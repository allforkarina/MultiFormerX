from __future__ import annotations

"""Train MultiFormer on MM-Fi memmap dataset."""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.memmap_dataset import MemmapDataset
from eval.metrics import heatmaps_to_keypoints, pck_score
from model.multiformer import MultiFormer, multistage_pose_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MultiFormer on memmap dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to memmap .npy data directory")
    parser.add_argument("--output-dir", type=str, default="runs/multiformer")
    parser.add_argument("--normalize", type=str, default="global_minmax",
                        choices=["global_minmax", "global_zscore", "zscore"])
    parser.add_argument("--train-subjects", nargs="+", default=None,
                        help="List of training subject IDs, e.g. S01 S02 ... S10")
    parser.add_argument("--val-subjects", nargs="+", default=None)
    parser.add_argument("--random-val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=15)
    parser.add_argument("--lr-gamma", type=float, default=0.7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pck-threshold", type=float, default=0.20)
    parser.add_argument("--paf-loss-weight", type=float, default=0.5)
    parser.add_argument("--pose-min", type=float, default=-0.8)
    parser.add_argument("--pose-max", type=float, default=0.8)
    return parser.parse_args()


def run_epoch(
    model: MultiFormer,
    data_loader: DataLoader,
    device: torch.device,
    paf_loss_weight: float = 0.5,
    pose_range: tuple[float, float] = (-0.8, 0.8),
    optimizer: AdamW | None = None,
    pck_threshold: float = 0.20,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_pck = 0.0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in tqdm(data_loader, dynamic_ncols=True, leave=False):
            csi = batch["csi"].to(device=device, dtype=torch.float32)
            kpts18 = batch["kpts18"].to(device=device, dtype=torch.float32)
            target_pcm = batch["pcm"].to(device=device, dtype=torch.float32)
            target_paf = batch["paf"].to(device=device, dtype=torch.float32)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(csi)
            loss = multistage_pose_loss(predictions, target_pcm, target_paf, paf_loss_weight=paf_loss_weight)

            final_pcm, _ = predictions[-1]
            predicted_keypoints = heatmaps_to_keypoints(final_pcm, pose_range=pose_range)
            batch_pck = pck_score(predicted_keypoints, kpts18, threshold=pck_threshold)

            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = csi.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_pck += float(batch_pck.detach().cpu()) * batch_size
            total_samples += batch_size

    total_samples = max(total_samples, 1)
    return total_loss / total_samples, total_pck / total_samples


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: MultiFormer,
    optimizer: AdamW,
    scheduler: StepLR,
    train_loss: float,
    train_pck: float,
    val_loss: float,
    val_pck: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_loss": train_loss,
        "train_pck": train_pck,
        "val_loss": val_loss,
        "val_pck": val_pck,
    }
    torch.save(checkpoint, output_dir / "last.pt")


def save_metrics_history(output_dir: Path, history: list[dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    fieldnames = ["epoch", "train_loss", "val_loss", "train_pck20", "val_pck20", "lr"]
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_training_curves(output_dir: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping training curve images")
        return

    epochs = [item["epoch"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_loss"] for item in history], label="train loss")
    plt.plot(epochs, [item["val_loss"] for item in history], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_pck20"] for item in history], label="train PCK@20")
    plt.plot(epochs, [item["val_pck20"] for item in history], label="val PCK@20")
    plt.xlabel("epoch"); plt.ylabel("PCK@20")
    plt.ylim(0.0, 1.0); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(output_dir / "pck20_curve.png", dpi=150); plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    pose_range = (args.pose_min, args.pose_max)

    common_kwargs = dict(
        data_dir=args.data_dir,
        normalize=args.normalize,
        pose_range=pose_range,
    )
    train_ds = MemmapDataset(
        split="train",
        train_subjects=args.train_subjects,
        random_val_ratio=args.random_val_ratio,
        seed=args.seed,
        **common_kwargs,
    )
    val_ds = MemmapDataset(
        split="val",
        train_subjects=args.train_subjects,
        random_val_ratio=args.random_val_ratio,
        seed=args.seed,
        **common_kwargs,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = MultiFormer().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    output_dir = Path(args.output_dir)

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pck = run_epoch(
            model, train_loader, device,
            paf_loss_weight=args.paf_loss_weight, pose_range=pose_range,
            optimizer=optimizer, pck_threshold=args.pck_threshold,
        )
        val_loss, val_pck = run_epoch(
            model, val_loader, device,
            paf_loss_weight=args.paf_loss_weight, pose_range=pose_range,
            pck_threshold=args.pck_threshold,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        save_checkpoint(output_dir, epoch, model, optimizer, scheduler,
                        train_loss, train_pck, val_loss, val_pck)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_pck20": train_pck, "val_pck20": val_pck, "lr": current_lr,
        })
        save_metrics_history(output_dir, history)
        save_training_curves(output_dir, history)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} train_pck20={train_pck:.4f} "
            f"val_loss={val_loss:.6f} val_pck20={val_pck:.4f} "
            f"lr={current_lr:.6g}"
        )


if __name__ == "__main__":
    main()
