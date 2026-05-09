from __future__ import annotations

"""Train MultiFormer on the local HDF5 MM-Fi pose dataset."""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from dataloader import DEFAULT_SPLIT_SCHEME, create_data_loaders
from metrics import heatmaps_to_keypoints, pck_score
from model import MultiFormer, multistage_pose_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MultiFormer")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to packed HDF5 dataset")
    parser.add_argument("--output-dir", type=str, default="runs/multiformer", help="Checkpoint directory")
    parser.add_argument("--split-scheme", type=str, default=DEFAULT_SPLIT_SCHEME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-step-size", type=int, default=15)
    parser.add_argument("--lr-gamma", type=float, default=0.7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pck-threshold", type=float, default=0.20)
    return parser.parse_args()


def move_batch_to_device(
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    csi_amplitude = batch["csi_amplitude"].to(device=device, dtype=torch.float32)
    keypoints = batch["keypoints"].to(device=device, dtype=torch.float32)
    target_pcm = batch["pcm"].to(device=device, dtype=torch.float32)
    target_paf = batch["paf"].to(device=device, dtype=torch.float32)
    return csi_amplitude, keypoints, target_pcm, target_paf


def run_epoch(
    model: MultiFormer,
    data_loader,
    device: torch.device,
    optimizer: Adam | None = None,
    pck_threshold: float = 0.05,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_pck = 0.0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in tqdm(data_loader, dynamic_ncols=True, leave=False):
            csi_amplitude, keypoints, target_pcm, target_paf = move_batch_to_device(batch, device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(csi_amplitude)
            loss = multistage_pose_loss(predictions, target_pcm, target_paf)
            final_pcm, _ = predictions[-1]
            predicted_keypoints = heatmaps_to_keypoints(final_pcm)
            batch_pck = pck_score(predicted_keypoints, keypoints, threshold=pck_threshold)

            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = csi_amplitude.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_pck += float(batch_pck.detach().cpu()) * batch_size
            total_samples += batch_size

    total_samples = max(total_samples, 1)
    return total_loss / total_samples, total_pck / total_samples


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: MultiFormer,
    optimizer: Adam,
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
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_pck20"] for item in history], label="train PCK@20")
    plt.plot(epochs, [item["val_pck20"] for item in history], label="val PCK@20")
    plt.xlabel("epoch")
    plt.ylabel("PCK@20")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "pck20_curve.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    data_loaders = create_data_loaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_scheme=args.split_scheme,
        return_pose_targets=True,
    )

    model = MultiFormer().to(device)
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    output_dir = Path(args.output_dir)

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pck = run_epoch(
            model,
            data_loaders["train"],
            device,
            optimizer=optimizer,
            pck_threshold=args.pck_threshold,
        )
        val_loss, val_pck = run_epoch(
            model,
            data_loaders["val"],
            device,
            pck_threshold=args.pck_threshold,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        save_checkpoint(
            output_dir,
            epoch,
            model,
            optimizer,
            scheduler,
            train_loss,
            train_pck,
            val_loss,
            val_pck,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_pck20": train_pck,
                "val_pck20": val_pck,
                "lr": current_lr,
            }
        )
        save_metrics_history(output_dir, history)
        save_training_curves(output_dir, history)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"train_pck20={train_pck:.4f} "
            f"val_loss={val_loss:.6f} "
            f"val_pck20={val_pck:.4f} "
            f"lr={current_lr:.6g}"
        )


if __name__ == "__main__":
    main()
