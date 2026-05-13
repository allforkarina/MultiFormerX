from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .pose_targets import generate_pose_targets


CSI_FILES = {
    "global_minmax": "csi_gminmax.npy",
    "global_zscore": "csi_gzscore.npy",
    "zscore": "csi_zscore.npy",
}


class MemmapDataset(Dataset):
    """Memory-mapped .npy dataset for fast training I/O.

    CSI is stored as pre-normalized .npy files, read via np.load(mmap_mode='r').
    Keypoints and meta are small enough to load entirely into RAM at init.
    PCM/PAF targets are generated on-the-fly in __getitem__.

    Split logic: group by subject, shuffle frames within each subject,
    80/20 train/val split per subject.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        envs: Iterable[str] | None = None,
        train_subjects: Iterable[str] | None = None,
        test_subjects: Iterable[str] | None = None,
        random_val_ratio: float = 0.2,
        seed: int = 42,
        normalize: str = "global_minmax",
        heatmap_size: int = 36,
        heatmap_sigma: float = 1.5,
        paf_width: float = 1.0,
        pose_range: tuple[float, float] = (-0.8, 0.8),
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be train/val/test/all, got {split}")
        if normalize not in CSI_FILES:
            raise ValueError(f"Unknown normalize mode: {normalize}, expected one of {list(CSI_FILES)}")

        self.split = split
        self.normalize = normalize
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.paf_width = paf_width
        self.pose_range = pose_range

        data_dir = Path(data_dir)

        self._csi = np.load(str(data_dir / CSI_FILES[normalize]), mmap_mode="r")
        self._kpts18 = np.load(str(data_dir / "ground_truth.npy"))
        meta = np.load(str(data_dir / "meta.npz"), allow_pickle=True)
        self._envs = meta["environment"]
        self._samples = meta["sample"]
        self._actions = meta["action"]

        self.indices = self._build_split(split, envs, train_subjects, test_subjects, random_val_ratio, seed)

    def _build_split(
        self,
        split: str,
        envs: Iterable[str] | None,
        train_subjects: Iterable[str] | None,
        test_subjects: Iterable[str] | None,
        random_val_ratio: float,
        seed: int,
    ) -> np.ndarray:
        num_total = len(self._samples)
        env_set = set(envs) if envs else None
        train_set = set(train_subjects) if train_subjects else None
        test_set = set(test_subjects) if test_subjects else None

        candidate_indices: list[int] = []
        for i in range(num_total):
            if env_set is not None and str(self._envs[i]) not in env_set:
                continue
            sample = str(self._samples[i])
            if split == "train" and train_set is not None and sample not in train_set:
                continue
            if split in {"val", "test"} and train_set is not None and sample in train_set:
                continue
            if split == "test" and test_set is not None and sample not in test_set:
                continue
            candidate_indices.append(i)

        if split == "all":
            return np.asarray(sorted(candidate_indices), dtype=np.int64)

        rng = random.Random(seed)
        grouped: dict[str, list[int]] = {}
        for idx in candidate_indices:
            grouped.setdefault(str(self._samples[idx]), []).append(idx)

        train_indices: list[int] = []
        val_indices: list[int] = []
        for subject, indices in sorted(grouped.items()):
            shuffled = indices[:]
            rng.shuffle(shuffled)
            pivot = int(round(len(shuffled) * (1.0 - random_val_ratio)))
            train_indices.extend(shuffled[:pivot])
            val_indices.extend(shuffled[pivot:])

        chosen = train_indices if split == "train" else val_indices
        return np.asarray(sorted(chosen), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        frame_idx = int(self.indices[index])
        csi = np.array(self._csi[frame_idx])
        kpts18 = self._kpts18[frame_idx].copy()

        pcm, paf = generate_pose_targets(
            kpts18,
            heatmap_size=self.heatmap_size,
            heatmap_sigma=self.heatmap_sigma,
            paf_width=self.paf_width,
            pose_range=self.pose_range,
        )

        return {
            "csi": torch.from_numpy(csi),
            "kpts18": torch.from_numpy(np.ascontiguousarray(kpts18)),
            "pcm": torch.from_numpy(pcm),
            "paf": torch.from_numpy(paf),
            "meta": {
                "env": str(self._envs[frame_idx]),
                "subject": str(self._samples[frame_idx]),
                "action": str(self._actions[frame_idx]),
                "frame_idx": int(frame_idx),
            },
        }
