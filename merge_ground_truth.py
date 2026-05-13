from __future__ import annotations

"""Merge per-trial COCO17 ground truth .npy files into a single OpenPose18 ground_truth.npy.

Reads individual per-trial files from ground_truth_npy/, sorted by (action, subject),
converts COCO17 → OpenPose BODY_18 (with neck interpolation), concatenates all frames,
and saves as a single ground_truth.npy in the project root.
"""

import sys
from pathlib import Path

import numpy as np


def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    """Convert a single frame from COCO17 (17, 2) to OpenPose BODY_18 (18, 2).

    Invalid keypoints (non-finite or all-close to zero) are set to [0.0, 0.0].
    The neck (OpenPose index 1) is interpolated as the midpoint of the left and
    right shoulders when both are valid; if only one is valid, that shoulder is
    used directly.
    """
    kpts17 = np.asarray(kpts17, dtype=np.float32)
    if kpts17.shape != (17, 2):
        raise ValueError(f"Expected keypoints with shape (17, 2), got {kpts17.shape}")

    def _is_valid(point: np.ndarray) -> bool:
        return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))

    # Start with zeros (invalid by default).
    kpts18 = np.zeros((18, 2), dtype=np.float32)

    # COCO17 → OpenPose BODY_18 mapping (matches spec table in
    # docs/superpowers/specs/2026-05-14-ground-truth-migration.md).
    # Standard COCO17 order:           OpenPose BODY_18:
    #   0  nose                         0  nose
    #   1  left_eye                     1  neck (interpolated below)
    #   2  right_eye                    2  right_shoulder
    #   3  left_ear                     3  right_elbow
    #   4  right_ear                    4  right_wrist
    #   5  left_shoulder                5  left_shoulder
    #   6  right_shoulder               6  left_elbow
    #   7  left_elbow                   7  left_wrist
    #   8  right_elbow                  8  right_hip
    #   9  left_wrist                   9  right_knee
    #  10  right_wrist                 10  right_ankle
    #  11  left_hip                    11  left_hip
    #  12  right_hip                   12  left_knee
    #  13  left_knee                   13  left_ankle
    #  14  right_knee                  14  right_eye
    #  15  left_ankle                  15  left_eye
    #  16  right_ankle                 16  right_ear
    #                                  17  left_ear
    _COCO17_TO_OPENPOSE18 = {
        0: 0,   # nose
        1: 15,  # left_eye
        2: 14,  # right_eye
        3: 17,  # left_ear
        4: 16,  # right_ear
        5: 5,   # left_shoulder
        6: 2,   # right_shoulder
        7: 6,   # left_elbow
        8: 3,   # right_elbow
        9: 7,   # left_wrist
        10: 4,  # right_wrist
        11: 11, # left_hip
        12: 8,  # right_hip
        13: 12, # left_knee
        14: 9,  # right_knee
        15: 13, # left_ankle
        16: 10, # right_ankle
    }

    for coco_idx, op_idx in _COCO17_TO_OPENPOSE18.items():
        if _is_valid(kpts17[coco_idx]):
            kpts18[op_idx] = kpts17[coco_idx]

    # Neck interpolation (OpenPose index 1).
    left_sh = kpts17[5]
    right_sh = kpts17[6]
    left_valid = _is_valid(left_sh)
    right_valid = _is_valid(right_sh)

    if left_valid and right_valid:
        kpts18[1] = (left_sh + right_sh) * 0.5
    elif left_valid:
        kpts18[1] = left_sh
    elif right_valid:
        kpts18[1] = right_sh

    return kpts18


def _parse_sort_key(filepath: Path) -> tuple[str, str]:
    """Extract (action, subject) sort key from filename E{env}_S{subject}_A{action}.npy."""
    parts = filepath.stem.split("_")  # e.g. ['E01', 'S01', 'A01']
    # parts[2] = action ("A01"), parts[1] = subject ("S01")
    return (parts[2], parts[1])


def main() -> None:
    root = Path(__file__).resolve().parent
    src_dir = root / "ground_truth_npy"
    output_path = root / "ground_truth.npy"

    if not src_dir.is_dir():
        print(f"ERROR: source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect and sort .npy files by (action, subject).
    files = sorted(
        [p for p in src_dir.glob("*.npy")],
        key=_parse_sort_key,
    )

    if not files:
        print(f"ERROR: no .npy files found in {src_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} .npy files in {src_dir}")

    all_frames: list[np.ndarray] = []

    for fpath in files:
        data = np.load(fpath)  # shape (N_frames, 17, 3)
        frames = data[:, :, :2]  # (N_frames, 17, 2) — discard confidence

        converted: list[np.ndarray] = []
        for i in range(frames.shape[0]):
            converted.append(coco17_to_openpose18(frames[i]))

        all_frames.append(np.stack(converted, axis=0))

    merged = np.concatenate(all_frames, axis=0)  # (N_total, 18, 2)
    np.save(str(output_path), merged)

    print(f"Total frames: {merged.shape[0]}")
    print(f"Output shape: {merged.shape}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
