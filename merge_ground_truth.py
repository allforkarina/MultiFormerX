from __future__ import annotations

"""Merge per-trial H36M-17 ground truth .npy files into a single OpenPose18 ground_truth.npy.

Reads individual per-trial files from ground_truth_npy/, sorted by (action, subject),
converts H36M-17 → OpenPose BODY_18 (with neck interpolation), concatenates all frames,
and saves as a single ground_truth.npy in the project root.
"""

import sys
from pathlib import Path

import numpy as np


def h36m17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    """Convert a single frame from H36M-17 (17, 2) to OpenPose BODY_18 (18, 2).

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

    # H36M-17 → OpenPose BODY_18 mapping.
    # H36M-17 order:                   OpenPose BODY_18:
    #   0  pelvis                       0  nose (← neck/head_base)
    #   1  r_hip                        1  neck (interpolated below)
    #   2  r_knee                       2  r_shoulder
    #   3  r_ankle                      3  r_elbow
    #   4  l_hip                        4  r_wrist
    #   5  l_knee                       5  l_shoulder
    #   6  l_ankle                      6  l_elbow
    #   7  spine                        7  l_wrist
    #   8  thorax                       8  r_hip
    #   9  neck/head_base               9  r_knee
    #  10  head                        10  r_ankle
    #  11  l_shoulder                  11  l_hip
    #  12  l_elbow                     12  l_knee
    #  13  l_wrist                     13  l_ankle
    #  14  r_shoulder                  14  — (NaN, no face data)
    #  15  r_elbow                     15  — (NaN)
    #  16  r_wrist                     16  — (NaN)
    #                                   17  — (NaN)
    _H36M17_TO_OPENPOSE18 = {
        9: 0,   # nose
        14: 2,  # r_shoulder
        15: 3,  # r_elbow
        16: 4,  # r_wrist
        11: 5,  # l_shoulder
        12: 6,  # l_elbow
        13: 7,  # l_wrist
        1: 8,   # r_hip
        2: 9,   # r_knee
        3: 10,  # r_ankle
        4: 11,  # l_hip
        5: 12,  # l_knee
        6: 13,  # l_ankle
    }

    for src_idx, op_idx in _H36M17_TO_OPENPOSE18.items():
        if _is_valid(kpts17[src_idx]):
            kpts18[op_idx] = kpts17[src_idx]

    # Neck interpolation (OpenPose index 1).
    left_sh = kpts17[11]
    right_sh = kpts17[14]
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
            converted.append(h36m17_to_openpose18(frames[i]))

        all_frames.append(np.stack(converted, axis=0))

    merged = np.concatenate(all_frames, axis=0)  # (N_total, 18, 2)
    np.save(str(output_path), merged)

    print(f"Total frames: {merged.shape[0]}")
    print(f"Output shape: {merged.shape}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
