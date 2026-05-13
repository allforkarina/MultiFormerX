# Ground Truth Migration — Design Spec

> **Goal:** Replace the current `kpts18.npy` labels (derived from raw COCO17 via ad-hoc normalization) with MM-Fi official ground truth keypoints, which are professionally normalized to pose_range [-0.8, 0.8] and preserve correct skeletal proportions.

**Architecture:** A merge script reads individual per-trial `.npy` files from `ground_truth_npy/`, orders them by `(action, subject)` to align with the existing CSI memmap, converts COCO17→OpenPose18 (with neck interpolation), and writes a single `ground_truth.npy`. The dataloader then reads this file instead of `kpts18.npy`.

**Tech Stack:** Python 3.10+, numpy, pathlib.

---

## Background

The current `kpts18.npy` was built by `build_memmap.py`, which:
1. Reads raw COCO17 keypoints from MM-Fi `rgb/frame*.npy`
2. Maps to OpenPose18 (with neck = midpoint of shoulders)
3. Normalizes to pose_range [-0.8, 0.8] via ad-hoc scaling (if abs_max > 10, divide by image dimensions then remap)

The ground truth files in `ground_truth_npy/` are MM-Fi's official pre-normalized labels, also in pose_range, but with professionally computed normalization that preserves correct inter-joint relationships.

## Input format

Each file in `ground_truth_npy/`: named `E{env}_S{subject}_A{action}.npy`, shape `(N_frames, 17, 3)`.

- Channel 0: x coordinate (pose_range [-0.8, 0.8])
- Channel 1: y coordinate (pose_range [-0.8, 0.8])
- Channel 2: confidence score (not used in this pipeline)

Keypoints are in COCO17 order (nose, L-eye, R-eye, L-ear, R-ear, L-shoulder, R-shoulder, L-elbow, R-elbow, L-wrist, R-wrist, L-hip, R-hip, L-knee, R-knee, L-ankle, R-ankle).

## Output format

Single file `ground_truth.npy`: shape `(N_total, 18, 2)`, float32.

Keypoints are in OpenPose BODY_18 order, with neck (index 1) interpolated as midpoint of left and right shoulders when both are valid.

## File sorting

Files are sorted by `(action, subject)`, matching `build_memmap.py:iter_trials()` order:

```python
def sort_key(path):
    # E{env}_S{subject}_A{action}.npy
    parts = path.stem.split('_')
    subj = parts[1]  # S01
    act = parts[2]   # A01
    return (act, subj)
```

This ensures frame-level alignment with the existing CSI `csi_*.npy` and `meta.npz` files.

## COCO17 → OpenPose18 mapping

Reuse the identical mapping from `build_memmap.py`:

| OpenPose idx | COCO17 idx | Keypoint |
|-------------|-----------|----------|
| 0 | 0 | nose |
| 1 | — | neck (midpoint of L-shoulder + R-shoulder) |
| 2 | 6 | R-shoulder |
| 3 | 8 | R-elbow |
| 4 | 10 | R-wrist |
| 5 | 5 | L-shoulder |
| 6 | 7 | L-elbow |
| 7 | 9 | L-wrist |
| 8 | 12 | R-hip |
| 9 | 14 | R-knee |
| 10 | 16 | R-ankle |
| 11 | 11 | L-hip |
| 12 | 13 | L-knee |
| 13 | 15 | L-ankle |
| 14 | 2 | R-eye |
| 15 | 1 | L-eye |
| 16 | 4 | R-ear |
| 17 | 3 | L-ear |

Neck interpolation: if both L-shoulder and R-shoulder are valid (finite, non-zero), neck = (L-shoulder + R-shoulder) / 2. If only one is valid, use that one.

## Datasheet output alongside ground_truth.npy

Optionally save a `ground_truth_metadata.json` with:
- `total_frames`: N_total
- `num_files`: count of source files processed
- `source_dir`: path to source directory

This is for validation only.

## MemmapDataset change

In `data/memmap_dataset.py` line 62, change:

```python
self._kpts18 = np.load(str(data_dir / "kpts18.npy"))
```

to:

```python
self._kpts18 = np.load(str(data_dir / "ground_truth.npy"))
```

No other dataloader changes needed — the label is still `(N, 18, 2)` OpenPose18 in pose_range.

## Verification

After merging, verify:
1. `ground_truth.npy` shape matches total frame count from `meta.npz`
2. `python -m py_compile` passes for both new script and modified dataloader
3. Sample sanity: load a few frames, verify x,y values are within [-0.8, 0.8]
