# Ground Truth Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge 1080 per-trial ground truth `.npy` files into a single `ground_truth.npy` and switch the dataloader to use it.

**Architecture:** A standalone merge script reads `ground_truth_npy/*.npy`, sorts by (action, subject) to align with existing CSI memmap, extracts x,y coordinates, maps COCO17→OpenPose18 with neck interpolation, concatenates, and saves. The dataloader change is a one-line filename swap.

**Tech Stack:** Python 3.10+, numpy, pathlib.

---

### Task 1: Create merge script

**Files:**
- Create: `merge_ground_truth.py`

**Purpose:** Read individual trial `.npy` files, merge into one aligned `.npy`.

- [ ] **Step 1: Write the merge script**

```python
from __future__ import annotations

"""Merge individual ground truth .npy files into a single memmap-compatible file.

Input:  ground_truth_npy/E{env}_S{subject}_A{action}.npy  (N_frames, 17, 3)
Output: ground_truth.npy  (N_total, 18, 2)  OpenPose BODY_18 in pose_range [-0.8, 0.8]
"""

import sys
from pathlib import Path

import numpy as np


COCO17_TO_OPENPOSE18: dict[int, int] = {
    0: 0,   # nose
    2: 6,   # r_shoulder
    3: 8,   # r_elbow
    4: 10,  # r_wrist
    5: 5,   # l_shoulder
    6: 7,   # l_elbow
    7: 9,   # l_wrist
    8: 12,  # r_hip
    9: 14,  # r_knee
    10: 16, # r_ankle
    11: 11, # l_hip
    12: 13, # l_knee
    13: 15, # l_ankle
    14: 2,  # r_eye
    15: 1,  # l_eye
    16: 4,  # r_ear
    17: 3,  # l_ear
}


def _valid_point(point: np.ndarray) -> bool:
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def coco17_to_openpose18(kpts17: np.ndarray) -> np.ndarray:
    kpts17 = np.asarray(kpts17, dtype=np.float32)
    kpts18 = np.zeros((18, 2), dtype=np.float32)
    valid = np.zeros(18, dtype=bool)
    for op_idx, coco_idx in COCO17_TO_OPENPOSE18.items():
        p = kpts17[coco_idx]
        if _valid_point(p):
            kpts18[op_idx] = p
            valid[op_idx] = True
    l_sh, r_sh = kpts17[5], kpts17[6]
    if _valid_point(l_sh) and _valid_point(r_sh):
        kpts18[1] = (l_sh + r_sh) * 0.5
        valid[1] = True
    elif _valid_point(l_sh):
        kpts18[1] = l_sh
        valid[1] = True
    elif _valid_point(r_sh):
        kpts18[1] = r_sh
        valid[1] = True
    kpts18[~valid] = 0.0
    return kpts18


def _sort_key(filepath: Path) -> tuple[str, str]:
    parts = filepath.stem.split("_")
    return (parts[2], parts[1])


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    source_dir = script_dir / "ground_truth_npy"
    if not source_dir.is_dir():
        print(f"ERROR: source directory not found: {source_dir}")
        sys.exit(1)

    files = sorted(source_dir.glob("*.npy"), key=_sort_key)
    if not files:
        print(f"ERROR: no .npy files found in {source_dir}")
        sys.exit(1)

    print(f"Found {len(files)} files in {source_dir}")

    all_kpts: list[np.ndarray] = []
    total_frames = 0
    for fp in files:
        data = np.load(str(fp))
        frames, _, _ = data.shape
        kpts17 = data[:, :, :2].astype(np.float32)
        kpts18 = np.empty((frames, 18, 2), dtype=np.float32)
        for i in range(frames):
            kpts18[i] = coco17_to_openpose18(kpts17[i])
        all_kpts.append(kpts18)
        total_frames += frames

    merged = np.concatenate(all_kpts, axis=0)
    output_path = script_dir / "ground_truth.npy"
    np.save(str(output_path), merged)
    print(f"Saved {output_path}  shape={merged.shape}  dtype={merged.dtype}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run syntax check**

Run: `python -m py_compile merge_ground_truth.py`
Expected: OK (no output)

- [ ] **Step 3: Run the merge script**

Run: `python merge_ground_truth.py`
Expected: `Found 1080 files ... Saved ground_truth.npy shape=(N, 18, 2) dtype=float32`

- [ ] **Step 4: Verify output shape against meta.npz**

Run: `python -c "import numpy as np; m=np.load('ground_truth_npy/../meta.npz', allow_pickle=True) if __import__('pathlib').Path('meta.npz').exists() else None; gt=np.load('ground_truth.npy'); print(f'ground_truth: {gt.shape}')"`
Expected: ground_truth shape is (N_total, 18, 2)

- [ ] **Step 5: Commit**

```bash
git add merge_ground_truth.py
git commit -m "feat: add merge script for per-trial ground truth npy files"
```

---

### Task 2: Switch dataloader to ground_truth.npy

**Files:**
- Modify: `data/memmap_dataset.py:62`

- [ ] **Step 1: Change kpts18.npy → ground_truth.npy**

In `data/memmap_dataset.py`, change line 62 from:
```python
self._kpts18 = np.load(str(data_dir / "kpts18.npy"))
```
to:
```python
self._kpts18 = np.load(str(data_dir / "ground_truth.npy"))
```

- [ ] **Step 2: Run syntax check**

Run: `python -m py_compile data/memmap_dataset.py`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add data/memmap_dataset.py
git commit -m "feat: switch dataloader to MM-Fi official ground truth keypoints"
```

---

### Post-implementation verification

After both tasks complete, run the full compile-check:

```bash
python -m py_compile merge_ground_truth.py data/memmap_dataset.py data/pose_targets.py eval/metrics.py train.py model/tfddt.py model/attention.py model/heatmap_decoder.py model/papm.py model/msfn.py model/multiformer.py
```
