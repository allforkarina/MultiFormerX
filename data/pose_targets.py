from __future__ import annotations

"""OpenPose-style PCM/PAF target generation from OpenPose18 keypoints."""

import numpy as np


OPENPOSE18_LIMBS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
    (2, 16),
    (5, 17),
)


def coco17_to_openpose18(keypoints: np.ndarray) -> np.ndarray:
    """Map normalized COCO-17 keypoints to OpenPose BODY_18 order."""

    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape != (17, 2):
        raise ValueError(f"Expected keypoints with shape (17, 2), got {keypoints.shape}")

    openpose_keypoints = np.full((18, 2), np.nan, dtype=np.float32)
    openpose_keypoints[0] = keypoints[0]
    openpose_keypoints[2] = keypoints[6]
    openpose_keypoints[3] = keypoints[8]
    openpose_keypoints[4] = keypoints[10]
    openpose_keypoints[5] = keypoints[5]
    openpose_keypoints[6] = keypoints[7]
    openpose_keypoints[7] = keypoints[9]
    openpose_keypoints[8] = keypoints[12]
    openpose_keypoints[9] = keypoints[14]
    openpose_keypoints[10] = keypoints[16]
    openpose_keypoints[11] = keypoints[11]
    openpose_keypoints[12] = keypoints[13]
    openpose_keypoints[13] = keypoints[15]
    openpose_keypoints[14] = keypoints[2]
    openpose_keypoints[15] = keypoints[1]
    openpose_keypoints[16] = keypoints[4]
    openpose_keypoints[17] = keypoints[3]

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    if np.isfinite(left_shoulder).all() and np.isfinite(right_shoulder).all():
        openpose_keypoints[1] = (left_shoulder + right_shoulder) * 0.5

    return openpose_keypoints


def _valid_point(point: np.ndarray) -> bool:
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def _pose_to_heatmap_coords(
    kpts: np.ndarray,
    size: int = 36,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float32).copy()
    lo, hi = pose_range
    scale = (size - 1) / (hi - lo)
    invalid = ~np.isfinite(kpts).all(axis=-1) | np.all(np.isclose(kpts, 0.0), axis=-1)
    kpts = (kpts - lo) * scale
    kpts = np.clip(kpts, 0, size - 1)
    kpts[invalid] = 0.0
    return kpts.astype(np.float32)


def generate_pose_targets(
    keypoints: np.ndarray,
    heatmap_size: int = 36,
    heatmap_sigma: float = 1.5,
    paf_width: float = 1.0,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate PCM and PAF targets from OpenPose18 keypoints.

    Args:
        keypoints: (18, 2) OpenPose BODY_18 in pose_range coordinates.
        heatmap_size: spatial size (36).
        heatmap_sigma: Gaussian sigma.
        paf_width: limb half-width.
        pose_range: (min, max) of keypoint coordinate range.

    Returns:
        pcm (19, H, W), paf (38, H, W).
        PCM ch0-17: 18 anatomical keypoints, ch18: mean background.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    grid_y, grid_x = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(np.float32)
    points = _pose_to_heatmap_coords(keypoints, size=heatmap_size, pose_range=pose_range)

    valid = np.array([_valid_point(k) for k in keypoints], dtype=bool)

    pcm = np.zeros((19, heatmap_size, heatmap_size), dtype=np.float32)
    for idx, point in enumerate(points):
        if valid[idx]:
            distance_sq = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2
            pcm[idx] = np.exp(-distance_sq / (2.0 * heatmap_sigma**2))
    pcm[18] = pcm[:18].mean(axis=0)  # background = mean of valid keypoint heatmaps

    paf = np.zeros((len(OPENPOSE18_LIMBS) * 2, heatmap_size, heatmap_size), dtype=np.float32)
    for limb_idx, (start_idx, end_idx) in enumerate(OPENPOSE18_LIMBS):
        if not (valid[start_idx] and valid[end_idx]):
            continue
        start = points[start_idx]
        end = points[end_idx]
        limb_vec = end - start
        limb_len = float(np.linalg.norm(limb_vec))
        if limb_len < 1e-6:
            continue

        unit_vec = limb_vec / limb_len
        rel_x = grid_x - start[0]
        rel_y = grid_y - start[1]
        proj = rel_x * unit_vec[0] + rel_y * unit_vec[1]
        perp = np.abs(rel_x * unit_vec[1] - rel_y * unit_vec[0])
        mask = (proj >= 0.0) & (proj <= limb_len) & (perp <= paf_width)

        paf[2 * limb_idx][mask] = unit_vec[0]
        paf[2 * limb_idx + 1][mask] = unit_vec[1]

    return pcm, paf
