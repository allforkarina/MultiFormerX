from __future__ import annotations

"""COCO-17 keypoints to OpenPose-style PCM/PAF target generation."""

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


def generate_pose_targets(
    keypoints: np.ndarray,
    heatmap_size: int = 36,
    heatmap_sigma: float = 1.5,
    paf_width: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate OpenPose-style PCM and PAF targets from normalized COCO-17 keypoints.

    Args:
        keypoints: Array with shape ``(17, 2)``. Coordinates are expected to be
            normalized to ``[0, 1]`` in the same convention used by the dataset.
        heatmap_size: Output spatial size for both PCM and PAF targets.
        heatmap_sigma: Gaussian standard deviation in heatmap pixels.
        paf_width: Limb half-width in heatmap pixels.

    Returns:
        ``pcm`` with shape ``(19, H, W)`` and ``paf`` with shape ``(38, H, W)``.
        PCM channels are 18 OpenPose BODY_18 keypoints plus background.
    """

    grid_y, grid_x = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(np.float32)
    points = coco17_to_openpose18(keypoints)
    points[:, 0] *= heatmap_size - 1
    points[:, 1] *= heatmap_size - 1

    valid_points = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] <= heatmap_size - 1)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= heatmap_size - 1)
    )

    pcm = np.zeros((19, heatmap_size, heatmap_size), dtype=np.float32)
    for keypoint_index, point in enumerate(points):
        if not valid_points[keypoint_index]:
            continue
        distance_sq = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2
        pcm[keypoint_index] = np.exp(-distance_sq / (2.0 * heatmap_sigma**2))
    pcm[-1] = np.clip(1.0 - np.max(pcm[:-1], axis=0), 0.0, 1.0)

    paf = np.zeros((len(OPENPOSE18_LIMBS) * 2, heatmap_size, heatmap_size), dtype=np.float32)
    for limb_index, (start_index, end_index) in enumerate(OPENPOSE18_LIMBS):
        if not (valid_points[start_index] and valid_points[end_index]):
            continue
        start = points[start_index]
        end = points[end_index]
        limb_vector = end - start
        limb_length = float(np.linalg.norm(limb_vector))
        if limb_length < 1e-6:
            continue

        unit_vector = limb_vector / limb_length
        rel_x = grid_x - start[0]
        rel_y = grid_y - start[1]
        projection = rel_x * unit_vector[0] + rel_y * unit_vector[1]
        perpendicular = np.abs(rel_x * unit_vector[1] - rel_y * unit_vector[0])
        mask = (projection >= 0.0) & (projection <= limb_length) & (perpendicular <= paf_width)

        paf[2 * limb_index][mask] = unit_vector[0]
        paf[2 * limb_index + 1][mask] = unit_vector[1]

    return pcm, paf
