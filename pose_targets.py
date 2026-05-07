from __future__ import annotations

"""COCO-17 keypoint to PCM/PAF target generation for MultiFormer training."""

import numpy as np


COCO17_LIMBS: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)


def generate_pose_targets(
    keypoints: np.ndarray,
    heatmap_size: int = 36,
    heatmap_sigma: float = 1.5,
    paf_width: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate PCM and PAF targets from one normalized COCO-17 keypoint array.

    Args:
        keypoints: Array with shape ``(17, 2)``. Coordinates are expected to be
            normalized to ``[0, 1]`` in the same convention used by the dataset.
        heatmap_size: Output spatial size for both PCM and PAF targets.
        heatmap_sigma: Gaussian standard deviation in heatmap pixels.
        paf_width: Limb half-width in heatmap pixels.

    Returns:
        ``pcm`` with shape ``(18, H, W)`` and ``paf`` with shape ``(38, H, W)``.
        The final PCM channel is the background channel.
    """

    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape != (17, 2):
        raise ValueError(f"Expected keypoints with shape (17, 2), got {keypoints.shape}")

    grid_y, grid_x = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(np.float32)
    points = keypoints.copy()
    points[:, 0] *= heatmap_size - 1
    points[:, 1] *= heatmap_size - 1

    valid_points = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] <= heatmap_size - 1)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= heatmap_size - 1)
    )

    pcm = np.zeros((18, heatmap_size, heatmap_size), dtype=np.float32)
    for keypoint_index, point in enumerate(points):
        if not valid_points[keypoint_index]:
            continue
        distance_sq = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2
        pcm[keypoint_index] = np.exp(-distance_sq / (2.0 * heatmap_sigma**2))
    pcm[-1] = np.clip(1.0 - np.max(pcm[:-1], axis=0), 0.0, 1.0)

    paf = np.zeros((len(COCO17_LIMBS) * 2, heatmap_size, heatmap_size), dtype=np.float32)
    for limb_index, (start_index, end_index) in enumerate(COCO17_LIMBS):
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
