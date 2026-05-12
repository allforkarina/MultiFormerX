from __future__ import annotations

"""Single-person pose metrics for MultiFormer PCM outputs."""

import torch


OPENPOSE18_TO_COCO17: tuple[int, ...] = (
    0,
    15,
    14,
    17,
    16,
    5,
    2,
    6,
    3,
    7,
    4,
    11,
    8,
    12,
    9,
    13,
    10,
)


def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    keypoint_indices: tuple[int, ...] = OPENPOSE18_TO_COCO17,
) -> torch.Tensor:
    """Decode normalized COCO-17 keypoint coordinates from OpenPose PCM heatmaps.

    Args:
        heatmaps: Tensor with shape ``(B, C, H, W)``. Only the first
            ``keypoint_indices`` channels are decoded; background is ignored.
        keypoint_indices: OpenPose PCM channel indices in target output order.

    Returns:
        Tensor with shape ``(B, K, 2)`` containing normalized
        ``x, y`` coordinates in ``[0, 1]``.
    """

    if heatmaps.ndim != 4:
        raise ValueError(f"Expected heatmaps with 4 dims, got {heatmaps.shape}")
    if not keypoint_indices:
        raise ValueError("Expected at least one keypoint index")
    max_keypoint_index = max(keypoint_indices)
    if heatmaps.shape[1] <= max_keypoint_index:
        raise ValueError(
            f"Expected heatmaps with channel index {max_keypoint_index}, got {heatmaps.shape[1]} channels"
        )

    keypoint_heatmaps = heatmaps[:, keypoint_indices]
    batch_size, _, height, width = keypoint_heatmaps.shape
    flat_indices = keypoint_heatmaps.flatten(start_dim=2).argmax(dim=2)

    y = torch.div(flat_indices, width, rounding_mode="floor").to(dtype=heatmaps.dtype)
    x = (flat_indices % width).to(dtype=heatmaps.dtype)
    x = x / max(width - 1, 1)
    y = y / max(height - 1, 1)
    return torch.stack((x, y), dim=2).reshape(batch_size, len(keypoint_indices), 2)


def pck_score(
    predicted_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    threshold: float = 0.20,
    right_shoulder_index: int = 6,
    left_hip_index: int = 11,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute torso-normalized single-person PCK on normalized coordinates."""

    if predicted_keypoints.shape != target_keypoints.shape:
        raise ValueError(
            "Predicted and target keypoints must have the same shape, "
            f"got {predicted_keypoints.shape} and {target_keypoints.shape}"
        )
    if predicted_keypoints.ndim != 3 or predicted_keypoints.shape[-1] != 2:
        raise ValueError(f"Expected keypoints shaped (B, K, 2), got {predicted_keypoints.shape}")
    num_keypoints = target_keypoints.shape[1]
    if right_shoulder_index >= num_keypoints or left_hip_index >= num_keypoints:
        raise ValueError(
            "Torso keypoint indices must be within keypoint dimension, "
            f"got right_shoulder_index={right_shoulder_index}, left_hip_index={left_hip_index}, "
            f"num_keypoints={num_keypoints}"
        )

    valid = torch.isfinite(target_keypoints).all(dim=2)
    torso_points = target_keypoints[:, (right_shoulder_index, left_hip_index)]
    torso_valid = torch.isfinite(torso_points).all(dim=(1, 2))
    torso_scale = torch.linalg.vector_norm(
        target_keypoints[:, right_shoulder_index] - target_keypoints[:, left_hip_index],
        dim=1,
    )
    valid = valid & torso_valid.unsqueeze(1) & (torso_scale > eps).unsqueeze(1)
    distances = torch.linalg.vector_norm(predicted_keypoints - target_keypoints, dim=2)
    normalized_distances = distances / torso_scale.clamp_min(eps).unsqueeze(1)
    correct = (normalized_distances <= threshold) & valid
    return correct.sum().to(dtype=torch.float32) / valid.sum().clamp_min(1).to(dtype=torch.float32)
