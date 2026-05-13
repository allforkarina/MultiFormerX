from __future__ import annotations

"""Single-person pose metrics for MultiFormer PCM outputs."""

import torch


def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> torch.Tensor:
    """Decode keypoint coordinates from PCM heatmaps.

    Args:
        heatmaps: (B, C, H, W). PCM channels: C-1 keypoints + 1 background (last channel).
        pose_range: (min, max) of keypoint coordinate system.

    Returns:
        (B, C-1, 2) keypoints in pose_range coordinates.
    """
    keypoint_heatmaps = heatmaps[:, :-1]  # exclude background channel
    batch_size, _, height, width = keypoint_heatmaps.shape
    flat_indices = keypoint_heatmaps.flatten(start_dim=2).argmax(dim=2)

    y = torch.div(flat_indices, width, rounding_mode="floor").to(dtype=heatmaps.dtype)
    x = (flat_indices % width).to(dtype=heatmaps.dtype)
    x = x / max(width - 1, 1)
    y = y / max(height - 1, 1)

    lo, hi = pose_range
    span = hi - lo
    x = x * span + lo
    y = y * span + lo

    return torch.stack((x, y), dim=2).reshape(batch_size, keypoint_heatmaps.shape[1], 2)


def pck_score(
    predicted_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    threshold: float = 0.20,
    right_shoulder_index: int = 2,
    left_hip_index: int = 11,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute torso-normalized single-person PCK.

    Both predicted and target keypoints must be in the same coordinate system.
    Valid keypoints are those with finite, non-zero values.
    """
    if predicted_keypoints.shape != target_keypoints.shape:
        raise ValueError(
            "Predicted and target keypoints must have the same shape, "
            f"got {predicted_keypoints.shape} and {target_keypoints.shape}"
        )

    num_keypoints = target_keypoints.shape[1]
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
