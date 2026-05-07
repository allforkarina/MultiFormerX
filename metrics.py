from __future__ import annotations

"""Single-person pose metrics for MultiFormer PCM outputs."""

import torch


def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    num_keypoints: int = 17,
) -> torch.Tensor:
    """Decode normalized single-person keypoint coordinates from PCM heatmaps.

    Args:
        heatmaps: Tensor with shape ``(B, C, H, W)``. Only the first
            ``num_keypoints`` channels are decoded; any background channel is
            ignored.
        num_keypoints: Number of COCO keypoint channels to decode.

    Returns:
        Tensor with shape ``(B, num_keypoints, 2)`` containing normalized
        ``x, y`` coordinates in ``[0, 1]``.
    """

    if heatmaps.ndim != 4:
        raise ValueError(f"Expected heatmaps with 4 dims, got {heatmaps.shape}")
    if heatmaps.shape[1] < num_keypoints:
        raise ValueError(f"Expected at least {num_keypoints} heatmap channels, got {heatmaps.shape[1]}")

    keypoint_heatmaps = heatmaps[:, :num_keypoints]
    batch_size, _, height, width = keypoint_heatmaps.shape
    flat_indices = keypoint_heatmaps.flatten(start_dim=2).argmax(dim=2)

    y = torch.div(flat_indices, width, rounding_mode="floor").to(dtype=heatmaps.dtype)
    x = (flat_indices % width).to(dtype=heatmaps.dtype)
    x = x / max(width - 1, 1)
    y = y / max(height - 1, 1)
    return torch.stack((x, y), dim=2).reshape(batch_size, num_keypoints, 2)


def pck_score(
    predicted_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Compute single-person PCK on normalized coordinates."""

    if predicted_keypoints.shape != target_keypoints.shape:
        raise ValueError(
            "Predicted and target keypoints must have the same shape, "
            f"got {predicted_keypoints.shape} and {target_keypoints.shape}"
        )
    if predicted_keypoints.ndim != 3 or predicted_keypoints.shape[-1] != 2:
        raise ValueError(f"Expected keypoints shaped (B, K, 2), got {predicted_keypoints.shape}")

    valid = torch.isfinite(target_keypoints).all(dim=2)
    distances = torch.linalg.vector_norm(predicted_keypoints - target_keypoints, dim=2)
    correct = (distances <= threshold) & valid
    return correct.sum().to(dtype=torch.float32) / valid.sum().clamp_min(1).to(dtype=torch.float32)
