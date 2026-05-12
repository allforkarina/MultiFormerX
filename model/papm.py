from __future__ import annotations

import torch
from torch import nn


class PAPM(nn.Module):
    """Pose Attention Perception Module (论文公式 7-8).

    从上一 stage 的 heatmaps (PCM+PAF, 57ch) 计算:
      - Channel attention: MLP(57 → hidden → 256) + sigmoid
      - Spatial attention: Conv2d(2 → 1, k=7) + sigmoid
    对特征图施加加权: Φ_i = Φ_{i-1} ⊙ W_C ⊗ W_S
    """

    def __init__(self, feature_channels: int = 256, heat_channels: int = 57, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(feature_channels // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Linear(heat_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_channels),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor, heatmaps: torch.Tensor) -> torch.Tensor:
        avg_pool = heatmaps.mean(dim=(2, 3))
        max_pool = heatmaps.amax(dim=(2, 3))
        channel_weight = self.sigmoid(self.channel_mlp(avg_pool) + self.channel_mlp(max_pool))
        channel_weight = channel_weight[:, :, None, None]

        spatial_avg = heatmaps.mean(dim=1, keepdim=True)
        spatial_max = heatmaps.amax(dim=1, keepdim=True)
        spatial_weight = self.spatial(torch.cat([spatial_avg, spatial_max], dim=1))

        return features * channel_weight * spatial_weight
