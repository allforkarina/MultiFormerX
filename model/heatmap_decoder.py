from __future__ import annotations

import torch
from torch import nn


class HeatmapDecoder(nn.Module):
    """Decode 2D features into PCM (19ch) and PAF (38ch) heatmaps (论文模块 C 子组件).

    共享 4 层 3×3 conv trunk (256→128→128→128→128),
    然后 bottleneck 1×1 conv (128→512) 接 PCM/PAF 1×1 heads.
    """

    def __init__(
        self,
        in_channels: int = 256,
        mid_channels: int = 128,
        hidden_channels: int = 512,
        pcm_channels: int = 19,
        paf_channels: int = 38,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for _ in range(4):
            layers.extend([
                nn.Conv2d(channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ])
            channels = mid_channels
        self.shared = nn.Sequential(*layers)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(mid_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.pcm_head = nn.Conv2d(hidden_channels, pcm_channels, kernel_size=1)
        self.paf_head = nn.Conv2d(hidden_channels, paf_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(x)
        x = self.bottleneck(x)
        return self.pcm_head(x), self.paf_head(x)
