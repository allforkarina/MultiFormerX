from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .tfddt import TFDDTTokenizer
from .attention import DualAttentionExtractor
from .msfn import MSFN


class MultiFormer(nn.Module):
    """MultiFormer: CSI → TFDDT → DualAttention → MSFN → PCM/PAF.

    输入: (B, 64, 3, 114) memmap CSI (time × ant × subc).
    输出: list of (pcm(B,19,36,36), paf(B,38,36,36)) × 3 stages.
    """

    def __init__(
        self,
        time_packets: int = 64,
        rx_antennas: int = 3,
        input_subcarriers: int = 114,
        embed_dim: int = 1296,
        num_heads: int = 8,
        depth: int = 8,
        dropout: float = 0.1,
        heatmap_size: int = 36,
        recon_channels: int = 64,
        feature_channels: int = 256,
        decoder_mid: int = 128,
        decoder_hidden: int = 512,
        stages: int = 3,
    ) -> None:
        super().__init__()
        self.tokenizer = TFDDTTokenizer(
            time_packets=time_packets,
            rx_antennas=rx_antennas,
            input_subcarriers=input_subcarriers,
            embed_dim=embed_dim,
        )
        self.extractor = DualAttentionExtractor(
            freq_tokens=input_subcarriers,
            time_tokens=time_packets,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            dropout=dropout,
            heatmap_size=heatmap_size,
            recon_channels=recon_channels,
        )
        self.input_proj = nn.Sequential(
            nn.Conv2d(recon_channels * 2, feature_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.msfn = MSFN(
            feature_channels=feature_channels,
            decoder_mid=decoder_mid,
            decoder_hidden=decoder_hidden,
            stages=stages,
        )

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        freq_tokens, time_tokens = self.tokenizer(x)
        features = self.extractor(freq_tokens, time_tokens)
        features = self.input_proj(features)
        return self.msfn(features)


def multistage_pose_loss(
    predictions: list[tuple[torch.Tensor, torch.Tensor]],
    target_pcm: torch.Tensor,
    target_paf: torch.Tensor,
    paf_loss_weight: float = 0.5,
) -> torch.Tensor:
    """Stage-wise MSE loss with weighted PAF auxiliary loss.

    Loss = Σ_i [ MSE(PCM_i, target) + paf_loss_weight × MSE(PAF_i, target) ]
    """
    if not predictions:
        raise ValueError("Expected at least one prediction stage")

    loss = target_pcm.new_tensor(0.0)
    for predicted_pcm, predicted_paf in predictions:
        loss = loss + F.mse_loss(predicted_pcm, target_pcm)
        loss = loss + paf_loss_weight * F.mse_loss(predicted_paf, target_paf)
    return loss
