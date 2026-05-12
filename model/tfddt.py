from __future__ import annotations

import torch
from torch import nn


class TFDDTTokenizer(nn.Module):
    """Time-Frequency Dual-Dimensional Tokenization (论文模块 A).

    将 memmap 预上采样后的 CSI 分解为频率 token 和时间 token 序列。
    输入 shape: (B, M=64, NR=3, NS=114) — time × antennas × subcarriers.
    """

    def __init__(
        self,
        time_packets: int = 64,
        rx_antennas: int = 3,
        input_subcarriers: int = 114,
        embed_dim: int = 1296,
    ) -> None:
        super().__init__()
        self.time_packets = time_packets
        self.rx_antennas = rx_antennas
        self.input_subcarriers = input_subcarriers

        freq_raw_dim = time_packets * rx_antennas  # 192
        time_raw_dim = input_subcarriers * rx_antennas  # 342

        self.freq_proj = nn.Linear(freq_raw_dim, embed_dim)
        self.time_proj = nn.Linear(time_raw_dim, embed_dim)

        self.freq_pos_embed = nn.Parameter(torch.zeros(1, input_subcarriers, embed_dim))
        self.time_pos_embed = nn.Parameter(torch.zeros(1, time_packets, embed_dim))
        nn.init.trunc_normal_(self.freq_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, M=64, NR=3, NS=114) -> freq (B, 114, 1296), time (B, 64, 1296)."""
        if x.ndim != 4:
            raise ValueError(f"Expected x shape (B, M, NR, NS), got {tuple(x.shape)}")
        bsz, packets, rx, subcarriers = x.shape
        if packets != self.time_packets or rx != self.rx_antennas or subcarriers != self.input_subcarriers:
            raise ValueError(
                f"Expected (M, NR, NS)=({self.time_packets}, {self.rx_antennas}, {self.input_subcarriers}), "
                f"got ({packets}, {rx}, {subcarriers})"
            )

        freq_raw = x.permute(0, 3, 1, 2).reshape(bsz, subcarriers, packets * rx)
        time_raw = x.permute(0, 1, 3, 2).reshape(bsz, packets, subcarriers * rx)

        freq_tokens = self.freq_proj(freq_raw) + self.freq_pos_embed
        time_tokens = self.time_proj(time_raw) + self.time_pos_embed

        return freq_tokens, time_tokens
