from __future__ import annotations

"""MultiFormer model components aligned with the local MM-Fi dataloader."""

import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import resample


class CSIAmplitudeTokenizer(nn.Module):
    """Convert CSI amplitude frames into MultiFormer frequency and temporal tokens.

    The local dataloader returns CSI amplitude as ``(B, 3, 114, 10)``, i.e.
    antenna, subcarrier, and packet axes. This module keeps all 114 subcarriers
    and uses Fourier resampling to increase the packet axis to 64.
    """

    def __init__(
        self,
        target_packets: int = 64,
        target_subcarriers: int = 114,
        num_antennas: int = 3,
        token_dim: int = 192,
    ) -> None:
        super().__init__()
        frequency_raw_token_dim = target_packets * num_antennas
        temporal_raw_token_dim = target_subcarriers * num_antennas

        self.target_packets = target_packets
        self.target_subcarriers = target_subcarriers
        self.num_antennas = num_antennas
        self.token_dim = token_dim

        self.frequency_projection = nn.Linear(frequency_raw_token_dim, token_dim)
        self.temporal_projection = nn.Linear(temporal_raw_token_dim, token_dim)
        self.frequency_position = nn.Parameter(torch.zeros(1, target_subcarriers, token_dim))
        self.temporal_position = nn.Parameter(torch.zeros(1, target_packets, token_dim))

    def forward(self, csi_amplitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return frequency ``(B, 114, D)`` and temporal ``(B, 64, D)`` tokens."""

        if csi_amplitude.ndim != 4:
            raise ValueError(f"Expected CSI amplitude with 4 dims, got {csi_amplitude.shape}")
        if csi_amplitude.shape[1] != self.num_antennas:
            raise ValueError(
                f"Expected {self.num_antennas} antenna channels, got {csi_amplitude.shape[1]}"
            )
        if csi_amplitude.shape[2] != self.target_subcarriers:
            raise ValueError(
                f"Expected {self.target_subcarriers} subcarriers, got {csi_amplitude.shape[2]}"
            )

        resampled = torch.as_tensor(
            resample(csi_amplitude.detach().cpu().numpy(), self.target_packets, axis=-1),
            dtype=csi_amplitude.dtype,
            device=csi_amplitude.device,
        )

        # Frequency tokens fix one subcarrier and concatenate all packets/antennas.
        frequency_raw = resampled.permute(0, 2, 3, 1).reshape(
            csi_amplitude.shape[0],
            self.target_subcarriers,
            self.target_packets * self.num_antennas,
        )

        # Temporal tokens fix one packet and concatenate all subcarriers/antennas.
        temporal_raw = resampled.permute(0, 3, 2, 1).reshape(
            csi_amplitude.shape[0],
            self.target_packets,
            self.target_subcarriers * self.num_antennas,
        )

        frequency_tokens = self.frequency_projection(frequency_raw) + self.frequency_position
        temporal_tokens = self.temporal_projection(temporal_raw) + self.temporal_position
        return frequency_tokens, temporal_tokens


class TransformerEncoderBlock(nn.Module):
    """One MultiFormer self-attention block: MHA, AddNorm, FFN, AddNorm."""

    def __init__(
        self,
        token_dim: int = 192,
        num_heads: int = 8,
        ffn_hidden_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, token_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attention_output, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.attention_norm(tokens + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(tokens)
        tokens = self.feed_forward_norm(tokens + self.dropout(feed_forward_output))
        return tokens


class DualTokenTransformerEncoder(nn.Module):
    """Independent frequency and temporal Transformer branches."""

    def __init__(
        self,
        token_dim: int = 192,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_hidden_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frequency_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    token_dim=token_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.temporal_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    token_dim=token_dim,
                    num_heads=num_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        frequency_tokens: torch.Tensor,
        temporal_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.frequency_layers:
            frequency_tokens = layer(frequency_tokens)
        for layer in self.temporal_layers:
            temporal_tokens = layer(temporal_tokens)
        return frequency_tokens, temporal_tokens


class MultiFormerTokenEncoder(nn.Module):
    """CSI amplitude tokenizer followed by the dual-token Transformer encoder."""

    def __init__(
        self,
        token_dim: int = 192,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_hidden_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = CSIAmplitudeTokenizer(token_dim=token_dim)
        self.encoder = DualTokenTransformerEncoder(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
        )

    def forward(self, csi_amplitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frequency_tokens, temporal_tokens = self.tokenizer(csi_amplitude)
        return self.encoder(frequency_tokens, temporal_tokens)


class TokenReconstructionLayer(nn.Module):
    """Reconstruct encoded tokens into 2D feature maps."""

    def __init__(
        self,
        token_dim: int = 192,
        token_count: int = 64,
        output_channels: int = 64,
        feature_size: int = 36,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.token_count = token_count
        self.feature_size = feature_size
        self.feature_projection = nn.Linear(token_dim, feature_size * feature_size)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(token_count, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with 3 dims, got {tokens.shape}")
        batch_size, token_count, token_dim = tokens.shape
        if token_dim != self.token_dim:
            raise ValueError(f"Expected token dim {self.token_dim}, got {token_dim}")
        if token_count != self.token_count:
            raise ValueError(f"Expected {self.token_count} tokens, got {token_count}")

        feature_map = self.feature_projection(tokens).reshape(
            batch_size,
            token_count,
            self.feature_size,
            self.feature_size,
        )
        return self.reconstruction(feature_map)


class MultiFormerFeatureExtractor(nn.Module):
    """CSI amplitude to fused 2D feature maps for MultiFormer pose estimation."""

    def __init__(
        self,
        token_dim: int = 192,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_hidden_dim: int = 768,
        dropout: float = 0.1,
        branch_channels: int = 64,
        feature_size: int = 36,
    ) -> None:
        super().__init__()
        self.token_encoder = MultiFormerTokenEncoder(
            token_dim=token_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
        )
        self.frequency_reconstruction = TokenReconstructionLayer(
            token_dim=token_dim,
            token_count=114,
            output_channels=branch_channels,
            feature_size=feature_size,
        )
        self.temporal_reconstruction = TokenReconstructionLayer(
            token_dim=token_dim,
            token_count=64,
            output_channels=branch_channels,
            feature_size=feature_size,
        )

    def forward(self, csi_amplitude: torch.Tensor) -> torch.Tensor:
        frequency_tokens, temporal_tokens = self.token_encoder(csi_amplitude)
        frequency_features = self.frequency_reconstruction(frequency_tokens)
        temporal_features = self.temporal_reconstruction(temporal_tokens)
        return torch.cat((frequency_features, temporal_features), dim=1)


class HeatmapDecoder(nn.Module):
    """Decode one MSFN feature map into PCM and PAF heatmaps."""

    def __init__(
        self,
        input_channels: int = 256,
        pcm_channels: int = 19,
        paf_channels: int = 38,
    ) -> None:
        super().__init__()
        self.pcm_decoder = self._make_branch(input_channels, pcm_channels)
        self.paf_decoder = self._make_branch(input_channels, paf_channels)

    @staticmethod
    def _make_branch(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, output_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pcm_decoder(features), self.paf_decoder(features)


class PoseAttentivePerceptionModule(nn.Module):
    """PAPM channel-spatial attention feedback from pose probabilities."""

    def __init__(
        self,
        feature_channels: int = 256,
        pose_channels: int = 57,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        hidden_channels = max(feature_channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Linear(pose_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, feature_channels),
        )
        self.spatial_attention = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(
        self,
        features: torch.Tensor,
        previous_pcm: torch.Tensor,
        previous_paf: torch.Tensor,
    ) -> torch.Tensor:
        pose_probabilities = torch.cat((previous_pcm, previous_paf), dim=1)

        avg_pool = pose_probabilities.mean(dim=(2, 3))
        max_pool = pose_probabilities.amax(dim=(2, 3))
        channel_weights = torch.sigmoid(
            self.channel_mlp(avg_pool) + self.channel_mlp(max_pool)
        ).unsqueeze(-1).unsqueeze(-1)

        spatial_avg = pose_probabilities.mean(dim=1, keepdim=True)
        spatial_max = pose_probabilities.amax(dim=1, keepdim=True)
        spatial_weights = torch.sigmoid(
            self.spatial_attention(torch.cat((spatial_avg, spatial_max), dim=1))
        )

        return features * channel_weights * spatial_weights


class MultiStageFeatureFusionNetwork(nn.Module):
    """Three-stage MSFN that iteratively refines PCM and PAF predictions."""

    def __init__(
        self,
        input_channels: int = 128,
        feature_channels: int = 256,
        pcm_channels: int = 19,
        paf_channels: int = 38,
        num_stages: int = 3,
    ) -> None:
        super().__init__()
        if num_stages != 3:
            raise ValueError("The standard MultiFormer reproduction uses exactly 3 MSFN stages")

        self.input_projection = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.decoders = nn.ModuleList(
            [
                HeatmapDecoder(
                    input_channels=feature_channels,
                    pcm_channels=pcm_channels,
                    paf_channels=paf_channels,
                )
                for _ in range(num_stages)
            ]
        )
        self.papms = nn.ModuleList(
            [
                PoseAttentivePerceptionModule(
                    feature_channels=feature_channels,
                    pose_channels=pcm_channels + paf_channels,
                )
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, features: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        refined_features = self.input_projection(features)
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

        pcm, paf = self.decoders[0](refined_features)
        outputs.append((pcm, paf))

        for stage_index, papm in enumerate(self.papms, start=1):
            refined_features = papm(refined_features, pcm, paf)
            pcm, paf = self.decoders[stage_index](refined_features)
            outputs.append((pcm, paf))

        return outputs


class MultiFormer(nn.Module):
    """Standard MultiFormer reproduction adapted to local amplitude-only CSI."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = MultiFormerFeatureExtractor()
        self.pose_estimator = MultiStageFeatureFusionNetwork()

    def forward(self, csi_amplitude: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        features = self.feature_extractor(csi_amplitude)
        return self.pose_estimator(features)


def multistage_pose_loss(
    predictions: list[tuple[torch.Tensor, torch.Tensor]],
    target_pcm: torch.Tensor,
    target_paf: torch.Tensor,
) -> torch.Tensor:
    """Stage-wise MSE loss for MultiFormer PCM and PAF predictions."""

    if not predictions:
        raise ValueError("Expected at least one prediction stage")

    loss = target_pcm.new_tensor(0.0)
    for predicted_pcm, predicted_paf in predictions:
        loss = loss + F.mse_loss(predicted_pcm, target_pcm)
        loss = loss + F.mse_loss(predicted_paf, target_paf)
    return loss
