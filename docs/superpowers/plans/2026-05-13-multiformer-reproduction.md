# MultiFormer 论文复现实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 MultiFormerX 的单文件 model.py 按论文架构拆分为 6 个模块，修正 TFDDT/Reconstruction/PAPM 维度对齐论文，切换到 memmap 数据管道，修正优化器和训练超参数。

**Architecture:** 参考 multiformer 项目的模块化设计，遵循论文 TFDDT → DualAttention → Reconstruction → MSFN 拓扑，CSI shape (B, 64, 3, 114)，embed_dim=1296，pose_range=[-0.8, 0.8]。

**Tech Stack:** PyTorch, NumPy, scipy (仅 memmap 读取不依赖 scipy.resample)

**参考实现:** `D:\Files\Projects\PythonProjects\PaperResuming\multiformer/models/*.py`
**设计文档:** `docs/superpowers/specs/2026-05-13-multiformer-reproduction-design.md`

---

### Task 1: 创建 model/tfddt.py — TFDDT 时频双维度 Token 化

**Files:**
- Create: `model/tfddt.py`
- Test: smoke via `model/multiformer.py` (Task 6)

- [ ] **Step 1: 写入 TFDDTTokenizer 模块**

```python
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
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -m py_compile model/tfddt.py
```

预期: 无输出（编译通过）

- [ ] **Step 3: Commit**

```bash
git add model/tfddt.py
git commit -m "feat: add TFDDTTokenizer with embed_dim=1296 and independent position embeddings"
```

---

### Task 2: 创建 model/attention.py — TransformerBlock + ReconstructionLayer + DualAttentionExtractor

**Files:**
- Create: `model/attention.py`
- Test: smoke via `model/multiformer.py` (Task 6)

- [ ] **Step 1: 写入 attention 模块**

```python
from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    """MultiFormer self-attention block: Pre-LN → MHA → residual → Pre-LN → FFN → residual."""

    def __init__(self, embed_dim: int = 1296, num_heads: int = 8, dropout: float = 0.1, ffn_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(embed_dim * ffn_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class ReconstructionLayer(nn.Module):
    """Reconstruct 1D token features into 2D spatial feature maps (论文模块 B).

    embed_dim=1296 → reshape to (token_count, 36, 36) → Conv2d → BN → ReLU.
    """

    def __init__(
        self,
        token_count: int,
        embed_dim: int = 1296,
        heatmap_size: int = 36,
        out_channels: int = 64,
    ) -> None:
        super().__init__()
        self.token_count = token_count
        self.heatmap_size = heatmap_size
        target_dim = heatmap_size * heatmap_size
        self.to_heatmap = nn.Identity() if embed_dim == target_dim else nn.Linear(embed_dim, target_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(token_count, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, token_count, _ = tokens.shape
        if token_count != self.token_count:
            raise ValueError(f"Expected {self.token_count} tokens, got {token_count}")
        x = self.to_heatmap(tokens)
        x = x.reshape(bsz, token_count, self.heatmap_size, self.heatmap_size)
        return self.conv(x)


class DualAttentionExtractor(nn.Module):
    """独立参数的双路 Transformer + Reconstruction (论文模块 B).

    频率路: 114 tokens → Reconstruction(114→64ch)
    时间路:  64 tokens → Reconstruction(64→64ch)
    Concat → (B, 128, 36, 36)
    """

    def __init__(
        self,
        freq_tokens: int = 114,
        time_tokens: int = 64,
        embed_dim: int = 1296,
        num_heads: int = 8,
        depth: int = 8,
        dropout: float = 0.1,
        heatmap_size: int = 36,
        recon_channels: int = 64,
    ) -> None:
        super().__init__()
        self.freq_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.time_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.freq_recon = ReconstructionLayer(freq_tokens, embed_dim, heatmap_size, recon_channels)
        self.time_recon = ReconstructionLayer(time_tokens, embed_dim, heatmap_size, recon_channels)

    def forward(self, freq_tokens: torch.Tensor, time_tokens: torch.Tensor) -> torch.Tensor:
        for block in self.freq_blocks:
            freq_tokens = block(freq_tokens)
        for block in self.time_blocks:
            time_tokens = block(time_tokens)
        freq_map = self.freq_recon(freq_tokens)
        time_map = self.time_recon(time_tokens)
        return torch.cat([freq_map, time_map], dim=1)
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -m py_compile model/attention.py
```

- [ ] **Step 3: Commit**

```bash
git add model/attention.py
git commit -m "feat: add DualAttentionExtractor with Pre-LN Transformer blocks and ReconstructionLayer"
```

---

### Task 3: 创建 model/heatmap_decoder.py — HeatmapDecoder

**Files:**
- Create: `model/heatmap_decoder.py`

- [ ] **Step 1: 写入 HeatmapDecoder**

```python
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
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -m py_compile model/heatmap_decoder.py
```

- [ ] **Step 3: Commit**

```bash
git add model/heatmap_decoder.py
git commit -m "feat: add HeatmapDecoder with shared conv trunk and PCM/PAF heads"
```

---

### Task 4: 创建 model/papm.py — PoseAttentivePerceptionModule

**Files:**
- Create: `model/papm.py`

- [ ] **Step 1: 写入 PAPM 模块**

```python
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
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -m py_compile model/papm.py
```

- [ ] **Step 3: Commit**

```bash
git add model/papm.py
git commit -m "feat: add PAPM with channel MLP and spatial Conv2d attention"
```

---

### Task 5: 创建 model/msfn.py — MultiStageFeatureFusionNetwork

**Files:**
- Create: `model/msfn.py`

- [ ] **Step 1: 写入 MSFN 模块**

```python
from __future__ import annotations

import torch
from torch import nn

from .heatmap_decoder import HeatmapDecoder
from .papm import PAPM


class MSFN(nn.Module):
    """Multi-Stage Feature Fusion Network (论文模块 C).

    三阶段迭代精修:
      Φ₀(128ch) → input_proj(128→256) → Decoder₁ → H₁
        → PAPM(H₁) → Φ₁ → Decoder₂ → H₂
        → PAPM(H₂) → Φ₂ → Decoder₃ → H₃ (final)
    """

    def __init__(
        self,
        feature_channels: int = 256,
        decoder_mid: int = 128,
        decoder_hidden: int = 512,
        stages: int = 3,
        pcm_channels: int = 19,
        paf_channels: int = 38,
    ) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError("MSFN needs at least one stage")

        self.decoders = nn.ModuleList([
            HeatmapDecoder(
                in_channels=feature_channels,
                mid_channels=decoder_mid,
                hidden_channels=decoder_hidden,
                pcm_channels=pcm_channels,
                paf_channels=paf_channels,
            )
            for _ in range(stages)
        ])
        self.papms = nn.ModuleList([
            PAPM(feature_channels=feature_channels, heat_channels=pcm_channels + paf_channels)
            for _ in range(stages - 1)
        ])

    def forward(self, features: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        current = features
        for idx, decoder in enumerate(self.decoders):
            pcm, paf = decoder(current)
            outputs.append((pcm, paf))
            if idx < len(self.papms):
                current = self.papms[idx](current, torch.cat([pcm, paf], dim=1))
        return outputs
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -m py_compile model/msfn.py
```

- [ ] **Step 3: Commit**

```bash
git add model/msfn.py
git commit -m "feat: add MSFN with 3-stage iterative refinement and PAPM feedback"
```

---

### Task 6: 创建 model/multiformer.py — MultiFormer 顶层 + Loss

**Files:**
- Create: `model/multiformer.py`
- Modify: `model/__init__.py`

- [ ] **Step 1: 写入 MultiFormer 和 multistage_pose_loss**

```python
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
        # input_projection: 128ch (concat 64+64) → 256ch
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
```

- [ ] **Step 2: 更新 model/__init__.py**

```python
from .multiformer import MultiFormer, multistage_pose_loss
```

- [ ] **Step 3: 语法检查**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -m py_compile model/multiformer.py && python -c "from model.multiformer import MultiFormer, multistage_pose_loss; print('import OK')"
```

- [ ] **Step 4: Commit**

```bash
git add model/multiformer.py model/__init__.py
git commit -m "feat: add MultiFormer top-level module with configurable PAF loss weight"
```

---

### Task 7: 删除旧 model/model.py

**Files:**
- Delete: `model/model.py`

- [ ] **Step 1: 删除旧文件并验证导入**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && rm model/model.py && python -c "from model.multiformer import MultiFormer, multistage_pose_loss; print('import OK')"
```

预期: `import OK`

- [ ] **Step 2: Commit**

```bash
git add model/model.py
git commit -m "refactor: remove old monolithic model.py, replaced by modular structure"
```

---

### Task 8: 创建 data/memmap_dataset.py — Memmap 数据管道

**Files:**
- Create: `data/memmap_dataset.py`

- [ ] **Step 1: 写入 MemmapDataset**

```python
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .pose_targets import generate_pose_targets


CSI_FILES = {
    "global_minmax": "csi_gminmax.npy",
    "global_zscore": "csi_gzscore.npy",
    "zscore": "csi_zscore.npy",
}


class MemmapDataset(Dataset):
    """Memory-mapped .npy dataset for fast training I/O.

    CSI is stored as pre-normalized .npy files, read via np.load(mmap_mode='r').
    Keypoints and meta are small enough to load entirely into RAM at init.
    PCM/PAF targets are generated on-the-fly in __getitem__.

    Split logic: group by subject, shuffle frames within each subject,
    80/20 train/val split per subject.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        envs: Iterable[str] | None = None,
        train_subjects: Iterable[str] | None = None,
        test_subjects: Iterable[str] | None = None,
        random_val_ratio: float = 0.2,
        seed: int = 42,
        normalize: str = "global_minmax",
        heatmap_size: int = 36,
        heatmap_sigma: float = 1.5,
        paf_width: float = 1.0,
        pose_range: tuple[float, float] = (-0.8, 0.8),
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be train/val/test/all, got {split}")
        if normalize not in CSI_FILES:
            raise ValueError(f"Unknown normalize mode: {normalize}, expected one of {list(CSI_FILES)}")

        self.split = split
        self.normalize = normalize
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.paf_width = paf_width
        self.pose_range = pose_range

        data_dir = Path(data_dir)

        self._csi = np.load(str(data_dir / CSI_FILES[normalize]), mmap_mode="r")
        self._kpts18 = np.load(str(data_dir / "kpts18.npy"))
        meta = np.load(str(data_dir / "meta.npz"), allow_pickle=True)
        self._envs = meta["environment"]
        self._samples = meta["sample"]
        self._actions = meta["action"]

        self.indices = self._build_split(split, envs, train_subjects, test_subjects, random_val_ratio, seed)

    def _build_split(
        self,
        split: str,
        envs: Iterable[str] | None,
        train_subjects: Iterable[str] | None,
        test_subjects: Iterable[str] | None,
        random_val_ratio: float,
        seed: int,
    ) -> np.ndarray:
        num_total = len(self._samples)
        env_set = set(envs) if envs else None
        train_set = set(train_subjects) if train_subjects else None
        test_set = set(test_subjects) if test_subjects else None

        candidate_indices: list[int] = []
        for i in range(num_total):
            if env_set is not None and str(self._envs[i]) not in env_set:
                continue
            sample = str(self._samples[i])
            if split == "train" and train_set is not None and sample not in train_set:
                continue
            if split in {"val", "test"} and train_set is not None and sample in train_set:
                continue
            if split == "test" and test_set is not None and sample not in test_set:
                continue
            candidate_indices.append(i)

        if split == "all":
            return np.asarray(sorted(candidate_indices), dtype=np.int64)

        rng = random.Random(seed)
        grouped: dict[str, list[int]] = {}
        for idx in candidate_indices:
            grouped.setdefault(str(self._samples[idx]), []).append(idx)

        train_indices: list[int] = []
        val_indices: list[int] = []
        for subject, indices in sorted(grouped.items()):
            shuffled = indices[:]
            rng.shuffle(shuffled)
            pivot = int(round(len(shuffled) * (1.0 - random_val_ratio)))
            train_indices.extend(shuffled[:pivot])
            val_indices.extend(shuffled[pivot:])

        chosen = train_indices if split == "train" else val_indices
        return np.asarray(sorted(chosen), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        frame_idx = int(self.indices[index])
        csi = np.array(self._csi[frame_idx])
        kpts18 = self._kpts18[frame_idx].copy()

        pcm, paf = generate_pose_targets(
            kpts18,
            heatmap_size=self.heatmap_size,
            heatmap_sigma=self.heatmap_sigma,
            paf_width=self.paf_width,
            pose_range=self.pose_range,
        )

        return {
            "csi": torch.from_numpy(csi),
            "kpts18": torch.from_numpy(np.ascontiguousarray(kpts18)),
            "pcm": torch.from_numpy(pcm),
            "paf": torch.from_numpy(paf),
            "meta": {
                "env": str(self._envs[frame_idx]),
                "subject": str(self._samples[frame_idx]),
                "action": str(self._actions[frame_idx]),
                "frame_idx": int(frame_idx),
            },
        }
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -m py_compile data/memmap_dataset.py
```

- [ ] **Step 3: Commit**

```bash
git add data/memmap_dataset.py
git commit -m "feat: add MemmapDataset with 3 normalization variants and subject-based split"
```

---

### Task 9: 修改 data/pose_targets.py — 适配 pose_range 坐标系

**Files:**
- Modify: `data/pose_targets.py`

- [ ] **Step 1: 修改 generate_pose_targets**

将 `generate_pose_targets` 函数签名和坐标映射逻辑改为接受 `pose_range` 参数。memmap 输入已是 OpenPose18 格式，跳过 COCO17→OpenPose18 转换。背景通道改为关键点均值 (对齐论文)。

```python
from __future__ import annotations

"""OpenPose-style PCM/PAF target generation from OpenPose18 keypoints."""

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


def _valid_point(point: np.ndarray) -> bool:
    point = np.asarray(point)
    return bool(np.isfinite(point).all() and not np.allclose(point, 0.0))


def _pose_to_heatmap_coords(
    kpts: np.ndarray,
    size: int = 36,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> np.ndarray:
    kpts = np.asarray(kpts, dtype=np.float32).copy()
    lo, hi = pose_range
    scale = (size - 1) / (hi - lo)
    invalid = ~np.isfinite(kpts).all(axis=-1) | np.all(np.isclose(kpts, 0.0), axis=-1)
    kpts = (kpts - lo) * scale
    kpts = np.clip(kpts, 0, size - 1)
    kpts[invalid] = 0.0
    return kpts.astype(np.float32)


def generate_pose_targets(
    keypoints: np.ndarray,
    heatmap_size: int = 36,
    heatmap_sigma: float = 1.5,
    paf_width: float = 1.0,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate PCM and PAF targets from OpenPose18 keypoints.

    Args:
        keypoints: (18, 2) OpenPose BODY_18 in pose_range coordinates.
        heatmap_size: spatial size (36).
        heatmap_sigma: Gaussian sigma.
        paf_width: limb half-width.
        pose_range: (min, max) of keypoint coordinate range.

    Returns:
        pcm (19, H, W), paf (38, H, W).
        PCM ch0-17: 18 anatomical keypoints, ch18: mean background.
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape != (18, 2):
        raise ValueError(f"Expected keypoints with shape (18, 2), got {keypoints.shape}")

    grid_y, grid_x = np.mgrid[0:heatmap_size, 0:heatmap_size].astype(np.float32)
    points = _pose_to_heatmap_coords(keypoints, size=heatmap_size, pose_range=pose_range)

    valid = np.array([_valid_point(k) for k in keypoints], dtype=bool)

    pcm = np.zeros((19, heatmap_size, heatmap_size), dtype=np.float32)
    for idx, point in enumerate(points):
        if valid[idx]:
            distance_sq = (grid_x - point[0]) ** 2 + (grid_y - point[1]) ** 2
            pcm[idx] = np.exp(-distance_sq / (2.0 * heatmap_sigma**2))
    pcm[18] = pcm[:18].mean(axis=0)

    paf = np.zeros((len(OPENPOSE18_LIMBS) * 2, heatmap_size, heatmap_size), dtype=np.float32)
    for limb_idx, (start_idx, end_idx) in enumerate(OPENPOSE18_LIMBS):
        if not (valid[start_idx] and valid[end_idx]):
            continue
        start = points[start_idx]
        end = points[end_idx]
        limb_vec = end - start
        limb_len = float(np.linalg.norm(limb_vec))
        if limb_len < 1e-6:
            continue

        unit_vec = limb_vec / limb_len
        rel_x = grid_x - start[0]
        rel_y = grid_y - start[1]
        proj = rel_x * unit_vec[0] + rel_y * unit_vec[1]
        perp = np.abs(rel_x * unit_vec[1] - rel_y * unit_vec[0])
        mask = (proj >= 0.0) & (proj <= limb_len) & (perp <= paf_width)

        paf[2 * limb_idx][mask] = unit_vec[0]
        paf[2 * limb_idx + 1][mask] = unit_vec[1]

    return pcm, paf
```

保留文件中原有的 `coco17_to_openpose18` 函数不作修改 (定义在文件开头，HDF5 管道 dataloader.py 仍需要它)。`generate_pose_targets` 现在接受 OpenPose18 格式输入，跳过 COCO17→OpenPose18 转换。

- [ ] **Step 2: 语法检查**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -m py_compile data/pose_targets.py
```

- [ ] **Step 3: Commit**

```bash
git add data/pose_targets.py
git commit -m "fix: adapt generate_pose_targets to pose_range coordinate system"
```

---

### Task 10: 修改 eval/metrics.py — 坐标映射适配 pose_range

**Files:**
- Modify: `eval/metrics.py`

- [ ] **Step 1: 修改 heatmaps_to_keypoints**

新增 `pose_range` 参数，将 argmax 解码出的 `[0, 1]` 归一化坐标映射回 `[pose_min, pose_max]`。保留 `OPENPOSE18_TO_COCO17` 映射。

```python
from __future__ import annotations

"""Single-person pose metrics for MultiFormer PCM outputs."""

import torch


OPENPOSE18_TO_COCO17: tuple[int, ...] = (
    0,   # nose
    15,  # l_eye
    14,  # r_eye
    17,  # l_ear
    16,  # r_ear
    5,   # l_shoulder
    2,   # r_shoulder
    6,   # l_elbow
    3,   # r_elbow
    7,   # l_wrist
    4,   # r_wrist
    11,  # l_hip
    8,   # r_hip
    12,  # l_knee
    9,   # r_knee
    13,  # l_ankle
    10,  # r_ankle
)


def heatmaps_to_keypoints(
    heatmaps: torch.Tensor,
    keypoint_indices: tuple[int, ...] = OPENPOSE18_TO_COCO17,
    pose_range: tuple[float, float] = (-0.8, 0.8),
) -> torch.Tensor:
    """Decode keypoint coordinates from PCM heatmaps.

    Args:
        heatmaps: (B, C, H, W). PCM channels in OpenPose18 order (ch18=bg ignored).
        keypoint_indices: OpenPose PCM channel indices → COCO17 output order.
        pose_range: (min, max) of keypoint coordinate system.

    Returns:
        (B, 17, 2) keypoints in pose_range coordinates.
    """
    if heatmaps.ndim != 4:
        raise ValueError(f"Expected heatmaps with 4 dims, got {heatmaps.shape}")
    max_keypoint_index = max(keypoint_indices)
    if heatmaps.shape[1] <= max_keypoint_index:
        raise ValueError(
            f"Expected heatmaps with at least {max_keypoint_index + 1} channels, "
            f"got {heatmaps.shape[1]}"
        )

    keypoint_heatmaps = heatmaps[:, keypoint_indices]
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

    return torch.stack((x, y), dim=2).reshape(batch_size, len(keypoint_indices), 2)


def pck_score(
    predicted_keypoints: torch.Tensor,
    target_keypoints: torch.Tensor,
    threshold: float = 0.20,
    right_shoulder_index: int = 6,
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
    if predicted_keypoints.ndim != 3 or predicted_keypoints.shape[-1] != 2:
        raise ValueError(f"Expected keypoints shaped (B, K, 2), got {predicted_keypoints.shape}")

    num_keypoints = target_keypoints.shape[1]
    if right_shoulder_index >= num_keypoints or left_hip_index >= num_keypoints:
        raise ValueError(
            f"Torso keypoint indices out of bounds, "
            f"got rs={right_shoulder_index}, lh={left_hip_index}, num={num_keypoints}"
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
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -m py_compile eval/metrics.py
```

- [ ] **Step 3: Commit**

```bash
git add eval/metrics.py
git commit -m "fix: adapt heatmaps_to_keypoints to map to pose_range coordinates"
```

---

### Task 11: 修改 scripts/train.py — 切换到 memmap dataloader + AdamW

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: 重写 train.py**

切换到 memmap dataset、AdamW 优化器、新增 `--normalize` 和 `--paf-loss-weight` CLI 参数。kpts18 直接从 memmap 读取，不需要 dataloader 的 COCO17 映射。

```python
from __future__ import annotations

"""Train MultiFormer on MM-Fi memmap dataset."""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.memmap_dataset import MemmapDataset
from eval.metrics import heatmaps_to_keypoints, pck_score
from model.multiformer import MultiFormer, multistage_pose_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MultiFormer on memmap dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to memmap .npy data directory")
    parser.add_argument("--output-dir", type=str, default="runs/multiformer")
    parser.add_argument("--normalize", type=str, default="global_minmax",
                        choices=["global_minmax", "global_zscore", "zscore"])
    parser.add_argument("--train-subjects", nargs="+", default=None,
                        help="List of training subject IDs, e.g. S01 S02 ... S10")
    parser.add_argument("--val-subjects", nargs="+", default=None)
    parser.add_argument("--random-val-ratio", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=15)
    parser.add_argument("--lr-gamma", type=float, default=0.7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pck-threshold", type=float, default=0.20)
    parser.add_argument("--paf-loss-weight", type=float, default=0.5)
    parser.add_argument("--pose-min", type=float, default=-0.8)
    parser.add_argument("--pose-max", type=float, default=0.8)
    return parser.parse_args()


def run_epoch(
    model: MultiFormer,
    data_loader: DataLoader,
    device: torch.device,
    paf_loss_weight: float = 0.5,
    pose_range: tuple[float, float] = (-0.8, 0.8),
    optimizer: AdamW | None = None,
    pck_threshold: float = 0.20,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_pck = 0.0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch in tqdm(data_loader, dynamic_ncols=True, leave=False):
            csi = batch["csi"].to(device=device, dtype=torch.float32)
            kpts18 = batch["kpts18"].to(device=device, dtype=torch.float32)
            target_pcm = batch["pcm"].to(device=device, dtype=torch.float32)
            target_paf = batch["paf"].to(device=device, dtype=torch.float32)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(csi)
            loss = multistage_pose_loss(predictions, target_pcm, target_paf, paf_loss_weight=paf_loss_weight)

            final_pcm, _ = predictions[-1]
            predicted_keypoints = heatmaps_to_keypoints(final_pcm, pose_range=pose_range)
            batch_pck = pck_score(predicted_keypoints, kpts18, threshold=pck_threshold)

            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = csi.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_pck += float(batch_pck.detach().cpu()) * batch_size
            total_samples += batch_size

    total_samples = max(total_samples, 1)
    return total_loss / total_samples, total_pck / total_samples


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: MultiFormer,
    optimizer: AdamW,
    scheduler: StepLR,
    train_loss: float,
    train_pck: float,
    val_loss: float,
    val_pck: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_loss": train_loss,
        "train_pck": train_pck,
        "val_loss": val_loss,
        "val_pck": val_pck,
    }
    torch.save(checkpoint, output_dir / "last.pt")


def save_metrics_history(output_dir: Path, history: list[dict[str, float]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    fieldnames = ["epoch", "train_loss", "val_loss", "train_pck20", "val_pck20", "lr"]
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_training_curves(output_dir: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping training curve images")
        return

    epochs = [item["epoch"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_loss"] for item in history], label="train loss")
    plt.plot(epochs, [item["val_loss"] for item in history], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_pck20"] for item in history], label="train PCK@20")
    plt.plot(epochs, [item["val_pck20"] for item in history], label="val PCK@20")
    plt.xlabel("epoch"); plt.ylabel("PCK@20")
    plt.ylim(0.0, 1.0); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(output_dir / "pck20_curve.png", dpi=150); plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    pose_range = (args.pose_min, args.pose_max)

    common_kwargs = dict(
        data_dir=args.data_dir,
        normalize=args.normalize,
        pose_range=pose_range,
    )
    train_ds = MemmapDataset(
        split="train",
        train_subjects=args.train_subjects,
        random_val_ratio=args.random_val_ratio,
        seed=args.seed,
        **common_kwargs,
    )
    val_ds = MemmapDataset(
        split="val",
        train_subjects=args.train_subjects,
        random_val_ratio=args.random_val_ratio,
        seed=args.seed,
        **common_kwargs,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = MultiFormer().to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    output_dir = Path(args.output_dir)

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_pck = run_epoch(
            model, train_loader, device,
            paf_loss_weight=args.paf_loss_weight, pose_range=pose_range,
            optimizer=optimizer, pck_threshold=args.pck_threshold,
        )
        val_loss, val_pck = run_epoch(
            model, val_loader, device,
            paf_loss_weight=args.paf_loss_weight, pose_range=pose_range,
            pck_threshold=args.pck_threshold,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        save_checkpoint(output_dir, epoch, model, optimizer, scheduler,
                        train_loss, train_pck, val_loss, val_pck)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_pck20": train_pck, "val_pck20": val_pck, "lr": current_lr,
        })
        save_metrics_history(output_dir, history)
        save_training_curves(output_dir, history)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} train_pck20={train_pck:.4f} "
            f"val_loss={val_loss:.6f} val_pck20={val_pck:.4f} "
            f"lr={current_lr:.6g}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法检查**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -m py_compile scripts/train.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "fix: switch to memmap dataloader, AdamW optimizer, configurable PAF loss weight"
```

---

### Task 12: 端到端 Shape Smoke Test

**Files:**
- None (test only)

- [ ] **Step 1: 运行导入和 shape smoke test**

```bash
cd "D:/Files/Projects/PythonProjects/PaperResuming/MultiFormerX" && python -c "
from model.multiformer import MultiFormer, multistage_pose_loss
from data.pose_targets import generate_pose_targets
from eval.metrics import heatmaps_to_keypoints, pck_score
import torch
import numpy as np

# 1. Model instantiation
model = MultiFormer()
print(f'Model created: {sum(p.numel() for p in model.parameters()):,} params')

# 2. Forward pass with memmap-shaped CSI
dummy_csi = torch.randn(2, 64, 3, 114)
outputs = model(dummy_csi)
assert len(outputs) == 3, f'Expected 3 stages, got {len(outputs)}'
for i, (pcm, paf) in enumerate(outputs):
    assert pcm.shape == (2, 19, 36, 36), f'Stage {i+1} PCM: {pcm.shape}'
    assert paf.shape == (2, 38, 36, 36), f'Stage {i+1} PAF: {paf.shape}'
print('Forward pass shapes OK')

# 3. Pose targets with pose_range
kpts18 = np.random.uniform(-0.8, 0.8, (18, 2)).astype(np.float32)
kpts18[3] = 0.0  # simulate missing keypoint
pcm, paf = generate_pose_targets(kpts18, pose_range=(-0.8, 0.8))
assert pcm.shape == (19, 36, 36), f'PCM target: {pcm.shape}'
assert paf.shape == (38, 36, 36), f'PAF target: {paf.shape}'
print('Pose targets shapes OK')

# 4. Loss with configurable PAF weight
loss = multistage_pose_loss(
    outputs,
    torch.from_numpy(pcm).unsqueeze(0).repeat(2, 1, 1, 1),
    torch.from_numpy(paf).unsqueeze(0).repeat(2, 1, 1, 1),
    paf_loss_weight=0.5,
)
assert loss.item() > 0, 'Loss should be positive'
print(f'Loss={loss.item():.6f}')

# 5. PCK with pose_range coordinates
from eval.metrics import OPENPOSE18_TO_COCO17

pred_kpts = heatmaps_to_keypoints(outputs[-1][0], pose_range=(-0.8, 0.8))
assert pred_kpts.shape == (2, 17, 2), f'Pred keypoints: {pred_kpts.shape}'
tgt_kpts = torch.from_numpy(kpts18[np.array(OPENPOSE18_TO_COCO17)]).unsqueeze(0).repeat(2, 1, 1)
pck = pck_score(pred_kpts, tgt_kpts, threshold=0.20)
print(f'PCK@20={pck.item():.4f}')

print('All smoke tests passed!')
"
```

预期: 所有 shape 检查通过，loss > 0, PCK 在正常范围。

- [ ] **Step 2: Commit**

```bash
git commit --allow-empty -m "test: end-to-end shape smoke test passed"
```

---

### 验证清单

完成所有任务后确认：

- [ ] `model/tfddt.py` — TFDDTTokenizer, embed_dim=1296, 两路独立 pos_embed
- [ ] `model/attention.py` — TransformerBlock (Pre-LN), ReconstructionLayer (token_count→64ch), DualAttentionExtractor
- [ ] `model/heatmap_decoder.py` — 4 层共享 trunk + PCM/PAF heads
- [ ] `model/papm.py` — Channel MLP(57→hidden→256) + Spatial Conv2d(2→1, k=7)
- [ ] `model/msfn.py` — 3-stage with input_proj 128→256 + PAPM feedback
- [ ] `model/multiformer.py` — MultiFormer + multistage_pose_loss (paf_loss_weight)
- [ ] `model/model.py` — 已删除
- [ ] `data/memmap_dataset.py` — 3 种归一化, subject-based split
- [ ] `data/pose_targets.py` — 接受 kpts18 + pose_range
- [ ] `eval/metrics.py` — heatmaps_to_keypoints 映射到 pose_range
- [ ] `scripts/train.py` — memmap dataloader, AdamW, --paf-loss-weight
- [ ] Shape smoke test 通过
