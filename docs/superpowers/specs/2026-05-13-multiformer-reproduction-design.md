# MultiFormerX 论文复现设计

> 基于 MultiFormer 论文 (MultiFormer.pdf) 四维度分析，在 MultiFormerX 项目基础上完整复现模型架构，适配 MM-Fi 数据集和 memmap 数据管道。

## 1. 决策汇总

| 决策项 | 选择 | 说明 |
|--------|------|------|
| 复现范围 | 完整论文架构，单人输出 | 不含多人 Pose Decoder (NMS + Hungarian) |
| 子载波策略 | keep (114) | 不下采样到 64，保留 MM-Fi 全部频域信息 |
| 姿态解码 | 单人 argmax | PCM 热图 argmax → OpenPose18→COCO17 映射 |
| embed_dim | 1296 (=36×36) | 完整复现论文设置，后续可调参 |
| 归一化 | 3 种变体, 参数选择 | global_minmax / global_zscore / zscore |
| 基础项目 | MultiFormerX | 参考 multiformer 模块设计，multiFormer 项目不动 |
| 数据管道 | memmap `.npy` | 零拷贝 mmap 读取，OS page cache 共享 |
| 坐标系 | `[-0.8, 0.8]` (pose_range) | 以 build_memmap.py 输出为准 |
| 模块拆分 | 6 个独立模块文件 | 从 model.py 拆出 |
| Position Embedding | 频率/时间两路独立 | 形状 114≠64, 语义不同, 不可共享 |
| PAF Loss | 可配置权重 `paf_loss_weight=0.5` | 辅助损失, 提供解剖学结构先验 |

## 2. 架构变更对照

### 2.1 TFDDT 模块 (`model/tfddt.py`)

基于论文公式 (3)(4)，将上位机预处理后的 memmap CSI 进行时频双维度 token 化。

```
输入 CSI:  (B, 64, 3, 114)   ← memmap 已预 upsample (时域 10→64)
            time × antennas × subcarriers

频率 token: (B, 114, 1296)    ← 固定子载波 j, 收集 M×NR 维 → Linear(192→1296)
时间 token: (B, 64, 1296)     ← 固定时刻 i, 收集 NS×NR 维 → Linear(342→1296)
            ↓ 各加可学习 position embedding
```

**与论文差异：**
- 论文频域上采样 30→64 (FFT 零插值+低通滤波)；MM-Fi 114 子载波已充足，不做频域操作
- 频率 token 数量 = 114 (非 64)，时间 token 数量 = 64 (与论文一致)
- 论文 token 长度声明为 1296，推导值为 192；本实现直接在 TFDDT 内用 Linear 投影到 1296

### 2.2 双路 Attention + Reconstruction (`model/attention.py`)

基于论文模块 B，独立参数的频率/时间双路自注意力 + 特征图重建。

```
Freq tokens → [TransformerBlock × 8] → [ReconstructionLayer] → (B, 64, 36, 36)
Time tokens → [TransformerBlock × 8] → [ReconstructionLayer] → (B, 64, 36, 36)
                                                                  ↓ concat
                                                              (B, 128, 36, 36) → Φ₀
```

**ReconstructionLayer (per 论文 Section III-B)：**
- embed_dim=1296 → Linear(1296→1296) identity → reshape (B, token_count, 36, 36)
- Conv2d(token_count → 64, k=3, s=1, p=1) + BN + ReLU
- 频率路: token_count=114 → 64ch；时间路: token_count=64 → 64ch
- Concat → 128ch 特征图

**与论文差异：**
- 频率路 Reconstruction 输入 token 数为 114 (论文 64)，输出通道对齐 64

### 2.3 MSFN (`model/msfn.py` + `model/heatmap_decoder.py` + `model/papm.py`)

基于论文模块 C，三阶段迭代姿态精修网络。

```
Φ₀(128ch) → [input_projection 128→256] → Φ₀'(256ch)
              ↓
         [HeatmapDecoder] → P₁(19ch), A₁(38ch)   (H₁ = 57ch)
              ↓
         [PAPM: H₁ → channel_attn + spatial_attn] → Φ₁
              ↓
         [HeatmapDecoder] → P₂(19ch), A₂(38ch)   (H₂)
              ↓
         [PAPM: H₂ → channel_attn + spatial_attn] → Φ₂
              ↓
         [HeatmapDecoder] → P₃(19ch), A₃(38ch)   (H₃, final)
```

**PAPM (论文公式 7-8)：**
- Channel Attention: heatmaps (57ch) → GlobalAvgPool + GlobalMaxPool → SharedMLP → 256ch weights
- Spatial Attention: heatmaps → ChannelAvgPool + ChannelMaxPool → Conv2d(2→1, k=7) → sigmoid
- Feature update: `Φ_i = Φ_{i-1} ⊙ W_C ⊗ W_S`

**与论文差异：**
- input_projection 128→256：两路 64ch concat 后需升维匹配 PAPM 输入
- 论文未显式说明此升维层；本实现添加 1×1 conv + ReLU

## 3. 数据管道变更

### 3.1 新增 `data/memmap_dataset.py`

从 `build_memmap.py` 预处理产物读取，替代原 HDF5 管道。

```
输入文件:
  data_dir/csi_{normalize}.npy    (N, 64, 3, 114) float32, mmap_mode='r'
  data_dir/kpts18.npy             (N, 18, 2) float32, [-0.8, 0.8]
  data_dir/meta.npz               environment, sample, action, frame_idx

输出 (per __getitem__):
  {
    "csi":          Tensor (64, 3, 114),
    "kpts18":       Tensor (18, 2),
    "pcm":          Tensor (19, 36, 36),   ← 动态生成
    "paf":          Tensor (38, 36, 36),   ← 动态生成
    "meta": {"env", "subject", "action", "frame_idx"},
  }
```

Split 逻辑：按 subject 分组，每组内随机 80/20 分配 train/val。

### 3.2 修改 `data/pose_targets.py`

`generate_pose_targets` 新增 `pose_range: tuple[float, float] = (-0.8, 0.8)` 参数。

```
坐标映射: kpts ∈ [pose_min, pose_max] → [0, heatmap_size-1] → 高斯热图
```

输入 kpts 由 memmap 直接提供 (已映射到 `[-0.8, 0.8]`)，跳过 COCO17→OpenPose18 转换（memmap 已存 OpenPose18）。

### 3.3 修改 `eval/metrics.py`

`heatmaps_to_keypoints` 解码坐标映射到 `[-0.8, 0.8]`：

```
argmax → [0, 1] 归一化坐标 → x * (pose_max - pose_min) + pose_min → [-0.8, 0.8]
```

PCK 保持 torso-normalized（右肩-左髋距离归一化），阈值默认 0.20。

## 4. 文件结构

```
MultiFormerX/
├── model/
│   ├── __init__.py
│   ├── tfddt.py              ← TFDDTTokenizer (论文模块 A)
│   ├── attention.py          ← TransformerBlock + ReconstructionLayer (论文模块 B)
│   ├── heatmap_decoder.py    ← HeatmapDecoder (论文模块 C 子组件)
│   ├── papm.py               ← PAPM (论文模块 C 子组件)
│   ├── msfn.py               ← MultiStageFeatureFusionNetwork (论文模块 C)
│   └── multiformer.py        ← MultiFormer + multistage_pose_loss
├── data/
│   ├── __init__.py
│   ├── memmap_dataset.py     ← 新增: memmap 数据读取 + 分 split
│   ├── dataloader.py         ← 保留: HDF5 数据管道 (可选)
│   └── pose_targets.py       ← 修改: pose_range 参数适配
├── eval/
│   ├── __init__.py
│   └── metrics.py            ← 修改: 坐标映射到 pose_range
├── scripts/
│   └── train.py              ← 修改: 切换 memmap dataloader
├── reference/
│   └── MultiFormer.pdf
├── .claude/
│   └── settings.local.json   ← auto-push hook
└── .gitignore
```

## 5. Shape 全链路

```
Layer/Operation                         Output Shape            说明
─────────────────────────────────────────────────────────────────────
[输入] Memmap csi                       (B, 64, 3, 114)         time × ant × subc
                                        (M=64, NR=3, NS=114)

[TFDDT]
freq raw tokens (permute+reshape)       (B, 114, 192)            NS × (M×NR)
time raw tokens (permute+reshape)       (B, 64, 342)             M × (NS×NR)
freq_proj: Linear(192→1296)             (B, 114, 1296)
time_proj: Linear(342→1296)             (B, 64, 1296)
+ pos_embed                             (B, 114/64, 1296)

[Transformer Encoder × 8]
LayerNorm → MHA(8 heads) → residual     (B, 114/64, 1296)
LayerNorm → FFN(1296→2592→1296) → res   (B, 114/64, 1296)

[Reconstruction]
Linear(1296→1296) identity              (B, 114/64, 1296)
Reshape                                 (B, 114/64, 36, 36)
Conv2d(token_count→64, k=3) + BN + ReLU (B, 64, 36, 36)         freq & time 各自

[Concat]
Φ₀                                      (B, 128, 36, 36)

[MSFN input_proj]
Conv2d(128→256, k=1) + ReLU             (B, 256, 36, 36)         Φ₀'

[MSFN Stage 1]
PAPM init (identity)                    (B, 256, 36, 36)
HeatmapDecoder → PCM₁                    (B, 19, 36, 36)          18 kpts + bg
HeatmapDecoder → PAF₁                    (B, 38, 36, 36)          19 limbs × 2 vec

[MSFN Stage 2]
PAPM(H₁) → Φ₂                           (B, 256, 36, 36)
HeatmapDecoder → PCM₂                    (B, 19, 36, 36)
HeatmapDecoder → PAF₂                    (B, 38, 36, 36)

[MSFN Stage 3]
PAPM(H₂) → Φ₃                           (B, 256, 36, 36)
HeatmapDecoder → PCM₃                    (B, 19, 36, 36)
HeatmapDecoder → PAF₃                    (B, 38, 36, 36)

[Output]
argmax PCM₃ → OpenPose18→COCO17         (B, 17, 2)               单人 2D 关键点

[Loss]
MSE(PCM₁₂₃, target_pcm) + paf_loss_weight × MSE(PAF₁₂₃, target_paf)
                                                    paf_loss_weight=0.5 (可配置)
```

## 6. 关键 Shape 一致性检查

| 检查项 | 结论 |
|--------|------|
| Token 维度 1296 = 36×36, reshape 后直接对齐热图空间 | ✅ |
| Φ_i 全程 (B, 256, 36, 36), 各 stage 入出口对齐 | ✅ |
| PAPM channel_attn 输出 (B, 256, 1, 1), spatial_attn (B, 1, 36, 36), 广播后 (B, 256, 36, 36) | ✅ |
| PCM 19ch (18 kpts + bg), PAF 38ch (19 limbs × 2) | ✅ |
| HeatmapDecoder 256→128→128→512→output, 分辨率 36×36 不变 | ✅ |
| freq token=114, time token=64, concat 后 64+64=128ch | ✅ |

## 7. Position Embedding 设计

频率路 (114 tokens) 和时间路 (64 tokens) 必须使用**独立**的 Position Embedding：

```python
# TFDDTTokenizer.__init__
self.freq_pos_embed = nn.Parameter(torch.zeros(1, token_subcarriers, embed_dim))
self.time_pos_embed = nn.Parameter(torch.zeros(1, time_packets, embed_dim))
```

| 约束 | 频率路 | 时间路 |
|------|--------|--------|
| Token 数量 | 114 (子载波) | 64 (时刻) |
| 物理语义 | 子载波间的频谱相关性 | 时刻间的时序依赖 |
| Position Embedding | `(1, 114, 1296)` | `(1, 64, 1296)` |
| 初始化 | trunc_normal(std=0.02) | trunc_normal(std=0.02) |

两路 token 长度不同 (114 ≠ 64)，物理语义不同，不可共享权重。各自独立参数，梯度独立更新。

## 8. PAF 辅助损失设计

推理阶段使用单人 argmax 解码 PCM，PAF 不参与前向推理。但训练时 PAF 分支提供**肢体方向和连通性的结构先验**，对 PCM 学习有正则化作用。

```
Loss = Σ_i [ MSE(PCM_i, target_pcm) + paf_loss_weight × MSE(PAF_i, target_paf) ]
       i ∈ {1, 2, 3}  (3 个 MSFN stage)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `paf_loss_weight` | 0.5 | 可配置超参，PAF 辅助损失的权重衰减 |

- PAF 38 通道 (19 limbs × 2 方向向量) 的 MSE 数值天然大于 PCM 19 通道，不降权会导致 PAF 在梯度中占比过高
- 0.5 权重让 PAF "参与但不主导"，优化主导权保留在 PCM
- 作为 CLI 可配置参数 (`--paf-loss-weight`)，支持消融实验：0.0 (无 PAF)、0.5 (默认)、1.0 (等权)

## 9. 训练策略

### 9.1 损失函数

基于论文公式 (13)-(15)，MSFN 各阶段的 PCM 和 PAF 热图通过 MSE 计算误差：

| 符号 | 公式 | 通道数 |
|------|------|--------|
| PCM 损失 | $f_{pcm}^{i} = \sum_{j=1}^{19} \|PCM_j^i - PCM_j^*\|_2^2$ | J=19 (18 kpts + bg) |
| PAF 损失 | $f_{paf}^{i} = \sum_{c=1}^{38} \|PAF_c^i - PAF_c^*\|_2^2$ | C=38 (19 limbs × 2) |
| 总损失 | $f = \sum_{i=1}^{3}(f_{pcm}^{i} + \texttt{paf\_loss\_weight} \times f_{paf}^{i})$ | n=3 stages |

> **设计偏差**：论文使用等权 ($f_{pcm} + f_{paf}$)，本实现使用可配置的 `paf_loss_weight` (默认 0.5)，理由见 Section 8。

### 9.2 超参数配置

基于论文 Table II，修正了已知的笔误：

| 参数 | 论文值 | 本实现 | 说明 |
|------|--------|--------|------|
| Learning rate | 1e-3 | 1e-3 | 初始学习率 |
| Batch size | 32 | 32 | |
| Epochs | 100 | 50 | 复现实验，50 轮足够观察收敛趋势 |
| Optimizer | — | AdamW | 论文未指定, 使用 AdamW |
| Weight decay | ~~0.7~~ | 1e-4 | ⚠️ 论文笔误: Table II 将 LR gamma 错标为 weight decay |
| LR step size | 15 | 15 | StepLR 衰减周期 |
| LR gamma | 0.7 | 0.7 | 每 15 epochs 衰减至 70% |
| Grad clip norm | — | 1.0 | 论文未指定, 保留现有值 |
| PCK threshold | — | 0.20 | torso-normalized |
| PAF loss weight | — | 0.5 | 可配置, `--paf-loss-weight` |

### 9.3 优化器修正

论文 Table II 将 "0.7" 在 weight decay 行列出，但 0.7 的 L2 惩罚会导致严重梯度消失。结合 "step size=15" 的上下文，0.7 实际是 StepLR 的 gamma。本实现：

```python
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=15, gamma=0.7)
```

## 10. 不作变更的范围

- 不实现多人 Pose Decoder (NMS + PAF 积分 + 匈牙利算法)
- 不引入 Teacher-Student 框架 (Teacher 为 OpenPose)
- 不修改 `multiformer/` 项目
- 不改变 CSI 原始数据的上位机预处理流程 (build_memmap.py)
