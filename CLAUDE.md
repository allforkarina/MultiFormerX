# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

MultiFormerX is a PyTorch reproduction of the MultiFormer paper for WiFi-based single-person pose estimation. The model takes mmap'd CSI amplitude tensors and outputs Part Confidence Maps (PCM) and Part Affinity Fields (PAF) heatmaps via a three-stage iterative network.

## Commands

```bash
# Compile-check all source files (minimum validation after any edit)
python -m py_compile model/tfddt.py model/attention.py model/heatmap_decoder.py model/papm.py model/msfn.py model/multiformer.py data/memmap_dataset.py data/pose_targets.py eval/metrics.py train.py

# Full shape smoke test (no dataset needed)
python -c "
from model import MultiFormer, multistage_pose_loss
import torch
model = MultiFormer()
outputs = model(torch.randn(2, 64, 3, 114))
assert len(outputs) == 3
assert outputs[-1][0].shape == (2, 19, 36, 36)
print('OK')
"

# Training (requires pre-built memmap .npy files on a Linux server)
python train.py \
    --data-dir /data/WiFiPose/dataset/mmfi_pose_v3 \
    --normalize global_minmax \
    --train-subjects S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 \
    --batch-size 32 --epochs 50 --paf-loss-weight 0.5
```

## Architecture

**Data flow:** CSI → TFDDT (dual-domain tokenization) → Dual Transformer (freq + time branches) → Reconstruction → Concat → MSFN (3-stage iterative refinement) → PCM/PAF heatmaps → argmax → keypoints.

**CSI shape:** `(B, 64, 3, 114)` = time packets × antennas × subcarriers. This is MM-Fi data pre-upsampled by `build_memmap.py` (on the Linux server), stored as memory-mapped `.npy` files. Three normalization variants exist: `global_minmax`, `global_zscore`, `zscore`.

**Keypoint coordinate system:** `pose_range = [-0.8, 0.8]`. All keypoints (input targets and decoded predictions) use this range. `generate_pose_targets` maps from pose_range to `[0, 35]` heatmap grid. `heatmaps_to_keypoints` decodes argmax → `[0, 1]` → maps back to pose_range.

**embed_dim = 1296 = 36×36** — tokens project directly to heatmap spatial dimensions. The transformer QKV operates on 1296-dim, making this a large model (~219M params). ReconstructionLayer uses `nn.Identity()` (no projection needed) since embed_dim already equals the target heatmap dim.

**Frequency tokens = 114** (one per subcarrier), not 64 as in the paper. This reflects MM-Fi's 114 subcarriers vs. the paper's 30→64 upsampled. **Time tokens = 64** (matching the paper). The two branches have independent position embeddings of different sizes.

**MSFN** receives 128ch features (64+64 concat) → `input_proj` 128→256 → three HeatmapDecoder stages with PAPM feedback between stages. Each PAPM takes 57ch heatmaps (19 PCM + 38 PAF) as attention signal.

**PCK metric:** Torso-normalized — distance between predicted and target keypoints divided by right-shoulder to left-hip distance, with threshold default 0.20.

**PAF loss:** Weighted at `paf_loss_weight = 0.5` (configurable via CLI). PAF is auxiliary only — not used in single-person argmax inference.

## Key modules

| Module | Role |
|--------|------|
| `model/tfddt.py` | CSI → freq/time token sequences with independent position embeddings |
| `model/attention.py` | Pre-LN TransformerBlock + ReconstructionLayer + DualAttentionExtractor |
| `model/heatmap_decoder.py` | 4-layer shared conv trunk → PCM/PAF 1×1 heads |
| `model/papm.py` | Channel attention (MLP) + spatial attention (Conv2d) from heatmap feedback |
| `model/msfn.py` | 3-stage decoder stack with PAPM between stages |
| `model/multiformer.py` | Top-level MultiFormer + `multistage_pose_loss` |
| `data/memmap_dataset.py` | Memory-mapped `.npy` dataset, subject-based 80/20 train/val split |
| `data/pose_targets.py` | PCM/PAF target generation from OpenPose18 keypoints |
| `eval/metrics.py` | Heatmap argmax decoding + torso-normalized PCK |

## Design constraints

- **Single-person only.** No NMS, no Hungarian bipartite matching, no multi-person Pose Decoder.
- **No Teacher-Student framework.** Targets are generated from COCO keypoints, not OpenPose video.
- **MM-Fi specific.** 114 subcarriers, 64 time packets, OpenPose BODY_18 keypoint ordering.
- **Memmap pipeline assumed.** Training expects pre-built `.npy` files from `build_memmap.py` on the server.
- **AdamW with weight_decay=1e-4.** The paper's Table II erroneously lists "weight decay = 0.7" (actually the LR gamma).
- **No `scipy.signal.resample` in the model pipeline.** Upsampling is done once during memmap preprocessing.
- **`model/__init__.py` and `data/__init__.py` provide re-exports.** Use `from model import MultiFormer` not `from model.multiformer import MultiFormer`.
