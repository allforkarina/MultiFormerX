# 发现记录

## 初始观察
- 当前项目 `pose_targets.py` 使用 COCO17 原始顺序生成 PCM：17 个关键点 + 1 个背景 = 18 通道。
- 当前项目 PAF 使用 `COCO17_LIMBS` 的 19 条 limb，因此 PAF 为 38 通道。
- 这造成 PCM 通道语义不是 OpenPose BODY_18，也没有 neck 通道。
- PCK 当前从最终 PCM 的前 17 通道 argmax 解码，并直接与 COCO17 keypoints 比较。

## 待核查
- 是否需要将训练 target 和评估 target 都统一到 OpenPose18 顺序。
- Frequency/Temporal reconstruction 是否应按 embedding dim `D=192` 重构到 `36×36`，而不是按 token count `114/64` 作为空间 grid。
- PCK 偏低是否主要来自通道顺序、坐标尺度、背景/neck 通道或阈值定义。

## 代码核查结果
- `pose_targets.py` 当前没有 COCO17→OpenPose18 映射，直接按 COCO17 索引生成 17 个关节点热图，再追加背景通道。
- `pose_targets.py` 当前 PAF limb 数为 19，输出 38 通道；但 PCM 是 18 通道，其中实际是 17 关键点 + 背景，不是 OpenPose18 关键点。
- `model.py` 当前 `TokenReconstructionLayer` 将 token count 解释为空间 grid：frequency 为 `114×1`，temporal 为 `8×8`，再插值到 `36×36`。
- `metrics.py` 当前 `heatmaps_to_keypoints` 默认解码前 17 个 PCM 通道，并与 COCO17 顺序标签比较；如果模型 target 改为 OpenPose18，评估也必须同步映射回 COCO17 或改用 OpenPose18 target。
- `train.py` CLI 默认 `--pck-threshold=0.20`，但 `run_epoch` 函数默认是 `0.05`；主训练路径会传入 CLI 值，直接调用 `run_epoch` 时可能用错阈值。

## 论文核查结果
- 论文 TFDDT 将 CSI 从 `10×3×30` 上采样到 `64×3×64`，产生 64 个 frequency tokens 和 64 个 temporal tokens。
- 论文 token embedding 维度为 `1296=36×36`，reconstruction layer 明确将 1D feature reshape 为 `36×36`，再接 `3×3 conv + BN + ReLU`。
- 当前项目保留 MM-Fi 的 `114` 子载波，并把 token count 当空间 grid，这不符合论文 reconstruction 语义。
- 论文 MSFN 中 `PCM Pi ∈ R^{19×36×36}`，对应 18 个 anatomical keypoints + 1 个 average/background-like 通道；`PAF Ai ∈ R^{38×36×36}`，对应 19 条 limb。
- 论文 PCK 公式使用 `||pd_j - gt_j|| / sqrt(rs^2 + lh^2) <= alpha`；按文字描述，`rs` 和 `lh` 分别与右肩、左髋位置相关。当前代码直接在归一化坐标中比较欧氏距离，没有按身体尺度归一。

## 已实施修改
- `pose_targets.py` 新增 `coco17_to_openpose18`，按 OpenPose BODY_18 顺序生成 18 个 anatomical keypoints，neck 使用左右肩均值。
- `pose_targets.py` PCM 改为 `(19,36,36)`，PAF 保持 `(38,36,36)` 并改用 OpenPose18 limb 连接。
- `model.py` reconstruction 改为 `Linear(192→1296)`，再 reshape 为 `(B, token_count, 36, 36)`，最后 `Conv2d(token_count→64)`。
- `model.py` frequency reconstruction 为 `(B,114,192) → (B,114,1296) → (B,114,36,36) → (B,64,36,36)`。
- `model.py` temporal reconstruction 为 `(B,64,192) → (B,64,1296) → (B,64,36,36) → (B,64,36,36)`。
- `model.py` PCM channel 改为 19，PAPM pose channel 自动对应 57。
- `metrics.py` 解码默认从 OpenPose18 PCM 通道映射回 COCO17 顺序，避免与 dataloader 的 COCO17 labels 顺序错位。
- `metrics.py` PCK 改为 torso-normalized：默认使用 COCO17 `right_shoulder=6` 和 `left_hip=11` 的距离作为尺度，判断 `distance / torso_scale <= alpha`。
- `train.py` 中 `run_epoch` 的默认 PCK 阈值改为 `0.20`，与 CLI 默认 `--pck-threshold=0.20` 保持一致。
- `.gitignore` 已添加，忽略 Python 缓存、虚拟环境、本地缓存、数据集、训练输出、checkpoint、日志和常见图像产物。

## 验证结果
- `python -m py_compile train.py metrics.py model.py dataloader.py pose_targets.py scripts\build_h5_dataset.py` 通过。
- CPU smoke：`generate_pose_targets` 输出 `pcm=(19,36,36)`、`paf=(38,36,36)`。
- CPU smoke：`MultiFormer(torch.randn(2,3,114,10))` 输出 3 个 stage，每个 stage 为 `pcm=(2,19,36,36)`、`paf=(2,38,36,36)`。
- CPU smoke：`heatmaps_to_keypoints` 输出 `(2,17,2)`，可继续与 COCO17 labels 对齐。
- PCK smoke：构造 torso scale 为 1 的样本，完全重合输出 `1.0`，整体偏移 `0.30` 且阈值 `0.20` 输出 `0.0`。

## Git 约束
- 提交应包含源码、项目约束/规划文件和必要文档。
- 不提交 `__pycache__/`、`.pyc`、`data/`、`runs/`、checkpoint、HDF5 数据集、本地缓存和训练产出。
