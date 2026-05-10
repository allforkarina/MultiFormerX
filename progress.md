# 进度记录

## 2026-05-10
- 收到 owner 对复现偏差的四点反馈。
- 启用 `grill-me`、`planning-with-files-zh`、`deep-learning-pytorch`。
- 建立持久规划文件，进入只读核查阶段。
- 注意：`git status --short` 显示 `__pycache__/train.cpython-312.pyc` 已修改；本轮不处理该生成文件。
- 只读核查 `pose_targets.py`、`model.py`、`metrics.py`、`train.py`，确认 PCM 语义、PAF 通道、reconstruction 和 PCK 链路存在耦合问题。
- 使用 `pdftotext` 核查 `MultiFormer.pdf`，确认论文要求 `1296=36×36` reconstruction、PCM 19 通道、PAF 38 通道和 torso-normalized PCK。
- 生成过临时 `MultiFormer.txt`，已删除。
- 收到 owner 确认：COCO17→OpenPose18 映射采用 neck 均值；reconstruction 采用 `Linear(192→1296)`、token 作为 channel、`Conv2d(token_count→64)`。
- 修改 `pose_targets.py`、`model.py`、`metrics.py`。
- 编译和 CPU shape smoke 均通过。
- 编译生成/更新了 `__pycache__` 下多个 `.pyc` 文件；这些是生成文件，不属于本次代码逻辑变更。
- 收到 owner 确认：PCK 改为 torso-normalized，使用 COCO17 右肩与左髋距离作为尺度。
- 修改 `metrics.py` 和 `train.py`，编译与 PCK smoke 均通过。
- 收到 owner 确认：清理 `__pycache__`，新增 `.gitignore`，提交时只保留源码和约束/规划文件。
- 新增 `.gitignore` 并删除已跟踪/未跟踪的 `__pycache__` 生成文件。
- 已提交并推送到 `origin/main`：`b9c8934 Align pose targets and reconstruction`。
