# MultiFormerX 复现修正计划

## 目标
核查并修正 MultiFormerX 复现中 OpenPose18/COCO17 映射、PCM/PAF 通道、token reconstruction 维度和 PCK 计算问题。

## 阶段
| 阶段 | 状态 | 内容 |
|---|---|---|
| 1 | complete | 只读核查当前实现与论文设计偏差 |
| 2 | complete | 确认 COCO17 到 OpenPose18 的映射策略 |
| 3 | complete | 设计最小代码修改方案 |
| 4 | complete | 获得项目 owner 明确确认后实施代码修改 |
| 5 | complete | 编译与 smoke shape 验证 |
| 6 | complete | 清理生成文件并提交推送 |

## 当前决策
- owner 已确认 COCO17→OpenPose18 映射和 reconstruction 修改。
- owner 已确认 PCK 使用 torso-normalized 公式。
- 优先修正语义错误，再考虑性能优化。
- 已采用 `Linear(192→1296) + reshape + Conv2d(token_count→64)` reconstruction。
- 已采用 COCO17 右肩到左髋距离作为 PCK torso scale。
- 已确认提交约束：只提交源码、规划/约束文件和必要文档，不提交生成文件或运行产出。

## 风险
- OpenPose18 的 neck 由 COCO17 左右肩均值估计；任一肩无效则 neck 无效。
- PCK 偏低可能来自关键点通道顺序不一致，而不只是 PCK 函数公式错误。
- PCK torso scale 过小或 torso 点无效时，该样本关键点会被排除在 PCK 统计之外。
