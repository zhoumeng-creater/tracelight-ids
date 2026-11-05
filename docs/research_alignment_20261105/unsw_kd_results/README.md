# 第十轮 UNSW 四臂验收证据

完整解释与推进决定见[详细验收](../../UNSW_KD_RESULTS_20261105.md)。本目录保存本地独立复算的紧凑记录，原数据、模型和逐行事件留在实验归档与本地分析目录。

| 文件 | 内容 |
|---|---|
| [integrity.json](integrity.json) | 外包与内包SHA、清单、固定源码、目标、共同起点及训练预算核验 |
| [compact_summary.json](compact_summary.json) | 八个R/C终点、三组配对主比较、四个失败指标、集中度 |
| [cost_training_audit.json](cost_training_audit.json) | 原始成本数组复算、顺序重建、160条训练轨迹核验及模型结构 |
| [score_mechanism.json](score_mechanism.json) | 共同起点、原教师缓存和K/H固定工作点的事件重叠 |
| [teacher_overlap_crosscheck.json](teacher_overlap_crosscheck.json) | 另一代理独立复核教师阈值、身份和事件重叠 |

原实验固定提交：`aa6221ad89dc81d5da3ad38a7587215b64bdbb2c`，标签 `lab-unsw-kd-20261105`。外包 `TraceLight-ALL-2026-11-05b.zip` 的SHA-256为 `91a85c124f43d8e1e297a08a5b94bcb6cddae15b61f05f52e1371f27a96c78df`。

本地完整分析目录：`E:\OneDrive - Coventry University\中传\毕业论文\工作\UNSW四臂验收_20261105`。复算脚本、六份逐行事件、各族与模式统计、150组配对成本及图表均保存在该目录。本次分析读取封存工件，新增拟合与模型推理均为0。
