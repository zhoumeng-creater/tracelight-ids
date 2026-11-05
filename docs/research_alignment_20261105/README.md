# 2026-11-05 成果审查与机制证据索引

本轮并行审查学生表示、教师与低误报目标、评价协议三个方向，共20项不重复的近期成果／项目条目。只把实际读到的内容写入依据；条目不是“已复现方法”的计数。外部项目均未在本轮真实数据上运行。

| 文件 | 范围 | 本轮实际作用 |
|---|---|---|
| [学生结构与数值表示](student_representation_review.md) | 10项；TabM、Learned Knots等核至公式和作者实现，GGPL全文缺口另列 | 发现并修复PLE零残差无梯度配置；把范围行为与容量、编码开销分开 |
| [教师与低误报目标](teacher_objective_review.md) | 7项；TabDistill、Pocket、PLD、STACO等核至实现／代数 | 排除换名式创新、二分类退化和概率／工作点混用；限制下一轮KD范围 |
| [评价协议](evaluation_review.md) | 5项；与第一方向重复TabArena、TabReD两项 | 固定组与来源边界、选择机会和配对成本口径 |
| [教师内容实证复核](teacher_content_review.md) | 只读原F/R/C分数，不新增推理／训练 | 分开UNSW与CIC的目标内容，确认UNSW三seed OOF分数完全重复 |
| [教师内容紧凑数据](teacher_content_summary.json) | 约38 KB，保留全体／逐族／逐折及seed一致性结果 | 便于复核定量结论；原始输入SHA和完整2.69 MB结果见原分析目录 |

原分析及复算脚本位于[教师内容目录](<E:/OneDrive - Coventry University/中传/毕业论文/工作/机制重设计_20261105/teacher_content>)；本仓库的教师内容文稿和紧凑JSON为该次已完成分析的副本。原始模型、数据、NPZ未随文献记录再次提交。

上述20项对应机制重设计与第九轮零拟合验证，已回传的结论见[验收报告](../INPUT_EXTRAPOLATION_RESULTS_20261105.md)，无需重跑。

## UNSW 普通 KD 后续协议

用户要求继续广泛参考并冻结全字段 small 普通 KD 最小方案，新增审查如下。11项与上面的20项存在重叠，不合计成31项独立新成果。

| 文件 | 实际内容 |
|---|---|
| [普通 KD 的11项近期成果审查](unsw_kd_sources.md) | 本轮实时核查，分公式、固定作者实现、浅层检索和不可得工件；解释四臂取舍 |
| [全字段工件审计](unsw_kd_artifacts/artifact_review.md) | 正确的all-small教师headroom、full-F/OOF、目标和重复结构、旧runner差异 |
| [绑定](unsw_kd_artifacts/binding.json)与[目标诊断](unsw_kd_artifacts/target_diagnostics.json) | 教师／起点／F/R/C哈希，固定模式置乱不变量及覆盖 |
| [独立设计复核](unsw_kd_design_review.md) | 4次续训、固定终点、验收及停止规则的反证审查 |
| [合成目标检查](unsw_kd_objective_verification.json) | float64独立logit上的等价梯度和全局分母检查；不代表实际训练入口已验证 |

最新下一步以[UNSW最小协议](../UNSW_KD_MINIMAL_PROTOCOL_20261105.md)及[JSON规格](../../configs/unsw-small-kd-minimal-20261105.json)为准：零新教师、零新warmup、四次续训；专用入口已实现，按[实验室指南](../UNSW_KD_LAB.md)执行。本地完整脚本与冻结目标NPZ保存在[方案工作目录](<E:/OneDrive - Coventry University/中传/毕业论文/工作/UNSW普通KD方案_20261105>)，仓库只收录报告、统计和哈希。
