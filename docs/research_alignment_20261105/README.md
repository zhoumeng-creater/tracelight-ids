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

已确认的问题、实际改动与后续取舍见[主报告](../PROJECT_MECHANISM_REDESIGN_20261105.md)；下一轮只执行[零拟合最小验证](../MINIMAL_VALIDATION_20261105.md)。本轮不追加候选损失、教师或大训练矩阵。
