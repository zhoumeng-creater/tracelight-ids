# R31 根因研究材料

[主报告](../R31_ROOT_CAUSE_AND_RESOLUTION_20261118.md)给出根因证据、明确取舍、选定问题及训练前仍需完成的两项设计工作。不是实验室运行入口。

- `analytic_basis.md`、`analytic_counterexample.py/json`：原单侧目标不足以控制组成变化的解析反例；不读取真实数据。
- `cache_guard_audit/`：从6份history的96次验证记录重建保护拒绝，核对39个正例与原工作点、来源/协议的关系，附逐身份CSV及复算脚本。
- `design_challenge.md`、`design_challenge_sources.json`：分组假设、风险族覆盖与作者依据。
- `literature_low_fpr.md/json`：Cooper、STACO、tapAUC原文与固定代码对照，以及直接套用的障碍。
- `transfer_np_author_review.md`：2025迁移NP仓库与2026正式论文补充的区别及实际约束实现。
- `author_sources/`、`author_sources_low_fpr/`、`author_npcs/`：仓库提交固定来源、版本与SHA清单；完整只读原件集中归档于项目 `实验归档/20261118_正常子群训练结果/根因研究/材料` 的同名子目录。本机阅读副本不纳入Git。未导入或执行作者训练。

`.source.txt`/`.code.txt` 为 notebook 代码单元的阅读提取，不是本项目运行入口。证据目录禁用Git自动换行转换，保留审计SHA；原作者格式不改。

本轮模型加载、前向、训练、真实数据重校准和阈值搜索均为0。解析有理数核验与缓存算术核对不计为新模型实验。
