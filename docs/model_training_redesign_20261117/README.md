# 模型与训练重审依据

2026-11-17。[主设计](../MODEL_TRAINING_REDESIGN_20261117.md)规定最终合同；本目录三份审查保留各自证据与推导。

| 文件 | 内容 |
|---|---|
| [architecture_author_audit.md](architecture_author_audit.md) | 对照R29实际源码快照、TabM/PLE固定作者实现、R26结构，核对初始化、输入、参数和消融归因 |
| [training_author_audit.md](training_author_audit.md) | 原history的选轮轨迹、P重训更新量、损失定义、作者示例与论文差异、四组采样及停止合同 |
| [supervision_identifiability.md](supervision_identifiability.md) | 协议×标签支持、A/B/D语义、去协议后完整输入冲突、V挑战面板与可归因范围 |
| [supervision_identifiability.json](supervision_identifiability.json) | 已有输入和标签的描述统计；不是新模型输出 |
| [static_review.json](static_review.json) | 文档/规格哈希、21项静态一致性核对及独立复审修正；不是模型运行验收 |

另外核对了TabReD作者固定提交`b5ef15b3749f30da7a1eb8fba21a5b54d706bf32`的[数据角色说明](https://github.com/yandex-research/tabred/blob/b5ef15b3749f30da7a1eb8fba21a5b54d706bf32/readme.md)及[神经基线训练脚本](https://github.com/yandex-research/tabred/blob/b5ef15b3749f30da7a1eb8fba21a5b54d706bf32/paper/bin/nn_baselines.py#L484-L501)：验证分数严格改善时保存，结束载回同一checkpoint。它支持选中工件直接交付这一实现选择，不提供本项目协议遮蔽、四组采样或D效果结论。

独立复审已修正：不能把已确认流程缺口直接写成唯一性能原因；A+B合并支持不等于每个家族都有TCP/UDP正例；PLE为字段独立、成员共享；D合并排序增益与TCP内部排序、检出严格增加分别判断；F拟合和V迁移不得混作任务梯度冲突；采用条件明确要求实现、机制和效果三者通过。

本次原始工作材料集中于项目`工作/模型训练重审_20261117`；静态归档集中于`实验归档/20261117_第二十九轮模型训练/复核/模型训练重审_20261117`。旧回传工件和原训练代码保持原值。
