# 近期方法与本项目的差异

2026-11-20。此次查阅范围包括低 FPR 排序、可微风险校准、正常条件建模及阶段弱监督。最终直接近邻选 CRT；其他路线只用于判定机制边界，没有扩展训练清单。以下“读代码”均为静态读取，没有运行作者代码或复现成绩。

## 直接近邻：Conformal Risk Training，NeurIPS 2025 主会

[正式原文](https://proceedings.neurips.cc/paper_files/paper/2025/file/6559542f75b4452ebaaf82094c7defb7-Paper-Conference.pdf) §4、Alg.2 将批次拆为伪校准／预测两部分，梯度穿过风险控制参数。它处理一般 OCE 风险及下游目标，实验包括息肉分割 FNR 控制与储能风险。随机批次校准、反传顺位或校准参数都不是本项目创新。

作者提交固定为 `b9b2276e3e1ccf26ab74f2c750e51e091aa409bd`：

- [run_polyp.py](https://github.com/chrisyeh96/conformal-risk-training/blob/b9b2276e3e1ccf26ab74f2c750e51e091aa409bd/run_polyp.py#L342) 第 342—368 行以前半批正像素确定校准边界，实际平均边界附近约 0.5% 像素，再对后半批计算可微 FPR 损失。不能把它写成精确单顺位实现。
- [run_storage.py](https://github.com/chrisyeh96/conformal-risk-training/blob/b9b2276e3e1ccf26ab74f2c750e51e091aa409bd/run_storage.py#L457) 第 457—489 行以批次前部计算 λ，再用其缩放后部决策并反传。
- [storage/data.py](https://github.com/chrisyeh96/conformal-risk-training/blob/b9b2276e3e1ccf26ab74f2c750e51e091aa409bd/storage/data.py#L70) 第 70—98 行随机拆训练／校准；第 122—128 行普通 mini-batch 训练打乱，全批模式例外。代码也支持保留最后 20% 为时间测试，因此不能声称作者没有时间测试；这不同于以真实前后组成构造训练内 Q→E。

本研究的差异是按真实时期组成定义 Q/E，以后期整数预算边界和前期校准边界的差训练，并兼顾攻击相对同一边界的分离。没有将作者的可交换风险保证迁移到本项目。三臂中的随机对应版本是任务适配控制，未经作者原任务完整复现，不能标作“CRT 复现成绩”。

## 排序与现成表格模型

TabM（ICLR 2025）与数值编码已有项目真实结果，继续作共同骨干。[既有作者核查](../r31_design_review_20261117/author_methods.md)固定了 TabM、STACO（TMLR 2025）与 tapAUC（2025）的实现及差异。STACO 的两向 pAUC 是已有排序方案；tapAUC 的零漏报优先目标不能直接支持本项目 1% 误报要求。R30 训练正负已经分离这一事实，要求新的损失说明对正常内部排序的作用；不等于已经证明从头训练 pAUC 无效。

## 正常条件与检索模型

[Deep Context-Conditioned Anomaly Detection for Tabular Data](https://arxiv.org/html/2509.09030v1) 是 2025 预印本，使用条件 Wasserstein 自编码器并选择上下文特征。本轮读取公开方法，未取得作者完整实现；“正常上下文不同”与本项目相关，但条件重建不直接保证各后续正常单元的告警预算。

[Retrieval-Based Deep Hybrid Models for Robust Network Intrusion Detection](https://onlinelibrary.wiley.com/doi/full/10.1002/spy2.70263) 于 2026-10-25 发表。RobustTabR 以检索邻居距离构成密度特征，给既有预测附加可信度。本文的 OOD 接受／复核问题、特征与评价不同；不能把 FPR@95 当作本项目正常单元 1% 告警率。本轮未核得可供锁定的作者实现，不将其直接列为实验臂。

## 阶段弱监督路线

[ProDisc-VAD](https://arxiv.org/html/2505.02179v3) 为 2025 预印本，作者代码固定于 `cd885300d7ec02773f2d686ecdb34f97354a3f14`。其正常原型注意力、极端实例选择及对比学习已有直接机制；[模型代码](https://github.com/modadundun/ProDisc-VAD/blob/cd885300d7ec02773f2d686ecdb34f97354a3f14/ProDisc_VAD.py#L51) 的包内相似矩阵与论文跨批正配对存在差别，不能把实例伪标签当成可靠恶意阶段真值。

[LogMILP](https://arxiv.org/html/2605.10988v1) 为 2026 预印本，要求删去高注意力实例后正包概率下降。作者提交 `ddaeec0f5643830e88531a70a6fb61804e86480f` 只有 [README 与 LICENSE](https://github.com/YUK1207/LogMILP/tree/ddaeec0f5643830e88531a70a6fb61804e86480f)，不具备完整复现代码。

这些来源说明原型、MIL 和删除响应已有先例。项目目前没有同阶段输入下简单集合／序列分类器的真实失败，故不从这批论文再推出一个待试的新骨干。

## 当前创新判断

待检验贡献是针对真实正常组成迁移的训练结构及其实际效果，不是新分位数、新 hinge 或新校准保证。三臂比较必须证明真实时期对应超过同目标随机对应；否则只能保留校准感知训练的适配结果。真实依据足以支持这项有限比较，不足以在运行前称其为已成立的新算法。
