# 直接低误报目标：有限训练池 CVaR 强对照

2026-11-04。状态：一手方法／固定作者源码已静态核查，独立实现及合成验证；没有在本地对真实数据拟合。它是已有直接目标的适配，不是候选 KD 的创新，也不是作者算法复现。

## 为什么必须有这一对照

旧轮次把大量工作放在升级约束和蒸馏变体上，尚未回答“直接优化低 FPR 目标的小学生是否已经足够”。如果新 KD 只胜过普通 BCE、却不能超过同样学生和监督预算的直接 pAUC 目标，就不能把结果归因为教师知识解决了低误报瓶颈。此对照完全不查询教师，不使用 R/C/V 标签选训练对。

## 原文和作者代码

- 最近原文：[Closing the Approximation Gap of Partial AUC Optimization: A Tale of Two Formulations](https://arxiv.org/html/2512.01213v1)，2025-12-01 作者预印本；正文标注 2025-11-23 接收。已读 §III–IV、§V 的问题边界和公式，本文不借用其 instance-wise 算法或收敛／总体泛化保证。该工作进一步说明直接 pAUC 是必要的已有方向，不能把“加入尾部排序”独立称为新颖性。
- 基础原文：[When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee](https://proceedings.mlr.press/v162/zhu22g.html)，ICML 2022。这里明确保留旧基础年份；不是新推荐的 2025/2026 方法。
- 作者实现：[Optimization-AI/LibAUC，固定 commit 的 auc.py](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/losses/auc.py#L344)，`pAUC_CVaR_Loss`，commit `3c02aa9877e0091114a00f94a8c6367698eb5637`。本次实时读取行 344–444：原接口追踪逐正例 `u_pos`，以 `eta` 更新阈值状态，默认 squared hinge，`auto=True` 的负类分母来自当前输入。没有运行该作者实现。现代码没有照抄源码，也没有把自己的优化器标成 SOPA。

## 本项目明确实现什么

设训练池正、负类索引为 $P,N$，$M=|N|$，上尾质量 $b=\beta M$。$\beta=0.01$ 与研究的低误报范围相同，是训练目标范围，不是部署 FPR 证书。单对损失为

$$
\ell_{ij}(s)=T\log\left(1+\exp\left(\frac{m+s_j-s_i}{T}\right)\right),\quad i\in P,j\in N.
$$

直接对照最小化已有的有限池 CVaR surrogate：

$$
L_\beta(s)=\frac1{|P|}\sum_{i\in P}\min_{u_i}\left[u_i+\frac1b\sum_{j\in N}(\ell_{ij}(s)-u_i)_+\right].
$$

由于 $\ell_{ij}$ 对负类分数严格单调，最坏的 $b$ 个负类质量对所有正类相同，可以在每个 macrostep 对**完整训练池**的当前负类分数求尾部权重。高于边界分数的每行取质量 1，余下质量在所有边界并列行上平分，再除以 $b$，得到概率 $w_j$。这保留了非整数 $b$ 的 fractional boundary，没有把 $\lceil b\rceil$ 行都等权纳入。于是

$$
L_\beta(s)=\frac1{|P|}\sum_{i\in P}\sum_{j\in N} w_j(s)\ell_{ij}(s).
$$

对当前参数冻结尾部选择／权重并反传，是该非光滑有限池目标的合法次梯度；边界并列选择对称次梯度。它不是对 mini-batch 负类求 top 1%，也不是人口分布 CVaR 已知。目标的精确性针对 **logistic CVaR surrogate**，不表示与离散经验 pAUC 数值相等。

完整枚举代价为 $|P|\times|\operatorname{supp}(w)|$。CIC 规模下默认不用枚举：每个 macrostep 从正类均匀抽样、从全局负类尾部按 $w_j$ 抽样，均有放回，固定 65,536 对。所有对的均值是同一精确有限池目标在冻结当前参数下的无偏估计；所有 chunk 反传完毕才进行一次优化器更新。若中途改模型参数、使用旧步尾部或按 chunk 均值再求均值，就改变了这个解释。该抽样器不含持久优化变量；调用方封存 NumPy `Generator.bit_generator.state` 或等价确定性步种子。

## 接口及对照一致性

实现位置 `src/ids/direct_pauc.py`：

| API | 返回／用途 |
|---|---|
| `global_negative_tail(scores, labels, beta=...)` | `weights` 为全池长度的 float64 概率（正类权重为 0），以及 cutoff、质量、并列计数等统计 |
| `sample_global_tail_pairs(..., rng=Generator, n_pairs=65536)` | 原始池的 `positive_indices`、`negative_indices` 和抽样统计；不隐式设随机种子 |
| `pairwise_pauc_rows(pos_logits, neg_logits, ...)` | 已对齐抽样对的逐对损失；调用方每 chunk 求和再除全对数 |
| `direct_pauc_loss(scores, labels, ...)` | 小池全枚举值及梯度检查；临时矩阵分块不消除保留 autograd 图的总开销 |

所有接口限制 `partition='fit'`。这只表达角色约束，真正的数据角色绑定仍由 runner 的训练身份清单承担。默认 $m=0,T=1$；其他设置必须配置冻结并向 hard／KD 提供相同调参预算。加入 BCE 时用显式系数记录，不能把强制添加的约束或额外标签隐藏在“pAUC baseline”名字里。

## 可证伪检查和目前证据

`tests/test_direct_pauc.py` 核查了与独立 CVaR 阈值最小化的数值等价、fractional mass、边界并列／行重排等变、全局与 batch-local 的反例、全枚举梯度、逐对分块全分母、极端分数的稳定性，以及 RNG 恢复后配对完全一致。固定随机种子的抽样梯度与全枚举梯度比较检验了估计对象。当前合成检查 36 项通过；它们不证明本项目实测优于 BCE，更不替代实验室对照。

如果本对照达到或超过候选 KD，就撤销“教师知识带来增益”的主张，报告直接目标已经足够。若 KD 只在增加训练对数、标签池或调参次数后获胜，须先完成等预算重跑再解释。历史 test 不进入这些选择；现有开发 V 结果只能支持开发阶段结论。
