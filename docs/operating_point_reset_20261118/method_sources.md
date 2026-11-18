# 固定操作点比较：方法来源与实现边界

2026-11-18。本轮只比较同一 R30 模型、同一 C 正常身份上的两条固定校准规则。NP 参照是项目此前已经采用的成熟标准方法，不是新的算法贡献，也不是迁移 NP 训练的复现。

## 1. 精确顺位公式及数值

设冻结模型在 C 的正常分数按升序排列为 $s_{(1)},\ldots,s_{(n)}$。两条规则都采用严格判决 $s(x)>t$，并保留相等分数，不随机打破并列。

原经验顺位为 $k_0=\lceil0.99(n+1)\rceil$。NP 参照固定 $\alpha=0.01$、$\delta=0.05$，取

$$
k^*=\min\left\{k\in\{1,\ldots,n\}:\sum_{j=k}^{n}{n\choose j}(1-\alpha)^j\alpha^{n-j}\le\delta\right\},\qquad t=s_{(k^*)}.
$$

等价计算为 `binom.sf(k-1, n, 1-alpha)`，或 `binom.cdf(n-k, n, alpha)`。这是由样本数与事先固定风险参数确定顺位，不是在 ED 上搜索分数阈值。

| $n=5511$ | 顺位 | 二项尾概率 | C 中严格超阈值行数上限 |
|---|---:|---:|---:|
| 原经验规则 | 5457 | 不作为该规则的置信参数 | 54 |
| NP 最小合格顺位 | 5469 | 0.03958062748674295 | 42 |
| NP 前一阶，仅核对最小性 | 5468 | 0.05382556310887155 | 43 |

实际 C 告警数可能因并列更少。NP 阈值不低于原阈值；在同一固定评分函数上，TP 和 FP 都不会增加。这项对照衡量收紧校准带来的实际取舍，不承担重排或新增检测能力的主张。

## 2. 原文、作者代码与项目复用

[Tong、Feng、Li，*Neyman–Pearson classification algorithms and NP receiver operating characteristics*，Science Advances 2018](https://doi.org/10.1126/sciadv.aao1659) 的 [作者全文](https://arxiv.org/pdf/1608.03109) Algorithm 1、Proposition 1 与式 (2) 给出上述二项尾界、最小合格顺位和严格大于判决。该方法属于 2018 年成熟基线；不能因代码或检索日期较新将其写成 2025—2026 年新算法。

[作者项目页](https://yangfengstat.github.io/projects/neyman_pearson/)直接链接 Python `nproc`。本轮只读核对固定提交 `cdff352221d072c62fc1c0635fd77c77435cc1ea` 的 [nproc/npc.py](https://github.com/ZhaoRichard/nproc/blob/cdff352221d072c62fc1c0635fd77c77435cc1ea/nproc/npc.py)：

- 第 308–313 行使用 `binom.cdf(n-r_upper, n, alpha)` 反解同一风险上界；第 254–255 行明确并列的最小、最大秩。
- 第 210–220 行依据风险上界取 cutoff；第 422 行采用严格 `score > cutoff`。

本轮使用论文的精确顺位公式，不调用完整 `nproc` 包。作者包还包含分数方向判断、风险参数网格和 NP-ROC 计算；这些流程不属于本次固定规则比较。没有执行作者训练或宣称完整复现该包。

项目已有实现位于固定提交 `5b15339ecc055042ea3d68c2cb5e4e93a8ecefc0` 的 [src/ids/risk_transfer/audit.py](https://github.com/zhoumeng-creater/tracelight-ids/blob/5b15339ecc055042ea3d68c2cb5e4e93a8ecefc0/src/ids/risk_transfer/audit.py#L32)：`order_rule` 第 48–57 行分别计算原经验顺位与精确 NP 顺位，第 59 行执行严格判决。此处是项目实现，不是作者源码。本轮复用的是这一顺位计算规则；旧函数中的 OR 保留通道不进入 R30 比较。新工作树基于 `9bb2bb3`，没有导入旧 `risk_transfer` 整个模块，也不重跑旧混合 HGB 审计。

## 3. 假设与本轮解释

原文上界针对：评分函数与留出正常校准样本独立，正常样本独立同分布，未来正常来自同一分布。概率作用于校准样本的随机性，控制总体 FPR 超过 $\alpha$ 的概率；它不保证每个有限单元的经验比例都不超过 1%。严格大于使分数并列保守。

本轮 C 已被多轮开发利用，采集内流的独立性未建立，B 到 D 的正常组成也有已观察差异。因此报告开发 TP/FP 和逐身份变化，不把二项尾值当作新的总体风险证书，也不将单模型的 $\delta=0.05$ 扩展成三 seed 或 15 单元的联合保证。

F/V 已承担原训练和选轮。本轮无需增加 F/V 阈值下限、风险分组或参数选择；新增这些策略会改变比较对象。ED 可行区间仅用于事后诊断，不能进入两条校准公式。全部规则及常数见 [固定规格](../../configs/operating-point-reset-20261118.json)。

## 4. 与近期迁移 NP 方法的区别

- [TLNP，AISTATS 2025](https://proceedings.mlr.press/v258/kalan25a.html) 的作者仓库固定提交为 [`f048fc973c01adffc8b3cacff8481957d644abe2`](https://github.com/eneugut/transfer-learning-neyman-pearson/tree/f048fc973c01adffc8b3cacff8481957d644abe2)。已存并核对的 `tlnp_lib/training_utils.py` 第 109–119 行拆分目标正常、目标攻击和来源攻击；README 第 15 行另列普通训练后调整 cutoff 的 Naive NP。它们不是本轮二项尾顺位参照的来源，项目没有执行其权重搜索或训练流程。
- [两类分布同时迁移的 NP，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33870b3e099880cd8e705cd07173ac27-Abstract-Conference.html) 是另一篇论文。[官方补充](https://proceedings.iclr.cc/paper_files/paper/2026/file/33870b3e099880cd8e705cd07173ac27-Supplemental-Conference.zip)已保存在 `r31_root_cause_20261118/author_sources/np_both_shift_2026`；其 ZIP SHA-256 为 `16d56a85ea354e088b27c10404763b6ea648b78f004b8caebee082445fd480d7`。`Nasa.source.txt` 第 435–445 行的训练入口使用目标正常、目标攻击及来源数据，示例 `alpha=0.10`；第 485–488 行在目标测试数据上比较。当前固定 C 分数比较没有这些目标域训练材料，不继承该迁移方法的结论。既有逐行核查见 [迁移 NP 作者实现核查](../r31_root_cause_20261118/transfer_np_author_review.md)。
