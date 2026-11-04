# 重构审查：把教师的同类次序作为一个可推翻的增量

2026-11-04。该页是本轮新研究入口的反方审查与损失实现依据。没有运行真实数据拟合。原v3/v4结果及失败解释保留；本文不会把新的坐标、采样和普通排序损失宣称为已成立的原创理论。

## 1. 先把三种缺口分开

研究主问题是：给定小学生的字段／参数预算，教师能否让其在低误报区域多保留一些检测能力，且超出充分训练hard和成熟直接pAUC目标？

- 输入缺口：同架构学生在全字段与受限字段之间的差值。它只是本模型类的经验观察，不是被删字段的因果重要性或不可恢复信息量证明。
- 容量缺口：在相同字段下，小学生与较大直接学生的差值。较大学生不是受限预算内的公平算法赢家，而是容量探针。
- 传递缺口：同一学生、同一数据标签、同一初始化与优化预算上，KD相对直接目标的差值。教师大于学生产生的体积差不属于这项KD增量。

完整字段HGB和受限字段HGB要保持拟合行、标签、树模型配置与查询角色一致。前者优于后者并不保证小学生可模仿；同信息教师蒸馏同样改善时，不能将全部收益归给额外字段。

## 2. 本轮实际核过的近邻

| 一手成果 | 本次核查与边界 | 对本构造的限制 |
|---|---|---|
| [TMLR2025 Rethinking Knowledge Transfer in Learning Using Privileged Information](https://openreview.net/forum?id=dg1tqNIWg3)；[可访问作者预印本](https://arxiv.org/html/2408.14319)；[作者代码](https://github.com/danilprov/rethinking_lupi/tree/04e72da7537426aa20c2056bff07114658e68824) | 预印本§2–4、正式身份及已有源码审查交叉核对；本次完整终稿仍未逐式取得，不称全文复现。Generalized Distillation结合标签与教师soft监督，分析去掉／替换PI等归因控制。 | 更多信息不自动带来学生收益；必须保留同信息教师、充分hard和置乱信息控制。 |
| [ICML2025 Enhancing Logits Distillation with Plug&Play Kendall’s τ Ranking Loss](https://proceedings.mlr.press/v267/guan25a.html)；[作者全文v2](https://arxiv.org/html/2409.17823v2)；[RankLoss源码](https://github.com/OvernighTea/RankingLoss-KD/blob/50a61305fd068dbbd14c04ad070ea9fb9b76c4d0/mdistiller/distillers/rankloss.py) | 正式记录、§4.1 Eq.(5)–(8)及完整损失代码已静态阅读，固定commit `50a61305fd068dbbd14c04ad070ea9fb9b76c4d0`。其目标是每个样本内的**类别通道**次序。 | Kendall软符号排序不是本项目发明。二分类只剩一个通道pair；它不能证明跨样本同类次序没有先例，也不适合原封不动充当丰富多类dark-knowledge。 |
| [NeurIPS2025 PLD: A Choice-Theoretic List-Wise Knowledge Distillation](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6b2921f24f82dc05cda53bbe50bab90-Abstract-Conference.html) | 正式记录与摘要已读：真实类先置顶，余类按教师排序，构成类别list。未核其完整作者源码，不进入本轮原码复现主张。 | 二分类只剩一个非目标类，复杂类序退化。不能把多类论文中“排序优于KL”的动机直接借给二元CSV。 |
| [AISTATS2025 Quantifying Knowledge Distillation using Partial Information Decomposition](https://proceedings.mlr.press/v258/dissanayake25a.html)；[作者代码](https://github.com/pasandissanayake/kd-rid) | 正式记录、摘要、作者代码入口已核，未审完整算法或运行；不借用RID理论或声称复现其信息量估计。 | 教师中的任务无关信息可能耗费有限学生容量。本文不使用“信息瓶颈”一词伪装已经估计了互信息或证明可迁移性。 |
| [2022基础近邻 Evaluation-oriented KD for Deep Face Recognition](https://arxiv.org/html/2206.02325) §3、Eq.(9)–(12) | 本次原文重新核：针对FPR／TPR差异挑critical relations，以多个模型各自阈值的平滑rank-count匹配。既有本仓adapted EKD结果保留。 | “面向低FPR／关键关系／排序替代绝对拟合”早已有明确构造。本文不能据几个新限定词主张首创。 |

近期推荐聚焦2025/2026；2022条目是必须正面比较的基础直接近邻。上述论文可见、代码静态已读、本项目实验室已运行是不同状态。

## 3. 一个最小候选，不加入投影助手或升级保护

首先由硬标签BCE和成熟全局尾部pAUC目标处理二元分离，作为各组共同的直接任务目标 $L_{task}$。pAUC的系数、尾部比例、优化预算和分母由共同协议固定。KD只增加一个同类跨样本关系项：

$$
L=L_{task}+\lambda L_{same},\qquad
L_{same}=\frac12\sum_{c\in\{0,1\}}\frac1{M_c}
\sum_{(i,j)\in P_c}\operatorname{BCEWithLogits}\!\left(\frac{s_i-s_j}{T},o^T_{ij}\right),
$$

$$
y_i=y_j=c,\ i\ne j,\quad
o^T_{ij}=\begin{cases}1&t_i>t_j\\1/2&t_i=t_j\\0&t_i<t_j.\end{cases}
$$

其中 $s$ 是学生raw logit，$t$ 是同一个完整F教师的native margin；$o^T$ 是**两个样本的次序偏好**，不是攻击概率。每类总权重固定为1/2，绝不把每chunk／batch改为独立均值。某类无合法pair，其项明确为0并报告缺失覆盖，不捏造或把另一类权重翻倍。

局部兴趣集合由该教师独立R正常分数的经验higher分位点定义：

$$
\gamma=5\alpha,\quad q_T=Q^{higher}_{1-\gamma}(t(R_0)),\quad A_c=\{i:y_i=c,t_i\ge q_T\}.
$$

每个pair先从 $A_c$ 均匀抽一个anchor，再从同类完整F池其余行中均匀抽partner，固定数量、有放回。该分布**不是**全体合格无序pair的均匀分布。因为教师高分攻击往往很多，这个兴趣集合并不是“只保留紧邻阈值的攻击”，也不保证其样本都困难。ties可让兴趣质量大于5%，必须报告；它不是1%FPR门禁。

硬偏好与higher次序统计对教师严格递增变换不变，故不会再把Portscan低绝对类别置信度直接当作应压低学生攻击概率的理由。它不能恢复被clip抹除的顺序，所以入口必须读取native分数，不能先clip再声称消除了旧问题。不同OOF教师的margin不可直接混排；本轮选择同一F教师，而不是跨折拼raw score。单一教师的训练内目标可以合法用于监督KD，但其过拟合风险由R／C审查和普通KL控制暴露，不假装OOF更严格就天然更好。

本损失是标准偏好BCE／关系蒸馏在任务上的独立适配。候选差异仅为**同类跨样本信息对象＋固定训练池teacher-R尾部兴趣采样**。尚无足够证据给其通用原创定理或首创名称。

## 4. 为什么可能有增量，以及为何也可能完全无效

标签和直接pAUC规定攻击相对正常尾部的次序，却没有给出同一真实标签内的难度顺序。教师的同类关系如果位于受限学生可表示的输入函数中，可能改变共享特征的学习、改善未见样本的低FPR分离。这是一个待证的机制假设。

但是，同类正确排序不是pAUC的必要条件。把教师认为困难的攻击继续排低可能损害TPR；保持高分正常相对其他正常更高也可能无益，甚至耗费本已紧张的模型容量。偏好损失不提供保真、总体FPR或学生可表示性保证。即使其训练偏好拟合更好，也不能直接写成检测知识转移成功。

本轮不另加投影网络、扩散助手、逐族保护或后置guard去掩盖该假设失败。若该单增量不成立，保留失败结论，回到资源预算与教师资格重新选择，而不是保护候选名称。

## 5. 近期近邻的真实二分类适配

ICML2025作者文件的全局 `c_pair_` 固定为 `combinations(range(1000),2)`，函数内部调用 `.cuda()`，原样不能接受本项目二维类别输出。独立适配函数 `binary_channel_rank_rows` 明确构造 $[-s/2,s/2]$，保持作者默认的无偏row-std加 `1e-7` 标准化、logistic软符号以及 `1/.0125=80` 输出尺度：

$$
L^{bin}_{RK}=-80\tanh(\beta\Delta\hat z_S/2)\tanh(\beta\Delta\hat z_T/2).
$$

对于标准化二维行，$\Delta\hat z\approx\sqrt2\operatorname{sign}(s)$，除零附近外几乎只保留类别margin方向。该适配用于实测“近期类别排序方法在二元任务是否退化”，不能称原1000类作者源码复现；它与本候选的跨样本顺序不同。默认输出尺度必须写在日志里，不能悄悄把80倍删掉后借用作者超参。

## 6. 不可删掉的反证与退出条件

| 对照 | 它能推翻的解释 |
|---|---|
| 同预算充分训练hard、同预算直接全局尾部pAUC | 若已解释收益，KD不是必要增量。额外教师／训练预算另记，不能把训练更久当知识。 |
| 普通native raw KL | 若已足够，不能声称排序对象或局部关系必要。 |
| ICML2025 binary类别RankLoss语义适配 | 检验类别通道关系在二分类的适用边界；不把其退化自动当本候选优越证据。 |
| 同类全池uniform pair KD | 若相同改善，不能把效果归因于低FPR局部化。 |
| 同类内置乱教师分数，再重新构造mask和pair目标 | 保持每类目标分布和tail支持，破坏实例对应。若同样有效，应撤回“有用教师实例知识”归因，检查正则化或抽样解释。 |
| 受限字段教师与完整字段教师 | 若相同改善，不得把全部收益归于特权字段；若完整教师更强而学生无增量，转移仍可能失败。 |
| 全字段小学生、同字段较大学生探针 | 区分输入损失与容量损失；不是把预算外模型当算法公平赢家。 |

报告C选定工作点之后的V实际FPR／TPR、每族支持与检出、配对纠正／新损失、pAUC口径和训练／查询／部署成本。C和旧V都已用于开发，新结果只能形成开发证据；不得回用已打开的旧test声称新独立成功。

停止升格规则：候选未超过直接pAUC，不作KD增益主张；未超过同类置乱，不作实例知识主张；未超过uniform，不作局部化主张；完整与受限教师无差异，不作PI主张。单种子微小改善只决定是否值得扩展，不提供稳定优胜结论。

实现文件 `src/ids/tail_rank_kd.py` 与 `tests/test_tail_rank_kd.py` 已覆盖两个端点解析梯度、教师严格单调不变、分块全局分母、同类置乱、尾部缺覆盖、ties、二类RankLoss退化与作者二维表达核对。本次19项合成检查通过；它们证明语义与数值实现，不证明实证收益或新颖性。
