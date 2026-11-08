# Scanning 下一步：上下文、标签粒度与观测信息

核查日期：2026-11-08。本文为设计证据与候选机制审查。只读论文、作者固定源码和已有工件；未下载权重、拟合数据或运行真实模型。来源登记见 [tfm_context_sources.json](tfm_context_sources.json)。

## 判断

优先保留两个可区分的解释：**当前二分类任务把 Scanning 与其他攻击混为一类，改变了 TabICLv2 的表示与排序；当前单流输入没有表达扫描行为所需的通信历史。** 前者可用同一 P 的标签干预直接检验；后者必须增加查询端真实可用的历史观测，并与匹配的无历史对照比较。支持重选不能替代这两个检验，也不应自动成为下一轮主线。

执行顺序应先做标签粒度探针。它保持观察信息不变，且标签写入模型的两个具体位置已经核实。若语义三类未产生超过伪三类的区分增益，再转入历史观测分支；不同时启动标签、检索与历史输入矩阵。

P 的 8192 行按正常／攻击各 4096 配额抽样；Scanning 512 是抽样结果，并非族配额。15 条失检查询有 P 中完整 48 字段相同的扫描行，排除了这些行“从未得到相同输入的正例支持”这一解释，但没有排除上下文中的标签粒度、重复权重或交互。第十五轮 B1/B2 都改变了目标分数却没有恢复旧 54 条，因此单纯源端口改写的收益解释已明显收缩。[项目工件与独立尾部审计](tail_audit.json)

“同 48 维模型再次失败”不能证明 48 维已经达到信息论上限。单流信息不足需要由缺失观测的有控增益支持；模型先验或任务定义失配仍是竞争解释。去源端口后的 47 维正负重合也不能直接当作完整 48 维的不可分证据。

## 2025—2026 直接近邻

| 成果与年份 | 原文机制及固定代码 | 本项目的取舍 |
|---|---|---|
| [TabICLv2，2026](https://arxiv.org/html/2602.11139v2) | 标签先进入列级嵌入，后进入 ICL；分类器是固定 MLP 输出头。作者 commit `d83091b845a8c5f7c087bd54964f755a9e560f45` 的 [ColEmbedding._compute_embeddings_with_cache](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/embedding.py#L736) 与 [ICLearning._icl_predictions_with_cache](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/learning.py#L675) 两处均将 P 标签写入缓存。 | **采用为标签粒度探针的机制依据。** 同一输入的攻击族标签可以改变早期表示；改标签后必须重建列级与 ICL 完整 KV。三类低于原生 10 类容量，无须多类层级扩展或新骨干。 |
| [LimiX，2025-09-03 首发](https://arxiv.org/abs/2509.03505)，[原文 §5](https://arxiv.org/html/2509.03505v2#S5) | 第一遍前向以特征注意力加权样本注意力，选择上下文；第二遍在选中行上预测。固定 commit `b4d06eadca6f0cd58ae259034b02c86f4a905950` 的 [InferenceAttentionMap.inference](https://github.com/limix-ldm-ai/LimiX/blob/b4d06eadca6f0cd58ae259034b02c86f4a905950/inference/v1_0/inference_method.py#L309) 与 [InferenceResultWithRetrieval.inference](https://github.com/limix-ldm-ai/LimiX/blob/b4d06eadca6f0cd58ae259034b02c86f4a905950/inference/v1_0/inference_method.py#L95) 对应两遍过程。 | **保留机制近邻，暂缓采用。** 证明有可靠近期查询条件检索路径，不证明当前 8192 行上下文需要检索。会引入首次评分、选择、重建上下文和额外推理；换 LimiX 同时换了模型和预训练，无法隔离 Scanning 根因。 |
| [Test-Time Compute for TFM，2026-10-08 预印本](https://arxiv.org/html/2610.12005v1) | 冻结 TabPFN-3，以末层 ICL 注意力取行；查询按检索集合分组并使用行集合并集。有同上下文大小随机对照。COMET_MC 检索将 log loss 从 0.0151 升至 0.2302；稀有类池占比 1.7%，检索上下文中位占比 40.3%；固定行数修正类比例仍未修复。固定 commit `45fb141b2acf15518e70885aa149c8e1898cd15c` 的 [capture_attention_topk](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/45fb141b2acf15518e70885aa149c8e1898cd15c/src/ttc/retrieval/host_probe.py#L155)、[_score_attn / _eval_modelaware](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models/blob/45fb141b2acf15518e70885aa149c8e1898cd15c/src/ttc/benchmarks/row_modelaware.py#L78) 可核。 | **采用其控制思路与失败证据，暂缓算法移植。** 上下文选择可能同时改变支持相关性与标签先验；不能把“找到更多稀有类”当作效果保证。作者研究用多组重建，且访问整批无标签查询；这不是当前单一可复用 KV 的同成本改动。 |
| [DistPFN，ICML 2026](https://proceedings.mlr.press/v306/lee26aw.html) | 用测试平均预测与 P 类先验的比值，对各类预测乘全局权重再归一化。固定 commit `6b375c8a7624e614e84072c419c4d127229c77b0` 的 [eval_TABPFN_shift_O.py](https://github.com/seunghan96/DistPFN/blob/6b375c8a7624e614e84072c419c4d127229c77b0/src/evals/eval_TABPFN_shift_O.py#L79) 可见公式。 | **排除为当前尾部排序修复。** 相同正权重作用于 C/E 时，二分类变换严格保序；重登记分位阈值后判决不变。若 C/E 分别估计权重，则告警又取决于查询批组成。P 二分类本身均衡，也不能把扫描族的失败直接称为二分类多数类先验偏置。 |

上述项目已核到代码，未复现论文效果。LimiX 代码固定的是 2026-11-06 仓库快照中的 `v1_0` 路径，方法来源仍是 2025 报告，未将 LimiX-2 的后续变化混写为 2025 方法。TTC 的检索结果来自 TabPFN-3，不能直接移作 TabICLv2 的实证结果。

DistPFN 的固定评估脚本还有两项复现问题：`P_test_avg` 是一维，但 `cross_entropy` 对 `axis=1` 求和；后文又用测试 ROC-AUC 在四个 ensemble 数中取最大值。这里只采用可核的重权公式，并据此作代数判断，不引用该脚本的提升量作为本项目收益依据。

## 固定 48 维可以检验什么

最小的标签粒度诊断保留同一 P/C/E 身份、同一 48 维输入、原预处理和冻结骨干。它不增加支持行，不重选支持，不改变正常 C，也不叠加源端口干预。

| 诊断臂 | 唯一变化 | 作用 |
|---|---|---|
| T0 | P 标签为 Normal / Attack | 重现已有二分类基准。 |
| TF | P 标签为 Normal / Scanning / OtherAttack，计数 4096 / 512 / 3584 | 检验已存档族信息经标签通道进入模型，是否改变攻击—正常尾部排序。 |
| TP | 正常类不变；按 P 最终 FP32 完整输入分组，整组以预登记哈希划为 AttackA / AttackB，计数 512 / 3584 | 与 TF 同类数、同类计数；区别是类别是否对应扫描族结构，防止把增加输出类或任意拆分的作用误称为族语义收益。相同攻击输入必须同伪类，避免新增矛盾标签。 |

三臂是**设计候选，尚未实现或冻结为实验室入口**。若只做 T0/TF 两臂，可以检验标签重新定义是否影响表现，但不能将收益进一步归因于真实族语义；TP 是后一个主张所需的对照。不要加上更多随机种子、更多族划分或检索版本来挑选赢家。

必须保持的定义：

- 三臂统一输出攻击联合概率的 log-odds：$s=\log p(\mathrm{Attack})-\log p(\mathrm{Normal})$；TF、TP 的 $p(\mathrm{Attack})=p_1+p_2$。不得改用 argmax、扫描单类分数、取两种分数最大值或在 E 上选择合并规则。多类 logits 应按固定集成约定还原概率再合并，或用等价 log-sum-exp；不能把两个 logits 相加称为攻击概率。概率截断／数值稳定实现应在合成核查中固定。
- 正常 C 只用于每臂既定 1% 阈值登记。比较整体 TP/FP、扫描 TPR、旧 54 与 15 条匹配子组及对应正常子组，同时保留成对新增／修复错误。检测目标仍是全攻击，不以扫描增益抵销其他族损失。
- TF 和 TP 使用同一三类编号与单一 seed，特征置换与其他模型设置固定；两层 KV 独立重建。类别映射写入工件，不能只更改输出标签名。C/E 的族字段只供事后分组评价，不进入预处理、查询模型输入或阈值计算；没有新优化器或任务内训练。
- 三臂使用同 8192 行标签对象，但 TF 使用更细的监督信息。其族标签已在存档中存在；这不等于在只能取得二分类标签的部署情境下零额外标注成本。
- TP 只在 P 攻击行中，按最终 FP32 的完整 48 维位模式分组。组整体按固定哈希排序，组大小不超过剩余 512 配额才纳入 AttackA；否则跳过，配额满后其余进入 AttackB。若最终不足 512，协议硬失败，不换种子或按效果补选。只要攻击单例组不少于 512，这一遍贪心必能填满：未填配额可由余下单例逐行补齐，无须求解器或多次搜索。
- TP 的哈希种子、组摘要、并列处理先固定，不按 E 选择；登记真实族与伪类交叉表，不强制 Scanning 在两个伪类中的比例。分组后抽样并非逐行均匀抽样，不把 Scanning 落入小类数量写成理论必然的 64。原有正常／攻击同输入冲突保持；TF 是否新增 Scanning／OtherAttack 同输入冲突也须审计。

原生标签编码容量为 10，官方 [`TabICL.__init__` 的 `max_classes=10`](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/tabicl.py#L155) 与论文 A.4 一致；三类无需换模型或启用层级扩展。固定 N、特征数、ensemble 和模型维度后，两个 KV 的张量形状应保持，输出概率列从 2 变 3，类别映射等元数据也改变；应在合成核查中检查实际形状与元素数，不能把这一预期写成实测包体字节相等。工程入口中的二分类限制仍须扩展，现有完整 KV 适配器不能只改配置名就作为三类入口。

固定作者源码的具体位置为：[`layers.py:55–89`](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/layers.py#L55) 的 `OneHotAndLinear` 按配置容量 one-hot 后线性编码；[`embedding.py:170–174`](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/embedding.py#L170) 创建列标签编码器，[780–789 行](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/embedding.py#L780) 将标签写入列缓存；[`learning.py:110–115`](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/learning.py#L110) 创建 ICL 标签编码器和固定输出头，[711–732 行](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/learning.py#L711) 将标签写入 ICL 缓存并解码，[798–808 行](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/learning.py#L798) 仅在类数超过 `max_classes` 时拒绝该缓存路径。

可证伪结果应写清：TF 若不能相对 T0 改善低误报排序，或收益不超过 TP，则降低“扫描族被二分类标签合并是主要可修复瓶颈”的优先级。TF 若对扫描及其易混正常行产生有方向的分离，并超过 TP，说明现有单流输入仍有可由任务表达释放的信号。两种情况都不等于已经证明新算法贡献。

TP 是一份固定的无族语义划分，不代表所有可能的有意义子类划分。TF 优于 T0/TP 支持当前族标签提供有效结构，不证明它是唯一根因，也不证明 Scanning 是所有类别分解中最优的一种。

## 为什么不先重选支持

当前 P 全部 8192 行已经参与推理。在同一 P 中选择仍为 8192 行的无重复子集，不会改变上下文；缩小子集改变 N，重复抽行改变权重，从更大池换行则改变可访问的标注预算。三者必须分开，不能以“同 N”掩盖候选池与标签成本。

若以后证据支持重选，需要单列一个冻结池，并区分：可访问标签总数、实际入上下文的行数、正常／攻击及族计数、唯一模式数、重复质量、检索评分成本、缓存数量。选择臂应与同 N、同类计数、同池的随机替换比较；若同时增加 Scanning 配额，还需说明测试的是族权重变化。当前没有证据要求启动这套实验。

查询条件检索只从已观察的 48 维中改变支持，不能恢复未输入的通信序列。精确相同的查询输入在相同上下文与推理规则下仍不能因未知的主机历史而得到不同判断。LimiX 或 TTC 的注意力相关性也不等于网络行为历史。

## 与跨流方向的关系及创新边界

补充历史观测更直接对应扫描行为，但它是**改变观测单位**，应另有因果可用性约束：只使用判决前已到达且可用的流，不用攻击标签，不把未来流或当前流结束后才确定的信息混入早期告警。应固定 P 标签定义和模型，先比较同一查询的“无历史／有历史”；原始 48 维继续保留，新增字段数、状态内存、更新时间和首次可报警时刻单列。不得为保持名义 48 维随意删除旧字段，否则无法区分新增信息与被删信息的影响。

已有工件可以分析历史变量是否改变重合结构，但只有同模型、同数据身份的实验室对照才能证明模型用到了新信息。历史打乱对照应保留字段边缘分布并破坏正确时间／主机关联，用来检验收益是否来自关联信息；具体实现和是否需要该臂由跨流协议冻结，不能拿它补做 E 选参。

标签拆分、增加若干计数／扇出字段、直接套用注意力检索，都是机制探针或任务适配。潜在研究问题是：在低误报与固定状态预算下，如何保留**能改变正常—攻击边界的任务条件证据**，并证明收益不只是新增监督、历史信息或额外计算带来的。当前先定位哪类证据确实有用，再决定是否形成新的压缩或上下文组织方法；QCOC、CRT 和“多加模块”都不能跳过这一步。

## 暂缓与排除记录

- **年份排除**：[LoCalPFN 首发 2024-06-07](https://arxiv.org/abs/2406.05207)、[TuneTables 首发 2024-02-17](https://arxiv.org/abs/2402.11137)、[TabDPT 首发 2024-10-23](https://arxiv.org/abs/2410.18164)。不因 2025 正式会议或 2026 修订重新列为新候选。
- **证据排除**：`arxiv:2607.26628` 延续上一轮排除，作者早期草稿虚构测量的固定 README 记录见[上一轮证据](../scanning_mechanism/tfm_evidence.md)。没有使用其效应值。
- **机制不一致**：[Two Is Enough，2026-10-20](https://arxiv.org/html/2610.27569v1) 将 TabICLv2 描述为无固定输出头的类原型比较器；与官方原文 A.4 和已核固定 MLP decoder 不一致。该文同时说明单变量比较仍在进行，本次未找到可核作者实现。不采用其机制来解释本项目，更不据此重训。
- **代码未核实**：TAAR 的 [OpenReview 原稿](https://openreview.net/pdf/55d78dfa9704c8813df8505e36609ed52f7b0590.pdf) 描述任务注意力检索，本次未取得可归属该稿的独立固定作者代码。不以它代替已核 LimiX/TTC，也不采信搜索摘要中的效果数字。
