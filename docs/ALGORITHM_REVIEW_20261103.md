# 算法设计审查：低误报检测、蒸馏与近期方法对照

审查日期：2026-11-03。依据：当前实现、前五轮核验结果、research v2预验证及下列原论文／官方实现。本文区分已实现、已验证和拟优化，不将论文中的成绩或理论保证移作本项目结论。后续真实训练、方法比较和正式资源测量在实验室RTX4090D执行，本机仅文献、代码、单元及合成检查。

## 1. 是否涉及算法设计

涉及，而且“如何在低误报预算和部署成本约束下保留攻击检出能力”应成为论文的算法主线。当前已有四类设计：

| 层次 | 已实现内容 | 证据与边界 |
|---|---|---|
| 特征选择 | 训练内MI、分组重采样频率聚合、Spearman贪心去冗余、固定字段预算 | 第四轮双环境消融已完成；稳定聚合没有普遍检测收益，不是已验证的新选择算法 |
| 蒸馏目标 | weighted-hard、严格matched-hard、普通KD、正确性／置信门控及gated-matched控制 | 数学与工程路径已检验；完整五seed的实验室v2研究尚未完成 |
| 低误报代理 | batch高分正常与低分攻击的有界pairwise soft-margin损失 | 是启发式困难样本排序，不等于精确优化1%FPR或pAUC |
| 教师与学生学习 | 训练内组隔离OOF教师、小MLP／大MLP容量控制、官方TabM适配 | 已有可运行实现；模型选择、充分调优和新确认仍需研究 |

数据隔离、独立校准、哈希与可追溯性属于必要的研究规范；它们增强可信度，但不能单独充当算法创新。当前可以准确称为“已有算法流程与训练目标设计，尚待证明有效且有辨识度的方法贡献”。

当前v2固定矩阵为13条件×5seed，每环境65个主拟合，另有15个OOF和5个全train教师参考。次数是预算账本，不是创新证据。现有本机pilot只使用单seed、有限样本和20epochs，不能代替实验室完整研究，也不能直接与历史全量成绩排名。

## 2. 原论文核查与适配结论

### A. TabM：ICLR 2025主会

**TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling**。[正式全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)，重点§3.3、§3.5及附录D.2／D.9；[固定作者版本](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/README.md)。

论文的设计重点是共享参数的多个MLP成员。训练逐成员计算损失再平均，推理对成员概率取平均，早停应监测集成后的性能。原适配器前两项正确，但用成员损失均值选最佳轮；本次已将其改为集成概率的weighted BCE选择，成员损失仍保留诊断。这是一次有论文依据的具体优化，不是新算法创新。

当前k=8、width=32是固定小预算适配，完整配置也沿用该尺寸。论文搜索包含k=32和更宽网络；20epoch小试中不少模型最佳轮仍在上限。因而不能由65.27%的旧pilot分数认定TabM弱，也不能仅因使用官方包就声称完成作者最强成绩复现。

建议先在实验室给TabM一个单列、可负担的调参预算，验证集成早停及数值嵌入，再决定它作为强模型参照、OOF教师，还是部署候选。训练较大k后裁剪推理成员可作额外压缩对照，不能直接假定优于小MLP。

### B. TabArena：NeurIPS 2025 Datasets and Benchmarks Track

**TabArena: A Living Benchmark for Machine Learning on Tabular Data**。[正式会议记录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)，[作者实现](https://github.com/autogluon/tabarena)。它是正式数据与基准赛道工作，不混称为新模型方法主赛道。

值得借鉴的是受控调参、数据分区、资源和集成口径。其较充分的搜索／交叉验证设定与本项目固定小网络不同；排行榜不能直接证明某方法在本项目的低FPR、组隔离及CPU条件下更好。公平比较应给各模型合理且记录清楚的搜索预算，不能让新模型只跑一个小配置后宣布传统模型更强。

### C. FSNID：KDD 2025主会

**Feature Selection for Network Intrusion Detection**，DOI10.1145/3690624.3709339。[正式作者全文](https://www.machineintelligencelab.ai/papers/kdd25.pdf)，重点§4与附录C；[作者代码](https://github.com/c-s-westphal/FSNID)。

其核心是条件信息贡献与顺序删除，而非独立地按单变量MI取前k。它为当前遗漏交互信息、重复占用字段预算的问题提供了直接对照。应保留方法自然输出的字段数，并补同字段数的普通选择基线。

公开入口与论文设置有需要记录的差异：默认网络／batch／迭代含义不同，入口全矩阵缩放和评价调用也不能直接替代本项目分区。不能由此推断论文成绩无效，但本项目必须做train-only适配、说明筛选原始字段还是编码列，并记录全部计算成本。默认成本较高且未核实明确代码许可，当前仍是已审查、未复现；不把现有MI改名为FSNID。

### D. 教师质量：ICLR 2026主会

**SGD-Based Knowledge Distillation with Bayesian Teachers: Theory and Guidelines**。[官方会议页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0bdbfac7d9df70021d694e5ac795bf9d-Abstract-Conference.html)，[全文](https://proceedings.iclr.cc/paper_files/paper/2026/file/0bdbfac7d9df70021d694e5ac795bf9d-Paper-Conference.pdf)，重点§3–4及附录F；[作者代码](https://github.com/Bayesian-KD/Bayesian-KD)。

该文把教师后验估计误差与学生训练噪声联系起来。适合借鉴的是“教师软目标质量需要单独检验”，而非“教师accuracy更高即可”。其理论有优化和可表示性假设，不能成为本项目AdamW、小MLP及1%FPR的保证。也没有证明本项目T=1、alpha=0.5最优。

下一步先诊断OOF教师在正常尾部和困难攻击上的质量。若确认教师成为瓶颈，再比较有真实多样性的教师集成或末层不确定性估计；不立即引入完整Bayesian训练。

### E. 置信与可学性：ICML 2026正式论文

**Toward Understanding Adversarial Distillation: Why Robust Teachers Fail**。[PMLR正式记录](https://proceedings.mlr.press/v306/lee26g.html)，[作者全文](https://arxiv.org/html/2605.21999v1)，重点§4及表2；[作者实现](https://github.com/HongsinLee/why-robust-teachers-fail)。

论文说明在特定对抗学习模型中，高置信教师可能要求学生拟合不可表达信息。它提示我们检验教师置信与学生容量是否匹配，不能据此直接认定当前Fuzzers失效的原因。该文是图像对抗设置，不给本项目普通二分类提供鲁棒性保证，也不构成增加PGD的理由。

### F. 容量不匹配：AAAI 2026主会

**Rethinking the Dark Knowledge and Kullback-Leibler Divergence Loss in Knowledge Distillation Under Capacity Mismatching**，RCKD。[正式条目](https://ojs.aaai.org/index.php/AAAI/article/view/39878)，[全文](https://ojs.aaai.org/index.php/AAAI/article/download/39878/43839)，重点公式8–10；[作者代码](https://github.com/yingchao-wang/RCKD)。

这是一个应当“参考但不机械移植”的例子。其类间关系向量有C(C−1)/2维，使用余弦比较。当前C=2时只剩一维，余弦基本退化为符号关系，无法保留原多类机制；单logit学生更不能直接替换损失来宣称复现。类似地，[CVPR2022 DKD](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.pdf)的非目标分布在二分类中只有一个类别。论文身份再强，也不能代替任务适配论证。

### G. 低FPR目标：ICML 2022基础与2026扩展

**When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee**。[ICML正式记录及全文](https://proceedings.mlr.press/v162/zhu22g.html)。这是直接相关的基础工作，不冒称近期论文。其CVaR形式与KL平滑形式的估计性质不同，应选择明确版本及优化器，不能只摘一个loss公式。

**Closing the Approximation Gap of Partial AUC Optimization: A Tale of Two Formulations**。[作者全文](https://arxiv.org/html/2512.01213v1)，DOI10.1109/TPAMI.2025.3637063，TPAMI2026；这是期刊工作。它的前身为NeurIPS2022 **Asymptotically Unbiased Instance-wise Regularized Partial AUC Optimization: Theory and Algorithm**（[正式记录](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fc9f83d9925e6885e8f1ae1e17b3c44b-Abstract-Conference.html)）。PAUCI和UPAUCI、会议与期刊版本不能混称，也不能把另一篇Yao等的NeurIPS2022 DC／Moreau论文算作该前身。

这些工作比增加一个无关大模型更直接地服务于本项目：使训练目标显式关注低FPR区间，并说明随机近似及优化代价。它们也不保证分布变化后的测试FPR。

### H. 其他前沿表格模型：TabICL，ICML 2025

**TabICL: A Tabular Foundation Model for In-Context Learning on Large Data**。[正式出版页及全文](https://proceedings.mlr.press/v267/qu25d.html)。它可以作为有条件的离线教师候选，但优先级低于先调好TabM。模型使用带标签的训练上下文，生成OOF目标时必须排除heldout关联组，不能把包含同一记录的上下文预测当折外监督。显存足够也不代表主机RAM与长上下文成本可负担。

若接入必须锁定作者版本和checkpoint；原论文v1、后续v1.1与当前v2不能混用。进一步核查确认v2已是[ICML2026正式论文](https://proceedings.mlr.press/v306/qu26a.html)，不是只停留在早期arXiv身份；其最新版本变化及选择优先级见第6节。该项目前只作方法选择审查，未接入或复现，不因“基础模型”名称而自动纳入全部矩阵。

## 3. 当前算法最需要优化的具体问题

### 3.1 10%困难正常流与1%FPR不是同一目标

当前代码在每batch选择最高分10%正常流、最多32条，并只取最低分的至多64条攻击，再加组均衡权重。改变的不仅是正常尾部范围，也包括正类分布及估计权重。它是可解释的代理，但不能声称精确优化[0,0.01]pAUC。

也不能只把0.1改成0.01就称优化完成：每batch若约300条正常，1%仅约3条，离散尾部估计波动较大。需要明确跨batch分位数／阈值机制或相应的pAUC优化器，并把随机配对作为控制，检验收益是否来自困难样本选择。

一个可研究的单侧CVaR代理为：

\[
\min_{w,\{s_i\}}\frac{1}{n_+}\sum_{i:y_i=1}
\left[s_i+\frac{1}{\beta n_-}\sum_{j:y_j=0}
\big(\ell(f_w(x_i)-f_w(x_j))-s_i\big)_+\right],\quad\beta=0.01.
\]

这是对给定成对代理损失的尾部风险形式，不等同于原始0–1pAUC，也不直接等于单点Recall@FPR=1%。随机优化时的每正样本阈值需要稳定身份与跨batch状态；不能每batch重新top-k后就借用论文收敛结论。

主指标继续保留独立工作点的攻击召回及实际FPR，pAUC仅作为预先定义的区域指标。若研究直接约束FPR下的FNR，可另参考[ICML2021 Implicit rate-constrained optimization](https://proceedings.mlr.press/v139/kumar21b.html)，不要把它与pAUC目标混为同一优化问题。

### 3.2 分组隔离与分组等权是两种决定

按会话／重复联合组隔离是防泄漏要求。按每组选中样本等权训练则改变风险的权重。主要测试FPR按流计数时，训练的等组风险与评价的等流风险未必一致。

建议主目标先匹配按流的操作定义，组用于隔离和不确定性估计；等组加权作为显式消融，不直接认定更正确。攻击家族保护也应单列，而非由当前batch组数暗中决定。

### 3.3 门控的置信代理需要检验

当前g=correctness×1{|2q−1|≥0.5}×|2q−1|，相当于正常只接收q≤0.25、攻击只接收q≥0.75的教师软目标。拒绝后仍有完整hard监督，不能说这些训练样本被删除。

只读既有pilot的OOF记录，正常接受3276/3865，攻击接受7408/8135；364条q≥0.5正常记录全部被拒绝。这只描述监督覆盖，不证明门控有效或无效。拒绝错误教师本身可能合理；需要进一步检查门控是否只在容易样本上学习，以及hard分支是否足以处理正常尾部。

balanced HGB分数也不是原分布真实后验。理想加权风险下，若p=N0/N1、真实后验为eta，则q=p·eta/(p·eta+1−eta)。因此不能将|2q−1|直接解释为校准置信，也不能盲目给教师去偏而忽略学生hard使用的同类权重。若校准教师，须在train内部继续交叉拟合，不能动旧calibration/test。

### 3.4 软知识、类权重与门控须继续分开归因

T=1、alpha=0.5是合法基线。设hard的正类权重为p，把KD中的q替成y后，有效正类权重变为(1−alpha)p+alpha；gated版本为(1−alpha g_i)p+alpha g_i。因此已有matched-hard与gated-matched-hard是必要控制。

普通KD相对matched-hard的旧pilot信号值得检验，但不能由单seed、20epochs认定教师信息已经被证明有效。温度／alpha搜索应有限且单列账本，不把追分过程藏入“新算法”。

## 4. 推荐的优化次序

| 优先级 | 动作 | 保持不变／最小控制 | 当前状态 |
|---|---|---|---|
| P0 | TabM以集成预测选择早停；head loss只用于训练及诊断 | 固定数据、结构、seed和优化器；验证新checkpoint选择 | 本次已修实现，仅合成测试 |
| P1 | 显式beta=0.01的低FPR区域目标，与现有hard-negative代理比较 | hard、matched-hard、KD、随机配对、当前排序及新目标 | 研究设计，未实现／未运行 |
| P2 | OOF教师尾部可靠性与学生可学性分析，再选择教师 | 同一学生、同一base损失、同样本；比较HGB／经调优TabM等 | 既有OOF可诊断，强教师比较待实验室 |
| P3 | 充分但受控的近期模型对照和字段压缩 | 先固定教师与目标，再比较all／缩减字段；保留容量匹配 | TabM adapter已有，正式对照和FSNID仍待完成 |

建议先在实验室完成小规模、控制充分的目标比较，再对过开发闸门的候选执行完整五seed及第二环境复验。现有13条件配置是机制基线矩阵，不能因它已写入YAML就当作最优路线或必须一口气完成所有笛卡尔积。

## 5. 可形成论文贡献的主线与证据

建议主线为：**低误报预算下，面向正常尾部与困难攻击保护的轻量模型学习与蒸馏。** 算法候选可以由监督项、显式低FPR代理及经验证的教师项组成；这些组件已有先例，创新必须落实到清楚的新设计、适配问题和可复核收益，不以新缩写替代。

```mermaid
flowchart LR
  A[已审计的train] --> B[组隔离OOF教师]
  A --> C[固定字段与小学生]
  B --> D[软目标质量与权重控制]
  D --> C
  C --> E[明确的低FPR训练目标]
  E --> F[validation选方案与失败分析]
  F --> G[实验室五seed及CPU取舍]
  G --> H[新来源准入后冻结确认]
```

至少需要分别回答：低FPR收益是否来自目标对齐；蒸馏是否超出重加权；困难家族是否受到保护；优势是否在第二环境和CPU部署中保留。只改善总体平均、选择稳定性、模型字节或某个seed中的一项，都不足以替代其余证据。

第五轮test已经使用。上述新设计仍在已用来源上开发；若要支持新的独立确认结论，必须另有未使用、schema与标签可审计的数据并预先冻结。旧test上的新分数只能是已见数据的探索，不能重标最终成绩。旧运行包、模型和报告保持原样；新源码必须新目录、新版本，实验运行中不得更新。

本次没有启动真实数据训练，也没有将文献结论宣称为本项目性能。算法是否变得更有效，交由实验室受控实验与后续确认决定。

## 6. 广泛参考与快速更新模型：补充研究边界

研究不能只围绕当前模块寻找支持。检索范围扩展为表格基础模型、网络流量预训练、教师质量与压缩、低误报／分布变化和部署策略；广泛比较后，仍应围绕一个可检验的问题选择主线。论文、当前代码和checkpoint需要分别核查，发布日期、软件版本与模型版本不能混为一项。

### 6.1 当前可直接关联CSV任务的基础模型

以下为截至2026-11-03检索到的官方版本信息，不是本项目已运行清单，也不将作者基准成绩直接移入IDS结果。

| 模型 | 已核对的较新身份 | 可能角色 | 需要先解决的条件 |
|---|---|---|---|
| TabICLv2 | ICML2026正式论文；软件2.2.0；分类checkpoint为tabicl-classifier-v2-20260212.ckpt | 优先审查的预训练教师／独立强参照 | 训练上下文、标签感知embedding与缓存全部限制在OOF训练折；先测24GB机器的真实上下文成本 |
| TabPFN3.5 | 2026-09-15发布，当前对应技术报告；checkpoint为tabpfn-v3.5-20260909.safetensors，另有Fast；软件9.2.0 | 独立性能参考；不默认加入蒸馏 | 非商业与输出用于竞争模型训练的限制需逐条核对；不因代码可获取就默认可发布KD路线 |
| TabDPT | NeurIPS2025主会；官方后续模型1.3／软件1.3.2与原论文版本分开 | 真实数据预训练路线的教师／参照 | 锁定模型、代码、上下文和预训练数据来源，区分原论文复现与新版适配 |
| TabM | ICLR2025；当前已有官方0.0.3 adapter | 强神经参照、可调优教师、子模型裁剪或小学生对照 | 修正早停后，单列合理调优预算；保留现tiny配置作为成本点，不拿它代表模型上限 |

直接来源：[TabICLv2正式论文](https://proceedings.mlr.press/v306/qu26a.html)、[软件2.2.0](https://github.com/soda-inria/tabicl/releases/tag/v2.2.0)、[TabICL权重](https://huggingface.co/jingang/TabICL)、[TabPFN官方模型卡](https://huggingface.co/Prior-Labs/tabpfn_3_5)、[官方变更记录](https://github.com/PriorLabs/TabPFN/blob/main/CHANGELOG.md)、[TabPFN3.5报告](https://arxiv.org/abs/2609.17895)、[模型许可证](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE)、[TabDPT正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html)、[TabDPT软件1.3.2](https://github.com/layer6ai-labs/TabDPT-inference/releases/tag/v1.3.2)、[TabDPT权重](https://huggingface.co/Layer6/TabDPT)。

TabDPT软件1.3.2在2026-11-02 23:04UTC上传，即北京时间11月3日07:04；它增加解释接口，不是另一个1.3.2模型。软件与模型分列可避免用新包名冒充更新的checkpoint。TabICLv2官方源码／权重标BSD-3-Clause，TabDPT模型标Apache-2.0；实际执行仍需锁住下载文件身份及相应条款。

TabICLv2新增目标感知表示、可扩展attention和预训练改进，不能把它仅视为v1更换权重。OOF时连上下文生成的表示／缓存也不能使用heldout组标签。其论文中更大规模的显存结果不能直接变成4090D24GB可行性保证；CIC现有train约128万行，需有明确的上下文预算与训练家族支持审计。允许在合法训练折内部选择上下文行，与禁止把关联组拆到fit／heldout两边是不同要求。

TabPFN3.5许可包含对用其输出训练竞争模型的限制。因此本项目不默认把它纳入可公开发布的蒸馏实现，优先研究授权边界更清楚的候选；性能参照、权重使用和下游训练分别记录。这里指出的是具体候选的实际适配条件，不据此停止其他可行研究路线。

### 6.2 网络流量基础模型与通用大语言模型

需要并行审查packet／burst预训练、自监督表示与时序模型，但输入契约决定能否实测。现有42／82字段CSV不包含恢复原始包字节序列所需的信息，不能把它转换成几段文本就称为同一任务复现。缺可靠时间元数据时，也不能按CSV行序伪造时序。

| 工作 | 已核实身份／输入 | 对本项目的价值与限制 |
|---|---|---|
| TrafficFormer | 作者发表页标示IEEE S&P2025；PCAP→flow→burst字节→bigram／BPE，微调还用包长、方向、时间等 | 包级预训练参照，需新增可核对的原始包输入；不是当前统计CSV的直接替代 |
| The Sweet Danger of Sugar／Pcap-Encoder | ACM SIGCOMM2025；原始包十六进制字段与T5表示 | 很重要的严格划分、捷径和简单强基线审查；预训练明确包含UNSW原始包，需核具体重叠 |
| Lens最新v5 | TMLR2026，期刊；每流前34包、语义字段遮盖与知识模板，不沿用旧版3包设置 | 可借鉴有语义的预训练任务；现有CSV不能原样复现，字段组遮盖须与成熟表格自监督方法另作比较 |
| netFound-640M-base | 2023预印本路线、2025公开checkpoint；包／burst／流token | 不是因2025发权重就变成2025顶会论文；输入、时序和数据来源均需另审计 |

直接来源：[TrafficFormer作者发表页](https://liuzhuotao.github.io/publications/trafficformer/)、[官方输入流程](https://github.com/IDP-code/TrafficFormer)、[SIGCOMM正式DOI](https://doi.org/10.1145/3718958.3750498)、[Pcap-Encoder作者代码](https://github.com/SmartData-Polito/Debunk_Traffic_Representation)、[Lens v5全文](https://arxiv.org/html/2402.03646v5)、[Lens作者代码](https://github.com/V-Enzo/Lens-foundation-model-for-network-traffic)、[netFound作者代码](https://github.com/SNL-UCSB/netFound)、[netFound权重卡](https://huggingface.co/snlucsb/netFound-640M-base)。这些候选本轮均未复现或运行。

预训练数据污染需要独立审计：某模型若已用UNSW原始包预训练，即便本项目没有对它微调，也不能直接称其在UNSW上独立泛化。包级模型可以成为背景、另一条有条件的数据路线或独立展示来源，不应在没有一致提取链时与当前CSV模型直接排性能表。

通用LLM应按具体问题评估：它可用于有真值可核查的字段语义整理、规则候选或解释，但这些辅助功能不能自动计为检测算法收益。若用作检测教师，须验证结构化分数、输入字段、提示选择、成本与标签泄漏；不能仅凭通用能力或版本新就把它接入告警核心。

有条件的扩展是“富包输入教师→CSV学生”：若公共PCAP与每条CSV有真实流对应，可研究额外信息能否转化成低成本学生收益。没有这种对应时不开展，不能凭样本位置配对。另一项候选是语义字段组遮盖／去噪，必须防止保留派生字段让模型只学算术重建，并比较同架构从头训练、随机字段遮盖和语义组遮盖；尚未排除成熟表格自监督近邻前，不主张它为新方法。这两项只作为条件研究空间，不同时铺开成默认训练矩阵。

### 6.3 在参考基础上创新的检查顺序

每个新算法候选须先写明：最近邻工作与实现身份、它没有解决的具体问题、本项目改变的数学目标或机制、能推翻该解释的对照，以及训练／部署成本。当前优先问题仍是低FPR正常尾部与困难攻击保护；“换成基础模型教师”“使用OOF”“门控加排序”本身都不足以主张首创。

新模型持续更新时，保留两个口径：固定版本的可复现实验，以及注明日期的前沿候选表。不能在实验过程中自动升级依赖或默认checkpoint；新发布不自动使旧实验无效，但需要解释为什么选用或不选用。后续真实实验仍在实验室执行，本次扩展只做文献与代码核查。

### 6.4 直接近邻与可检验的创新主线

“基础模型教师→高效学生”已经有直接近邻：[Jeong等的Distillation of Tabular Foundation Models into Efficient Predictors](https://arxiv.org/html/2610.01435v1)，2026-10-01预印本，尚未核实主会身份或作者仓库可运行状态；[Tanna等的医疗表格蒸馏](https://arxiv.org/html/2605.18702v1)，2026-05-18，作者标为[ICML2026 SD4H workshop](https://pratinavseth.github.io/publications/distilling-tfms-structured-health-data/)，明确使用OOF。后者不是ICML主会。因此TFM、小学生、OOF及部署时移除教师都不能单独承担新贡献。

推荐主线是**面向低FPR工作点的风险尾部迁移**：固定一个经验证的教师，研究教师在正常流量99%分位附近的判别关系如何被小学生保留。与普通KD相比，区别应落实到工作点相关的信息选择／目标，而非只换模型。它仍是待验证假设，需要继续核查工作点感知蒸馏近邻；本次不宣称首创。

关键消融不能把低FPR损失在所有条件都固定后，又声称已经分解了它的贡献。建议分阶段固定以下比较：

| 比较 | 回答的问题 |
|---|---|
| hard vs hard＋CVaR(beta=0.01) | 低FPR目标本身是否有效 |
| HGB普通KD vs HGB普通KD＋CVaR | 同一教师下目标对齐是否有效 |
| HGB普通KD＋CVaR vs HGB工作点感知KD＋CVaR | 迁移规则的独立作用 |
| HGB普通KD＋CVaR vs TabICLv2普通KD＋CVaR | 教师变强是否真正带来收益 |
| TabICLv2普通KD＋CVaR vs 同教师工作点感知KD＋CVaR | 新规则是否在近期强教师上仍有效 |
| 相应matched-hard／同gate且q=y控制 | 排除类权重、逐行重加权解释 |

每阶段先冻结预算及闸门；先检查教师自身的低FPR能力，再花费成本产生完整OOF。上下文、embedding和缓存均折内生成，必要的教师工作点估计也只来自合法训练范围。特征压缩与容量改变后置，避免同时改变所有因素。目标含低FPR区域时，单点Recall@1%仍需单独报告。

备线是**整体FPR约束下的代价感知级联**，参考[ICML2025 A Unified Approach to Routing and Cascading for LLMs](https://proceedings.mlr.press/v267/dekoninck25a.html)。若现有小MLP与HGB确有互补错误，可先用缓存开发预测估计升级收益，再训练路由；主比较保留学生、全HGB、同覆盖率随机路由和简单置信路由。要统一校准整个级联，统计新增误报、挽救漏报、升级比例及端到端p95。LLM路由论文不给IDS误报保证；不建议把昂贵TFM放到最终CPU级联末端。

特征可用性变化可参考[ICML2025 TabFSBench](https://proceedings.mlr.press/v267/cheng25e.html)作为压力测试轴；没有真实时间元数据时不把随机缺失／噪声叫真实漂移。主线无效且已有模型错误互补时才启动备线，不将广泛参考变为同时实现所有模型。
