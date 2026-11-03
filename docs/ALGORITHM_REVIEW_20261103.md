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

若接入必须锁定作者版本和checkpoint；原论文v1、后续v1.1与当前v2不能混用。该项目前只作方法选择审查，未接入或复现，不因“基础模型”名称而自动纳入全部矩阵。

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
