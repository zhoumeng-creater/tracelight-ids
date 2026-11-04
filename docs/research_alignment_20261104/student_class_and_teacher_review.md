# 学生函数类与教师选择：本轮实施依据

2026-11-04。承接[核心机制重设计决定](../ALGORITHM_REDESIGN_DECISION_20261104.md)。本次重新查阅一手论文、模型卡和固定作者源码，并只读审查 `native_trees.py` 与第一阶段筛查入口。当前证据为文献、实现和合成验证；新的真实拟合与资源测量在实验室执行。

## 1. 本轮改变了什么

撤回同类尾部排序后，不能把研究继续收窄为“固定窄MLP，再换一个KD损失”。本轮先回答：**在相同F监督下，当前差距来自字段、MLP容量、直接优化目标，还是学生函数类选择？**

四组 all/MI24 × small/wide MLP 可以提供字段和容量诊断，但仍只覆盖标量输入、标准化、全连接ReLU这一类模型。增加宽度和改变表格表示不是同一件事。新增CatBoost两档容量，让研究包含一个直接表达字段阈值交互、原生处理类别列的学生家族。它是否更好、更轻，由相同协议的实测决定。

第一阶段保留两位HGB、四组MLP warmup、对应的hard/direct_pAUC续训，再加两档全字段CatBoost。**本阶段不训练新的KD候选，不将筛查结果自动扩展为下一轮矩阵。** 原同类排序代码、固定标签与负结果继续保留。

CatBoost相对MLP改变了类别处理、函数类和优化方式。因此它胜出可以支持更换部署候选，但不能仅凭该差值证明“MLP不可表示”或“类别编码是唯一原因”。

## 2. CatBoost适配的原理与代码核查

本次核实官方稳定版为 **1.2.10**，固定release commit为 `b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`。[官方发布](https://github.com/catboost/catboost/releases/tag/v1.2.10)

官方`CrossEntropy`将数值标签解释为正类概率；`Logloss`则按`target_border`解释数值标签。固定源码 `CalcCrossEntropyDerRangeImpl` 使用连续targets，第一导数为 `target−p`，第二导数为 `−p(1−p)`，之后乘行权重。故普通软标签蒸馏可以直接使用官方交叉熵，不需要自写树损失。[API语义](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit#y)、[固定源码第304–350行](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo_helpers/error_functions.cpp#L304)

本仓实现已具备以下行为：

- hard和soft均使用`CrossEntropy`；原生类别列、数值缺失值交给CatBoost，拟合只接收F，没有eval_set或隐式早停。
- 所有行权重由F内原始hard y计算，soft目标不改变类别平衡规则。
- 输出保留`RawFormulaVal`的float64坐标，避免先转float32概率、裁剪再还原logit；阈值登记由共同调用方处理。
- 保存官方`.cbm`与预处理规则元数据，记录树数、总叶数、解析后的实际参数与部署包字节，读取校验模型hash。

设教师概率为q，当前API的参数为非负`kd_weight=λ`，混合目标为

\[
t=\frac{y+\lambda q}{1+\lambda}.
\]

温度为1、hard和soft共享行权重时，`BCE(y,p)+λ KL(q||p)`对学生分数的梯度等于`(1+λ) BCE(t,p)`的梯度。实现优化归一化后的BCE；它不声称树提升的完整轨迹与未归一化目标相同。若写成`(1−α)y+αq`，则 **α=λ/(1+λ)**，不能把两个系数当同一个数。

CatBoost的类别目标统计也可能随软目标改变。后续soft/hard比较测量的是完整树学生训练中的KD增量，不能只归因于叶值平滑。类平衡教师输出也不自动成为自然部署分布下的校准概率。

两档固定容量与`Plain`、`max_ctr_complexity=1`是当前预算探针，不代表充分调优CatBoost的上限。浅树可以是部署学生，较大树可以是质量参照；标签由实际成本决定，不能只凭模型名称决定谁“轻量”。

## 3. 为什么暂不加入更多结构

**数值嵌入作为后备，复用现有TabM。** 本仓`recent_baselines.py`已有官方`TabM.make`与`PiecewiseLinearEmbeddings`，分箱仅拟合训练集。若树学生不满足部署预算，再比较现成PLE与无embedding设置。要控制并报告总参数和实测成本，不能以相同hidden width声称同容量。

**DeepGBM暂不移植。** KDD2019已经提出树输入选择、叶划分与输出知识迁移。作者固定commit `6926d35bdcc3249befc127973d313e3c061f2bb2` 的 `models/gbdt2nn.py` 按树组建立多个BatchDense子网络，再用固定输出权重合成；embedding MSE与标签损失联合训练。它增加多个子网络和叶嵌入，且原任务强调在线更新。本项目当前没有证据说明需要承担这些成本。[原论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/08/deepgbm_kdd2019__CR_.pdf)、[固定作者源码](https://github.com/motefly/DeepGBM/blob/6926d35bdcc3249befc127973d313e3c061f2bb2/models/gbdt2nn.py)

**TabDistill继续作为直接近邻。** 2026-04预印本已覆盖TFM交互发现→GAM/EBM。作者当前commit仍为 `64214da0edf7eef6e8bf645332471d78b30345e8`；下游实现按交互计数选项，EBM以原始y训练。换成IDS、尾部mask或成本权重首先是任务适配；没有合法遮蔽、预算和标签驱动交互对照，不应建立新的原创主张。[原文](https://arxiv.org/html/2604.13332v1)、[固定实现第154–225行](https://github.com/Clouddelta/tab-distill/blob/64214da0edf7eef6e8bf645332471d78b30345e8/experiments/TabDistill_downstream_comparison/compare_index_performance.py#L154)

## 4. 当前教师与近期原文的实查结论

TabICLv2已正式发表于ICML2026，作者仓库HEAD为 `d83091b845a8c5f7c087bd54964f755a9e560f45`，默认分类checkpoint为`tabicl-classifier-v2-20260212.ckpt`，默认8成员。它是后续唯一优先TFM资格候选。原论文百万行结果使用不足50GB显存，不能据此保证实验室24GB直接处理完整CIC。[正式记录](https://proceedings.mlr.press/v306/qu26a.html)、[固定官方说明](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/README.md)、[模型卡](https://huggingface.co/jingang/TabICL)

TabPFN3.5官方模型卡列有主版、Fast和实验性多类版，主文件为`tabpfn-v3.5-20260909.safetensors`。另有2026-10-28公布的Thinking升级，但首发是API专属、逐步开放，不能写成已取得本地checkpoint。模型身份、新版发布和本项目实测分开记录。[3.5模型卡](https://huggingface.co/Prior-Labs/tabpfn_3_5)、[Thinking官方公告](https://priorlabs.ai/blog-posts/tabpfn-3-5-thinking-upgrade?waitlist=open)

2026-10的TFM蒸馏预印本已比较TabM与树学生，研究full-context和teacher-only监督。这支持扩大可检验学生类，而不是只接受窄MLP。其通用表格结果没有证明本项目1%误报工作点有效；full-context训练内目标也不能充当教师样本外可靠性的证据。[原文§3–4](https://arxiv.org/html/2610.01435v1)

本次没有安装或运行上述TFM、DeepGBM与TabDistill。CatBoost本机验证使用合成数据，不属于IDS效果证据。

## 5. 筛查如何支持下一步决定

F/R/C沿用原冻结身份。C参与阈值与checkpoint选择，C误报满足经验预算本身不是泛化证据。MLP每epoch或macrostep选择checkpoint，固定容量树只评价最终模型；拟合次数不能代替选择机会或调参预算。因此第一阶段的跨家族表用于诊断，不能称公平调优后的性能排名。

先看共同工作点的检出缺口、错误互补、软目标与hard标签的差别及完整包体积。CPU流程成本必须在同机同线程下实测，包含预处理、预测和阈值判定；没有CPU结果时不确定部署赢家。训练与教师查询成本另列。多个seed共用同一数据划分时，只刻画优化随机性。

若进入KD阶段，冻结一个字段集合、一个学生容量和一个教师，至少比较hard、普通soft、同类内固定置乱soft；神经学生继续保留direct pAUC及pAUC+普通soft。所有条件使用同样的F标签、相同初始化／容量、训练预算和C选择额度。先查询真实F行，不同时增加合成查询或交互解释器。

以下观察直接改变研究结论：

- 同预算hard与soft相当：KD增量未获支持，不将学生本来较小的体积归功于KD。
- 真实soft与置乱soft相当：优先解释为平滑或正则作用，不声称迁移了教师实例知识。
- 较大直接树已经满足实际部署预算：优先将其列为部署候选，小MLP须证明节省成本值得检测损失。
- 教师在目标区域既无优势又缺少可利用互补：停止扩展该教师；一次预先限定的TabICLv2资格检查可作为下一选择，不启动多教师全组合。
- C上的进展未在冻结后的评价中保持：保留开发过拟合结果，不继续用该评价集选模型。

本轮的新增贡献是修正研究次序、补齐学生函数类与直接基线、建立可执行筛查。CatBoost接入、混合BCE、阈值登记和阶段化执行均不单独作为原创算法主张。
