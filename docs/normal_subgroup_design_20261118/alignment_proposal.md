# 正常子群单侧分位目标：模型与训练候选

日期：2026-11-18。状态：设计附件，已合入总设计；本文件没有执行拟合、模型前向、新评分、阈值搜索或候选判决。

## 1. 唯一采用的改动

保留 R30 blind TabM+PLE 及每个种子的选中工件，增加一段**冻结单侧分位目标的受保护微调**。从 F 正常数据预先确定子群；对每个子群，按 R30 原分数的组内分位，把偏高的正常分数向 F 全体正常同分位压低；已经低于目标包络的正常记录保持原目标。正例保留原分数作为单侧保护。部署仍只有原结构的 TabM，不增加分组阈值、教师、组合模型或规则。

此设计直接作用于正常子群之间的相对分数。它不依赖已经饱和的正常—攻击 pair loss 来间接改变正常子群排序，也不要求扩大模型或继续增加 B 正例重复次数。

所有目标在微调前冻结。每步在相同 256 行上分别做 train-mode BCE 前向和 eval-mode 辅助损失前向，再共同反传、一次 clip 和 step；不在线对全部 14691 条正常构图排序。它是固定经验分位目标的函数拟合，**不称为每步重新求解的最优传输，也不声称严格实现所有子群分布相等**。

## 2. 模型、输入和数据边界

模型保持五数值、五缺失标记、协议 UNKNOWN、F 中拟合的预处理与 PLE、32 成员及概率平均。数值字段为 duration_seconds、orig_pkts、resp_pkts、orig_ip_bytes、resp_ip_bytes。输入宽度、层宽、dropout、成员数和参数量不改变。

F 是唯一分组拟合及训练目标来源。R30 同种子的完整 F 原分数和 identity 清单作为冻结锚表。V 只读选轮；C/E 不参与分组、目标、梯度、超参数或选轮。五个 D 错误形态不写入组规则。分组在 A/B 的 F 正常各建根，按 orig_pkts、resp_pkts、orig_ip_bytes、resp_ip_bytes 固定轮转优先级进行平衡二叉分箱；max_depth=4、min_child=256、同值不拆、选择最接近半数的合法边界，同距取较小边界，最多 32 组。协议、duration、分数、标签误差和 D 模式不参与切分。A/B 是研究数据来源元信息，不是新增模型特征。精确树、组数和每组计数在总设计的 group contract 冻结；下列加权公式也允许未来接口中的组跨 A/B 来源。

若节点无法满足 min_child=256，停止切分并保留该父节点为叶，不额外制造 pooled 回退组。F/V 的来源必须为 A/B，否则输入合同拒绝；V 的低支持组保留在报告和既定评价合同中，不能因候选表现删组。D 不属于 A/B，不强行分配 F 来源树，不参与 V 的 H；D 通过完整正常分母、正例身份和原 15 单元验收评价。部署不读取来源或 group，因此未知来源仍由同一个 TabM 评分。经验小组尾部分位不作为未来 1% FPR 的统计保证。

## 3. 固定目标的精确定义

记原工件为 $\theta_0$，成员 logit 为 $z_{ik}$，$K=32$。训练与评价都沿用实际部署分数：

$$
s_\theta(x_i)=
\log\sum_{k=1}^{K}\sigma(z_{ik})
-\log\sum_{k=1}^{K}\sigma(-z_{ik}).
$$

两项的 $\log K$ 抵消。实现调用现有 stable_scores：FP32 网络、FP64 logsigmoid/logsumexp；保留梯度，不能 detach 当前分数。它不是成员 logits 的平均，也不能先分别对齐成员后假定平均概率已经对齐。

设 $F_0$ 为 14691 条 F 正常，$s_i^0=s_{\theta_0}(x_i)$。令 $Q_0(u)$ 为全体 F 正常原分数的左连续广义逆经验分位，即升序第 $\max(1,\lceil |F_0|u\rceil)$ 个值。组 $g$ 有 $n_g$ 条正常，记录 $i$ 的升序平均秩为 $\bar r_i$，相等分数共用平均秩：

$$
u_i=\frac{\bar r_i-\tfrac12}{n_g},\qquad
\widetilde t_i=\min\{s_i^0,Q_0(u_i)\}.
$$

组内排序、分位与初始目标一次计算。相同原分数同组记录共享初始目标，不用 identity 的排序人为拆开 ties。这里使用离散经验分位的中点规则，不把它称为不同样本数连续 Wasserstein 积分的精确解。

A/B 是训练元信息，可能把相同模型输入分到不同组。部署网络无法区分相同输入，因此再按冻结预处理后的 FP32 numeric、缺失标记和 UNKNOWN 协议输入构造等价类 $\mathcal E(i)$，最终统一为：

$$
t_i=\min_{j\in\mathcal E(i)}\widetilde t_j.
$$

这一步只读 F，保证相同网络输入的正常目标一致，仍不抬高正常。它是本项目的可实现性修正，不能再把最终目标称为逐组精确分位传输。F 的已知异标签同输入冲突为 0；目标构造前仍按冻结输入合同断言该条件。只在未来登记入口计算目标，本设计阶段没有生成任何新 target。

冻结尺度：

$$
\Delta=\max\{Q_0(0.75)-Q_0(0.25),1\}.
$$

1 是防零尺度的固定工程值；不根据候选分数重新估计尺度，不为不同种子搜索常数。目标必须保存 identity、group、模型输入等价类、原分数、组内平均秩、$u_i$、$Q_0(u_i)$、$\widetilde t_i$、最终 $t_i$、$\Delta$ 及原工件/数据哈希。

有四个直接性质：

1. $t_i\le s_i^0$：目标构造不抬高任何正常记录。
2. 初始 $\widetilde t_i$ 在同组内关于原分数单调不减。等价类取最小的最终修正可能改变局部次序，必须报告被修正数量，不能继续声称最终目标逐组单调。
3. 跨组可以改变相对排序：同一原分数在不同正常组中的分位不同，压低幅度不同。
4. 若模型完全实现目标并保留正例原分数，F 中已有正负严格间隔不会被目标本身破坏。实际网络是共享参数，必须另做逐身份核验，不能由目标性质推出 D 检出保证。

## 4. 一个完整目标，不再拼接待定项

每个原 source/class 批次各 64 条，仍是 A0/A1/B0/B1。基础项是原 member-wise BCE：

$$
L_{\mathrm{BCE}}=\frac1{256K}\sum_{i\in B}\sum_{k=1}^{K}
\operatorname{BCEWithLogits}(z_{ik},y_i).
$$

正常目标项按组等权，而非按组在 F 中的频率等权：

$$
A(\theta)=\frac1G\sum_{g=1}^{G}\frac1{n_g}
\sum_{i\in F_0:g_i=g}
\left(\frac{s_\theta(x_i)-t_i}{\Delta}\right)^2.
$$

**A 覆盖全部正常记录。** 未下调记录的 $t_i=s_i^0$，也有平方锚点，因此没有给“把所有正常一起向下移动”留下零代价。对被下调记录使用双侧目标误差：下调到固定目标后不能继续无代价降低。固定非恒定目标与平方误差限制正常分数塌缩；这仍是软约束，最终用秩和 tie 检查拒绝塌缩伪改善。

正例项按 A1/B1 各占一半，避免 90 条 B 正例被 5123 条 A 正例淹没：

$$
P(\theta)=\frac12\sum_{p\in\{A1,B1\}}\frac1{N_p}
\sum_{i\in F_p}
\left[\frac{s_i^0-s_\theta(x_i)}{\Delta}\right]_+^2.
$$

最终唯一目标：

$$
L=L_{\mathrm{BCE}}+A+P.
$$

三项系数全部固定为 1。没有温度、额外 pair margin、组别专用阈值或 lambda 网格。

### 批次估计量

原队列每次分别抽取 64 条 A0/A1/B0/B1。记正常记录来源的 F 正常总数为 $N_{p(i)}$。固定参数下，A 的组等权采样校正为：

$$
\widehat A=
\sum_{i\in B_0}
\frac{N_{p(i)}}{64\,G\,n_{g_i}}
\left(\frac{s_\theta(x_i)-t_i}{\Delta}\right)^2.
$$

这一权重即使组跨来源也成立，因为该行在对应来源 64 个抽样中的期望计数为 $64/N_{p(i)}$。**不能在上述求和之外再除以 128 或 256。** 周期洗牌的边界批次按原循环队列合同；每个完整周期覆盖身份，不用 replacement 制造额外独立样本。

$$
\widehat P=\frac1{128}\sum_{i\in B_1}
\left[\frac{s_i^0-s_\theta(x_i)}{\Delta}\right]_+^2.
$$

每步先在 train mode 做原 BCE 前向；随后切换 eval mode，在同一批数据上开启 autograd 计算 A/P，再恢复 train mode。eval mode 只关闭 dropout，不进入 no_grad。BCE先backward，辅助项随后backward累积到同一参数梯度，最后一次global norm clip、一次optimizer step。这样辅助目标作用于部署分数，避免把 dropout 扰动当成分位偏移；两个臂的 BCE dropout 流仍完全相同。锚表与检查点评分都使用 eval mode。

## 5. 必要对照与训练预算

每个种子 17/29/43 从自己的原 blind 工件分叉两条轨迹，共六条：

- transport-on：正常目标是上述 $t_i$。
- transport-off：只把正常目标改为 $t_i=s_i^0$，其他损失、权重、数据、保护、优化器、随机流、预算和选轮算法完全相同。

这叫“transport 开/关、同保护微调”比较。对照不是纯 BCE，也不是重跑 R30。原 R30 同种子工件同时作为零步锚点；六条轨迹共同解释新增训练的效果，不能只挑表现最好的一条。

固定参数：

| 项目 | 决定 |
|---|---|
| 可训练参数 | 原网络全部参数；不增参数 |
| AdamW | lr=0.0002，weight_decay=0.0003；新建且两臂相同的优化器状态 |
| 网络与评分 | FP32 网络；stable_scores 的 FP64 标量计算；不启用 AMP |
| 梯度 | 全参数 global norm clip=1，非有限即失败 |
| dropout | BCE 前向开启 0.1；辅助前向及检查关闭；同种子两臂复位相同 RNG |
| 采样 | 原四组各 64 的独立循环洗牌，batch=256 |
| 更新数 | 固定 139×16=2224；不因训练 loss 或 F 分离提前停 |
| 检查点 | step 0 与 139,278,…,2224，共 17 个 |
| 学习率调度 | 无 |
| 外延预算 | 不增 seed、不追加 epoch、不搜索 loss 系数 |
| 教师成本 | 未来登记预检以相同确定性 CPU 路径从原工件生成 F/V 锚表并冻结；每步不运行教师；学生同批两次前向 |
| 交付 | 保存选中工件与完整候选/拒绝理由表；失败回退原工件 |

与 R30 相比，lr 和固定追加预算是本次微调合同的必要组成；不能声称“所有训练条件完全没变，只替换 loss”。新对照隔离的是固定微调合同下 transport 目标的净作用。

## 6. 机制和保护必须同时成立

训练 A 下降只说明更接近目标，不能证明误报修复。机制评价必须用 V 正常的共同秩尾部，计算每个 F 定义组在 pooled V 正常最高 10%、5%、1% 中的占比/富集及最坏组超额；边界相等分数采用 fractional ties，不用 identity 任意拆分。共同严格单调变换保持此量不变，因此不能靠缩放伪造机制改善。

V 的端点是选择规则，不是新的部署阈值。逐检查点还须核验 F 正例分数保护、V 逐身份正例排序不退化、原 pAUC 保护，以及最大 normal tie 块比例不增加。精确安全门与秩指标字典由总设计的 evaluation contract 固定，不能在见到轨迹后放宽。

选择只在安全检查点中进行，先最小化正常秩尾部最坏组超额 H，同 H 取最早步。原 V pAUC 是保护门，不另作 tie-break。两臂都用相同规则。没有安全且机制改善的候选时，结论是该固定方案失败，返回原 R30；不能“换一个阈值继续试”。

最终开发验收仍按三个种子分别满足：

- B 72/72、D 1964/1964，且原 TP 身份丢失 0。
- 全部 15 个单元各自 $FP\le\lfloor n_{\rm normal}/100\rfloor$，不相互抵消。
- 逐身份列出 FP 新增/删除，F/V 正常 group 覆盖完整；D 不强行映射 A/B 树，保留原完整单元分母。
- 同时优于原 R30 和 transport-off 的正常秩尾部机制端点。只优于原 R30、却不优于同预算对照，不能归因给 transport。

完整分布相等时，组混合比例改变不会改变正常分数分布。本设计使用单侧固定经验目标，目的是削弱最坏子群尾部集中；它没有强制所有未来条件分布相等，故必须直接检验上述机制及开发端点。

## 7. 应修改的位置和实现验证

实现放在单独模块 normal_subgroup_training，不覆盖 R30 注册和已完成的历史逻辑。

| 文件/位置 | 修改内容 | 验证标准 |
|---|---|---|
| groups.py | 只用 F 输入建立、序列化 group contract；稀疏与 UNKNOWN 固定处理 | 禁止 V/C/E 输入；每条 F 正常恰属一组；顺序置换不改分组 |
| targets.py | 读取锚表、midrank、经验逆分位、单侧 target、输入等价类取最小、固定 Delta | $t_i\le s_i^0$；同输入同最终 target；初始 ties 同 target 及组内单调；最终非恒定；所有身份哈希绑定 |
| losses.py | 原成员 BCE + 精确加权 A/P | 穷举小样本抽样期望等于总体公式；不重复除 batch size；32 成员实际 aggregate 有梯度 |
| training.py | 载入 R30、六条固定 2224 步轨迹、相同随机流、17 检查点 | 对照只改 target 值；断言同初始参数、组表、锚表及 sampler sequence；保存原始梯度范数 |
| selection.py | rank-tail、安全门、tie 检查、固定 lexicographic 选择 | 全局单调变换不改变机制指标；正常常数不能冒充通过；无合格工件必回退 |
| evaluation.py | 正例身份、15 单元预算、FP 新增/删除、组支持 | 分母守恒；零 TP 身份丢失；无种子冠军筛选 |

实现阶段只用人工数值张量测试数学与控制流。真实工件前向、预备评分和拟合，须等总设计、注册和入口门均完成后再开始。此设计阶段不执行它们。

退出条件包括：F 数据/锚表不匹配；目标生成仅为统一单调变换且不能改变组间尾部；组覆盖不足以评价预定机制；训练非有限；全部检查点破坏保护门；transport-on 不优于同预算 off；V 改善而实际逐单元预算仍失败；任一 R30 TP 身份丢失。退出后记录失败原因，不原地搜索阈值、lambda 或新组。

## 8. 2025 作者实现的实际采用边界

[Auditing and Enforcing Conditional Fairness via Optimal Transport，AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33847)明确讨论条件完整分布，不只比较均值。本方案借此确定正常标签条件下的分布对象；没有把 FairBiT/FairLeap 的原算法或实验结果冒称为本方案，当前未找到可核实的对应作者仓库。

[Towards Marginal Fairness Sliced Wasserstein Barycenter，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8711bd7223057ce98a6030a687609b87-Abstract-Conference.html)的[作者仓库](https://github.com/khainb/MFSWB/tree/fb5c3630841c0a02405a26233d92c5cd49f21668)提供了实际读取的排序分位代码和分组分布训练代码：

- SWAE/metrics/wasserstein.py 第 8—13 行：searchsorted/gather 经验逆分位。
- 同文件第 38—97 行：相等样本数排序配对、不同样本数累计权重及分位区间积分。
- 同文件第 134—166 行：组列表上的分布距离与最坏组聚合。
- SWAE/swae/trainer.py 第 89—116 行：按类切分表示，分布项加入基础重建损失。
- 同文件第 35—43 行：反向传播与优化器更新路径。

原论文应用是几何平均及 autoencoder 的 class-fair representation，不是 IDS、条件正常分数或 1% FPR。我们没有采用随机投影、多维 barycenter 或原重建网络。单侧固定目标、F-only group、TabM ensemble aggregate、正例保护、加权采样和尾部验收都是本项目适配。

作者四个必要文件及 MIT LICENSE 已按原字节保存于 author_sources/khainb_MFSWB；source_manifest.json 对每个文件记录 Git blob SHA-1、SHA-256 与字节数，全部 blob 校验一致。仅阅读，没有 import 或执行作者代码。
