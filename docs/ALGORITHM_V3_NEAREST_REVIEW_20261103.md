# V3 约束蒸馏设计：近邻工作与数学红队审查

日期：2026-11-03。本文只读核查论文、作者代码及候选数学设计，不记录真实数据训练结果，不修改源码或配置。**建议把“显式正常误报增量／攻击保留约束”作为原型主线，把梯度投影作为同目标的优化器对照。两者都已有成熟算法基础；当前不能称首创、无负迁移保证、精确 pAUC 优化或具有总体 NP 保证。**

本文区分三种陈述：**事实**指可核到的一手论文／代码内容；**审查推断**指本次独立推导；**待验证设计**指尚未被真实实验支持的候选。已读全文不等于已复现；代码可见不等于跑通。主会、期刊和预印本分开标注。

## 1. 最危险的直接近邻

| 编号 | 原始工作与身份 | 已覆盖的机制 | 与本候选仍可检验的差异 |
|---|---|---|---|
| N01 | Hayder et al., **DTO-KD: Dynamic Trade-off Optimization for Effective Knowledge Distillation**，ICLR 2026 主会 | 任务与 KD 梯度冲突／尺度主导、多目标梯度组合 | 它没有本候选的 class-conditional FP 增量、正常修复信用与攻击保留预算；不能把“KD 梯度冲突优化”当新增机制 |
| N02 | Yu et al., **CORD: Conflict-Aware Optimization for Joint RL and Distillation in Agents**，2026-10-21，arXiv:2610.29965v1，预印本 | 指定被保护目标、冲突门控、单侧最近点投影 | 换成 IDS 风险梯度只是目标适配；投影几何本身已有 |
| N03 | Zhao et al., **OPOD: On-Policy Omni Distillation**，首发 2026-07-23，v3 2026-08-04，arXiv:2607.20918，预印本 | 单侧教师缺口、各模态预算及独立对偶系数 | 它控制生成 token 的教师似然缺口，非真标签条件下的新增 FP／原 TP 丢失 |
| N04 | Yan et al., **Positive-Congruent Training: Towards Regression-Free Model Updates**，CVPR 2021 主会，旧直接基础 | 冻结旧模型、负翻转、按旧模型正确性强化蒸馏 | 本候选只能以具体风险 surrogate、净正常误报与攻击不抵扣的差异接受检验 |
| N05 | Ricci et al., **Mitigating Negative Flips via Margin Preserving Training**，AAAI 2026 主会；预印本首发 2025-11-11 | 旧模型间隔保护、旧模型与独立新模型双源 focal distillation | 其任务为类别递增，使用新类 logit bias；不同于按 anchor 每个样本间隔缩放的二分类事件上界 |
| N06 | Ghiani et al., **Regression-Aware Continual Learning for Android Malware Detection**，IEEE TIFS 2026，期刊；原预印本 2025 | 安全任务的 benign／malware 负翻转及旧正确样本蒸馏 | 已覆盖“安全检测中保护旧检测能力”；并非显式低 FPR 约束 |
| N07 | Kalan et al., **Neyman-Pearson Classification under Both Null and Alternative Distributions Shift**，ICLR 2026 主会 | 两类条件分布同时迁移下的 NP 目标、适应性程序与统计／计算条件 | 本候选不能沿用其凸程序、独立样本和函数类假设取得的保证 |
| N08 | Sikder et al., **Promoting Intersectional Fairness Through Knowledge Distillation**，ECAI 2025 正式论文 | 表格 KD 与各交叉组 FPR 方差、最大差及绝对幅度惩罚 | 组公平不等于所有组 FPR≤1%；未提供本候选的硬事件上界与 anchor 修复分解 |
| N09 | Huang et al., **Evaluation-Oriented Knowledge Distillation for Deep Face Recognition**（EKD），CVPR 2022 主会，旧直接基础 | 指定 FPR、师生各自阈值、关键关系及相对工作点蒸馏 | “教师工作点 margin”本身不能主张新颖；须加入 adapted EKD |

### 1.1 一手入口与阅读边界

- **N01，全文方法／定理／算法已核**：[正式身份](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e546ce4230c15bc422a500bb47b341ea-Abstract-Conference.html)、[正式全文，§3.1、式 (9)–(17)](https://proceedings.iclr.cc/paper_files/paper/2026/file/e546ce4230c15bc422a500bb47b341ea-Paper-Conference.pdf)。未核得作者可运行代码入口。
- **N02，全文方法及优化器局限已核**：[全文，§3、附录 B.3](https://arxiv.org/html/2610.29965v1)、[作者代码](https://github.com/LvAoAo/CORD-release)、[梯度实现入口](https://github.com/LvAoAo/CORD-release/blob/main/verl/trainer/ppo/cord_utils.py)。原文主比较每方法一个训练种子、四次解码评估；该误差条不代表训练种子稳定性。
- **N03，全文 §3.1–3.6 及代码控制路径已核**：[全文](https://arxiv.org/html/2607.20918v3)、[版本日期](https://arxiv.org/abs/2607.20918)、[作者实现](https://github.com/VincentZhao2002/OPOD/blob/main/training/method/opd_loss_patch.py)。代码按模态累计和／计数并更新系数，缺失模态不更新；本轮未执行。
- **N04，全文 §4.1–4.3 已核**：[CVPR 正式页面](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html)、[全文](https://arxiv.org/html/2011.09161)。
- **N05，全文方法及作者仓库说明已核**：[AAAI 正式论文](https://ojs.aaai.org/index.php/AAAI/article/download/37825/41787)、[作者全文](https://arxiv.org/html/2511.08322)、[作者代码](https://github.com/miccunifi/negative_flip_MPT)。不能将其类别递增实验直接称为 IDS 复现。
- **N06，全文方法已核**：[作者全文](https://arxiv.org/html/2507.18313)、[正式期刊 DOI](https://doi.org/10.1109/TIFS.2026.3714132)。TIFS 21:6845–6854；不能继续仅标为 2025 预印本。
- **N07，全文目标、迁移程序及假设已核**：[ICLR 正式身份](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33870b3e099880cd8e705cd07173ac27-Abstract-Conference.html)、[作者全文](https://arxiv.org/html/2511.06641)。未核得可直接套用本项目 MLP 的作者实现。
- **N08，全文 §3–4 已核**：[作者全文，式 (5)–(6)、(12)–(14)](https://fahimsikder.com/uploads/intersectional-KD.pdf)、[出版 DOI](https://doi.org/10.3233/FAIA251214)。论文页眉为 ECAI 2025；出版平台另有 2026 页面日期，不能据此改称 2026 主会论文。
- **N09，全文及作者代码入口此前已核**：[CVPR 正式身份](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Evaluation-Oriented_Knowledge_Distillation_for_Deep_Face_Recognition_CVPR_2022_paper.html)、[作者全文](https://arxiv.org/html/2206.02325)、[作者代码](https://github.com/Tencent/TFace/tree/master/recognition/tasks/ekd)。它使用人脸关系分数；二分类单行分数版本应明确标为 adapted。

旧梯度基础也不能省略：[GEM，NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf) 以多个历史任务风险梯度形成半空间约束；[A-GEM，ICLR 2019](https://openreview.net/pdf?id=Hkf2_sC5FX) 使用平均参考梯度；[PCGrad，NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) 对冲突任务梯度作投影。[Group-DRO，ICLR 2020](https://arxiv.org/abs/1911.08731) 已提供已知组 mixture 最坏风险目标及过拟合警示。这些属于旧基础，不作为 2025／2026 文献数量充数。

## 2. 已覆盖什么，尚缺什么

**事实：**N04 的 focal distillation 采用旧模型正确性相关权重，讨论减少 negative flip 与允许 positive flip 的矛盾；N05 加入旧／新双源模型及间隔校准；N06 将负翻转问题带入恶意软件检测。由此，“用同容量 hard 模型作冻结 anchor，保护它原先正确的样本，再从强教师学习”已经有很近的结构，不能仅更换领域后宣布新方法。

**事实：**N03 的教师缺口取正部，并按模态设置固定预算及更新对偶系数。因此“单侧损失＋不同组各自预算＋原始-对偶”也已被覆盖。其似然信号不等于教师真正确；当前候选使用真标签条件下的事件，是应明确比较的目标差异。

**审查推断：**目前最具体的候选差异只剩以下联合设计：用冻结同容量参考 B 定义风险变化；正常允许保守的旧 FP 修复信用，攻击只限制原 TP 的丢失；以 B 的逐点绝对工作点间隔调节 hinge 斜率。三项都需单独消融，不能以联合命名代替证据。本次检索没有发现完全相同的式子，**不构成不存在先验的证明**。

正常净误报分解本身是事件恒等式：

$$
\operatorname{FPR}_S-\operatorname{FPR}_B
=P(B=0,S=1\mid Y=0)-P(B=1,S=0\mid Y=0).
$$

攻击漏报也可相同分解；选择禁止新增 TP 为旧 TP 丢失抵扣，是特定安全偏好，不是普遍最优性。以上分解不能称为新定理。真正尚缺的是：所选连续 surrogate 是否足够紧、优化是否可行、低 FPR 泛化是否更好，以及相对 PCT／EKD／成熟低 FPR 目标是否还有收益。

## 3. 候选 A：投影 KD 梯度的数学边界

### 3.1 直接最近点投影已经成熟

设受保护梯度为 $a$，候选 KD 梯度为 $v$，下降更新为 $\theta^+=\theta-\eta d$。N02 的单约束几何可写为：

$$
\widetilde v=\arg\min_w\tfrac12\|w-v\|^2\quad\text{s.t. }a^\top w\ge0,
\qquad
\widetilde v=v-\frac{\min(a^\top v,0)}{\|a\|^2}a.
$$

这是已有保护方向的最近点投影；数值实现还需处理 $\|a\|\approx0$。其来源可引 N02／A-GEM，不应另命名为本项目发明。[N02 §3.2](https://arxiv.org/html/2610.29965v1)

**审查推断：**保护交叉熵梯度只能保护交叉熵的局部线性近似，不能推出保护 FPR。若要与候选 B 比优化器，A 和 B 应使用同一组风险约束；否则混淆了“换目标”和“换求解器”。

对真实连续约束 $c_k(\theta)\le0$，一阶可行条件应为：

$$
c_k(\theta-\eta d)\approx c_k(\theta)-\eta\nabla c_k^\top d\le0
\quad\Longrightarrow\quad
\nabla c_k^\top d\ge c_k(\theta)/\eta.
$$

简单要求 $\nabla c_k^\top d\ge0$ 只是局部不增，既未利用可行余量，也不保证修复已违反预算。若保留 hard 方向 $g_H$、只投影 KD 分量，正确的线性条件是：

$$
\nabla c_k^\top\widetilde v\ge c_k/\eta-\nabla c_k^\top g_H.
$$

单独让 $\nabla c_k^\top\widetilde v\ge0$ 并不能阻止 $g_H$ 自己破坏风险。多个半空间逐个只投影一次，还可能重新违反此前约束；需要联合 QP、收敛的交替投影，或明确软 slack／残差。相反约束可使非零进展困难；要求同时修复相反的已违反约束甚至可能不可行。

### 3.2 原始梯度满足约束，不代表 AdamW 的实际参数位移满足

**事实：**N02 明确把一阶保护限制在优化器变换前，并讨论 AdamW、动量与预条件的作用。[N02 附录 B.3](https://arxiv.org/html/2610.29965v1)

**本次算术反例：**取 $a=(1,1)$、$v=(1,-3)$，投影得 $\widetilde v=(2,-2)$；若保留 $a$，合成 $d=(3,-1)$，则 $a^\top d=2>0$。但正对角预条件 $P=\operatorname{diag}(0.1,10)$ 后，$a^\top Pd=-9.7$；下降 $-Pd$ 反而增大受保护线性损失。动量、weight decay、逐层裁剪及有限步曲率还会引入其他偏差。

因此，A 不能宣称“加一次投影即可保证 no-harm”。若真需要一步可行性，应约束实际位移，并另付风险复算／回溯成本；这已经超出简单投影 KD。即使一步训练 surrogate 可行，也仍无独立总体 FPR 保证。

### 3.3 DTO-KD 的原文公式不能未经检查直接移植

**事实：**N01 式 (9) 要求 $\pi$ 属于 simplex；Theorem 3.1 的式 (10) 只写和为 1，随后闭式没有裁剪。其梯度为 $g_1=\nabla\log L_{KD}$、$g_2=\nabla\log L_H$。[N01 §3.1](https://proceedings.iclr.cc/paper_files/paper/2026/file/e546ce4230c15bc422a500bb47b341ea-Paper-Conference.pdf)

**本次审查反例：**$g_1=1,g_2=2$ 时，其 affine 闭式给 $\pi=(2,-1)$，违反式 (9) 的非负条件。正确的两梯度 simplex 最小范数组合应在分母非零时使用：

$$
\pi_1=\operatorname{clip}\!\left(\frac{\|g_2\|^2-g_1^\top g_2}{\|g_1-g_2\|^2},0,1\right),\quad\pi_2=1-\pi_1.
$$

相同／零梯度需另处理；边界裁剪后“两个目标贡献相等”不能普遍沿用。原文式 (16) 的正范数下界也不能对任意梯度成立：$g_1=1,g_2=-1$ 时最小范数组合为零。这里是对**原文无额外条件表述**的反例，不声称已复核或推翻整篇实验。若实现该对照，应标明修正 simplex 求解，并稳定处理接近零的 loss；不能把作者所有保证搬入论文。

## 4. 候选 B：正常修复信用与攻击不抵扣的事件上界

本节是对本轮提出设计的独立推导，**不是从近邻论文摘录的定理，也不是已完成实现说明**。

### 4.1 固定对象、符号与 tie 必须一致

- $B$：冻结、与学生同容量的 hard-label 参考模型；$T$：提供额外知识的强教师，两者角色不同。
- $z_B,z_S$：二分类 attack logit；$t_B,t_S$：冻结的训练参考阈值。
- $a_i=z_B(x_i)-t_B$，$m_i=z_S(x_i)-t_S$，$d_i=\max(|a_i|,\kappa)$，$\kappa>0$；$a,d$ 停梯度。
- $b_i=\mathbf1\{a_i\ge0\}$，$s_i=\mathbf1\{m_i\ge0\}$，阈值相等判 attack。若实际实现采用严格大于，事件边界与测试必须同步改写。

定义正常新增 FP 上界及旧 FP 修复信用：

$$
u_i^+=(1-b_i)\left[1+\frac{m_i}{d_i}\right]_+,
\qquad
\ell_i^-=b_i\operatorname{clip}\!\left(-\frac{m_i}{\kappa},0,1\right).
$$

逐点有：

$$
s_i\le b_i+u_i^+-\ell_i^-.
$$

理由分两种情况即可：$b_i=0$ 时，$m_i\ge0$ 对应 hinge 至少为 1；否则右侧非负。$b_i=1$ 时，$m_i\ge0$ 信用为 0；$m_i<0$ 时信用至多为 1。对任意固定正常 cohort 取均值，得到：

$$
\widehat{\operatorname{FPR}}_S\le
\widehat{\operatorname{FPR}}_B+
\frac1{n_0}\sum_{i:y_i=0}(u_i^+-\ell_i^-).
$$

攻击原 TP 丢失的上界可取：

$$
v_i=b_i\left[1-\frac{m_i}{d_i}\right]_+,
\qquad b_i(1-s_i)\le v_i.
$$

其均值只约束“B 原先检出的攻击又被 S 漏掉”；$b_i=0$ 的新增 TP 无法为丢失原 TP 抵扣。由此也有 $\widehat{\operatorname{TPR}}_{S,f}\ge\widehat{\operatorname{TPR}}_{B,f}-\overline v_f$。这些都是对固定 cohort、固定阈值成立的代数界；对未知总体还缺泛化／依赖假设。

### 4.2 可行性、松弛与可优化性问题

1. **anchor 不自动满足零预算。** 只有 $m=a$ 且 $|a|\ge\kappa$ 时相应 hinge 才为零。弱边界样本在 anchor 原点的 slack 为 $1-|a|/\kappa$。必须记录初始约束值；将预算设为零后不能因为 B 的硬决策不变就称初始可行。
2. **相同权重不够。** 初始化 S=B 时还需 $t_S=t_B$ 或严格等价的 bias 对齐，才能令 $m=a$。独立为学生重估阈值会改变这一性质。
3. **改变分数尺度可改变松弛而不改变分类。** 对弱边界样本扩大同号 margin 可降低 hinge；因此收益可能只是继续 hard 训练／margin sharpening，必须有同起点、同更新数的 no-KD 对照。
4. **减去修复信用使正常约束非凸。** $-\operatorname{clip}(-m/\kappa,0,1)$ 是负 clipped ramp，即便把 m 视为标量也不能称整个约束凸；非凸神经网络的原始-对偶更新没有自动全局最优或可行保证。
5. **修复信用保守而且有平坦段。** 旧 FP 刚越过 0 只有部分信用，降到 $m\le-\kappa$ 才得到完整信用；$m>0$ 时风险项本身不给旧 FP 修复梯度，需 hard loss 等项启动修复。正常侧有净抵扣不等于每个正常子群都受保护。
6. **平滑 sigmoid 不能替代这个硬事件上界。** 如要改 softplus，应保留在 $m=0$ 时至少为 1 的归一化及逐点验证；普通 sigmoid 对 $m\ge0$ 小于 1，无法直接上界硬报警事件。
7. **所有均值的分母要冻结含义。** 正常分母为全部 $Y=0$；家族 f 攻击分母为全部 $Y=1,F=f$。若改为仅 B-negative／B-positive 条件均值，应乘对应 mask 覆盖率后才能代入上述界。
8. **家族保护可能空洞。** 某家族没有 B 原 TP，则 $\overline v_f=0$ 无法提供任何学习该家族的要求。需同时报告全家族召回、原 TP 支持数及预设稀有家族处理；不得见到结果后删除不利家族。

### 4.3 与最直接先验相比如何定位

N04／N06 已保护旧模型正确样本；N05 已引入旧参考和独立新模型、使用 margin；N03 已有单侧缺口和分组对偶预算。**候选 B 站得住的只能是一个可执行研究问题：这一特定净正常风险 majorizer，加上攻击不抵扣及 anchor 自适应斜率，在同风险协议下是否比它们更紧、更有效。**

其中“正常修复信用”接近净错误变化的连续保守化；“攻击不抵扣”接近 class-conditional negative-flip 控制。两者概念不新；自适应 denominator 也需要与固定 hinge 比较。若无法胜过负翻转蒸馏或固定 surrogate，不保留新的核心算法主张。

## 5. 可实现的原始-对偶规格与禁止偷换的预算

设冻结 hard 项与工作点 KD 项的基目标为：

$$
F(\theta)=L_H(\theta)+\alpha L_{KD}^{op}(\theta;T).
$$

工作点 KD 要具体给出教师、学生 margin 的尺度处理；不同模型 raw logits 任意缩放会改变匹配目标。若使用训练参考尺度归一化，应在 train 内估计、冻结并写入元数据；不能每 batch 归一化后仍称固定工作点。N09 已有师生各自阈值的工作点蒸馏，故这里不是创新点。

**正常只选一种预算，配置中明确命名：**

$$
c_0^{abs}=\widehat{\operatorname{FPR}}_B+\overline{u^+-\ell^-}_0-\beta_0,
\quad\text{或}\quad
c_0^{inc}=\overline{u^+-\ell^-}_0-\epsilon_0.
$$

前者约束绝对 empirical FPR 上界，后者只限制相对 B 的增量；若 B 自己 FPR 超标，后者并不能保证目标 1%。各攻击家族使用 $c_f=\overline v_f-\epsilon_f$。归一化到“原 TP 条件丢失率”则是另一种 estimand，需显式说明。

待验证的标准 Lagrangian 原型为：

$$
\min_\theta\max_{\lambda\ge0}
F(\theta)+\lambda_0c_0(\theta)+\sum_f\lambda_fc_f(\theta),
\qquad
\lambda_k\leftarrow\operatorname{clip}\bigl(\lambda_k+\eta_\lambda\operatorname{sg}(\widehat c_k),0,\lambda_{max}\bigr).
$$

这里采用成熟原始-对偶方式，不宣称新优化器。$\lambda_{max}$ 截断意味着约束实际上可能被软化；系数长期饱和且风险超标应判为违反预算，不应把它藏在平均指标中。乘子、教师输出、anchor margin 与 mask 应停止不应有的梯度；不默认对线上乘子更新做二阶反传。

实现必须处理：

- batch 缺少某家族时保持该乘子，不能用 0 当风险；保存完整 epoch 的 sum／count 审计真实约束。均匀类内抽样下比值估计与稀有组的噪声也应记录，不能把每 batch 等权平均当完整 cohort 均值。
- balanced batch 可用于训练，但需保留各 class-conditional 风险的定义；duplicate-cap 或 group 等权若进入风险均值，会改变估计对象，不能仍报告 per-flow 上界。
- 对所有条件固定 hard 项、初始化、更新步数、教师 cache、阈值来源与选择规则。保存 B／T hash、训练行集合、mask 来源、阈值、$\kappa$、预算、乘子和优化器状态。
- 连续风险满足不代表实际独立评价风险满足；至少同时报告 surrogate slack、硬新增 FP／修复 FP、原 TP 丢失／新增 TP。不能只报告总 AUC。

## 6. mixture group、重复 group 与校准的边界

**事实：**N07 已研究两类条件分布迁移；Group-DRO 已研究已知组 mixture 下最坏风险。但本候选的非凸网络、经验界和相关 flow 不自动满足这些原文保证。[N07 正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/33870b3e099880cd8e705cd07173ac27-Paper-Conference.pdf)、[Group-DRO](https://arxiv.org/abs/1911.08731)

**审查推断：**若已知正常环境 g 内分别有固定总体 $\operatorname{FPR}_{S,g}\le\beta_g$，则对这些固定条件分布的任意 mixture q，FPR 是 $\sum_gq_g\operatorname{FPR}_{S,g}$。这是基本全期望分解，不是新定理；训练经验约束尚未建立该总体前提。不同环境之间若允许共享修复信用，一个环境的新 FP 可被另一环境旧 FP 修复抵扣，只能声称固定 pooled mixture 的目标，不能声称任意 mixture 保护。

N08 已在 KD 中使用交叉组 FPR 惩罚；其式 (5) 用 hard argmax indicator 定义 FPR，式 (6) 加方差、最大差和幅度项。已核原文未交代可直接迁移的可微硬事件上界，不能照抄后声称有梯度约束保证。公平性 parity 本身也允许各组同样很高的 FPR；绝对风险上界需另给预算。[N08 §4.3](https://fahimsikder.com/uploads/intersectional-KD.pdf)

**相同特征 hash 组不是环境组。** 对确定性模型，同一输入的报警相同，其组内 hard FPR 常只有 0 或 1；若每个正常 hash 组都约束≤1%，实际上要求所有已见正常模式零报警。会话／重复组更适合划分、防泄漏、采样贡献控制与相关性诊断；环境组才可能承载 mixture 解释。训练时组等权会改变 estimand，相关样本的有效独立数量也不能由行数替代。

**FC 与 OOF 分开说。** 使用合法 train 标签拟合 full-context／full-fit 教师再蒸馏，并非测试泄漏。OOF 有助于本项目诊断“教师在未见行上的错误／负翻转”，不是所有 KD 的必需通则。若 B 或 T 使用 OOF mask，训练约束对应多个折模型；不能不经说明就称保护最终 full-fit 模型 B 的风险。需要公平 FC／OOF 对照或明确参考对象。

**最终校准不能继承训练界。** $t_S$ 改变即改变 $m,s$ 及 surrogate；最终 calibration 可能改善或破坏训练期预算。必须分别报告固定训练阈值的风险机制和部署校准后的实际 FPR／TPR，重算 bound gap。只有单调分数校准且阈值也按同一函数映射、从而硬事件完全不变时，才可保留事件结论；这仍不构成总体保证。最终 test 不用于选择预算、$\kappa$、阈值或家族集合。

## 7. 最少实验与停止条件

### 7.1 原型正确性先于训练

下列是建议交给原型测试的边界；本审查没有运行真实数据或 fit：

1. 枚举 $a,m$ 的符号、0 tie、$\pm\kappa$，检查正常／攻击逐点界、修复信用≤1；拒绝 $\kappa\le0$。
2. 检查 $m=a$ 时强 margin 零 slack、弱 margin 初始非零 slack；检查阈值不一致会破坏该性质。
3. 检查 detached denominator、全 cohort 分母、空家族及原 TP 为零；与直接 sum／count 结果比对。
4. 检查正常旧 FP 修复可抵扣，而攻击新增 TP 不抵扣原 TP 丢失；保存前后 FP／TP 转移表。
5. 投影对照检查多约束残差、退化梯度以及实际优化器位移；DTO 对照必须覆盖 simplex 裁剪反例。

本轮仅做过独立算术核验：$a,m\in\{-10,-2,-1,-0.5,-0.1,0,0.1,0.5,1,2,10\}$、$\kappa=0.5$ 共 121 对，正常／攻击界未发现违反；另算得 DTO affine 权重 $(2,-1)$ 及预条件反例的内积 $-9.7$。这是公式边界检查，不是模型有效性证据。

### 7.2 分阶段证伪，避免一次铺满组合

**第一阶段建议 5 条件 × 3 个种子，即 15 个学生 fit**：只选一个固定教师、一个 B、一个数据环境。不是给已有研究追加无条件 15 次；是否启动由主流程决定。

| 条件 | 保持不变的部分 | 回答的问题 |
|---|---|---|
| C1：从 B 继续 hard 训练 | 起点、更新数、hard 项、采样一致 | 只是继续训练／margin sharpening 吗？ |
| C2：普通工作点 KD | 同 T、同更新数与 threshold 协议 | 比普通 KD 是否有必要？ |
| C3：PCT-style focal distillation 到 B＋同一 T 的 KD | 显式分开旧参考和强教师；适配超参仅用开发集 | 是否只是成熟负翻转保护？ |
| C4：候选风险约束＋无 KD | 同 $\kappa$、预算、乘子更新及起点 | 收益来自知识，还是仅来自风险训练？ |
| C5：完整候选 B | 同上，加入同一工作点 KD | 联合目标是否有额外收益？ |

若第一阶段通过，再做**小的第二阶段**，每项先固定一个最有希望而非 test 选出的配置：固定 denominator hinge；去掉 normal repair credit；攻击改为净抵扣；相同风险目标的投影优化器；adapted EKD；成熟 CVaR-pAUC／hard 低 FPR 目标。不能把这六项与所有教师／容量／特征／种子直接作笛卡尔积。N09 与成熟低 FPR 目标是正式方法主张前所需的强对照；第一阶段的 15 fits 不足以代替它们。

CVaR-pAUC 的成熟主干可使用 [When AUC meets DRO，ICML 2022](https://proceedings.mlr.press/v162/zhu22g.html) 及其[作者团队 LibAUC](https://github.com/Optimization-AI/LibAUC)。它与固定阈值的 surrogate 约束不同；不能把本候选称为该论文的精确 pAUC 实现。

### 7.3 必须报告与击穿条件

- 在相同校准预算／FPR 工作点比较召回与家族保护；固定训练阈值和最终部署阈值分开报告。新增 FP、修复 FP、丢失原 TP、新增 TP、各 mask 支持数不可只合并成净变化。
- 报告原始／最终 constraint、乘子饱和比例、$\kappa$ 截断比例、logit 尺度、bound gap、真实环境各自风险及 CPU 学生部署成本；教师训练／查询成本另列。
- 若 C4 与 C5 相当，放弃“知识迁移产生收益”的解释；若 C1 相当，优先解释为继续训练／margin sharpening；若 C3 相当，不主张优于既有负翻转机制。
- 若固定 hinge 与自适应 denominator 相当，删除该复杂性；若无 repair credit 已足够，信用机制不作为贡献；若净攻击预算更好且原 TP 丢失可接受，承认 no-credit 偏好存在代价。
- 若优势依赖提高实际 FPR、忽略最差环境、丢弃稀有家族、或乘子长期饱和，则判定预算设计失败，不能只用总体准确率宣布胜出。

## 8. 最终建议与允许的贡献措辞

优先实现 **B：冻结同容量参考、固定参考阈值、显式正常净增量与攻击原 TP 丢失 surrogate、标准原始-对偶**。它比单纯“投影 KD 梯度保护 hard loss”更贴近低误报研究问题，也较容易给出逐点事件核验、风险分解及失败诊断。投影 A 保留作同约束的优化器对照；不作为另一条独立算法创新主线。

在独立实验之前可写：“提出并原型化一种面向固定工作点的参考模型风险约束设计，研究正常修复抵扣与攻击保留不抵扣的作用。”不能写：“首次提出风险蒸馏／首次约束 KD／保证低 FPR 无退化／证明任意分布迁移安全／精确优化 pAUC”。

最终能否成为算法贡献，取决于三个窄问题的证据：**事件 surrogate 是否比固定 hinge 更有效；正常信用和攻击不抵扣是否各有必要；在同校准风险下是否稳定胜过 PCT-style／EKD／成熟低 FPR 对照。** 如不能击穿最接近先验，贡献应降为透明、可复现的 IDS 适配与风险分析，不用新缩写维持新颖性。
