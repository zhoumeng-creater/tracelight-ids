# 统一操作点下的风险损失与正例约束

已实现一个可合成核验的数学内核：正常风险代理、正例约束和真实判决共用同一分位点；完整正常身份决定尺度和分母，完整保护身份决定对偶变量。训练与评价的正例保护由此使用同一严格判决，替换 R31 中绝对分数软锚与逐身份排序硬保护的错配。风险族尚未批准，当前没有新训练注册、优化器配置或运行入口。

## 1. 分数、数据角色与阈值

底座保持原 32 成员模型。成员 logit 是 FP32，部署分数仍为

$$
s_\theta(x)=\operatorname{logit}\left(\frac1{32}\sum_{m=1}^{32}\sigma(z_{\theta,m}(x))\right).
$$

复用 `route_batch/tabm_training.py::stable_scores` 的 FP64 `logsigmoid/logsumexp` 实现，不先计算会饱和的概率，也不把平均 logit 当作该分数。合成测试检验 32 个成员都有梯度及 ±1000 logit 的有限输出。

参考正常集合 $Q$、风险正常集合 $N$、保护正例集合 $P$ 按身份两两不交，内部身份唯一。$N$ 的风险组是完整分区，所有预登记组非空。相同可见输入或相同分数的不同身份保留各自一行；重复身份不能增加权重。`FrozenRiskManifest` 固定这些身份与顺序，任何缺行、重排、重号或漏组均拒绝。

对当前 $Q$ 分数使用既有序位规则：

$$
k_Q=\left\lceil\frac{99(|Q|+1)}{100}\right\rceil,
\qquad t_Q(\theta)=s_{Q,(k_Q)},
\qquad d_\theta(x)=\mathbf1\{s_\theta(x)>t_Q(\theta)\}.
$$

序位用整数运算，至少需要 99 条正常参考。前向取精确 kth 值；后向在全部等于该值的参考身份间均分导数。这是预先声明的广义梯度约定。没有插值、抖动、分组阈值或停止阈值梯度。

`build_risk_state` 的实参 `point` 用于身份和数值核对；随后从传入的当前 `reference_scores` 重建同值分位点图。旧图或脱图的 `point` 因而不会悄悄切断当前 $Q$ 的梯度。

## 2. 正常风险代理及其尺度

采用完整正常身份并集 $Z_0=Q\cup N$，各身份权重恰为一次：

$$
\sigma_\theta=\operatorname{std}_{\mathrm{population}}
\{s_\theta(x):x\in Z_0\},\quad
r_i=\frac{s_\theta(x_i)-t_Q(\theta)}{\sigma_\theta},
$$

$$
\widetilde R_g(\theta)=\frac1{|N_g|}\sum_{i\in N_g}[1+r_i]_+,
\qquad L_N(\theta)=\max_{g\in\mathcal G}\widetilde R_g(\theta).
$$

尺度保留梯度，不使用冻结尺度、数值下限或小批量估计。最差组并列时，`amax` 在并列组间均分梯度。每组分母是清单中的完整正常身份数，不能只用当前抽到的行数。

逐行有 $\mathbf1\{r_i>0\}\le[1+r_i]_+$，因此这是当前共同阈值下经验正常风险的上界；它没有被解释为精确风险、概率上界的置信保证或 D 的误报保证。对实数运算中的统一正仿射变换 $s'=as+b$、$a>0$，有 $t'=at+b$、$\sigma'=a\sigma$，故 $r'=r$。值与梯度测试同时覆盖这一性质，避免只在前向归一化、后向却通过固定尺度缩放分数。

**正例不进入尺度。** 初稿采用 $Q\cup N\cup P$，存在直接压缩攻击余量的捷径。若最差组全部 hinge 活跃、其正常均值 $\bar s_g<t$，且某个正例 $p$ 高于全体均值 $\mu$，则该初稿有

$$
\frac{\partial L_N}{\partial p}
=-\frac{(\bar s_g-t)(p-\mu)}{M\sigma^3}>0.
$$

仅最小化正常项就会降低这个正例分数。最终实现改为 $Z_0$ 后，$L_N$ 与独立的正例分数张量没有梯度连接；只改正例分数不能改变正常代理。模型参数仍由正常与正例共享，因此这不等于正常更新不会影响正例预测。

若 $\sigma_\theta=0$，全部正常分数相同，严格判决的正常风险已经为零。此时返回 `DegenerateNormalScale`，不加 floor；由真实评价检查正例是否仍检出。常数正常、分离正例可以是有效终态，不能据此宣布合同不可行。

## 3. 正例代理与真实残差分离

对完整 $P$ 定义连续归一化余量 $r_i=(s_i-t_Q)/\sigma_\theta$，原始更新使用可微代理 $\widetilde g_i=-r_i$，对偶更新使用

$$
v_i=\mathbf1\{s_i\le t_Q\}-r_i
=\begin{cases}-r_i,&s_i>t_Q,\\1-r_i,&s_i\le t_Q.\end{cases}
$$

正常数值表示下，$v_i\le0$ 与严格 TP 完全一致；阈值并列及漏检的 $v_i\ge1$，安全正例的 $v_i<0$。单位跳变取自 0/1 判决指示，没有增加可调 margin，也没有放宽严格检出条件。指示项使用原 FP64 分数比较，不从归一化余量的符号推断。

每个登记正例保留一个非负乘子，训练侧每次计算完整 $P$，不把观察到的批次重定义成全部保护身份：

$$
L_{\mathrm{primal}}=L_N+\sum_{i\in P}\operatorname{stopgrad}(\lambda_i)\widetilde g_i,
\qquad
L_{\mathrm{dual}}=\sum_{i\in P}\lambda_i\operatorname{stopgrad}(v_i),
$$

$$
\lambda_i^{+}=\max\{0,\lambda_i+\eta_\lambda v_i\}.
$$

因此阈值并列且 $\lambda_i=0$ 时，一次对偶更新得到 $\eta_\lambda$；安全余量为正时，乘子可以下降。每个身份单独更新，一个漏检不能被其他安全正例的负残差均值抵消。代码核对 $v_i\le0$ 与实际严格判决一致，并单独拒绝安全正例的归一化余量下溢为零；不应用数值容差。

这是一套明确的代理/真实残差分离规则，不是同一个光滑拉格朗日函数。在非边界区域，指示项局部不变，连续代理提供残差的局部导数；阈值边界则由单位对偶信号和声明的连续代理共同处理。原始与对偶项使用同一次更新前的状态。内核不选择步长、优化器、权重衰减或初始乘子，也不叠加旧 BCE/MSE。

`nextafter(t_Q,+∞)` 只记录可表示的最小严格通过分数，不参与更新。弃用的初稿曾以 $(\operatorname{nextafter}(t_Q,+∞)-s_i)/\sigma_\theta$ 驱动对偶；它虽能表示严格可行集，但阈值并列只产生 ULP 量级增量，不能据此称已解决恢复能力。最终的单位漏检指示解决这一特定弱驱动问题；并不保证每次参数更新可行或非凸优化收敛。零乘子初始化仍意味着第一次原始更新没有正例约束力，真实逐身份判决负责拒绝丢失保护正例的检查点。

连续余量和正常代理具有实数运算中的统一正仿射不变性。有限精度可能改变相邻可表示数及比较结果，因此不声称 ULP 诊断或浮点变换完美等变。

## 4. 代理下降不能代替真实端点

完整合成例固定

$$
Q=\operatorname{linspace}(0,1,99),\quad
N=(0,0.5,0.9,1.01,1.2),\quad P=(2).
$$

所有分数统一做严格单调变换 $s'=\log(1+s)$ 后，最终正常尺度版本的 $L_N$ 从 **0.6725370824** 降到 **0.6475162409**，正常报警仍为 $(0,0,0,1,1)$，正例仍检出。测试保留这个反例；不把它改写成有效风险改善。

任何只依赖严格排序、且对所有严格单调变换都不变的目标，在排序不变的开区域内局部为常数，无法同时提供有用的普通梯度。因此本内核消除了统一平移与正缩放捷径，但不声称所有非线性分数重塑都已排除。

选择应在另一个冻结验证清单及其共同参考 $Q_V$ 上，重算真实整数 FP 与完整保护正例判决：先要求全部登记正例严格检出，再比较最差组的精确分数风险 $\max_g FP_g/|N_g|$ 是否严格改善。仅代理下降、H 下降或分数更负均不构成候选改善。D/C/E 不进入这项梯度、风险族选择或验证选轮；最终仍使用预先冻结的原部署校准与独立评估合同。

这一选轮规则能拒绝上述伪改善，不能反过来证明代理一定会找到有效改进。风险族必须先有支持与来源解释；当前精确模式及粗叶的覆盖审查没有批准一个可直接开训的主风险族。

## 5. 作者实现中复用什么、我们新增什么

参照 [Cooper（2025）](https://arxiv.org/abs/2504.01212)，固定作者提交 `30b158e0262f7073c7710bd1c4996e462af4ff35`（2025-06-09）：

- [`constraints/constraint_state.py:15–49,60–79`](https://github.com/cooper-org/cooper/blob/30b158e0262f7073c7710bd1c4996e462af4ff35/src/cooper/constraints/constraint_state.py) 区分可微 `violation` 与更新对偶的 `strict_violation`，缺省时复用同一残差。
- [`formulations/utils.py:49–88`](https://github.com/cooper-org/cooper/blob/30b158e0262f7073c7710bd1c4996e462af4ff35/src/cooper/formulations/utils.py) 分别计算 `detach(lambda) * violation` 与 `lambda * detach(violation)`。
- `simultaneous_optimizer.py:56–76` 分别反传后更新；`constrained_optimizer.py:59–95` 使用对偶上升并投影乘子。

本地保存的前两份源码 SHA-256 分别为 `ab438ea23a53c6408075d94855953b0e4bf9aa1b90deba288d7417e8eebf5cc4`、`622b60ffe24ad22d68e5ef25715fbe238764ed434fd70daaf7c7ef9771ac991f`。实际采用的是分离原始/对偶计算图与有符号残差机制；统一分位点、正常尺度、hinge 风险代理、带单位指示的正例真实残差、身份清单和选轮条件是本项目设计，不能归给作者，也不继承非凸训练的收敛保证。

## 6. 实现、核验与进入训练前剩余工作

`risk_losses.py` 提供 `aggregate_member_scores`、`build_risk_state`、`RiskLossState` 和 `DegenerateNormalScale`；复用已有 `operating_point.py` 和 `evaluation.py`。`constrained_step.py` 接收 `normal_objective`、有序 `positive_ids`、`positive_proxy_violation`、脱图的 `positive_true_violation`。

本轮仅运行人工 logits/分数测试：本模块 **28 项通过**，覆盖 32 成员稳定聚合、全量身份与分母、正仿射值及梯度、正例不参与正常尺度、严格阈值并列、同图分位点、单位对偶恢复、漏检不能被均值抵消、安全负残差、常数正常终态和单调变换反例。与分位点、真实评价、原始/对偶三个模块联跑共 **117 项通过**；Ruff 通过。真实模型加载、真实前向、拟合及 C/E 搜索均为 0。

全量目标并不要求同时保留全部模型前向图：后续可用固定 eval 状态下的两遍向量雅可比积，先得全分数及对分数的梯度，再按原身份顺序分批重算并累积参数梯度，最后只更新一次。两遍期间参数、状态及随机性必须不变，分数须一致；本轮未实现或执行此模型级流程。

进入训练前还需固定有依据的风险族、Q/N/P 身份角色、完整比较基线、优化预算和同精度检查点规则，并验证基线在新操作点的真实保护状态。若固定预算内没有满足真实保护且真实风险改善的点，就保留原基线；不靠移动阈值、缩减保护身份或改风险族追认成功。
