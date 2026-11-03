# V4 前置审查：工作点、可行更新与成熟低 FPR 对照

核查日：2026-11-03。本文只读本项目回传 metadata、近期原文和作者项目，未运行真实训练、未复现外部项目、未修改算法源码。真实训练仍采用 GitHub 交付→用户在实验室 RTX4090D 拉取运行→回传工件；没有尝试 SSH。此文是后续设计依据，不代表 V4 已实现或有效。

**建议先修“可行起点＋真实更新接受”，再判断软信息是否值得保留。** 仅提高 dual、改名为 safe KD，或把教师 raw score 拟合解释成边界校准，都不能回应本轮失败。若成熟 pAUC/CVaR 或同标签 hard＋安全更新已解释全部收益，就撤销新的蒸馏机制主张。

本文续接[已有最近邻审查](README.md)、[OPOD/FPR 源码审查](opod_fpr_code_audit.md)、[PCT/MPT 源码审查](pct_mpt_code_audit.md)和[全景定位](../landscape_20261103/README.md)。PCT、MPT、OPOD、C-GUARD、RCAT 的既有机制不再重复计作文献增量。

## 1. 本轮结果指向哪些问题

只读入口：`local/risk-v3-return-audit-20261103/metadata/risk-v3-{unsw,cicids}/summary.json` 与 `students/risk_kd-s{17,29,43}/history.json`。以下是保存 metadata 的独立计数，**不是本文重新读取预测分数计算指标，也不替代外层 ZIP/完整 inventory 验收**。

| 环境／risk_kd seed | 120 步中 actual_C_feasible | surrogate_feasible | eligible | 能确认的现象 |
|---|---:|---:|---:|---|
| UNSW 17 | 77 | 0 | 0 | 大量真实事件预算可行步骤仍未通过代理 |
| UNSW 29 | 58 | 0 | 0 | 同上；最终 normal 实际 FPR 也超限 |
| UNSW 43 | 8 | 0 | 0 | 真实与代理可行均较困难 |
| CICIDS 17 | 92 | 6 | 6 | 有可行候选，不能称所有 risk KD 都训练失败 |
| CICIDS 29 | 89 | 37 | 37 | 同上 |
| CICIDS 43 | 91 | 6 | 0 | 曾满足 C 代理，但没有同时满足现有 validation 筛选 |

一个具体反例：UNSW seed17 最终 risk_kd 的 normal 实际 FPR=0.0093112、normal 代理=0.0093892，均低于 1%；未通过的是 `family_0/2/3` 代理，分别约 0.0102545/0.0106545/0.0107003，而真实丢失 TP 比例分别为 0/0/0.00174027。因此“normal FPR 已压住”不等于整体合格，“有代理超限”也不等于对应真实事件超限。

其第一步 normal 实际 FPR 从锚的 0.0102041 变到 0.0172194，代理达 0.0824086；此时风险 Lagrangian 为 0（初始 dual 为 0）。这是明确的更新轨迹现象，**尚不能单凭日志证明 KD 梯度是唯一原因**。应分别核查 hard、soft、约束梯度与 AdamW 实际位移；不能用梯度范数替代位移或事件翻转证据。

当前必须区分三种失败：①不存在／尚未找到符合完整要求的起点；②可行点被实际更新破坏；③真实事件可行，但连续代理过紧。三种情况需要不同修复，不能统一归为“再训练久一点”。

## 2. 实时核实的文献与作者项目

读取层级：“全文”指方法及限制的相应章节；“代码”指下列固定文件的静态阅读；均没有安装、执行或复现。版本日期限制到 2026-11-03。正式身份和理论适用范围分别记录。

| ID | 正式身份／原文入口 | 本次精读证据与作者工件 | 对本项目的约束 |
|---|---|---|---|
| V01 | **Feasible Learning，AISTATS 2025**，[PMLR 正式记录](https://proceedings.mlr.press/v258/ramirez25a.html)、[全文](https://arxiv.org/html/2501.14912v1) | §2–3、Eq.(FL)/(RFL)、§2.1 的 infeasible problems；[作者代码](https://github.com/juan43ramirez/feasible-learning)，固定 commit `864a585ff685cdbb8e091983c98026b2ec38c39f`（2025-01-20）；全文＋代码 | 逐样本损失约束、动态权重和 slack 已有；“feasible”方法名不代表每个中间迭代满足原预算 |
| V02 | **Dual Optimistic Ascent (PI Control) is the Augmented Lagrangian Method in Disguise，ICLR 2026 主会**，[正式记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c66afa3f372ceb9307736004d322d88a-Abstract-Conference.html)、[全文 v2](https://arxiv.org/html/2509.22500v2) | §3、Thm.1–2、§4.4；[作者代码](https://github.com/juan43ramirez/pi-control-is-alm)，commit `edb340719b739aa130df221d6c9690349d4c6fa8`（2026-01-26）；全文＋代码 | 等式有特定初始化下的轨迹对应；不等式不是任意迭代轨迹恒等，也不是逐步可行保证 |
| V03 | **Neyman-Pearson Classification under Both Null and Alternative Distributions Shift，ICLR 2026 主会**，[正式会议 PDF](https://openreview.net/pdf?id=pHckxhmBlI)、[作者公开全文 v1](https://arxiv.org/html/2511.06641v1) | 正式 PDF 搜索索引核到 published 身份，直接打开受 OpenReview 验证阻挡；精读的是 2025-11 的 v1 §3–5、Assumptions1–3，不能称已逐式核对终稿。所核原文未提供作者代码入口 | 类条件漂移下源域约束不保证目标域约束；凸性／复杂度等条件不能套到当前固定小 MLP |
| V04 | **Closing the Approximation Gap of Partial AUC Optimization: A Tale of Two Formulations，TPAMI 作者接收版**，[全文](https://arxiv.org/html/2512.01213v1)、[相关正式 DOI](https://doi.org/10.1109/TPAMI.2025.3637063) | 全文首页载明 2025-11-23 接收；作者[主页](https://statusrank.github.io/)列 2026 卷期，出版社页本次未成功打开。§III–IV、Eq.(3)/(6)及 Thm.1；全文，未核到对应完整作者实现 | top-k 负例、阈值学习与 instance-wise reformulation 已有；不能把“动态困难正常样本＋阈值变量”本身称新。本文不采用主页上另列的同名 CVPR 条目作为已核身份 |
| V05 | **Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control，Annals of Applied Statistics 19(2), 2025**，[出版社 DOI](https://doi.org/10.1214/24-AOAS1998)、[IMS 正式期刊 PDF](https://www.imstat.org/publications/aoas/aoas_19_2/aoas_19_2.pdf) | [公开全文](https://arxiv.org/html/2110.01052v5) §1.1–2.4、Thm.1、Eq.(2)；[作者项目](https://github.com/aangelopoulos/ltt) README／目录。研究始于 2021，不能因正式刊年 2025 称全新机制 | 候选训练与风险验收、有效 p 值、多重候选选择必须区分；不是在 C 上反复训练后即可获得证书 |
| V06 | **Certifying Model Upgrades with Slice-Wise Non-Regression and Incumbent Fallback，2026 预印本**，[全文 v1](https://arxiv.org/html/2609.13714v1) | §2–4、§6–7及公开 digits 的 underpowered fallback；全文。原文 reproducibility statement 宣称补充材料，所读正文未给可直接核实的作者 GitHub，不写成已开源完整项目 | 独立配对验收＋保留 incumbent 已有；不把该预印本视作成熟大规模部署效果证据 |
| V07 | **When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee，ICML 2022**，不可回避的成熟对照，[正式原文入口](https://proceedings.mlr.press/v162/zhu22g.html) | 作者维护的 [LibAUC](https://github.com/Optimization-AI/LibAUC)、[公式文档](https://docs.libauc.org/api/libauc.losses.html#libauc.losses.pAUC_CVaR_Loss)与固定代码，commit `3c02aa9877e0091114a00f94a8c6367698eb5637`（2026-08-21）；原文机制＋文档＋代码 | CVaR-pAUC 已有可查实现；需要同学生／同监督／同预算强对照，不能只胜普通 BCE 后称新 |

**排除的“近年低 FPR”候选：**[tapAUC，2025 预印本](https://arxiv.org/html/2502.11570v2)及[作者 notebook](https://github.com/ArnaudBougaham/tapAUC)主要追求 zero false negatives／TPR 接近 100%，原文平均 FPR 达 20.43%。其目标方向不能因标题含 partial AUC 就当作本项目“固定 1% FPR”直接对照；本次只作方向核查，没有执行。

## 3. 论文公式与固定代码到底允许借用什么

### 3.1 Feasible Learning 不能替代可行性检查

V01 的 FL 是 $\min_\theta 0$，使每个训练样本满足 $\ell_i(\theta)\le\epsilon$；RFL 引入非负 slack 并最小化其平方范数。原文明确讨论不可行约束使乘子增长，及先更新 dual 的实际作用。**允许 slack 后问题可行，不等于原始预算已满足。** [§2–3 原文](https://arxiv.org/html/2501.14912v1)

固定代码的 [configs/task.py L39–60](https://github.com/juan43ramirez/feasible-learning/blob/864a585ff685cdbb8e091983c98026b2ec38c39f/configs/task.py#L39-L60)采用 `AlternatingDualPrimalOptimizer`／`nuPI`；[fl_trainer.py L33–48](https://github.com/juan43ramirez/feasible-learning/blob/864a585ff685cdbb8e091983c98026b2ec38c39f/src/trainers/fl_trainer.py#L33-L48)按观察到的样本索引更新并记乘子。本次所核路径没有 post-step 事件验收／回退。可参考可行性诊断和顺序；不能声称搬用 FL 就保证当前每步 FPR。

### 3.2 PI／ALM 是优化器参照，不是安全过滤器

V02 的标准 Lagrangian 是 $L=f+\lambda^Tg$；不等式 ALM 的原始方向使用 $[\lambda+c g(\theta)]_+^T\nabla g(\theta)$。Thm.2 讨论满足严格互补等条件的局部稳定点，§4.4 限定单步一阶情形。它没有声称当前非光滑 hinge／AdamW／离散事件每次都可行。[原文 §3–4](https://arxiv.org/html/2509.22500v2)

作者固定 [src/trainer.py L44–54](https://github.com/juan43ramirez/pi-control-is-alm/blob/edb340719b739aa130df221d6c9690349d4c6fa8/src/trainer.py#L44-L54)区分 dual-first PI 与 primal-first ALM；初始化对齐有专门处理，而非只把一个 dual 学习率换大。可列为优化消融，不能给未改的更新套上论文局部收敛保证。

### 3.3 pAUC/CVaR 要区分目标公式与用于反传的源码标量

令正例为攻击、负例为正常，$L_{ij}=\ell(f(x_i^+)-f(x_j^-))$。V07 的 CVaR 目标为

$$
\min_{\theta,s}\frac1{n_+}\sum_i\left[s_i+\frac1{\beta n_-}\sum_j(L_{ij}-s_i)_+\right],\qquad\beta=0.01.
$$

固定 [LibAUC auc.py L463–488](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/losses/auc.py#L463-L488)按当前 batch 负例数归一化，使用正例索引保存 `u_pos`，先更新它，再用 detach 的尾部 mask 反传 pairwise loss。**返回标量不是完整静态 CVaR 目标值**，不能把其日志直接与上式数值等同。另见 [SOPA 作者优化器](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/optimizers/sopa.py)。

直接适配需固定样本索引、辅助状态、正负采样、margin／分数尺度及恢复规则；不能提取一个 `relu` 就称作者复现。V04 Eq.(6)也明确是对高分正常尾部的 pairwise surrogate，Thm.1 使用有界分数及特定平方损失。不能把其理论直接转授给任意 raw-logit hinge。

## 4. 教师 raw-score matching 不等于工作点校准

以下是本项目的推导与建议，不是以上论文声称的新定理。

1. **概率校准：**输出与真实条件概率／频率的关系。class-balanced HGB 的 `predict_proba` 不自动具有原始类别先验下的 posterior 语义。
2. **工作点对齐：**在合法训练内参考正常样本上确定教师阈值 $t_T$，使其报警规则对应预定的经验 FPR；学生使用自己的 $t_S$。
3. **分数匹配：**最小化 raw 分数／logit 差，只约束数值坐标；不自动实现前两者。

**可推翻 raw matching 解释的反例：**$f_T$ 与 $f_T+10$ 的排序和 ROC 完全相同，教师参考阈值也同移 10；但固定学生阈值 $t_S$ 后，拟合两种 raw 分数会给出不同报警。更直观地，若教师正常分数不超过 2、攻击分数为 3，学生冻结阈值为 5，则学生精确拟合 raw 分数仍漏报该攻击；拟合以 $t_T\approx2$ 为中心的教师 margin，才具有相同报警侧含义。

可实施的适配是先固定折内教师参考阈值／正尺度，然后形成

$$m_T=(f_T-t_T)/s_T,\quad m_S=(f_S-t_S)/s_S,\quad q_T=\sigma(m_T/T).$$

这里 $q_T$ 是工作点 soft target，**仍不称经过概率校准的 posterior**。尺度可以来自预声明的参考正常分数分位差并设正下限；选择哪种尺度属于待检验设计，不能用 validation／旧 test 反调。T 改变软信息梯度，不能当作重新校准 FPR 的替代品。

必须对照：raw matching、仅中心对齐、中心＋尺度、同目标幅度但硬标签 matched、教师 margin 幅度在同标签／同报警侧内置乱、正仿射变换教师后重算参考阈值。若单纯中心化已解释收益，就只报告工作点适配收益，不声称新知识选择机制。尺度变化对优化有效但仍是已有归一化机制。

V03 进一步提醒：一套类条件分布上的低 FPR 约束不保证另一套分布；当前 UNSW/CIC 是分别训练评价，不能把“教师→学生”冒充论文中的“源域→目标域迁移”，更不能引用其保证宣称跨环境安全。

## 5. 最小可实施修复：先有可行基线，再接受真实更新

### P0：先证明起点属于本轮完整经验可行集

“保留 anchor”与“有 feasible anchor”不同。本轮 UNSW seed17 的 C 锚 FPR 已为 0.0102041，超过 1%；原点还有弱边界代理 slack。故不能不检查就把 step0 当完整可行解，更不能从代理中扣去 slack 制造可行。

建议对新的开发协议明确定义：先使用 F∪C 的相同 hard 标签预算做固定 Phase-I；保存所有实际事件、代理、支持量和原始阈值。若选择采用比 1% 更严格的 R 参考目标建立余量，必须事先预声明，不根据 C/V 结果反复试阈值。所有后续条件从同一个通过完整准入检查的状态初始化；对照拥有相同硬标签和 Phase-I 成本。

Phase-I 也可能失败。正常 FPR 与“相对原 anchor 丢失原 TP”可能冲突；全判正常虽 FPR=0，却未必满足攻击保留预算。应输出 `no_feasible_start`，而非把放宽后的限值仍称原 1% 合约。未经本轮准入的旧 incumbent 可以作为操作 fallback 保留，但必须标明它没有得到新合约认证。

若找到可行起点，**step0 必须进入候选集合**。所有后续更新都失败时返回精确起点，防止“已有合格模型却仅因新增训练找不到合格 checkpoint 而输出最后诊断态”。这不是算法新颖性；V05/V06 已明确候选搜索与失败回退的角色。

### P1：接受门禁放在实际 AdamW 位移之后

设当前可行参数为 $\theta_k$。普通 dual 仅改变提议方向，不能保证有限步长提议仍可行；逐坐标 Adam 预条件、动量及 decoupled weight decay 还会改变实际位移。只检查原始梯度内积／梯度裁剪不足以声称 safe update。

可实施的最小事务：

1. 保存模型、AdamW moments／step、scheduler／AMP 等相关状态；若涉及随机层，固定本次 RNG 状态。
2. 生成真实 AdamW 候选；在完整 C、相同固定 margin／阈值下计算全部准入条件。
3. 若拒绝，完整恢复状态，以预声明的缩步序列重新生成候选；试次计数有上限。不能只回滚参数却保留拒绝提议产生的 moments／step。
4. 接受后才提交候选状态和与接受状态一致的 dual 更新；全拒绝则保持原状态。记录每次缩步、拒绝原因、实际位移、额外 forward／时间。
5. 最终以实际部署 CPU 的 raw-logit 规则重新验收。若训练设备与 CPU 决策不同，只能按最终 CPU 结果认定合格；保留差异，不用概率转换掩盖。

这只是**固定 C 上的经验接受策略**，不是人口 FPR／新来源风险／优化收敛证书。训练内反复使用 C 完全合法，但也使它不能兼任 V05/V06 所需的独立验收集。缩步不保证进展：可行边界可能阻止所有正步长，而 $\alpha=0$ 只是保持基线。

需先约定门禁究竟要求：实际事件可行、连续代理可行，还是两者都可行。不同版本应分别命名和报告，不偷偷更换条件。要判断代理保守性，所有方法都同时报告两类口径；不能只展示 stricter gate 后的候选数量而归因于模型能力。

**关键反例与单元检查：**

- 当 $g(\theta)=\theta\le0$、起点 $\theta=0$、目标 $f=-\theta$ 时，任意正向步都违规；backtracking 应拒绝并保持基线，不能声称总能找到改进。
- 构造原梯度满足约束线性化、AdamW 实际位移却越界的二维例子，确认门禁检查真实参数后的风险。
- 强制首次更新拒绝，比较模型、optimizer、dual、RNG／scheduler 恢复后与“未做该提议”路径一致；只比较权重不够。
- 全部提议失败应精确返回 step0 hash；起点不合格应明确失败，不能用全拒绝伪造“安全”。
- 接受时实际事件达标但代理不达标的例子必须保留，分别测试两种预声明门禁，不能改报告标签消去失败。

## 6. 成熟且直接的强对照

最小必要矩阵不是再添加十个教师，而是同一个小学生、相同 F∪C 硬标签、相同初始化／参考阈值和开发选择预算下比较：

| 控制 | 解决的辨识问题 | 不能借此声称什么 |
|---|---|---|
| feasible hard step0／hard 继续训练 | 余量、共同 hard、初始化是否已足够 | 起点保证不是新 KD 效果 |
| hard＋P1 接受／回退 | 是否只是优化安全过滤产生收益 | 超过普通 hard 不能归因于教师 |
| hard＋单侧 CVaR 正常损失 | 是否只需强化高风险正常尾部 | 它不是 pairwise pAUC，也不是恰好 1% FPR 保证 |
| hard＋CVaR-pAUC，及其 KD 版本 | 已有低 FPR 排序目标是否解释收益 | pAUC 区间改善不等于工作点达标或家族非劣 |
| boundary KD＋同一 P1；同幅度 matched 控制 | soft information 的独立增量 | 教师重加权／尺度收益不等于新机制 |
| risk-hard／risk-KD＋同一 P1 | 显式约束及 KD 各贡献多少 | 不能利用不同可行门槛制造胜出 |

**单侧 CVaR 正常损失的简单适配：**对完整训练正常池的 $\ell_j=\operatorname{softplus}(m_j)$，使用 $\min_\eta\eta+(\beta n_0)^{-1}\sum_j[\ell_j-\eta]_+$。这是成熟 CVaR 聚合，不是新损失；它强调正常尾部，但没有直接比较攻击与正常排序，必须与 V07 的 pairwise 版本分开。

**避免 minibatch 1% 尾部伪精确：**若 batch 仅有 32 个正常样本，$0.01\times32<1$。取 1 个 hardest normal 实际对应 3.125% 的离散尾部；不能声称精确优化全体 1%。优先在固定完整训练正常池计算尾部／辅助阈值，或采用已核 V07 的索引状态估计并报告估计误差。所有 pair 只来自合法训练角色，chunking 后必须保持全局分母，不能每块独立 top-k 后不加权平均。

pAUC 还需声明：区间 $[0,0.01]$、raw area／除以区间宽度／McClish standardized 三者是哪一种，ties 如何处理。选择开发阈值和计算 ROC 曲线是不同操作；报告固定阈值 FPR/TPR、家族原 TP 丢失、候选通过率与成本，不能将区间指标代替主准入条件。

## 7. 可保留的研究问题与退出条件

可研究的问题是：**在已验证可行的小学生起点、相同后置接受门禁和相同监督／计算预算下，折内工作点教师 margin 是否仍能提供超出 hard、matched 和 CVaR-pAUC 的改善？** 这比“叠 gate＋dual＋难样本”更可反驳，但当前并无成功证据，也不能由组件组合推出首创。

1. 若 hard＋P1 已取得同样结果：归因为可行优化／安全更新适配，不归为 KD 创新。
2. 若 matched 与真实教师相同：撤销软信息增益解释。
3. 若中心化控制已足够：报告工作点适配，停止制造新的 confidence 名称。
4. 若风险代理长期拒绝真实事件可行且有用的候选：公开代理代价；可另立预声明的 actual-gate 版本，不事后替换原结论。
5. 若 backtracking 几乎全拒绝、最终总是 step0：这是安全保持但无有效升级；停止按“算法改善”包装。
6. 训练、选择、C 接受／回退均使用过的数据不能提供新来源确认。将来若应用 LTT／独立配对非劣检验，必须重新规划真正未使用的验收数据与独立单位；hash 分组隔离不自动证明样本 i.i.d.，逐组等权又改变原来的逐行风险目标。

当前允许交付的是文献审计、可执行的最小设计与必要控制。旧 calibration/test 不为 V4 排名、反调阈值或认证；若缺新数据，不借近期论文的保证填补这个缺口。

## 8. 本轮完成与缺口

- 已实时核查：七项机制来源的原始入口；其中 FL、PI/ALM、NP 迁移、PAUC reformulation、LTT 与 slice-wise fallback 精读相应方法／限制；三个作者代码路径固定版本静态核查。
- 已只读计数：两个回传环境的 risk_kd 可行步骤及具体代理拒绝现象。未在本文重算预测成绩，未把保存 metadata 当作全部工件已通过验收。
- 未完成：V4 代码、Phase-I／backtracking／CVaR-pAUC 在本项目的合成实现与实验室实测、独立新来源确认。
- 尚有来源缺口：NP 正式终稿直接访问受验证阻挡；PAUC 出版商页未打开、对应作者代码未核到；slice-wise 预印本补充代码无可核公开 URL。均没有补造或称完整作者复现。
