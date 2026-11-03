# Algorithm v3：锚定残差风险约束的数学与统计审查

日期：2026-11-03。范围：设计审查、有限合成算例与工程单测；不读取真实训练/保留分区，不进行真实训练。本文不证明方法新颖、有效、人口风险保证或优化收敛。相关先例与创新边界见 [全景综述](landscape_20261103/README.md)和[最近邻审查](landscape_20261103/nearest_work_and_novelty.md)。

## 1. 核心建议

候选可行，但必须把三个对象分开：**固定 hard 学生的决策锚、用于学习的风险代理、冻结决策后的独立核验**。建议采用基线决策分区的一侧 hinge 上界，正常风险给有限修复信用，攻击保留不给新增 TP 抵账。它比“softmax 置信＋困难样本”更清楚地定义了要保护的错误，但仍可能只是已有约束学习与蒸馏的任务适配，须由近邻和反例检验。

最重要的统计边界：参与梯度、dual 更新、选轮、预算调整或停止决策的 audit pool，已经是 **constraint-training pool**。其经验上界可以逐样本成立，其结果不能因此升级为未见数据风险保证。另留 holdout 也不自动消除研究者历史使用旧数据造成的适应性。

## 2. 一个具体候选：同容量 hard 锚下的残差风险约束蒸馏

### 2.1 固定对象与单位

固定 hard 基线 $B$ 的权重、预处理器、阈值 $t_B$ 及接受并列的规则。学生与它具有相同部署架构，从同一权重初始化；额外目标和 dual 仅在训练存在。令

$$
a_i=z_B(x_i)-t_B,\qquad m_i=z_S(x_i)-t_S,\qquad B_i=\mathbf 1\{a_i\ge0\},\qquad S_i=\mathbf 1\{m_i\ge0\}.
$$

默认训练期固定 $t_S=t_B$，让输出层自行适应。若把 $t_S$ 作为可学习截距，须明确它属于训练参数，并与所有对照采用相同自由度；不能边看验证结果边改阈值而不记账。阈值、score 空间、tie 规则必须绑定到同一个 decision version。

取 $\kappa>0$，adaptive 模式用 $d_i=\max(|a_i|,\kappa)$；fixed 模式用 $d_i=\kappa$。二者都可用于下述点态界，但优化几何不同，不能混为同一方法。

### 2.2 正常类：新增 FP 上界减保守修复信用

$$
u_i^+=\mathbf 1\{B_i=0\}\,[1+m_i/d_i]_+,\qquad
\ell_i^-=\mathbf 1\{B_i=1\}\,\operatorname{clip}(-m_i/\kappa,0,1).
$$

当 $S_i=1,B_i=0$ 时 $m_i\ge0$，故 $u_i^+\ge1$；其他情形新增事件为零。反过来，只有 $S_i=0,B_i=1$ 时 $\ell_i^-$ 才可能为正，且不超过 1。因此

$$
\mathbf 1\{B_i=0,S_i=1\}\le u_i^+,\qquad
\ell_i^-\le\mathbf 1\{B_i=1,S_i=0\}.
$$

对一个正常约束掩码 $M_0$，分母必须是该掩码中**全部正常行** $n_0$，不是只数基线阴性/阳性、有效 hinge 或 batch 行数。相同固定样本及决策下有

$$
\widehat{\mathrm{FPR}}_S\le U_0
=\widehat{\mathrm{FPR}}_B+
\frac1{n_0}\sum_{i\in M_0}(u_i^+-\ell_i^-).
$$

以 $U_0\le\alpha$ 作为绝对正常 FPR 经验代理约束。它比单独限制新增 FP 更直接地避免候选靠放宽 FP 预算胜出。仍需分别记录新增、修复和基线 FP；repair credit 不是“负的误报概率”。关闭信用时 $U_0=\widehat{\mathrm{FPR}}_B+\operatorname{mean}(u^+)$，是必要消融。若另要求 net FP 不超过锚，等价设置更严格上限 $\min(\alpha,\widehat{\mathrm{FPR}}_B)$，但不要悄悄将 absolute-budget 任务换成 net-budget 任务。

### 2.3 攻击类：只约束丢失的基线 TP

$$
v_i^-=\mathbf 1\{B_i=1\}\,[1-m_i/d_i]_+,
\qquad
\mathbf 1\{B_i=1,S_i=0\}\le v_i^-.
$$

对攻击族 $f$，建议定义

$$
U_{1,f}=\frac1{n_{1,f}}\sum_{i:y_i=1,\,f_i=f}v_i^-\le\epsilon_f.
$$

分母为该族全部攻击，故 $\epsilon_f$ 的单位是**该族 TPR 的绝对损失**；并有 $\widehat{\mathrm{TPR}}_{S,f}\ge\widehat{\mathrm{TPR}}_{B,f}-U_{1,f}$。若改除基线已检出的攻击数，则单位成为基线 TP 的相对保留率，二者不得混报。基线全漏的族约束自动为零，不代表该族已被保护；该族必须另报支持与实际召回。新增 TP 不抵消丢失 TP，以免用一种容易攻击补偿另一种困难攻击。

### 2.4 训练目标与教师附加信号

$$
\min_\theta\max_{0\le\lambda_j\le\lambda_{\max}}
L_{\mathrm{hard}}(\theta)+\beta L_T(\theta)
+\sum_j\lambda_j\,(U_j-c_j).
$$

这是有界原始—对偶启发式；深网非凸、hinge/clip 和随机采样不提供收敛保证。乘子在 primal loss 中 detach；dual 更新只使用 detach 的完整约束估计。失败时不能用乘子到上限或训练结束冒充约束已满足。最终需另检查代理违反量及真实错误计数。

教师默认仍是附加信号，不能与风险锚混用。建议先选**工作点中心化 margin 蒸馏**：教师攻击得分 $q$ 在合法训练范围产生，原得分阈值 $\tau_T$ 由训练内正常参考集确定。若用 logit 空间，须同时定义 $z_T=\operatorname{logit}(\operatorname{clip}(q,\varepsilon,1-\varepsilon))$ 和 $t_T=\operatorname{logit}(\operatorname{clip}(\tau_T,\varepsilon,1-\varepsilon))$，再按训练内冻结尺度迁移 $z_T-t_T$；不能用 logit 减概率阈值。也可迁移正常参考 CDF 中的位置；后者是 tail percentile，不是攻击概率。二者先择一冻结，学生/教师尺度、剪裁和损失系数显式记录，不在单个方法内部无账本搜索。

class-balanced 教师的 $q$ 不等于部署后验。理想加权 BCE 下，$\operatorname{logit}q=\operatorname{logit}p+\log(w_1/w_0)$，说明权重会平移阈值意义；有限树模型不保证严格满足此式。用同一 score 空间中心化可以避免盲信 0.5，但不能修复教师排序错误或外部漂移。teacher margin/目标必须 detach；转换、剪裁、尺度和有效教师覆盖都须落盘。用真实标签替代教师目标、保持同权重/同更新的控制不可省略。

## 3. $\kappa$、基线 slack 与不可行性

当 $m=a$ 且 $|a|\ge\kappa$，新增/丢失 hinge 都为零；弱边界 $|a|<\kappa$ 时可以为正。基线自己的非零 slack 是保守上界的代价，**不得减掉后仍声称保有同一界**。

建议 $\kappa=\eta s_B$：$s_B$ 是训练内冻结的 baseline logit 尺度，例如正常 logit 的 IQR；若尺度为零，明确失败或使用事先规定且记录的替代尺度，不静默兜底。$\eta$ 没有通用正确值；以小公共候选集合在开发阶段比较，并报告锚自身 slack、约束可行率、最大梯度/clip 率及最优值敏感性。

单位测试必须验证：同时将 $a,m,\kappa$ 乘任意正数 $c$，风险代理与实际事件不变；只缩放 logits 而固定 $\kappa$ 则不应声称不变。这是代理的尺度性质，不是整个 BCE 训练轨迹不变。

若 proxy 不可行但真实错误已达标，不能擅自扣 slack 或放宽预算。可在预先记录的共同开发预算内比较 fixed/adaptive 和 $\kappa$，或如实淘汰该代理。模型容量相同且输入冲突时，正常约束与攻击保留本来就可能没有可行解；应允许返回 hard 锚，并报告候选失败。

## 4. 正常组的意义与分母

对于正常 stratum $g$，以 $M_{0,g}=\{i:y_i=0,g_i=g\}$ 计算绝对上界；对于攻击 family $f$，用其全部攻击计算丢失上界。缺支持的约束应拒绝构建或明确 unsupported，不能用 0 填充。

**分组约束不等于分组采样。** 逐组风险 $U_{0,g}$、组均值和逐行风险是不同估计目标。实际逐行风险必须按 $n_{0,g}/N_0$ 加权；小组不能凭组均衡平均掩盖一个大组的大量 FP。多个 stratum 约束也不自动覆盖所有 feature-hash 簇，更不控制未见组。

当前 `group_id`、会话标识、特征相同簇、实验场景应按真实来源区分。相同特征哈希可用于避免重复泄漏和研究集中度，但不证明独立网络会话；不能把其数量直接当独立样本数。若公开输入缺真实会话/时间，结论限定为可观测关联簇的经验结果。真实 family 标签仅用于训练约束/审计，不进入部署输入。

## 5. 稀有尾部与 minibatch 支持

若某新增 FP 事件真实频率为 0.001，256 个独立正常样本的 batch 完全看不到该事件的概率约为 $0.999^{256}=0.774$。batch “零违反”十分常见，不能据此减小 dual 或认定满足预算；相关组使支持更弱。

建议 normal/family 约束的 dual 在**完整固定 constraint pool 一次扫描**后更新，至少先采用这个易审计版本。primal 可用有明确概率的分层采样，但须按目标分母还原权重。hard mining 用于训练可以，不能把其偏置均值当总体约束均值；固定数目的困难样本也不能丢掉未选中的风险支持。计数和风险评估不能只保留 top-k。

每次完整审计保存：正常/族/stratum 支持，基线阳阴性数、新增/修复 FP、丢失/新增 TP、代理、slack、违反量、抽样模式。聚合分子/分母，不平均大小不同 batch 的条件均值。空支持、单组支持和少量基线 TP 须显式失败/降级，不能仅由 `min_support=1` 推出科学支持充分。

## 6. 数据角色与阈值版本

| 角色 | 可做什么 | 不可声称什么 |
|---|---|---|
| fit / teacher fit | 拟合 hard 锚、教师和候选的监督部分 | 同一行上的 in-sample 教师误差不是 OOF 可靠性 |
| anchor reference（训练内） | 确定工作点、冻结 score 尺度及教师边界 | 它不是未来 final calibration；若也用于选模型，须记作开发 |
| constraint_train（训练内） | 计算约束梯度、dual、训练日志 | 不再是独立 audit，不提供未见风险保证 |
| inner holdout audit | 冻结候选后的一次性 sanity check | 反复看结果改方案后不再独立；历史旧 train 重划也不是新的确认来源 |
| development validation | 比较候选、预算/参数选择、早停，全部记账 | 重复使用不提供无偏确认 |
| 新来源 final calibration / test | 冻结方案后分工校准阈值与一次预定确认 | 旧 calibration/test 不重新取得盲态；test 不能反调阈值 |

所有拆分按可核组隔离，且写出每个 fit/阈值/尺度/教师目标的依赖。若 constraint pool 给候选额外标签，hard 对照也必须得到相同标签、更新与数据；“core-only hard 锚”与“继续 hard 训练的公平对照”分开报告。OOF 行身份、fold/组、阈值来源和 cache 均按身份绑定。

**任何后续阈值变化都会改变被约束的决策。** 训练期以 $t_S$ 建立的上界，不能自动转成 final calibration 新阈值上的证书。可以分别报告训练工作点/开发阈值/最终冻结阈值；在新的冻结决策上重新计算实际计数和同一界，不能复用旧证书。“相同 1% 校准预算”也不等于 test 实际 FPR 恰好相同；若实际误报不同，须照实报告，而非在 test 上调到同 FP。

## 7. 有限样本能支持到哪里

点态界不需要 IID，故同一有限池上的事件计数核验始终可做。把经验值推广到总体则额外需要固定候选/阈值、独立核验样本及与目标对应的抽样假设；优化过程中反复使用的 pool 不满足这些前提。本候选默认只主张经验上界。

作为支持量级算例：在独立 Bernoulli 且候选已冻结的理想条件下，$n$ 次观察零事件的一侧 95% 上界为 $1-0.05^{1/n}$。20 个攻击零丢失仍允许约 **13.91%** 丢失；2,000 个正常零新增仍允许约 **0.1497%** 新增风险；要把该上界压到 0.1%，至少约 2,995 个独立观察。多族/多预算同时判断还需计入多重选择。该算例不是本项目保证，也不能用相关流行数替代独立单位数。

真实组内相关时，可报告按组重采样的经验区间，但组数少/来源不明时不称精确风险保证。若改评估“一个组是否出现任何新增 FP”，其 Bernoulli 单位变成组，目标也变成组事件概率；不能把该上界写成逐流 FPR。

## 8. 可直接推翻错误实现/过强主张的合成反例

| 检查 | 构造与应有结果 | 能推翻什么 |
|---|---|---|
| 点态界网格 | 穷举正负/零 $a,m$ 与多种 $\kappa$；独立计算真新增/修复 FP、lost TP，检验上下界及有限梯度 | 符号、并列规则、广播/分母错误 |
| 弱边界 slack | $a=-0.01,m=a,\kappa=0.1$，无新增 FP，但 $u^+=0.9$；攻击 $a=0.01,m=a$ 时 $v^-=0.9$ | “锚必然可行”“slack 就是实际错误” |
| 禁止扣基线 slack | 上例将 $m$ 改成 0.001：真正新增 FP 为 1，$u(m)-u(a)=1.01-0.9=0.11<1$ | 减 baseline proxy 后仍称事件上界 |
| 修复信用保守 | 正常 $a=1,m=-0.01,\kappa=0.1$：真正修复 1，信用仅 0.1；必须保留残余上界 | 过度 repair 抵账、负风险 |
| 攻击不抵账 | 两攻击中 baseline 检出第一个、漏掉第二个；候选恰好对调，整体 TPR 不变但 lost-TP 风险为 1/2 | 用新增 TP 掩盖困难攻击丢失 |
| 组平均非流平均 | 999 行组全误报、1 行组无误报：逐流 99.9%，两组均值 50% | 把 group-balanced 目标称逐流预算 |
| 家族分母 | 一个族 100 攻击、仅 1 个 baseline TP 且候选丢掉：绝对族损失 1%，相对 baseline TP 损失 100% | 分母无声切换、虚假的家族保护 |
| 基线零覆盖族 | 某族 baseline/candidate 均全漏，lost-TP 约束为零，但召回是零 | “满足保留约束即检出该族” |
| 稀有尾部 batch | 构造 0.1% 风险行；随机 batch 大量为零，整池计数仍非零 | 从 minibatch 的零违反推出全池可行 |
| 阈值重新校准 | 正常 $z_B=-1,t_B=0,z_S=-0.2,t_S=0$，候选无 FP；事后 $t_S=-0.3$ 后变为 FP | 旧 margin 的证书自动适用新阈值 |
| teacher 单调重标度 | 例如 weighted-score logit 加常数，0.5 方向可变但正常 rank/相对同尺度边界不变 | 将 $q=0.5$ 当部署可靠性、把 score 当后验 |
| 容量不可分 | 正常与攻击输入完全相同，预算要求正常不报且攻击必检出 | 原始—对偶必定存在可行解/必定收敛 |

其中前三项和阈值反例是不可妥协的单测；它们直接检验算法声称的数学对象，不是只复写实现公式。

## 9. 必要消融与最强击穿实验

最小矩阵应区分四个来源：同数据更充分的 hard 训练、风险约束、教师信息、组处理。建议：hard continued、hard＋同风险约束、teacher-margin KD、teacher-margin KD＋正常约束、再加攻击保留、完整候选；配同一更新/标签预算、同容量/初始化/批次，并加入已有工作点方法或 pAUC 强对照。

对于完整候选至少再做：关闭 repair credit；移除 group 约束但保留 pooled 预算；teacher 目标换真实标签且保持权重；teacher 在合法范围内置乱；同预算单调 score/温度重标度；fixed/adaptive 同公共尺度搜索。组采样带来的数据再权重必须有无教师版本。

若“hard＋同约束”在相同误报预算下同样好，结论应是约束训练有效，而不是新知识迁移成功。若 teacher 置乱不影响收益，应撤销教师机制解释。若只有允许更多 FP 才改善召回，不构成同风险改进。若仅 constraint pool 满足而 holdout/新来源失效，说明经验拟合，没有可迁移风险证据。若 κ 轻微变化便翻转结论，应报告不稳定，不挑唯一成功值。

## 10. 建议冻结的 profile 与最小 API

配置至少记录：`algorithm_version`、anchor/model/schema/row IDs 哈希、`partition=constraint_train`、score/tie/threshold 版本、`majorizer=adaptive|fixed`、`margin_floor`、`normal_repair_credit`、每条 constraint 的 `name/kind/limit/group`、正常 stratum 和攻击 family 的来源、各自支持数、teacher score/工作点/尺度/目标版本、primal 与 dual 更新频率/步长/上限、失效与回退策略。

风险模块只接受已计算的 1-D margins、合法二元标签及整数 strata/family。reference 应 detach/复制 anchor、labels、metadata，sample IDs 必须完整同序匹配；拒绝 calibration/test partition、未知组、重复 ID、缺支持、非有限值和无效预算。只依赖 Tensor version counter 不能发现 `.data` 内容变更；每次完整报告重算冻结内容的 SHA 可以防止无声变更，但其 CPU 拷贝/同步/哈希时间必须计入约束训练成本。拒绝错误角色只能防误传，不证明文件真实来源。`min_support=1` 是工程下限，不是统计门槛。

`normal_fpr` 的返回 value 是绝对正常上界，`attack_loss` 是全攻击/指定全族的丢失上界；同时给 actual、baseline slack、违反量及名称，不做隐藏减 slack。dual state 必须绑定完整约束定义/顺序和预算；checkpoint 不能在同名改预算后默默恢复。部署预测不读 family/stratum/teacher，最终模型容量和 CPU 全流程另验。

本报告支持开展合成工程测试与预先冻结的开发试验；不支持跳过近邻对照、直接声称新算法成功，或把训练约束改名为独立安全保证。
