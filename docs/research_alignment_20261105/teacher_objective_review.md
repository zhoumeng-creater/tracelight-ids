# 教师知识对象、KD 与低误报目标审查

2026-11-05。状态：原文／作者实现调研与既有证据复核；没有训练，没有安装近期方法，没有读取旧 test 或新增 V 分数。本文只新增研究记录，不修改算法源码。

**当前没有足够依据把 native raw KL、温度搜索或新排序损失定为下一轮主方法。最小下一步是用已有 F/R/C 分数做一次分层的目标内容诊断：教师在哪些行提供了区别于硬标签的监督，这些行是否对应低误报检出的有效余量。UNSW 的普通 T=1 KD 可保留为有依据的机制对照；CIC 目前应先解释训练内目标近 hard 与 OOF 概率坐标失配。**

本结论对应[分阶段重构计划](../PROJECT_RECONSTRUCTION_20261104.md)的阶段 C 前置问题。阶段 A/B 的诊断已经有结果，不等于阶段 C/D 的蒸馏增益和独立确认已经完成。

本次随后已完成上述零拟合分层检查，见[教师内容实证复核](teacher_content_review.md)。UNSW目标非退化且有跨族纠错，普通KD保留为受控候选；CIC全字段F误差质量98.82%集中于两个冲突模式，仍无广泛新soft目标依据。UNSW三seed的OOF原生分数逐元素相同，不能算独立重复。当前下一轮实验室任务以[统一输入审计](../MINIMAL_VALIDATION_20261105.md)为准，本文件后续的目标内容检查清单保留为已执行依据。

## 1. 先从本项目的失败机制出发

证据来源是[教师独立复核](<E:/OneDrive - Coventry University/中传/毕业论文/工作/实验结果分析_20261105/teacher_review/teacher_review.md>)与[本轮分析第7–9节](<E:/OneDrive - Coventry University/中传/毕业论文/工作/实验结果分析_20261105/本轮实验详细分析_20261105.md>)。下表不使用历史 V 成绩来挑选新方案。

| 已确认的现象 | 支持的判断 | 不能据此推出 |
|---|---|---|
| CIC 实际 OOF 的旧概率裁剪饱和约82.47%；重新 sigmoid 后，native／legacy raw 概率最大差约1.0133×10⁻⁶ | 保存 native 有利于保留分数坐标和排序细节；T=1 raw KL 的逐行目标改变量却极小 | 不能说普通 KD 的知识被裁剪摧毁99%，也不能把 native 替换本身当作算法改进 |
| 旧目标为 sigmoid(裁剪后 logit−该折 R 阈值)；CIC 正常目标约99.63%接近0.5 | 已定位到裁剪与阈值中心化组合造成的目标退化 | 不能把它混同普通 raw KL，或归因为与目标设计无关的纯数值问题 |
| CIC 的139,618条 F 内 Portscan 全在 fold2 的 holdout；三个 seed 的该族 raw 概率中位数约0.001682、0.001659、0.028935，按各折自己的 R 阈值检出率均为100% | 教师确有低误报排序知识，但 raw sigmoid 的0.5边界不对应任务工作点 | 不能从教师检出率高推导 raw 概率就是合适的蒸馏目标；不能跨折混排绝对 margin |
| 新 screen 的 F 内教师 mean\|q−y\|：UNSW all／24为0.0428883／0.0603577；CIC all／24约3.07545×10⁻⁶／4.88071×10⁻⁶ | 两数据集的概率监督增量差异很大；CIC 的全体平均目标几乎是 hard | 全体均值小不等于每个困难族、正常尾部都无有效信号；也不能因此加大 λ 制造“KD作用” |
| CIC 24字段 MLP 的2,272个漏报集中于一个完整特征模式，同输入 HGB 能识别这些行 | 该模式在现有输入上可被某种学生函数类表示 | 单凭此表不能将漏报归因为缺少教师监督 |

三种对象必须分别命名：**概率** $q_T$、**低误报工作点上的决定** $\mathbf1[z_T>\tau_R]$、**教师选择的输入结构**。它们回答不同问题，不能只用一个“教师强”指标替代。

主分析随后新增的零拟合证据进一步定位了CIC这个模式：`Fwd Packets/s=2,000,000` 超过F最大值约1,090,909，按既有预处理后的z约50.51；F中有同19字段、时长13–190的16条攻击模板，均被识别。将全部24列统一夹到F内min/max、保持hard/wide权重不变，已报告两者都恢复2,272条；固定原阈值时hard增加74个FP、wide的FP不变。这个结果首先支持输入范围外推解释，后续应由两环境、全部既有MLP的统一审计复核代价，不能把它记成KD收益。本文件不另启动该审计。

## 2. 近期成果初筛：七项，四项继续核到作者实现

以下身份和链接均在本轮实时核验。预印本不写成正式录用；源码可读不写成已运行或已复现。

| 成果与一手来源 | 本轮读取深度及实际对象 | 对本项目的处理 |
|---|---|---|
| [TabDistill：Selecting Feature Interactions for GAMs，2026-04预印本](https://arxiv.org/html/2604.13332v1)；[作者仓库](https://github.com/Clouddelta/tab-distill) | §3公式／Algorithm 1，作者交互搜索与下游 EBM 实现 | 直接结构知识近邻，深读。不是 soft-label KD；不能把换成 IDS 或加尾部权重称为原创 |
| [Pocket Foundation Models，2026-05预印本](https://arxiv.org/html/2605.18654v1)；[作者 TabTune](https://github.com/Lexsi-Labs/TabTune) | §3目标、§4.4消融；交叉预测、MLP损失、树学生实现 | 与训练内目标近 hard 直接相关，深读；论文和当前代码存在实质差异，不能直接照搬整条管线 |
| [Distillation of Tabular Foundation Models into Efficient Predictors，2026-10预印本](https://arxiv.org/html/2610.01435v1) | §3 Eq.1–3、查询生成、附录协议 | 与 Pocket 的 OOF 主张不同；论文所链 [TFM_Distillation](https://github.com/nums-ai/TFM_Distillation) 在本轮网页和 GitHub API 均返回404，源码未取得，不列为可复现实装候选 |
| [TabDistill：Distilling Transformers into Neural Nets for Few-Shot Tabular Classification，2025-11预印本](https://arxiv.org/abs/2511.05704) | 作者全文§2：把 transformer 表示映射到 MLP 参数空间，用最终分类器的监督损失训练 | 与2026同名工作不同；是少样本超网络式生成学生，不能当作“普通树教师 soft KL 有效”的证据。本轮未取得作者代码 |
| [PLD，NeurIPS 2025主会](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6b2921f24f82dc05cda53bbe50bab90-Abstract-Conference.html)；[全文与附录C代码](https://arxiv.org/html/2506.12542v3) | §4损失、Appendix C 作者 PyTorch 实现 | 深读并做二分类代数化简。类别排序在二分类下退化，不作为本项目样本排序 KD 的直接依据 |
| [STACO，TMLR 2025](https://openreview.net/pdf?id=M3kibBFP4q)；[作者全文](https://arxiv.org/html/2505.21944v2)；[官方 LibAUC 教程](https://docs.libauc.org/examples/tpauc_staco.html) | Eq.1–3、Algorithm 1/2，`tpAUC_CVaR_loss` 与 `STACO` 源码 | 深读。是教师无关的直接低FPR监督近邻；适合限定预算后的对照，不是新KD知识对象 |
| [Closing the Approximation Gap of Partial AUC Optimization，TPAMI作者接收稿](https://arxiv.org/html/2512.01213v1) | 作者稿载明2025-11接受；§III–IV的OPAUC/TPAUC与instance-wise改革写法 | 新 UPAUCI 实现本轮未取得，不拿旧 PAUCI 前身冒名；用于区分低FPR区域优化、代理近似与单一工作点保证 |

### 2.1 两篇近期 TFM 蒸馏论文并不支持同一个默认配方

Pocket 主张用 OOF 防止 ICL 教师直接看到查询标签；2026-10工作则比较

$$
q_i^{\mathrm{OOF}}=q_T(x_i\mid D\setminus D_{f(i)}),\qquad
q_i^{\mathrm{FC}}=q_T(x_i\mid D)
$$

后报告 full-context、teacher-only 在所测组合中最好，并加入按字段重组的 CutMix 查询。它研究的是 TabICLv2／TabPFN v3 到 TabM／XGBoost 的任务，二分类主指标是整体 ROC-AUC；不能直接外推到本项目的 HGB、成组 IDS 输入和1%误报工作点。[2026-10原文§3](https://arxiv.org/html/2610.01435v1#S3)

本项目 F 内目标近 hard，与 ICL 上下文直接包含标签不是同一因果机制：HGB 是在 F 上拟合出的树，不是查询时读取包含该行标签的上下文。OOF 同样会改变拟合支持；CIC fold2没有见过 Portscan 就是实际例子。因此应先比较训练内与逐折目标内容、族支持和工作点，再选择教学来源，不能先宣布 OOF 或 full-F 普遍正确。也不能用新增合成查询把问题绕过去：流持续时间、包数、速率等字段可能有联合约束，逐列混合是否仍为合法流记录尚未核验。

## 3. 深读一：TabDistill 2026迁移的是结构

原文的学生形式是

$$
g(\mathbb E[y\mid x])=\beta+\sum_j f_j(x_j)+\sum_{I\in\mathcal I}f_I(x_I).
$$

教师经 SPEX 得到局部交互，按出现频次选择交互集合 $\mathcal I$，下游仍拟合原标签。它提供的思路是“让小模型接收教师发现的结构”，不是强制小模型模仿所有预测值。[原文§3与Algorithm 1](https://arxiv.org/html/2604.13332v1#S3)

作者仓库本轮仍为 `64214da0edf7eef6e8bf645332471d78b30345e8`：

- [`tabarena_single_mulindex.py:176–189`](https://github.com/Clouddelta/tab-distill/blob/64214da0edf7eef6e8bf645332471d78b30345e8/experiments/interaction_search/tabarena_single_mulindex.py#L176) 当前默认 TabPFN v3、查询预算1000、局部top_k=10、交互阶数2–4；与论文v1的TabPFN-2／预算500／最高3阶不同。
- [该文件:567–600](https://github.com/Clouddelta/tab-distill/blob/64214da0edf7eef6e8bf645332471d78b30345e8/experiments/interaction_search/tabarena_single_mulindex.py#L567) 遮蔽值来自训练数据的数值均值／类别众数；二分类值函数是裁剪到 $[10^{-10},1-10^{-10}]$ 后的概率 logit。
- [`compare_index_performance.py:154–225`](https://github.com/Clouddelta/tab-distill/blob/64214da0edf7eef6e8bf645332471d78b30345e8/experiments/TabDistill_downstream_comparison/compare_index_performance.py#L154) 按计数选交互，`ebm_spex.fit(X_train, y_train)`；不是用教师概率替代 `y_train`。
- [`pyproject.toml`](https://github.com/Clouddelta/tab-distill/blob/64214da0edf7eef6e8bf645332471d78b30345e8/pyproject.toml) 要求Python≥3.12、TabPFN8.2.0、torch2.9.1，并指定CUDA13.0源；公开代码不等于当前环境可直接运行。

**对应本项目：**它说明概率 KD 不是唯一知识对象；但本轮还没有证明该模式必须由教师才能发现。若未来研究结构迁移，最能推翻解释的控制是：相同交互数、相同学生和拟合预算下，标签驱动的 FAST／树分裂选择、随机交互能否得到同样收益。还必须核查遮蔽后的 IDS 行是否合法、教师查询成本，以及改变 score 坐标是否改变所选结构。当前不追加解释器或交互模块。

## 4. 深读二：Pocket／TabTune有可借鉴诊断，但不能整体照搬

论文§3.2采用硬标签项与温度 KD 混合，并以教师熵设定逐行温度和权重。它的§4.4自身也报告：五个低维二分类任务的 hard 消融优于完整 MLP 管线。这不是“多加温度和权重必然更好”的证据。[Pocket原文](https://arxiv.org/html/2605.18654v1#S3.SS2)

本轮锁定作者 `Lexsi-Labs/TabTune@5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e`，提交时间2026-11-04；仓库[MIT许可证](https://github.com/Lexsi-Labs/TabTune/blob/5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e/LICENSE)。当前代码与论文至少有以下差别：

| 已读位置 | 实际行为 | 对复现／迁移的影响 |
|---|---|---|
| [`distiller.py:380–481`](https://github.com/Lexsi-Labs/TabTune/blob/5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e/tabtune/distillation/distiller.py#L380) | 按 `inference` 策略触发交叉预测；使用逐行 `StratifiedKFold`，失败则普通 `KFold`，保存 `predict_proba` | 没有本项目的组隔离／攻击族支持约束；换成自己的成组折是协议适配 |
| [`mlp_student.py:440–472`](https://github.com/Lexsi-Labs/TabTune/blob/5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e/tabtune/distillation/students/mlp_student.py#L440) | 学生logit除以T；教师概率只clip和归一化，没有同温处理；硬项还有label smoothing | 不是直接按同温Hinton公式实现；不能不加说明地称标准温度KD |
| [`losses.py:68–105`](https://github.com/Lexsi-Labs/TabTune/blob/5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e/tabtune/distillation/losses.py#L68) | 默认基础T=3时，自适应范围为1.5–6；权重为归一化熵的二次函数加0.5 | 与论文的T∈[1,5]和高斯形权重不同；不能把当前源码当论文固定配置 |
| [`gbdt_student.py:75–111`](https://github.com/Lexsi-Labs/TabTune/blob/5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e/tabtune/distillation/students/gbdt_student.py#L75)；[`distiller.py:577–588`](https://github.com/Lexsi-Labs/TabTune/blob/5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e/tabtune/distillation/distiller.py#L577) | 每个类训练一个概率回归器；`y`用于标签映射，未传入α/T，也没有硬软混合目标 | 当前树分支不是论文混合KL，也不是对logit做MSE；概率MSE与KL本来就不严格等价 |
| [`distiller.py:535–546`](https://github.com/Lexsi-Labs/TabTune/blob/5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e/tabtune/distillation/distiller.py#L535) | 高层分发只接受mlp/lgbm，其他类型报错；底层类虽有xgb/catboost构造 | README的xgb/catboost示例不能据此视为当前入口已可运行 |

这里区分论文方案与当前公开实现，不用当前源码差异否定作者历史实验。对本项目最有价值的是它要求直接检查教学目标和学生类型，而不是其多组件默认值。

若只在学生侧变温，纯KD最优条件为 $\sigma(z_S/T)=\sigma(z_T)$，即 $z_S=Tz_T$；同温KD则要求 $z_S=z_T$。这项代数差异足以改变教师坐标的含义。将代码复制过来之前必须决定到底检验哪个目标；目前没有理由把它作为新增变量。

## 5. 深读三：PLD的类别排序在二分类下没有额外类序

PLD把真类放首位，再按教师排序其余类，目标为

$$
\mathcal L_{\mathrm{PLD}}=
\sum_{k=1}^{K}q^T_{\pi_k}
\left[-s_{\pi_k}+\log\sum_{j=k}^{K}e^{s_{\pi_j}}\right].
$$

二分类时 $\pi=(y,1-y)$，第二项精确为0，故

$$
\boxed{\mathcal L_{\mathrm{PLD}}=q^T_y\,\mathrm{CE}(y,p_S).}
$$

这是一项本次独立代数化简。作者[附录C代码](https://arxiv.org/html/2506.12542v3#A3)按升序排列非真类、真类放最后，再计算 `logcumsumexp−s_perm`，与上述反向表示一致；二分类没有可以额外传递的非目标类顺序。[原文§4.2及附录C](https://arxiv.org/html/2506.12542v3)

本项目中该目标相当于教师置信度加权 hard BCE，尤其可能降低 raw 概率很小的 Portscan 正样本权重。它不对应跨样本的低FPR排序，也没有解除概率坐标问题。若未来作为近邻适配，只需与相同权重的 BCE 做等价性检查，不应为两个等价表达各跑一轮并宣称排序机制胜出。本文没有实现PLD新模块。

## 6. 低FPR直接监督：STACO可作近邻，但它不是KD

STACO原文Eq.1在全部训练分数上选低分正样本和高分负样本，再最小化二者的单调pairwise surrogate；Eq.2–3把选择写为CVaR阈值及原始／对偶优化。正样本比例与负样本比例是两个独立参数，不能把两向指标偷换为当前单向1%指标。[作者原文§3](https://arxiv.org/html/2505.21944v2#S3)

已读作者 `Optimization-AI/LibAUC@3c02aa9877e0091114a00f94a8c6367698eb5637`：

- [`auc.py:731–853`](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/losses/auc.py#L731)：`tpAUC_CVaR_loss` 按正样本身份维护 `s1/dual_var`，默认squared-hinge；`forward`本身更新这些状态。只保存网络优化器还不足以恢复完整状态。
- [`staco.py`](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/optimizers/staco.py)：模型更新支持Adam/SGD及epoch正则；不是把任意pairwise项相加就获得原算法保证。
- 当前源码[第829–837行](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/losses/auc.py#L829)仅在 `len(index)==len(y_pred)` 时定义 `ids_p`，而[教程训练段](https://docs.libauc.org/examples/tpauc_staco.html#Training)传正样本索引子集。静态阅读可知该路径将使用未定义变量；整batch身份输入可以避开这一路径，仍需合成验证。当前未运行作者方法。

本项目 [`direct_pauc.py:59`](../../src/ids/direct_pauc.py#L59) 是每个macrostep重算全F正常上尾、对并列分数分配分数质量，再抽样正负对；[`pairwise_pauc_rows:139`](../../src/ids/direct_pauc.py#L139) 是logistic surrogate。它是明确命名的独立直接目标实现，不是STACO或UPAUCI复现。论文和LibAUC中CVaR平均、未经条件归一化的成对和、本项目条件尾部平均可能相差固定比例；一旦与BCE混合，比例会改变有效系数，不能因最优解的单目标尺度等价而忽略。

当前应保留已有直接目标作为教师内容的反证控制，先核其优化和选择预算。继续叠加一个STACO分支不能回答raw目标失配；优化整个低FPR区域也不自动保证冻结阈值下的FPR和召回。

## 7. 本项目必须先完成的代数与坐标检查

令 $q=\sigma(z_T)$、$p=\sigma(z_S)$，硬标签为 $y$。当温度为1、hard与KD使用相同行权重时：

$$
\mathrm{BCE}(y,p)+\lambda\mathrm{KL}(q\Vert p)
=(1+\lambda)\mathrm{BCE}\!\left(\frac{y+\lambda q}{1+\lambda},p\right)-\lambda H(q).
$$

归一化后目标 $r=(y+\lambda q)/(1+\lambda)$ 相对hard的logit梯度变化为

$$
\Delta g=(p-r)-(p-y)=\frac{\lambda}{1+\lambda}(y-q).
$$

因此这项诊断无需拟合学生，甚至不需要学生分数；不能把 soft-target BCE 与 T=1 hard+KL包装为两个创新。项目 [`native_trees.mixed_targets`](../../src/ids/native_trees.py#L50) 已实现归一化目标，hard/soft行权重由相同硬标签产生；[`budget_transfer._backward`](../../src/ids/budget_transfer.py#L197) 的NN分支则分别累加hard和KD。未来跨实现比较必须同时记录总损失尺度、权重衰减、梯度裁剪和训练路径；目标梯度等价不保证有限步boosting或Adam轨迹完全相同。

本次另用70组标量组合核对两个恒等式：PLD二分类损失的最大绝对差为 $1.04\times10^{-16}$，归一化BCE+KL梯度与软目标BCE梯度的最大绝对差为 $1.67\times10^{-16}$。这里只计算标量表达式，没有拟合模型。

以既有λ=0.3仅作代数说明，CIC all／24的全体平均 $|r-y|$ 约为 $7.10\times10^{-7}$／$1.13\times10^{-6}$。这不是选择λ的建议；它说明不能仅靠一次“soft训练”标签宣称已经加入大量新知识。

另有两个不同的坐标问题：

1. **温度缩放不能移动二分类零边界。** 对任何 $T>0$，$\sigma(z/T)>0.5$ 当且仅当 $z>0$；它可以缓和置信度，却不会把Portscan负margin变成正margin。按R阈值中心化确实移动边界，但得到的是相对工作点的响应，不能未经校准便称攻击后验概率。
2. **类别权重与概率含义不能混用。** 理想加权BCE的最优输出满足 $\operatorname{logit}q_w=\operatorname{logit}\eta+\log(w_1/w_0)$。这是条件推导，不是对本轮极端HGB margin的已证实因果解释；有限树、缺族折和分布变化都可能另有影响。不能把raw sigmoid天然当作原始总体的校准后验。

## 8. 下一步：一项零拟合目标内容诊断

**不新增模型、不搜索温度／λ、不读新V。** 输入只用已经保存的F-heldout／F-in-sample、各教师R／C分数、标签与折身份；严格区分旧OOF、新full-F、全字段与24字段。输出一张可逐行追溯的汇总表，而不是新的资格guard。

| 检查 | 应输出的量 | 它要排除的错误解释 |
|---|---|---|
| 目标是否只重复hard | 每教师／折／标签／攻击族的 $q$ 分位数、熵、$|q-y|$ 分位数及总和；固定λ下 $\Delta g$ 的绝对质量 | 全体均值掩盖少量困难行；通过增大λ把数值变化冒充知识 |
| 概率和工作点是否一致 | 同行的 $\mathbf1[q>0.5]$ 与 $\mathbf1[z_T>\tau_R]$ 交叉表；按各折自己的R阈值计算 | 用很强的低FPR排序替概率校准背书；跨折混排margin |
| 教师变化落在哪里 | 正常R尾部对应区域、各攻击族、已保存的学生错误／教师纠错与反向损失集合上的 $|q-y|$；记录计数和分母 | KD变化只发生在已易学样本，或主要推动学生接受教师错误 |
| 实例对应关系是否有内容 | 在每个合法OOF折和硬标签内固定置乱q，另用该层均值代替q；比较目标差异分布与关键集合的对应关系 | 只有类别平滑／重加权效果，却称迁移了实例知识；置乱跨折造成坐标混淆 |
| 内容是否被一个重复模式主导 | 行加权与特征模式等权的诊断统计并列，并列出主要模式的样本数／族／会话数 | 把2,272条同模式当2,272个独立知识点；把模式hash当独立会话 |

均值替代和置乱只是**零拟合对象检查**，不产生学生效果结论。若现有F学生分数不存在，先输出与学生无关的前两项及目标质量表，不为补表触发训练；后续是否查询已封存学生F分数单独记录成本。任何校准映射若要拟合，已超出本项“零拟合”，需要在后续协议中固定数据角色，不能悄悄使用C标签寻找最有利目标。

**继续条件：**能够指出同一学生可见输入上的具体教师增量，给出支持数、目标改变量与反向损失，而不是只给teacher总体召回；没有折身份或角色混用；普通目标未在核心集合上退化为hard或系统性相反监督。

**停止条件：**若有效区域的目标仍近hard，或主要知识只存在于工作点排序而raw目标持续与其相反，就停止该raw-KL解释的扩展，不以扫温度、扩大λ、追加mask／guard替代知识对象论证。若内容只集中在一个既有模式，优先处理学生对该模式的直接可学习性。不能按seed剔除不利教师后只汇报成功条件。

## 9. 有依据的最小KD候选及能推翻它的对照

**目前仅保留普通T=1软监督作为条件式基线，优先回答UNSW教师纠错是否能被一个已选部署学生吸收。** 它已有清晰目标与实现，本轮证据也没有显示UNSW存在CIC式裁剪／中心化退化；但尚不足以冻结架构、字段数、λ和拟合预算，更不足以命名新算法。CIC暂不由“换native”直接进入相同矩阵。

通过上一节后，只选择一个学生／字段预算／教师身份，另行冻结相同F标签、初始化或随机规则、行权重、更新量、查询量、C选择机会和主比较：

- hard、普通KD、折内同标签固定置乱KD：真实KD若不优于置乱，撤回实例知识解释；类条件均值／均匀平滑作为事先指定的平滑控制，不能看到结果后再挑。
- 学生字段少于教师时，保留同信息教师控制：若只有补回字段才改善，不能称KD已解决不可观测信息；若同信息教师已足够，也不把全信息教师当必要条件。
- 神经学生保留直接pAUC及pAUC+普通KD：若普通KD不超过同预算hard／直接目标，停止此组合的自定义损失扩展。不能用更长续训或更多C检查点解释为知识增量。

树学生的普通soft可用已有CatBoost适配器，NN学生可用已有raw KL；不将更换学生、预处理、训练长度和教学目标同时进行。只有出现可重复的相对控制增量，才进入预定seed稳定性和真正未使用来源的独立确认。
