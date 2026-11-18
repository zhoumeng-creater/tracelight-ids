# 低误报约束训练：2025/2026 原文与作者实现核查

核查日期：2026-11-18。本文区分作者已实现的机制、作者实证范围与本项目需要新增的设计。只下载并静态阅读原文和源码；没有加载模型、计算真实数据新分数、拟合、选阈值或执行作者训练。

**结论：下一版可借 Cooper 的“真实约束更新乘子、可微代理更新模型”实现；不能把 STACO 或 tapAUC 直接换进来，宣称解决了保检出与正常尾部问题。** 三项资料均未证明：只用现有 F 数据训练，就能在未见正常分布上改善 1% 误报工作点，并保留原模型检出的每个正例。它们也不能证明本轮有限轨迹没有找到可行点就等于约束不可满足。

| 主对照 | 核实年份与作者代码 | 实际对象 | 本项目决定 |
|---|---|---|---|
| Cooper | 2025-04-01 预印本；作者库 `30b158e0262f7073c7710bd1c4996e462af4ff35`，提交于 2025-06-09 | 非凸约束优化的原始/对偶更新；代理与真违例分开 | 借鉴约束状态、身份乘子和更新隔离；不照搬示例学习率、不宣称可行性保证 |
| STACO | 2025 论文，TMLR；核对库版本 `3c02aa9877e0091114a00f94a8c6367698eb5637` 提交于 2026-08-21，非论文发表当日快照 | two-way pAUC 的嵌套 CVaR 排序目标 | 保留为排序机制对照；不作为本轮的直接替换方案 |
| tapAUC | 2025-02-17 首版、2025-04-01 修订；作者库 `24c2dde9248348e73094a156a64b323a6b41203d`，2025-04-02 | 选取部分正常样本的配对平方 hinge，加训练侧零漏报阈值 | 不采用原脚本；选样与阈值实现存在需先修复的不一致，且没有逐身份保护 |

## 1. Cooper：能借的实现与必须补齐的项目语义

[Cooper 原文](https://arxiv.org/html/2504.01212v1)明确给出原始/对偶一阶更新、非可微约束的代理机制，并说明一般 min-max 问题的 GDA 可能发散。作者示例是范数受约束逻辑回归，没有本项目的跨域 IDS、1% FPR、32 成员聚合或保留原 TP 身份实验。这里采用的是实现机制，效果仍需由项目数据验证。

作者源码与本地快照均固定在上述 SHA；以下位置是原始文件行号：

| 原始路径和位置 | 已核实行为 | 对本项目的含义 |
|---|---|---|
| `src/cooper/constraints/constraint_state.py:15–49,60–79` | `violation` 给可微约束；`strict_violation` 给对偶使用的真违例。后者省略时回退到前者 | 必须显式传入真实 FP/TP 违例，不能只填平滑损失就称已在约束真实工作点 |
| `src/cooper/formulations/formulations.py:108–127,165–195` | primal 路径使用 `violation`，dual 路径换成 `strict_violation` | 一次状态记录两类量，防止代理下降被当成真约束改善 |
| `src/cooper/formulations/utils.py:49–88` | primal 为 `sum(detach(lambda) * proxy)`；dual 为 `sum(lambda * detach(true))` | 两套梯度隔离；这里是求和，不是平均，不能把现有平均损失的权重或示例学习率原样搬入 |
| `src/cooper/optim/constrained_optimizers/simultaneous_optimizer.py:56–76` | 清梯度、计算状态、两类 Lagrangian 分别 backward，然后更新原始与对偶变量 | 一个模型和乘子同步更新；不是固定 `lambda=1` 的三个损失相加 |
| `src/cooper/optim/constrained_optimizers/constrained_optimizer.py:59–95` | 对偶优化器要求 `maximize=True`；更新后执行乘子投影 | 真违例驱动约束压力；不等式乘子保持非负 |
| `src/cooper/multipliers/multipliers.py:66–95,111–119,137–170` | 默认乘子为零；`IndexedMultiplier` 可按身份索引取对应乘子 | 可为正例身份建状态表；零初始化并不保证首步不破坏正例，未观察的身份也不自动得到本步保护 |

可直接复查[ConstraintState](https://github.com/cooper-org/cooper/blob/30b158e0262f7073c7710bd1c4996e462af4ff35/src/cooper/constraints/constraint_state.py#L15)、[梯度隔离](https://github.com/cooper-org/cooper/blob/30b158e0262f7073c7710bd1c4996e462af4ff35/src/cooper/formulations/utils.py#L49)、[实际更新](https://github.com/cooper-org/cooper/blob/30b158e0262f7073c7710bd1c4996e462af4ff35/src/cooper/optim/constrained_optimizers/simultaneous_optimizer.py#L45)。MIT 许可证与原始字节已保存。

该版本实际 `SimultaneousOptimizer.roll` 更新一套模型参数，返回本步状态；没有要求把历史模型随机混合后部署。由此只能说**作者代码不强制随机混合**，不能引用其他随机分类器的理论，替一个确定的非凸 TabM checkpoint 提供可行性保证。选轮仍要用完整、真实的约束值；代理损失最低、乘子稳定或最后一步都不是自动合格条件。

### 对应本项目的对象

以下是项目需要定义的对象，不是 Cooper 已完成的 IDS 方法。部署分数应沿用

$$
s_\theta(x)=\operatorname{logit}\left(\frac1{32}\sum_{k=1}^{32}\sigma(z_{\theta,k}(x))\right).
$$

计算路径仍须沿用项目稳定的 FP64 实现。对每个成员独立加约束、约束平均 logit，均不等于约束这个部署分数。

令 $Q$ 为**训练权限内**预先冻结的正常参考集，$t_Q(\theta)$ 为在当前模型分数上按同一有限样本校准规则得到的阈值，告警判定沿用严格 $s>t$。正常组的真实风险可以写为

$$
R_g(\theta)=\frac1{|N_g|}\sum_{x\in N_g}\mathbf1[s_\theta(x)>t_Q(\theta)].
$$

组预算若为预先冻结的 $b_g$，对应真违例是 $R_g-b_g$，而不是到某张旧分数表的 MSE。预算不能事后按结果选定；也不能未经可行性审查就令每组都必须低于 1%。既有正例身份 $i$ 的判决保护则检查

$$
I_i(\theta)=\mathbf1[s_\theta(x_i)\le t_Q(\theta)],
$$

保护哪些身份必须在训练权限内冻结。训练不能从 C/E 的既有命中身份反推保护名单，也不能读 C/E 来更新 $t_Q$、乘子或选轮。

如果用 $I_i\le0$ 作为逐身份约束，其真违例只有 0/1，满足时不会产生让乘子下降的负值；如果改用有符号 margin，必须重新明确尺度、严格不等号和 ties。两者不是同一优化问题，不能由默认实现替我们作决定。

下一版要闭合的实现条件只有以下几项：

1. **同一阈值函数。** 正常风险和正例约束必须使用相同 $t_Q(\theta)$；不能正常端用分位数，正例端只保原始绝对分数。$Q$ 的来源、与正常风险样本是否分离、秩序统计量、取整及 ties 规则必须写死。最终 C 只在冻结后作原协议校准。
2. **代理梯度。** Cooper 不提供经验分位数的求导或平滑化。本项目须明确代理是否通过 $t_Q(\theta)$ 回传，以及 ties 处的处理。直接 detach 阈值会遗漏它随模型移动的影响；若采用该近似必须标明，不能等同于完整校准风险的梯度。
3. **真实违例与采样。** 为每个组及固定正例身份记录代理、真违例、乘子。身份采样只更新观察到的部分，全量检查仍必需。求和/平均与抽样权重应匹配冻结的目标，不能依靠调大统一罚项代替。
4. **判决不变性与选择。** 对全部分数作共同严格单调变换后，秩序统计校准阈值同步变换，$R_g$ 和 $I_i$ 应保持不变。平滑代理可能受尺度影响，必须同时记录真违例。最终候选靠真约束与已冻结的选择规则接受，不能靠 loss 下降或后验删除守卫接受。

这能纠正“优化绝对分数几何、验收重标后的判决”的目标错位，却不补充训练数据没有包含的正常机制。训练侧同一校准函数与最终 C 校准属于同一程序规则，数值阈值和分布并不因此相同；未见域迁移仍需要单独的数据与假设依据。

## 2. STACO：严格分离并不自动意味着平方 hinge 无梯度

[2025 原文](https://arxiv.org/html/2505.21944v2)优化 two-way partial AUC，通过正负样本尾部的嵌套 CVaR 表达限制 TPR/FPR 的关注区间。它提供该优化问题的算法及理论，不是“现有分类器每个已检出身份不得退化”的约束学习。

[作者损失源码](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/losses/auc.py#L731)的 `tpAUC_CVaR_loss` 默认使用平方 hinge，`threshold=0.5` 实际保存为 `self.margin`；维护逐正例 `s1/u/dual_var` 和公共 `s2`。`auc.py:829–851` 先以 `loss > s1` 选门控，再用 `dual_var` 加权模型损失。因此能否产生模型梯度取决于 margin、门控和对偶权重。

令 $d_{ij}=s_i^+-s_j^-$。作者默认代理为

$$
\ell_m(d_{ij})=[m-d_{ij}]_+^2.
$$

F 严格分离只说明 $d_{ij}>0$；只有所有有权重的配对均满足 $d_{ij}\ge m$，或相关门控/权重全部为零时，该项才没有模型梯度。**不能由 AUC=1 推出 STACO 默认损失在本项目上已饱和。** 本轮未执行真实梯度计算，不能据此断言它一定有或一定没有梯度。即使仍有梯度，学习目标也仍是所选正负配对的间隔；它没有提供每个正常子群在重标阈值下的风险约束。

作者示例使用原始模型输出（`last_activation=None`），训练 `theta_0=theta_1=0.5`，报告 `max_fpr=0.3/min_tpr=0.7` 的 tpAUC。它每轮评估 train/test，没有本项目逐身份可行性选轮。原始文件位置：`examples/18_Optimizing_tpAUC_Loss_with_STACO_on_Imbalanced_data.ipynb` 的代码提取件第 107、131、135–158、179–193 行。

**已核到一个具体接口不一致：**同一固定版本示例第 158 行传 `index[:num_pos]`，但损失 `auc.py:829–830` 只在 `len(index)==len(y_pred)` 时定义 `ids_p`，随后无条件使用。正负混合 batch 下，按示例传正例长度的 index 会走未定义路径。这是当前固定库与示例接口的问题，不是否定论文算法；它说明不能只引用库名就当作可运行且适配的实现。

决定：保留 STACO 对关注 ROC 区间的启发及版本核查记录；不把它当作正常子群风险和 TP 身份约束的替代。若未来单独使用，其评分尺度、状态索引、优化器和选轮契约都要重新接线，不能只增加一个 pair loss。

## 3. tapAUC：训练零漏报目标没有转成目标域逐身份保证

[2025 原文](https://arxiv.org/html/2502.11570v2)提出动态选取部分正常样本、训练配对排序代理，并保留训练侧无漏报阈值。摘要报告 6 个数据集平均 TPR 92.52%、FPR 20.43%；这不是 1% FPR 下保留原模型每个 TP 的实证。

作者仓库固定 SHA 中只包含 README 与 notebook，完整树没有 LICENSE。源码只作核查快照，不作为已许可的本项目依赖引入。[作者 notebook](https://github.com/ArnaudBougaham/tapAUC/blob/24c2dde9248348e73094a156a64b323a6b41203d/tapAUC.ipynb)的可读代码提取件 `tapAUC.code.txt` 已保存，关键位置如下：

| 行号 | 实际行为 | 对结论的影响 |
|---|---|---|
| 87–98 | `relu(negative + gamma - positive)**2`，模型输出为 sigmoid 概率 | 严格分离仍不必达到 gamma；“配对饱和”须核实分数间隔，不能从 AUC 推断 |
| 162、166–172 | 按分数降序排序，随后取正常索引列表的末端；`neg_count==1` 时取最后一个 | 实际选的是低分正常端，与原文描述关注高分困难正常样本不一致 |
| 100–118、186–187 | 无给定阈值时，ZFN 用当前传入正例最低分；调用处是训练 batch 循环结束后的 `y_pred_train/y_batch_train` | 该阈值来自最后一个训练 batch；不是完整训练集的零漏报阈值，判定还采用 `>=`，与本项目严格 `>` 不同 |
| 190–206 | 每轮读测试 fold 指标，最终保存固定最后 epoch 的报告值 | 没有逐身份安全选轮，也没有替本项目隔离 C/E 的训练协议 |

决定：不采用原脚本，不推荐只换成 tapAUC loss。它没有补上共同校准函数下的组误报约束、逐身份保护或跨域正常尾部信息；作者报告本身也没有达到“测试零漏报”。上述代码问题与适用目标的差异应分别处理，不能用修脚本推导项目任务就能成功。

## 4. 两项边界对照

**SACSIM，2026。** [期刊原文](https://doi.org/10.1093/biomtc/ujag138)与 [PMC 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13455594/)对应 *Shift-adjusted Neyman–Pearson classifiers via single index modeling*，期刊页面出版日期 2026-08-10。作者随刊 R 包 `sacsim 0.1.0`、许可与脚本已按原始字节保存；`func_wts.R` 的 `getWv.pstr/getWy.npindex` 根据源/目标协变量、目标类比例及单指标假设构造权重，`rule_tools.R` 的 `getRule` 用加权风险求阈值。`proc_simu.R` 将 `mean(y.vld)` 传入目标比例。它需要目标域信息及分布假设，不是 F-only 深网训练修复，也没有现有 TP 身份约束；用于说明迁移依赖什么信息，不采用其阈值调整作为本项目下一步。

**FairPOT，2025。** [AIES 原文](https://ojs.aaai.org/index.php/AIES/article/view/36660)明确是分数组后处理，以部分 OT 变换和参数曲线权衡 AUC/xAUC；它不提供本项目所需训练约束，故排除。未核到可固定的作者代码，不据此否定其是否存在。

## 5. 资料与执行记录

`author_sources_low_fpr/manifest.json` 记录 33 个文件的 SHA256、字节数、固定提交或随刊来源；所有条目已逐一重算校验。Notebook 的 `.code.txt` 是按原顺序提取代码 cell 的可读副本，原 `.ipynb` 字节保留，没有执行。Cooper/STACO 的 MIT、SACSIM 包 GPL 与文章许可证信息已记录；tapAUC 的无 LICENSE 判断有固定完整仓库树存档。

本文件给出源码支持的实现选择与不能外推的边界。完整新训练设计应在项目主稿中完成统一的数据权限、校准函数、约束代理、对偶更新与选轮定义，不能把本文的对象公式直接当作已冻结运行合同。
