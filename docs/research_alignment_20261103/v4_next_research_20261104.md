# v4回传后的研究收敛：先查阈值可行域与重复拒绝，再补pAUC

核查日：2026-11-04。只读新回传 `UPGRADE_V4_UNSW.zip` 与随后到达的 `UPGRADE_V4_CICIDS.zip` 的JSON、既有协议、原论文和作者代码；未读取prepared特征、未加载joblib、未重训。本文件不代替root的全包SHA／预测复算。

## 1. 本轮已经回答什么

新包是full配置：3seed、8条件、每条件60步。seed17在共同hard Phase-I第11步找到C可行点，seed29从step0可行；seed43在30步后仍未找到可行点，八条件全部记 `no_feasible_start`。前两seed16条件均得到合格开发候选，不能把这16个条件化结果写成三seed全部成功。

| seed | 条件 | V检出／17,902 | V误报／8,400 | 相对hard新增检出 |
|---|---|---:|---:|---:|
| 17 | hard | 15,075 | 82 | — |
| 17 | raw_kd | 15,076 | 83 | +1，即+0.005586个百分点 |
| 17 | balanced_raw_kd | 15,081 | 83 | +6，即+0.033516个百分点 |
| 29 | hard | 15,143 | 73 | — |
| 29 | raw_kd | 15,172 | 76 | +29，即+0.161993个百分点 |
| 29 | balanced_raw_kd | 15,161 | 78 | +18，即+0.100547个百分点 |

这些是在相同R参考预算下选择的开发结果，不是V实际FPR完全相等的比较。数个新增检出不能独立证明KD新颖或稳健胜出，尤其seed43被共同起点挡住。legacy EKD在两seed均只选step0；guarded条件相对各自unguarded版本召回更低，不能先把更新门禁当成有益升级。

CIC并未启动本轮八条件训练：`n_students=0`、`n_skipped_no_feasible_start=24`。三个seed在共同Phase-I的31个记录状态（含step0）中，失败项均只有normal，家族实际保留预算均未超限。

| CIC seed | step0 C误报 | step30 C误报／187,861 | 允许误报 | step30冻结旧阈值下全部C实际预算 |
|---|---:|---:|---:|---|
| 17 | 1,952 | 1,982（1.055035%） | 1,878 | 未全部通过 |
| 29 | 1,987 | 1,930（1.027355%） | 1,878 | 全部通过 |
| 43 | 1,988 | 1,937（1.031081%） | 1,878 | 全部通过 |

这支持检查所选R工作点与C预算的失配。对CIC29/43及UNSW43，还可从相同模型的阈值单调性直接证明存在满足R+C预算的阈值，见下一节。CIC不存在“v4 raw KD已改善／未改善”的新实测结论，因为三seed所有该条件都未训练。

最需要修的两个问题是：①R的一个极端分位点是否错误地将“该分数模型无可行阈值”与“所选R阈值在C不合格”混为一谈；②完整回滚后是否进入相同提议的重复拒绝。它们应先于更大的KD矩阵。

## 2. 固定分数下，阈值可行域可以精确求出

本节是对有限池事件的基本次序统计推导，不是新NP定理。所有预测先转float64，判定规则固定为 $z\ge t$；仅有C训练标签和R参考标签参与，V不能决定 $t$。

对正常池 $S\in\{R_0,C_0\}$，令 $N_S=|S|$、$A_S=\lfloor\alpha N_S\rfloor$。其分数降序排列为 $z^{S}_{[1]}\ge\cdots\ge z^{S}_{[N_S]}$。当 $A_S<N_S$ 时，允许至多 $A_S$ 个误报的充要条件是

$$
t>z^S_{[A_S+1]}.
$$

浮点实现取 $\ell_S=\operatorname{nextafter}_{64}(z^S_{[A_S+1]},+\infty)$，联合下界 $\ell=\max(\ell_R,\ell_C)$。相等分数必须整组拒绝，不能把阈值舍回float32。

对某个攻击集合 $G$，记分母为全部攻击行 $N_G$，原锚点TP集合为 $B_G$，允许丢失数 $L_G=\lfloor\delta N_G\rfloor$。将**候选模型在原锚点TP行上的分数**升序排列为 $b^G_{(1)}\le\cdots\le b^G_{(|B_G|)}$。丢失事件是 $z<t$，所以当 $L_G<|B_G|$ 时，其充要条件为

$$
t\le u_G=b^G_{(L_G+1)}.
$$

等号保留该TP，不能再向下nextafter。若 $L_G\ge|B_G|$，该集合不产生上界；若锚点TP为空同样无上界，须明确无原TP可保护。集合包括攻击总池和每个已启用保护族，分母不能换成锚点TP数，新增TP也不能抵扣旧TP损失。

令 $u=\min_G u_G$。因此，有限R/C上的全部实际事件约束有解，当且仅当 $\ell\le u$。非空时最小阈值 $t=\ell$ 在这些约束内获得最多报警／检出；空区间应返回决定下界的正常池和决定上界的攻击族。阈值有限性和超出可表示范围也须显式处理。

必要测试：正常ties、TP ties、$L=0$、无锚点TP、$L\ge|B_G|$、两下界由不同池主导、空区间、$\ell=u$、原始阈值nextafter及小支持族整数预算。每次应在完整分数上回算事件计数，不能只信排序索引。

**已有非空区间证据：**下表原阈值取自v3 hard的run.json，并逐一核对该文件SHA与新v4包内 `v3-evidence/RISK_RESEARCH_COMPLETE.json` 清单一致。均为Phase-I末步同一个模型的两种阈值评价。

| 环境／seed | 原锚点阈值 $t_{old}$ | 当前模型R阈值 $t_R$ | 原阈值下C正常误报／允许 | 原阈值下C全部保留约束 |
|---|---:|---:|---:|---|
| UNSW43 | 1.5085414648 | 1.4698911905 | 77/78 | 通过 |
| CIC29 | -2.7168436050 | -2.9631576538 | 1,540/1,878 | 通过 |
| CIC43 | -4.3014707565 | -4.7451453209 | 1,394/1,878 | 通过 |

因为 $t_{old}>t_R$，同一R分数上的报警集合满足 $\{z\ge t_{old}\}\subseteq\{z\ge t_R\}$。$t_R$已满足R预算，故 $t_{old}$也满足；加上日志显示原阈值下C全部实际预算通过，三例已经存在联合可行阈值，不能解释为这些模型无可行工作点。此前“R精确计数未知所以无法证明R可行”的判断遗漏了这一单调性，现明确纠正。

现有记录尚不能给出精确最小 $\ell$、全部 $u_G$、R在原阈值下的精确计数；这些仍需原始R/C分数审计。失败Phase-I参数没有保存，还需在实验室保存／按state hash核验重放。联合可行阈值存在也不能追认旧v4准入：旧协议固定选最小R阈值，且没有可部署的该失败模型工件。

一旦用C确定上述区间并选 $t$，C就是阈值训练信息；这必须作为新训练协议，不能继续称R-only校准，更不能借统计校准文献的独立性保证。旧v4结果保持原判据。

## 3. 完整回滚可能形成确定性的吸收态

当前full-population梯度、无dropout、固定F/C顺序；一个guard步骤全部拒绝后，参数、AdamW moments／step和RNG均恢复。下一宏步若控制器、学习率和回溯分数仍不变，梯度及完整提议也相同，从而重复相同拒绝。增加宏步数不改变这个状态。

优先记录提议前state hash、提议delta hash、失败约束／计数、最小尝试fraction、真实移动范数；确认重复后终止，记录 `stalled_repeated_rejection` 和首次发生步。正确回滚应保留，不能靠残留被拒动量打破重复。

后续若研究如何继续移动，需另立控制：拒绝后缩小下一次trust-region／lr，或加入针对活动约束的修正方向；控制器自身状态必须保存。缩小参数到float32无位移只能记no-op，不能算有效更新。与“原地终止并保留当前模型”对照后，才判断额外复杂度是否值得。

## 4. 本次实时复核的一手来源与实际差异

| 来源 | 本次核实层级与固定工件 | 对下一步的实际约束 |
|---|---|---|
| [Learn then Test，AOAS2025正式刊载](https://doi.org/10.1214/24-AOAS1998)，[公开全文§1.1–2](https://arxiv.org/html/2110.01052v5) | 本次重新访问出版社／原文。机制始于2021，不能称2025新发明 | 阈值候选与多风险验收已有；原文以独立i.i.d. calibration和有效p值/FWER为基础。自适应训练C的精确计数区间没有其总体风险保证 |
| [双类条件漂移NP，ICLR2026正式记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33870b3e099880cd8e705cd07173ac27-Abstract-Conference.html)，[作者全文v1](https://arxiv.org/html/2511.06641v1) | 正式身份本次已由会议页确认，方法精读仍为公开v1；未取得对应作者实现 | 约束与目标分布都变化时，源池达标不等于新流量达标。本项目的有限池区间不是该迁移算法复现，也不借其凸性等假设 |
| [Feasible Learning，AISTATS2025](https://proceedings.mlr.press/v258/ramirez25a.html)，[原文§2–3](https://arxiv.org/html/2501.14912v1) | 作者代码此前固定 `juan43ramirez/feasible-learning@864a585ff685cdbb8e091983c98026b2ec38c39f`，本次再核README／原文 | 可参考集中优化违反项的Phase-I；引入slack会改变问题，不能用slack可行冒充原1%实际事件可行 |
| [PI=ALM，ICLR2026作者全文v2](https://arxiv.org/html/2509.22500v2)，[固定trainer](https://github.com/juan43ramirez/pi-control-is-alm/blob/edb340719b739aa130df221d6c9690349d4c6fa8/src/trainer.py) | 本次读§3–4及对应初始化／SGD、dual-first代码 | 不等式局部稳定点对应不等于任意AdamW每步可行；不用于给当前guard吸收态套收敛结论 |
| [Closing the Approximation Gap of Partial AUC，作者全文](https://arxiv.org/html/2512.01213v1)，[作者正式卷期列表](https://qianqianxu010.github.io/publications/) | 原文2025-11接收，作者页列TPAMI48(3):3482–3498，2026。该条只有pdf链接；本次未核到UPAUCI对应作者代码，出版社DOI直开失败 | PAUCI与UPAUCI不同；UPAUCI增加实例选择变量以消除近似偏差。原实验主要β=.3/.5、sigmoid分数、ResNet18；不能把其成功直接外推到本项目β=.01及MLP |
| [Pairwise AUC Optimization Needs Corrective Power，2026-10全文](https://arxiv.org/html/2610.21743v1)，[作者代码](https://github.com/JiaShen-tech/pairwise-auc-corrective-power/tree/3e0aba756b4b208086b9287d0016a7eabc1df9d7) | 新近邻；作者／arXiv comments称NeurIPS2026 accepted，本次未独立取得正式proceedings。已读§2–4及 `losses.py`、`diagnostics.py`，MIT，固定SHA `3e0aba756b4b208086b9287d0016a7eabc1df9d7` | full-AUC pairwise更新需要困难误排序的实际纠正力，不能仅换loss。其SGD／U-statistic条件不是本项目非pairwise KD或guard的实证归因 |

最后一项作者代码在binary路径先sigmoid，使用 $m=s^+-s^-$，squared hinge为 $[\gamma-m]_+^2$，梯度权重为 $2[\gamma-m]_+$。它是下一轮pairwise强控制的实现依据，不是把raw logit margin和bounded sigmoid margin混用的许可。困难pair梯度、当前工作点边缘和被门禁阻断的更新必须分开测。

## 5. pAUC仍是应补的强对照，但先完成诊断闭环

低FPR区间排序改善与满足某个R/C阈值预算不同。先用现有已保存V分数报告raw OPAUC[0,.01]及除以区间宽度的归一值；不要把sklearn McClish标准化值与前两种混称。只做描述性复算，不在旧V反复调参寻找漂亮结果。

可执行成熟实现优先[LibAUC官方CVaR代码](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/losses/auc.py#L344-L444)和[SOPA](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/optimizers/sopa.py)。这是不可回避的2022基础算法、2026固定工件，不伪称2026新论文。目标为

$$
\frac1{n_+}\sum_i\left[s_i+\frac1{\beta n_-}\sum_j(\ell(f_i^+-f_j^-)-s_i)_+\right],\quad\beta=.01.
$$

本次核代码：`forward`维护每正例索引的 `u_pos`，先更新辅助阈值再用detach尾部mask反传；返回标量不是完整目标值。`u_pos`不是注册buffer，必须显式保存；拒绝更新时也要回滚它。完整pair矩阵复杂度为 $O(B_+B_-)$，需预算matched的正负采样与持久索引；batch只有32正常时取一个hardest等于3.125%尾部，不能叫精确1%尾部。

先不把pAUC、KD、guard三者一起堆入当前救火轮。诊断闭环后最小矩阵可为hard、raw KD、hard+pAUC、KD+pAUC，全部相同学生／标签／起点／调参预算，使用同一预声明阈值策略；guard作为另一个单独因素，在确认重复拒绝已被处理后再加。

## 6. 下一轮的最小可执行顺序与退出规则

| 顺序 | 实际动作 | 可推翻的解释／退出 |
|---|---|---|
| P0 工件诊断 | 保存失败Phase-I模型及R/C同序分数；对旧可用模型和Phase-I末状态求精确阈值区间；按旧state hash验证重放 | 若区间空，报告哪族与正常尾部冲突，不称仅阈值可修；若区间非空，优先修阈值策略，停止不必要重训 |
| P0 停滞诊断 | 验证重复提议／拒绝后提前停止，保留当前模型；报告节约步数和成本 | 若所有后续提议不同，撤回确定性吸收态归因；不以接受次数替代有益移动 |
| P1 限量干预 | 同一state比较R-only与明确的R+C有限池阈值；若仍需训练，比较活动违反项Phase-I与相同计算预算hard | C参与阈值就承认训练协议改变，不给独立证书；相同hard已解释收益则撤回新机制解释 |
| P2 强基线 | 冻结上述诊断后，实施合法LibAUC/SOPA适配及pAUC控制，记录分数尺度、sampling、状态、成本 | 没有优于同预算hard／pAUC，不宣称KD算法创新；无新来源确认，只保留开发证据 |

不放宽1%预算、不换seed、不拆CIC巨组。共同可行起点失败须列为结果；CIC本轮未启动新条件，不能对它们排序。未来独立确认仍需要真正未使用来源，原calibration/test不参与新模型选择。
