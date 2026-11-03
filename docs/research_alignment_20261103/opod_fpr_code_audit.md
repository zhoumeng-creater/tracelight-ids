# OPOD、交叉组 FPR 蒸馏与 C-GUARD：V3 原文／源码参照审计

核查日期：2026-11-03。只读论文、作者仓库与本项目代码；本轮没有安装依赖、导入外部训练框架、运行模型或修改算法。范围限于 OPOD、ECAI 2025 Intersectional KD，以及同名消歧后的恶意软件 C-GUARD；PCT／MPT 由另一份审查负责。

**结论：V3 不能把“单侧项＋分组预算＋对偶更新”或“表格 KD＋组 FPR”作为首创。** OPOD 已有前者；ECAI 2025 论文已有后者。V3 当前真正变化的是所约束的事件、连续包络、参考模型角色及固定完整风险池。它的 KD 本身仍是工作点 soft-target BCE；不对称性主要在对 hard anchor 的风险约束中，不能直接写成“复现 OPOD 的一侧蒸馏”。

本文标签：**原文事实**、**源码事实**、**审查推断**、**建议对照**。看到源码不等于代码正确或已运行；作者报告的效果不等于本项目复现。未取得实现不证明实现不存在。

## 1. 固定版本与可复核入口

| 对象 | 本次锁定版本 | 状态与证据边界 |
|---|---|---|
| OPOD 论文 | *OPOD: On-Policy Omni Distillation*；arXiv:2607.20918v3，2026-08-04；首发 2026-07-23 | 2026 预印本；未核实正式主会录用身份。[原文 §3](https://arxiv.org/html/2607.20918v3)、[版本页](https://arxiv.org/abs/2607.20918) |
| OPOD 作者代码 | `VincentZhao2002/OPOD@7d50ff93ff5cca5294c4069e1301d880b53e1674`，2026-08-28 | [固定 commit](https://github.com/VincentZhao2002/OPOD/commit/7d50ff93ff5cca5294c4069e1301d880b53e1674)。`opd_loss_patch.py` blob=`6e8b339622265e124da271390676f3f782313098`；不是把 blob SHA 当 commit |
| OPOD 宿主 | 作者集成说明指定 Relax `cbb1a82` | [固定集成说明](https://github.com/VincentZhao2002/OPOD/blob/7d50ff93ff5cca5294c4069e1301d880b53e1674/training/integration/README.md#L8-L20)。本轮访问宿主短 SHA 得到仓库迁移重定向，未完成完整宿主 SHA／环境锁定，不称可直接复现 |
| Intersectional KD 论文 | Sikder、Ramachandranpillai、de Leng、Heintz，*Promoting Intersectional Fairness Through Knowledge Distillation*；ECAI 2025，3427–3434；DOI `10.3233/FAIA251214` | 正式会议论文，不是 FairX 的 ECAI 2024 workshop 工具论文。[作者发表页](https://fahim-sikder.github.io/publications/promoting-intersectional-fainress/)、[全文](https://fahim-sikder.github.io/uploads/intersectional-KD.pdf) |
| 作者 Code 链接所指 FairX | `fahim-sikder/FairX@417c3d9f7d29776429950a0f45af8093100de175`，2025-10-29 | [固定 commit](https://github.com/fahim-sikder/FairX/commit/417c3d9f7d29776429950a0f45af8093100de175)；作者发表页的 Code 链接确实指向该仓。完整递归树已查，`truncated=false`；不意味着该论文完整训练代码已发布 |
| C-GUARD | Imran、Al-Essa、Franchini、Pirlo，*C-GUARD: Context-Adaptive Conformal Gating for Improving Robustness Against Evasive Windows PE Malware*；Future Internet 18(8):431，2026-08-13 | 正式期刊，非顶会。[出版原文](https://www.mdpi.com/1999-5903/18/8/431)。不是 arXiv:2608.00180 的 Constitution-Grid C-Guard，也不是其他 LLM guard |
| 本项目 V3 | 审查时 HEAD=`9cf05876c8fac1b26fc5a00230517b4d955ffbd2`；`risk_distillation.py` 实际内容 SHA256=`19cf30ff7868b238be60e62ab9c4224e87f2dd8b43d84614c4314a146e66881b` | HEAD 不等于未提交源码已包含于该 commit；以内容 SHA 和实际函数共同界定本次映射，不拿新文献改写旧实验结论 |

## 2. OPOD：原始目标与作者代码并不是同一层对象

### 2.1 原文目标

**原文事实：**§3.3 对学生生成 token 定义教师相对 log probability 缺口的正部；每条 rollout 先对 token 平均。§3.4 在每模态样本均值上设 warm-up 后冻结的预算，使用截断对偶系数。总目标还包含 RL 与 reference KL，不能删掉它们后称完整 OPOD。[原文式 (1)–(5)](https://arxiv.org/html/2607.20918v3)

为比较估计对象，将原文这部分记为：

$$
c_i=\frac1{T_i}\sum_t[\log\pi_T(y_{it})-\log\pi_S(y_{it})]_+,
\qquad
\overline c_m^{paper}=\frac1{|\mathcal B_m|}\sum_{i\in\mathcal B_m}c_i.
$$

**审查推断：**“教师给当前 token 更高概率”不是“教师预测真正确”，也不直接上界正常误报或丢失原 TP。此处缺口并非二分类报警工作点 margin；不能把 token log-ratio 换成 IDS logit 后继续沿用相同风险语义。

### 2.2 固定源码实际路径

作者文件是插入宿主框架的 patch excerpt，文件头明确不是独立模块。以下均依据[固定版本 `opd_loss_patch.py`](https://github.com/VincentZhao2002/OPOD/blob/7d50ff93ff5cca5294c4069e1301d880b53e1674/training/method/opd_loss_patch.py)，没有运行该文件。

| 定位 | 真实行为 | 对 V3 的含义 |
|---|---|---|
| L294–324 | 先算 `rk=student_logp-teacher_logp`；rectified 时控制量为 `relu(-rk)`，再传入模态 dual controller；另有非默认 ExOPD 分支 | 必须锁定 rectified、extrapolation 与 reward-gating 配置，不能只写“采用 OPOD loss” |
| L333–351 | rectified 时用 `min(rk,0)`，以 `adv - coef*penalty` 修改 RL advantages | 是 advantage injection 路径；不能未经推导就声称等价于监督分类中直接反传一个 hinge 标量 |
| L96–108、L171–181 | 按模态累计所有 token 的 sum／count，经 data-parallel all-reduce 后更新 beta；无 token 的模态保持 beta | 风险总体与分母须明确，不能搬一个 mean 而忽略权重／分布式聚合 |
| L87–94 | 预留 4 个模态代码；缺少 modality 时退化成共享桶 | 代码槽位包含 video 不等于论文验证了 video；本项目不能把任意 hash group 当模态／环境语义 |
| L110–123 | beta 保存在 actor 进程的 args；默认 beta 下限 .1、上限 1.5 | 下限非零意味着预算满足后仍保留教师拉力；不同于 V3 的非负乘子允许回到 0 |
| L129–169 | warm-up 将 beta 暂置下限，累计每步的模态平均，再冻结 epsilon、重置 beta | 此预算是数据驱动的教师缺口尺度，不是使用者指定的 1% FPR 合约 |
| L323–337 | dual 系数覆盖另一种 adaptive-KL 系数；correctness floor 还可再缩放正确样本系数 | 开关相互作用会改变实际目标；不能把所有可选分支都当正式默认方法 |

**重要估计对象差异，审查推断：**上述 dual 代码使用

$$
\overline c_m^{code}
=\frac{\sum_{i\in\mathcal B_m}\sum_t c_{it}}{\sum_{i\in\mathcal B_m}T_i},
$$

不是原文逐 rollout 等权的 $\overline c_m^{paper}$。例如两条同模态响应，长度分别为 1 和 100，token 缺口分别恒为 1 和 0，则原文均值为 $1/2$，代码控制量为 $1/101$。这是按原文与固定源码的直接代入，未执行实验。两种目标在长度相等时一致，变长时不一致；不能把代码的 token-pooled 控制解释为已证明论文 rollout 均值达标。

warm-up 又将**各步的上述均值等权平均**，而非对整个 warm-up 的所有 token 重新汇总；[固定 `opd_process.py` L228–250](https://github.com/VincentZhao2002/OPOD/blob/7d50ff93ff5cca5294c4069e1301d880b53e1674/training/method/opd_process.py#L228-L250) 明确其输入是 `sum_of_per_step_mean_kl, n_steps_seen`。引用论文、引用作者实现、实现自己的估计对象，需要分别说明。

**开关陷阱，源码事实：**[32-GPU recipe L290–311](https://github.com/VincentZhao2002/OPOD/blob/7d50ff93ff5cca5294c4069e1301d880b53e1674/training/recipes/run-qwen3-30B-A3B-opod-32xgpu.sh#L290-L311) 用 shell 环境变量“非空”检测开关，故把 `OPD_RECTIFIED` 或 `OPD_DUAL_KL` 设为字符串 `0` 仍会加启用 flag。复现消融需 unset／空值并核最终 argv；不能只记录配置表中的 0／1。

本次只核到 excerpt 的进程内 beta 状态；完整宿主是否将 beta、warm-up 状态与 epsilon 一起 checkpoint，尚未确认。不能据此宣布作者续跑失效，也不能未经核查默认其完整恢复。

### 2.3 可复用部分、输入与许可

可复用的是：各约束独立系数、sum／count 聚合、缺席组不更新、目标尺度／预算冻结、warm-up 与正式控制状态分离、记录系数饱和。它们是可借鉴的成熟工程机制，引用来源即可，不作为新发明。**不应为方便而把正式 FPR 预算自动改成初始训练风险的 warm-up 均值。**

原实现需要对齐的学生／教师 token log probabilities、rollout advantages、模态代码、分布式进程组，以及宿主 Relax／Megatron／SGLang；不是输入一张 IDS 特征表即可 `fit` 的分类器。作者 README 说明训练使用多卡 CUDA（论文 32×H20），方法 excerpt 也依赖宿主类型与全局导入。[固定集成输入 schema](https://github.com/VincentZhao2002/OPOD/blob/7d50ff93ff5cca5294c4069e1301d880b53e1674/training/integration/README.md#L65-L102)

代码／文档为 [Apache-2.0](https://github.com/VincentZhao2002/OPOD/blob/7d50ff93ff5cca5294c4069e1301d880b53e1674/LICENSE)；模型权重遵循各自基础模型许可，训练数据不被此代码许可覆盖。[固定 README 许可说明](https://github.com/VincentZhao2002/OPOD/blob/7d50ff93ff5cca5294c4069e1301d880b53e1674/README.md#L297-L301)。本轮未复制作者源码进入算法模块。

### 2.4 V3 究竟改变什么，怎样击穿

| 维度 | OPOD 原文／作者实现 | 当前 V3 |
|---|---|---|
| 被约束量 | 路由教师在生成 token 上的单侧概率缺口 | 真标签条件下的正常报警上界、hard anchor 原 TP 丢失上界 |
| 参考角色 | 专家教师、起始 reference policy | 强教师 T 供 soft target；同容量 hard B 定义风险；数据角色 R 只冻结工作点，三者不能混称 reference |
| 不对称位置 | teacher-student token guidance | KD 为 soft-target BCE；不对称位于 B→S 风险项，正常有 repair credit、攻击无 rescue credit |
| 聚合 | 原文 rollout 等权；固定控制代码 token 等权 | 完整 C 中指定 class／family 的全部成员，分母绑定且不会只对被筛中样本平均 |
| 乘子 | warm-up epsilon，beta floor 默认 .1；当步控制路径 | 明确风险预算，乘子可为 0；完整 C 后重算，再更新供后续参数步使用 |
| 推理 | 丢弃教师，仅学生 | 同样仅学生；此部署形态已有，不能当新贡献 |

**最强击穿对照，建议而非本轮执行：**先比较相同 V3 风险代理的固定系数 penalty 与 dual 更新，保持 F∪C hard 监督、teacher、起点与更新预算一致。如果调好的固定 penalty 同样有效，则不把 dual 作为关键贡献。再做一个清楚标为 **OPOD-inspired adaptation** 的单侧 log-probability 缺口控制到 B、保留相同 T 的 soft supervision，在同实际 FPR 下比较；其约束单位是 log probability，不能与 V3 的事件质量预算直接填相同数值后宣称公平。使用真类概率、学生采样类概率或全二元分布期望都会改变原目标，应预先固定其中一种并解释，不能称原论文逐项复现。

## 3. ECAI 2025 Intersectional KD：有直接目标先验，当前公开项目不足以证明完整实现

### 3.1 原始目标与数学可执行性

**原文事实：**§4.3 的学生目标组合分类、归一化表征蒸馏、预测 KL、FPR／DP／条件独立性与对抗项；系数通过 Hyperopt 选择，并非本文所见的显式风险预算 dual 更新。式 (5) 按真非该类、指定交叉组计算 hard argmax FPR；式 (6) 组合组间方差、最大差与加权绝对幅度。[原文式 (5)–(14)](https://fahim-sikder.github.io/uploads/intersectional-KD.pdf)

其 FPR 部分可简记为：

$$
L_{FPR}=\sum_c\left[\operatorname{Var}_g(FPR_{g,c})
+\max_{g,g'}|FPR_{g,c}-FPR_{g',c}|+\sum_gw_{g,c}FPR_{g,c}^2\right].
$$

**审查推断：**若严格按式 (5) 的 hard argmax indicator 对神经网络参数求普通梯度，该部分几乎处处梯度为零；主文未交代可直接照搬的平滑替代或梯度估计器。没有对应训练源码时，不能判定作者实际程序如何处理，更不能把该未解问题照抄进 V3。组 parity 可以在各组风险同样很高时成立；幅度 penalty 也不是设定 $FPR\le0.01$ 的硬上限或总体证明。

这篇是 **2025 正式 ECAI 方法论文**。其作者 Code 页面指向 FairX；FairX 本身是另一篇 **ECAI 2024 co-located workshop** 工具论文，不能把两种身份互换，也不根据出版平台晚出现的 2026 页面日期改写方法年份。

### 3.2 固定作者仓的实际发布状态

| 核查点 | 固定代码／文档事实 | 允许得出的结论 |
|---|---|---|
| [README L29–31](https://github.com/fahim-sikder/FairX/blob/417c3d9f7d29776429950a0f45af8093100de175/README.md#L29-L31) | 方法已接收，未来会发布并合入 FairX | 不是声明该固定版本已经提供完整方法 |
| [README L51–58](https://github.com/fahim-sikder/FairX/blob/417c3d9f7d29776429950a0f45af8093100de175/README.md#L51-L58) | Intersectional KD 仍为未勾选项 | 项目计划条目不能当运行证据 |
| [inprocessing 导出](https://github.com/fahim-sikder/FairX/blob/417c3d9f7d29776429950a0f45af8093100de175/src/fairx/models/inprocessing/__init__.py) | 仅导出 TabFairGAN、Decaf、FairDisco；递归树中未核到对应 KD 模型／损失／训练入口 | 在本次固定默认版本中没有得到可审完整 KD 实现；不排除其他分支、补充工件或作者尚未发布代码 |
| [FairnessUtils](https://github.com/fahim-sikder/FairX/blob/417c3d9f7d29776429950a0f45af8093100de175/src/fairx/metrics/fairness/main_fairness.py) | `evaluate_fairness` 接受 train/test tuple，另训 XGBoost，对其预测调用 Fairlearn DPR／EOR | 这是后续公平性评估工具，不是该论文学生 FPR loss，也不会自动评价传入的已冻结 V3 学生 |
| [正文参考 [27]](https://fahim-sikder.github.io/uploads/intersectional-KD.pdf) | 另列 Zenodo DOI `10.5281/zenodo.15866767` | 本轮 Zenodo 页面与 API 返回访问失败／403，未取得文件列表；不能说补充材料没有代码 |

因此，当前能称 **“论文目标已核、官方 Code 指向的固定仓状态已核、完整方法代码未取得”**；不能称“作者已完整开源并已复现”，也不能反过来断言作者没有任何实现。

### 3.3 可复用部分、许可与输入限制

可直接借鉴：交叉组支持表、每组 FPR、组差与绝对风险同时报告，以及区分效用／公平指标。不可直接借用：未明确可微路径的 hard FPR loss、未经任务核准的敏感属性移除或默认重新训练一个 XGBoost 代替评价目标学生。

该方法训练需要合法可用的敏感属性／交叉组标签、真标签、教师输出及可对齐的表征；主文数据包括 Adult、COMPAS、German Credit 与 MIMIC。网络流的 duplicate／session hash 是隔离单位，不能自动替代其人口交叉组定义；缺少真实环境标签时也不能假设有 group-mixture 保证。若只研究 pooled FPR，则不应堆一套敏感属性对抗模块制造算法深度。

FairX 代码为 [MIT](https://github.com/fahim-sikder/FairX/blob/417c3d9f7d29776429950a0f45af8093100de175/LICENSE)；方法论文 PDF 标为 CC BY-NC 4.0，作者网站页脚另有网站内容许可，不能混作代码许可。未知补充工件需单独核许可；原始数据的访问条件也不由 FairX 的 MIT 覆盖。本轮未下载数据、未安装其 Python 3.8 旧依赖栈。

### 3.4 V3 的具体变化与最强击穿对照

V3 没有将公平性方法重命名为安全方法；它去掉了 DP／敏感信息对抗目标，改为指定正常事件的绝对预算和原 TP 丢失预算，提供固定阈值下的逐点经验上界。它也没有完成对 ECAI 原始全目标的复现，所以论文应写“对直接先验的比较与可微风险适配”，而非宣称已完整复现后超越。

**建议对照：**在相同固定教师、学生容量、F∪C 标签和完整风险统计下，比较简单 pooled-FPR hinge penalty／明确组的 worst-group penalty 与 V3；有真实可解释环境组时，再做组 FPR variance＋max-gap＋magnitude 的干净实现。若给式 (5) 换平滑 surrogate，名称写 **Intersectional-KD-inspired FPR penalty adaptation**，公开差异，不伪称作者原始代码。其余六项损失只有在输入和研究目标都适用时才考虑，不能为形式完整硬加。若普通幅度 penalty 已解释收益，V3 不能把“加入误报训练”当独立创新。

## 4. C-GUARD：可比的是操作点与非退化规则，尚不能审源码实现

### 4.1 已核原文机制与本轮访问边界

**原文事实：**正式论文描述三折 OOF 构造 baseline-negative 的 TN-vs-FN rescue 数据；辅助模型分数先定义 context，再作验证集选择的 conformal gate；仅将 baseline-negative 翻为 malware，不覆盖原 positive。作者另做 validation FP-matched 的单阈值放宽对照。[原文 §3–4 与附录](https://www.mdpi.com/1999-5903/18/8/431)

本轮原文直连多次返回 429／403，可从出版源检索缓存核对上述公开方法。前轮报告记录实验工件需申请；本轮没有重新取得可固定 commit 的作者实现，亦未核到可执行公开仓。因此此节**不填写假想“源码真实行为”**，不声称完成复现或证明不存在公开代码。若取得作者工件，应先核 baseline／gate 阈值、OOF 归属、验证集复用、组合 FPR 计数及许可，再决定接入。

### 4.2 V3 与它的真正差别

**审查推断：**C-GUARD 的单向 override 在同一固定 baseline 上从结构上保留 baseline-positive；V3 是重新训练单个学生，只约束原 TP 丢失的经验上界，因此仍可在预算内丢失旧 TP。两者不等价；不能把 V3 的软约束写成 C-GUARD 那样逐样本不覆盖保证。

C-GUARD 推理需要 baseline、辅助模型与 gate。V3 计划只部署学生，教师与 B 只在训练使用，但尚需实测 CPU 成本。辅助分数先参与 context 判定，意味着“少翻转”不直接等于“少计算辅助模型”；它也不是 V3 已实现的 feature-acquisition 方法。其 conformal compatibility 在 malware/FN 侧构造，不直接提供正常总体 FPR≤1% 的 NP 合约。

**可复用部分：**冻结阈值、给出新增 FP 的明确代价、使用 OOF 错误而非训练内高置信错误作 rescue 构造、保留更简单单阈值方案作反证。原文的 validation 预算不自动保证 test 风险；不能仅借用 conformal 一词建立 V3 的统计主张。

**最强击穿对照：**若将来采用 rescue／级联，必须有同开发 FPR 的单阈值放宽、同调用比例随机升级、简单置信升级，并报告端到端 CPU 成本。若只靠放宽阈值就有同等回收，则没有必要增加辅助系统。当前单学生 V3 主线不应为凑近期基线，提前同时引入另一条级联研究。

**许可与输入：**这是 Windows PE／EMBER2024 的静态特征检测，不是网络流 schema；不能把文件特征原样移入 UNSW／CIC。未取得可用代码 LICENSE／工件，不假定论文开放可读就授权一套未取得实现；对作者提供材料及原数据许可需另核。本轮没有申请、下载或运行相关工件。

## 5. 从参照到当前代码的逐项映射

以下是审查时实际源码，不是未来计划已经完成的声明；行号与 §1 内容 SHA 配套。

| V3 入口 | 可追溯参照 | 当前实际改动／仍需验证 |
|---|---|---|
| `working_margins`、`freeze_margin_threshold`（L52、L71） | OPOD 明确 log-probability 信号单位；C-GUARD 明确冻结 baseline 工作点 | V3 采用 raw-logit float64 次序统计与 nextafter，绑定实际 floor-count。是数值合约，不是新统计理论；未知总体及 sigmoid 导出等价仍未验收 |
| `build_risk_reference`（L162） | OPOD 的独立桶与 sum／count 提示；ECAI 的组条件分母 | V3 冻结完整 C、身份、成员、预算、anchor margins 与签名；C 是训练开发池，不是独立证书 |
| `constraint_risks`（L237） | OPOD 单侧缺口与 ECAI FPR 目标均为相关先验 | 改成正常非负事件包络、保守 repair credit、attack no-credit；必须与固定 hinge、无信用和成熟负翻转机制比较 |
| `risk_lagrangian`、`ProjectedDual`（L320、L337） | OPOD 及一般 constrained learning 已有系数控制 | V3 对原始 violation 乘 detached 非负权重、显式保存状态／参考签名。求解器本身不作创新；上限饱和不代表可行 |
| `boundary_kd_objective`（L423） | 通用 KD／工作点蒸馏；不是 OPOD advantage injection 的等价实现 | 固定 hard 项＋相对工作点的 soft alarm BCE；正式 student hard 要覆盖相同 F∪C，不能只把 F kernel 的返回值当完整监督 |
| 训练角色与验证 | FairX 评估会另训 downstream XGBoost，OPOD 的 rollout 权重和 C-GUARD 的 gate 选择各有协议 | 不能直接调用外部评价工具改掉目标模型，或把多加 C 标签的收益归于风险机制；验证、校准与确认角色继续分开 |

[本项目源码](</E:/OneDrive - Coventry University/中传/毕业论文/代码/tracelight-ids/src/ids/risk_distillation.py>)、[V3 设计与数据角色](</E:/OneDrive - Coventry University/中传/毕业论文/代码/tracelight-ids/docs/ALGORITHM_V3_DESIGN_20261103.md>)。本轮没有根据上述参照修改任何函数。

## 6. 下一次修改前应形成的最小参照包

这不是额外实验矩阵，而是将用户“先参考成果再动手”的要求具体落到当前设计：

1. 每个拟改组件先填：原论文公式／假设、固定作者代码路径／commit、源码与论文差异、输入／许可限制，以及当前是否真的跑通过。
2. 将 V3 新规则写成与参照的最小差分；普通 dual、one-sided ReLU、KD、独立桶不记为新增贡献。所谓“未找到”同时写明查了哪些版本和未取得哪些工件。
3. 为该差分预设最强击穿条件：同 proxy 的 fixed penalty 若不差，停止宣称 dual 必要；固定 hinge 若不差，删除 adaptive 复杂性；无 KD 若不差，不解释为知识迁移；简单 FP-matched 阈值若足够，不加 rescue 系统。
4. 真实训练前统一 F∪C 标签／forward／教师查询／选轮预算，再冻结条件。不能让候选更多监督、对照更少计算，之后用“算法更深”解释收益。
5. 作者原始实现、干净适配版和本项目候选分开命名；未经环境与输入验收的仓库只能标“已读源码”，不能标“已复现”。

**本次落点：保留 V3 为有明确先验、可实现、可被击穿的风险代理研究；不把 OPOD 或 ECAI 机制重新命名成创新。** 是否值得正式实验，仍由最简单强对照和数据支持决定。

固定源码版本回执：OPOD `7d50ff93ff5cca5294c4069e1301d880b53e1674`；FairX `417c3d9f7d29776429950a0f45af8093100de175`。**本轮上述外部项目均未运行、均未复现；C-GUARD 未取得可固定版本实现。** 本文件只提供原文／源码参照与待验证差分，不是实验结果。
