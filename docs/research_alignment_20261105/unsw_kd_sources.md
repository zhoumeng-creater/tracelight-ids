# UNSW 全字段 small 普通 KD：论文、作者实现与最小控制依据

核查日期：2026-11-05。本文服务于下一轮协议制定；没有训练、加载实验模型、查询新分数或读取 V/test。项目事实来自本目录已有 `teacher_objective_review.md`、`teacher_content_review.md`；教师与 full-small 的最新逐行比较以本轮工件审查为准。联网核查覆盖表格蒸馏、正则化解释、教师偏差、二分类退化、验证协议和分布变化；仅将直接改变本轮设计的内容展开。

## 1. 对最小方案的结论

建议固定现有全字段 full-F HGB 教师、原 small `[32,16]` 学生和原输入处理；只比较 hard、普通 KD、模式分层固定置乱 KD、二分类标签内均值 KD。置乱按模式的 Normal / Attack 行数分层，精确保留两类目标分布。温度固定 $T=1$，旧配置的蒸馏相对权重固定 $\lambda=0.3$，不搜索教师、温度、权重或新架构。

这四臂回答一个问题：**教师置信度与训练样本的对应关系，是否能让同预算 small 学生获得超过硬标签、类内平均软化和随机置信度扰动的低 FPR 收益？** 论文支持提出问题与选择控制；实际答案必须来自本项目的配对实验。

本轮没有新算法主张。普通 KL、混合标签、类内置乱和类内均值属于已有蒸馏与受控分析工具。新任务、重新命名或增加验收规则不构成方法新颖性。

## 2. 本轮实时核查的来源及采用决定

| 来源与当前身份 | 读到的证据 / 作者工件状态 | 对本轮的具体影响 |
|---|---|---|
| [Pocket Foundation Models，2026-05 预印本](https://arxiv.org/html/2605.18654v1) | 深读 §3、§4.4；作者 TabTune 代码取得并固定提交。论文强调 ICL OOF；5 数据集 MLP 消融 hard AUC .863，完整管线 .829 | 必须保留同学生 hard；不把多个软目标组件的组合当成收益来源；ICL 的结论不直接决定 HGB 教师 |
| [Distillation of Tabular Foundation Models into Efficient Predictors，2026-10 预印本](https://arxiv.org/html/2610.01435v1) | 深读 §3.1 式 1–2：在其四种 TFM–学生组合上 full-context teacher-only 最好；论文链接的作者仓库 `nums-ai/TFM_Distillation` 在本次 GitHub API 返回 404 | full-F / OOF 不能按口号选；不导入其 teacher-only、CutMix 或额外教师矩阵 |
| [Rethinking Self-Distillation: Label Averaging and Enhanced Soft Label Refinement with Partial Labels，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9d4824d834b5fe8e6b53dcfe42cab8d2-Abstract-Conference.html) | 读正式论文的理论设定、p. 5 的 top-2 构造及作者 `SelfPLL` 目标代码；理论条件为固定特征、结构化相关性及标签噪声 | 正则化与信息传递必须分开；其 top-2 在二分类会覆盖全部类别，不能新增为有信息的蒸馏臂 |
| [PLD: A Choice-Theoretic List-Wise Knowledge Distillation，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6b2921f24f82dc05cda53bbe50bab90-Abstract-Conference.html) | 本次核对正式身份和[原文 §3.1](https://arxiv.org/html/2506.12542v3)：普通 KD 对双方 logits 使用同温度，并使用归一化 hard / soft 权重；前轮已审其附录实现 | $T=1$ 混合目标是标准 KD 的等价实现；不再引入类排序模块；二分类退化详见既有 objective review |
| [Quantifying Knowledge Distillation using Partial Information Decomposition，AISTATS 2025](https://proceedings.mlr.press/v258/dissanayake25a.html) | 本次核对正式论文条目及摘要；讨论容量有限学生承接教师无关信息的问题；没有在本轮读其实现或复现 RID | 只借用“保真度不等于任务收益”的判读，不能把拟合教师 KL 降低当作成功；不加入信息分解优化器 |
| [Label Smoothing in Disguise?，2026-10-31 预印本](https://arxiv.org/html/2611.01065v1) | 读摘要、§7–8、表 2–3 和工件说明；固定末轮的 KD 优势可与 CE 早停及 teacher-free LS 解释竞争；正文承诺发布工件，但未找到可固定的作者代码入口 | 同预算 hard 必须与 KD 同时训练；记录固定末步和学习过程，不用给 hard 更多过拟合机会制造优势；本文是语言生成研究，只用于设计控制 |
| [Distilling Balanced Knowledge from a Biased Teacher，CVPR 2026 官方条目](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Distilling_Balanced_Knowledge_from_a_Biased_Teacher_CVPR_2026_paper.html)；[原文 v2](https://arxiv.org/html/2506.18496v2) | 官方检索结果可核会议身份；本次直接打开官方网页 403，arXiv 全文可读；读 §3.1 的跨组 / 组内 KL 分解；未取得作者代码 | 教师总体强不代表每个攻击族受益，验收必须保留族级新增 FN；不把多类别长尾分组公式移植到二分类输出 |
| [TabM，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html) | 本次核对正式身份、作者 README 与固定提交；论文是参数高效集成架构，前轮已读训练 / 预测实现 | 作为有质量的表格学生近邻记录，本轮固定已有 small；不能同时换学生再把增益归因于 KD |
| [TabArena，NeurIPS 2025 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html) | 官方摘要明确指出验证方式、预算及验证集过拟合的影响；本次审作者协议校验代码 | 四臂相同预算、初始化、数据顺序及输出时点；C 已经用于开发，不能再次包装成独立证据 |
| [Beyond IID / BeyondArena，2026-06 预印本](https://arxiv.org/html/2606.30410v1) | 本次读任务定义、分组 / 时间验证设定；作者当前 TabArena 实现明确 task-specific split 元数据 | 不用换行随机切分制造新 holdout；不能把同一数据拆分的多 seed 说成独立数据泛化 |
| [Selecting Feature Interactions for GAMs by Distilling Foundation Models，2026-04 预印本](https://arxiv.org/html/2604.13332v1) | 本次复核原文方法及作者链接；既有 review 已深读固定作者实现，是交互发现后以硬标签拟合 GAM | 已覆盖结构蒸馏近邻；本轮不加交互挖掘、GAM、mask 或新表征 |

筛查时排除了身份或用途不合适的来源：`Calibration Transfer via Knowledge Distillation` 虽有 2025 出版记录，[CVF 官方会议身份是 ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Hebbalaguppe_Calibration_Transfer_via_Knowledge_Distillation_ACCV_2024_paper.html)，不计为 2025 新成果；`LLS: Regulating Neural Network Training via Learnable Label Smoothing` 搜索结果存在 withdrawn 标识，未核为正式录用，不作为优秀正式成果依据。搜索到的跨任务仿真论文网页返回 429，未取得原文，不用其摘要支持具体控制设计。

## 3. 直接近邻：公式和实现不能混用

### 3.1 TabTune：固定作者提交后的实际行为

作者仓库 HEAD 于本次查询为 `5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e`（2026-11-04）。以下文件已在线读取，没有运行外部训练：

| 文件 | Git blob | 本次核对位置 |
|---|---|---|
| [mlp_student.py](https://github.com/Lexsi-Labs/TabTune/blob/5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e/tabtune/distillation/students/mlp_student.py#L440) | `7ef11d2cc997f760d4331e17c03a9a426d58085b` | 440–472：学生除以每样本温度；教师概率只裁小值、归一化；hard 项同时带 label smoothing；两项共享样本权重 |
| [losses.py](https://github.com/Lexsi-Labs/TabTune/blob/5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e/tabtune/distillation/losses.py#L13) | `530251b83aefa017df6746a145c7e830b431bb65` | 13–56 普通入口同样只给学生温度变换；68–105 的自适应温度和置信权重与论文表述存在差别 |
| [distiller.py](https://github.com/Lexsi-Labs/TabTune/blob/5c014e2aaf48da4c6b9dfda1f5364dfb04d4776e/tabtune/distillation/distiller.py#L380) | `6ed02d029dcf7fd60f7e13d74fdbe824939ba78d` | 380–420：仅 ICL inference 且 folds>1 走 OOF，其他预拟合 / 微调教师直接预测；OOF 入口使用按行分层划分 |

因此，“作者代码默认参数”不等于本项目所需的普通对称温度 KD，也不等于只改变教师信息的一项控制。采用 $T=1$ 可移除温度坐标歧义；不复制自适应温度、置信加权或额外 LS。

### 3.2 SelfPLL：二分类退化与代码差异

[正式论文 p. 5](https://proceedings.iclr.cc/paper_files/paper/2025/file/9d4824d834b5fe8e6b53dcfe42cab8d2-Paper-Conference.pdf) 将 top-2 类别各赋 $1/2$ 权重。作者仓库固定为 `4092b2d141d812ad4df8d2f60c17c37148f0c6f8`（2025-03-19）：

- [train_linear.py:277–281](https://github.com/Hyeonsu-Jeong/SelfPLL/blob/4092b2d141d812ad4df8d2f60c17c37148f0c6f8/train_linear.py#L277)，blob `2348a8fded7c9c1685ab286609ad098d2e6cfd01`：ordinary 分支使用教师 softmax 目标 CE；partial 分支调用 `two_hot` 和 `partial_loss`。
- [util.py:285–289](https://github.com/Hyeonsu-Jeong/SelfPLL/blob/4092b2d141d812ad4df8d2f60c17c37148f0c6f8/util.py#L285)，blob `a219b133b4fe6379e8ac41bc4c07c6f8473466f1`：实际 partial loss 是对 top-2 的 $(1-p^q)/q$ 取均值，默认 $q=0.7$，并非直接对等权标签做 CE。

本项目的二分类代数检查：top-2 恒为两类。论文等权目标是 $(1/2,1/2)$；代码的单行目标为

$$
L(p)=\frac{2-p^{0.7}-(1-p)^{0.7}}{1.4},
$$

最小值也在 $p=1/2$。两者均丢弃所有实例相关的教师置信信息，故不加入新方法臂。作者脚本 [198–207 行](https://github.com/Hyeonsu-Jeong/SelfPLL/blob/4092b2d141d812ad4df8d2f60c17c37148f0c6f8/train_linear.py#L198) 还直接按 `test_acc` 存 best checkpoint；该实现行为不可移植为本项目的选择协议。

### 3.3 验证协议项目的可复查位置

TabArena 当前作者提交为 `2bf318ef1ccc0b549e91e03081c258e7bbee6d26`（2026-11-04）；[validation_protocol.py](https://github.com/autogluon/tabarena/blob/2bf318ef1ccc0b549e91e03081c258e7bbee6d26/packages/tabarena/src/tabarena/benchmark/validation_protocol.py#L204) 的 blob 为 `bf1c47dc03b9045c2fdcc1443a2bebed3fb46152`。204–213 行声明任务相关分组 / 时间验证；331–361 行检查 bagged 实验的协议及禁止覆盖任务 split 元数据。这里采用的是固定分割来源和验证机会的原则，不引入其 bagging 预算。

TabM 当前作者提交为 `28e47ae301c92ec37787dde1ce923a0793f405b4`；本次读取的 [README](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/README.md) blob 为 `2349caedbb18cf24048f32ae42bf595b45fbfd1b`。本次仅核作者项目与结构定义，没有运行 TabM 或新增架构比较。

## 4. 为什么必须归一化损失

记学生概率 $p_i=\sigma(z_i)$，固定教师概率 $q_i$，硬标签 $y_i\in\{0,1\}$。本轮 $T=1$ 时存在精确恒等式：

$$
\operatorname{BCE}(y_i,p_i)+\lambda\operatorname{KL}(q_i\Vert p_i)
=(1+\lambda)\operatorname{BCE}\left(\frac{y_i+\lambda q_i}{1+\lambda},p_i\right)-\lambda H(q_i).
$$

右侧熵项与学生参数无关。采用相同、冻结的类别权重 $w(y_i)$ 和相同 reduction 时，四臂统一优化

$$
L_a=\frac{\sum_i w(y_i)\operatorname{BCE}(r_i^{(a)},p_i)}{\sum_i w(y_i)},\qquad
\frac{\partial L_a}{\partial z_i}=\frac{w(y_i)}{\sum_jw(y_j)}(p_i-r_i^{(a)}).
$$

若现有训练器使用固定样本数而非权重和作为分母，四臂仍应保持那个相同分母；关键是不得在 KD 臂独自改变 reduction。$w$ 必须由硬标签计算，不能随软目标重新计算类别权重或 `pos_weight`。

`hard + 0.3 KL` 和 `hard` 直接比较，会同时改变总梯度尺度。Adam 的近似缩放不变性不足以消除有限步数、$\epsilon$、梯度裁剪及权重衰减的交互。统一混合目标能去掉这个混杂。此处 $\lambda=0.3$ 对应归一化 soft 比重 $\alpha=3/13\approx0.230769$，不能写成 $\alpha=0.3$。

## 5. 四臂分别识别什么

令 $F_c=\{i\in F:y_i=c\}$，$\bar q_c=|F_c|^{-1}\sum_{i\in F_c}q_i$。现有类别平衡权重在同类中为常数，故普通均值也等于类内加权均值。

以冻结的全字段输入定义完全相同的特征模式 $g(i)$，同模式教师值为 $q_g$，模式内 Normal / Attack 行数为 $(n_{g0},n_{g1})$。只在计数对相同的模式之间生成一次固定置换 $\pi$，令 $q_i^{S}=q_{\pi(g(i))}$。每层先将模式按稳定标识排序，再用 `SeedSequence([17,353])` 对模式随机排列，最后循环移位 1 位；单例层保持不动。全部在训练前一次构造，不按结果挑选 seed。该构造不是对所有可能置换的均匀抽样。

由于置换前后每个 q 都携带相同的 Normal / Attack 重复次数，两类的完整行级 q 边际分布都保持不变，同时维持同输入同教师值。

| 臂 | 目标 | 保留 / 删除的信息 |
|---|---|---|
| H：hard | $r_i=y_i$ | 同预算硬标签控制 |
| K：普通 KD | $r_i=(y_i+0.3q_i)/1.3$ | 保留教师置信度与实例的实际对应 |
| S：模式分层固定置乱 | $r_i=(y_i+0.3q_{\pi(g(i))})/1.3$ | 保留每类完整 q 分布、模式规模和标签组成；破坏层内 q 与特征模式的对应 |
| M：标签内均值 | $r_i=(y_i+0.3\bar q_{y_i})/1.3$ | 保留每类平均软化幅度；删除类内置信度异质性 |

四臂只使用 $F$ 构造目标。类别指 Normal / Attack，不使用攻击族 `attack_cat` 定义置换桶或均值。否则会额外保留攻击族监督，改变二分类信息预算。

### 5.1 不采用统一 $q=0.5$ 的原因

uniform 控制在 $\lambda=0.3$ 时对每行产生 $|r-y|=3/26\approx0.1153846$。已有 UNSW 全字段 F 真实教师的等行均值为 $0.00989731$，uniform 的平均标签移动约大 **11.66 倍**。因此它同时改变实例对应、平均正则强度和两类不对称性，不适合占用本次仅有的第四臂。

M 是利用 F 硬标签构造的类别条件平滑控制，**不是声称复现标准均匀 label smoothing**。它不使用实例特征，但依赖 $y$，因此不是一个只依赖输入的合法教师函数 $q(x)$。同输入异标签模式在 M 下产生不同混合目标，来源是硬监督的定义，不能误报为新增输入或教师随机性。K 胜过 M 可支持实例级信息有增量；不能据此写成胜过所有 LS 超参数。

### 5.2 同类置乱能够和不能识别什么

K 与 S 的差异检验“相同规模、相同标签组成的模式中，置信度配给正确模式”是否重要；S 与 M 的差异反映保留分层结构的随机异质性相对类均值软化的作用。K 同时优于 H、S、M，才支持本轮存在超出平均软化及该随机目标控制的迁移信号。

层内置乱也会改变教师置信度与攻击族、困难程度的对应；这些是被检验的实例信息组成部分，不是单独可归因的机制。一次固定置换不能给出关于所有置换的显著性结论。不要每轮重置乱：每轮重新抽取会逐步逼近各分层的均值目标，同时引入新的训练噪声。

简单按行类内置乱会向相同输入分配不同教师值；本方案改用上述模式层置换，避免这个额外冲突。K>S 单独仍可能仅表示随机目标损害更大，故需要 K>M 和 K>H。置换前后至少核对每类 q 排序数组完全相同、均值一致、同模式教师值唯一、模式固定点比例，以及目标文件哈希。循环移位使非单例层没有模式映射固定点；仍可能因不同模式原本有相同 q 而数值不变，单例层也保持原值，两者须分别计数。

这个控制有明确的识别范围：模式大小和标签组成与 q 的关系被保留，单例层不能置换。因此必须报告可置换的模式数、行数、$|q-y|$ 质量覆盖率及实际 $|q^{S}-q|$；不能只报告置换函数已运行。若可变软信号覆盖不足，K 与 S 接近应记为控制无力，不能判定实例信息不存在。

若以后增加非类别常数的样本权重，普通类内置换不再保证加权软目标质量不变，当前控制必须重新推导；本轮不引入这种权重。

## 6. 教师、温度与预算的冻结理由

**教师。** 选择现成 full-F 全字段 HGB 作为本轮唯一教师：已有 $F$ 分数非硬标签，差异不是由单一重复模式主导；有无低 FPR headroom 应以当前 full-small 对应工件核查。选择 full-F 是有现成证据、零新增教师拟合的最小问题，不是证明它普遍优于 OOF。UNSW 三个 seed 的 OOF 原生数组相同，也不能拿来冒充三份独立教师证据。

**温度。** $T=1$ 检验现有教师原生概率坐标。HGB sigmoid 分数未经额外校准，不能称为真实总体后验；本轮也不以 C 标签拟合温度或校准器。若这次失败，结论是该冻结教师坐标、该混合权重、该学生预算下未获收益，不能反向推出所有蒸馏温度都无效。温度探索是后续另一假设，不能在本轮临时追加。

**训练预算。** 四臂从同一 seed17、同一 full-small warmup 参数出发；优化器如何重置由主协议统一明确。共用行顺序、随机数规则、学习率、类别权重、裁剪、数据暴露和固定 40 个 macro-step，故最多 4 次 continuation，无教师重训，无架构搜索。预先冻结固定终点有助于消除四臂不等的 C 选点机会；中间轨迹用于解释，不择优替换终点。真实训练放在实验室 RTX 4090D。

**判据。** 论文不提供 UNSW 的可接受误差阈值。主协议将冻结：按相同 C 阈值登记规则，K 相对 H、S、M 的 TPR 均至少提高 0.002（至少 34 TP），且 pAUC 不降；R 相对 H 不降；各攻击族净 TP 不降；正常 FP 相对 H 净不增加；删除最大正收益 group 后仍有净正收益。新增与移除错误须分开交付，不以净值掩盖。C 既有重复开发使用，单 seed 只能形成继续研究或停止本候选的开发判断；进入独立确认需要另立未参与选择的数据来源，旧 V/test 不能承担这个角色。

## 7. 结果判读与停止规则的文献依据

- K 不胜 H：没有本预算下的 KD 应用增益，不扩大温度 / 教师矩阵补救。
- K 胜 H，但不胜 M：类平均软化足以竞争，不能声称教师实例知识带来收益。
- K 胜 H、M，但不胜 S：教师特定实例对应机制未通过控制；可能是异质目标的正则化作用。
- K 只胜 S：只能说随机破坏对应伤害更大，没有实际收益证据。
- K 胜 H、S、M，且通过低 FPR 及新增错误约束：有依据做小规模配对稳定性复核；不据此宣称新算法或独立泛化。

这些判断是本轮目标代数与控制含义的推论，不是任何外部论文已经证明 UNSW 的预期结果。所有外部项目在本轮的状态均为阅读 / 静态核查；没有将其标为已运行或已复现。
