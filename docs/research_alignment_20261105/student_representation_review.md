# 学生结构与数值表示：2025–2026 成果复核

核查日期：2026-11-05。承接[上一轮学生与教师审查](../research_alignment_20261104/student_class_and_teacher_review.md)，结合 round8 已回传结果，重新检查表示、函数类与监督对象。本次初筛 10 项近期方法／研究项目，深入读取 TabM、样条编码研究的原文公式和作者关键实现，并检查 GGPL 的可用代码及原文取得缺口。没有执行真实数据训练，没有读取 calibration/test。

## 1. 结论与本次实际改动

**下一步优先检查输入表示与已有适配器的有效性，不能直接换一个更大的学生或追加 KD 损失。** 两份数据需要不同问题：

- CIC24：24 字段 HGB 能检出那 2,272 条主导漏报模式，说明这些字段在该批 C 记录上仍可用于判别；small/wide MLP 均未解决。新增冻结模型诊断发现，按 F 取值范围裁剪输入可以恢复这 2,272 条，但 small 的误报也增加，结果随模型与阈值而变。下一步先在全部冻结模型和两个环境中完成零拟合审计，检查 F 内尺度、取值模式、支持范围与整体误报代价。固定 PLE 保留为有后续证据才启用的备选，不立即进入训练。
- UNSW：保留全字段 small MLP。hard 续训对 hard 续训时，MI24 少检出 1,297 条（13,969 对 12,672），包仅缩小约 14%，暂不值得作为默认预算。全字段 HGB 相对全字段 hard MLP 修复 730 条、破坏 251 条；迁移问题应围绕这份互补性，不先强制压缩字段。
- 监督问题独立存在：CIC F 内 T=1 教师概率几乎等于标签。修好输入表示只能支持更强的直接学习，不能直接成为 KD 增益；UNSW 的非硬概率较丰富，也仍需真实／置乱教师对照。

本次源码检查发现并修复一个**先前 TabM 适配器的实际错误**：`PiecewiseLinearEmbeddings(version="B", activation=True)` 的分段残差在零初始化后经过 ReLU，初始及后续梯度均为零，只留下仿射数值嵌入。现已按作者配置改为 `activation=False`；训练协议号由 2 改为 3，包格式号、可选依赖、默认入口均未改变。新增残差梯度和旧零残差包预测兼容测试，相关合成测试 **22 passed、0 skipped**。**round8 screen 没有使用 TabM，这个错误不是本轮 CIC24 失败的原因；修复也没有产生新的真实性能证据。** 具体推导见第 3 节。

本轮结果依据：[详细分析](<E:/OneDrive - Coventry University/中传/毕业论文/工作/实验结果分析_20261105/本轮实验详细分析_20261105.md>)、[24 个模型的独立复核](<E:/OneDrive - Coventry University/中传/毕业论文/工作/实验结果分析_20261105/screen_review/screen_review.md>)。

## 2. 广度筛查：每项都对应一个实际决策

年份按正式发表身份判断；2024 年首发但正式发表于 2025 年的工作标明这一点。公开、已读和本项目已运行分开记录。

| 成果／项目 | 实时核实的身份与一手来源 | 对本项目的作用 | 当前取舍 |
|---|---|---|---|
| TabM，含 PLE、TabM-mini | ICLR 2025；arXiv v3 为 2025-02-18；官方稳定包 0.0.3。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)、[代码](https://github.com/yandex-research/tabm) | 将数值表示与参数共享集成拆开；现有接口可以复用，但本次发现其 PLE 支路错误 | 优先采用其正确 PLE 作为表示对照；完整 TabM 是强神经参照，不预定为部署赢家 |
| From Uniform to Learned Knots | 2026；最新版 arXiv v2（8 月 17 日）；作者仓库与作者机构均标注 TMLR 2026，OpenReview 决定页本次被验证页面阻断。[原文](https://arxiv.org/html/2604.05635v2)、[作者代码](https://github.com/mkumar73/tdl-numerical-encodings)、[作者机构记录](https://www.hsbi.de/publikationsserver/record/7112) | 固定骨干比较标准化、PLE 和不同样条，直接覆盖本项目的表示疑问 | 优先固定 PLE；不先引入可学习高阶样条及多个学习率 |
| GGPL | KDD 2026，正式标题为 *Efficient Piecewise-Linear Embeddings for Deep Tabular Regression by Guided Breakpoint Allocation*。[会议目录](https://kdd2026.kdd.org/papers/)、[作者机构](https://www.lgresearch.ai/publication/list)、[作者代码](https://github.com/LGAI-Research/gbdt-guided-piecewise-linear-embeddings) | 树的分裂阈值已经被用于指导神经数值断点；直接限制未来“结构迁移”的原创性主张 | 保留为直接近邻；先解决固定 PLE，且取得正式全文后才考虑完整适配 |
| Tabular Numeric Stretch Transformation | 2026-08 预印本，v1；作者页仍标在审，未核到该论文的作者代码链接。[原文](https://arxiv.org/html/2608.09162v1)、[作者页](https://ziiihooo.com/publications/) | 不扩维的分段单调变换，直接回应重尾数据与小模型预算 | 作为低成本表示假设的依据，不宣称已具备可复现作者实现 |
| TabPrep | 2026-06 预印本 v1；作者公开包装器，生成器在 AutoGluon。[原文](https://arxiv.org/html/2606.02384v1)、[作者代码](https://github.com/atschalz/tabprep) | 将数值算术结构、类别条件结构及伪类别交互显式化，提醒不能把所有差距解释为网络容量 | 不整套接入；当前没有算术缺口证据，默认大规模生成特征与轻量目标不相称 |
| ModernNCA | ICLR 2025，首版预印本为 2024。[正式论文](https://openreview.net/pdf?id=JytL2MrlLT)、[作者方法说明](https://www.lamda.nju.edu.cn/yehj/research/tabular-learning/index.html)、[作者实现](https://github.com/LAMDA-Tabular/TALENT/blob/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/models/modernNCA.py) | 学习邻域几何，提供另一种局部结构表达；已读检索与预测实现 | 不作为首选轻量学生：部署需保留候选样本及表示，不能只统计网络权重 |
| TabKANet／TabMLPNet 数值编码研究 | *Knowledge-Based Systems* 330（2025），114697；与原 2024 TabKANet 预印本不是相同标题。[正式 DOI](https://doi.org/10.1016/j.knosys.2025.114697)、[作者仓库及引用](https://github.com/AI-thpremed/TabKANet) | 将数值增强、归一化、编码分开，考虑偏斜数值 | 只作备选：KAN、BatchNorm、噪声与骨干同时改变，当前不足以识别本项目的单一原因；未深核其最终论文全部实现 |
| TabArena | NeurIPS 2025 Datasets and Benchmarks Track。[正式记录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)、[维护项目](https://github.com/autogluon/tabarena) | 提供现代 MLP、GBDT 及调参／集成的比较口径；指出验证策略影响排名 | 用于设计对照预算；不把跨数据集排行榜分数移植为 IDS 低 FPR 结论 |
| TabReD | ICLR 2025，首版预印本为 2024。[正式记录](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html)、[作者项目](https://github.com/yandex-research/tabred) | 时间漂移和较丰富字段会改变模型排名；简单 MLP 与 GBDT 仍是有力基线 | 保留全字段简单模型；后续独立确认关注时间／来源，而非只换随机 seed |
| TALENT | JMLR 26(226), 2025，软件论文。[正式记录](https://www.jmlr.org/papers/v26/25-0512.html)、[作者项目](https://github.com/LAMDA-Tabular/TALENT) | 同一工具箱提供数值编码和标准／分位数／power／robust 变换，避免从头再造每种表示 | 用其实现核查简单控制，不整套引入 35+ 模型。其另一篇 *A Closer Look* 原始预印本为 2024，不误写为新 2026 方法 |

RealMLP、PLE 的最初论文分别属于 2024、2022，作为上述近期研究使用的背景基线保留，不将它们改写成年份符合要求的新推荐。CatBoost/HGB 是本项目已实测简单基线；small CatBoost 的不利结果继续保留，不因阅读更多论文而撤销。

## 3. 深入核查一：TabM 与真正有效的数值嵌入

### 3.1 论文与作者实现

TabM §3.2–3.5 的关键机制是共享矩阵、保留成员适配参数：

$$
h_i=s_i\odot\bigl(W(r_i\odot x_i)\bigr)+b_i,
\qquad
L=\frac{1}{k}\sum_{i=1}^{k}\ell(y,z_i),
\qquad
\hat p=\frac{1}{k}\sum_{i=1}^{k}\sigma(z_i).
$$

成员分别承担监督损失，部署取平均概率；不能改成先平均 logit 再计算训练损失。TabM-mini 只保留首个输入缩放适配器。论文附录 D.8–D.9 将分位数 PLE 作为独立表示部件；作者预处理默认采用加微量噪声的分位数正态化，并非本项目当前的 StandardScaler。[原文 §3、附录 D](https://arxiv.org/html/2410.24210v3)、[线性共享实现 L518–658](https://github.com/yandex-research/tabm/blob/a507095893d784c5702059d737ddfbd1299c41dd/tabm.py#L518)、[论文预处理 L262–298](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/paper/lib/data.py#L262)。

PLE 在区间内部使用分段坐标

$$
\phi_j(x)=\operatorname{clip}_{[0,1]}\!\left(\frac{x-b_{j-1}}{b_j-b_{j-1}}\right).
$$

作者库对首尾区间保留线性外推；因此这不是“全部截到 [0,1]”的实现。固定版本的 `compute_bins` 先计算 F 分位数，再删除重复分位点；低基数、重复值和常量列需要检查实际有效箱数。[分箱 L348–463](https://github.com/yandex-research/rtdl-num-embeddings/blob/7381c5a4ed01253fbef9ea8cfed137f907265552/package/rtdl_num_embeddings.py#L348)、[边界行为 L634–681](https://github.com/yandex-research/rtdl-num-embeddings/blob/7381c5a4ed01253fbef9ea8cfed137f907265552/package/rtdl_num_embeddings.py#L634)。

### 3.2 本次发现并修复的死支路

官方 version B 的计算是仿射嵌入加分段残差。若使用 `activation=True`，本仓原设置等价于

$$
e(x)=a x+b+\operatorname{ReLU}(U\phi(x)),\qquad U_0=0.
$$

PyTorch 在零点使用 ReLU 导数 0，因此 $\nabla_U L=0$；没有偏置的残差投影会一直停在零。网络仍有可训练参数，但它没有学习声称的分段残差。[零初始化及 forward L768–805](https://github.com/yandex-research/rtdl-num-embeddings/blob/7381c5a4ed01253fbef9ea8cfed137f907265552/package/rtdl_num_embeddings.py#L768)。作者稳定版 README 明确推荐 `version="B", activation=False`，这与论文 Q-L 表示一致。[官方推荐 L253](https://github.com/yandex-research/rtdl-num-embeddings/blob/7381c5a4ed01253fbef9ea8cfed137f907265552/package/README.md#L253)。

本次相同初始化的合成反向结果：开启 ReLU 的残差梯度最大绝对值为 0；关闭后为 0.282293；两者仿射分支梯度均为 0.120850。修复位置是 [recent_baselines.py](../../src/ids/recent_baselines.py)，验证为 [test_recent_baselines.py](../../tests/test_recent_baselines.py)。新协议使用 $e(x)=a x+b+U\phi(x)$。旧训练协议正常生成的零残差工件重新加载后预测保持逐项一致；未改写旧工件。

此前“官方接口能拟合、能重载”的验证不足以证明具体分段分支真的参与学习。本次新增的梯度断言检验这一机制。既往真实 TabM 工件若使用此设置，需要按实际训练协议重新标注；不能把旧结果称作有效 PLE 的实证。

### 3.3 成本和可以推翻解释的对照

固定版本：TabM 0.0.3，tag commit `a507095893d784c5702059d737ddfbd1299c41dd`；rtdl_num_embeddings 0.0.12，tag commit `7381c5a4ed01253fbef9ea8cfed137f907265552`。两者模块与本次 HEAD 下载内容逐字节相同。TabM 的最小依赖为 PyTorch、rtdl_num_embeddings、typing_extensions，无预训练 checkpoint；许可证 Apache-2.0。当前已在隔离环境使用官方包完成合成检查，未做新 IDS 训练。[包配置](https://github.com/yandex-research/tabm/blob/a507095893d784c5702059d737ddfbd1299c41dd/pyproject.toml#L15)。

不能将“参数共享”写成“几乎没有推理成本”。成员仍需计算；作者通常用 $k=32$，本项目的单条 CPU 预算必须另测。直接套本仓默认 48 箱、16 维嵌入也不轻：对 24 个纯数值字段和 [32,16] 骨干，若各列均有 48 箱，version B 嵌入加骨干约为 **32,065 个可训练参数**，原标量骨干为 **1,345 个**；另有分箱和编码缓冲区。这是静态上界配置计算，不是测得的包大小。

必要控制：保持相同 F 标签、字段和选择次数，先比较标量 MLP 与固定 PLE MLP；再用相同表示比较单骨干与 TabM，不能同时换表示、集成和训练预算后只归因于其中之一。若等成本的标量 MLP 同样修复错误，就不能声称收益来自分段表示；若 PLE 后直接 hard 已解决缺口，则先记录表示收益，再单独检验 KD。

## 4. 深入核查二：固定与可学习样条并非同一成本层级

2026 年样条研究的 §3.1.4 用正区间宽度保证断点有序：

$$
\pi_r=\operatorname{softmax}(a)_r,\quad
w_r=\delta+\bigl(1-(K+1)\delta\bigr)\pi_r,\quad
\kappa_\ell=\sum_{r\leq\ell}w_r.
$$

式 (10)–(12) 与作者 `LearnableKnots.widths/internal_knots` 一致；固定分位数节点只依赖训练输入，CART/LightGBM 节点额外使用训练标签。作者比较表明分类上 PLE 更稳健，可学样条的收益和开销依赖骨干与编码宽度。[原文](https://arxiv.org/html/2604.05635v2#S3.SS1.SSS4)、[固定源码 L15–76](https://github.com/mkumar73/tdl-numerical-encodings/blob/a3b8149576ccc48fc0a8d13bb881f1e292f668c0/src/models/spline_expanders.py#L15)。

这对 CIC 的价值是提出可检验的局部区间表示，而非证明需要复杂样条。已核到两个会影响迁移的实现差异：

- `LearnableSplineBase` 的训练范围 min-max 变换把新输入截到 [0,1]；固定 B-spline 也裁到训练范围。若主导错误位于 F 取值范围外，这会把不同极值合并，不能直接称其更保留异常信息。[可学习变换 L18–67](https://github.com/mkumar73/tdl-numerical-encodings/blob/a3b8149576ccc48fc0a8d13bb881f1e292f668c0/src/models/learnable_spline_base.py#L18)、[固定 B-spline L155](https://github.com/mkumar73/tdl-numerical-encodings/blob/a3b8149576ccc48fc0a8d13bb881f1e292f668c0/src/preprocessors/bspline.py#L155)。
- 公开项目是完整研究管线，Python≥3.13、NumPy≥2.3.2、PyTorch≥2.8、scikit-learn≥1.7.2，并依赖 LightGBM 等，与本项目当前锁定环境不同。没有为本项目安装或运行；不应为一个表示探针升级整个环境。[固定依赖](https://github.com/mkumar73/tdl-numerical-encodings/blob/a3b8149576ccc48fc0a8d13bb881f1e292f668c0/pyproject.toml)。

先采用固定 PLE 的理由是：其作者代码、稳定包和本仓接口已具备，能用较少变量检验局部分段表示。若再考虑可学习节点，需要固定节点数、相同骨干、相同选择预算及固定节点对照；若固定节点同样有效，就撤销“节点优化必要”的解释。对监督节点，还需同节点预算的无监督分位数对照，不能把额外 F 标签使用隐藏在“预处理”里。

## 5. 深入核查三：GGPL 已覆盖树引导断点，当前不直接移植

GGPL 的正式 KDD 2026 身份已由会议与作者机构核实。正式 ACM DOI 为 [10.1145/3770855.3818128](https://doi.org/10.1145/3770855.3818128)。本次 ACM 正文／PDF 返回 403，原 ICLR 投稿页也被验证页面阻断，**没有取得 KDD 最终原文公式**；以下只陈述已读作者源码行为，不冒称最终论文已逐式复现。

固定 commit `8cfc0b77c58c13a69cf312c063f6fa257a3101c8` 的预处理从拟合 GBDT 提取阈值及分裂分数，将所有字段候选一起排序，选择全局 `n_breakpoints` 预算，然后添加各字段训练最小／最大值。[全局选择 L291–321](https://github.com/LGAI-Research/gbdt-guided-piecewise-linear-embeddings/blob/8cfc0b77c58c13a69cf312c063f6fa257a3101c8/gbdt_guided_piecewise_linear_embeddings_preprocessing.py#L291)。这不是每个字段各取 `n_breakpoints` 个节点。

从源码可直接写出其断点参数化：$\Delta_j=\operatorname{softmax}(a_j)\,\mathrm{range}_j$，再累加得到有序断点；训练可随机停用内部断点，推理使用全部断点并缓存仿射系数。嵌入仍是仿射基项加零初始化的 PLE 残差，**没有在残差之后附加导致死支路的 ReLU**。[参数化 L286–307](https://github.com/LGAI-Research/gbdt-guided-piecewise-linear-embeddings/blob/8cfc0b77c58c13a69cf312c063f6fa257a3101c8/gbdt_guided_piecewise_linear_embeddings.py#L286)、[初始化与输出 L380–446](https://github.com/LGAI-Research/gbdt-guided-piecewise-linear-embeddings/blob/8cfc0b77c58c13a69cf312c063f6fa257a3101c8/gbdt_guided_piecewise_linear_embeddings.py#L380)。

它提供的是教师结构知识的近邻例子：部署时无需运行树，学生以树阈值作为表示先验。若本项目改用 HGB 阈值、换二分类低 FPR 任务或给阈值加权，首先属于适配，不能直接称原创。要主张树结构有用，必须排除“相同节点数的分位数／随机节点也一样有效”；要主张低 FPR 选择必要，还应与普通分裂收益选择对照。没有必要现在同时加入这两组新选择。

可用代码是作者单文件模块加最小 trainer，README 明确说明 trainer 不是原研究完整管线；无需预训练权重。仓库没有 release，`pyproject` 标记 0.1.0，并约束 PyTorch 2.11，按后端另需 XGBoost／LightGBM／CatBoost、pandas、joblib。本项目 PyTorch 2.6 不在其声明验证范围。示例自行随机分 train/validation/test，不能直接对本项目运行。正式结果主要针对回归；本项目二分类目标及分组划分的适配还没有证据。[作者说明及依赖](https://github.com/LGAI-Research/gbdt-guided-piecewise-linear-embeddings/blob/8cfc0b77c58c13a69cf312c063f6fa257a3101c8/README.md)、[环境约束](https://github.com/LGAI-Research/gbdt-guided-piecewise-linear-embeddings/blob/8cfc0b77c58c13a69cf312c063f6fa257a3101c8/pyproject.toml#L10)。

## 6. 其余候选为何暂不进入实验矩阵

**Numeric Stretch：理论上适合低成本，但代码证据不足。** 原文式 (2) 在相邻分位点之间用 $s(x)=c_{t-1}+w_t(x-b_{t-1})/(b_t-b_{t-1})$，无监督等宽分配近似经验 CDF，监督版本根据目标变化分配宽度。它提醒我们，改变输入几何不一定需要扩展输入维度。本次查原文、作者页和仓库搜索，未找到对应作者实现；先采用成熟、可核实的 F-only 数值变换控制，不独立复刻整套监督 Stretch 并称作者复现。[原文 §3](https://arxiv.org/html/2608.09162v1#S3)。

**TabPrep：特征生成不是免费的归一化。** 已核原文 §3.2／附录 C 的预算算术生成，以及作者包装器 commit `fee7f189764070836b59e8cf29207050468589f0`。AutoGluon commit `e3cb3b71386c54ff11155d6e7a67f6b43f117208` 实现依次生成除、乘、减、加，默认上限 2,000 新特征；RSFC 将字段子集哈希后进行 OOF 目标编码。后者不能把本项目的会话／重复组跨 OOF 折共享而仍声称严格组外监督。[包装器 L79–114](https://github.com/atschalz/tabprep/blob/fee7f189764070836b59e8cf29207050468589f0/tabprep.py#L79)、[算术生成 L121–173](https://github.com/autogluon/autogluon/blob/e3cb3b71386c54ff11155d6e7a67f6b43f117208/features/src/autogluon/features/generators/arithmetic/preprocessor.py#L121)、[RSFC L295–355](https://github.com/autogluon/autogluon/blob/e3cb3b71386c54ff11155d6e7a67f6b43f117208/features/src/autogluon/features/generators/rsfc.py#L295)、[默认 KFold 逻辑 L105–120](https://github.com/autogluon/autogluon/blob/e3cb3b71386c54ff11155d6e7a67f6b43f117208/features/src/autogluon/features/generators/oof_target_encoder.py#L105)。当前失败模式还没有显示必须生成某种算术关系，故不整套接入。

**ModernNCA：近邻支持库必须计入部署。** 已核 TALENT commit `39b54024902955e51063a34bce17d514cd68ff42`：训练随机抽候选，预测计算查询到候选表示的 `torch.cdist`，softmax 后对候选标签加权。其成本随候选库大小变化；不能拿小网络权重文件与本项目完整模型包比较。[模型 L65–116](https://github.com/LAMDA-Tabular/TALENT/blob/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/models/modernNCA.py#L65)。本项目约 90 万 F 行下，没有理由先引入完整检索库；近邻关系可用于有限诊断，而非默认部署。

**TabArena／TALENT：强基线的价值是控制，不是扩大排行榜。** TALENT 的标准、min-max、quantile、power、robust 分支都能核对训练内拟合范围。[变换实现 L387–424](https://github.com/LAMDA-Tabular/TALENT/blob/39b54024902955e51063a34bce17d514cd68ff42/TALENT/model/lib/data.py#L387)。先选择能区分当前假设的控制；不因这些选项存在就全跑。TabArena 当前代码 commit 为 `2bf318ef1ccc0b549e91e03081c258e7bbee6d26`，其大预算调参／集成结果不能直接对照本轮一次终态的树模型。

## 7. 本项目下一步的可证伪顺序

### 7.1 先做零拟合输入机制检查

报告定稿时新增的冻结响应证据进一步提高了这一步的优先级：F 极值范围裁剪能够恢复 CIC 的主导漏报模式，并伴随模型相关的误报变化。因此先跨全部现有模型、两个环境检查保留旧阈值和按既定 R+C 规则登记阈值的结果，记录 TP、FP、受影响模式和其他攻击族。这里的阈值登记仍使用开发集，不能成为独立确认；也不能根据当前一个模式直接把裁剪设置为默认预处理。

对 CIC 主导模式，仅用已允许的 F/R/C 和冻结模型，登记每列原值、F 分位位置、标准化值、缺失填补、有效唯一值数及 float32 前后的可分辨性；核对 F 中同值／近邻模式的标签与组支持，检查所选小网络的首层激活及最终分数。**StandardScaler 的非退化仿射变换本身并不丢弃实数信息**；只有取得异常尺度、有限精度、填补合并或优化响应证据，才能进一步说明哪一环导致困难。

量化变换同样要查边界：经验分位数通常压到训练支持边界；简单对数不适合直接套在标志位、类别编号或允许负数且语义不同的字段上；本次核到的样条代码还有训练范围裁剪。不能因“重尾”这个名称就先裁掉异常大值。

### 7.2 若表示假设仍有支持，先冻结最小直接学习对照

| 要回答的假设 | 必要比较 | 会使解释失效的结果 |
|---|---|---|
| 标量尺度造成优化困难 | 原小 MLP；同维度、F-only 的一个事前选定单调数值变换＋同骨干 | 新变换未改善目标错误，或改善完全来自额外选模机会；不能继续以缩放作为已知原因 |
| 局部分段表示比单纯重标度必要 | 有效固定 PLE＋同骨干；标量变换控制；记录完整参数／缓冲区／CPU 成本 | 等成本标量模型表现相同，或仅扩大容量就能解释收益 |
| 集成比单模型必要 | 先固定同一种表示，再比较单骨干与一个冻结预算的 TabM 配方 | 代价超过部署约束，或收益仅来自更大搜索／成员选择机会 |
| 教师提供实例知识 | 已确认学生上比较 hard、真实目标、固定置乱目标，必要时保留直接目标 | 真实目标不超过 hard 或置乱目标，不支持实例知识解释 |

这不是已冻结的新训练矩阵。先确定包大小、单条延迟和允许质量损失的验收门槛，再决定哪一项值得拟合。旧 F/R/C 身份、阈值规则和开发集属性保留；所有分位点、字段策略和标签驱动节点只能在 F 拟合。C 的 2,272 行是一个高重复模式，不能因为修复它就称数千个独立难例均得到改善，也不能删去该组提高成绩。

对 UNSW，若现有全字段小 MLP 已满足部署预算，就先围绕同字段教师的错误互补研究；新数值嵌入必须说明其额外成本。对 CIC，先定位可分结构为何没被直接 hard 学到，再选择有额外信息的监督对象。两者都不以“引用了新论文”作为启动更大矩阵的理由。

## 8. 本次验证与可追溯材料

- 修改范围：本报告、`src/ids/recent_baselines.py`、`tests/test_recent_baselines.py`；未改默认 screen 配方或历史结果。
- 官方包合成检查：`tests/test_recent_baselines.py` 共 22 项通过、0 跳过，7.21 秒；17 条提示来自常量数值列只有一个箱。新测试覆盖从零初始化获得非零分段梯度，以及旧协议零残差包重载预测不变。
- 原文和源码状态：TabM、样条研究已读原文公式及固定作者源码；GGPL 已读关键实现，最终原文取得失败；TabPrep 已读机制与相关生成器；ModernNCA 已读关键检索代码；Numeric Stretch 未取得作者实现。除 TabM/rtdl 合成检查外，没有运行这些方法。
- 工作证据目录：`E:\OneDrive - Coventry University\中传\毕业论文\工作\项目重审_20261105\student_sources`。其中保存各作者库 commit、下载的只读源码、稳定 tag 对照以及 `ple_version_b_synthetic_backward.json`。源码已读不等于原论文实验已复现。
