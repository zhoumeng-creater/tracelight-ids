# R29架构、输入与作者实现重审

2026-11-17。范围：R29实际源码与回传源码快照、R26小MLP源码、已有input_audit/summary、TabM固定作者包与paper实现。没有加载模型、前向、训练、新评分或新判决。本次只新增本文，不改主设计或训练源码。

## 1. 明确判断

**目前没有发现R29在PLE、首层初始化、成员损失、概率平均这些关键环节写错。** 它确实使用作者推荐的TabM＋PLE实现。能确认的设计问题，是把一种合法的表格学习配置，当成了对D迁移已经有机制保证的方案；另有消融归因和作者来源描述不够准确。

需要保留和改正的事项分开处理：

- 保留PLE version B、activation=False、按原字段分组初始化、32成员分别计算BCE、平均成员概率。这些是正确实现。
- 把“有无PLE只变编码”改成“有无PLE的作者配方比较”：无PLE还会自动改变首层缩放初始化和参数维度。
- 把“照作者论文训练”改成“采用作者包/example的实用默认配置，并作任务适配”。paper主实验的批量策略、预处理和调参配置不同。
- 不再因D失败推断模型缺少特征交互。两层稠密非线性主干已经能表达协议、时长、包数和字节数的交互；问题在于监督是否识别了应当保留的交互。
- 若新设计研究显式协议依赖，使用固定UNKNOWN协议槽的配对设计在结构上成立；它只识别协议字段的作用，不证明已解决D，也不消除其他字段携带的传输信息。

## 2. 审查确实对应R29运行源码

下列当前repo文件与R29回传source_snapshot逐文件SHA-256完全一致：

| 文件 | SHA-256 |
|---|---|
| src/ids/recent_baselines.py | bedec25677916ba61a475f53b4295862757af830039a3652730259d359560079 |
| src/ids/model_training/preprocessing.py | 9e711cbc890e1fe21f4b67b2bace3aed70649cc9a3747259089f6fe8488d8d6c |
| src/ids/model_training/models.py | 982f033adacf4046114a46905586a04fcf8fffd8e9cc37b74e514b9bc9772d89 |
| src/ids/route_batch/tabm_training.py | 946d7085c340fddbe691426eca7b2168886ef45c8f99c3a92c30269151ec2aa0 |
| src/ids/flow_history/model.py | e64afcd77bbec68fe4bc819c77ab6adbb072435e258f754996d95cda3a0bc36f |
| src/ids/support_diagnostic/model.py | 6750a3a6cf2d095502dcd255785db785fe53b5b4dca5ed870cb4c5a648bb18ae |

repo为 E:/OneDrive - Coventry University/中传/毕业论文/代码/tracelight-ids/.worktrees/model-training-20261116；回传为 E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261117_第二十九轮模型训练/回传。

这是源代码与已有审计核查；本次没有重新实例化网络证明运行输出逐值等价。

## 3. R29完整结构与精确参数

连续输入固定顺序为duration_seconds、orig_pkts、resp_pkts、orig_ip_bytes、resp_ip_bytes。每个连续字段先识别NaN/-1缺失，按拟合池中位数填补，然后log1p、均值标准差缩放，转FP32。

类别输入为协议6类和五个缺失标记各2类：

$$
[6,2,2,2,2,2].
$$

协议固定映射为[1,6,17,58,132,UNKNOWN]。五个连续字段各经16维PLE，拼接后的输入为：

$$
5\times16+6+5\times2=96.
$$

完整网络为：

1. 五个独立字段的PLE version B；共享于32成员。
2. 类别one-hot，拼接为96维。
3. EnsembleView复制为32个成员视图。
4. 96→512的LinearBatchEnsemble、ReLU、Dropout(0.1)。
5. 512→512的LinearBatchEnsemble、ReLU、Dropout(0.1)。
6. 32个独立512→1输出头；输出头后无dropout。
7. 每成员BCE取平均训练；推理平均成员概率，再稳定计算FP64 log-odds。

首层缩放初始化按以下原字段块共享一个初始随机值：

$$
[16,16,16,16,16,6,2,2,2,2,2].
$$

每成员、每块独立抽标准正态；首层输出缩放及第二层输入/输出缩放初始为1。缩放参数训练后不强制块内保持相同。首层是先对输入做成员缩放，再混合字段，符合作者关于多样性必须在首个字段混合之前引入的要求。[固定TabM源码](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/tabm.py#L1353-L1416)，[字段块推导](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/tabm.py#L1632-L1672)

已有R29模型最大实际区间数为48，参数计算如下：

| 部分 | 参数 |
|---|---:|
| PLE线性支路：5×16权重与5×16偏置 | 160 |
| PLE残差投影：5×48×16，无偏置 | 3840 |
| 第一BE层：512×96＋32×96＋32×512＋32×512 | 84992 |
| 第二BE层：512×512＋3×32×512 | 311296 |
| 独立输出头：32×512＋32 | 16416 |
| 合计 | **416704** |

这与三份已有PLE summary.json一致。若后续使用新拟合池，必须登记实际最大区间数；两臂复用同一边界与embedding状态。最大区间数仍为48时参数仍为416704，不能用未经核查的新边界直接沿抄数字。

## 4. 逐项作者核对

| 项目、实际source/function | 固定作者依据 | 差异及作用 | 具体处置与验证标准 |
|---|---|---|---|
| recent_baselines.py:179 make_tabm_network | TabM.make、TabM.__init__ | 正确把五字段PLE作为num_embeddings传入；没有把96维误当96个连续原字段，首层块划分正确 | 保留。未来规格显式记录resolved activation/start_scaling_init/chunks，避免只记录隐式默认；静态声明应为ReLU/normal/[16×5,6,2×5] |
| recent_baselines.py:190 PiecewiseLinearEmbeddings | [rtdl固定实现723–805行](https://github.com/yandex-research/rtdl-num-embeddings/blob/7381c5a4ed01253fbef9ea8cfed137f907265552/package/rtdl_num_embeddings.py#L723-L805)；[paper/lib/deep.py的包装81–93行](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/paper/lib/deep.py#L81-L93) | version B是线性支路＋零初始化PLE残差；activation=False让残差可学习。若误加ReLU，零点导数会阻断该初始残差支路；R29没有此错 | 保留，禁止为“加强非线性”再加embedding ReLU |
| recent_baselines.py:157 _numeric_bins | [compute_bins固定实现](https://github.com/yandex-research/rtdl-num-embeddings/blob/7381c5a4ed01253fbef9ea8cfed137f907265552/package/rtdl_num_embeddings.py#L447-L462) | 对拟合池计算48分位区间上限，去重重复边界；不是每字段一定48个区间 | 保留；归档实际边界和区间数。常量fallback本轮未触发，不能当失败解释 |
| model_training/preprocessing.py:51/83/90 | [作者transform_num](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/paper/lib/data.py#L262-L298)；[example](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb) | R29为median＋log1p＋standard；paper具体配置为noisy-quantile，example同时给standard与quantile选项。作者并没有强制所有任务分位高斯化 | 合法任务适配，保留；不因论文用了quantile就直接换。改预处理会改变距离、分位内插和范围外行为，需另有机制依据 |
| preprocessing.py:15 _raw | TabM官方cat_cardinalities/one-hot接口 | 协议one-hot避免把协议码当有序量，missing单独输入；这不自动获得协议不变性 | 保留语义映射，输入策略必须单独声明；不得把“类别正确”写成“可跨协议” |
| TabM.make隐式默认 | [固定make 1693–1708行](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/tabm.py#L1693-L1708) | 有PLE：normal；无PLE：random-signs。R29两臂都显式2层，无PLE覆盖了作者默认3层 | 文案改为“固定两层下的作者有/无嵌入配方”；不能把已有TP差值全部归于PLE映射单因素。无须新增消融 |
| _MLPBackboneBase/MLPBackboneBatchEnsemble | [作者两层定义](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/tabm.py#L1159-L1193) | 两层都带dropout；不是只第一层，末层也未被遗漏；没有额外LayerNorm或attention | 保留。没有证据支持加深/加宽/加归一化来修D |
| recent_baselines.py:204 tabm_head_loss | 作者README Training、paper模型独立成员训练 | logits(B,32,1)，标签扩展到(B,32)，默认mean同时平均行和成员；R29没有额外pos_weight | 正确；不得改成BCE(mean logits)或BCE(mean probabilities)后仍称原TabM训练 |
| route_batch/tabm_training.py:44/55 _log_probabilities/stable_scores | 作者README Inference | FP64 logsumexp(logsigmoid)分别计算mean p和mean(1-p)的对数，再相减；等于平均概率的log-odds，未把FP32概率先截断 | 保留。没有发现成员均值语义错误 |
| model_training/models.py:260–280 | [example](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb)；[paper固定配置](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/paper/exp/tabm-piecewiselinear/california/0-tuning.toml) | R29共享batch、FP32、全参数AdamW；paper配置独立成员batch、AMP及调参。共享batch是作者明示的实用简化，并非实现错误 | 保留一致来源表述；训练/选轮细节由同目录训练审查单独处理，不把改成paper配置当现成修复 |
| V2普通MLP照抄TabM lr=.002 | [paper普通MLP＋PLE配置](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/paper/exp/mlp-piecewiselinear/california/0-tuning.toml) | 作者该MLP配置学习率范围3e-5至1e-3，TabM范围1e-4至3e-3；通用参数一致不等于各模型均最优 | 当前新设计不增加MLP拟合；若以后比较，明确是共同优化配方下结构对照，不声称作者最佳MLP复现，不展开LR网格 |

## 5. 输入是否被PLE、缺失或FP32破坏

已有审计三seed一致：

| 部分 | typed唯一输入 | FP64唯一输入 | FP32唯一输入 | PLE后唯一输入 |
|---|---:|---:|---:|---:|
| F | 5810 | 5810 | 5810 | 5810 |
| V | 4607 | 4607 | 4606 | 4606 |
| 最终P | 10405 | 10405 | 10404 | 10404 |

原实验分析已确认FP32合并的是两条正常记录，没有新增异标签冲突。**在这些已有F/V/P审计中，PLE没有进一步把不同FP32输入合并。** 该结果不能扩展为未计算的D嵌入逐行结论。

F实际区间数为[48,36,41,45,46]；P为[48,36,43,45,47]；constant_bin_columns为空。不是分箱失败把五个数值全部变为常量，也不是所有数值落入同一个硬箱。

PLE编码本身在区间内连续线性；第一、最后分量保留范围外线性延伸，不是将所有范围外数值夹到常数。version B另有原数值的线性支路。因此不能把它描述为“离散分箱丢失时长”，也不能因有线性支路就断言任意外推可靠。[固定编码forward](https://github.com/yandex-research/rtdl-num-embeddings/blob/7381c5a4ed01253fbef9ea8cfed137f907265552/package/rtdl_num_embeddings.py#L656-L680)

缺失输入只有duration在既有支持中出现两种状态：F137条、V140条、P277条缺失；其余四字段缺失均为0。四个恒定missing one-hot给主干增加常量/零列，可被偏置吸收，没有增加区分信息。D也只有duration缺失（308条）；主要UDP误报中，仅PLE17有1条缺失，另两个seed没有。因此不能把修改missing编码当成主要误报修复。

## 6. R26小MLP与R29不是只差模型大小

| 项目 | R26 target/ports_zero8 MLP | R29 TabM＋PLE |
|---|---|---|
| 输入槽 | 原8个Zeek字段，两个端口固定0 | 五连续＋协议类别＋五missing类别 |
| 实际数值变换 | 八槽均signed log1p，再标准化；协议也被log压缩 | 五连续median/log1p/standard，协议one-hot |
| 缺失表达 | -1经signed log1p后保留数值哨兵 | median填补＋独立missing one-hot |
| 主干 | 8→128→64→1，GELU，零dropout | 96→512→512→32独立输出，ReLU，两个0.1 dropout |
| 集成 | 单模型 | 32成员共享大部分权重 |
| 参数 | 按源码结构计算9473 | 已有summary确认416704 |
| 目标 | 类别总质量平衡BCE | 自然行权成员BCE |
| 优化 | AdamW .001/.01，batch512，固定80轮 | AdamW .002/.0003，batch256，选轮后P重训 |

R26的通用_transform代码注释说协议/端口不压缩，但它引入的是NF字段名CATEGORICAL=(L4_SRC_PORT,L4_DST_PORT,PROTOCOL,L7_PROTO)。实际Zeek列名不匹配，故八列都进入signed log1p。support_diagnostic/model.py的模块注释和fit记录已经如实记载这一行为。它是遗留通用注释不准确，不是本次新发现训练执行偏离了已登记的实际流程。不能重新按注释解释R26。

R29比R26有更高容量和不同几何、正则与目标。R26部分D表现较好，只说明这些既有输入存在某种可行归纳偏好，不能单独证明GELU优于ReLU、128/64优于512/512、类别权重优于自然权重，或one-hot导致失败。

## 7. D卡在哪里：模型表达能力与监督可识别性必须分开

R29在F看到TCP正常5028条、UDP正常604条、UDP控制90条，没有TCP控制。P中TCP也全部正常。D控制则是TCP。于是“某些活动形态在UDP是控制，在TCP是否仍为控制”没有被R29训练标签直接确定。

首层为稠密矩阵，协议one-hot、数值PLE、missing一起进入ReLU主干。它既可以学习协议条件下的活动特征，也可以学习主要依赖协议的捷径。架构没有强制选择其中一种；作者一般表格成绩也不能替代这项识别。

这并不证明R29已经实际采用了协议捷径。R26也能表达协议依赖，却有较强D检出；必须把“存在合法但不希望的解”与“已证明最终模型使用这个解”分开。已有分数不能唯一反推出内部依赖，当前又不允许新模型干预评分。

主要UDP正常误报与TCP漏检仍是两类错误。取消显式协议对TCP迁移有清晰的归纳约束，但它可能削弱正常业务区分、增加UDP误报；不能把解决一类的动机充当另一类已解决的证据。

[TabM 2025原文](https://arxiv.org/html/2410.24210v3)明确给出MLP＋嵌入和TabM的有效性，同时将共享权重的泛化效果作为实证结果；[TabReD固定作者说明](https://github.com/yandex-research/tabred/blob/b5ef15b3749f30da7a1eb8fba21a5b54d706bf32/README.md)强调时间分布与真实特征条件。两者都不提供“采用该结构即可跨TCP/UDP控制实现”的保证。

## 8. 对拟议固定UNKNOWN协议槽的结构审查

在新拟合的两臂中，保留完全相同的96维模型：

- 感知臂：categorical[:,0]保留实际协议索引。
- 固定槽臂：训练和推理都将categorical[:,0]设为5，即UNKNOWN。
- 两臂其余五连续、五missing、预处理、PLE边界、完整初始state_dict、数据顺序和Dropout随机流一致。

这个做法比删掉6维更能保持参数和初始化规则相同，且不用修改作者内部网络。协议one-hot在固定槽臂为[0,0,0,0,0,1]：

1. 前五列输入恒0，对模型数据梯度没有贡献；其参数仍可能受AdamW衰减，但这些列不参与输出。
2. 第六列输入恒1，是经成员缩放和共享矩阵作用的常量通道，可视为附加偏置来源。不是六维全删。
3. 首层protocol块仍是6维，初始化chunks和参数总数不变。
4. 两臂参数初始状态相同，但输入不同，所以**初始函数值不保证相同**。不应把“同state_dict”写成“同初始预测”。
5. 若五连续和missing固定，仅改变传入的原协议码，固定槽臂输出在数学上不变。这个保证来自强制覆盖接口，不需要结果指标证明。
6. 对真实TCP/UDP流量整体不具有该不变性：时长、包数、字节数本身携带传输形态，模型仍可能从这些量识别协议。

实施位置应在新的模型输入策略层，避免原地修改两臂共用的categorical张量：复制或构建本臂输入后再覆盖protocol列。策略同时写入工件与加载/推理入口，不能仅在训练loader里覆盖。旧R29预处理与工件不修改。

这项配对判断能识别的是：**在共同新F_AB监督、共同训练规则下，允许显式协议输入相较于强制忽略该槽，对后续检出和误报有什么作用。** 新A监督、四组采样或选轮若同时改变，均为两臂共同条件；不能将新结果全部归为对旧R29失败的单因素解释。

## 9. 可落地的修改与验证清单

| 编号 | 修改位置 | 修改内容及依据 | 通过标准 |
|---|---|---|---|
| A01 | 新设计的模型声明、构造器元数据 | 明列TabM0.0.3、rtdl0.0.12、2×512、k32、ReLU、dropout0.1、PLE-B/16/48上限、normal、字段块；不只写“作者默认” | 静态展开与固定源码一致；两臂结构、参数名称/形状一致 |
| A02 | 新输入策略接口及工件schema | 分actual_protocol/fixed_unknown两策略；固定槽覆盖索引5，其他类别不变 | 非模型静态输入核对只允许协议列变化；训练/验证/部署共用策略；禁止原地污染共享输入 |
| A03 | 新运行构造流程 | 每seed建立一份完整初始参数状态，复制给两臂；保持相同结构与RNG调用形状 | 未来实现核验比较完整初始state_dict哈希；不以初始预测相等为标准；本次未构造或运行 |
| A04 | 新预处理/embedding登记 | 两臂共用F_AB数值拟合与PLE边界；实际区间数与最大区间数显式保存 | F以外输入不影响预处理；边界逐项一致；若最大区间48则参数416704；不把预处理差异混入协议比较 |
| A05 | 新损失与预测接口 | 原样保留成员BCE、平均概率语义、稳定FP64 log-odds | 源码及未来小规模合同核验对应作者定义；不用新阈值/拼接补模型结果 |
| A06 | 设计说明和报告 | 修正R29有无PLE归因、paper/example来源、R26实际协议预处理说明 | 不再写“只改PLE映射”“完整paper复现”“已定位protocol为唯一根因” |
| A07 | 采用与退出声明 | 两类错误分别验收；无新增架构、宽度、损失或seed网格 | 不能只改善TCP召回却隐藏UDP误报；不能只赢配对臂却退掉已有强模型能力；不合格即退出该设计 |

上述前五项定义的是未来实现合同，本次只作静态审查。当前已经确认关键作者语义正确，并把可检验的新架构约束收敛到显式协议槽；尚未用实验认定它能够修复D。
