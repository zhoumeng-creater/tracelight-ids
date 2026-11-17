# TraceLight IDS

**2026-11-17 R30结果已复核：D检出改善，采用条件未通过**　[完整分析与处置](docs/MODEL_TRAINING_REDESIGN_RESULTS_20261117.md)。6次拟合、0次重训执行有效；三个主模型均B72/72、D1964/1964，但误报合规分别只有8/15、5/15、4/15单元，`adoption_pass=false`。协议对照已检出D1962，不能把共同训练修改带来的数百条增益单独归于协议屏蔽。当前不追加训练、调阈值或模型拼接；原运行标签及注册保持冻结，下方为历史交付记录。

**2026-11-17 模型与训练修改已实现并核验**　[实验室运行说明](docs/MODEL_TRAINING_REDESIGN_LAB.md) · [逐项实现与核验](docs/MODEL_TRAINING_REDESIGN_IMPLEMENTATION_20261117.md)。版本`model-training-redesign-r30-v1`，独立入口`RUN_MODEL_TRAINING_REDESIGN.ps1`：A+B四组分层批、协议输入唯一配对、纯排序选轮、F选中checkpoint直接交付，6次拟合、0次重训。96项合成检查和真实档案零模型预检通过；真实训练尚未执行。旧R29与下方设计记录保留，当前按本段实验室说明执行。

**2026-11-17 模型与训练设计已重审，尚未实现或开训**　[完整设计、修改位置和验证标准](docs/MODEL_TRAINING_REDESIGN_20261117.md) · [静态规格](configs/model-training-redesign-20261117/design.json)。保留作者TabM+PLE，改为A+B合并监督、四组分层批、纯排序选轮和F选中工件直接交付；主候选固定协议槽，只与保留协议的同合同版本作三seed配对。已确认原F/V无TCP正例、NLL驱动饱和后选轮、P重训增加75%更新并替换函数；这些缺口尚不能作为唯一因果归因。下一步是实现与合同核验；本轮没有训练、前向、新评分或阈值搜索。下方均为历史安排，不再推进阈值或模型拼接候选。

**2026-11-17 固定缓存验证已完成，候选未通过并关闭**　[完整结果与处置](docs/RETAINED_INCREMENT_RESULTS_20261117.md)。三seed最终判决完全相同：B72TP/12FP，D1694TP/18FP，全部15cell误报合规；但D增量0TP/3FP，未达D增益和强参照保持条件。主/独立实现逐身份核对一致。没有新训练、模型前向或阈值搜索；不扫描零下限、不挑种子、不自动恢复V2。下方设计为验证前冻结记录。

**2026-11-17 后续设计已收敛，新增训练为0**　[完整模型、判决、验收与退出设计](docs/RETAINED_INCREMENT_DESIGN_20261117.md) · [静态规格与工件绑定](configs/retained-increment-design-20261117/design.json)。冻结旧HGB∪规则保留通道，复用R29三份PLE，仅在原生零分界以上追加告警，原C阈值只可收紧；不重训、不扫描下限。尚未计算该候选的新判决，状态`zero_fit_candidate_specified_not_evaluated`。下一步限定为已有缓存的一次固定公式核查；V2七轨迹不再是默认下一步。下方为历史设计及结果记录。

**2026-11-17 失败机制核查完成，V2尚不具备开训依据**　[具体错误、阈值反证和设计处置](docs/R29_FAILURE_MECHANISM_RESOLUTION_20261117.md)：已定位短UDP正常尾部与约31秒TCP控制损失；新增A未补充UDP/443正常，四组权重尚无针对性修复证据。开发规则改为Q_dev正常校准→V_eval实测，保留原正式C/E及验收；名义1%不代表所有有限单元同时≤1%。候选规格状态为`candidate_mechanism_not_ready`，不启动模型实现或实验；下方定稿表述是被本次核查修订的历史记录。

**2026-11-17 完整模型与训练设计 V2已定稿，尚未实现或实验**　[完整设计与逐项修改/验收表](docs/MODEL_AND_TRAINING_DESIGN_V2_20261117.md) · [静态设计规格](configs/model-design-20261117/design.json)。A/B历史监督、共同工作点选轮、直接保留最佳checkpoint；TabM+PLE／同PLE普通MLP／HGB共7条轨迹，0次P重训；完整D继续只作评价。当前不运行新实验，旧R29脚本不能代替V2入口。下方为历史记录。

**2026-11-17 R29已完成，当前先修订设计**　14次拟合执行与结果复核通过，但七个新模型均未达到采用条件。见[问题定义与设计修订](docs/R29_PROBLEM_DEFINITION_AND_DESIGN_REVISION_20261117.md)：区分已证问题与原因假设，列明R26/R29多项同时变化，保留强参照和PLE消融。当前不启动新的训练、模型前向或阈值搜索；下方“尚未执行”为历史交付记录。

**2026-11-16模型与训练实现**　[实验室运行指南](docs/MODEL_TRAINING_LAB.md)：`model-training-r29-v1` 已接通语义输入、HGB/官方 TabM/去 PLE 消融、V 选轮、P 重训、全模型冻结、C 校准、逐单元比较与独立成本测量。最多 14 次拟合，三种子全部报告。[104项合成检查和真实输入入口校验通过](docs/model_training_implementation_20261116/validation.json)，真实训练尚未执行；下方设计阶段记录保留。

**2026-11-16模型与训练设计重审**　[完整决定](docs/MODEL_TRAINING_DESIGN_REVIEW_20261116.md)：对照官方模型，固定输入语义、训练/选轮/校准角色及结果分支。未来仅新增官方TabM和去数值嵌入消融，与HGB比较，最多14次拟合；本轮为设计审查，没有开始训练。旧权重补算只解释R26。

**2026-11-16训练诊断推进**　[新证据与路线决定](docs/TRAINING_DIAGNOSIS_DECISION_20261116.md)：端口置零造成63组旧来源内部异标签输入，涉及3657条记录；这些组未精确覆盖B/D攻击。已准备[8份冻结权重的训练支持诊断](docs/TRAINING_DIAGNOSIS_LAB.md)，0拟合，B/D分数复用；真实训练支持评分等待实验室执行。

**第二十六轮开发诊断**　[历史监督、源数据增量与端口依赖](docs/SUPPORT_DIAGNOSTIC_LAB.md)，版本`support-diagnostic-lab-v1`，入口`scripts/RUN_SUPPORT_DIAGNOSTIC.ps1`。TQH-B完整单元按时间分配，D只作冻结外评；20次新拟合、4份旧权重复用，保留同标签target-only、端口置零和已知强规则对照。全部材料已开，不称新算法或盲测。旧TabM等完成项不重跑。

**已有结果展示**　[逐日误报、攻击得失与就绪状态](docs/flow_replay/index.html)已接入第二十五轮修复结果，保留此前真实逐行回放；明确区分全日和原未开攻击子集。

**第二十五轮修复补算入口**　[就绪修复、逐类得失与排序上界指南](docs/READINESS_REPLAY_LAB.md)，固定版本为 `readiness-replay-lab-v1`，入口为 `scripts/RUN_READINESS_REPLAY.ps1`。只读复用第二十五轮完整缓存，42 个来源、五组学习器；修复 399 条与一小时共同就绪条件，多核评价并一次输出控制、下载的正常排名。新增拟合、模型前向、历史构建与下载均为 0。

**第二十五轮实验室入口**　[行为信息与本地阈值迁移指南](docs/BEHAVIOR_TRANSFER_LAB.md)，冻结版本为 `behavior-transfer-lab-v1`，入口为 `scripts/RUN_BEHAVIOR_TRANSFER.ps1`。覆盖两个来源的 42 个日期块，复用五组学习器与已有 CUDA 环境，比较 18 种固定判决，输出逐日误报、逐类新增和丢失、事件覆盖及实际资源利用。

**第二十三轮实验室批次**　[固定版运行指南](docs/RESPONSE_OBSERVABILITY_LAB.md) · [原始抓包与零计数的核验](docs/response-evidence/OBSERVABILITY.md)。检验回包观测质量、多核历史构建，以及 1% 误报预算下的扫描、控制和下载检出。

**第二十二轮实验室批次（2026-11-12）**　[固定版运行命令与判读标准](docs/CONTACT_EVIDENCE_LAB.md) · [数据与离线准备](docs/CONTACT_EVIDENCE_DATA.md) · [第二十、二十一轮离线回放](docs/flow_replay/index.html)。比较四种流历史输入、两种告警追加策略及独立的扫描和心跳场景。

**第二十一轮前向阈值复验已完成**　[结果与逐行复算](docs/FORWARD_THRESHOLD_RESULTS.md) · [固定版本与复现命令](docs/FORWARD_THRESHOLD_LAB.md)。已定位 CIC2018 校准分数的主机集中现象。IoT-23 两段新评价均未取得符合预设误报预算的扫描和 C&C 联合收益。执行版为 `forward-threshold-lab-v1.2`。

**第二十轮实验室入口（2026-11-11）**　[CIC2018 独立采集复验](docs/INDEPENDENT_CAPTURE_LAB.md)。在同一批已核对标签的记录上比较原 48 字段 HGB 与加入四项 60 秒历史的 HGB，同时测扫描、未见 Bot、正常误报与逐流成本。固定标签为 `independent-capture-lab-v1`。

**第十九轮实验室入口（2026-11-11）**　[过去 60 秒的流行为与低误报检测](docs/FLOW_HISTORY_LAB.md)。在旧开发集和前向新通信组上比较原始 48 字段与四项历史观测，复用旧分数；GPU 训练小型 MLP，多核 CPU 训练树模型，分别测量含历史状态的部署成本。固定标签为 `flow-history-lab-v1`。

**第十八轮复现入口（2026-11-09）** [TabICL 重放诊断与 TabPFN 一次准备指南](docs/ROUTE_FOLLOWUP_LAB.md)，固定标签 `route-followup-lab-v1`。只运行两个 TabICL 诊断和一个 TabPFN 开发集补测；不重跑已完成的 TabM 或新通信组确认。统一脚本准备独立环境与固定权重，保存完整 C/E 冷重放、逐行差值、判决翻转和独立 CPU 成本；失败也保留报告并打包。旧 run 全部文件在运行前后核对 SHA，新结果写入独立目录。下方批次指南为历史执行记录。

**当前实验室批次（2026-11-08）：** [一次完整运行指南](docs/ROUTE_BATCH_LAB_20261108.md)已准备，比较输入表示、监督训练、容量与学习率：24 条 TabM 轨迹、24 次全 P 重训，并列已有完整 KV／QCOC、官方 TabICL 1／8 视图与可选 TabPFN 2.5；冻结全部可用最终模型后自动进行新通信组确认。用户不设总时限，候选和轮数固定。135 项新增合成测试及 Ruff 通过，真实结果等待实验室运行。当前以[批次设计](docs/ROUTE_BATCH_DESIGN_20261108.md)为准，下方历史记录保留。

**实验室执行（2026-11-08）：** [Scanning标签粒度三臂指南](docs/SCANNING_LABEL_LAB_V1.md)已接通。使用冻结标签`scanning-label-lab-v1`，依次执行`RUN_SCANNING_LABEL.ps1`的Preflight、Run、Verify、Pack；复用第十四轮数据、预处理器和权重，只比较T0二分类、TF真实扫描三类、TP整组伪三类。二分类兼容、多类冷加载和三臂逐位复现已做合成验收，真实对照留在实验室。回传本轮ZIP与SHA，完整命令见指南；下述“待接入”为前一阶段状态。

**下一步设计（2026-11-08）：** [Scanning标签粒度对照](docs/SCANNING_NEXT_DESIGN_20261108.md)先保持48字段与8192条支持不变，比较二分类、真实拆出Scanning、同计数伪三类，统一判断全攻击与正常。标签会进入两级缓存，当前问题不能只归因于源端口或支持缺失。工件复算、23项纯数组检查及随机微型官方模型核查完成；正式多类适配器与runner待接入，真实实验仍在实验室。跨流历史作为后续条件分支，压缩研究先等待可用的完整KV基准。

**第十五轮结果（2026-11-08）：** [归档分析](../../实验归档/20261108_第十五轮Scanning机制/实验分析.md)已完成。源端口重编号／遮蔽使目标分数变化，但未恢复旧54条扫描，整体TP相对基线分别−1／−15；不能据此声称源端口机制被否证。停止这两个具体修复候选。脚本参数问题另以`scanning-mechanism-lab-v1.1`修复，不改变原实验协议，无需重跑；此前执行记录保留。

**当前实验（2026-11-08）：** [Scanning输入机制三臂](docs/SCANNING_MECHANISM_PROTOCOL_20261108.md)已接入正式入口。冻结上一轮P/C/E和预处理，只比较完整KV的原表示、源端口重编号、源端口遮蔽；先通过B0回放，再运行两项干预。本机288项合成回归通过，真实模型留在实验室。[预检—运行—验收—归档命令](docs/SCANNING_MECHANISM_LAB_V1.md)对应标签`scanning-mechanism-lab-v1`，当前不追加QCOC或训练矩阵。以下为此前阶段记录。

面向网络入侵检测的轻量化机器学习研究工程：在明确的正常流量误报预算下，比较检测效果、可用字段、模型包和CPU流程成本，并保存逐行可复核工件。

**当前算法主线（2026-11-06）：** [校准裕量敏感的上下文缓存压缩](docs/CALIBRATION_AWARE_CONTEXT_DESIGN_20261106.md)。采用 TabICLv2（2026）与 QCOC（2026）共享 KV 编译框架，借鉴 CRT（2025）的校准反馈；以每个源状态对攻击—正常边界差值的敏感度改变原型分配与中心，在固定槽数下保护低误报判定信息。新研究采用的方法、项目与文献限于 2025—2026 年。

**当前实现（2026-11-07）：** [原始 CSV 与完整 KV／QCOC 实验室入口](docs/CONTEXT_CACHE_CSV_LAB_20261107.md)已接通支持集独占的官方预处理、文本标签映射、两个模型各自的正常样本阈值、逐行错误配对和四文件独立部署包；110 项合成测试通过，双方在冷进程中从原始特征 CSV 逐位复现分数和决策。[配置模板](configs/context-cache-lab-minimal.example.json)需在实验室填写实际数据和 checkpoint 登记；校准裕量几何、CRT 更新及真实模型比较尚未执行。原生张量接口见[前一阶段说明](docs/CONTEXT_CACHE_IMPLEMENTATION_20261106.md)。

**设计证据：** [数学和求解器记录](docs/research_alignment_20261106/calibrated_context/README.md)保留支持与不利证据：六分区穷举显示准则差异，实际迭代案例尚未显示新方法优势。工程核查不替代真实 IDS 效果与正式资源测量。

**实验室交付（2026-11-07）：** [完整命令](docs/CONTEXT_CACHE_LAB_RUN_20261107.md)与[专用环境安装脚本](scripts/setup_context_cache_lab.ps1)已准备。首轮选择 NF-ToN-IoT-v3（2025），固定 P8192／正常 C5000／E20000、M512、seed20261107；自动按通信组准备 CSV、登记实际哈希和行数。固定权重 revision／SHA，交付版 120 项合成测试通过。执行标签为 `context-cache-lab-20261107-v1`；本机未执行真实对照。

## 历史设计与实验记录

以下保留此前已提交记录；旧候选不因仍在仓库中而恢复默认执行。

**第十一轮已验收（2026-11-06）：** [零拟合诊断分析](docs/UNSW_DIAGNOSTIC_RESULTS_20261106.md)确认9项有效性通过。anchor的hard/teacher余弦0.998321，不支持起点整体梯度对抗；K40仍有明显F拟合缺口，未分离优化与表示。H40单条p95 2.39274ms、HGB 3.20741ms，当前进程RSS约519/154MiB；HGB保留主要实际参照。已有依据保留预先限定H/K两臂优化机制对照，但本次未启动训练；不把软目标熵当作不可拟合噪声，也不改判第十轮KD失败。

**当前实验入口（2026-11-06）：** [H/K mini-batch 实验室指南](docs/UNSW_KD_MINIBATCH_LAB.md)已实现固定标签拉取、零推理预检、单次运行及自动打包回传。严格执行[冻结协议](docs/UNSW_KD_MINIBATCH_PROTOCOL_20261106.md)：两臂各40遍F、batch=1024，共6,720次更新；同一原anchor、教师q和40份共同排列，只评分两个终点。质量门槛与交互门槛分别报告，不增加实验组合。本机未启动真实训练。第十一轮零拟合诊断已完成，[旧指南](docs/UNSW_DIAGNOSTIC_LAB.md)保留为复现记录。

**四臂后的研究重定（2026-11-05）：** 用户确认没有模型体积、内存或延迟硬上限，按效果和实际成本比较。[最新机制研究](docs/UNSW_KD_FAILURE_RESEARCH_20261105.md)撤销“必须压成35KB small”的默认前提，将现成HGB与CatBoost-reference列为强实际参照。F缓存显示教师函数在同一混合目标上的损失0.06264，当前K最后记录0.12454；原warmup和续训同为40遍F，但更新数为3360与40，尚不能判定容量上限。[最小验证规格](docs/research_alignment_20261105/kd_failure_research/最小机制验证规格_20261105.md)及原验收条件保持，普通KD不自动追加训练。

**当前研究决定（2026-11-05，第十轮已验收）：** [UNSW普通KD四臂结果](docs/UNSW_KD_RESULTS_20261105.md)确认实验完整、目标与预算正确，但冻结候选未通过：C上K相对H仅净增5 TP，低于34条门槛；K相对S/M的+63/+62不能替代H对照。R净TP−1、pAUC略降，Analysis/DoS各−1；六块成本中一块p95比值1.400056超过1.25。停止该候选，不追加训练或调整原验收规则。

**此前机制依据：** 24次基础screen及教师审计已回传。[机制重设计报告](docs/PROJECT_MECHANISM_REDESIGN_20261105.md)定位CIC24的主导漏报与训练范围外推有关；冻结裁剪能修复2,272行，也会产生模型相关的新误报。CIC full-F软目标接近hard。[20项成果审查](docs/research_alignment_20261105/README.md)记录原文、作者代码、实际取舍和缺口。

**第九轮已完成验收：** 两环境16模型零拟合审计与正式成本已回传。[冻结标准验收结果](docs/INPUT_EXTRAPOLATION_RESULTS_20261105.md)：CIC指定模式机制复现，16/16成本通过，14/16无新FN；有C检出收益的候选均存在新增错误，不作默认部署替换。small24 warmup／direct因重登记新增1 FN停止候选。无需重跑本轮；下一步只保留UNSW原全字段small普通KD最小协议的研究依据，不自动启动训练矩阵、CIC KD／PLE或读取V/calibration/test。

此前已修复PLE零残差分支无梯度的配置错误；它不解释未使用TabM的round8结果。工程检查见[验证记录](docs/MECHANISM_REDESIGN_VERIFICATION_20261105.md)，真实结果以[第九轮验收](docs/INPUT_EXTRAPOLATION_RESULTS_20261105.md)为准。

**UNSW 普通 KD 冻结方案与已完成运行：** [主协议](docs/UNSW_KD_MINIMAL_PROTOCOL_20261105.md)与[规格 JSON](configs/unsw-small-kd-minimal-20261105.json)保留一个 full-F HGB、原全字段 small warmup 和 H/K/S/M 四次续训，每臂40次全F更新，固定第40步。采用同尺度混合BCE、保持重复模式的固定置乱和类别均值软化对照；[11项近期成果审查](docs/research_alignment_20261105/unsw_kd_sources.md)解释取舍。四臂均已完成；[运行指南](docs/UNSW_KD_LAB.md)保留为复现记录，当前执行决定以[第十轮验收](docs/UNSW_KD_RESULTS_20261105.md)为准。

**已有证据与保留结果：** [v3复核](docs/RISK_V3_RESULTS_20261104.md)未支持候选胜出；[v4复核](docs/UPGRADE_V4_RESULTS_20261104.md)覆盖16个学生、32个未启动条件，不能给CIC新KD效果结论。[根本设计复盘](docs/RESEARCH_ROOT_CAUSE_REVIEW_20261104.md)说明重构原因；旧源码、配置、失败和[零拟合审计](docs/THRESHOLD_AUDIT_LAB.md)均保留。历史V已被开发使用，不能作为新的独立确认。

**历史基线：** 前五轮及正式题目保留。第四轮CIC普通MI24小MLP曾相对全字段同款模型减少43.73%部署包、CPU单条p95约7.01→2.28ms；这些是旧验证协议结果，不能当成本轮KD效果。[第五轮协议](docs/STAGE5_PROTOCOL.md)的最终test已经打开，不再用于新模型选择。近期原文和固定作者源码分别见[关系KD审查](docs/research_alignment_20261104/new_design_review.md)、[pAUC审查](docs/research_alignment_20261104/direct_pauc_source.md)及[成果全景](docs/landscape_20261103/README.md)。

## 已实现

- UNSW-NB15 的 42 个原始输入字段校验、来源 SHA-256、标签与重复审计；相同输入记录整组划分，防止内部训练/验证/校准泄漏。
- 训练、验证、校准、测试四分区；填补、标准化、类别编码、互信息特征选择只拟合训练部分。
- Logistic Regression、Random Forest、HistGradientBoosting 和小型 PyTorch MLP；全字段基线及 8/16/24 原始字段实验配置。
- 正常误报预算下的保守阈值、攻击召回、Macro-F1、AP、ROC-AUC、按攻击类型的检测召回及 Wilson 区间。
- 配置、代码、数据、环境、预测、模型、训练历史的记录与完整性验证；重复执行不覆盖实验。
- 分开测量 CPU 单条推理延迟与批量吞吐；HTML 离线报告可比较模型并核查逐条预测。
- 受控字段预算配置、按保存顺序执行的整套 CPU 基准、多种子研究汇总与可交互离线对照图。计时文件绑定具体模型、数据和源码。
- 训练集内部按精确特征组交叉拟合 HGB 教师软标签；同结构小 MLP 的硬标签/蒸馏对照，统一早停规则。部署只需要学生模型。
- 三轮 CPU 测量、同种子配对的恢复检出与新增漏检、教师额外训练成本、运行前协议/源码快照，以及校验后打包。
- 分组重采样 MI、数值相关冗余消融、跨模型复用的训练内选择池、独立外层选集稳定性诊断。
- CNS2022 Improved CICIDS2017 独立 82 字段契约；固定来源哈希、Attempted 标签规则、会话与重复记录联合分组、磁盘分块准备。
- 第四轮一键运行及可核验恢复；两环境各自报告选择器配对、家族表现、实际部署成本，并自动生成结果 ZIP。

## 本地快速验证

Python 3.11 / 3.12；建议 3.12。先在代码目录之外建立虚拟环境，再进入本仓库：

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q
python -m ids doctor --output local/doctor.json
python -m ids synthetic --output local/raw
python -m ids prepare --train-csv local/raw/UNSW_NB15_training-set.csv --test-csv local/raw/UNSW_NB15_testing-set.csv --output local/prepared --data-kind synthetic
python -m ids suite --data-dir local/prepared --config configs/smoke.yaml --output-root local/runs --device cpu
python -m ids report --run-root local/runs/smoke --output local/report.html
```

`local/report.html` 可直接用浏览器打开。其指标来自本次实际运行，页面显著标明 **synthetic**；不能引用为 UNSW-NB15 性能结果。`local/` 已忽略，不提交到仓库。

实验室 RTX 4090D 24GB 的环境准备、CUDA 安装、运行和结果带回步骤见 [LAB.md](docs/LAB.md)。树模型和 LR 使用 CPU；MLP 才使用指定 GPU。小型表格模型未必需要占满 24GB 显存，先完成可信对照。

## 真实数据与实验顺序

按 [DATA.md](docs/DATA.md) 获取官方训练/测试 CSV。若使用固定版本镜像，必须显式选择镜像下载并传 `--data-kind mirror`，不能标成已核验官方数据。原始数据和模型权重不随代码再分发。

```powershell
python -m ids prepare --train-csv /path/to/UNSW_NB15_training-set.csv --test-csv /path/to/UNSW_NB15_testing-set.csv --output /path/to/prepared --data-kind official
python -m ids suite --data-dir /path/to/prepared --config configs/baselines.yaml --output-root /path/to/runs --device cuda
```

1. `prepare` 读取原始文件，审计并固定划分。官方测试集在此时只参与完整性/分布/重复审计，不参与拟合或调参。
2. `suite` 只读取 train 和 validation。MLP 依据 validation loss 早停；验证集阈值属于探索记录。
3. 依据验证结果选定模型，记录选择理由和最终比较组，然后 `freeze --run-dir ...`。
4. `evaluate --run-dir ... --data-dir ...` 单独用 calibration 确定阈值，再打开 test。默认拒绝覆盖已开始的最终评估。
5. 多次实验、跨种子和特征预算的推断边界见 [PROTOCOL.md](docs/PROTOCOL.md)。测试集 FPR 可能超过校准预算，这本身是需要报告的结果。

最终评估的命令门控是防误操作机制，并非阻止研究者手动读取文件的安全隔离。请在正式实验前保留方案与代码版本。当前只实现精确重复分组，不能宣称解决了近重复、时间泄漏或跨域泛化问题。

## 实验工件

第五轮附带基于第四轮 validation 的离线演示。固定 seed 17，分别展示每个环境的候选和参照。回放页可查看同一条流量在两个模型中的分数、阈值、判定与原始字段，并导出特征 CSV。在仓库根目录使用原有 Python 环境，可对符合对应 schema 的新流量特征表生成真实的本机推理报告：

```powershell
python -m workflows.stage5.demo predict --experiment 'D:\TraceLightLab\stage4-stable-selection\experiment' --dataset unsw --input 'flows.csv' --output 'inference.html'
```

UNSW 输入 42 个原始字段；CIC 用 `--dataset cicids` 及该数据集的 82 字段。生成的网页用第四轮开发阈值解释两个模型的分数；第五轮最终成绩单独保存在最终评估报告。

每个 run 包含 `config.json`、`code.json`、`environment.json`、`dataset.json`、`run.json`、模型、完整验证预测与指标。报告的散点图和明细固定取稳定 ID 的前 200 条，汇总指标使用完整验证集。分数不是经过概率校准的概率；没有输入来源 IP/时间的表格不会生成虚假的拓扑图或时间轴。

```powershell
python -m ids benchmark --run-dir /path/to/run --data-dir /path/to/prepared --output /path/to/benchmark.json --threads 1
python -m ids predict --run-dir /path/to/run --input /path/to/unlabelled.csv --output /path/to/predictions.csv
```

`predict` 接受相同 42 字段的无标签 CSV；无标签输入只给出预测，不计算准确率。仅加载自己生成或信任的 `model.joblib`，因为 joblib/pickle 可执行代码。基准范围为已经载入内存的 DataFrame 到分数，包含预处理，不包含 CSV 解析、网络采集或 UI。

## 分支与后续工作

- `main`：测试通过、可复现的稳定版本。
- `dev`：特征选择、实验配置、报告和后续系统开发。通过测试后经评审合入 `main`。

第五轮的两个同种子比较分别是 UNSW 全字段小 MLP 对全字段 RF、CIC 普通 MI24 小 MLP 对全字段小 MLP。两份数据独立训练与评估，结论限定在各自的数据环境内。报告保留实际测试误报数、总体和逐家族召回、同种子差值及原工件的部署成本。

核心依赖固定为当前验证版本；这不表示它们是最新版本。源码采用 MIT，数据遵循各自条款。研究来源和实现依据见 [REFERENCES.md](docs/REFERENCES.md)。
