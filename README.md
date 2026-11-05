# TraceLight IDS

面向网络入侵检测的轻量化机器学习研究工程：在明确的正常流量误报预算下，比较检测效果、可用字段、模型包和CPU流程成本，并保存逐行可复核工件。

**当前研究决定（2026-11-05）：** 24次基础screen及教师审计已回传。[机制重设计报告](docs/PROJECT_MECHANISM_REDESIGN_20261105.md)定位CIC24的主导漏报与训练范围外推有关；冻结裁剪能修复2,272行，也会产生模型相关的新误报。CIC full-F软目标接近hard，UNSW的普通KD仅保留为受控候选。最新[20项成果审查](docs/research_alignment_20261105/README.md)记录原文、作者代码、实际取舍和缺口。

**第九轮已完成验收：** 两环境16模型零拟合审计与正式成本已回传。[冻结标准验收结果](docs/INPUT_EXTRAPOLATION_RESULTS_20261105.md)：CIC指定模式机制复现，16/16成本通过，14/16无新FN；有C检出收益的候选均存在新增错误，不作默认部署替换。small24 warmup／direct因重登记新增1 FN停止候选。无需重跑本轮；下一步只保留UNSW原全字段small普通KD最小协议的研究依据，不自动启动训练矩阵、CIC KD／PLE或读取V/calibration/test。

此前已修复PLE零残差分支无梯度的配置错误；它不解释未使用TabM的round8结果。工程检查见[验证记录](docs/MECHANISM_REDESIGN_VERIFICATION_20261105.md)，真实结果以[第九轮验收](docs/INPUT_EXTRAPOLATION_RESULTS_20261105.md)为准。

**UNSW 普通 KD 最小方案已冻结：** [主协议](docs/UNSW_KD_MINIMAL_PROTOCOL_20261105.md)与[规格 JSON](configs/unsw-small-kd-minimal-20261105.json)仅保留一个 full-F HGB、原全字段 small warmup 和 H/K/S/M 四次续训，每臂40次全F更新，固定第40步。采用同尺度混合BCE、保持重复模式的固定置乱和类别均值软化对照；[11项近期成果审查](docs/research_alignment_20261105/unsw_kd_sources.md)及工件检查解释取舍。K须相对三个对照各净增至少34个C TP，并通过误报、逐族、集中度和成本条件。专用入口、合成端到端和打包流程已实现；实验室按[运行指南](docs/UNSW_KD_LAB.md)先预检，再执行四臂。旧预算迁移脚本不用于本轮。

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
