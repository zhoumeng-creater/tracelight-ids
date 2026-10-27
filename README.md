# TraceLight IDS

Reproducible lightweight machine learning for network intrusion detection, with an offline evidence viewer.

面向网络入侵检测的轻量化机器学习实验工程。目标是在明确的正常流量误报预算下，比较检测效果、原始字段数、模型体积与 CPU 推理开销。每条展示记录都能追溯到数据划分、配置、模型和决策阈值。

**当前阶段：可运行的研究基线与验证工具。尚无经过最终封存测试的论文结论。** 合成数据只用于工程测试；公开镜像数据保留单独的来源标记。本工程不把历史 CSV 回放称为实时网络监测。

## 已实现

- UNSW-NB15 的 42 个原始输入字段校验、来源 SHA-256、标签与重复审计；相同输入记录整组划分，防止内部训练/验证/校准泄漏。
- 训练、验证、校准、测试四分区；填补、标准化、类别编码、互信息特征选择只拟合训练部分。
- Logistic Regression、Random Forest、HistGradientBoosting 和小型 PyTorch MLP；全字段基线及 8/16/24 原始字段实验配置。
- 正常误报预算下的保守阈值、攻击召回、Macro-F1、AP、ROC-AUC、按攻击类型的检测召回及 Wilson 区间。
- 配置、代码、数据、环境、预测、模型、训练历史的记录与完整性验证；重复执行不覆盖实验。
- 分开测量 CPU 单条推理延迟与批量吞吐；HTML 离线报告可比较模型并核查逐条预测。

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

每个 run 包含 `config.json`、`code.json`、`environment.json`、`dataset.json`、`run.json`、模型、完整验证预测与指标。报告的散点图和明细固定取稳定 ID 的前 200 条，汇总指标使用完整验证集。分数不是经过概率校准的概率；没有输入来源 IP/时间的表格不会生成虚假的拓扑图或时间轴。

```powershell
python -m ids benchmark --run-dir /path/to/run --data-dir /path/to/prepared --output /path/to/benchmark.json --threads 1
python -m ids predict --run-dir /path/to/run --input /path/to/unlabelled.csv --output /path/to/predictions.csv
```

`predict` 接受相同 42 字段的无标签 CSV；无标签输入只给出预测，不计算准确率。仅加载自己生成或信任的 `model.joblib`，因为 joblib/pickle 可执行代码。基准范围为已经载入内存的 DataFrame 到分数，包含预处理，不包含 CSV 解析、网络采集或 UI。

## 分支与后续工作

- `main`：测试通过、可复现的稳定版本。
- `dev`：特征选择、实验配置、报告和后续系统开发。通过测试后经评审合入 `main`。

后续重点：多种子正式基线、8/16/24 字段预算、选择稳定性、消融与跨域验证；展示进一步加入受控流量回放、不同方法同屏对照与失败案例。跨域数据、真实流量采集、模型解释和这些演示尚未实现，不能当作已有成果。

核心依赖固定为当前验证版本；这不表示它们是最新版本。源码采用 MIT，数据遵循各自条款。研究来源和实现依据见 [REFERENCES.md](docs/REFERENCES.md)。
