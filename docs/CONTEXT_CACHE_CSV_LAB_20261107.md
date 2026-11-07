# 原始 CSV、误报阈值与完整 KV／QCOC 最小对照

更新：2026-11-07。`ids.context_cache` 已接通原始 CSV → 支持集预处理 → 完整 KV／普通 QCOC → 各自阈值 → 开发评价 → 独立 CSV 部署包。本机用随机小型官方模型完成合成核查；真实权重与真实 IDS 比较由实验室执行。

本轮已选定 NF-ToN-IoT-v3，固定 P=8192、C=5000、E=20000、M=512 和 seed=20261107。[实验室完整命令](CONTEXT_CACHE_LAB_RUN_20261107.md)包含拉取固定标签、安装、官方数据／权重下载、`lab-prepare`、`lab-register`、预检、正式运行和回传。交付版共 **120 项合成测试通过**。下文说明通用 CSV 接口和阈值语义。

本入口沿用 [TabICLv2 2026 与 QCOC 2026 的适配器](CONTEXT_CACHE_IMPLEMENTATION_20261106.md)。两候选共享 checkpoint、支持行、预处理、FP32 精度和 CPU 评分配置，只改变 Stage 3 上下文缓存。一次配置只有一个 M 和一个 seed。新设计中的逐状态校准裕量几何、CRT 式 V-only 更新仍未加入；它们的依据与待检验主张见[主设计](CALIBRATION_AWARE_CONTEXT_DESIGN_20261106.md)。

## 数据角色和预处理

配置必须指定三个独立 CSV。入口读取全部登记行，不抽样、不重新划分、不按结果补候选。

| 文件 | 标签要求 | 用途 |
|---|---|---|
| `support` | 正常、攻击两类均有 | 拟合预处理；构建完整上下文；为普通 QCOC 提供支持查询 |
| `calibration` | 仅正常；出现攻击即报错 | 分别计算完整 KV 和 QCOC 的分数、登记各自阈值 |
| `evaluation` | 正常、攻击两类均有 | 用已登记阈值做逐行配对开发评价 |

每个文件登记路径、SHA-256、精确行数，checkpoint 也登记路径和 SHA-256。相对路径以**配置文件目录**为基准。预检检查跨角色的完整特征重复、声明的样本 ID 重叠和会话组重叠；角色内部重复特征保留并计数。不声明 `group_column` 时，会话关联检查记为未执行。以上检查覆盖声明的身份和完全重复输入，不把不同哈希视为总体独立性的证明。

CSV 使用 UTF-8（可带 BOM）、逗号分隔和标准引号。表头须唯一且无首尾空白，数值列须为有限数或空单元格。仅空单元格表示缺失，`NA`、`001` 等类别文本保持原样。特征按 `schema.features` 明确选择，额外列不进入模型；标签、ID、会话组和攻击族列不得作为特征。

- `numeric` 和 `categorical` 划分全部特征。省略两者时，全部按数值处理。
- 默认标签为正常 `0`、攻击 `1`。原始文本标签可登记 `"label_mapping": {"BENIGN": 0, "DDoS": 1}`，须列出实际出现的所有类别；未知、空标签报错。映射保存进预处理包。
- `id_column`、`group_column`、`family_column` 可设为 `null`。声明后，三个实验文件必须具有对应列；ID 在文件内唯一，ID／会话组不得为空。未提供 ID 时使用角色和行序号生成报告键。
- `family_column` 与标签列须不同。攻击族只用于逐族报告，不参与预处理、编译或阈值选择。

预处理复用[固定官方源码](https://github.com/soda-inria/tabicl/tree/d83091b845a8c5f7c087bd54964f755a9e560f45)的单视图链：`TransformToNumerical` → `UniqueFeatureFilter` → `PreprocessingPipeline(normalization_method="none", outlier_threshold=4.0, random_state=0)`。`none` 仍包含官方缩放和离群值处理。编码、填补、常量特征筛除与缩放统计只在支持集拟合；后续只调用 transform。输出顺序为官方类别列在前、数值列在后，再移除支持集常量列。没有特征／类别置换或多视图集成，不能把这条单视图结果写成官方默认集成成绩。

## 误报阈值登记

分数为原生二分类 `logit_attack - logit_normal`：先在 FP32 中相减，再提升到 float64 排序和比较。它不是概率；不使用历史接口的 `[0,1]` 限制或 `>=` 判定。

对 n 条正常校准分数、误报目标 α，取精确有理数计算的秩：

```text
r = ceil((n + 1) * (1 - alpha))
threshold = 第 r 小的正常分数（从 1 开始）
predicted_attack = score > threshold
```

严格 `>` 使阈值处并列分数保持正常。完整 KV 与 QCOC 使用**各自分数、各自阈值**，共享正常校准行和 α，不沿用完整 KV 阈值评价 QCOC。`threshold.json` 保存 n、r、阈值、边界并列数、校准 FP/FPR、实际分数哈希，以及模型／预处理／校准 CSV 的三个哈希绑定。

若 r>n，数学端点为正无穷，JSON 保存 `threshold: null`、`threshold_kind: "positive_infinity"`、`reject_all: true`，即所有输入均判正常。合成测试覆盖该端点；真实开发配置会在读取数据和模型前拒绝不足以形成有限阈值的行数。α=1% 时至少需 99 条正常校准记录，99 条只够取最大值，不能据此声称充分估计了尾部。

登记范围为 `development_calibration`，评价范围为开发比较。秩构造的统计解释依赖样本交换性等条件；本轮记录的是明确规则和实际误报计数，不将其写成部署总体 FPR 的高置信保证。旧已打开 test 不进入此入口。

## 实验室运行

依赖和官方源码检出沿用[固定版本说明](CONTEXT_CACHE_IMPLEMENTATION_20261106.md#3-固定依赖)及 [requirements-context-cache.txt](../requirements-context-cache.txt)。源码 commit 为 `d83091b845a8c5f7c087bd54964f755a9e560f45`，加载核对 15 个模型文件和 2 个预处理文件的 LF 规范化哈希。入口没有权重下载行为；实验室提供已核实来源和哈希的本地 `tabicl-classifier-v2-20260212.ckpt`。

自备三份 CSV 时，可使用[通用配置模板](../configs/context-cache-lab-minimal.example.json)并通过 `lab-register` 登记实际路径、哈希与行数。通用模板的字段和行数只是格式示例。本轮 NF-ToN-IoT-v3 使用另一个[固定模板](../configs/context-cache-nf-ton-v3.json)，由上述完整命令自动准备和登记，无需手工填写文件摘要。

以下命令在实验室仓库根目录运行。`local/context-cache-lab.json` 是填写后的配置，输出路径必须不存在。预检会读取登记 CSV、核对文件哈希和数据角色，但不反序列化 checkpoint、不拟合预处理、不运行模型。

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m ids.context_cache --tabicl-source local/vendor/tabicl lab-preflight --config local/context-cache-lab.json --output local/context-cache-preflight.json
python -m ids.context_cache --tabicl-source local/vendor/tabicl lab-compare --config local/context-cache-lab.json --output local/context-cache-lab-run
```

`build_device` 为 `cuda` 时，仅支持缓存构建和 QCOC 编译在 GPU；两个已导出候选依次冷加载到 CPU 做校准与评价。CPU 评分与部署统一使用 FP32、登记的 `threads` 和 `query_batch_size`。`query_chunk_size` 仅控制编译中间工作区，仍使用全部支持查询拟合 V。

运行目录主要工件：

| 工件 | 内容 |
|---|---|
| `protocol.json`、`data_audit.json`、`source_snapshot.json`、`environment.json` | 登记配置、数据角色检查、适配器源码哈希、实际环境 |
| `preprocessor.joblib` | 支持集拟合的官方预处理状态 |
| `full/`、`qcoc/` | 每个候选各一份可独立加载的四文件部署包 |
| `full_calibration.csv`、`qcoc_calibration.csv` | 每个候选的正常校准分数、标签、决策 |
| `paired_evaluation.csv` | 同行标签、双方分数／决策、修复／新增 FP/FN；可含攻击族 |
| `comparison.json` | 双方 TP/FP/FN/TN、TPR/FPR、逐族计数、配对变化、编译记录和成本计数 |

报告同时列出“修复漏检／新增漏检、修复误报／新增误报”和净 TP/FP，不能只用净值掩盖新增错误。完整部署文件字节包含模型、预处理、阈值和清单；原生张量字节另列。计时是一次运行的墙钟时间，不是 p95 延迟或 RSS 基准。

## 独立 CSV 预测

每个候选的四文件目录为：

```text
full/ 或 qcoc/
  model.pt
  preprocessor.joblib
  threshold.json
  deployment.json
```

`model.pt` 包含骨干、列缓存及完整／压缩 Stage 3 缓存；`preprocessor.joblib` 包含 schema、类别编码和拟合统计。包中不存原始支持行、支持标签数组或拟合查询。加载时核对文件哈希及模型—预处理—阈值绑定。joblib 文件按项目可信工件加载；部署使用相同固定源码和依赖环境。

独立进程只需上述目录和原始查询 CSV，无需支持文件、原始 checkpoint 或校准文件。查询 CSV 可以只有登记的特征列，ID／会话组／标签／攻击族均可省略；提供了标签时仍验证其编码，但预测不使用标签。

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m ids.context_cache --tabicl-source local/vendor/tabicl predict-csv --deployment local/context-cache-lab-run/qcoc --input local/queries.csv --output local/qcoc-predictions.csv
```

输出为 `row_id,score,predicted_attack`。冷加载恢复登记线程数和查询批大小，使预测沿用阈值登记时的 CPU 评分配置。

## 本机合成核查

本轮合成流程生成 32 条支持行、199 条正常校准行、80 条开发评价行，以及随机初始化的小型官方模型；M=4、α=1%。包含文本标签、缺失值、未见类别、字面类别 `NA`、仅支持集为常量的特征，以及互不重叠的 ID／会话组。

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m ids.context_cache --tabicl-source local/vendor/tabicl synthetic-csv-check --output local/context-csv-synthetic-new
```

核查分别复制完整 KV 与 QCOC 的四文件包到冷目录；新进程读取剥离全部元数据的特征 CSV，两者的分数和登记决策均逐位复现。旧数值张量入口继续保留。

针对性回归命令：

```powershell
$env:PYTHONPATH = "$(Resolve-Path src);$(Resolve-Path local/vendor/tabicl/src)"
python -m pytest -q tests/test_weighted_context_cache.py tests/test_context_compiler.py tests/test_tabicl_cache_adapter.py tests/test_context_cache_cli.py tests/test_context_csv_preprocessing.py tests/test_context_calibration.py tests/test_context_csv_workflow.py
```

合并回归为 **110 passed、0 skipped**；Ruff 通过。实际执行结果、依赖版本和源码哈希见[本轮证据](research_alignment_20261107/context_csv/README.md)。合成检查验证 CSV、阈值、编译与部署接口；真实模型的检测变化和资源取舍仍待实验室结果。
