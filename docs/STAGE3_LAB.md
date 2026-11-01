# 第三轮实验室操作：全字段小 MLP 与蒸馏

本轮保留已有 42 字段输入，完成 4 种配置 × 3 个 seed 的 12 个主模型，以及 9 次 train 内折外 HGB 教师拟合。小 MLP 的 hard/KD 对照与新 120 epoch 大 MLP 参照见 [STAGE3_PROTOCOL.md](STAGE3_PROTOCOL.md)。前两轮结果继续保留；真实 prepared 数据不重新生成，最终 calibration/test 不在本轮使用。

公开示例统一使用变量 `$labRoot`，请改为实验室实际根目录。仓库、既有虚拟环境、prepared 数据的子目录若不同，同步调整相应变量；不要迁移或覆盖已有实验来迎合示例路径。

本文所列入口已按第三轮实现核对：运行前仍以所更新版本的 `--help` 和 `--dry-run` 确认；尚未通过 CPU/CUDA smoke 的版本不开始真实实验。接口校验通过只表示工程流程可用，不表示研究结果已完成。

## 1. 更新代码与检查环境

以下 Windows PowerShell 命令在同一会话中顺序执行。每一步失败就停止并保留错误输出。

```powershell
$labRoot = 'D:\TraceLightLab'
$repo = Join-Path $labRoot 'tracelight-ids'
$prepared = Join-Path $labRoot 'data\prepared-v1'
$stage3 = Join-Path $labRoot 'stage3-distillation'
$py = Join-Path $labRoot 'venv\Scripts\python.exe'
Set-Location -LiteralPath $repo

git status --short
# 有本地修改时先保留并核对，不使用 reset/clean 丢弃修改。
git switch main
if ($LASTEXITCODE -ne 0) { throw '切换失败；先保留并处理当前修改' }
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw '更新失败；停止' }
git rev-parse HEAD

function Invoke-Stage3Python {
    & $py @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: exit $LASTEXITCODE" }
}

Invoke-Stage3Python -m pip check
Invoke-Stage3Python -m pytest -q
Invoke-Stage3Python scripts/run_stage3.py --help
Invoke-Stage3Python scripts/package_run.py --help
Invoke-Stage3Python -m ids doctor --require-cuda --output (Join-Path $labRoot 'stage3-preflight.json')
```

本轮继续使用原环境；`pip check` 失败时先核对锁定依赖，不用无约束的批量升级“修好”环境。若版本更新明确新增依赖，按仓库锁定的安装说明处理，再重复检查。GPU 型号、显存、CPU 和库版本以此次 doctor 为准，本机与实验室结果分别保存。

真实数据目录应已有 `manifest.json`、`manifest.sha256`、`audit.json` 及 train/validation/calibration/test 四个 Parquet 文件。入口需核对数据身份；不要手改清单、移动几份数据凑目录，或把合成 prepared 路径传给真实配置后写成研究结果。

启动后保持源码、配置和协议不变；不在训练中 pull、切分支或更改 loss。使用本地 SSD，并为完整模型、OOF 教师和压缩包预留空间。所需容量以实际工件为准，不把预留空间当作测得的部署需求。

## 2. 新配置的 CPU 与 CUDA 合成 smoke

第三轮增加了 OOF 与软目标训练，旧 smoke 通过不能替代新路径验证。使用独立的新合成目录；下面的 prepare 只针对新合成数据，不触碰真实四分区。

```powershell
$smokeStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$smokeRoot = Join-Path $labRoot "stage3-smoke-$smokeStamp"
$smokeRaw = Join-Path $smokeRoot 'raw'
$smokeData = Join-Path $smokeRoot 'prepared'
$smokeCPU = Join-Path $smokeRoot 'cpu'
$smokeCUDA = Join-Path $smokeRoot 'cuda'

Invoke-Stage3Python -m ids synthetic --output $smokeRaw --n-train 600 --n-test 120
Invoke-Stage3Python -m ids prepare --train-csv (Join-Path $smokeRaw 'UNSW_NB15_training-set.csv') --test-csv (Join-Path $smokeRaw 'UNSW_NB15_testing-set.csv') --output $smokeData --data-kind synthetic

Invoke-Stage3Python scripts/run_stage3.py --data-dir $smokeData --work-dir $smokeCPU --config configs/distillation-smoke.yaml --device cpu --benchmark-rounds 1 --dry-run
Invoke-Stage3Python scripts/run_stage3.py --data-dir $smokeData --work-dir $smokeCPU --config configs/distillation-smoke.yaml --device cpu --benchmark-rounds 1
Invoke-Stage3Python scripts/run_stage3.py --data-dir $smokeData --work-dir $smokeCUDA --config configs/distillation-smoke.yaml --device cuda --benchmark-rounds 1
```

检查两种设备路径都完成硬标签/蒸馏训练、OOF 分数覆盖与重复组隔离检查，并产生合成数据报告。smoke 用缩短的专用配置及单轮计时，分数不能作为论文证据。CPU/CUDA 使用不同输出目录；不得以 CPU 回退结果声称 CUDA 已通过。

## 3. 启动真实第三轮

确认 `$prepared` 仍指向前两轮同一份真实 prepared 数据，`$stage3` 是新输出目录。默认主配置为 `configs/distillation.yaml`，此处显式写出，便于复核。

```powershell
Invoke-Stage3Python scripts/run_stage3.py --data-dir $prepared --work-dir $stage3 --config configs/distillation.yaml --device cuda --dry-run
Invoke-Stage3Python scripts/run_stage3.py --data-dir $prepared --work-dir $stage3 --config configs/distillation.yaml --device cuda
```

dry-run 应显示第三轮流程，不包含真实数据 prepare、freeze 或 evaluate。运行包含 12 个主模型、9 次辅助教师拟合及后续 3 个完整 CPU 测量轮次；真实配置默认三轮，不传 `--benchmark-rounds 1` 降低它。训练期间 MLP 使用指定 CUDA，HGB 使用 CPU。

CPU 计时开始前停止其他训练和占用 CPU 的任务，保持相同电源模式。每轮测量全部 12 个主模型，CPU threads=1；辅助教师不参与部署排名。应分别记录 measurement_round=1/2/3、planned_rounds=3，顺序种子为 20261101/20261102/20261103。所有轮次保留，不能只交最快一轮。

训练和计时可能跨较长时间，保留控制台日志、错误和重试现场。不要提前判断“某个 seed 很差”后删掉或重新找种子；三种子结果完整进入报告。

## 4. 中断后恢复

保留源码、配置、prepared 与输出目录原状，在原命令上增加 `--resume`：

```powershell
Invoke-Stage3Python scripts/run_stage3.py --data-dir $prepared --work-dir $stage3 --config configs/distillation.yaml --device cuda --resume
```

恢复须校验完整 OOF 包、主 run 与已完成计时的绑定和哈希；不能仅凭存在一个模型或 CSV 就跳过。源、环境或配置不匹配时，保留报错并调查，不能删校验文件来强行继续。未完成尝试应保留，是否从折/主 run 重启以入口实际恢复日志为准，不把模型训练从头恢复误称为从某个 batch 断点继续。

已完成工作会按校验结果复用；计时恢复可能发生在新的时段，报告保留真实时间，不能声称全程处于同一个不间断独占时段。更新研究配置后应新建输出目录，不能与原配置结果拼成一轮。

## 5. 完成检查与阅读结果

完整交付应同时包括：

- 12 个主 run：每个完成标记、模型、配置、来源、环境、源码、验证预测/指标/阈值、训练历史均存在且可校验。
- 3 套 OOF 软目标，每套 3 个辅助教师拟合：折划分、覆盖/组隔离审计、每折模型和预处理、行 ID/分数、配置、成本与哈希齐全。
- 3 轮 × 12 个主模型的 36 份 CPU 测量，以及每轮顺序计划、原始时间观察值与校验文件。报告明确区分轮次，没有遗漏或覆盖。
- 主研究 HTML/JSON：4 个配置、各 3 seed；能查看 KD−small-hard 和 small-hard−large-hard 的配对差，全部家族（重点 Exploits、Fuzzers）、模型大小与多轮 CPU 数据。
- 协议、配置和代码快照，以及本轮候选判定。KD 未达到条件也是完整有效结果；不能以模型分数不够高判定整个实验运行失败。

自动报告不是唯一证据。抽查原始计数、分母、三个 seed 和三轮计时都在；若界面汇总缺项，保留工件并修复报告，不补造数据。验证集阈值和筛选结果均标为 development，不叫最终测试或生产误报保证。

真实配置的主要产物路径如下：

```text
stage3-distillation/
  STAGE3_PLAN.json
  STAGE3_COMPLETE.json
  protocol/                         # 源码、配置、协议快照
  environment/doctor.json
  runs/distillation-dev-v1/          # 12 个主 run
    teachers/                       # 3 套 OOF 及 9 次辅助教师拟合
  benchmarks/round-1/
  benchmarks/round-2/
  benchmarks/round-3/
  reports/validation-records.html
  reports/distillation-study.html
  reports/distillation-study.json
```

入口最后核验覆盖和文件哈希后写入 `STAGE3_COMPLETE.json`。这个标记说明开发工作流完成，不表示 KD 达到收益条件或最终测试完成。不能只保留截图或摘要表。

## 6. 打包带回

真实运行与核验完成后，调用第三轮打包入口；不要从公开仓库下载历史模型充当本次产物。

```powershell
$archiveStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$archive = Join-Path $labRoot "TraceLight-Stage3-$archiveStamp.zip"
Invoke-Stage3Python scripts/package_run.py --work-dir $stage3 --output $archive
Get-FileHash -LiteralPath $archive -Algorithm SHA256
```

打包工具先核对 `STAGE3_COMPLETE.json` 中的文件哈希，再写入整个本轮新输出、全部 OOF 教师与软目标、12 个主模型、36 份计时、报告、配置/代码/协议快照及 `MANIFEST.sha256`，并重新读取 ZIP 校验其中内容。成功时输出 `Verified ZIP`，ZIP 旁生成 `.zip.sha256`；两者一起带回。保留原始输出，在接收端完整核验前不要删除。打包工具不自动带走原始 prepared 大数据；实验室继续保留同一份 prepared，接收端如缺少它，另行复制并对照清单核验，不能重新 prepare 后假定逐字节相同。

打包会拒绝完成后新增、未列入封存清单的非 `.log` 文件。手工补充说明放在 work 目录之外，单独随实验包保存；不要修改完成清单或把说明塞进已封存的模型目录。

公开 GitHub 保存代码、配置、测试和文档。真实数据、模型、软目标、实验日志及带个人路径的环境/归档记录不因发布代码而自动公开；实验包按研究归档流程带回。最终部署只需要选定学生及其预处理，研究包仍保留教师以便复核。

## 7. Linux 对应入口

在已准备好并通过检查的环境中，替换目录变量即可使用同一入口。合成 CPU/CUDA smoke 与 Windows 相同，仍须使用专用 smoke 配置和不同输出目录。

```bash
set -euo pipefail
lab_root="/path/to/TraceLightLab"
repo="$lab_root/tracelight-ids"
prepared="$lab_root/data/prepared-v1"
stage3="$lab_root/stage3-distillation"
lab_python="$lab_root/venv/bin/python"
cd "$repo"
"$lab_python" -m pip check
"$lab_python" -m pytest -q
"$lab_python" -m ids doctor --require-cuda
"$lab_python" scripts/run_stage3.py --data-dir "$prepared" --work-dir "$stage3" --config configs/distillation.yaml --device cuda --dry-run
# 完成上述两条合成 smoke 路径后才启动真实实验。
"$lab_python" scripts/run_stage3.py --data-dir "$prepared" --work-dir "$stage3" --config configs/distillation.yaml --device cuda
"$lab_python" scripts/package_run.py --work-dir "$stage3" --output "$lab_root/TraceLight-Stage3.zip"
sha256sum "$lab_root/TraceLight-Stage3.zip"
```

Linux 的硬件、驱动、环境和计时条件另行记录，不能与 Windows 的绝对延迟混为同机测量。
