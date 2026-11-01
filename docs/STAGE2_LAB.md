# 第二轮：受控字段预算实验

首轮基线已经在 RTX 4090D 上完成并通过工件与指标复算。现有结果可以保留，本轮回答的是：减少输入字段后，检测效果和资源开销如何变化？设计及本轮筛选规则见 [STAGE2_PROTOCOL.md](STAGE2_PROTOCOL.md)。不要重复执行首轮，也不要提前执行 `freeze` 或 `evaluate`。

## 需要准备什么

- 保留首轮的完整 `prepared-v1`，包括 `manifest.json`、`manifest.sha256`、`audit.json` 和四个分区。继续使用同一份文件，不重新 prepare。
- 保留原虚拟环境，更新仓库后先运行测试。代码和输出目录分开；运行期间不更新源码或配置。
- 36 次训练：RF、HGB、MLP × 42/8/16/24 原始字段 × 17/29/43 三个种子。树模型在 CPU，MLP 在指定 GPU。
- 所有训练结束后，顺序测量全部 36 个模型的 CPU 开销。计时阶段暂停其他训练和占用 CPU 的任务；保持相同电源模式。
- 预留 3GB 输出余量。这里只是保守准备量，不代表最终工件大小。软件环境另占空间。

## Windows PowerShell

以下路径沿用首轮示例。如果实验室实际位置不同，只修改前三个目录变量和 Python 路径。

```powershell
$labRoot = 'D:\TraceLightLab'
$repo = Join-Path $labRoot 'tracelight-ids'
$prepared = Join-Path $labRoot 'data\prepared-v1'
$stage2 = Join-Path $labRoot 'stage2-controlled'
$py = Join-Path $labRoot 'venv\Scripts\python.exe'
Set-Location -LiteralPath $repo

git status --short
# 如果存在未提交的修改，先保留修改，不执行 reset/clean。
git switch main
if ($LASTEXITCODE -ne 0) { throw '切换失败，先处理当前修改' }
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw '更新失败，停止' }
git rev-parse HEAD

function Invoke-Stage2Python {
    & $py @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: exit $LASTEXITCODE" }
}
Invoke-Stage2Python -m pip check
Invoke-Stage2Python -m pytest -q
Invoke-Stage2Python scripts/run_lab.py --data-dir $prepared --work-dir $stage2 --config configs/controlled-features.yaml --device cuda --with-study --dry-run
Invoke-Stage2Python scripts/run_lab.py --data-dir $prepared --work-dir $stage2 --config configs/controlled-features.yaml --device cuda --with-study
```

首次安装仍按 [START_HERE.md](START_HERE.md) 完成环境准备。本轮没有新增第三方依赖。

## Linux

```bash
cd /path/to/tracelight-ids
git status --short
git switch main && git pull --ff-only
PY=/path/to/venv/bin/python
"$PY" -m pip check && "$PY" -m pytest -q
"$PY" scripts/run_lab.py --data-dir /path/to/data/prepared-v1 --work-dir /path/to/stage2-controlled --config configs/controlled-features.yaml --device cuda --with-study
```

## 中断后继续

保留代码版本、配置、数据和输出原状，在同一启动命令后加 `--resume`。已完成训练会校验后跳过；未完成训练会保留现场并从该种子重新开始。CPU 测量按保存的随机顺序恢复，并验证已有计时文件校验和。换了代码、环境或计时参数则拒绝拼接，使用一个新输出目录完整运行。

```powershell
Invoke-Stage2Python scripts/run_lab.py --data-dir $prepared --work-dir $stage2 --config configs/controlled-features.yaml --device cuda --with-study --resume
```

## 完成标准与带回结果

成功结束应输出 `Development complete`，并得到：

- `runs/controlled-features-v1/summary.json`：36 项，且每个目录存在 `run.json`。
- `benchmarks/controlled-features-v1/plan.json`：随机计时顺序及设置；旁边为 36 个计时 JSON 和各自的 SHA-256 文件。
- `reports/study-controlled-features-v1-*.html` 及同名 JSON：离线交互研究报告。
- `reports/controlled-features-v1-*.html`：逐条验证预测查看器。
- `environment/doctor-*.json`：本机实际硬件与版本。

先直接打开研究 HTML 检查是否显示 **12 个配置、每组 3 个种子**，再将整个 `stage2-controlled` 文件夹压缩带回，同时记录 `git rev-parse HEAD`。保留首轮 prepared 数据及原始归档；无需把数据或模型上传公开代码仓库。不要只保留截图或汇总表，完整模型、预测和原始计时才能复核结论。

当前本机开发验证与实验室复现分开记录；本机 RTX 4050 的训练时间、CPU 延迟不能冒充 4090D 实验室结果。离线图中呈现的是验证集开发比较，阈值也来自该验证集，不是最终测试或线上保证。
