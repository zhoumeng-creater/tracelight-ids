# 实验室第一次运行：从 GitHub 克隆到首轮实验

**如果已完成第二轮：使用 [第三轮小型 MLP 与蒸馏指南](STAGE3_LAB.md)。仅完成首轮则使用 [第二轮指南](STAGE2_LAB.md)。复用原 prepared 数据和环境，不重复已完成实验。**

这份指南可独立使用。首次目标是：确认环境可用、完成 CPU/CUDA 合成数据测试、准备真实数据、跑完 **4 种模型 × 3 个种子 = 12 次开发基线**，保存可核验的结果。这里不会运行最终测试评估。

仓库提供代码、锁定依赖、数据下载脚本、配置、测试和报告生成器。**原始数据、训练好的模型、个人电脑上的报告不随 Git 克隆下载**；以下步骤会在实验室生成它们。无需复制个人电脑上的虚拟环境。

## 0. 开始前确认

- Windows 或 Linux；Git 和 **Python 3.12**（含 venv/pip）已可用。
- NVIDIA 驱动可用；先运行 `nvidia-smi`，核实 GPU 型号和显存。预期是 RTX 4090D 24GB，记录实际结果。
- 能访问 `github.com`、`raw.githubusercontent.com`、`pypi.org`、`files.pythonhosted.org`、`download.pytorch.org`。安装依赖和下载数据需要网络；完成后训练和 HTML 报告可在本地运行。
- 选一个有写权限的本地 SSD 目录。建议预留至少 10GB 可用空间用于环境、缓存和首轮产物；这是准备余量，不是测得的最低需求。不要直接在同步目录或公共环境里安装依赖。
- 每一步失败就停在该步，保留完整报错。Python/Git 缺失、驱动或权限问题需要先在实验室解决，clone 不会自动安装这些系统前置条件。

只执行与你的操作系统对应的一节。路径均为示例，首次创建使用新目录。后续继续实验不需要重新下载、prepare 或从头安装。

## 1. Windows：PowerShell 按顺序执行

### 1.1 前置检查和克隆

```powershell
git --version
python --version
nvidia-smi

# 修改为实验室实际存在的 Python 3.12 路径。
# 如果 python --version 已是 3.12，可保留这一行。
$python312 = (Get-Command python).Source
& $python312 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Need Python 3.12"'
if ($LASTEXITCODE -ne 0) { throw '先选择 Python 3.12，再继续' }

# 可以换盘；后续路径均由此派生。
$labRoot = 'D:\TraceLightLab'
New-Item -ItemType Directory -Path $labRoot -Force | Out-Null
Set-Location -LiteralPath $labRoot
git clone --branch main https://github.com/zhoumeng-creater/tracelight-ids.git
if ($LASTEXITCODE -ne 0) { throw '克隆失败；若目录已存在，请按文末继续运行说明操作' }
Set-Location -LiteralPath (Join-Path $labRoot 'tracelight-ids')
git rev-parse HEAD
```

公共仓库克隆无需登录 GitHub。首次运行固定使用 `main`。启动实验后不要在运行中切分支、拉取代码或编辑配置；每次运行会记录源码指纹。

### 1.2 创建独立环境并通过安装检查

```powershell
$venv = Join-Path $labRoot 'venv'
.\scripts\setup_lab.ps1 -Python $python312 -VenvDir $venv -Device cuda
$py = Join-Path $venv 'Scripts\python.exe'

# 后续 Python 命令都调用这个虚拟环境，无需激活。
# 函数检查退出码，失败时不继续后面的阶段。
function Invoke-LabPython {
    & $py @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: exit $LASTEXITCODE" }
}
Invoke-LabPython -m ids doctor --require-cuda --output (Join-Path $labRoot 'preflight.json')
```

成功标准：安装脚本最后输出 `Environment ready`，依赖检查和测试通过，doctor 中 CUDA 可用。脚本固定安装 PyTorch 2.6.0+cu126；不需要先执行 README 中用于纯 CPU 环境的安装示例。若执行策略拦截 `.ps1`，按实验室管理员允许的方式运行，不更改全机策略。

### 1.3 合成数据验证 CPU 和 CUDA 两条路径

```powershell
$smokeRaw = Join-Path $labRoot 'smoke\raw'
$smokeData = Join-Path $labRoot 'smoke\prepared'
Invoke-LabPython -m ids synthetic --output $smokeRaw
Invoke-LabPython -m ids prepare --train-csv (Join-Path $smokeRaw 'UNSW_NB15_training-set.csv') --test-csv (Join-Path $smokeRaw 'UNSW_NB15_testing-set.csv') --output $smokeData --data-kind synthetic

Invoke-LabPython scripts/run_lab.py --data-dir $smokeData --work-dir (Join-Path $labRoot 'smoke\cpu') --config configs/smoke.yaml --device cpu
Invoke-LabPython scripts/run_lab.py --data-dir $smokeData --work-dir (Join-Path $labRoot 'smoke\cuda') --config configs/smoke.yaml --device cuda
```

两条命令使用不同输出目录，避免设备不同却复用同一套件。成功时各自打印 `Development complete` 和报告路径；报告必须标明合成数据。这一步的分数只说明工程路径工作，不能作为论文实验成绩。

### 1.4 下载真实数据，再准备四分区

```powershell
$raw = Join-Path $labRoot 'data\raw'
$prepared = Join-Path $labRoot 'data\prepared-v1'
Invoke-LabPython scripts/fetch_unsw.py --output $raw --source mirror --accept-unverified-mirror
Invoke-LabPython -m ids prepare --train-csv (Join-Path $raw 'UNSW_NB15_training-set.csv') --test-csv (Join-Path $raw 'UNSW_NB15_testing-set.csv') --output $prepared --data-kind mirror
```

这里显式使用固定版本公开镜像，**没有与独立官方副本逐字节比对**。下载器记录来源、哈希、结构和行数，不静默切换来源。若网络无法下载，可从另一台机器复制两份 CSV 和配套 `provenance.json` 到 `$raw`，跳过下载步骤；不要只复制一个文件。使用官方来源的替代流程见 [DATA.md](DATA.md)。

检查 `$prepared` 中的 `manifest.json`、`manifest.sha256`、`audit.json` 和四个 Parquet 文件。当前固定镜像及默认划分种子的预期行数：train 122737、validation 26302、calibration 26302、test 82332。实际来源、参数或划分实现变化时重新审查清单，不能为凑行数手改文件。审计报告还记录重复与标签冲突，保留这些信息。

### 1.5 启动 12 次开发基线

```powershell
$work = Join-Path $labRoot 'work'
Invoke-LabPython scripts/run_lab.py --data-dir $prepared --work-dir $work --config configs/baselines.yaml --device cuda --dry-run
Invoke-LabPython scripts/run_lab.py --data-dir $prepared --work-dir $work --config configs/baselines.yaml --device cuda
```

LR/RF/HGB 在 CPU 训练，MLP 使用 CUDA。成功时 `work/runs/baseline-dev-v1` 下应有 12 个完成的 run，`summary.json` 有 12 条记录；报告位于 `work/reports`。浏览器直接打开生成的 HTML 即可，不需要启动网站。

## 2. Linux：Bash 按顺序执行

以下命令在同一个 Bash 会话执行；`set -e` 遇到失败会退出当前执行流程。先确认 `python3.12`、Git 和驱动已由实验室提供；缺少 venv 时按实验室的系统管理方式安装。

```bash
set -euo pipefail
git --version
python3.12 --version
nvidia-smi

# 改为本地 SSD 上有写权限的新目录。
lab_root="$HOME/TraceLightLab"
mkdir -p "$lab_root"
cd "$lab_root"
git clone --branch main https://github.com/zhoumeng-creater/tracelight-ids.git
cd tracelight-ids
git rev-parse HEAD
bash scripts/setup_lab.sh --python python3.12 --venv-dir "$lab_root/venv" --device cuda
lab_python="$lab_root/venv/bin/python"
"$lab_python" -m ids doctor --require-cuda --output "$lab_root/preflight.json"

"$lab_python" -m ids synthetic --output "$lab_root/smoke/raw"
"$lab_python" -m ids prepare --train-csv "$lab_root/smoke/raw/UNSW_NB15_training-set.csv" --test-csv "$lab_root/smoke/raw/UNSW_NB15_testing-set.csv" --output "$lab_root/smoke/prepared" --data-kind synthetic
"$lab_python" scripts/run_lab.py --data-dir "$lab_root/smoke/prepared" --work-dir "$lab_root/smoke/cpu" --config configs/smoke.yaml --device cpu
"$lab_python" scripts/run_lab.py --data-dir "$lab_root/smoke/prepared" --work-dir "$lab_root/smoke/cuda" --config configs/smoke.yaml --device cuda

"$lab_python" scripts/fetch_unsw.py --output "$lab_root/data/raw" --source mirror --accept-unverified-mirror
"$lab_python" -m ids prepare --train-csv "$lab_root/data/raw/UNSW_NB15_training-set.csv" --test-csv "$lab_root/data/raw/UNSW_NB15_testing-set.csv" --output "$lab_root/data/prepared-v1" --data-kind mirror

"$lab_python" scripts/run_lab.py --data-dir "$lab_root/data/prepared-v1" --work-dir "$lab_root/work" --config configs/baselines.yaml --device cuda --dry-run
"$lab_python" scripts/run_lab.py --data-dir "$lab_root/data/prepared-v1" --work-dir "$lab_root/work" --config configs/baselines.yaml --device cuda
```

数据来源、预期分区行数及成功标准与 Windows 相同；报告在 `work/reports`，12 次基线记录在 `work/runs/baseline-dev-v1`。

## 3. 完成之后核验、测量和带回

1. 保留实验控制台输出；核查 12 个 run 都存在 `run.json` 且 `status` 为 `complete`，不要用 `failure.json` 当完成记录。
2. 对第 1.5 或第 2 节的实际运行命令加 `--resume` 再执行一次。代码会校验完成工件、源码和开发数据，输出 12 条 `Verified completed run`。它还会生成新的环境记录和报告，不会重新训练已完成模型。
3. 固定 CPU 线程，分别测单条延迟和批量吞吐。下面以 LR、seed 17 为例；其余 run 使用各自目录和新的输出文件名。CPU 测量尽量独占机器，关闭无关计算任务；不要与 GPU 训练并行计时。

Windows（保持前面变量及函数的会话）：

```powershell
Invoke-LabPython -m ids benchmark --run-dir (Join-Path $work 'runs\baseline-dev-v1\00-lr-s17') --data-dir $prepared --output (Join-Path $work 'benchmarks\00-lr-s17-t1.json') --threads 1
```

Linux：

```bash
"$lab_python" -m ids benchmark --run-dir "$lab_root/work/runs/baseline-dev-v1/00-lr-s17" --data-dir "$lab_root/data/prepared-v1" --output "$lab_root/work/benchmarks/00-lr-s17-t1.json" --threads 1
```

4. 运行结束后备份：`preflight.json`、**完整 `smoke/` 和 `work/`**、`data/raw/`（含 provenance）、**完整 `data/prepared-v1/`**，以及实际使用的代码提交号和改动。`smoke/` 保留 CPU/CUDA 验收证据；`work/` 包含环境、配置、代码指纹、逐条预测、模型包、指标、训练历史和报告，不能只带走截图。
5. 这些数据和模型不提交到公开 GitHub。没有本地改动的源码可凭提交号重新克隆；有改动时另行保留补丁和未跟踪的源码。虚拟环境不需要跨机器复制，应按依赖文件重建。
6. 复制后比对源与备份的文件数量和 SHA-256。在相同版本的项目环境中，对备份的 `work/runs/baseline-dev-v1` 再执行 `ids report`，指定新的报告输出位置；报告生成会验证每个 run 的工件哈希。数据文件还应与 prepared 清单记录的 SHA-256 一致。不能把“复制完成”或“HTML 能打开”单独当作整套实验备份完整的证明。

## 4. 暂停、恢复与下一步

- **新终端会话**：先进入已克隆的仓库，重新设置上述路径变量、虚拟环境解释器及 PowerShell 函数；不要重新 clone 到已有目录。
- **中断后继续**：使用原源码、数据、配置和输出路径，在实际运行命令末尾加 `--resume`。未完成的尝试会被保留后重新训练该模型；这不是 epoch 级续训。具体行为见 [LAB.md](LAB.md)。
- **完成一轮后更新代码**：先备份结果，查看 `git status` 确认自己的改动，然后在没有实验运行时 `git pull --ff-only`。源码或配置变了，就使用新的套件名或新的实验工作目录，不与旧结果混合。
- **环境检查失败**：保存 `preflight.json`（如已生成）、`nvidia-smi` 和完整异常。不要把 CUDA 失败后偷偷改成 CPU 的运行标成 GPU 实验。
- **开发基线完成后**：分析多种子差异与错误样本，再做 `configs/feature-pilot.yaml` 的 8/16/24 字段初筛。这个配置是初筛用途，不是已经配平全部训练预算的正式消融方案。
- **首次实验不运行 `freeze` / `evaluate`**。这些命令留到研究方案、模型选择和正式比较组确定后使用；遵循 [PROTOCOL.md](PROTOCOL.md)。

## 验证范围

实现曾通过 111 项本地测试和 Windows/Ubuntu CI；具体证据见 [VALIDATION.md](VALIDATION.md)。2026-11-01 从 GitHub 独立克隆 `main` 后，111 项测试再次通过，CPU/CUDA 两套四模型合成数据流程及报告生成均成功；固定镜像的两个下载地址响应正常。此次只补充文档，未改变实验代码或配置。

新克隆的功能验证使用个人电脑 RTX 4050 Laptop 和既有 Python 依赖，不能替代实验室新虚拟环境的安装检查。RTX 4090D 实验室机器仍需实际执行前置检查及 CUDA 训练；本指南不承诺未知驱动、网络和权限条件下零故障。
