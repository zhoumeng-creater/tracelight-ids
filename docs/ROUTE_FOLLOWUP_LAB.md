# 第十七轮工程补测：TabICL 重放诊断与 TabPFN 准备

2026-11-09。本次只执行三个任务：官方 TabICL 1／8 视图的缓存构建路径诊断，以及 TabPFN-2.5 的旧开发集补测。**不重训或重新评分已完成的 TabM，不重跑新通信组确认，不改写第十七轮原结果。** TabPFN 后补结果使用已打开的旧 P／C／E，单独标为开发比较，不能加入原确认表后称为独立确认。

发布分支为 `research/route-followup-20261109`，实验室固定标签为 `route-followup-lab-v1`。新 checkout 使用 `D:\zhoumeng\Graduate\tracelight-route-followup`，脚本通过该目录的 `src` 加载补测代码；已有 core 解释器只读复用，不在旧环境内安装或升级依赖。

本地固定源码后，244 项合成回归及 Ruff 通过，包括实际子进程的随机微型官方 TabICL 1/8 视图完整流程，以及 Windows PowerShell 5.1、pwsh 入口。TabPFN 固定权重已实际下载并验 SHA，未加载；真实 GPU/CPU 路径诊断、PFN 环境安装和检测效果由实验室执行。[验证记录与原始 JUnit](research_alignment_20261109/route_followup/validation.json)。

## 首次执行

在实验室 PowerShell 复制下面整段。目标 checkout 应尚不存在；保留旧 `tracelight-route-batch`、第十四轮工件、已登记 P／C／E 和 TabICL checkpoint。

```powershell
Set-Location -LiteralPath 'D:\zhoumeng\Graduate'
git clone --branch route-followup-lab-v1 --single-branch https://github.com/zhoumeng-creater/tracelight-ids.git tracelight-route-followup
if ($LASTEXITCODE -ne 0) { throw 'Git clone failed' }
Set-Location -LiteralPath 'D:\zhoumeng\Graduate\tracelight-route-followup'
Set-ExecutionPolicy -Scope Process Bypass
git describe --tags --exact-match
git rev-parse HEAD
.\scripts\RUN_ROUTE_FOLLOWUP.ps1
```

标签应为 `route-followup-lab-v1`。入口已带实验室默认路径：

| 参数 | 默认位置 |
|---|---|
| `Python` | `D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe` |
| `OriginalRun` | `D:\zhoumeng\Graduate\tracelight-route-batch\local\route-batch-20261108\run` |
| `OutputRoot` | 新 checkout 下的 `local\route-followup-20261109` |
| `TabPFNPython` | `D:\zhoumeng\Graduate\venvs\tracelight-route-tabpfn-followup\Scripts\python.exe` |
| `TabPFNCheckpoint` | 从原 `protocol.json` 的 `tabpfn_checkpoint.path` 读取 |

原协议中的 TabPFN 权重默认位于 `D:\zhoumeng\Graduate\tracelight-context-cache\local\checkpoints\tabpfn-v2.5-classifier-v2.5_default.ckpt`。只有实际存放路径不同时才在首次执行中显式传入对应参数；恢复时沿用同一组路径。`OutputRoot` 与 `OriginalRun` 必须互不包含。

入口先准备 TabPFN 环境和权重，再核对旧实验身份，执行三个独立任务并打包。TabPFN 准备失败时仍尝试两项 TabICL 诊断，保存具体失败阶段；不会把缺少解释器、依赖或权重记为模型效果不佳。旧工件身份、源码或路径不符等全局预检错误会停止执行。

## TabPFN 环境与权重

不需要另跑旧的 `setup_route_batch_lab.ps1`。新入口自动调用 [PREPARE_ROUTE_TABPFN.ps1](../scripts/PREPARE_ROUTE_TABPFN.ps1)，在新的 `tracelight-route-tabpfn-followup` 目录创建独立 PFN 环境，避免旧 optional 目录的残缺安装影响本次准备。已存在且完整匹配的环境直接复用，本入口带标记的中断安装可以恢复；显式指定的已有不兼容环境保留并报错，不覆盖升级。

准备固定为 64 位 Python 3.12、`torch==2.6.0+cu126`、`tabpfn==9.2.0`、`numpy==1.26.4`、`scikit-learn==1.5.1`、`huggingface-hub==0.28.1`，其余直接与传递依赖见 [requirements-route-batch-tabpfn.txt](../requirements-route-batch-tabpfn.txt)及其引用文件。准备阶段执行 `pip check`、依赖与模型接口导入、CUDA 小矩阵检查和合成测试，保存安装报告、完整 freeze、JUnit 和机器可读状态；不构建真实模型。

权重只从官方 Hub 的固定版本取得：

| 项目 | 固定值 |
|---|---|
| 官方模型库 | `Prior-Labs/tabpfn_2_5` |
| revision | `6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b` |
| 文件名 | `tabpfn-v2.5-classifier-v2.5_default.ckpt` |
| 大小 | 42,935,499 字节 |
| SHA-256 | `5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12` |
| LICENSE | TABPFN-2.5 License v1.1，16,905 字节 |
| LICENSE SHA-256 | `ac1d23429431a622b91c73db741868e69a0140d401ebb20b9e869a835e036e28` |

2026-11-09 已核实该固定版本公开可读，并用本入口实际下载、校验了权重与 LICENSE；本机没有加载权重。用户已授权本课题的权重准备，脚本按[官方公开条款](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b/LICENSE)直接下载，不登录或伪造 Prior Labs 账户已接受许可的状态。若官方访问条件变为 gated，或返回 401／403，则记录 `access_required`，按官方访问流程处理，不换来源绕过限制。完整本地权重通过文件名、字节数和 SHA 校验后可直接复用；现有不匹配文件不覆盖。

`preparation/prepare-report.json` 只有在解释器可启动、全部固定依赖、CUDA、合成测试及权重身份均通过后才为 `ready`。下载完成本身不代表整个准备完成。当前 default 权重经过真实数据继续训练，训练清单未出现 NF-ToN-IoT 名称不能证明无数据重叠，详见[来源记录](research_alignment_20261108/route_batch/sources.md)。

统一入口还会在全局预检后、真实任务前，使用 core 解释器执行定向合成回归，覆盖 foundation、构建路径重放、完整冷重放、成本、补测队列和 CLI，并保存 JUnit。这里使用 mock 或随机初始化的微型官方模型，不加载真实预训练权重；无需另执行测试命令。

## 三项任务各自回答什么

| 任务 | 实际工作与用途 |
|---|---|
| `tabicl_official_1` | 按旧配方重建 1 视图，分别使用原 `cuda:0` 构建后转 CPU、直接 CPU 构建；核对旧 C／E 分数、原始概率、重复调用和阈值判决变化。 |
| `tabicl_official_8` | 对 8 视图执行同样的两条构建路径诊断；保存两路径的逐行差值及原阈值／重新校准后的得失。 |
| `tabpfn25_official_8` | 只用旧 P8192 构建 8 个 estimator，明确采用 CPU 构建、CPU 评分；旧正常 C 独立登记阈值，输出旧 E 全部指标、各族和原始概率。 |

原设备重放用于辨认旧分数能否在原构建条件下复现，CPU 路径用于评价一个明确记录的新部署路径。CPU 路径的成绩、阈值与成本单列，不拿它补写或替换第十七轮原 GPU 构建系统的成本。

各路径先按旧实验的 C→E 顺序生成主分数，再对前 128 条 E 连续查询两次；后置探针不参与阈值登记。报告分列主分数与旧结果的差异、同一对象重复调用的差异，避免把查询顺序变化混入设备差异。

正式计时前，在独立进程中按同一 CPU 路径冷重建，复核完整旧 C／E。进入独立成本进程须同时满足三项条件：全部 C／E 分数在 `atol=3e-5, rtol=3e-5` 内；E 在保存阈值下判决不变；E 在冷重放 C 重新校准后判决也不变。原始概率、分数差值和 C 阈值附近的判决变化分别保留。单行 p50／p95 使用 64 条固定记录各 3 次，128／512 行批次各 3 次；RSS 是热态进程的 10 ms 采样峰值，加载／重建时间另记，不能把配方文件大小或进程 RSS 当作模型独占成本。

成本进程还会从冷加载后的第一次查询重放前 128 条 E；分数超容差或告警翻转都会先保存证据、再停止计时，不能用前一个进程的完整重放通过代替这项检查。

任务 `complete` 表示本项诊断／补测及同路径成本流程完成，不表示两条 TabICL 构建路径与原分数完全一致，也不表示已找到唯一故障原因。`diagnostic_failed`、`measurement_failed`、`failed`、`unavailable` 需分别查看对应日志与重放报告。

## 断点恢复与状态

保持本标签代码、旧 run、输入工件及路径，重新执行：

```powershell
Set-Location -LiteralPath 'D:\zhoumeng\Graduate\tracelight-route-followup'
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\RUN_ROUTE_FOLLOWUP.ps1 -Resume
```

恢复只针对新补测目录：已完成任务直接复用，其余任务写入新的 `attempt-NNN` 后重试。TabPFN 准备会重新核验已安装内容并继续本入口拥有的中断安装；旧 TabM 和已打开的确认集始终不进入队列。不得在原 `route-batch` run 上执行旧批次的恢复入口来运行这次修复。

查看三项任务状态与准备结果：

```powershell
Import-Csv -LiteralPath '.\local\route-followup-20261109\run\summary.csv' | Format-Table task,status,attempt
Get-Content -LiteralPath '.\local\route-followup-20261109\preparation\prepare-report.json' -Raw
```

新 `run/binding.json` 绑定旧实验、PFN 路径和补测源码；`original_files.json` 与 `original_preservation.json` 核对旧 run 全部文件的前后身份，包括 TabM 和 confirmation。总体 `all_tasks_complete=true` 同时要求三项任务完成、旧 run 未改变。若旧数据或源码改变，恢复会拒绝，不放宽原摘要检查。

## 回传

脚本结束后上传以下目录下最新的 ZIP 和同名 `.sha256.json`：

```text
D:\zhoumeng\Graduate\tracelight-route-followup\local\route-followup-20261109\route-followup-results-<UTC时间>.zip
D:\zhoumeng\Graduate\tracelight-route-followup\local\route-followup-20261109\route-followup-results-<UTC时间>.zip.sha256.json
```

若单项失败，入口仍先打包已有新结果和错误证据，再返回错误状态。ZIP 生成不等于全部任务完成，应查看 `summary.json`。回传包保留准备报告、pip freeze、JUnit、预检、逐行分数／概率、重放差值、成本和失败日志；不包含模型、预训练权重或旧整包。模型、缓存重建配方与权重继续保留在实验室。

主要文件位置：

| 文件 | 用途 |
|---|---|
| `run/summary.json`、`summary.csv` | 三任务状态及旧 run 是否保持不变 |
| `run/preparation-evidence/` | 本次准备、环境、安装、JUnit、预检和 wrapper 证据 |
| `run/tasks/tabicl_official_*/attempt-NNN/diagnostic/report.json` | 原设备与 CPU 路径的汇总比较 |
| 同上 `diagnostic/original/`、`diagnostic/cpu/` | 各路径的 `probe.csv`、C／E 分数与原始概率、阈值及 `path_report.json` |
| CPU 部署目录中的 `cold_replay_report.json`、`replay_diagnostic.json`、`replay_scores.csv` | 独立冷重建复核和成本入口的逐行一致性证据 |
| CPU 部署目录中的 `cost.json`、`cost-worker.log` | 通过重放检查后生成的 CPU 成本，或测量失败日志 |
| `run/tasks/tabpfn25_official_8/attempt-NNN/` | 旧开发 C／E 的 PFN 补测结果、概率、阈值、重放与成本 |
