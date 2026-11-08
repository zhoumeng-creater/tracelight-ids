# 同信息检测路线比较：一次完整实验室批次

2026-11-08。源码与合成验证已完成，真实模型尚未运行，执行地点为实验室 RTX4090D。用户已允许持续运行，不设总时限；候选表与每条训练轨迹的 200 epochs 仍固定。

设计与结果解释见 [批次设计](ROUTE_BATCH_DESIGN_20261108.md)，模型年份、官方实现、固定版本和权重见 [来源记录](research_alignment_20261108/route_batch/sources.md)，本机验证见 [validation.json](research_alignment_20261108/route_batch/validation.json)。

## 1. 这一批会执行什么

| 部分 | 完整工作量 |
|---|---|
| TabM 监督学习 | 2 种表示 × 2 个宽度 × 2 个学习率 × 3 个种子，共 24 条 T/V 轨迹，每条 200 epochs |
| TabM 最终模型 | 每条轨迹只按 V BCE 选轮数，再从头在全部 P 上训练，共 24 次 refit |
| 已有系统 | 第十四轮完整 KV、普通 QCOC 复用核验后的分数，重载模型复测成本 |
| 官方系统 | TabICLv2 官方 1／8 视图；有指定本地权重和环境时加 TabPFN 2.5 的 8 视图 |
| 开发评价 | 每个模型分别用同一 C 登记阈值；输出 E 全部攻击、各族、逐行得失和 CPU 成本 |
| 新通信组确认 | 全部可用最终模型冻结；排除第十三／十四轮六份 P/C/E 的所有通信组及行身份，新取 C′ 正常 10,000 行、E′ 未按标签分层的 50,000 行；自动评分与汇总 |

执行队列共 29 项，其中每个 TabM 队列项包含轨迹与全 P 重训两次拟合。缺少 TabPFN 时是 28 个可执行项加 1 个不可用项；不会假装完成该参照。新通信组来自同一数据源剩余部分，报告与旧 E 分开，不称为外部数据或前向时间验证。

全部比较使用原有 48 字段、P8192 和正常校准规则。低误报主工作点为 1%，另输出预定敏感性工作点。保存 50／200 轮与 V 最佳轮的诊断，不用 E 选择最佳 epoch。CPU 成本在新的纯推理进程内串行测量，排除训练状态残留；另记冷加载／缓存重建和 512 行 CSV 解析到评分的耗时。

## 2. 从 GitHub 拉取固定版本，保留旧实验目录

发布标签为 `route-batch-lab-v1`，开发分支为 `research/route-batch-20261108`。实验室使用固定标签，在新的 `tracelight-route-batch` 目录运行；旧仓库、数据和环境继续保留，脚本从旧目录读取已登记工件。

在 PowerShell 执行，目标目录应尚不存在：

```powershell
Set-Location -LiteralPath 'D:\zhoumeng\Graduate'
git clone --branch route-batch-lab-v1 --single-branch https://github.com/zhoumeng-creater/tracelight-ids.git tracelight-route-batch
if ($LASTEXITCODE -ne 0) { throw 'Git clone failed' }
Set-Location -LiteralPath 'D:\zhoumeng\Graduate\tracelight-route-batch'
git describe --tags --exact-match
git rev-parse HEAD
```

标签核对应输出 `route-batch-lab-v1`。随后执行本节的路径设置、第 3 节环境安装和第 4 节完整运行命令。此前交付的源码 ZIP 仍可使用，其源码摘要与本标签的运行代码一致；当前指南包含 GitHub 拉取步骤。

以下命令沿用此前实验室路径；若实际存放位置不同，只调整对应路径变量。首次登记后，路径、配置和源码均绑定到结果，恢复时不改动它们。

```powershell
Set-Location -LiteralPath 'D:\zhoumeng\Graduate\tracelight-route-batch'
Set-ExecutionPolicy -Scope Process Bypass

$OldRepo = 'D:\zhoumeng\Graduate\tracelight-context-cache'
$CoreVenv = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core'
$PFNVenv = 'D:\zhoumeng\Graduate\venvs\tracelight-route-tabpfn'
$BasePython = 'D:\zhoumeng\Graduate\venvs\context-cache\Scripts\python.exe'
$Round14 = Join-Path $OldRepo 'local\context-input-v2'
$Round13 = Join-Path $OldRepo 'local\context-cache-lab-20261107\prepared'
$Checkpoint = Join-Path $OldRepo 'local\checkpoints\tabicl-classifier-v2-20260212.ckpt'
$PFNCheckpoint = Join-Path $OldRepo 'local\checkpoints\tabpfn-v2.5-classifier-v2.5_default.ckpt'
$SourceZip = 'D:\zhoumeng\Graduate\data\nf-ton-iot-v3\NF-ToN-IoT-v3-official.zip'
$OutputRoot = 'D:\zhoumeng\Graduate\tracelight-route-batch\local\route-batch-20261108'
```

`BasePython` 必须为 64 位 Python 3.12。安装器创建新环境，并固定 TabICL 作者源码及依赖；不会在旧环境内安装。`CoreVenv` 和 `PFNVenv` 首次安装前应不存在。

## 3. 一次安装核心与可选环境

若准备纳入 TabPFN，先执行下面的完整安装。安装器记录依赖解析、安装报告、`pip check`、依赖版本、CUDA 可用性和合成测试；本步骤不下载或加载预训练权重。

```powershell
.\scripts\setup_route_batch_lab.ps1 -Python $BasePython -CoreVenvDir $CoreVenv -WithTabPFN -TabPFNVenvDir $PFNVenv
```

TabPFN 权重需先按 [官方许可流程与固定文件说明](research_alignment_20261108/route_batch/sources.md#实验室环境与权重准备)取得，放到 `$PFNCheckpoint`。该文件应为 42,935,499 字节，SHA-256 为：

```text
5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12
```

```powershell
Get-FileHash -LiteralPath $PFNCheckpoint -Algorithm SHA256
```

若这次不具备 TabPFN 权重，只安装核心：

```powershell
.\scripts\setup_route_batch_lab.ps1 -Python $BasePython -CoreVenvDir $CoreVenv
```

上述两个安装命令择一执行。核心已安装后补可选环境用 `-TabPFNOnly`，不重复核心安装。要让 TabPFN 同时进入开发和新通信组确认，应在下面的首次完整运行前准备好它；确认阶段一旦冻结，不再加入新模型。

## 4. 一条命令完成开发比较、新通信组确认和打包

```powershell
$RunArgs = @{
    Python = Join-Path $CoreVenv 'Scripts\python.exe'
    OutputRoot = $OutputRoot
    Round14Root = $Round14
    Checkpoint = $Checkpoint
    TabPFNCheckpoint = $PFNCheckpoint
    TabPFNPython = Join-Path $PFNVenv 'Scripts\python.exe'
    SourceZip = $SourceZip
    Round13Prepared = $Round13
}
.\scripts\RUN_ROUTE_BATCH.ps1 @RunArgs
```

入口依次完成登记、身份预检、29 项任务、新通信组准备与确认、汇总及 ZIP 校验。SourceZip 和 Round13Prepared 必须一起提供；上述完整命令已包含二者。首次运行前核对路径存在，尤其保留旧完整 KV／QCOC 部署包，不能只留下预测 CSV。

单项模型异常会保存日志和失败状态，继续其余独立任务。确认数据准备或整体确认失败时，仍先打包已有开发结果和错误记录，再返回失败状态。数据／代码身份不符等全局错误会停止入口，防止在错误输入上继续训练。

不设总超时，也不按检测分数决定是否追加训练。24 条轨迹均完整训练 200 轮；全 P 重训轮数来自各自的 V 选择。运行期间保持机器唤醒，避免并行占用 GPU 或干扰 CPU 成本测量。

## 5. 中断恢复与查看进度

保持原源码、环境、登记配置及路径，重新打开 PowerShell 后重建上面的路径变量和 `$RunArgs`，执行：

```powershell
.\scripts\RUN_ROUTE_BATCH.ps1 @RunArgs -Resume
```

这是任务级恢复：已完成任务直接复用；被中断且没有结果文件的任务保留旧 attempt，再从相同 seed 重启该任务。不会从中断的 optimizer 状态接着训练，也不会覆盖旧记录。已有明确失败结果的任务不会被普通 Resume 无限重试。

如果某项因解释器／权重缺失被记为 `unavailable`，在原登记路径补齐后可用：

```powershell
.\scripts\RUN_ROUTE_BATCH.ps1 @RunArgs -Resume -RetryUnavailable
```

开发部分只重试不可用项。确认部分只重试原 freeze 已纳入但运行失败／不可用的模型，不能借恢复把后来新增的模型加入已打开的确认集。代码或协议变更也不能冒充原批次恢复。

查看进度：

```powershell
Get-Content -LiteralPath (Join-Path $OutputRoot 'run\REPORT.md')
Import-Csv -LiteralPath (Join-Path $OutputRoot 'run\summary.csv') | Format-Table id,status,tp,fp,scanning_tp,p95_ms
```

当前任务日志位于 `run\tasks\<任务名>\attempt-NNN.log`，训练曲线位于该 attempt 的 `search/history.json`。每轮历史含 T/V 损失、梯度与更新时间，不能把尚无最后结果的长任务当成失败。

## 6. 回传内容与读数顺序

成功或部分失败的批次都会留下状态表。完整命令结束后回传 `$OutputRoot` 下最新的 `route-results-*.zip` 及同名 `.sha256.json`。回传包包含协议、源码摘要、环境、分数、逐行错误、训练历史、成本、失败原因和新确认结果；默认不重复携带模型与预训练权重。

完整模型保留在实验室 `run` 及原第十四轮目录，不能因回传了分析包就删除。新源码 ZIP 自带源码逐文件清单；结果 ZIP 另有逐成员校验清单。

| 文件 | 用途 |
|---|---|
| `run/summary.csv`、`REPORT.md` | 开发比较总表，明确 complete／failed／unavailable／measurement_failed 等状态 |
| 每项 `result.json`、`selection.json`、`search/history.json`、`refit/summary.json` | 实际训练与选轮、全 P 重训、参数量及效果 |
| 每项 `evaluation.csv`、`calibration.csv`、`metrics.json` | 独立阈值、逐行预测与相对旧完整 KV 的配对得失计数 |
| 每项 `cost.json`、`cost-worker.log` | 独立推理进程中的 CPU 延迟、RSS、加载／重建、CSV 到评分和完整部署字节 |
| `confirmation/freeze.json`、`data_binding.json`、`prepared/preparation.json` | 全模型冻结与新数据来源、排除组、完整源文件校验 |
| `confirmation/summary.csv`、各模型的结果和配对文件 | 新通信组上的全部模型与全部种子结果 |

先查完成状态、实际误报与总攻击检出，再看 Scanning 和其他攻击族的得失，然后比较容量、表示、学习率与训练轨迹，最后看部署成本和新通信组结果。不能只挑旧 E 上最好的 seed，也不能把训练错误记作算法无效。更宽模型、较长训练或另一模型家族若有效，需要在对应配对条件下解释；这批现成方法比较本身不称为原创算法。
