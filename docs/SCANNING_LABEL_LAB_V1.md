# Scanning 标签粒度三臂：实验室执行指南

本轮只运行一次 **T0二分类／TF真实扫描三类／TP整组伪三类**。固定标签为 `scanning-label-lab-v1`，设计见[标签粒度协议](SCANNING_NEXT_DESIGN_20261108.md)，数据与参数身份见[冻结配置](../configs/context-scanning-label-v1.json)。本机已完成合成核查，真实权重与真实数据比较只在实验室执行。

本机完整合成回归397项通过、零失败/错误/跳过；两项旧冷进程测试的中文路径日志编码已修正，并以线程告警视为错误补验通过。记录见[实现验收](research_alignment_20261108/scanning_next/implementation_validation.json)。

实验室环境已经安装好，直接从下面第1节开始。复用第十四轮P/C/E、冻结预处理器和已校验权重；第十五轮B0已精确复现这份参考。本轮不重新下载原始ZIP，不重新抽样或拟合预处理，不运行旧源端口三臂。

三臂各重新构建完整的列级与ICL缓存。CUDA用于构建，CPU FP32单线程、批128评分；总计3次完整KV构建、15000条C评分、60000条E评分，0优化器更新、0 QCOC编译。

## 1. 获取冻结代码，设置已有路径

在PowerShell中执行。路径沿用既有实验室目录；如果目录已迁移，只改路径变量。

```powershell
$ErrorActionPreference = 'Stop'
$repo = 'D:\zhoumeng\Graduate\tracelight-context-cache'
$py = 'D:\zhoumeng\Graduate\venvs\context-cache\Scripts\python.exe'
Set-Location -LiteralPath $repo
git status --short
git fetch origin --tags
if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
git checkout --detach scanning-label-lab-v1
if ($LASTEXITCODE -ne 0) { throw 'frozen checkout failed' }
git rev-parse HEAD

$round14 = Join-Path $repo 'local\context-input-v2'
$checkpoint = Join-Path $repo 'local\checkpoints\tabicl-classifier-v2-20260212.ckpt'
$runRoot = Join-Path $repo 'local\context-label-v1'
$labelArgs = @{
    Python = $py
    Round14Root = $round14
    Checkpoint = $checkpoint
    RunRoot = $runRoot
}
```

若checkout因本地修改失败，先保留修改与日志，不执行 `reset --hard`。`$runRoot`必须是不存在的新目录；不要覆盖第十四、十五轮，也不要把旧结果复制到新目录中。

`$round14`内必须有以下文件，预检会核对冻结SHA：

```text
prepared/support.csv
prepared/normal_calibration.csv
prepared/development_evaluation.csv
prepared/preparation.json
comparison/full/preprocessor.joblib
comparison/full/threshold.json
comparison/full_calibration.csv
comparison/evaluation_scores.csv
```

不需要第十四轮的 `model.pt`。若缺少上述工件，从已归档第十四轮恢复原文件到同一目录；不要重跑数据准备来生成替代文件。

## 2. 确认固定TabICL源码位置

沿用Python3.12、Torch2.6.0+cu126、TabICL2.2.0及既有固定依赖。上轮曾因缺少 `local/vendor/tabicl` 导致合成测试跳过，本轮先明确核对：

```powershell
$vendor = Join-Path $repo 'local\vendor\tabicl'
if (-not (Test-Path -LiteralPath (Join-Path $vendor 'src\tabicl\_model\tabicl.py'))) {
    throw 'Restore the existing pinned TabICL checkout at local\vendor\tabicl first.'
}
git -C $vendor rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'TabICL checkout is unavailable' }
& $py --version
if ($LASTEXITCODE -ne 0) { throw 'Python environment is unavailable' }
```

TabICL提交必须为 `d83091b845a8c5f7c087bd54964f755a9e560f45`。可继续使用上轮已建立的目录联接；脚本会将这个固定源码和当前仓库的 `src` 置于Python导入路径，不使用临时修改版脚本。

只有换了机器、尚无环境时，才使用已有安装入口，然后重新执行上述核对：

```powershell
& .\scripts\setup_context_cache_lab.ps1 `
    -VenvDir 'D:\zhoumeng\Graduate\venvs\context-cache' `
    -Python 'python'
```

已完成环境安装的这台机器无需再次运行安装入口。下面的预检只用 `--no-deps --no-build-isolation --editable .` 登记本轮代码，不升级依赖。

## 3. 合成核查与只读预检

```powershell
& .\scripts\RUN_SCANNING_LABEL.ps1 -Stage Preflight @labelArgs
```

成功标准：合成测试零失败、零错误、零跳过，并生成 `$runRoot\preflight.json`。脚本同时保存 `synthetic-junit.xml`、`preflight.log` 和路径登记后的 `protocol.json`。

预检核对固定代码清单、运行环境与CUDA，验证P/C/E、原预处理、权重、参考分数、阈值的SHA，以及P的4096正常／512扫描／3584其他攻击构成。它只读二进制摘要、CSV和JSON，不加载真实权重或预处理器、不拟合数据、不生成真实伪类。

全部结果集中在 `$runRoot`；合成临时文件集中在仓库 `local/synthetic/label-v1`。源码冻结清单为 [scanning-label-code-freeze-v1.json](../configs/scanning-label-code-freeze-v1.json)。

## 4. 正式运行一次三臂

```powershell
& .\scripts\RUN_SCANNING_LABEL.ps1 -Stage Run @labelArgs
```

执行顺序如下：

1. 加载原冻结预处理，保留相同48维输入；只从P生成TF与TP支持标签，并在首次模型评分前保存。
2. 运行T0。C/E分数与阈值必须满足 `3e-5 + 3e-5 * abs(reference)`，逐行告警判决必须一致。失败即停止，TF/TP不会构建。
3. 运行TF与TP，各自重新构建两层完整KV。TP按完整FP32攻击输入整组分配，精确512行进入小类；无法满足时停止，不换种子、不拆组补样本。
4. 每臂用同一正常C独立登记1%阈值，再评分同一E。最终分数始终是全攻击相对正常的联合log-odds；三分类不使用argmax或Scanning单类概率作为告警。

不要调整阈值、换伪类种子或追加模型。阴性结果同样回传。现有E已打开，本轮用于定位机制；通过候选门槛也不自动启动历史输入或独立确认实验。

## 5. 复算、打包与回传

```powershell
& .\scripts\RUN_SCANNING_LABEL.ps1 -Stage Verify @labelArgs
& .\scripts\RUN_SCANNING_LABEL.ps1 -Stage Pack @labelArgs
```

`Verify`不加载模型或预处理器，使用保存的P输入、标签、logits、分数和阈值复算：整组伪类规则、类别映射与工件绑定、T0重放、逐行判决、各族与配对指标、低误报区间pAUC和候选门槛。`Pack`建立SHA清单并回读ZIP逐成员核对。

回传以下两个文件，直接放入本项目既有“实验归档”目录：

```text
local/context-label-v1-results.zip
local/context-label-v1-results.zip.sha256
```

只回传这一轮结果，不再重新压缩全部历史轮次、原始数据ZIP或预训练checkpoint。结果包保留三个模型工件、共享预处理、P输入与标签审计、完整C/E logits及分数、阈值、代码/环境记录、测试和运行日志。

运行结束可先查看：

```powershell
Get-Content -LiteralPath (Join-Path $runRoot 'comparison\state.json')
Get-Content -LiteralPath (Join-Path $runRoot 'comparison\baseline_replay.json')
$report = Get-Content -LiteralPath (Join-Path $runRoot 'comparison\report.json') -Raw | ConvertFrom-Json
$report.candidate_gate | ConvertTo-Json -Depth 8
```

候选门槛同时要求TF扫描检出超过T0/TP、整体TP不低于T0、FP不高于T0，以及固定FPR区间 `[0,0.01]` 的扫描归一化pAUC超过两者。报告还会区分TF与TP的实际误报代价，不以单个扫描检出数宣布胜出。

## 6. 出错时保留什么

任一步失败后停止后续评分，保留 `$runRoot` 和对应 `*-failure.txt`、日志。若comparison未完成，可直接执行 `Pack` 回传部分工件与失败记录；不要覆盖目录或改参数重试。若结果已声明complete但 `Verify` 失败，`Pack`也会拒绝完整性不符的包，此时先回传失败文本、日志与 `state.json`，保留全部原工件待定位。

本轮执行入口为 `RUN_SCANNING_LABEL.ps1`。旧 `RUN_SCANNING_MECHANISM.ps1` 对应已完成的端口三臂，不用于这轮。
