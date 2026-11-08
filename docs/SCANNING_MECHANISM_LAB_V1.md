# Scanning 三臂实验室执行说明

固定标签：`scanning-mechanism-lab-v1`。运行协议见[机制设计](SCANNING_MECHANISM_PROTOCOL_20261108.md)，参数与文件身份见[冻结配置](../configs/context-scanning-mechanism-v1.json)。本轮复用第十四轮P8192／正常C5000／E20000、冻结预处理和TabICLv2权重，不重新抽样、不拟合预处理、不训练权重。

三臂依次为`baseline`、`src_frequency_bijection`、`src_mask_zero`。CUDA分别构建完整上下文，保存后在CPU FP32、单线程、批128评分。总计3次完整缓存构建、15000条C评分、60000条E评分、0优化器更新、0 QCOC编译。B0的分数和阈值容差固定为`3e-5 + 3e-5 * abs(old)`，C/E判决必须逐行一致；失败即停止，不执行B1/B2。

## 1. 切换冻结代码

下面路径沿用第十四轮实验室目录。若目录迁移，只修改路径变量。`$round14`必须同时包含`prepared`与`comparison`，其中保留`comparison/full/preprocessor.joblib`、`full/threshold.json`、`full_calibration.csv`和`evaluation_scores.csv`。无需旧完整模型实体，也无需重新下载原始ZIP或权重。

```powershell
$ErrorActionPreference = 'Stop'
$repo = 'D:\zhoumeng\Graduate\tracelight-context-cache'
$py = 'D:\zhoumeng\Graduate\venvs\context-cache\Scripts\python.exe'
Set-Location -LiteralPath $repo
git fetch origin --tags
if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
git checkout --detach scanning-mechanism-lab-v1
if ($LASTEXITCODE -ne 0) { throw 'frozen checkout failed' }

$round14 = Join-Path $repo 'local\context-input-v2'
$checkpoint = Join-Path $repo 'local\checkpoints\tabicl-classifier-v2-20260212.ckpt'
$runRoot = Join-Path $repo 'local\context-scanning-v1'
$scanArgs = @{
    Python = $py
    Round14Root = $round14
    Checkpoint = $checkpoint
    RunRoot = $runRoot
}
```

沿用既有Python3.12／Torch2.6.0+cu126／CUDA12.6／固定TabICLv2源码环境。预检核对已归档版本，脚本只以`--no-deps --no-build-isolation -e .`登记本次代码，不更新依赖。不要重新执行旧的`lab-prepare-v2`或`lab-input-compare`。

## 2. 合成回归与只读预检

```powershell
& .\scripts\RUN_SCANNING_MECHANISM.ps1 -Stage Preflight @scanArgs
```

脚本要求新建的`$runRoot`，将合成JUnit、日志、路径登记协议和预检记录集中保存在其中。临时合成工件在仓库`local/synthetic/scanning-v1`，不进入结果ZIP。合成检查只使用随机小模型和合成CSV；必须零失败、零错误、零跳过。

正式预检只读文件摘要、CSV和JSON，核对：

- P/C/E与旧预处理、权重、参考分数及阈值的冻结SHA；CSV行顺序、标签与攻击族。
- 旧C完整阈值登记、严格`>`判决以及旧C/E逐行判决。
- [源码清单](../configs/scanning-mechanism-code-freeze-v1.json)中的LF规范化摘要、固定依赖与CUDA可用性。

预检不反序列化真实权重或预处理器，不构造真实源端口映射，不运行模型。配置登记只替换文件路径，不能修改种子、样本数、阈值规则或干预。

## 3. 正式运行

```powershell
& .\scripts\RUN_SCANNING_MECHANISM.ps1 -Stage Run @scanArgs
```

运行时加载原冻结预处理，不调用`fit`。源端口映射仅由P生成，在第一次模型评分前保存；B1保留P源端口分布、相等关系与已见状态。B2在最终48列输入上置零，只删除源端口槽，不再删除常量列。每臂重新构建列缓存与完整KV，分别用同一正常C登记自己的1%阈值。

B0复现通过后才进入B1/B2。运行失败会保存`comparison/state.json`和`failure.txt`，不得放宽容差、改阈值、换种子或覆盖目录后重跑。

## 4. 工件复算与归档

```powershell
& .\scripts\RUN_SCANNING_MECHANISM.ps1 -Stage Verify @scanArgs
& .\scripts\RUN_SCANNING_MECHANISM.ps1 -Stage Pack @scanArgs
```

`Verify`只读取已保存工件，不额外运行模型。它校验三臂模型／共享预处理／映射／阈值的绑定、C/E行身份与标签、完整阈值登记、配对错误、逐族指标和工作量。`Pack`生成文件SHA清单，并回读ZIP逐成员核对。回传这两个文件：

```text
local/context-scanning-v1-results.zip
local/context-scanning-v1-results.zip.sha256
```

成功包保留三份完整缓存模型、共享预处理与B1映射、冻结配置、源码／环境记录、合成JUnit、预检、旧参考分数、三臂逐行分数和判决、cohort掩码、输入碰撞、映射变化审计及分析报告。不会把原始CSV、下载checkpoint、旧轮总包或合成临时目录再次装入ZIP。

若预检或正式运行失败，停止后续评分，只运行以下命令回传失败证据：

```powershell
& .\scripts\RUN_SCANNING_MECHANISM.ps1 -Stage Pack @scanArgs
```

封包会标记`partial`或`failed`，不会把失败包视为完成结果。

## 5. 回传后的判断顺序

先核`baseline_replay.json`与`state.json`，再读`mapping_audit.json`中目标54／15条的实际移动比例。只有目标受到干预，B1阴性结果才对相应解释有意义。

`report.json`分别记录总体、所有攻击族及指定子组的TP/FP、修复与新增FN/FP、分数变化、各臂C分位变化和阈值变化。Scanning对正常的ROC面积单独报告，低误报面积固定积分区间`[0,0.01]`，原面积及除以0.01并列；不从E重选阈值。

源端口敏感性不自动等于检测收益。B1改变输出只能支持具体编号参与判断；B2输入碰撞与模型响应分别解释。此次E已用于机制设计，结果用于定位问题，不作新方法的独立确认。真实结果返回前，不宣称扫描已恢复或源端口应被删除。
