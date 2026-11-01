# 第四轮实验室操作：稳定字段选择与第二数据环境

本轮只执行新工作：UNSW 30 个主 run、作者修复 CICIDS2017 的 18 个主 run、独立选择稳定性诊断及 3 轮 CPU 计时；完成后自动核验与打包。研究范围见 [STAGE4_PROTOCOL.md](STAGE4_PROTOCOL.md)。第三轮结果和原数据目录保留。

沿用实验室已经工作正常的 Git、Python 虚拟环境和 CUDA，不要求重新安装，不重复旧三轮全套实验。新的工程 smoke 与测试由发布端完成；这里不要求使用者手工运行一长串内部阶段。更新仓库后使用一个运行入口，由入口依次检查环境、验证旧数据、准备第二数据环境、执行本轮、生成报告和 ZIP；失败即停止，保留现场。

本文的真实实验入口沿用已有环境；合成预检由入口自动执行。发布端工程验证与实验室真实 48 个 run 是两件事，不能用预检分数作为研究结果。已证实检查、发布前选择池修正和待完成验证见仓库中的 [STAGE4_VALIDATION.md](https://github.com/zhoumeng-creater/tracelight-ids/blob/main/docs/STAGE4_VALIDATION.md)。

## 1. 更新已有仓库

使用原来的终端和虚拟环境。只调整以下目录变量以匹配实际位置；路径有空格时不需要删除空格或迁移数据。

```powershell
$labRoot = 'D:\TraceLightLab'
$repo = Join-Path $labRoot 'tracelight-ids'
$prepared = Join-Path $labRoot 'data\prepared-v1'
$stage4 = Join-Path $labRoot 'stage4-stable-selection'
$py = Join-Path $labRoot 'venv\Scripts\python.exe'
Set-Location -LiteralPath $repo

git status --short
git switch main
if ($LASTEXITCODE -ne 0) { throw '切换失败；先保留并处理当前修改' }
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw '更新失败；停止' }
git rev-parse HEAD
```

有本地修改时保留并核对，不使用 `reset --hard` 或 `clean` 丢弃。`$prepared` 指向前几轮已用的真实 UNSW prepared；`$stage4` 是本轮新的输出目录。入口自动核对同一份原 prepared，正式配置要求其 manifest SHA-256 为 `6348a936f68a67a11e241452eff9abdf334d4ea02f71472f805f38c532c002d4`。指错目录或不同 split 会停止，不会重建数据替换原实验。不要重新 prepare UNSW，不用此前第三轮输出目录承接第四轮。

## 2. 一条命令执行与自动打包

在上面同一个 PowerShell 会话中执行：

```powershell
& $py scripts/run_stage4.py --unsw-data-dir $prepared --work-dir $stage4 --device cuda
if ($LASTEXITCODE -ne 0) { throw '第四轮已停止；保留目录和错误输出，不跳过失败阶段' }
```

入口需自行完成环境/设备检查、UNSW 清单核验、CIC 官方下载哈希核验与独立 prepared 准备、选择诊断、48 个主 run、三轮计时、报告、完整性核验和结果 ZIP。CIC 下载或导入失败时停止，不自动改用其他数据集或低配矩阵；真实配置不得静默退回 CPU、减少抽样次数或省去某个 seed。

发布配置自动构造 train 内按家族比例抽行的共同选择池，预算 50,000 行、稀有家族至少一行，保留原组身份；内部 50 次和外层 5 次仍按池内组抽样，全部主模型使用完整 train。不需要使用者手工抽样、补 Portscan 或改变 seed。某次子样本缺少家族会写入审计，不悄悄换样本。

运行期间不 pull、不切分支，不修改源码或配置；保持电源与 CPU 模式一致，关闭其他训练和长期占用 CPU 的任务。MLP 的 CUDA 训练结束后仍要完成 CPU 测量，不因为 GPU 空闲而认为程序卡住。

新数据准备需要下载、解压和 Parquet 空间；结果包不等于原始数据全部备份。入口保存具体来源与哈希，真实所需空间以下载和产物为准。沿用本地 SSD，不在 OneDrive 同步目录内进行大批训练和计时。

## 3. 中断后续跑

在**原命令**后增加 `--resume`；使用同一仓库版本、原数据目录和同一 `$stage4`。不要为了续跑先更新源码，不删失败记录或校验文件。

```powershell
& $py scripts/run_stage4.py --unsw-data-dir $prepared --work-dir $stage4 --device cuda --resume
if ($LASTEXITCODE -ne 0) { throw '恢复核验未通过；保留现场并处理错误' }
```

完成阶段仅在身份与哈希核验通过后复用。未完成模型按同 seed 从头训练，属于工件级恢复，不是从某个 epoch 接着训练。源、数据、配置或环境变化时拒绝拼接；保留错误输出，查明变化后决定恢复原环境或建立新实验。

完成后再次 resume 先核验已封存工件，再复用完成结果，不重复训练。入口最后打印报告及 ZIP 的实际路径，以这份输出作为回收位置。

## 4. 带回哪些文件

入口最后输出报告路径、ZIP 路径及 SHA-256。带回 ZIP 和其 `.sha256` 文件，保留实验室原输出及两份 prepared 数据，直到接收端完整核验。不要只发截图、摘要表或单个模型。

默认产物位于 `$stage4`：

```text
stage4-stable-selection/
  data/                          # 新数据下载与prepared；不进入结果ZIP
  experiment/
    STAGE4_COMPLETE.json         # 工件覆盖与校验完成标记
    reports/index.html           # 两数据环境的报告入口
    protocol/                    # 本轮源码、配置与协议快照
    unsw/                        # UNSW选择、主run、三轮计时与来源
    cicids/                      # CIC选择、主run、三轮计时与来源
  TraceLight-Stage4.zip
  TraceLight-Stage4.zip.sha256
```

旧 UNSW prepared 仍在 `$prepared`，不会因上述容器结构被复制成新版本或覆盖。结果 ZIP 不包含原始/预处理大数据，需继续保留整个 `$stage4` 和原 UNSW 数据供复验。入口内部预检产生的工程记录不算 48 个科学主 run。

本轮完成应对应 48 个主 run、两个数据环境的选择诊断、144 份 CPU 计时和可核验报告。总览已自动生成稳定方法与 MI、去冗余条件之间的配对表，并链接各数据环境的 `method-comparisons.json`，无需手算同 seed 差值。运行完成不代表稳定选择成功；负结果、家族短板和额外选择成本都要带回。报告只属于 validation 开发证据，本轮没有执行最终 calibration/test。

公开 GitHub 放代码、配置、测试和说明；真实模型、数据、日志与结果包不随本次代码更新自动公开。对外结果表和论文结论以接收端核验后的实际工件为依据。

## 5. Linux

同样沿用既有环境：

```bash
set -euo pipefail
lab_root='/path/to/TraceLightLab'
repo="$lab_root/tracelight-ids"
prepared="$lab_root/data/prepared-v1"
stage4="$lab_root/stage4-stable-selection"
lab_python="$lab_root/venv/bin/python"
cd "$repo"
git status --short
git switch main
git pull --ff-only
git rev-parse HEAD
"$lab_python" scripts/run_stage4.py --unsw-data-dir "$prepared" --work-dir "$stage4" --device cuda
```

中断后保留目录与代码，原运行命令加 `--resume`，不要先重新 pull。Windows 与 Linux 的绝对延迟分别报告，不合并为同机计时。
