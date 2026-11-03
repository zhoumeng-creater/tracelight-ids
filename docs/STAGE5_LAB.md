# 实验室操作 · 第五轮最终评估

沿用第四轮实验室 SSD 上的仓库、虚拟环境、UNSW prepared 和完整工作目录。工作目录须有 `experiment/STAGE4_COMPLETE.json` 及 `data/cicids-prepared/`。第四轮回传 ZIP 里只有 CIC 清单和来源，正式评估需要实验室原 prepared 文件。第五轮读取既有模型，在 CPU 上完成校准、测试与打包。

## 首次运行

在 PowerShell 里将下面几个路径改成实际位置。`$stage5` 指向第四轮目录外的全新输出位置。

```powershell
$labRoot = 'D:\TraceLightLab'
$repo    = Join-Path $labRoot 'tracelight-ids'
$py      = Join-Path $labRoot 'venv\Scripts\python.exe'
$unsw    = Join-Path $labRoot 'data\prepared-v1'
$stage4  = Join-Path $labRoot 'stage4-stable-selection'
$stage5  = Join-Path $labRoot 'stage5-final-evaluation'
Set-Location -LiteralPath $repo

git switch main
if ($LASTEXITCODE -ne 0) { throw '请保留并处理当前修改，再更新主分支' }
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw '更新失败，请核对仓库与网络' }
git rev-parse HEAD

& $py workflows/stage5/run.py --stage4-work-dir $stage4 --unsw-data-dir $unsw --output-dir $stage5
if ($LASTEXITCODE -ne 0) { throw '第五轮已停止，请保留现场和错误输出' }
```

入口验证发布提交、项目依赖及第四轮材料绑定。它按固定清单校准 12 个模型，统一锁定阈值，再完成两份数据的测试和报告。若校验提示清单或模型不符，请找回原实验室目录。中断后保留已有阈值、测试记录及错误信息。

## 恢复与回传

恢复时使用同一发布版本、同一环境和目录，沿用首次运行的变量。

```powershell
& $py workflows/stage5/run.py --stage4-work-dir $stage4 --unsw-data-dir $unsw --output-dir $stage5 --resume
if ($LASTEXITCODE -ne 0) { throw '恢复校验未通过，请保留当前输出' }
```

结束后在 `$stage5/reports/index.html` 阅读结果，`$stage5/reports/summary.json` 保存机器可读配对数据。将 **`$stage5/TraceLight-Stage5.zip` 和同目录的 `TraceLight-Stage5.zip.sha256`** 一起回传。保留两份 prepared、原第四轮目录和第五轮输出，直至接收端核验完毕。完整评估定义见 [第五轮协议](STAGE5_PROTOCOL.md)。
