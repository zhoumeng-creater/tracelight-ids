# v4 实验室交接：复用v3、先跑机制pilot

2026-11-04。固定发布标签 `upgrade-v4-lab-20261104`。你在实验室拉取并运行，开发机不运行真实训练。本轮依据[v3复核](RISK_V3_RESULTS_20261104.md)、[固定协议](UPGRADE_V4_PROTOCOL.md)与[近邻审查](research_alignment_20261103/risk_v4_literature_audit.md)，先验证目标退化、工作点漂移和实际更新门禁，不预设新方法胜出。

## 1. 新checkout、原解释器、原v3与prepared

以下是可替换的目录布局示例。`$labRoot`沿用你原实验室根目录；旧v3输入应是含 `RISK_RESEARCH_COMPLETE.json` 的完整环境目录，**不是 packages 或回传ZIP的小元数据目录**。使用原SSD和原CUDA环境，保留旧代码与所有结果。

```powershell
$labRoot = 'D:\TraceLightLab' # 改为原实验室根目录
$repo = Join-Path $labRoot 'tracelight-upgrade-v4'
$py = Join-Path $labRoot 'venv\Scripts\python.exe'
$unsw = Join-Path $labRoot 'data\prepared-v1'
$cicids = Join-Path $labRoot 'experiments\stage4-stable-selection\data\cicids-prepared'
$old = Join-Path $labRoot 'experiments\risk-v3-full-20261103'
$v3unsw = Join-Path $old 'risk-v3-unsw'
$v3cicids = Join-Path $old 'risk-v3-cicids'
$check = Join-Path $labRoot 'experiments\upgrade-v4-check-20261104'
$pilot = Join-Path $labRoot 'experiments\upgrade-v4-pilot-20261104'
$full = Join-Path $labRoot 'experiments\upgrade-v4-full-20261104'

git clone --branch upgrade-v4-lab-20261104 --single-branch https://github.com/zhoumeng-creater/tracelight-ids.git $repo
if ($LASTEXITCODE -ne 0) { throw '拉取失败；不要覆盖旧代码' }
Set-Location -LiteralPath $repo
$env:PYTHONPATH = Join-Path $repo 'src'
git rev-parse HEAD
& $py -c "import sys,ids,torch; print(sys.executable); print(ids.__file__); print(torch.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
if ($LASTEXITCODE -ne 0) { throw '先核原解释器' }
& $py scripts/run_upgrade_lab.py --help
```

不需要新增依赖或重装torch/驱动。入口继续锁定原基础包、PyTorch2.6.0与CUDA0 RTX4090D。正在运行时不pull、切分支、改配置。另建代码目录可以保留原v3完整源码和输出；不要移动或重算旧结果来适配新哈希。

## 2. 只读预检和双环境pilot

```powershell
& $py scripts/run_upgrade_lab.py --profile pilot --unsw-v3-dir $v3unsw --unsw-data-dir $unsw --cicids-v3-dir $v3cicids --cicids-data-dir $cicids --work-dir $check --dry-run
if ($LASTEXITCODE -ne 0) { throw '预检失败，保留记录和报错' }

& $py scripts/run_upgrade_lab.py --profile pilot --unsw-v3-dir $v3unsw --unsw-data-dir $unsw --cicids-v3-dir $v3cicids --cicids-data-dir $cicids --work-dir $pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot停止；保留现场，不覆盖重跑' }
```

dry-run只读JSON清单/预检与代码，不读parquet、不加载模型。真正运行会核旧v3全部文件哈希及开发数据，再读取相同角色；预检不表示已训练。`$check`与`$pilot`分开。pilot固定seed17，每环境最多8学生×30步，复用完整F/R/C/V，不做CIC小cap，不拟合新教师/锚点；不能把旧teacher成本说成从未发生。

先进行最多30步共同hard Phase-I，只依据C找第一个实际可行起点，不使用V。`no_feasible_start`表示该seed没有合格起点、八条件全部跳过；结果仍可封存作失败诊断，不能标为方法有效。guarded条件可能全部回退step0；需检查实际参数移动和相对warm start的效果。

每步记录R重校阈值下V结果及固定原阈值下兼容性结果。v3代理预算继续记录，v4主门槛改为共同实际事件预算；不能把新判据套回旧表重判胜出。

## 3. 回传文件与核验

运行结束回传 `$pilot\packages\` 下：

- `UPGRADE_V4_UNSW.zip` 与 `.zip.sha256`
- `UPGRADE_V4_CICIDS.zip` 与 `.zip.sha256`
- `UPGRADE_V4_LAB_PREFLIGHT.json` 与 `.json.sha256`

ZIP含新学生/预测/history、Phase-I、首步hard/soft梯度诊断、教师目标退化统计、R/旧阈值双口径、源代码与旧v3身份凭据。旧教师/锚点不重复打包，请继续保留原v3目录与prepared数据。包生成、SHA匹配和 `development_complete` 仅表示诊断链完整；是否改善要看返回的失败、回退、共同准入及成本。

```powershell
& $py -m ids.upgrade_research --output (Join-Path $pilot 'upgrade-v4-unsw') --verify
if ($LASTEXITCODE -ne 0) { throw 'UNSW工件不完整' }
& $py -m ids.upgrade_research --output (Join-Path $pilot 'upgrade-v4-cicids') --verify
if ($LASTEXITCODE -ne 0) { throw 'CIC工件不完整' }
```

## 4. 中断、复核与后续完整配置

`--resume`只复用完整工件并核原v3/source/data/解释器/硬件身份，不能恢复半程epoch。错误或中断保留全部目录和报错；纠错版本用新checkout、新输出目录，不能编辑完成标记。完整目录的安全复核可在相同命令末尾加 `--resume`。

先带回pilot复核，不自动扩大。full配置已准备：同三seed、每条件60步，最多24学生/环境；Phase-I成本另计。

```powershell
& $py scripts/run_upgrade_lab.py --profile full --unsw-v3-dir $v3unsw --unsw-data-dir $unsw --cicids-v3-dir $v3cicids --cicids-data-dir $cicids --work-dir $full
if ($LASTEXITCODE -ne 0) { throw '完整研究停止；保留现场' }
```

两环境均可只传其中一组路径独立运行。全部真实计算包括C验收的CPU forward均在实验室；本机仅分析回传工件和做合成检查。calibration/test仍不读取，未来需要真正未使用来源完成新主张确认。
