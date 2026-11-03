# v3 实验室交接：GitHub 拉取、4090D 运行、回传复核

工作方式沿用前几轮：开发机完成调研、实现和合成检查并推送 GitHub；你在实验室 SSD 上拉取代码，用既有 RTX4090D 环境与 prepared 数据运行，再回传结果。无需远程 SSH。本机不进行本轮真实训练。

**本轮先运行 UNSW pilot，再回传。** v3 是待验证候选；完成标记表示实验链及文件覆盖完整，不表示效果改善或创新成立。旧 calibration/test 不读取，也不作为新方法的独立确认集。

## 1. 获取冻结代码，保留原实验

本轮发布标签为 `risk-v3-lab-20261103`，仓库 `zhoumeng-creater/tracelight-ids`。建议另建代码目录，避免改动正在运行的 v2 和旧封存环境。以下是布局示例；数据与解释器应改成实验室原有的实际路径，不重新 prepare、不复制回传包中的小清单目录冒充完整数据。

```powershell
$labRoot = 'D:\TraceLightLab'
$repo = Join-Path $labRoot 'tracelight-risk-v3'
$py = Join-Path $labRoot 'venv\Scripts\python.exe'
$unsw = Join-Path $labRoot 'data\prepared-v1'
$cicids = Join-Path $labRoot 'stage4-stable-selection\data\cicids-prepared'
$check = Join-Path $labRoot 'risk-v3-check-20261103'
$pilot = Join-Path $labRoot 'risk-v3-pilot-20261103'
$full = Join-Path $labRoot 'risk-v3-full-20261103'

git clone --branch risk-v3-lab-20261103 --single-branch https://github.com/zhoumeng-creater/tracelight-ids.git $repo
if ($LASTEXITCODE -ne 0) { throw '拉取失败；不要覆盖已存在的代码目录' }
Set-Location -LiteralPath $repo
$env:PYTHONPATH = Join-Path $repo 'src'
git status --short
git rev-parse HEAD
& $py -c "import sys,ids,torch; print(sys.executable); print(ids.__file__); print(torch.__version__); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
if ($LASTEXITCODE -ne 0) { throw '先核对原实验室解释器与环境' }
& $py scripts/run_risk_lab.py --help
if ($LASTEXITCODE -ne 0) { throw 'v3入口不可用' }
```

已有独立 v3 checkout 可用 `git fetch origin dev --tags` 后 `git switch --detach risk-v3-lab-20261103`；有本地修改应保留处理，不能 reset/clean 丢弃。训练开始后不再 pull、切版本或改配置。全流程使用同一个 `$py`。

本轮只依赖原项目锁定基础包与 PyTorch 2.6.0 CUDA，不需要新增 TabM/LibAUC 包，也不应升级原 torch、驱动或 CUDA。入口核对实际 Python 包版本、CUDA0 RTX4090D 和数据 schema；型号不符会停止，不回退 CPU/4050。HGB 教师用实验室 CPU，MLP 用实验室 GPU。

若 prepared 路径不确定，查看实验室第四轮的 `experiment/STAGE4_PLAN.json` 与 `preflight.json`。`$unsw`、`$cicids` 必须是各自包含完整 train/validation parquet 的目录；CIC 使用 CNS2022 repaired 联合会话/重复组版本。

## 2. 预检后执行 UNSW pilot

```powershell
& $py scripts/run_risk_lab.py --profile pilot --unsw-data-dir $unsw --work-dir $check --dry-run
if ($LASTEXITCODE -ne 0) { throw '预检未通过，保留报错和预检记录' }

& $py scripts/run_risk_lab.py --profile pilot --unsw-data-dir $unsw --work-dir $pilot
if ($LASTEXITCODE -ne 0) { throw 'pilot已停止；保留目录和报错，不覆盖重跑' }
```

预检只读源码、环境与 prepared manifest 元数据，不读 parquet、不生成角色、不训练。训练入口会核验 train/validation 文件哈希与 audit 后再读取。`$check` 与 `$pilot` 分开，不能把 dry-run 当作实验结果。若有意在同一预检目录转入真实运行，配置/源码/解释器/环境/路径必须相同，并用 `--resume`。

pilot 固定 seed17；整组取 train≤12,000、validation≤6,000，再将选中的 train 确定性整组划为 F/R/C，目标比例70%/10%/20%。比例是软目标，不能拆组、删掉不方便的家族或换 seed 追求好结果。三折 HGB＋一个完整 F 教师、一个锚点、九个学生，共14次拟合；锚点最多30 epochs，学生各30个完整 F+C 优化步。全批步不等于 minibatch epoch，不与旧轮耗时直接混比。

终端会打印教师折、锚点及每个条件的进度；各学生目录内的 `history.json` 在运行中更新。尚未完成时这些文件只是现场诊断，不能当封存结果。若出现 OOF 类别不足、角色支持不可行、非有限损失或显存不足，保存原目录和报错回传，**不换 seed、不偷偷缩数据、不关闭约束**。

## 3. 回传哪些文件

成功后回传 `$pilot\packages\` 下四个文件：

- `RISK_V3_UNSW.zip`
- `RISK_V3_UNSW.zip.sha256`
- `RISK_V3_LAB_PREFLIGHT.json`
- `RISK_V3_LAB_PREFLIGHT.json.sha256`

ZIP 内含源码/配置快照、实际环境、F/R/C/V 身份与缺族账本、OOF教师及预测目标、锚点、九种学生、逐步损失/约束/对偶记录、CPU原始logit逐条验证预测、汇总和全部文件 SHA。打包会回读各 ZIP 成员核验。数据本体不打包；保留原 prepared 与工作目录至接收端复核完毕。

本机可先自行核验环境结果：

```powershell
& $py -m ids.risk_research --output (Join-Path $pilot 'risk-v3-unsw') --verify
if ($LASTEXITCODE -ne 0) { throw '工件核验失败' }
Get-Content -LiteralPath (Join-Path $pilot 'risk-v3-unsw\summary.json')
```

`development_complete` 只表示链路完成。`eligible_development_candidate` 还要核CPU固定阈值下的C实际预算/V误报；risk条件额外需要C代理约束达标。`diagnostic_no_eligible_checkpoint` 必须作为不合格/负结果报告，即使平均召回好看也不能升格。选中检查点与最终轨迹的对偶状态分别保存，不混称可恢复的完整训练状态。

## 4. 中断与恢复

`--resume` **只复用已经完整且验收通过的环境结果并完成打包**，会重新核对源码、配置、实际解释器/硬件/数据绑定。没有 epoch/半程恢复。完整 pilot 的安全复核命令：

```powershell
& $py scripts/run_risk_lab.py --profile pilot --unsw-data-dir $unsw --work-dir $pilot --resume
if ($LASTEXITCODE -ne 0) { throw '恢复核验失败；保留原结果' }
```

半程目录会被拒绝且保留。纠错应提交新版本、使用新输出目录，不能修改旧完成标记或哈希来绕过校验。

## 5. 完整开发研究入口（先审查 pilot）

先回传 pilot，检查数值、缺族、约束可行性、教师质量、CPU决策一致性和实际成本，再决定完整运行。完整配置已准备：每环境 seeds17/29/43、无行数cap、27学生＋12教师＋3锚点，共42次拟合；两环境串行合计84次。**拟合数是预算，不是创新或有效性的证据。** CIC禁止 bounded pilot，以免139,618行的 Portscan 巨关联组被小cap静默排除。

```powershell
& $py scripts/run_risk_lab.py --profile full --unsw-data-dir $unsw --cicids-data-dir $cicids --work-dir $full
if ($LASTEXITCODE -ne 0) { throw '完整研究停止，保留现场' }
```

也可只给其中一个数据路径，保持各自独立结果。full同样回传 packages 下 ZIP、各 SHA 及原始预检凭据。即使完整研究成功，也仍是已知 train/validation 上的开发证据，不能重新打开旧test给新模型调参。

## 6. 原始 logit 决策复核

v3 冻结的是 `float64(raw_logit) >= frozen_R_threshold`。不能直接把阈值做 sigmoid 后交给旧概率推理接口；边界舍入可能改变误报数，run.json会报告这些差异。复核入口默认只重放保存的 validation cohort，同序输出，不读取 calibration/test：

```powershell
& $py scripts/predict_risk.py --research-dir (Join-Path $pilot 'risk-v3-unsw') --condition risk_kd --seed 17 --data-dir $unsw --split validation --output (Join-Path $labRoot 'risk-v3-replay-new.parquet')
if ($LASTEXITCODE -ne 0) { throw '原始logit复核失败' }
```

输出路径必须全新并位于封存目录及prepared外；这一步复核决策，不赋予不合格候选部署资格。协议与近邻适配差异见 [RISK_V3_PROTOCOL.md](RISK_V3_PROTOCOL.md)。
