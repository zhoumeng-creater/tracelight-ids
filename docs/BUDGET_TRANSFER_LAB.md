# 重构研究：GitHub 拉取、实验室运行与回传

2026-11-04后续决定：[同类排序核心机制重设计](ALGORITHM_REDESIGN_DECISION_20261104.md)。尚未启动的本页46次矩阵暂缓，当前优先[零拟合价值审计](THRESHOLD_AUDIT_LAB.md)。本页及原固定标签保留作对照复现；已开始的运行保留原代码／协议／结果。

2026-11-04。固定发布标签 `budget-transfer-20261104`，在新的 checkout 运行。真实训练和正式 CPU 测量均在实验室 RTX4090D；本机只做代码、单元和合成检查。

本次重建[研究主线与算法](BUDGET_TRANSFER_DESIGN_20261104.md)，[科学协议](RESEARCH_RECONSTRUCTION_PROTOCOL_20261104.md)固定教师价值、字段／容量瓶颈和反证对照。没有 phase-I、旧 TP 保留契约或后置 guard。

## 先运行完整样本单种子 pilot

pilot 与 full 使用相同的完整 F/R/C/V 身份、40 个 hard warmup epoch 和40个继续训练macro步；仅 seed17 与 seed17/29/43 不同。没有行截断、巨组拆分或家族重抽。**先跑两环境 pilot 并回传，审阅后再决定是否扩展 full。** 不自动扩展。

每环境每 seed：2 个 HGB 教师、4 个 hard warmup（all／MI24 × small／wide）、17 个小学生继续训练条件，共23次。两环境 pilot 共46次拟合。all 的同信息教师条件与候选相同，记别名不重复拟合；教师 headroom 不佳也保留全部 KD 结果。

仍使用现有 lab Python 和 prepared 数据，默认路径与上一轮一致：

| 用途 | 根目录下的路径 |
|---|---|
| 根目录 | `D:\zhoumeng\Graduate` |
| Python | `venv\Scripts\python.exe` |
| v3 UNSW | `experiments\risk-v3-full-20261103\risk-v3-unsw` |
| v3 CIC | `experiments\risk-v3-full-20261103\risk-v3-cicids` |
| UNSW prepared | `data\prepared-v1` |
| CIC prepared | `experiments\stage4-stable-selection\data\cicids-prepared` |
| 新 checkout | `tracelight-budget-transfer` |

v3 用来核验完整数据身份和整组角色。新教师／学生仅在 F 重训，不复用旧目标／锚点权重；不依赖 v4 包或前一版零拟合审计先完成。零拟合审计仍可单独运行。

## 推荐入口

在实验室 PowerShell 执行：

```powershell
$labRoot = 'D:\zhoumeng\Graduate'
$checkout = Join-Path $labRoot 'tracelight-budget-transfer'
git clone --branch budget-transfer-20261104 --single-branch https://github.com/zhoumeng-creater/tracelight-ids.git $checkout
Set-Location -LiteralPath $checkout
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\RUN_BUDGET_TRANSFER.ps1 -LabRoot $labRoot -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\RUN_BUDGET_TRANSFER.ps1 -LabRoot $labRoot
```

若已取得私人交接包中的 `RUN_BUDGET_TRANSFER.ps1`，可直接运行该脚本；它会按固定标签 clone 到新目录，再调用同一入口。不要更新正在运行的旧实验源码。

`-DryRun` 只核源码、锁定依赖、GPU0、prepared manifest、v3 completion/plan 元数据；不查询模型、不读 prepared 正文、不拟合。它写入单独的 `budget-transfer-pilot-check-20261104`；正式目录为 `budget-transfer-pilot-20261104`，先核完整旧源与开发分区哈希。

路径不同时覆盖参数（正式运行去掉 `-DryRun`）：

```powershell
.\scripts\RUN_BUDGET_TRANSFER.ps1 -LabRoot 'E:\Graduate' -V3Root 'E:\Runs\risk-v3-full-20261103' -UnswData 'E:\Data\prepared-v1' -CicidsData 'E:\Data\cicids-prepared' -DryRun
```

若需重新核验已完成工件／补齐打包，使用相同参数加 `-Resume`；仅接受已经封存完成的环境，不续接部分epoch。中断目录与终端错误应保留，不覆盖重跑。

## 直接 Python 入口

支持单环境或自定义工作目录：

```powershell
$env:PYTHONPATH = Join-Path $checkout 'src'
& (Join-Path $labRoot 'venv\Scripts\python.exe') scripts/run_budget_transfer_lab.py --unsw-v3-dir (Join-Path $labRoot 'experiments\risk-v3-full-20261103\risk-v3-unsw') --unsw-data-dir (Join-Path $labRoot 'data\prepared-v1') --work-dir (Join-Path $labRoot 'experiments\budget-transfer-unsw-pilot-20261104') --profile pilot
```

双环境再传 `--cicids-v3-dir`／`--cicids-data-dir`。full 用 `--profile full` 或 PowerShell `-Profile full`，使用另一个新目录。先解释pilot，不自动扩大；两次均跑的 seed17 不算第四个独立种子。

## 输出与回传

将正式工作目录下 `packages` 的**全部六个文件**回传到原实验归档位置：

```text
D:\zhoumeng\Graduate\experiments\budget-transfer-pilot-20261104\packages
  BUDGET_TRANSFER_UNSW.zip
  BUDGET_TRANSFER_UNSW.zip.sha256
  BUDGET_TRANSFER_CICIDS.zip
  BUDGET_TRANSFER_CICIDS.zip.sha256
  BUDGET_TRANSFER_LAB_PREFLIGHT.json
  BUDGET_TRANSFER_LAB_PREFLIGHT.json.sha256
```

包内含源码／配置绑定、MI选择、教师native训练分数、全部模型与history、R/C分数、教师价值／相同输入歧义、`FROZEN_SELECTION.json`、晚读V逐行预测及全部条件清单。教师和四组warmup也冻结后评V。资源记录包含模型包／维数／参数、同口径CPU流程、训练／教师查询耗时和forward行数；相同步数不声称相同FLOPs。

## 结果口径

1. 阈值为 R/C 正常阈值的最大值，C参与选择；有限训练池控制不是部署总体保证。V超1% FPR照实报告，不凭高召回升格。
2. headroom 仅作分层诊断，不决定删seed。先比较 candidate 对 direct pAUC 和 direct raw KL，再看 uniform、固定类内置乱、同信息教师。
3. 比较未成立就撤回相应知识／局部化／特权字段归因；不能只凭近期二分类 RankLoss 退化声称创新。
4. 四组warmup在相同阶段比较容量和字段。CPU只测已有CSV字段到报警，不含抓包／流特征提取。
5. 旧V多轮开发使用，三seed共享拆分；本轮仍是开发证据。固定系数筛查不称调优后的SOTA；新未使用来源确认仍需另做。

一手原文／固定作者源码见[关系KD审查](research_alignment_20261104/new_design_review.md)和[直接pAUC审查](research_alignment_20261104/direct_pauc_source.md)。
