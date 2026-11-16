# 第二十六轮冻结权重的训练诊断

本批回答一个问题：混合训练模型在自己的 source/target 训练支持上已经失效，还是主要在后续 B 开发单元和 D 迁移单元失效。复用 R26 权重、标签、校准阈值和 B/D 连续分数；新增拟合为 0。

[固定登记](../configs/training-diagnosis-20261116/registration.json)列出八个权重：`target/mixed × ports_zero8 × HGB/MLP17/MLP29/MLP43`。每个权重只对同一批 18,874 条训练联合行评分一次，共 150,992 个逐行分数。source 为 8,192 行，target 为 10,682 行。HGB 在 CPU 上评分，MLP 使用实验室 CUDA；B/D 直接读取旧分数，不重复模型前向。八个权重顺序执行，逐权重保存分数与回执，恢复时复用已完成项。

## 已有证据与需要补的判断

本机已从原始归档恢复训练数值并核对身份、标签与权重。双端口置零后发现的 63 组精确输入标签冲突位于 source 内部；没有发现 source 与 target 在完全相同输入上互相矛盾的组。因此不能据这 63 组认定混合训练发生了源域与目标域标签冲突。

本批分别报告原始输入和各 MLP 实际标准化 FP32 输入的冲突、原权重下的经验 BCE 下确界、最终训练 BCE，以及训练/后续 B/D 的分组错误。训练损失与该下确界的差距同时包含函数类限制、优化与正则化影响，不能直接叫作优化误差。即使最终训练损失偏高，也不能自动判定模型太小、训练轮数不足或 AdamW 有问题；这些原因需要后续有针对性的区分，不能由本次结果自动触发加宽、续训或阈值搜索。

## 运行入口

默认解释器为原 route-core 环境：`D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe`。正式执行前使用当前固定 checkout；`-R26Run` 指向保留的 R26 运行目录，目录中应有 `protocol.json`、`models`、`evidence/replay` 和 `local/training/mixed.parquet`。

```powershell
.\scripts\RUN_TRAINING_DIAGNOSIS.ps1 -PlanOnly
.\scripts\RUN_TRAINING_DIAGNOSIS.ps1 -CheckOnly
.\scripts\RUN_TRAINING_DIAGNOSIS.ps1 -R26Run 'D:\zhoumeng\Graduate\tracelight-support-diagnostic\local\support-diagnostic-r26'
```

`-PlanOnly` 由 PowerShell 直接读取登记，不要求 Python 或 R26 目录存在。`-CheckOnly` 运行本次诊断、训练输入恢复和原 `test_support_diagnostic_*.py` 合成测试；可用 `-Python` 指定本机兼容解释器。

省略 `-R26Run` 时只检查相邻的 `tracelight-support-diagnostic`、`task-reset-20261115`、`task-reset` 及本 checkout 下对应 worktree 的固定路径，并核对 protocol SHA。恰好一个匹配时使用它；零个或多个匹配均提示明确传参，不递归搜索全盘。

输出默认写入当前 checkout 的 `local/training-diagnosis-r28`；用 `-OutputRoot` 指定其他独立目录。输出不能与旧 R26 运行目录重叠。`-Threads` 默认为 4。

## 只有回传包时恢复训练输入

R26 小回传包含权重、训练身份及 B/D 输入分数，但未包含原 `local/training/mixed.parquet`。已有原运行目录时直接复用该文件。只有回传包时，先用保留的原始缓存恢复输入，再显式传入 `-RecoveredTraining`：

```powershell
$py = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe'
& $py .\scripts\recover_r26_training_rows.py `
    --r26-run 'D:\path\to\r26-return' `
    --raw-dir 'D:\path\to\response-evidence-20261112' `
    --output 'D:\path\to\recovered-training.parquet'
.\scripts\RUN_TRAINING_DIAGNOSIS.ps1 `
    -R26Run 'D:\path\to\r26-return' `
    -RecoveredTraining 'D:\path\to\recovered-training.parquet'
```

恢复只需两份 CTU gzip 和 `TQH-C2_labels_v110.zip` 中的六个已登记训练单元。脚本只解析所选 CTU 物理行；为核对来源，仍读取完整解压字节计算 SHA。gzip 原始日志内容按 `protocol.source_specs` 校验，压缩载体与实验室不同会单列记录；TQH 包及成员按协议校验。输出包含原始八字段、身份、域、标签和 mixed 权重，不伪造抽样前的累计日志送达时间，也不声称逐字节恢复原 parquet。

缺失缓存、身份错配或 SHA 不符会报告具体文件。程序不下载、不重训；需要修复的是缺失输入或目录选择。

## 本机只分析输入与已有分数

```powershell
.\scripts\RUN_TRAINING_DIAGNOSIS.ps1 `
    -Python 'D:\python\python.exe' `
    -R26Run 'E:\path\to\r26-return' `
    -RecoveredTraining 'E:\path\to\recovered-training.parquet' `
    -OutputRoot 'E:\path\to\training-diagnosis-inputs' `
    -InputsOnly
```

`-InputsOnly` 不构造或执行神经网络。它读取已保存的归一化张量，检查恢复输入能否精确复现各 MLP 的训练 mean/scale，统计原始及 FP32 输入冲突，并复算已保存的 B/D 分数。输出状态为 `inputs_and_cached_scores_complete`，训练分数诊断仍未完成；不能把这个状态写成八权重实验完成。

## 输出与判读

| 输出 | 内容 |
|---|---|
| `evidence/input_conflicts.json` | 原始输入及模型实际输入的精确冲突、经验损失下确界 |
| `evidence/cached_BD.csv` | 旧 B/D 连续分数的分组损失与错误 |
| `evidence/training_scores/*.parquet` | 八个冻结权重在共同训练联合行上的分数与身份 |
| `evidence/training_metrics.csv` | 按来源、类别等分组的最终训练错误 |
| `evidence/final_training_objectives.json` | 原训练权重下的最终 BCE 及其与经验下确界的差距 |
| `outcome.json` | 输入分析或完整诊断的实际完成状态 |

若模型已在自身训练支持上分不开，应先解释训练目标、输入冲突和剩余拟合差距；若训练支持表现良好而 B/D 恶化，应研究支持覆盖与迁移。两种情况可以同时存在。本批是已开数据的机制诊断，不提供新的独立确认，不改变旧阈值，不自动选出下一种方法。

完整执行结束后回传生成的 `training-diagnosis-results-*.zip` 及其 SHA 文件。保留原 R26 权重和原始缓存；本批不重新打包它们。
