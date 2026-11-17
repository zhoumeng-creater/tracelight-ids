# R30结果复核证据

主报告：[结果复核与处置](../MODEL_TRAINING_REDESIGN_RESULTS_20261117.md)。对应执行源码`6ca3c92ec1ff8501f8c1a4c1ba19ce6e84517da8`，标签`model-training-redesign-r30-v1`。

- `archive_audit_integrity.json`：352份载荷、195份源码、22份收据、冻结顺序及模型字节核验。
- `training_audit.py/json/md`：190次选轮事件、缓存F/V分数组、配对随机流及原成本记录。
- `detection_audit.py/json/md`：原C阈值、6模型原决策、历史参照身份和原验收结果。
- `detection_cells.csv`：6模型×15单元原判决计数。
- `detection_identity_changes.csv`：旧参照已检出身份的新增/丢失及配对对照误报身份变化。
- `manifest.json`：以上交付文件的SHA-256，用于核对本次分析材料。

完整模型和原始分数仍在项目的`实验归档/20261117_第三十轮模型训练重设计/回传`。原R29材料从`实验归档/20261117_第二十九轮模型训练/回传`读取。累计包`TraceLight-ALL-2026-11-17.zip`保留原始内包，完整SHA为`84a4d73721ffc4794fe59f337bcfd8368afb7bbe7c477c9831ab661f8368ce93`。

以下两个脚本只复算已保存的数据，不导入模型、不执行训练或前向，不生成候选阈值。需已有numpy、pandas、scikit-learn及parquet读取依赖。`--project`指向含“工作”和“实验归档”的毕业论文根目录；路径示例与当前归档一致。

```powershell
$taskProject = 'E:\OneDrive - Coventry University\中传\毕业论文'
$taskEvidence = '.\docs\model_training_redesign_results_20261117'
$taskOutput = Join-Path $taskProject '工作\第三十轮结果分析_20261117'
python "$taskEvidence\training_audit.py" --project $taskProject --output "$taskOutput\training_audit.json"
python "$taskEvidence\detection_audit.py" --return-dir "$taskProject\实验归档\20261117_第三十轮模型训练重设计\回传" --output-dir $taskOutput
```

本轮已运行并通过：训练`all_checks_pass=true`，检测`independent_reproduction_passed=true`。复算不改变冻结注册、模型、分数或原判决；输出写入工作目录。
