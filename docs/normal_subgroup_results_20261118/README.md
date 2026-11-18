# R31 结果复核材料

本目录是已有回传的离线分析，不是下一次训练入口。主结论见[结果与处置](../NORMAL_SUBGROUP_RESULTS_20261118.md)：执行有效、候选失败、全部回退原 R30。

| 材料 | 内容 |
|---|---|
| `training_audit.md/json/py` | 96 个验证事件、保护判定、损失/梯度/采样及逆序幅度 |
| `effect_audit.md/json/py` | 6 份最终工件、C/E 分数、固定判决、原 TP 身份和 15 单元预算 |
| `effect_audit.csv` | 6 模型 × 15 单元的 90 条记录 |
| `effect_audit_protocol_family.csv` | 逐单元协议与类别分母 |
| `effect_audit_fp_identities.csv` | 最终误报身份与 R30 继承关系 |
| `effect_audit_cross_seed_fp.csv` | 333 个跨种子误报并集身份 |
| `design_interpretation.md/json`、`audit_design_interpretation.py` | H 主导分位、目标几何、随包解释纠正、固定作者源码依据 |
| `integrity_cost_audit.json` | 正式离线核验、207 份源码绑定、配对合同、资源与成本 |
| `source_receipt.json` | 外层与所取内层文件的原始 SHA-256 |
| `trajectory_summary.png`、`plot_trajectory_summary.py` | 三种子 H / 最高 1% Hq 的缓存轨迹图 |
| `effect_audit_path_replay.json` | 效果复算脚本改用显式路径后的结果一致性核验 |

完整回传保存在项目 `实验归档/20261118_正常子群训练结果/回传`，R30 比较工件在 `实验归档/20261117_第三十轮模型训练重设计/回传`。本目录不复制模型、F/V 数据及 C/E Parquet。

本目录保留审计文件原字节，禁用 Git 自动换行转换，使 `delivery_manifest.json` 中的 SHA-256 在检出后仍可核对。

## 本机复算

依赖 Python、NumPy、pandas、scikit-learn、PyArrow；绘图另用 Matplotlib。脚本只读取已经保存的数据和工件字节，不加载模型或执行训练。

```powershell
$thesis = 'E:\OneDrive - Coventry University\中传\毕业论文'
$auditOutput = Join-Path $thesis '工作\正常子群结果分析_20261118'
$returned = Join-Path $thesis '实验归档\20261118_正常子群训练结果\回传'
python docs/normal_subgroup_results_20261118/training_audit.py --archive $returned --output $auditOutput
python docs/normal_subgroup_results_20261118/effect_audit.py --thesis-root $thesis --output $auditOutput
```

`audit_design_interpretation.py` 使用文件顶部所列本机固定归档路径；`plot_trajectory_summary.py` 读取同目录 `training_audit.json`。异机复算前须指向相同哈希的归档，不能用训练脚本补生成资料。原始随包说明保留在归档 `来源` 下，与本次复核结论分开。
