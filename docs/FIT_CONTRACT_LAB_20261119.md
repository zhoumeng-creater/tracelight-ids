# 固定 HGB8 / HGB27 实验室运行

运行版本：`source-fit-contract-v1.1`。仅在既定实验室 RTX4090D 工作站运行真实拟合；本比较的 sklearn HGB 使用该工作站的 CPU。项目 `pyproject.toml` 固定 scikit-learn 1.5.1、NumPy 1.26.4 等依赖，沿用项目虚拟环境。

在实验室项目目录执行：

```powershell
git fetch origin --tags
git switch --detach source-fit-contract-v1.1
.\scripts\RUN_FIT_CONTRACT.ps1 -Threads 4
```

入口先核对四份注册输入的 SHA。默认输出到 `results/fit-contract-20261119-comparison`；非空输出目录不会被覆盖。输入已随版本库交付，不需要在实验室重新下载 PCAP 或重跑 Zeek。

若只需核验输入，执行下列命令；它不构造模型或生成分数。本机已经完成同一 Python 预检路径。

```powershell
.\scripts\RUN_FIT_CONTRACT.ps1 -InputsOnly
```

## 实际执行内容

1. 只用 santos F 的 471 条记录拟合 HGB8、HGB27，各一次。两臂固定 200 轮、31 叶、学习率 0.1、种子 20261111，不早停；缺失值由 HGB 原生处理。
2. 两个模型工件均保存后，分别评分同一 25 条 C。阈值为该模型 C 分数最大值，`score > threshold` 告警，不扫描阈值。
3. 固定阈值后评分 V、全部注册动作候选和 OpTC 外源开发。V 不参与重拟合或选轮；保留来源不在推理表中。
4. 两臂都报告。单流正常分母、动作候选下上界、未解析项和逐身份变化分开记录，不把一个多候选动作当作多次独立攻击。

25 条 C 的 1% 经验预算为零误报；这是已筛选 ownCloud 业务子集，不能据此宣称全网总体误报率为 1%。C 的日期晚于 V，故 V 是开发回测；OpTC 是已开跨源历史开发评价。

## 回传内容

回传整个结果目录，包括：

- `source_receipt.json`、`run_config.json`、`output_receipt.json`：输入、参数、环境与文件哈希。
- `hgb8.joblib`、`hgb27.joblib`、`thresholds.json`：两份模型及固定工作点。
- `scores.jsonl.gz`、`paired_decisions.jsonl.gz`：逐 ID 分数、判决和两臂新增/丢失。
- `event_results.jsonl.gz`、`normal_unit_results.jsonl.gz`、`metrics.json`：动作候选命中范围、31 个正常单元误报和角色汇总。
- `cost.json`、`summary.json`：实际 CPU 拟合/评分时间、线程、模型体积及调用次数。完整背景导出成本另见数据 `summary.json`，不把离线排序时长当作在线延迟。

解释结果时，先看控制与下载的逐事件变化和各正常业务单元预算，再看汇总指标。只有相同预算下的历史信息增益与实际成本都得到支持，才继续研究同状态预算的选择与压缩；否则保留简单方案并记录负结果。这次不追加模型、种子或窗口矩阵。
