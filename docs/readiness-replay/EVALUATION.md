# 固定决策修复的逐行评价

评价入口不训练模型、不重新评分、不登记阈值。它读取同一批原始与修复后布尔判决，比较日期预算、攻击得失和事件时延。

`prepare_context(features, truth)` 为一个来源建立可复用上下文；同一来源的五组学习器共用该对象。特征和真值 DataFrame 在复用期间保持不变。`evaluate_frames(features, scores, truth, old_decisions, repaired_decisions, *, source, group, output, context=None)` 返回本组所有输出路径。`evaluation_scopes(old_decisions, repaired_decisions)` 可供排序诊断直接复用。

## 分期和计数

| phase | 使用的行 |
|---|---|
| all | 全来源 |
| startup | 修复判决标记为未就绪的行 |
| after | 修复判决标记为已就绪的行 |
| original_startup | 原 v1 启动行 |
| original_after | 原 v1 启动期后的行 |

每个 phase 再按 truth.window 的 UTC 日期统计。all 包含启动期，两个版本的分期可能交叠，不能把这五项相加。评价器直接采用封存判决中的 phase；就绪条件由独立 readiness 模块决定。

预算表分别保存 `version=original/repaired`。正常分母为该日期、phase 内全部 Benign 行，整数预算为 `normal // 100`，通过条件为 `fp <= cap`。没有正常流时，FPR 与通过状态为空。Unresolved 的行数和告警数另列。

paired、services、events 均以修复后的 policy 为 after，比较三个原判决参照：`same_policy`、`raw_old`、`add_quality_old`。comparison 字段区分它们；例如 policy 本身为 raw_old 时，两个参照可以得到相同数字，不能重复汇总。before、after、gained、lost 满足 `before + gained - lost = after`。

paired 保留 family、native_group、evidence_level、exposure。services 按协议、目的端口、攻击 family 保存配对结果，并同时记录修复策略与原参照在同服务正常流上的误报。normal_service_fp 按原、修复两版本保存有误报的服务组。

## 事件与样本

事件编号保持原生含义。首次记录时间在**整个来源**的有效攻击事件组上确定，切到启动期、就绪后或另一个 UTC 日时不重置。输出同时保存修复策略和原参照的首次告警、相对首次记录的时延。这个起点是日志中首次观测，不是作者攻击启动时刻；跨日期、学习器和 scope 重复出现的事件编号不代表新的独立攻击。

`rare_rows.parquet` 对 C&C、ExternalC2、PureCC、C2、Download、FileDownload 全部逐行保留，不抽样。每行含 UID、row_id、事件、证据层、暴露身份、原/新 phase、数值特征、输入分数及全部原/新策略标志：

- `feature__*`：数值特征；
- `score__*`：传入的数值分数；
- `original__*`、`repaired__*`：封存策略标志。

其他类别在每个日期、family、native_group、策略、参照的 gained/lost/both_negative 中最多保留前三行。`examples.csv.gz` 保存配对关系和前后标志，`example_values.parquet` 按 row_id 去重保存对应数值。两表通过 row_id 连接。`sample_rows(...)` 提供同样的宽表构造入口，供其他诊断复用。

## 效率和验证

类别、服务和事件分组只在 prepare_context 中编码一次。每组学习器按原/新分期的四种组合做 NumPy 聚合，再组合出五种 scope；不在“每个日期 × phase × policy”内反复调用 pandas groupby。

12 项合成检查通过，包括与原 R25 judge 的预算、paired、services、events、normal_service_fp 逐列一致；原预算的 all/original_startup/original_after 可直接映射回旧 all/startup/after。检查还覆盖正常数为零、空原生类别、整数预算边界、同分时钟、事件跨 phase/UTC 日、全部稀有行保留及身份错位拒绝。

本机合成探针使用 18 个策略，复用上下文评价两组，计时包含文件写入：10 万行上下文 0.058 秒、单组评价 0.195–0.209 秒；50 万行、扫描占约 60% 时，上下文 0.421 秒、单组评价 0.910–1.207 秒，采样进程峰值 RSS 518,729,728 字节。探针没有读取真实来源、模型或 GPU。复算脚本与原始数字分别为 `synthetic_evaluation_benchmark.py`、`synthetic_evaluation_benchmark.json`；实验室整批耗时由主入口另行记录。
