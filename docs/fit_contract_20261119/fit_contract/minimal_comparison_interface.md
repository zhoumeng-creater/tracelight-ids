# 固定实验室入口

源码：`NEW/scripts/fit_contract/run_minimal_comparison.py`；合成测试：同目录 `test_minimal_comparison.py`。真实训练位置为实验室 RTX4090D 工作站，sklearn HGB 使用 CPU。

参数为 `--model-rows`、`--inference-rows`、`--events`、`--normal-units`、`--output`、`--threads`。四个输入均支持 JSONL 或 JSONL.gz；后三个数据输入可省略，实际交付应提供以取得完整动作/正常单位报告。线程数必须显式给定，两臂共用；输出目录须为空以保留每次执行记录。

## 模型与推理表

每行包含 `flow_id`、`dataset_role`、`label`、`features8`、`features27`；两臂的前八维必须一致。JSON null 原样转换为 NaN，HGB 原生处理，未填零。可附 `stable_flow_id`、`clock_basis`、`context_group`、`replay_export_time`，评分表保留这些字段。

model_rows 只包含 F/V/C 的明确 0/1 标签；F 两类均需存在，C 固定 25 条 label=0。inference_rows 接受 F、V、C、C_background、external_development，label 可为 null。未标注动作候选照常评分，但不进入拟合或正常分母；保留确认角色拒绝评分。相同 flow_id 两表合并只评分一次，特征/角色/标签冲突则报错。

## 动作与正常单位

事件每行：`event_id, label, kind, candidate_flow_ids, association_mode, candidate_set_complete`。

- `one_of`：仅一个候选为真实连接且候选集合完整，确定命中取所有候选告警，可能命中取任一候选告警。
- `all_confirmed`：列出的连接均已证属于动作，任一连接告警即命中。
- 集合不完整、无候选或存在未评分候选：保留身份并标 unresolved，不造上下界。未知动作仍应保留在统一原件；二值评价文件由装配方明确筛选。

正常单位每行：`unit_id, flow_ids`，可另附 count/budget。入口从具体身份重算分母和 `floor(n/100)` 预算，不使用外部计数字段替代成员。每个成员必须有 label=0 的评分记录；未知不能冒充正常。重复成员、重复单位及缺失正常分母成员在拟合前报错。

## 固定执行与产物

先写输入 source_receipt，再只用 F 拟合 HGB8、HGB27 各一次。固定参数 200 轮、31 叶、学习率 0.1、early_stopping=False、seed=20261111。保存两份 joblib 后，才计算各自 C25 分数并写 thresholds.json；阈值为最大正常分数，严格大于才告警。之后评分剩余登记身份，每身份每臂一次。

两臂均报告；V 不选臂、不选轮、不再拟合。C 校准早于评分顺序，但其历史日期晚于 V，因此 V 是开发回测。外源是已开历史开发，保留组不评分。C 只有经验零误报规则，没有 95% 总体误报保证。

输出包含两工件、逐 ID scores.jsonl.gz、动作 event_results.jsonl.gz、正常单位 normal_unit_results.jsonl.gz、逐角色 metrics.json、阈值、输入/输出回执及 cost.json。F 指标是训练诊断。cost 只记这次实际 fit/score 的时间、行数、线程与模型体积，不作为独立延迟基准；上下文导出成本引用装配方另存的 receipt。

## 真实表只读预检

加 `--inputs-only` 只调用相同的 prepare 与 metadata 检查，不构造模型、不拟合、不评分、不定阈值；无需 `--threads`。输出 source_receipt.json、inputs_audit.json、output_receipt.json。source_receipt 包含输入哈希；audit 包含 F/V/C/外源角色标签数、8/27 维检查、缺失值数、每个事件候选引用及每个正常单位分母检查，并明确记录零模型构造/拟合/评分/阈值计算。事件缺失引用如实列出，不生成假分数或假命中。

真实表预检和真实拟合使用不同输出目录；预检产物不是模型工件。当前 7 项合成单测及 Ruff 通过，详情见 minimal_comparison_checks.json。

配对产物 `paired_decisions.jsonl.gz` 从已生成分数直接导出，每个 flow_id 一行，含两臂 score/pred/threshold、label/role、新增/移除告警。metrics.json 的 paired_by_role 分别统计 TP 新增/丢失、FP 新增/移除；未知标签只统计告警变化。没有追加评分、选择模型或重定阈值。
