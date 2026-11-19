# 当前统一证据表

已实际合并 47,285 个事件、209,722 个唯一观察身份和 102,409 条事件—流关系。本表保存当前事实与候选分配，不注册新的训练、选模或校准批次。

主表为 `events.jsonl.gz`、`observations.jsonl.gz`、`event_flow_links.jsonl.gz`、`split_assignments.jsonl.gz`。`flow_rows.jsonl.gz` 是 observations 的同身份精简视图，只提供 raw8/features8；两表不能相加计数。没有导出 features27 或历史摘要。

正常业务保留全部 255 个 episode、16,374 个服务连接候选；其中 7,483 条实际 episode—流边进入主 links。没有 episode 的候选仍在 `service_connection_candidates.jsonl.gz`，没有虚构事件。

正常标签证据成立的连接为 3,475 条；观察质量、model_input_ready、calibration_candidate 与 label 分列。SF 状态和 episode 显式闭合不是正常标签的必要条件。无正常标签证据的候选保持 null；正常事件不会自动给所有关联流贴标签。

旧 C25 的原角色和标签保存在 `historical_C25.jsonl.gz`，当前观察不复制。旧 v1 的 365/618/270 分配保存在 `historical_normal_v1_assignments.jsonl.gz`；当前正常分配取 v2。current_role 与 historical_roles 分开；C_candidate 类角色是候选身份，不等于已采用校准批次。

SSH 接入 58 个事件、18 条关系，7 个引用连接全部复用基底身份；仅 1 条新增完整正常控制候选标签，来源为 fox。其余命令尝试、未结束会话与候选关系保留，未扩展正常整流标签。新 SSH 材料当前为 opened_source_development，保留原来源 F/V 关联。

Unraveled 当前接入状态：`full_capture_ingested`。旧两片段只保存在 historical_attack_slice_records/aliases 和 exposure 引用中，不与整场观察相加。整场接入时整个 campaign 使用 opened_source_development，不能拆成 F/V。

所有主观察标记 new_batch_registered=false、training_admission=false、calibration_admission=false；这表示本次合并没有创建实验批次，不否认已成立的开发标签。AIT/Unraveled 的 replay_export_time 是离线 Zeek 时钟；实际 detector 接收时刻没有被补造。

重建基底及正常/SSH：`python merge_current.py`。完整攻击来源冻结后：`python merge_current.py --attack-source ../attack_coverage/full_capture`。脚本按源稳定身份核对 raw8、合并证据和引用；SHA-256 见 source_receipt/output_receipt。拟合、评分、阈值调用均为 0。
