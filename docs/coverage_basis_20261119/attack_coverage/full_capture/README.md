# Unraveled 完整来源装配

完整原件、固定 Zeek 提取和全部作者流关联已完成。该版本替代此前 `network_mapping/` 的两个定向片段；片段仅保留为历史和关联别名，不重复进入统一表。

## 实际数量

| 层级 | Normal | C2 | 其他攻击 | 未知 | 合计 |
|---|---:|---:|---:|---:|---:|
| 目标主机作者流记录 | 12,100 | 473 | 9 | — | 12,582 |
| 目标主机 Zeek 连接观测 | 12,308 | 473 | 11 | 32 | 12,824 |

全捕获保留 47,032 条连接、45,960 条合法作者流和 47,303 条候选关联。全部合法作者流的双向包数、双向 L2 字节和毫秒首末包时间均与原包核对通过。目标主机 12,582 条作者流中，12,466 条有一个候选，116 条有多个候选；全部保留。计数差异来自 NFStream 与 Zeek 切流方式不同，不把作者流、物理连接和攻击动作混为同一层级。

473 条作者 C2 全部映射到 473 条连接。主事件表仍只有 **1 段 C2 活动事件**，另有目标主机 Normal、其他攻击和其他主机 Normal 观测集合，共 4 个聚合事件。它们不代表 473 次已知执行命令，也不代表多个独立攻击链。

所有数据使用 `unraveled:apt_2021_single_campaign` 作为 campaign/source_group。当前材料是已开开发来源，可支持新增 C2 实现的监督；F/V 角色由主分配清单记录，本装配器没有拆分攻击链、训练或评分。攻击下载动作仍未证实。

## 标签与输入字段

- `label`、`author_attack_label`：作者二分类事实。其他攻击也可为 1。
- `target_kind`：`normal / c2 / other_attack / unknown`。
- `target_label`：仅 normal/c2 为 0/1；其他攻击和未知为 null。目标主机分别为 12,308 / 473 / 43。
- `author_activity`、`author_stage`、`author_signature` 及 `author_annotation`：全部候选的原作者字段值，未猜具体框架名称。
- `raw8`：固定 Zeek 8.0.11 `-C` 的八个字段。`feature_eligible` 和 `raw8_missing_fields` 与标签分别保存。

目标 Normal 中 12,235 条 raw8 齐全，73 条存在缺失，仍保留作者正常事实；473 条 C2 的 raw8 全部齐全。没有 SF 门槛，重置、单向、超时连接也可保留来源标签。32 条未知没有合法作者注释候选，没有补成 Normal。177 条列数异常作者原行单列保存，没有猜移列后的标签。

原包关联按双向五元组和原作者毫秒时窗构建候选，同时保留 Zeek 从首包到实际导出时刻的所有候选。每个原包都以**原 PCAP SHA256、全局记录编号、字节偏移**确定身份；候选包集合和关联摘要可从 `packet_membership.jsonl.gz` 复算。来源标签一致、候选原包覆盖闭合时，可给一对多/多对一集合来源标签；混合或未覆盖保持未知，不要求切流逐一相等。

C2 聚合事件的 `association_mode=all_confirmed`、`candidate_set_complete=true`，范围是**这一个捕获中作者显式标注的 473 条 C2 流及其观测包**，不代表六周整个 campaign 或每条连接已完成全部后续通信。

## 正常业务分层

| 作者应用类别 | 作者 Normal 流 | 对应 Normal 连接 |
|---|---:|---:|
| DNS | 4,386 | 4,388 |
| HTTP | 1,593 | 1,645 |
| NTP | 624 | 884 |
| 其他 TCP | 56 | 27 |
| 其他 UDP | 262 | 263 |
| SSH | 51 | 20 |
| TLS | 5,128 | 5,081 |

这些是作者显式 Normal 的应用层分组，不把每条流称作已证独立业务动作。正常事实不要求逐条额外提供进程日志。源中未标注、Background、列数异常和未闭合关联仍不能当作正常分母。完整表给出此捕获、此主机的已知正常观测集合，不声称覆盖自然环境全部正常业务。

## 原件末尾与输入质量

完整下载原件为 3,518,820,352 B，SHA256：
`6da54d26f3396e78c067ce8c66dfee5ba47c48c331394d0f3e2f782cd2a15813`。

S3 对象本身在最后一条 record 中途结束：偏移 3,518,820,278 处记录头声明 caplen=orig_len=1,251，但文件只剩 58 B 帧数据。原件和第一次 Zeek 失败日志保留。派生副本仅移除这条不完整 record 共 74 B，前面 **2,592,965 条完整 record 的记录头和帧字节全部原样保留**，没有修改 caplen、orig_len 或 IPv4 长度；派生 SHA256 为 `2111f2b2ef43d6364d90731c4f55fc5c25cf899e994f3d3aefe9eadf39851ec3`。

受影响连接为：
`unraveled:20210621:net1013x:545013ee97aaea0602462426cda3ed03`，即 `10.1.3.8:54992 ↔ 10.8.10.84:443`。保留作者 C2 标签，标记 `partial_capture=true`；`raw8_observation_scope`、`partial_capture_basis` 说明输入仅来自实际完整 record。其 orig/resp 包数为 7/6，IP 字节为 965/419；没有补计最后一包或恢复缺失内容。

全捕获起止时刻为 UTC 2021-06-21 07:08:58.198585 至 2021-06-22 07:29:02.143561，不是 UTC 自然日。历史在捕获开始处冷启动；`actual_available_at` 与 `available_at` 均为 null，保留的 `replay_export_time` 仅是离线 Zeek 导出时钟。

## 交付与复跑

- `connections.jsonl.gz`：全部背景连接和稳定 ID。稳定键由原完整 PCAP SHA、原始协议/端点/端口、微秒首包时刻生成；保留 Zeek UID。
- `events.jsonl`：4 个来源观测集合，其中 C2 仅 1 段；`links.jsonl`：全部候选及 `source_annotation_id`。
- `author_flow_matches.jsonl.gz`：全部 45,960 条作者行；`target_host_author_flow_matches.jsonl.gz`：目标主机完整 12,582 条匹配表。
- `target_host_normal_business_strata.json`：正常分层与实际连接 ID。
- `packet_membership.jsonl.gz`、`packet_stats.json.gz`、`packet_links.jsonl.gz`：原包身份、包集合摘要及关联证据。
- `old_slice_aliases.jsonl`：335 条片段记录通过五元组/首包精确对应，46 条保留未匹配；不靠猜测补别名。完整版本替换旧片段，而不是合并后双计。
- `assembly_receipt.json`、`extraction_receipt.json`：状态、计数、哈希及固定运行环境；原件、派生及尾记录收据在 `../source/unraveled/full_capture/`。

在父目录依次运行 `download_full_pcap.py`、`export_full_capture.py`、`assemble_full_capture.py`。下载可按范围收据恢复；已有完整提取和包映射可复用。最终数据表已冻结，README 更新不改变核心表哈希。
