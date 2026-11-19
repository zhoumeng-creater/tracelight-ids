# fox、santos 完整来源与网络装配

2026-11-19 已完成四个指定 PCAP 的完整下载、CRC32/SHA256 校验，以及固定 254 个事件的实际网络关联。四件原始文件共 **3,000,244,737 B**，保留在 `source/`；成员、下载范围和 SHA 见 `selected_pcap_download_plan.json`、`pcap_receipts/`。未训练、评分或随机拆分流。

## 实际网络交付

| run | 固定事件 | 全候选连接 | 可用正例连接 | B 级条件正常连接 | 未定连接 |
|---|---:|---:|---:|---:|---:|
| santos | 133 | 562 | 32 | 439 | 91 |
| fox | 121 | 660 | 17 | 430 | 213 |

连接级正例构成：santos：webshell_command=30, attack_download=1, reverse_shell_channel=1；fox：webshell_command=15, attack_download=1, reverse_shell_channel=1。

四个攻击下载事件都有真实网络候选。WPHashCrack 对应 GitHub 请求和 codeload 传输两条连接；它们可以共同组成一次动作，不是假定互斥的两种解释。事件的 one-of 最小/最大界只作为保守包络。rockyou 有 actor 完成回执、精确 GET、同窗口端点和 SF 连接；若 Zeek 缺少响应状态字段，记录其缺失，不补写 200。

反向 shell 的原始候选全部保留。明确 REJ、无完整握手且双向应用数据为零的尝试，只从“成功反连”的 `evaluation_candidate_flow_ids` 中排除，并写入逐 ID 的 `candidate_exclusions`。真实双向应用数据连接与 actor 接通回执关联；失败尝试不参与成功通道的评估上界。

`network_mapping/<run>/` 中：

- `events.jsonl` 保留每个动作、原始候选集、评估候选集、排除依据和日志证据。两组共 254 条，全部有网络候选。
- `links.jsonl` 保留每条候选关联及端点、时间、HTTP/TLS/传输证据。
- `connections.jsonl.gz` 只包含事件涉及的全部候选连接，`label` 为 0、1 或 null；raw8 保持规范化表原值，None 和负数哨兵不会取得训练标签。
- `summary.json` 给出数量、正例种类和完整背景表路径。

完整背景由 `../ait_extracted/<run>/<capture_host>/connections.jsonl.gz` 提供。按 `flow_id` 合并本目录连接标签；没有事件关联的背景保持 null。每件捕获内有非本机连接，不能把发起端一律作为监控主机。`capture_host_inventory.json` 按原始 facts 接口列出全部本机 IPv4/IPv6、源行号和 SHA，供主机视角筛选。

## 标签依据

原始来源是作者 [AIT Log Data Set V2.1](https://zenodo.org/records/19483937)，许可 CC-BY-NC-SA-4.0。官方明确规定：配套标签文件中未列出的日志事件可视为正常；仅在作者给出的仿真区间和对应日志/标签范围内使用这一规则。来源说明存于 `metadata/zenodo_record_19483937.json`，其 SHA 和全部服务日志原件 SHA 均绑定在连接的 `qualification_reasons.service_provenance`。

正常资格检查完整连接期间所有潜在 Apache 请求，而非只检查一个正常动作。还要求真实正常动作锚点、无攻击标签/正例关联、SF、完整 raw8，以及整条连接处于仿真区间。共享代理后的 TCP 事务归属不能仅由代理 IP 唯一确定，因此仍标 **B 级条件正常**。

已取两组全部 34 份正常 actor 状态机日志，重新检查 86 个固定 ownCloud 事件。它们各自仍只有一个同时间、同对象的正常 actor 候选；其他 actor 候选完整保存在 `all_actor_attribution.json`。这核实了动作归属，不等于精确识别某条 TCP 事务。254 个目标事件没有增加。

## 原始捕获与八字段表示

原 PCAP 存在 IPv4 声明长度大于实存帧字节的记录。早期四次首 1 MiB 探针保留于 `capture_probe_summary.json`，属于来源检查历史，不能代表攻击流是否完整。

后续规范化采用已验证的声明长度表示：仅对符合规则的记录提高 PCAP `orig_len`，原始包字节和 `caplen` 不改。Zeek 8.0.11 `-C` 从完整捕获导出；实际 IP 字节字段表示声明长度计量，payload 缺口仍然存在。各件规范化、报文头关联及导出的完整收据位于 `../ait_extracted/`。本目录使用这些已冻结表，不自行补 payload 或填零。

## 重跑与分配

`fetch_selected_pcaps.py` 默认刷新计划和完成收据；定向下载需给出 `--execute --run santos|fox --host intranet_server|cloud_share`。`assemble_action_candidates.py` 重算原始 actor/服务日志关联；`associate_all_normal_actors.py` 复核全部正常 actor；`map_network_events.py --run all` 重算网络关联；`finalize_delivery.py` 读回校验并刷新交付文档和 SHA 收据。

整攻击链 ID 为 `ait:santos:wpdiscuz_complete_chain`、`ait:fox:wpdiscuz_complete_chain`。实际攻击时间为 santos 01-17、fox 01-18，F/V 按整链时间顺序交主合同登记。两个 run 来自同一攻击实现家族，命令、下载连接和重复尝试不能充作独立攻击链。harrison 与 OpTC Meterpreter 预留材料未被本目录读取。
