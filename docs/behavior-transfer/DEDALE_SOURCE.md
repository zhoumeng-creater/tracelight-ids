# DEDALE 的来源、真值与前向回放

本批登记 DEDALE V2 的绿色内网传感器，保留 2024-12-23 至 2025-01-19 全部 28 个原生日文件。前 14 个文件为作者正常业务前缀，D15（2025-01-06）已用于事件核查和开发，D16–D28 是尚未评分的同一试验台后续时段。后续时段包含同一攻击链的延续，其结论属于同场前向验证。[作者发布页](https://doi.org/10.57745/Y5JLDG) · [作者方法](https://dedale.inria.fr/download.html)

作者将动作的已知时间、地址和部分端口用于连接标签匹配。客户端和服务端日志能给部分动作提供独立佐证。本文档记录了实验前实际核验程度，未来评分结果应按证据层报告。[冻结注册](../../configs/behavior-transfer-20261114/dedale.json) · [作者逐动作标注代码](https://gitlab.inria.fr/mlanvin/dedale_labeling/-/blob/master/network_log_labeling/labelling_script_on_single_conn.log.py)

| 登记角色 | 文件 | 本批用途 |
|---|---|---|
| prefix | D01–D14 | 全量正常时段的误报与漂移对照 |
| development | D15 | 已打开的事件开发案例 |
| forward_same_testbed | D16–D28 | 冻结方法在后续原生日文件上的检验 |

每个原生日文件从空历史启动，文件内保留原始 conn.log 物理顺序，以连接结束时间的累计最大值构建回放时钟。window 按该回放时钟的 UTC 日期保存。长连接和日志尾部清理可能跨到其他自然日，因此一个原生日文件不保证只含一个 UTC 日期。D15 原始头部的 #open 是离线 Zeek 处理时间，不能用作攻击发生时刻。实验室报告会保留原始起止时刻、结束时间逆序数、最大时钟滞后和并列数，便于判断这一回放假设。

作者的 conn_labeled.csv 按起始时间排序。程序读取完整标签索引，再按原始 conn.log 的 UID 重排真值，核对 UID 集合、唯一性、两端地址、两端端口、协议和起始时刻。观测中的未知时长继续保存为 -1，原生 ip_proto 用于区分 IPv4 ICMP 与 IPv6 ICMP。地址只供历史关系分组，UID 只供身份核验，标签、事件描述、URI 和主机日志都存于独立真值或审计文件。

| 原生标签或动作 | 输出 family | 证据 |
|---|---|---|
| label=0 | Benign | author_benign，保留全部正常流 |
| label=2 | Unresolved | author_related_rule，保留攻击相关含义 |
| label=1 且明确为下载动作 | Download | author_action_rule，按具体动作映射 |
| label=1 且明确为信标或控制轮询 | ExternalC2 | author_action_rule，按具体动作映射 |
| 其余 label=1 | OtherAttack | 原始 step、technique、comments 保留 |

native_type 为简短动作名称，native_detail 保存作者原文，event_id 使用原生日文件及作者 step，表示作者动作阶段覆盖。一个 step 可能包含多次预定信标，因此其数量不能充当独立任务数或每次命令数。阶段覆盖与流检出分别计数。同为 T1104 的信标与可执行文件传输有不同的动作描述，程序分别映射。遇到未登记的新描述会保留在 OtherAttack，避免仅凭“C2”或 ATT&CK 技术编号硬归成下载。

D15 有五个已核查 UID。CI7cJo10RfMA3xxRC5 的 HTTP 请求返回 200，文件 F7QbbjrQy2jkZEEke 为完整的 1,105 B ZIP。受害机 PowerShell 记录同一下载地址，随后 Sysmon 记录 ZIP 创建、程序落盘和启动。实验室准备程序会重新核对这些记录，成功后标记 host_and_network_confirmed。CnOKRT2GLSyyTe1kRf 为 TLS 大响应，随后出现可执行文件落盘，标记 host_correlated_tls。其余三条控制连接标记 author_action_rule_tls，表示作者动作标签与成功 TLS 连接相符。只有 D15 取这些辅助日志，其余日期维持作者动作规则层，不把标签自动升级为完整文件交付。

D15 全量作者正常 TCP/80 有 26,325 条，TCP/443 有 201,275 条。正常 TCP/80 中有 170 条响应 IP 字节达到 65,536，已核实的恶意小 ZIP 响应 IP 字节只有 1,569。正常 TCP/80 还出现 76 条 ZIP 文件记录，对应 76 个不同连接 UID 和 29 个去重 FUID，包含微软更新的 HTTP 206 分段响应。这些数值来自已打开的开发日，不用于挑选后续评价样本。攻击专用服务端上没有同日正常流，模型输入应保持既有八字段和因果历史约束。

prepare_sources(cache_dir, output_dir, workers=4) 返回 28 个来源结果，每项包含 name、source_id、role、status 及 features、truth、audit 路径。准备失败的来源返回 error，其他来源可继续。prepare_one(source_id, cache_dir, output_dir) 支持单个来源重试。

实验室按冻结 ZIP 成员目录取字节范围，所需成员合计 1,502,394,505 压缩字节，包括 28 天的 conn.log/标签 CSV、D15 三种应用日志及一份系统事件成员。完整网络 ZIP 为 3,538,305,367 字节。已经持有完整 ZIP 时，可放在 cache_dir 下复用。否则缓存各成员的 gzip 投影，中断传输保留压缩片段，重试接续读取。准备程序核对冻结目录 SHA-256、成员名、长度和 CRC-32，再保存解压内容 SHA-256。除已核主机事件成员外，注册时尚未取得所有成员的解压 SHA-256，实验室收据会记录实际值。输出特征和真值各有 SHA-256，只有两者与注册收据一致时才复用已完成步骤。

转换按原生日文件并行，最多八个进程，并按可用内存为每个进程预留 1.5 GiB。每个进程分批读取连接日志，一次持有一天的标签索引。磁盘上的 Parquet 由完整 UID 核验通过后发布，准备收据写在两份输出完成之后。

2026-11-14 已完成 D15 全文件真实准备，789,091 条原始记录与标签一一对应。原生标签 0/1/2 分别为 789,063/6/22，动作映射得到下载 2 条、控制 3 条、其他攻击 1 条。五条事件的辅助证据均复核成功。这一步没有模型预测或阈值估计。原日志有 201,471 次结束时刻逆序，累计时钟产生 632,014 次并列，最大滞后 3,598,987 ms，约一小时。历史计算和启动校准必须沿用已登记的物理回放时钟并报告这个滞后，不能把它写成已验证的实时包到达顺序。[源文件、代码与输出校验记录](dedale-D15-source-check.json)
