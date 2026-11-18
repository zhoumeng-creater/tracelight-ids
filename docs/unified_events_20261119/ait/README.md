# AIT 正常下载连接装配结果（2026-11-19）

**31 次已确认正常下载全部获得实际后端连接候选，没有丢弃事件。** 完整连接时间窗内，31 次均存在多个候选；进一步检查请求所在秒实际出现的代理出站 TCP payload，27 次对应唯一候选，4 次仍有 2—3 个候选。这里的唯一性指包中基于 SYN/端点归组的连接候选，不等于文件内容已在 TLS 中识别，也不等于整个 TCP 期间只有该次下载。

既有 intranet 全目录的 3,975 个 IPv4 TCP 候选中没有代理 `172.19.130.68`，无法关联这些下载。此次按 ZIP 成员范围取回唯一 cloud_share PCAP，未取整 ZIP，也未追加 webserver PCAP：

| 项目 | 实测值 |
|---|---|
| 官方成员 | `gather/cloud_share/logs/suricata/log.pcap.1642684647` |
| ZIP 内压缩字节 / 解压字节 | 464,709,023 / 525,862,169 |
| CRC32 | 3,074,706,299 |
| SHA-256 | `e7039504139d6cadab64ad5c1772233a04705c06757fde3251c11beefb491bac` |
| 采集起止（epoch） | 1642684647.874864 — 1643093816.233258 |
| 总包数 / packet 连接候选 | 959,549 / 18,591 |
| Zeek 8.0.11 strict / ignore_checksums 记录数 | 38,734 / 21,925 |

原件选择、范围收据、校验和代码分别见 [source_selection.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/source_selection.json>)、[cloud-source-receipt.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/source/cloud-source-receipt.json>)、[fetch_cloud_member.py](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/fetch_cloud_member.py>)。所有事件沿用已打开的 russellmitchell 开发材料身份。

**可直接并入统一事件表的文件：**

- [normal_event_connections.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/normal_event_connections.json>)：36 个原始事件 ID，含 31 个已确认正常事件、5 个身份未明的 VPN 事件；保留前一层动作—服务端—代理证据、所有包级候选和两臂 Zeek 链接。
- [event_flow_links.csv](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/event_flow_links.csv>)：事件→连接长表，包含动作窗、请求秒、请求秒 payload 的独立标记。多个事件指向同一连接时不会复制成独立流。
- [referenced_connections.jsonl](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/referenced_connections.jsonl>)：上述所有事件候选实际引用的 1,050 条 Zeek 记录及 raw8、TLS、时间字段；完整 60,659 条记录另存 `connections.jsonl`。
- [cloud_packet_connections.csv](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/cloud_packet_connections.csv>)：原始包解析的全部连接候选及首末包时间、SYN、端点、包数和捕获 TCP payload 字节数。它们不是 Zeek IP 字节数。
- [checksum_mapping_audit.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/checksum_mapping_audit.json>)：每个请求活跃 packet 候选对应的两臂记录数、完整 raw8、导出时间与 EOF 状态。

31 个事件的完整动作窗涉及 263 个不同 packet 候选，全部保留。36 个事件的全候选集合中，76 个 packet 候选关联不止一个事件，关联列表写在 `shared_with_event_ids`。后端已观察到的方向为 `172.19.130.68:<临时端口> → 172.19.130.106:443`；客户端到代理的一段仍由原动作与代理日志支持，未虚构前端五元组，未把两条腿算成两个下载事件。

四个请求秒仍多义的事件如下；没有按响应体积选出“最像下载”的连接：

| 原事件尾部 | 文件/目录 | 请求时刻（UTC） | 请求秒有代理 payload 的 packet ID |
|---|---|---|---|
| `.1:1057` | Marketing 目录 | Jan24 09:44:38 | 17178、17179、17180 |
| `.1:2782` | ownCloud Manual.pdf | Jan24 13:47:59 | 17643、17644、17645 |
| `.4:5152` | Photos/Lake-Constance.jpg | Jan21 20:17:24 | 4469、4470 |
| `.4:5488` | Photos/Lake-Constance.jpg | Jan21 20:53:06 | 4628、4629、4630 |

这些请求秒共涉及 **38 个不同 packet 候选**。ignore_checksums 中对应请求秒的 38 条记录均识别为已建立的 TLS 1.3，SNI 均为 `cloud.dmz.smith.russellmitchell.com`，两臂均无该批记录的 HTTP 明文请求。因此包内容没有直接给出文件 URI；上表并发请求无法仅凭已取原件唯一分离。27 个唯一活跃候选及这 4 个多义事件均保留在同一主表中。

**两种校验和处理产生实质不同的流字段，不能互换。** 对上述 38 个 packet 候选的完整生命周期，34 个对应 `strict 2 条 / ignore_checksums 1 条` Zeek 记录，另外 4 个对应 `3 条 / 2 条`。在请求所在秒，每个 packet 候选恰有一条对应记录可跨两臂比较，但 **38 对 raw8 无一完全相同**：协议和两端端口全部相同，持续时间、双向包数、双向 IP 字节数五项全部不同。这里同时保留两臂，没有按匹配数、字节数或模型效果选臂。

例如 `.1:673` 的 PDF 下载，请求秒唯一 packet 候选为 `#17101`，实际端点 `172.19.130.68:58380 → 172.19.130.106:443`：

| 指标 | strict 请求秒记录 | ignore_checksums 请求秒记录 |
|---|---:|---:|
| duration_seconds | 3.6331260204315186 | 93.61733889579773 |
| orig_pkts / resp_pkts | 357 / 0 | 360 / 4,416 |
| orig_ip_bytes / resp_ip_bytes | 24,299 / 0 | 24,455 / 6,366,028 |
| replay_export_time | 1643016214.302176 | 1643016305.579088 |

该 packet 候选首包为 `1643016205.66737`、末包为 `1643016299.2851`；员工状态机在 `1643016220.0218325` 报告下载完成。strict 还生成另一条后续记录，保存在同一候选下。可见动作成功、包末次出现、Zeek 导出不是同一个时刻。38 个 packet 候选所对应的完整两臂记录中，EOF/shutdown 标记均为 false；真实检测器收到记录的时间与正常标签实际反馈时间仍为 null，离线重放时钟没有被改称线上可用时间。

包记录中没有 `captured frame length < original frame length`，但有 27,914 个 IPv4 包的头部总长度大于实际捕获 IP 字节数；本轮记录这一事实，没有据此认定丢包、卸载或损坏原因。[包计数](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/cloud_packet_counts.json>)

复算使用现有固定版 Zeek/原始 packet parser，代码及哈希写入结果。下载器复用已校验原件；连接构建器复用已校验的两臂导出，不重复前向或训练：

```powershell
python 'E:\OneDrive - Coventry University\中传\毕业论文\工作\统一事件与方法落实_20261119\ait\fetch_cloud_member.py'
python 'E:\OneDrive - Coventry University\中传\毕业论文\工作\统一事件与方法落实_20261119\ait\build_ait_connections.py'
python 'E:\OneDrive - Coventry University\中传\毕业论文\工作\统一事件与方法落实_20261119\ait\audit_checksum_mapping.py'
```

两次 Zeek 导出分别耗时约 22.2 秒与 20.1 秒，CPU 时间与峰值内存记录在 `summary.json` 的 `export_receipts`。本轮新增模型拟合、模型前向及阈值搜索均为 0。ownCloud 正常文件下载与 WordPress 控制/外部工具下载的服务差异继续保留，不能由同为 443 推成相同业务。
