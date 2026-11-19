# AIT 两个下载动作：载荷缺口与头部统计的分离验证

两次下载的完整载荷没有恢复，但三条连接的八维头部统计已通过可复算的截断记录表示验证。原先把“载荷不完整”直接当成 raw8 不可提取，门槛过严：本次只调整派生 PCAP 的 record `orig_len`，不改 `caplen`、时间戳或任何帧字节，Zeek 8.0.11 `-C` 即可计入全部记录和 IPv4 声明字节。原件与旧禁用结果保持原样；新表示尚未获得训练准入。

## 替代捕获点的实际结果

作者配置证明 intranet 默认网关 `10.143.0.1` 是 inet-firewall 的 ens7；防火墙 ens3 为 `192.168.230.4`，Suricata 捕获 ens3，Shorewall 将 `10.143.0.0/17` 出口 MASQUERADE 到该接口。原件来自[作者数据记录](https://zenodo.org/records/19483937)，配置、URL、CRC32、SHA256 均保存在 `source/capture_config/receipt.json`。

只取了覆盖动作时间的一个成员 `gather/inet-firewall/logs/suricata/log.pcap.1642957916`：压缩 694,128,142 字节，解压 789,200,482 字节，CRC32 `2278223961`，SHA256 `ac24fa275bab9fc462e634ca3fac076602ecf128b5f4724510152104b71888ad`。实际覆盖 epoch `1642957916.053478–1643093820.1755`，三条连接的 SYN 与 FIN 都在成员内部，不需要相邻轮转补边界。

原 3574/3575/3578 分别唯一对应防火墙 3596/3597/3598；源地址经 NAT 改为 `192.168.230.4`，端口、双向初始序号、载荷片段吻合。完整配对在 `recovered_download_observations.json`。

把所有实际捕获的 TCP 载荷都按序号合并，包括 IPv4 声明长度异常包中的实存前缀，得到：

| 原 packet ID | 实存响应字节 | FIN 序列要求的响应字节 | 实际缺字节 | 实际缺口数 |
|---|---:|---:|---:|---:|
| 3574 | 4,020 | 5,309 | 1,289 | 1 |
| 3575 | 25,242,008 | 55,637,730 | 30,395,722 | 15,935 |
| 3578 | 17,643,346 | 139,921,803 | 122,278,457 | 12,179 |

三条连接均无重叠字节冲突，对端 ACK 均确认到 FIN 后一字节。异常包之后的序号推进几乎全部等于 IPv4 声明的 TCP 载荷长度，而非实际保存的 1,420 字节：分别为 1/1、15,935/15,935、12,178/12,179。两个 TLS 连接的最初三个 record 完整，第四个分别缺少 1,130、1,129 字节；HTTP `200 OK` 的 `Content-Length=139921500`，加 303 字节响应头，恰好等于 FIN 序列要求的 139,921,803 字节。

这证明缺失的是捕获文件中的真实字节，不只是 IPv4 长度字段偏大。`actual_tcp_byte_evidence.json` 保留缺口、序号推进、ACK、TLS record 和 HTTP 头依据；`cross_capture_byte_identity.json` 进一步证明两个捕获点的所有实存区间边界及完整区间 SHA256 一致，合并两份原件也不会增加任何载荷字节。完整载荷恢复路线到此结束。

## 仅调整记录元数据的验证

[Zeek 8.0.11 的 IP 解析器](https://raw.githubusercontent.com/zeek/zeek/v8.0.11/src/packet_analysis/protocol/ip/IP.cc)在 record 原始长度小于 IPv4 TotalLen 加链路层偏移时丢弃该记录；[TCP 解析器](https://raw.githubusercontent.com/zeek/zeek/v8.0.11/src/packet_analysis/protocol/tcp/TCP.cc)分别处理声明长度和实际捕获长度，能保留载荷截断的完整头部统计。公开数据也有保留完整传输层头部的截断 PCAP，例如作者发布的 [CTU-13-Extended](https://www.stratosphereips.org/blog/2015/7/17/new-dataset-ctu-13-extended-now-includes-pcap-files-of-normal-traffic)。

本次规则为：对固定候选范围内 IPv4/TCP 头部完整、IPv4 校验和正确、无分片，且声明长度落在已有 TCP 序号及 ACK 证据范围内的记录，令：

`orig_len = max(orig_len, L2_offset + IPv4_TotalLen)`

`caplen`、包时间戳和全部捕获字节完全不变。该值表示被截断的**声明 IP 数据报长度**，不声称重新测得物理以太网帧长度；offload 聚合也意味着记录数不等于物理链路包数。

固定范围为三条下载、32 条合格控制、38 条完整 cloud 候选。派生 intranet 文件保留完整原捕获时间上下文和所有其他原记录，只修改下载候选中的 28,115 个 `orig_len`；逐条索引、原值、新值、头部条件和 ACK 上界保存在 `declared_length_representation/intranet-record-changes.jsonl.gz`。32 条控制没有改动；cloud 38 条应用同一规则后零改动，复用同 SHA 原件和既有固定导出。

原 intranet SHA256 为 `12c798074e27969ea40719700ecc8cb07ecda76e4f0751d746e02e2398067790`；派生文件为 `96f342e34f387b521999c70572915f97918b1fc7b2f95b87bb8836bafa2e808c`。独立逐记录比较确认，二者唯一差异为允许修改的 record `orig_len`。原 PCAP 执行前后 SHA256 均未改变。

只新增一次固定 Zeek 8.0.11 `-C` 导出，实耗约 12.99 秒。三条下载得到：

| packet ID | 起端/响应端端口 | 时长（秒） | 起端/响应端记录数 | 起端/响应端 IPv4 声明字节 |
|---|---|---:|---:|---:|
| 3574 | 45736 / 443 | 1.5610861778259277 | 12 / 9 | 1,242 / 5,785 |
| 3575 | 42126 / 443 | 16.899959087371826 | 6,872 / 17,814 | 380,100 / 56,564,066 |
| 3578 | 60448 / 80 | 2.544288158416748 | 8,320 / 12,529 | 432,813 / 140,574,009 |

三条连接的协议均为 TCP，端点、起点和时长均与原 `-C` 输出一致；所有记录数和声明 IP 字节数与原始头部逐包计算完全吻合。32 条控制候选对应的 33 条 Zeek 记录 raw8 完全不变；38 条 cloud 候选对应的 42 条记录 raw8 不变。候选 TCP 连接数与 Zeek 记录数分开记录，没有把尾段拆流算作新动作。

## 可交付内容与当前决定

- `declared_length_representation/validation.json`：三条下载和全部对照的旧/新 raw8、UID、时间字段、逐连接守恒及规范表示说明；`training_admission=false`。
- `declared_length_representation/intranet-declared-length.pcap`：只改允许范围 record 原始长度的派生文件。
- `declared_length_representation/representation_receipt.json` 与改动日志：源/派生 SHA、范围、逐字节不变证明。
- `verify_declared_length_representation.py`：表示规则和一次 Zeek 验证的复算脚本。
- `actual_tcp_byte_evidence.json`、`cross_capture_byte_identity.json`：真实载荷缺口及两个捕获点不能互补的证据。

结果支持把“完整载荷”与“可提取的头部统计”分开。缺失载荷、文件内容完整性及原 Zeek `missed_bytes` 不因表示转换而消失；三条下载的新记录仍保留这些缺失事实。是否将该表示纳入所有来源共同输入合同、如何保留抽取器/聚合差异、哪些动作关联可用作监督，由主方案据此确定。本轮没有训练、模型前向或阈值选择，也没有修改事件标签或已有划分。
