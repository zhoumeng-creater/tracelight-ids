# AIT cloud 校验和语义结论（2026-11-19）

**这 38 个已审计 packet 连接候选应以固定 Zeek 8.0.11 的 `-C` 输出作为规范离线输入。strict 保留作采集质量诊断，两臂不能拼成同一模型输入。** 依据是采集主机的发送校验和卸载状态、逐包校验、完整握手与 ACK，以及两臂包数/字节数的精确守恒；没有使用匹配数量、模型得分或训练结果择臂。既有映射文件保持冻结，规范口径单独登记在 [checksum_semantics.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/checksum_semantics.json>) 的 `semantic_decision`。

本轮只读已取 cloud PCAP，逐包检查原来 38 个候选的 **65,758 个包**。按双向端点及已冻结首末包时间归属，没有重复归属；每个候选的双向包数均与冻结目录完全一致。IPv4 头校验覆盖完整头部；TCP 校验使用 IPv4 伪首部及 IP 总长度指定的完整 TCP 段。分片或缺失字节按“无法校验”处理，不把它们算成错误。[TCP 校验定义](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.1)

| 方向 | 包数 | IPv4 可校验/正确/错误/无法校验 | TCP 可校验 | TCP 正确 | TCP 错误 | TCP 无法校验 |
|---|---:|---|---:|---:|---:|---:|
| 代理 → cloud | 6,601 | 6,601 / 6,601 / 0 / 0 | 6,601 | 6,601 | 0 | 0 |
| cloud → 代理 | 59,157 | 59,157 / 59,157 / 0 / 0 | 59,157 | 36 | 59,121 | 0 |

这批包没有零 IPv4 总长度，也没有 IP 声明长度超过捕获字节或 PCAP 捕获长度小于原长的情况。**59,121 个错误的出站 TCP 校验和，全部等于对应伪首部的折叠和。** 这正是 Linux 发送校验和卸载在待补算字段中保存的部分和：Linux 4.15 的 `CHECKSUM_PARTIAL` 分支会预置这一值，后续由发送路径完成计算。[Linux 4.15 实现](https://github.com/torvalds/linux/blob/v4.15/net/ipv4/tcp_ipv4.c#L526)、[同版本说明](https://raw.githubusercontent.com/torvalds/linux/v4.15/Documentation/networking/checksum-offloads.txt)

**作者实际主机状态直接支持该机制。** 本轮从同一官方 AIT 场景 ZIP 取回了 5 个配置/facts 小原件，压缩合计 122,401 bytes，没有新增 PCAP。`cloud_share/facts.json` 记录：

- 默认 IPv4 地址 `172.19.130.106`，接口 `ens3`，内核 `4.15.0-139-generic`。
- `ansible_ens3.features.tx_checksumming = on`，`rx_checksumming = on [fixed]`。
- TSO、GSO、GRO 均为 off。校验和卸载与分段/聚合卸载不是同一个开关。
- 快照时间 `2022-01-25T06:55:43.262732Z`，在该 PCAP 最后一个包之前约 73 秒。
- `suricata_interface = ens3`、`suricata_listenmode = af-packet`。

原件 SHA-256 为 `48ec85facf56fe029a7f68c8901415a0e42011a6dd77d7249a8786827ad6adeb`。JSON 中已摘录上述实际字段，未把整份 facts 的其他信息复制进结果。[作者原件](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/source/capture_config/gather/cloud_share/facts.json>)、[范围下载与校验收据](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/source/capture_config/receipt.json>)

实际 `suricata.yaml` 第 488—489 行指定 `af-packet/ens3`，第 310—329 行开启 PCAP 日志、`use-stream-depth: no`、`honor-pass-rules: no`。第 543—553 行解释接口 checksum 默认使用 kernel 指示；`#checksum-checks: kernel` 本身是注释。第 1491—1497 行也有卸载/校验和设置的注释，不能把注释当成已关闭 TX 校验和卸载的实测结果；接口 facts 明确仍为 on。[实际采集配置](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/source/capture_config/gather/cloud_share/configs/suricata/suricata.yaml:488>)

**strict 丢失回包的机制已经闭合。** 38 个连接都存在序列号和确认号匹配的 SYN→SYN/ACK→ACK，所有 SYN/ACK 在本机 PCAP 中的 TCP 校验和都是错误的；对端随后仍确认握手，并 ACK 后续服务器数据。再结合已核实的服务器 HTTP 200、状态机下载成功以及 `-C` 下 TLS 1.3 会话的 established 状态，不能把 strict 中的“没有回包”解释成真实连接未建立。

逐包统计与已有 Zeek 输出的守恒结果如下，比较覆盖每个 packet 候选的完整生命周期，而非只截取请求秒：

| 方向 | 原包总数 / IP 总字节 | 校验正确包数 / IP 字节 | strict 实际计数 | `-C` 实际计数 |
|---|---|---|---|---|
| 代理 → cloud | 6,601 / 421,306 | 6,601 / 421,306 | 6,601 / 421,306 | 6,601 / 421,306 |
| cloud → 代理 | 59,157 / 85,066,292 | 36 / 3,018 | 36 / 3,018 | 59,157 / 85,066,292 |

strict 恰好只留下校验正确的包，`-C` 恰好保留全部捕获包；包数和 IP 字节数均完全吻合。strict 的 80 条记录中有 **38 条 S0**，对应 packet 候选的真实握手均已完成。被丢弃的 SYN/ACK 与回包令分析器缺少真实双向状态，后续同端点活动再生成额外记录，因此出现先前的 `2:1` 与 `3:2` 拆分差异。Zeek **8.0.11** 官方说明明确该版本默认丢弃校验和错误包，`-C` 忽略校验和，并说明本机卸载抓包需要这样处理；这里引用固定版本，不使用行为已变化的 master 文档。[Zeek 8.0.11 官方说明](https://docs.zeek.org/en/v8.0.11/quickstart.html#live-traffic)

可复查的第一个 PDF 下载候选 `#17101` 中，包号 826593 的 SYN/ACK 存储 TCP 校验和为 23,812，恰等于伪首部部分和；按完整段重算应为 12,396。下一个包 826594 的客户端 ACK 校验正确，确认号为服务器初始序列号加一；后续客户端 ACK 覆盖超过 6.13 MB 的服务器序列空间。逐连接握手、校验值及 ACK 证据均保存在 JSON，不靠大流量特征给它指定正常标签。

规范口径的应用对象是这次完整审计的 **38 个 cloud 端候选**。整份 PCAP 其他连接先前出现的长度异常，以及 intranet 另一观察点的采集语义，不能由此自动通过。这里不把 strict 与 `-C` 两臂合并，也不把四个多义事件选成单一文件流；前端/后端关联和标签范围均沿用冻结表。后续统一模型表应明确引用上述规范臂及原始 flow ID，不能把两臂当成重复样本或可互换特征。本轮没有训练或前向。

复算脚本 [diagnose_checksum_semantics.py](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/ait/diagnose_checksum_semantics.py>) 还核对了诊断前后六份冻结产物的 SHA-256，均未变化：

```powershell
python 'E:\OneDrive - Coventry University\中传\毕业论文\工作\统一事件与方法落实_20261119\ait\diagnose_checksum_semantics.py'
```
