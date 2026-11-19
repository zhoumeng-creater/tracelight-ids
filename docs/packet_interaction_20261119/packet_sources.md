# 现有包头来源与连接映射

现有五份开发来源已抽取 4,961,557 条 IPv4 TCP 包头。时间、方向、seq、ACK、flags、IP 长度和 TCP 头长均可用；包载荷内容未读取或保存。本轮没有模型拟合、打分、阈值搜索或新下载。

AIT 只打开 santos 和 fox 的 intranet_server、cloud_share 原包及其既有连接表；没有打开 russell、harrison、Meterpreter 或其他保留源内容。Unraveled 提取目标主机 10.1.3.8 的全部 IPv4 TCP 包，包含所有端口和未标注连接。抽取不使用攻击标签或动作时间挑包。

## 可直接复用的结果

- 包头目录：`E:\OneDrive - Coventry University\中传\毕业论文\工作\连接内部交互研究_20261119\headers`。每份 `*.jsonl.gz` 是包头；同名 `.connections.jsonl.gz` 给出 0 起始 `connection_index` 到既有 flow_id、stable_flow_id、原始端点、raw8 与时钟的映射。
- 抽取脚本：`E:\OneDrive - Coventry University\中传\毕业论文\代码\tracelight-ids\.worktrees\packet-interaction-20261119\scripts\packet_evidence\extract_headers.py`。
- 关联核验：`E:\OneDrive - Coventry University\中传\毕业论文\代码\tracelight-ids\.worktrees\packet-interaction-20261119\scripts\packet_evidence\source_association_check.py`；结果为 `E:\OneDrive - Coventry University\中传\毕业论文\工作\连接内部交互研究_20261119\source_association_check.json`。
- `packet_sources.json` 保存原包路径、字节数、SHA256 依据、包头产物 SHA256 与逐源计数。原包字节数本轮复核；原包 SHA256 采用既有下载/抽取收据，本轮没有再次通读原包计算 SHA256。

## 来源与采集表示

| 来源 | 原始完整记录数 | 抽取 TCP 头 | 声明长度超过捕获 | IP 长度 >1500 | IP 长度 >9000 |
|---|---:|---:|---:|---:|---:|
| ait_santos_intranet_server | 599,875 | 550,828 | 42,871 | 42,861 | 5,287 |
| ait_santos_cloud_share | 1,066,804 | 1,006,750 | 22,522 | 22,515 | 39 |
| ait_fox_intranet_server | 1,716,027 | 1,633,622 | 56,545 | 56,529 | 6,877 |
| ait_fox_cloud_share | 1,356,225 | 1,241,603 | 34,560 | 34,547 | 28 |
| unraveled_target | 2,592,965 | 528,754 | 0 | 75,146 | 24,248 |

AIT 中，原 PCAP 的 caplen 与 orig_len 相同，但部分 IPv4 头声明的长度大于实际捕获字节。现有 declared-IP 派生仅修正记录 orig_len；本轮直接从未改动的原包读取头部。四源合计 156,498 个 TCP 记录存在这种缺载荷现象。头部长度完整、IPv4 头校验均有效，仍不能把声明字节当成独立验证过的物理线速字节。

Unraveled 原对象长 3,518,820,352 B；最后 74 B 属于未完成记录。使用既有 complete_records 派生（3,518,820,278 B），2,592,965 个完整记录的头和捕获字节保持原样。目标 TCP 记录没有声明长度超过捕获的情况，但有 75,146 个 IP 长度超过 1500、24,248 个超过 9000。这些长度使物理包数、分段大小、ACK 频率受采集聚合或大帧影响；仅凭包头不能区分具体 offload 设置与链路 MTU。

AIT 四源相邻记录没有时间倒退；Unraveled 全源有 1,467 次倒退。文件保留捕获记录顺序和精确微秒时间，没有重排或把负间隔改成零。seq/ACK 按原始 32 bit 整数保存，重传、乱序和回绕需要在后续表示中处理。

## 连接候选与查询时刻

Unraveled `packet_membership.jsonl.gz` 每行为 `[record_index, byte_offset, timestamp_us, caplen, declared_ip_bytes, author_indices, zeek_indices]`。zeek_indices 指向 `base_connections.jsonl.gz` 的 0 起始行序。本轮逐行确认 base 与已标注 connections 的 flow_id 顺序完全一致；每个目标 TCP 包的 record_index、偏移、时间、caplen、IP 长度均与既有账本吻合。

AIT 对每个包按双向五元组和连接生命周期查找全部候选。生命周期截止为 `max(zeek_reported_end, replay_export_time)`，匹配允许 ±10 μs 舍入误差；另外保存严格 observed interval 的候选。没有强行把多个候选中的一个当成真值。

| 来源 | lifecycle-only 包 | 其中正声明载荷 | 多候选包 | 涉及歧义连接 | 包数与 Zeek 不同的连接 |
|---|---:|---:|---:|---:|---:|
| ait_santos_intranet_server | 1,658 | 0 | 8 | 13 | 8 |
| ait_santos_cloud_share | 10,006 | 0 | 68 | 132 | 68 |
| ait_fox_intranet_server | 2,240 | 1 | 1 | 2 | 1 |
| ait_fox_cloud_share | 25,844 | 0 | 55 | 107 | 55 |
| unraveled_target | 8,896 | 6 | 52 | 66 | 52 |

所有候选关联都满足包时间不早于连接开始、不晚于 replay_export_time；实际没有用到导出后的 10 μs 容差。lifecycle-only 的 48,644 个记录中，48,637 个声明载荷为零，主要是 ACK/RST，另有少量 FIN。仅截取 observed interval 会漏掉连接收尾记录。这里支持的是完整连接导出时刻的比较，不能据此宣称首 N 包时刻已获得这些记录。

五源逐方向 Zeek 包数总和都与实际提取包数相等。对于没有歧义包的连接，orig/resp 包数及 IP 字节数全部与 Zeek 一致。184 个多候选包各被保留给两个连接：全部候选的计数合计比 Zeek 多 184，只有 singleton 的计数比 Zeek 少 184；所有逐连接不一致都出现在歧义连接。后续分析拒绝任何含多候选包的连接，并记录其缺失数量，不以计数碰巧相等解除歧义。

## 能比较什么

| 来源 | 同主机 TCP443 控制/C2 | 同主机 TCP443 确认正常 | 控制导出时段内确认正常 |
|---|---:|---:|---:|
| AIT santos intranet | 30 | 125 | 0 |
| AIT fox intranet | 15 | 73 | 0 |
| Unraveled 10.1.3.8 | 473 | 5081 | 769 |

这些连接共享同一采集主机、TCP443 和同一套头部提取规则，可以比较连接内部双向字节与确认进展。AIT 攻击和正常客户端不同，正常连接未出现在那段短攻击时窗；Unraveled 提供同攻击导出时段的 769 个正常 TCP443，但 C2 对端没有对应正常业务。三组都不能用端点、绝对时间或周期标签作特征来声称机制区分。

AIT 正常 ownCloud 下载已在 cloud_share 全 TCP 头部中保留，恶意下载候选在 intranet_server 全 TCP 头部中保留。两者主机与服务不同；one-of 动作映射仍是候选集合，不能把动作候选全部升为确定正常或攻击连接。Unraveled 的 473 个 C2 是同一攻击活动中的连接，且 1 个连接受原包末尾不完整记录影响；不能当作 473 次独立指令。

## 机制分析的约束

- 字节进展使用声明 TCP 序列区间的首次覆盖并显式处理重传/重叠，避免把重复抓取或重传当成新应用字节；声明覆盖仍不等于载荷完整。
- ACK 是累计传输确认；收到 ACK 不能推出应用已处理请求，也不能单凭方向交替推出命令—下载因果关系。未见握手、序列缺口、重置、时间倒退、映射歧义和末尾截断分别保留标记。
- 多个完整大记录的切分方式可以改变包数和突发数。若候选机制只能利用这些切分差异，当前来源不能支撑应用机制创新；若依赖独立于切分的唯一字节和累计确认关系，应让同信息、同预算近邻也看到同一表示。
