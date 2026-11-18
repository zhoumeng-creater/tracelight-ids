# OpTC 正常业务已装配：HTTP 文件对象与 TLS 浏览连接

2026-11-19。本轮已得到可追溯的正常业务反例：**5,026 个正常网页文件对象，其中 42 条 HTTP 连接的全部事务均满足完整业务证据规则；另有 41,588 条正常背景中的浏览器 TLS/443 连接。** 六条 news.com PE 下载保持 unknown。没有训练、模型评分或阈值选择。

这批数据可用于已开开发外源持出的业务对照。两个指定业务子集的正常分母已经完整枚举；全网正常分母、正常程序更新和合法远程运维分母没有成立。

## 1. 9 月 19 日的正常背景声明来自哪里

官方目录实际为 [ecar-bro/benign](https://drive.google.com/drive/folders/1XOGWT8R8ovw6J2DgNsui09ZwvrfOXIVa) → [19Sep19](https://drive.google.com/drive/folders/1ALKFkSXaWLnuC4oa51DNyOhN1Jz2XnSq) → [AIA-501-525](https://drive.google.com/drive/folders/1l0T20hYOwfBnUxPBBT89XPn9QrWHsVaV)。这里的 benign 来自作者目录，不是根据日期早于 9 月 23 日推断。原始分析论文第 2 页第 3.1 节也将 9 月 19–23 日描述为 normal collection，并说明 eCAR-Bro 的 bro-id 关联作用。[论文](https://arxiv.org/abs/2103.03080)

作者的背景声明作为基础证据，再与具体进程、会话和业务对象核对。它不能覆盖矛盾证据：9 月 19 日从 `132.197.158.98:8000 / news.com` 下载 `met.exe`、`mal.exe`、`explorer.exe` 的六条 PE 仍为 unknown，未纳入正常表。也没有把其余 IP 或该日期的所有流量统标 normal。

本轮复用已取的 9 月 19 日 17–18 UTC HTTP/files，定向补取该时段 conn、ssl、x509，以及上述 25 主机分片的 eCAR-Bro，新增原始日志 **110,332,302 bytes**。所有原件的 URL、大小、SHA256 和目录依据在 [source_receipt.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/source_receipt.json>)。

## 2. 实际正常事件与证据强度

| 已交付业务 | 数量 | 标签证据 | 可以支持什么 |
| --- | ---: | --- | --- |
| 正常网页文件对象 B1 | 5,026：4,779 图片、247 CSS | 作者背景；浏览器进程；精确 uid/五元组；HTTP GET 200；同站父 HTML 页面在同一 actor 前 120 秒内实际出现；HTTP/files 长度一致、无缺失、无超时 | 正常 HTTP 文件传输对象，不能替代正常 EXE/安装包下载 |
| 完整正常 HTTP 连接 B1 | 42 | 上述规则覆盖该连接内**全部** HTTP 事务；conn 为完整 TCP/SF，边界在该小时内 | 与恶意 HTTP 下载作连接级对照，不把同连接中一个正常对象扩展为整条连接正常 |
| 正常 TLS 浏览连接 B2 | 41,588 | 作者背景；具体 Firefox/Chrome 进程；eCAR uid 与五元组精确对应；TLS/443 建立成功；具体站点 SNI | 与 Empire 的 TLS/443 控制通道作同服务对照；没有 HTTPS 载荷、URL或每次业务动作真值 |

B1、B2 是本次来源证据分级。B2 的依据是作者正常背景与明确浏览器通信行为，不是对加密载荷内容的核验。表中 41,630 条正常连接覆盖 **17 台主机、327 个“主机—浏览器 actor—站点”组**；这些组与连接数量均不等于独立真实部署次数。

一个与恶意下载体量接近的实际反例：

- 正常事件 `optc_20190919_normalflow_CzBYVl2SykMQhic3ak`，Sysclient0520 的 Firefox PID 3368，17:50:49.974227–17:50:56.226402 UTC，HTTP/80。
- 该连接只有一个合格图片对象 `FwlBSj3bZQA9tVW5dl`：文件体 59,678 bytes，conn 响应 59,965 bytes，52 个响应包；有父网页、实际响应和 files 完整性证据。
- 对应恶意文件体为 59,392 bytes、conn 响应 59,598 bytes、52 个响应包。图片不是程序安装包，但确实构成同体量 HTTP 文件传输反例。正常对象全集中有 494 个落在恶意文件体 0.5–2 倍范围，42 条完整正常 HTTP 连接中有 8 条处于对应连接响应字节范围；该范围只用于描述可比性，没有用于挑选标签或模型调参。

实际 TLS 反例 `optc_20190919_normalflow_CK5ONi312mzKMCfbL7`：Sysclient0520 的 Firefox PID 4344，SNI=`www.noaa.gov`，TCP/443、TLSv1.2、建立成功，conn 响应 53,951 bytes；eCAR 与 conn 五元组一致。SNI 表示测试床中的站点身份，不等于对今日公共网站或互联网证书的验证。TLS 会话恢复记录可能不带证书，原字段保留。

本片段没有可确认的正常程序下载、系统更新、业务文档传输或远程运维事件。eCAR 全件 177,023 条记录中，176,812 条带浏览器路径；其余 211 条缺少进程路径（80 端口 6 条、443 端口 205 条），不能把它们补成运维行为。

## 3. 正常分母成立到什么范围

**已成立的是两个具体业务子集的连接分母：42 条 HTTP 网页文件传输、41,588 条 TLS 浏览。** 所有满足固定规则的记录均输出，未按分数、检测结果或是否容易区分筛选。HTTP 的 5,026 个文件对象分属多条连接，不能直接当成 5,026 条独立流；混有未满足规则事务的连接未被赋予整流正常标签。

**全网正常分母仍未成立。** 这一个小时的其他主机、未证明业务的流量、失败连接、跨小时连接，以及缺少唯一关联的记录，没有补正常标签。因此可以计算指定业务子集的误报比例，不能据此宣称企业全部正常流量上的 1% FPR。完整筛除计数写入 [normal_business_manifest.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/normal_business_manifest.json>)，没有丢掉六条 PE 的 unknown 状态。

## 4. 模型输入与时间

连接表保留同一 Zeek 语义的 8 个原始数值：`duration`、`orig_bytes`、`resp_bytes`、`orig_pkts`、`resp_pkts`、`orig_ip_bytes`、`resp_ip_bytes`、`missed_bytes`。不在此处宣称它们就是最终模型契约；新方法须以统一输入契约为准。

主机/IP、PID、进程路径、actor、uid/fuid、URL、referrer、SNI、MIME、原件行号和标签只用于来源证据，不作为检测器输入。正常 HTTP 在端口 80，恶意下载在 8080；端口保留审查，但不能让这一差异成为主要检测依据。TLS 浏览与 Empire 控制均为同一 TCP/443 服务。

每条连接的可用时间下界为观测流结束时间；文件对象另保留 files 观测结束时间，实际日志输出/超时/入库时间为 null。9 月 19 日 eCAR 与 Bro 约有 **30.069–30.479 秒**时钟差，记录保留两端原始时间，关联依靠官方 uid 和五元组，没有强行套用 9 月 24 日约 44 秒的时差。

## 5. 文件与复现

- [normal_flow_events.jsonl.gz](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/normal_flow_events.jsonl.gz>)：41,630 条实际正常连接，逐条带事件 ID、证据等级、业务组、源行号、原始数值及时间。
- [normal_http_file_objects.jsonl.gz](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/normal_http_file_objects.jsonl.gz>)：5,026 个实际文件对象，逐条带 URL、父页面、fuid、完整性依据和连接引用。
- [normal_event_examples.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/normal_event_examples.json>)：便于人工复核的原始 HTTP/files/conn/eCAR 示例。
- [normal_business_manifest.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/normal_business_manifest.json>)：规则、全部扫描与筛除计数、证据分级、分母和数据文件 SHA256。
- [fetch_normal_sources.py](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/fetch_normal_sources.py>) 与 [assemble_normal_business.py](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/assemble_normal_business.py>)：在本目录运行 `python fetch_normal_sources.py --offline`，再运行 `python assemble_normal_business.py`，即可从现有原件重建；第二条命令不会下载数据。

本批事件的角色为已开开发来源资格材料，后续训练、阈值选择和外源持出分配服从统一登记；不得把本批已经查看的对象改称未开盲测。

## 6. 统一观测表：补齐五元组，不扩大正常标签

[normal_flow_observations.jsonl.gz](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/normal_flow_observations.jsonl.gz>)已从同一原始 conn 按 UID 装配 **42,147 条唯一连接**：41,630 条既有正常流，与 5,026 个 HTTP 对象所属的 559 条连接取并集；交集为 42 条，额外对象连接为 517 条。

41,630 条的 `normal_flow_label_eligible=true`；额外 517 条为 false。文件对象的正常证据不会使其整个父连接自动变成 normal。每个对象 UID 都能引用一条观测，未匹配数为 0，原有两张事件表 SHA256 未变。

观测表保留 source/destination、原始协议及数值 ip_protocol、双向端口、完整时间与原始源行；统一 raw8 为 `ip_protocol, orig_port, resp_port, duration_seconds, orig_pkts, resp_pkts, orig_ip_bytes, resp_ip_bytes`，八列缺失数均为 0。另保留 payload bytes 和 missed_bytes。这里保存原始字段，不改变第 4 节的模型输入限制；端口是否进入最终方法仍由统一契约决定。

`flow_id=optc:20190919:<UID>`；`zeek_reported_end` 与 `available_at_lower_bound` 为 conn.ts+duration，`actual_available_at=null`。复现：运行 [assemble_normal_observations.py](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/assemble_normal_observations.py>)。完整计数、SHA256 和原表未变证明在 [normal_flow_observations_receipt.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/normal_flow_observations_receipt.json>)；本步没有新增下载或修改正常筛选。

## 7. 历史上下文的完整声明背景输入

[context_inputs.jsonl.gz](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/context_inputs.jsonl.gz>)已从既有两份完整 conn 轮转导出 **45,309 个主机视角、45,309 条物理连接**：

- `optc:20190919:declared_normal_business_subset`：正常事件涉及的 17 台主机全部已观测地址，保留该轮转中涉及这些地址的全部 44,944 条连接。
- `optc:empire_custom:20190924_continues_20190925`：主机 142.20.57.246，保留该轮转中全部 365 条连接，含 362 TCP、2 UDP、1 ICMP；1 条为响应端视角。

本次导出不按标签、协议或完整性过滤。每个目标主机有一个视角，`physical_flow_id=optc:DATE:UID`，`flow_id=physical_flow_id:host`；若两个端点都属于目标主机，代码输出两个视角、共享物理流 ID。此次没有双目标记录。raw8 始终保留原始连接方向，不因响应端视角而交换源/目的字段。

原始导出中两条 Empire 连接的 duration 缺失，raw8 的该维及 `flow_end_at` 保留 null。不能仅因 duration 缺失就用 start 替代；第 8 节为这两条具备单包证据的记录另作有依据的离线时钟推导。全部 `available_at=null`，观测结束时间排序不是在线日志实际到达顺序。

这里的“完整”仅指**指定主机在指定 conn 轮转中的全部记录**。组开始时历史为空，更早轮转中的前序连接未补齐；该轮转可能包含名义小时之前开始的连接，后续轮转也可能迟报更早启动的连接。因此不声称恢复了完整小时实时流或全场历史。“normal”组名表示主机选取依据，不给这些背景记录整体赋正常标签。未读取 Meterpreter 预留分支，也未运行 prefix 模块或模型。

复现脚本：[export_context_inputs.py](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/export_context_inputs.py>)。源 SHA、17 主机地址映射、全量扫描计数、缺失值与时间边界见 [context_inputs_receipt.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/context_inputs_receipt.json>)。

## 8. 单观测包的离线末时钟推导

两条 UDP DNS 记录 `CPdC8j3SCUkw9Gccff`、`C4hCrq1WFLFdAFIKf9` 均为 orig_pkts=1、resp_pkts=0。既然只观测到一个包，而原始 ts 表示首包时间，已观测末包时间必然等于该 ts。官方 [Bro v2.6.1 conn 定义](https://github.com/zeek/zeek/blob/v2.6.1/scripts/base/protocols/conn/main.bro#L18-L19)说明 ts 的首包语义，[写入逻辑](https://github.com/zeek/zeek/blob/v2.6.1/scripts/base/protocols/conn/main.bro#L229-L245)仅在 duration>0 时写 duration，也允许无正 duration 时记录包数。这里以官方版本作字段语义参考，未核实 OpTC 的精确运行构建版本。

[context_replay_inputs.jsonl.gz](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/context_replay_inputs.jsonl.gz>)保留全部 45,309 行，只给上述两行补入离线观测末时刻及 `observed_end_basis`，其余 45,307 行逐字典不变。两行原始 raw8.duration 仍为 null，全部实际 `available_at` 仍为 null，原始 context_inputs 文件未改。

这不推导 UDP 超时、连接状态移除、日志写出或入库时间；仅供显式的离线观测末时钟重放。运行 [resolve_single_packet_clock.py](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/resolve_single_packet_clock.py>)可复现，断言与完整依据在 [context_replay_inputs_receipt.json](<E:/OneDrive - Coventry University/中传/毕业论文/工作/统一事件与方法落实_20261119/optc/context_replay_inputs_receipt.json>)。
