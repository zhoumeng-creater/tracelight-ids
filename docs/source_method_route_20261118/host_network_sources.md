# 新增主机—网络来源核查：OpTC 与 Unraveled

核查日期：2026-11-18。已按授权完整取得 OpTC 9 月 24 日 ecarbro、HTTP、files、conn 四件压缩原件，合计 **108,166,157 bytes（约 108.2 MB）**，另保存 453,244-byte 官方红队 PDF。已完成全量扫描和事件关联，未训练、未评分。以下第 1–4 节保留初轮选型依据；实际装配结果见第 5 节。

## 决定

**OpTC 的指定切片已经装配完成：24 条已知 Empire 控制会话、2 次同文件下载；其中 1 次下载与红队动作记录吻合同一秒。** 两次下载及 24 次控制连接属于同一攻击情景，不能算 26 条独立攻击链。Unraveled 保留为第二候选，本报告不安排进一步扩取。

OpTC 已能沿“红队动作—主机进程—bro_uid—conn/http/files”追查具体正例。正常侧仍不就绪：根代理全量检查 9 月 19 日同小时 HTTP/files 后发现的 6 条 PE 下载，来自同一红队服务器 news.com，文件为 met.exe、mal.exe、explorer.exe，均须保持 **unknown**，禁止把红队之前整段数据或攻击匹配后的剩余流量统标正常。正常更新、可比业务文件下载和合法远程运维仍未得到合格分母。Unraveled 的同服务样例及合法管理代码保留作来源证据，逐流标注规则与精确攻击执行记录尚未找到。

这次来源建设的作用是给算法创新提供可以证伪的独立任务。来源核查、阈值修补或展示程序均不能代替尚未完成的算法创新。

## 1. OpTC：主选，动作关联证据最完整

### 已确认的事实

[作者官方仓库](https://github.com/FiveDirections/OpTC-data)说明：测试床有 1,000 台主机，采集其中 500 台；约两周数据，约 1 TB **压缩 JSON 兼容格式**；红队进入前以及红队活动期间均有持续正常业务。官方提供 `bro` 网络日志、`ecar` 主机事件及 `ecar-bro` 关联记录。`short` 目录存在缺失，不选它作为主数据入口。数据标为 public domain、Distribution A，允许公开发布。

本轮实读 [2019-09-24、AIA-501-525 的 ecarbro.json.gz](https://drive.google.com/file/d/1-45syji9pX-_QMxR1E_xIfjxcZ2eAPLB/view)前 64 KiB 压缩字节。记录确有 `timestamp`、`hostname`、`actorID`、`pid`，以及 `properties.bro_uid`、进程 `image_path`、方向和五元组。例如首条为 SysClient0525 的 Firefox 进程，时间 `2019-09-24T10:10:37.454-04:00`，`bro_uid=CXcnGn2tUpxqJzwvL`。这证明官方关联机制实际落在数据中；该示例自身不是攻击。

本轮又分别实读 2019-09-24 17:00–18:00 的 `conn`、`http`、`files` 各前 64 KiB 压缩字节：

- `conn` 包含原始 Zeek/Bro 语义的 duration、双向 payload bytes、IP bytes、包数、conn_state、history、service 和 uid。可接现有流特征路线；仍须统一旧版 service 名称和缺失值处理。
- `http` 有 uid、method、host、uri、响应长度、状态码、resp_fuids；`files` 有 fuid、conn_uids、mime_type、seen_bytes、missing_bytes。两者确有共同文件与会话 ID。
- 实例 `CAbcEB153YU8V7Jnc5` 的 HTTP 200 JavaScript 响应，与 `files` 中 `FtYQt74pHmJXW1ULHb` 对应。这是完整关联结构的实例，尚未完成红队下载本身的逐条关联。

参考：[官方 eCAR 格式](https://github.com/FiveDirections/OpTC-data/blob/master/ecar.md)、[已知问题](https://github.com/FiveDirections/OpTC-data/blob/master/errata.md)。模块加载哈希缺失、重复 process 对象和部分 FLOW 信息不完整，决定了标签不能只靠单个 PID 或哈希。

### 控制和下载的动作原件

[OpTCRedTeamGroundTruth.pdf](https://github.com/FiveDirections/OpTC-data/blob/master/OpTCRedTeamGroundTruth.pdf)为官方 7 页红队记录，本轮读取全文，原件 453,244 bytes，Git blob `fd6e91bf6b888c7239661b454bd8cdc9feb8c6a2`。

| 情景 | 可直接定位的动作 | 可承担的标签 |
| --- | --- | --- |
| 2019-09-23，Plain PowerShell Empire | C2 为 news.com:80；给出 profile URI、5 秒间隔、受害主机、PID 与逐项时间；11:23:29 从 news.com:8000 下载 runme.bat | 指定 Empire 进程/服务器的控制通道；带上下文的 stager 下载 |
| 2019-09-24，Custom PowerShell Empire | 10:46:02 改用 sports.com:443；13:26:57 在 Sysclient0501 经 RDP 后用 Chrome 从 news.com:8080 下载 fileTransfer1000.exe（nc.exe）；14:37:02 在 Sysclient0005 从 news.com:4445 下载 movingonup.exe | 控制链中的明确下载动作；可核查浏览器进程及 HTTP 对象 |
| 2019-09-25，Malicious Upgrade | 10:29:42 Sysclient0051、11:23:31 Sysclient0351 下载恶意 update.exe；随后 Meterpreter reverse TCP 回连 8080，记录进程和注入对象 | 恶意升级下载及其后控制通道 |

上述标签来自红队动作记录，**不是“只要端口为某值就算恶意”**。仍需把攻击时间、受害进程和 HTTP 对象接到实际会话，不能把整个受害主机当日流量统标恶意；也不能把每条 C2 连接都表述为已证实执行了一条控制命令。

有三天的情景记录，但主要是 Empire 的两种配置和 Meterpreter 两类实现，且第二天控制活动延续到第三天。不能算三个互相独立的网络，也不能仅按自然日切开同一攻击链。

### 正常业务证据与缺口

另取红队进入前 **2019-09-19 17:00–18:00** 的 HTTP 和 files 各前 128 KiB 压缩字节，解压片段约有 2,885 条 HTTP 行、3,990 条文件行。可确认正常网页 HTML、CSS、JS、图片下载。例如 `Cc3Adz1UMaNVe5FiRk` 的 HTTP 200 HTML 响应 81,422 bytes 与 `files` 记录相符。

这已支持“正常 HTTP 请求/响应与文件对象存在”，**尚未支持“可比正常软件更新、业务文档/安装包下载、合法远程管理的数量足够”**。片段中字符串 `update` 只出现在网页图片/JS 名称，不能拿它冒充软件更新。大流量背景也不能自动代替这些反例。

### 初轮锁定的切片原件

| 原件 | 已核实大小 | 用途 |
| --- | ---: | --- |
| [9 月 24 日 AIA-501-525 ecarbro.json.gz](https://drive.google.com/file/d/1-45syji9pX-_QMxR1E_xIfjxcZ2eAPLB/view) | 19,992,689 bytes | Sysclient0501 进程—会话关联 |
| [9 月 24 日 conn.17:00:00-18:00:00.log.gz](https://drive.google.com/file/d/1EYNYdkVSIDw78XfDcUGD9cRFHzjsSmMH/view) | 58,407,076 bytes | 攻击及同小时正常流分母 |
| [同小时 http](https://drive.google.com/file/d/14zNUSYWap4CiCKdTv4HZ9v56P6bFIPiQ/view) | 17,159,303 bytes | 下载 URI、响应状态及长度 |
| [同小时 files](https://drive.google.com/file/d/1AkEqZRG7Ox-PwsnywhG3vxY_iDnDoz1e/view) | 12,607,089 bytes | 文件对象与会话关联 |
| [9 月 19 日同小时 conn](https://drive.google.com/file/d/1fJvUt8nEhfROmzP3AjsofKo4GTHSvhd-/view) | 60,278,906 bytes | 红队进入前正常分母 |
| [9 月 19 日同小时 http](https://drive.google.com/file/d/1rFfW740CFN4EKbj_szEiPE52gYyMzgqW/view) | 16,636,170 bytes | 正常 HTTP 业务类型 |
| [9 月 19 日同小时 files](https://drive.google.com/file/d/110oZbclIrKkQ3OPWOUSxr7y4XYwNI-Vh/view) | 12,629,303 bytes | 正常文件下载类型 |

七个完整原件合计 197,710,536 bytes，约 189 MiB；本轮只读取其中的小片段。13:26:57 的红队动作应先核对与 eCAR `-04:00` 的时区一致性，再查 UTC 17:26:57 左右，不能直接选文件名 13 点的 Bro 日志。

初轮据此确定 `event_id → 红队原文时间/主机/动作 → actorID/PID/进程 → bro_uid → HTTP/files 对象` 的关联方式。9 月 24 日装配已经完成；正常更新/文件传输分母没有成立，不能宣称已经解决了恶意下载与正常下载的区分。

## 2. Unraveled / DAPT2021：第二候选，同服务反例和业务代码更强

### 实际可取得的工件

[作者仓库](https://gitlab.com/asu22/unraveled)、[官方 S3 说明](https://dapt2021.s3.amazonaws.com/README.txt)、[数据说明](https://gitlab.com/asu22/unraveled/-/blob/master/data/README.md)均可读取。约 6 周、约 1 TB 原始数据、约 22 GB 流数据、约 680 万条流；原始 PCAP、主机日志、Snort 和标注后的 NFStream 流分开发布。公开 S3 支持匿名列表与 HTTP Range，无需下载整个桶。

仓库及数据说明标 GNU 许可，仓库 LICENSE 为 GPLv3 系列，文件末尾写明 GPLv3 or later。它不是 public domain；使用时保留作者引用和随附许可，不把 GPL 许可解释为数据事实质量保证。

本轮对 `processed/network-flows/Week5_Day2_06222021/net1013x_Flow_labeled.csv` 实读前 131,072 bytes，服务器返回 HTTP 206、全长 21,798,174 bytes。样例确有五元组、毫秒时间、双向字节/包数、nDPI 名称及四个标签。两条具体记录：

| 同一主机 10.1.3.8 的会话 | 首次时间，Unix ms | 字节 | 作者标签 |
| --- | ---: | ---: | --- |
| 55002 → 10.8.10.84:443，TCP/TLS | 1624347137277 | 882 | Activity=Encrypted Channel: Symmetric Cryptography；Stage=Establish Foothold；Signature=APT |
| 53792 → 142.250.72.131:443，TCP/TLS.Google | 1624347147729 | 865 | Activity=Normal；Stage=Benign；Signature=None |

因此，样例已经有同主机、同端口的正常/恶意通信，不是简单端口标签。这两条的 `DefenderResponse` **都为 Benign**；该列表示防守方没有认定恶意，不能当真值标签。

NFStream 的字节核算、会话切分和应用识别不能直接等同于当前 Zeek 特征。作者的 [pcap2flows_nfstream.py](https://gitlab.com/asu22/unraveled/-/blob/master/tools/feature_extraction/pcap2flows_nfstream.py)确认用 NFStreamer、statistical_analysis=True、n_dissections=20 转换。主路线如继续使用现有特征，应从选定 PCAP 重提，不靠列名硬映射。

### 控制、下载和正常业务

作者说明及工具表记录了 Covenant、Sliver、Metasploit、恶意网页载荷、加密控制通道与后续数据库外传。Skilled Hackers 情景通过邮件恶意附件入侵，附件被下载、执行尝试被阻断；APT 情景在员工常访问网站部署载荷，随后建立控制通道。它们不是把正常流量人工改成恶意标签的单一端口规则。

正常业务有作者代码依据：

- [tools/README.md](https://gitlab.com/asu22/unraveled/-/blob/master/tools/README.md)明确用 Ansible 管理员工主机，Linux 用 SSH，Windows 用 WinRM 5986；恶意横移也利用为 Ansible 开放的 SSH。
- [copy.yml](https://gitlab.com/asu22/unraveled/-/blob/master/tools/employee_behavior_generation/playbook/copy.yml)复制用户配置到工作节点；[work.yml](https://gitlab.com/asu22/unraveled/-/blob/master/tools/employee_behavior_generation/playbook/work.yml)远程执行 worker。这是合法管理行为的可复查实现。
- [worker_check_mail.py](https://gitlab.com/asu22/unraveled/-/blob/master/tools/employee_behavior_generation/worker/worker_check_mail.py)第 126 行点击 Download；仓库有正常邮件附件样本及邮件生成/接收脚本。这比仅在背景说明里写“正常办公”更强。
- 实读流表已见 Normal 的 HTTP.UbuntuONE 和 TLS.Google，前者不足以单独证明软件更新下载完成。

**仍缺精确攻击动作到会话的标注依据。** 本轮完整枚举 tools 的 164 个目录/文件项，看到清洗、行为生成、特征提取和模型代码，未找到逐流 labeling 脚本或精确红队执行日志。现有 Activity 标签能定位 C2 类流，但仅凭它还不能排除“某 IP 在较宽时间段内所有流统一标注”的情况。正常代码存在也不等于已得到足量同服务正常样本，仍需在选定原件中核实实际执行。

### 独立数量、捕获边界和已识别原件

有 Amateur、Skilled Hackers、APT 三种攻击者轨迹，**不是六周的独立攻击重复**；其中完整持续控制/外传主要来自一条 APT 轨迹。一个测试床中的多主机、多天和多条 C2 流不应重复计为独立攻击链。

作者明确说明：子网到外网流量会同时出现在子网接口和网关接口；同子网内流量不经过网关，未被捕获。不可将 net1013x 与 netgw 当成两份独立样本，也不可用“没捕获到”判定横移没有发生。

初轮核查识别到 2021-06-22、10.1.3.x 子网的以下原件，本轮不再获取：

- [对应已标注流表](https://dapt2021.s3.amazonaws.com/processed/network-flows/Week5_Day2_06222021/net1013x_Flow_labeled.csv)，21,798,174 bytes；只用于标签核查、事件关联和建立索引。
- [对应 PCAP](https://dapt2021.s3.amazonaws.com/raw/network-captures/Week5_Day2_06222021/net1013x.pcap)，官方目录实际大小 2,623,311,872 bytes。本轮未下载。
- `raw/host-logs/10_1_3_8/sys-logs/` 下有 syslog、syslog.1 及轮转 gzip；`raw/host-logs/10_1_3_17/user-ansible/` 下有四类 Windows event CSV，大小分别约 9.27、9.36、26.76、11.00 MB。目录已实查，内容未读。
- 精确攻击执行时间表及逐流标注规则：当前公开仓库未查得，是这条来源用于主确认前必须补齐的原件；不能用作者论文的一张阶段示意图替代。

## 3. 未选 ATLAS 为本轮第三条主路线

[ATLAS 官方 README](https://github.com/purseclab/ATLAS)的任务是从审计日志建立实体图与序列、识别攻击实体；[raw_logs](https://github.com/purseclab/ATLAS/tree/main/raw_logs)实际列出 M1–M6、S1–S4 十个 ZIP。本轮查到的是审计实体管线与压缩包目录，没有核实到可复用的 conn 字节/包数语义或 PCAP。ZIP 尾部有界读取请求未成功，没有据此断言包内不存在网络数据。相较 OpTC 已直接验证的 Bro 和 eCAR 关联，它目前增加输入重建工作，不能作为这轮控制/下载流确认的优先来源。

## 4. 独立确认的准确表述

OpTC 采于 2019 年，Unraveled 采于 2021 年；均早于当前 2026 年训练来源。保留未用事件、固定方法后评估，可以支持**跨来源、跨攻击实现的历史独立事件确认**，不能称为从当前训练时间走向未来的时间前向验证。若要研究单一来源内部的时间前向，需要重新明确该来源自己的历史训练段和后续整条攻击链，并处理已发生的数据检查；不能把开发样例重新命名为盲测。

本轮已打开的结构样例和指定诊断切片用于资格验证与源定位，不是方法评价。OpTC 切片的全量装配见下；Unraveled 保留为第二来源记录，本轮不扩取其余日期或数据集。

## 5. 全量装配完成：控制通道与下载动作

下载事件脚本：[assemble_optc_event_slice.py](<scripts/assemble_optc_event_slice.py>)。输出：[optc_event_slice.json](<evidence/optc/optc_event_slice.json>)。四件压缩原件和红队 PDF 的完整 SHA256、大小、URL、原始记录行号及脚本 SHA256 均写入 JSON。

完整扫描量为 eCAR-Bro **261,326** 条、HTTP **381,579** 条、files **397,496** 条、conn **1,553,112** 条。文件名查询、主机与端口查询的全部候选均保留。

| 下载候选 | 首次 HTTP 请求 UTC | 与红队 13:26:57（−04:00）的差 | 原始网络观测结束 UTC |
| --- | --- | ---: | --- |
| CSHrim2uG4Jj6MUi31 | 17:26:57.933649 | +0.933649 秒，同一报告秒 | 17:26:57.936547 |
| CJ2kfq1Fa0PZCafIMe | 17:42:58.458456 | +961.458456 秒，重复下载候选 | 17:42:58.463468 |

两条均来自 Sysclient0501 的 Chrome PID 2340，GET `/fileTransfer1000.exe`，服务器 `132.197.158.98:8080`，HTTP 200，响应为 **59,392 bytes** 的 `application/x-dosexec`。files 的 seen_bytes=total_bytes、missing_bytes=0、未超时，且 uid、五元组及 fuid 精确相连。原始 conn 完整保留。第一条为红队记录动作的最强对应证据；第二条保留为重复下载，不伪造第二条明确动作记录，也不增加独立攻击链数量。原件未给该可执行文件内容哈希，不能声称已核对可执行文件字节身份。

控制通道脚本：[assemble_optc_control_channel_slice.py](<scripts/assemble_optc_control_channel_slice.py>)。输出：[optc_control_channel_slice.json](<evidence/optc/optc_control_channel_slice.json>)。该查询只读取上述已下载原件，没有增加下载，也没有修改下载事件 JSON。

官方 PDF 第 4 页给出 Sysclient0501、agent 6H8SZPCW、PID 1748 与服务器 `sports.com / 202.6.172.98:443` 的对应；第 5 页说明该 agent 在 13:05:03 执行 ls，13:11:23 启动反向 SSH 后失去联系。PID 648、5076 对应的 agent 已于 11:35:53 被杀，不能统标整个下午为活动。

在现有 conn 文件中，目标主机到该服务器 443 共 **24 条**记录，全部通过 eCAR 的 uid、五元组、PID 1748、PowerShell 进程和官方活动时间核对；没有无 eCAR 对应、五元组冲突或多个 agent 争用同一匹配的记录。网络开始时间范围为 **16:59:55.019358–17:10:45.926148 UTC**；小时日志包含少量 17 点前开始的连接，因此按原始 ts 报告实际边界。它们可标为**已知 Empire 控制通道**，不能声称每条连接都执行了某个指定命令。

eCAR 整件文件含该主机到该服务器的 342 条记录：PID 648 为 120 条、5076 为 23 条、1748 为 199 条；其中 318 条未在这个小时的 conn 文件中找到对应，保留原始候选与计数，不能据此标正常。24 条已确认通道无明文 HTTP 匹配；files 的 2 条匹配为 SSL 的 `application/pkix-cert` 证书对象，不计恶意载荷下载。

跨日志时钟差原样保留：控制连接的 Bro 开始时间比 eCAR 时间晚约 **44.055–44.065 秒**；两次下载的 HTTP 时间分别晚 **44.098、44.136 秒**。关联依靠官方 bro_uid 及五元组，不擅自做全局时钟校正。每条连接的 `available_at_lower_bound` 至少为 `conn.ts + duration`，HTTP 和 files 的完结证据另列；实际日志输出/超时/入库时刻未知，保留 null，不把特征可用时刻倒填到红队动作开始。

根代理的正常业务结果见 [benign_file_business.json](<evidence/optc/benign_file_business.json>)。9 月 19 日的 6 条 PE 下载均来自 `132.197.158.98:8000 / news.com`，响应 73,802 bytes；当前官方 7 页红队 PDF 未提供 9 月 19 日预演说明。它们继续标 unknown。此次交付建立了可追溯控制/下载正例，尚未建立可比正常文件与运维分母；这里没有检测性能、误报率或算法增益结论。

复现已落盘证据：先运行 `python scripts/assemble_optc_event_slice.py --with-conn --offline`，再运行 `python scripts/assemble_optc_control_channel_slice.py`。第二个脚本核对全部原件 SHA256，并记录下载事件 JSON 查询前后的相同 SHA256。
