# AIT 与 CICAPT-IIoT 原始证据核查（2026-11-18）

**首选继续使用 AIT-LDS 的 russellmitchell，把本轮新找到的正常文件下载接入已有连接证据。** 现有材料确有正常下载，过去只检查两名员工的日志，漏掉了实际使用 ownCloud 的另外两名员工。此次已经取回原日志，把 31 次正常下载动作、服务端成功响应、代理所见客户端唯一配对。它使 AIT 从“只有正常浏览，缺正常下载”推进到“有正常下载动作和客户端证据，尚待同一抓包视角和服务可比性核验”。它还没有证明正常软件更新或远程管理已可用，也没有增加新的攻击实现家族。

**AIT-ADS 不能另算一个独立数据来源。** 2026-11-18 查询作者 Zenodo 的 `versions/latest`，仍返回 AIT-ADS version 1、record 8263181。它是在 AIT-LDSv2 原始记录上运行 Suricata、Wazuh、AMiner 得到的告警；`labels.csv` 标记攻击阶段起止。作者提到软件更新等正常行为引发误报，但这一说明不能替代某次更新的原始命令、下载对象、执行结果及连接映射。[作者记录](https://zenodo.org/records/8263181)、[告警生成代码](https://github.com/ait-aecid/alert-data-set)

**现有场景与其他版本。** 当前已打开并详细检查的是 `russellmitchell`，有效期为 UTC `2022-01-21 00:00:00 ≤ t < 2022-01-25 00:00:00`。2026-04-09 发布的 AIT-LDS `v2_1` 仍包含 2022 年 1—2 月八次采集，并提供去除 PCAP 的 ZIP；它不是 2026 年重新采集的新环境。其他七个名称为 fox、harrison、santos、shaw、wardbeck、wheeler、wilson；它们有分开的执行时间、环境和参数，但共用攻击生成体系。可以检验跨运行与环境变化，不能称为七种独立控制实现。特别是 wheeler 明确没有密码破解分支，不能预设其有相同工具下载；shaw 的 DNS 日志看不到外传。[作者数据记录](https://zenodo.org/records/19483937)

如需要第二次 AIT 运行，优先取 **harrison 的小原件**：它是另一日期的高扫描量运行，可与当前低扫描量场景形成预先指定的跨运行验证。此选择依据采集差异，不依据模型成绩；其控制及下载成功仍须由该运行的原件确认。它不能代替其他作者、其他控制实现的外部验证。已经打开的 russellmitchell 证据属于开发审查材料，不能重新命名为未见测试集。

**本轮取得的正常下载证据。** 从作者 `russellmitchell_no-pcaps.zip` 按成员范围取回两个员工 `sm.log` 和 cloud_share 两份 Apache 日志，校验 ZIP CRC32、解压长度并记录 SHA-256；没有下载完整 ZIP 或新 PCAP。原件在 [ait_small_sources](<ait_small_sources>)，接收记录分别为 [receipt.json](<ait_small_sources/receipt.json>)、[employee_download_receipt_retry.json](<ait_small_sources/employee_download_receipt_retry.json>)。首次部分请求遇到 HTTP 500 或超时；员工原件在官方文件 URL 的一次重试中成功取得。

| 原始成员 | 有效期内实查结果 | SHA-256 |
|---|---|---|
| `gather/internal_employee_2/logs/sm.log` | 19 次 `Downloaded file`，0 次 `Download failed`；账号 tstevenson | `18d5d3e7425bb7a2f92fa353960ab7487a9e1b09ed77fc35e0e3f294e9e0cc2c` |
| `gather/internal_employee_3/logs/sm.log` | 28 次 `Downloaded file`，0 次 `Download failed`；账号 rblake | `5c2ff0fc8f338490a375f19319c9c1bc0e75b7bbe6c9e3db4665fc9eb7045800` |
| `gather/cloud_share/logs/apache2/other_vhosts_access.log.1` | Jan24 有 19 次文件/文件夹 GET，均 200 | `dd84f0feb1700360023d7d821aa5b2dd2602b0890f18b536ea80f9e3a414b134` |
| `gather/cloud_share/logs/apache2/other_vhosts_access.log.4` | Jan21 有 17 次文件/文件夹 GET，均 200 | `0c140f45b8367d20f876ed853252fa41e8908f8c8505b7339c1b2ee5e8d9b987` |
| `gather/webserver/logs/apache2/cloud.smith.russellmitchell.com-access.log.1` | Jan24 的 19 条服务端下载均唯一对应；解压 1,281,400 bytes，压缩 59,631 bytes | `b87709fc6b7385ae12324688892f5a7ded0f9967951e4297862c9b8b0d008294` |
| `gather/webserver/logs/apache2/cloud.smith.russellmitchell.com-access.log.4` | Jan21 的 17 条服务端下载均唯一对应；解压 1,789,416 bytes，压缩 72,459 bytes | `34fa3568b528da7b8387c48b3fcfadddb38b2c1b6ea2505e18a6b9f92208fef9` |

服务端只计 `GET /remote.php/webdav/...` 与 `GET /index.php/apps/files/ajax/download.php`，排除预览、CSS、图标、上传。36 次响应的 Apache 字节计数字段合计 106,451,409；这不是文件内容字节数。配对要求：员工同一 `transition_id` 的下载动作与完成事件、同名文件或目录，以及服务端秒级时间落在动作区间内（开始时刻向下取整到秒）。结果为 **31 个服务端事件各有唯一动作，且对应 31 个不同 transition_id**；其中 Jan21 12 个、Jan24 19 个，配对事件的服务端计数字段合计 81,894,524。其余 5 个 Jan21 服务端事件未配到这两名员工，未补猜身份。两名员工完整有效期的 47 个成功动作中，Jan22/23 还有 16 个；相应 `.2/.3` 服务端日志本轮未成功取到，未把它们计入 31 个双侧配对。

完整动作和配对在 [normal_download_action_server_matches.json](<ait_small_sources/normal_download_action_server_matches.json>)；服务端筛选结果在 [normal_download_server_events.json](<ait_small_sources/normal_download_server_events.json>)。

**代理/客户端映射已完成。** 在两份代理日志的有效期内，以请求方法、解码路径、完整查询参数、HTTP 版本、状态码作精确匹配；保留所有匹配候选，未按客户端、时间差或字节数删候选。36 条服务端下载全部只有一个代理候选，时间差均为 0 秒。31 个已有成功动作进一步全部匹配到所属员工的配置 IP：`10.143.2.25` 16 次、`10.143.3.65` 15 次，没有冲突或多义候选。另 5 条代理客户端均为 VPN 汇聚地址 `172.19.131.174`，仍缺具体员工动作，保持 unknown，不加入正常事件数。

三层映射已写入 [normal_event_manifest.json](<normal_event_manifest.json>)，包含全部 36 条服务端事件、31 个正常业务标签、5 个未决事件、前一层配对、全部 47 个成功动作、输入 SHA-256、原文行号和代理候选。每条 `flow_label` 仍为 null，`packet_flow_association_status` 为 `not_associated`；业务动作确认不等于整条网络流已判为纯正常。代理和后端的字节字段可能不同，原值分别保留，未用相等字节数作身份条件。

复算脚本为 [assemble_ait_normal_manifest.py](<assemble_ait_normal_manifest.py>)。运行一次通过了六份原件哈希、原行内容、动作 transition_id、文件/目录路径、时序和 31 个动作身份无重复断言。它只读取小原件，默认离线运行；Python 环境需有 PyYAML：

```powershell
python 'E:\OneDrive - Coventry University\中传\毕业论文\工作\独立来源与算法主线_20261118\assemble_ait_normal_manifest.py'
```

两份代理原件的范围响应和校验记录为 [proxy_download_receipt.json](<ait_small_sources/proxy_download_receipt.json>)、[proxy_download_receipt_retry.json](<ait_small_sources/proxy_download_receipt_retry.json>)；`.4` 首次超时后在官方 API 文件入口重试成功。本次只新增这两份代理原件，没有扩展 PCAP 下载。

| 可复查实例（UTC） | 员工动作与完成 | 服务端原件 |
|---|---|---|
| Jan24，ownCloud Manual.pdf | internal_employee_2 第 7059—7063 行：09:23:25.3420815 开始，09:23:40.0218325 完成；文件元数据 size=6,110,772 | `.1` 第 673 行：09:23:25，200，Apache 字段 6,128,196 |
| Jan24，Marketing 目录 | internal_employee_3 第 7259—7263 行：09:44:37.9098926 开始，09:44:57.7178237 完成；目录下载 | `.1` 第 1057 行：09:44:38，200，Apache 字段 7,656,444 |
| Jan21，Photos/Lake-Constance.jpg | internal_employee_3 第 1424—1428 行：06:33:21.467912 开始，06:33:41.3076112 完成 | `.4` 第 213 行：06:33:22，200，Apache 字段 541,394 |

作者固定代码中，ownCloud `Download` 在点击下载后等待文件下载状态，再记录 `Downloaded file` 或 `Download failed`；等待逻辑检查文件存在及临时下载后缀等条件。这比只看到请求意图强，但不等于文件内容哈希校验。此次另外配上了明确文件路径及 HTTP 200 服务端记录。[actions.py 固定提交](https://github.com/ait-aecid/kyoushi-statemachines/blob/13fdbac51c63e40718588f469b7844a712ce2eaf/src/cr_kyoushi/statemachines/owncloud_user/actions.py#L498)、[wait.py 固定提交](https://github.com/ait-aecid/kyoushi-statemachines/blob/13fdbac51c63e40718588f469b7844a712ce2eaf/src/cr_kyoushi/statemachines/owncloud_user/wait.py#L308)

此前检查的 internal_employee_1 与 remote_employee_1，在有效期内各 29 次 ownCloud 登录尝试均对应状态机失败、无下载事件；这个结果仍成立，但不能推广到全场景。新查到的两名成功员工与此前两名失败员工应同时保留，避免再次按局部日志下整体结论。

**现有控制与攻击下载证据没有因此失效，也没有自动完成流标签。** 旧审查已经核实 32 条作者标记的 webshell 命令请求；`whoami` 回传 `www-data` 支持实际执行。WPHashCrack 工具下载后有 53 项解包输出，随后有字典下载和破解执行证据，强于单纯“开始下载”日志。它们仍需落到具体连接；32 条请求不等于 32 次独立攻击，工具下载和控制也属于同一攻击链。[既有原件审查](<E:/OneDrive - Coventry University/中传/毕业论文/工作/独立事件与风险迁移_20261115/候选来源核验.md:23>)

**还差什么，下一步只取什么。**

1. **客户端身份已经补齐，网络五元组尚未补齐。** 已确认方向是员工 `10.143.2.25/10.143.3.65 → webserver 代理 → cloud_share`。后端明确记录来源 `172.19.130.68`、虚拟站点 `cloud.dmz.smith.russellmitchell.com:443`，配置给出 cloud_share=`172.19.130.106`。前端代理日志给出客户端 IP，但没有监听端口或客户端临时端口，代理配置地址也不自动等于 DNAT 前的包级目的地址。两名员工的 `Opening owncloud` 日志入口均为 `http://cloud.smith.russellmitchell.com`（internal_employee_2 第 486 行、internal_employee_3 第 1411 行起）；不能仅凭后端 443 就把前端端口也填为 443。manifest 已分开两段，未知端口和实际网络目的地址保留 null。
2. **当前 intranet PCAP 的观察覆盖尚未证明。** 已登记 PCAP 是 `gather/intranet_server/logs/suricata/log.pcap.1642684613`，配置主机为 `10.143.2.4`。上述正常链的客户端、代理、云服务地址均不包含它，因此这些记录不是“正常用户访问受害 WordPress 主机”的连接证据。但没有全局镜像配置，不能仅凭文件所在主机就断言 PCAP 一定看不到。已读取当前 [contract.json](<E:/OneDrive - Coventry University/中传/毕业论文/代码/tracelight-ids/.worktrees/ait-flow-contract-20261118/configs/ait-flow-contract-20261118/contract.json>) 和同目录 anchors.json；它们未提供这 31 个事件的包级关联。manifest 的 `current_pcap.membership_status` 为 `not_established`，当前包级已关联事件数为 0。接下来可直接拿已有连接提取结果，按 manifest 的动作时段、客户端/代理/云服务端点保留所有候选；该步骤不需要先下载另一份 PCAP。
3. **“正常文件下载存在”与“攻击/正常同服务可比”分别判断。** 本轮证实员工通过 ownCloud 下载，云服务后端为 443；攻击则由受害 WordPress 主机获取工具及字典。端点、应用、角色不完全相同。二者可用于一般文件传输行为的异环境检验，但不能直接宣称已控制服务差异，更不能把域名、主机、代理路径当成攻击行为。若研究问题要求同服务、同受害主机的正常更新/工具下载，此项目前仍缺。优先查该主机有效期内包管理器日志、系统日志、audit 与原始执行记录；只有已发生的更新才能入池，部署期安装不能充数。
4. **正常远程管理只有生成能力的公开证明，暂无本轮逐动作确认。** 作者论文与代码提供 SSH 管理员角色和预定义命令；新取得的两名员工有效期日志未出现 SSH/command 消息。应从场景 `host_vars/<host>/kyoushi.yml` 确定被赋予 SSH 角色的员工，再定向取其 sm.log、目标 auth/audit 小片段，确认命令、成功和连接。不要因框架支持 SSH 就登记成已验证负例。[作者论文 §3.2.2](https://www.skopik.at/ait/2023_tdsc-2.pdf)、[作者环境代码](https://github.com/ait-aecid/kyoushi-environment)
5. **第二次 AIT 执行只先取小证据。** harrison 的 `dataset.yaml`、`servers.yaml`、`attacks.log`、攻击状态机结果、对应 webshell/audit 标签，以及实际成功的正常文件下载动作与服务端记录；这些足以决定是否值得取网络原件。若没有控制执行回传、下载落地证据或可比正常事件，就不因另一个场景名称而扩张实验。

上述本地配置为 [servers.yaml](<E:/OneDrive - Coventry University/中传/毕业论文/代码/tracelight-ids/data/raw/ait-lds/source-audit-20261115/processing/config/servers.yaml:116>)。现有第一天已有 DNSteal，不能把整天当纯正常校准；本轮正常事件由具体动作确认，与“该天没有攻击”无关。

**唯一补充候选：CICAPT-IIoT2024，目前排在 AIT 小证据之后。** 作者公开 PCAP、auditd/SPADE 进程与文件/网络 socket 溯源，以及包含攻击时间、PID、类别的 Attack Information。它提供来自另一团队、另一平台的关联证据路径；论文列 T1105 工具传输和 C&C，但不能直接把战术类别当作本项目的独立控制动作标签。[UNB 官方数据页](https://www.unb.ca/cic/datasets/iiot-dataset-2024.html)

作者论文的正常阶段约 96 小时，攻击阶段约 72 小时，主要业务包括工业协议/平台。其表 4 的 C&C 数字为 **56 个包和 16 个溯源节点**，不是 56 个控制会话或独立攻击。公开说明未给出可直接核验的同服务正常下载、更新或远程管理动作—成功链；大量正常包不能补这个缺口。[作者论文，§3—4、表 4](https://www.cs.unb.ca/~sray/papers/CICAPT_IIoT_paper_Mobiquitous24_author_copy.pdf)

最小待取原件为 `Attack_info.csv`、产生它的 Caldera operation/ability 输出（命令、结果、PID、端点）、`pcap2csv.py` 或等价标注逻辑、对应 T1105 与控制动作的短溯源记录，以及正常业务生成脚本和成功事件。只有这些小件支持明确动作与正常对照，才值得取约 10 GB 全量数据。官方下载入口目前要求填写身份信息；本轮只读页面，未提交表单、未取得上述数据原件。[官方获取入口](https://cicresearch.ca/IOTDataset/CICAPT-IIoT-Dataset/)
