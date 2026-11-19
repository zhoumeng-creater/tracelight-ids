# 正常 SSH 控制的实际来源证据

已有原件中找到 **3 个有效期内、同一 intranet 主机上的正常 SSH 会话**，都唯一关联现有连接；均来自 fox。只有其中 1 条具有直连用户身份、完整退出和 SF 记录，可作为完整正常控制候选。没有 cloud 主机 SSH，也没有 SSH 命令触发的下载。

| 来源与时间（UTC） | 实际连接 | 正常命令／输入 | 证据状态 |
|---|---|---|---|
| fox internal_employee_3，2022-01-15 07:09–07:17 | `CNArPw2fO9dbfOUI3` | 两次 `tail /var/log/syslog`、两次 `ip addr`、两次 Ctrl-C，6 次均有执行回执 | actor 原 IP 直连，连接／命令／退出齐全，SF，raw8 完整 |
| fox remote_employee_2，2022-01-15 09:33 | `CxDSdw1n1YLIhfhBC6` | `tail`、Ctrl-C 有执行回执；`top` 只有尝试 | 观测源为 172.17.130.196，S1；两天后的断开属于异常恢复，不能当作网络完整退出 |
| fox remote_employee_2，2022-01-17 20:30 | `CTfMByqi3ynZyXu0c` | `top` 只有尝试 | 同一观测源，S1；没有命令完成回执 |

完整候选的原始证据是 `拟合来源与划分落实_20261119/ait_new_runs/source/fox/gather/internal_employee_3/logs/sm.log` 第 745–825 行。连接开始于 actor 的 Connecting—Connected 时间范围内；六次命令／输入都落在同一连接观测区间，原始输出按行号和哈希保留。`Executed command` 表示模拟器记录的执行完成，没有虚构进程退出码。

四个 actor 原日志共有 19 次 SSH 连接、39 次命令／终端输入尝试、32 次执行回执。目标为 14 次 mail、5 次 intranet；其中 santos 的 intranet 会话在 1 月 13 日，fox 另一会话在 1 月 14 日，都早于各自作者有效期。全部记录均保留，有效期外的动作不补作 F 正常控制。实际命令只有 `date`、`ip addr`、`ps aux`、`tail /var/log/syslog`、`top` 与 Ctrl-C，没有 wget、curl、apt 等下载工具调用。

已导出：

- `events.jsonl.gz`：58 个会话／命令事件；有效期、完成回执和原 source role 分开记录。
- `event_flow_links.jsonl.gz`：18 条实际候选关系，未给未匹配事件挑近邻流。
- `referenced_connections.jsonl.gz`：7 条原有测量记录，复用原 flow_id、stable_flow_id、raw8 和时间；非本机观察的上下文引用明确为 null。
- `complete_control_candidates.jsonl.gz`：上述 1 条完整正常控制候选，`normal_management_candidate_label=0`；保留 fox/V 来源，尚未授训练资格。
- `same_host_sessions.json`、`summary.json`、`source_receipt.json`：同主机会话、全部计数与原件哈希。

这是同主机的正常 SSH 控制证据。其服务为 TCP/22，与 HTTPS webshell 不同；一个会话内的六次输入也不能算六个独立连接。当前材料仍没有补出有效期内 santos/F 的完整正常控制，也没有补出正常控制触发下载的动作链。

没有新增下载、重跑 Zeek、训练或修改旧表。
