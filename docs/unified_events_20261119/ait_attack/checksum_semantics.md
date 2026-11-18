# AIT intranet 固定攻击连接的采集语义核查

32 条控制候选可采用 `ignore_checksums`（Zeek `-C`）口径。3544 及三条下载候选 3574、3575、3578 的捕获记录缺少 IPv4 声明的完整段字节，禁止进入完整 raw8 模型表。两次下载事件及其候选关系照常保留；首条控制请求仍有 3544/3545 两候选，不能把排除受损模型行当成消除了事件关联歧义。

## 固定范围与复算

读取 R27 已有 32 条控制请求的全部 33 个 TCP 候选，以及两次下载的 3574、3575、3578，共 36 条。5044 持续日志传输连接 0 不属于已锚定的外连下载，未扩入本次逐包范围。按双向端点与原始首末包时刻关联，分别断言 origin/responder 包数与 R27 目录一致；没有重复归属包。输入共 64,744 包，36 条连接全部具备序列号匹配的三次握手。

运行 `python diagnose_checksum_semantics.py` 可重建本报告及 JSON。脚本复用相邻 `ait/diagnose_checksum_semantics.py` 的 RFC 校验和函数与原 R27 PCAP 读取器，不重跑 Zeek、不改包、不改冻结 contract。全部 contract 文件执行前后 SHA256 一致。原 PCAP SHA256：`12c798074e27969ea40719700ecc8cb07ecda76e4f0751d746e02e2398067790`。

## 校验和与长度分别判断

intranet 主机 facts 的快照为 `2022-01-25T06:55:29.698260Z`：`ens3=10.143.2.4`、`virtio_net`、`tx_checksumming=on`、`tx_checksum_ip_generic=on`，GRO/GSO/TSO 均为 off，Suricata 使用 ens3/af-packet。facts 来自作者 ZIP 的固定成员，CRC32 2309763221；SHA256 `fae3b9e4e17b5236e88a5645f0b3b0ecc67e36a4d4bd19eee28e758d1d04746f`。快照本身不是整个采集期配置不变的证明；逐包模式提供了同方向的直接证据。

| 方向 | 包数 | IPv4 正确 | TCP 可校验 | TCP 正确 | TCP 错误 | TCP 无法校验 |
|---|---:|---:|---:|---:|---:|---:|
| intranet → 远端 | 29,226 | 29,226 | 29,226 | 2 | 29,224 | 0 |
| 远端 → intranet | 35,518 | 35,518 | 7,395 | 7,166 | 229 | 28,123 |

发送方向的 29,224 个错误 TCP checksum 全部等于 pseudoheader 部分和，符合 Linux TX checksum offload 的预填值。接收方向的 229 个可校验错误值也全部等于该部分和，但不能仅用 intranet 的 TX 配置解释远端记录。无法校验的 28,123 包全部是 IPv4 声明长度超过实际捕获字节；没有 IP 分片或零长度，PCAP 的 captured/original frame length 字段却相等，因此不能把它们写成普通 snaplen 截断，也不能据此断言网络真实丢包。[Linux 4.15 checksum offload 说明](https://raw.githubusercontent.com/torvalds/linux/v4.15/Documentation/networking/checksum-offloads.txt)、[对应 TCP 实现](https://github.com/torvalds/linux/blob/v4.15/net/ipv4/tcp_ipv4.c#L526)。

## 32 条完整控制候选

这 32 条共有 intranet 发送 13,788 包、远端发送 4,756 包。发送包全部是与 offload 部分和吻合的错误 checksum，远端包全部校验正确，长度完整。strict 恰好丢弃全部本机发送包；`-C` 的双向包数与 IP 字节数逐连接吻合原始记录。因此 `-C` 是由采集语义确定的口径，strict 只保留诊断用途，二者不可混作等价模型输入。[Zeek 8.0.11 官方说明](https://docs.zeek.org/en/v8.0.11/quickstart.html)明确说明本机 checksum offload 会影响默认校验，可用 `-C` 忽略。

这里的“可用”仅解决这 32 条固定连接的校验和与完整长度问题，不改变动作标签、首请求双候选、可用时间或同服务正常分母。完整候选 ID 与两臂所有 UID、raw8、导出时间均在 JSON。

## 四条受损候选

| packet ID | 远端原始包数 | 声明长度缺字节的包数 | -C 实际计入远端包数 | -C missed_bytes |
|---|---:|---:|---:|---:|
| 3544 | 410 | 8 | 402 | 21709 |
| 3574 | 9 | 1 | 8 | 2709 |
| 3575 | 17,814 | 15,935 | 1,879 | 53023422 |
| 3578 | 12,529 | 12,179 | 350 | 139572637 |

这四条的 `-C` 都标成 SF，但逐包对账显示：`-C` 计入的恰好是长度完整的包，缺少完整段字节的记录全部未计入。strict 计入的则恰好是双 checksum 正确的包。两种口径都不能恢复缺失数据；`SF`、握手完成或业务成功不等于完整流特征。

例如 3574 的原始第 574025 包声明 IPv4 长度 2,761，实际仅 1,472 字节；3575 第 574045 包声明 2,759、实际 1,472；3578 第 599934 包声明 7,042、实际 1,472。JSON 保留每方向/异常类型的最早包号、时间、长度与 checksum 数值。

当前原件足以确认发送 checksum offload，也足以确认四条候选的输入缺损；尚不足以唯一确定远端/虚拟网络聚合与捕获长度处理的哪一层产生了缺损。执行决定是保留事件及全部关系，阻止四条受损完整流模型行，不用改 checksum、补长度或挑较完整候选来修饰下载结果。
