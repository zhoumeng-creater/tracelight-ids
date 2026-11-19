# 完整捕获的共同提取合同

合同版本 `declared-ip-truncation-v1`，实现位于新工作树 `scripts/fit_contract/pcap_normalize.py` 与 `pcap_extract.py`。

规范化器读取完整 PCAP，不接收动作、标签、端点名单或时间筛选。对完整且校验和正确的 IPv4 头、完整 TCP/UDP/已支持 ICMP 头、无 IP 分片的记录，仅在 `orig_len < L2_offset + IPv4_TotalLen` 时提升 record `orig_len`。`caplen`、时间戳和全部捕获字节不变。无效、不完整、不支持的头部和分片保持原样，按原因计数；不填零，不重建载荷。

该表示统计的是捕获记录和 IP 头声明的数据报字节，不是重新测得的物理链路包数或线速字节。完整帧和不适用的记录继续使用固定 Zeek 原生解析；非 IPv4、零长度或其他异常的处理不能凭合同名称当成已经完成 IPv4 规范化。每源 receipt 保存具体状态计数，不能只看连接表的模式名称。

每个完整捕获使用相同的 Zeek 8.0.11 `-C` 与既有 export policy。只有源/派生 SHA 完全相同、版本/模式/政策及日志 SHA 均吻合，才允许复用既有完整出口。正常背景也完整保留，不把事件关联子集当成背景流。`replay_export_time`、EOF 与 shutdown 标记保留；实际 detector 接收时间仍为 null。

## 工件

- `normalization/normalization_receipt.json`：原始/派生 SHA、全量包数、修改与异常计数、逐记录不变验证、转换器 SHA。
- `normalization/record_changes.jsonl.gz`：每条修改的源记录索引、原/新 `orig_len`、`caplen`、完整头部依据。
- `normalization/capture_truncation_records.jsonl.gz`：全部可见 IPv4 声明长度大于实存长度的记录，包含端点、时间、缺字节数及是否修改。
- `zeek/`：完整 conn/http/ssl/files 等原始 JSON 行日志及固定出口 receipt。
- `connections.jsonl.gz`：所有连接、raw8 八字段及原始 `zeek_record`，未关联标签为 null。
- `uid_flowkeys.jsonl.gz`：新 UID 与稳定关联键。键绑定原 capture SHA、协议、有向端点、端口与精确开始时间；重复键保留所有记录并报告 cardinality。
- `extraction_receipt.json`、`schema_example.json`：运行输入、代码指纹、输出哈希和实际一行 schema。

连接 ID 为 `ait:{scenario}:{capture_host}:{uid}`。跨出口关联使用 `flow_key_id`，不能假定随机 UID 不变，也不能把同一真实 TCP 的多个 Zeek 尾段当成多个独立动作。

`raw8_missing_fields`、`duration_missing`、`raw8_complete` 保留缺字段含义。`capture_provenance` 用协议、双向端点及包时间落在 Zeek 观测区间内关联全部候选；保留改动数、截断数、声明缺字节的记录求和及歧义数。该求和不是去重后的 TCP 流缺口；Zeek `missed_bytes` 和原始包级证据分别保留。未匹配到截断记录也不自动证明载荷完整。

本组件不赋正常/攻击标签、不决定 F/V/C、不授训练资格，也不运行模型或阈值搜索。后续监督和校准使用哪个事件及连接，由独立登记确定。
