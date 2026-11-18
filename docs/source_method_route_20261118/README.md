# 事件证据包

现行决定见 [SOURCE_METHOD_ROUTE_20261118.md](../SOURCE_METHOD_ROUTE_20261118.md)，完整状态和核心工件 SHA256 见 [outcome.json](outcome.json)。本目录保存公开来源的必要记录和复现脚本；原始 gzip、PCAP、模型及论文全文未纳入 Git。

- [OpTC 下载事件](evidence/optc/optc_event_slice.json)：2 个同文件会话，其中 1 个吻合红队记录的同一秒；保留四表全部关联和时间差。
- [OpTC 控制通道](evidence/optc/optc_control_channel_slice.json)：24 条属于同一 Empire 控制链的会话；不提供逐命令执行标签。
- [OpTC 未定下载候选](evidence/optc/benign_file_business.json)：6 个 PE 对象均为 unknown，未赋正常标签。
- [AIT 正常事件](normal_event_manifest.json)：31 条员工动作、后端响应、代理请求唯一匹配；5 条额外 VPN 请求保持 unknown，网络流关联数为 0。

OpTC 脚本使用 Python 3 和 requests。将固定来源原件放入 `evidence/optc/` 后可离线复算；下载事件脚本也可按内置指定清单取得这些公开日志。

```text
python scripts/assemble_optc_event_slice.py --with-conn --offline
python scripts/assemble_optc_control_channel_slice.py
python scripts/inspect_optc_benign.py
```

AIT 脚本使用 Python 3 和 PyYAML，仅离线读取六份小日志、已保存的动作—服务端配对，以及场景配置。异地检出应显式指定原件目录和两份已有配置：

```text
python assemble_ait_normal_manifest.py --sources ORIGINAL_AIT_SMALL_SOURCES --servers ORIGINAL_SERVERS_YAML --contract EXISTING_AIT_CONTRACT_JSON --output OUTPUT_JSON
```

脚本检查固定原件 SHA256；未取得同一原件时不能用另一次采集代替。正常动作的前端端口、实际目的地址、五元组及 PCAP 覆盖保留未知，不从后端 443 补猜。

本包核验了正常日两个 gzip 的完整读取、哈希与全部候选对应，OpTC 四表 UID／FUID／五元组与文件完整性，以及 AIT 三方唯一性。脚本编译及 Git 差异检查通过。所有模型加载、前向、训练及新阈值选择均为 0。
