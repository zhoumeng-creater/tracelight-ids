# AIT 独立事件与同服务正常业务取证

固定版本：`risk-audit-evidence-v1`。本批只补网络证据，**没有训练、模型前向或误报率效果评价**。第二十六轮的六项固定风险规则已在本机复算，实验室无需重跑。判读依据见[风险与来源审计](R26_RISK_AUDIT_20261115.md)。

## 运行

在实验室仓库 PowerShell 中执行：

```powershell
git fetch origin tag risk-audit-evidence-v1
git worktree add .worktrees/risk-audit-evidence risk-audit-evidence-v1
Set-Location .worktrees/risk-audit-evidence
.\scripts\RUN_AIT_PACKET_EVIDENCE.ps1
```

默认复用既有 `D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe`；可用 `-Python '实际的python.exe'` 指定其他 Python。取证程序只依赖标准库，不升级原模型环境。合成检查可单独运行 `-CheckOnly`，此选项需要既有 pytest。

如已有这个确切 PCAP，可指定 `-Pcap '实际路径\log.pcap.1642684613'`，长度和 CRC 通过后跳过下载。其他 PCAP 不得同名替代。

没有 PCAP 时自动从固定 Zenodo 版本按成员范围下载，使用 2 个并发请求和 1 MiB 分块，逐块记录 SHA-256。中断后保留已经校验的块，重复同一命令续传。完整压缩体约 278.4 MiB，PCAP 约 327.3 MiB；不用下载约 6.64 GiB 的完整场景 ZIP。完整成员通过长度、解压终止和 CRC 检查后才替换正式文件，并清理自己的压缩缓存。已有不匹配 PCAP 会停止，不覆盖用户文件。

## 一次输出的证据

| 文件 | 用途 |
|---|---|
| `run-status.json` | 本批是否完成；失败原因也进入回传包 |
| `packet-evidence-summary.json` | PCAP 完整性、时间覆盖、解析统计、资源耗时、固定字段口径与准入缺口 |
| `tcp-connection-candidates.csv` | 有效四日内全部 IPv4 TCP 连接候选，含双向观测、SYN/FIN/RST、捕获字节和起止时间 |
| `tcp-service-distribution.csv` | 全部 TCP 候选的服务、方向及数量分布；未解析背景不标正常 |
| `action-connection-candidates.csv` | 68 次受控正常 HTTPS 动作与 32 条控制请求的零/单/多连接候选；保留发起、成功、连接起止和正常确认未知状态 |
| `victim-download-connection-candidates.csv` | 两个下载动作窗口中的受害机外连，以及无 SYN 时方向未定的候选；不把所有候选标为下载 |

控制请求候选允许 80/443，明确 HTTPS 的正常任务限定 443；共享 VPN IP 不当成独占身份。连接可复用，重复命令保留为动作记录。候选单一也不等于标签纯净。所有捕获 TCP payload 计数均不替代 Zeek IP 字节字段，输出不能直接送进现有八字段模型。

脚本自动打包输出目录顶层 JSON/CSV/log 为 `local/ait-packet-evidence-results-时间.zip`，并生成 `.sha256`。**回传这个 ZIP 和 SHA 即可**；PCAP 留在 `data/raw`，不重复打包原件、分块、权重和旧实验。

## 资源安排与结束条件

本批瓶颈是约 292 MB 的远端传输，使用两个并发请求；完整 PCAP 只解析一次，内存中复用连接索引完成全部动作关联。当前串行解析一个有状态包流，不宣称已用满 CPU 多核。没有神经网络工作，GPU 任务数为 0；不为提高占用率重跑 TabM、MLP 或 HGB。

成功表示网络证据清单已完整输出。缺 SYN、未匹配、多匹配、正常确认时刻未知均须保留；它们不是程序失败，也不能被忽略后继续评分。下载/校验失败则保留失败回传和已验证块。返回证据后决定是否具备独立来源回放条件，不自动扩展模型或阈值矩阵。
