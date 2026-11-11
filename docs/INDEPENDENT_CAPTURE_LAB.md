# 第二十轮实验室指南：CIC2018 独立采集上的两组 HGB

本轮检验第十九轮的四项 60 秒历史观测，在另一套采集、另一台扫描源和多攻击族训练条件下，能否提高低误报扫描检出，并记录对其他攻击的代价。只重新拟合两组直方图梯度提升树（HGB）。标签及时间切分依据采集元数据确定，在查看任何本轮模型分数前已固定。标签为 `independent-capture-lab-v1`。

## 实验室启动

在 Windows 实验室的 PowerShell 执行。使用已有 route-core 环境和新的 checkout，默认从昆士兰大学下载并验证 780,113,687 字节的 ZIP，展开 4,222,783,755 字节的 CSV。原始抓包与修订流表已在准备阶段核对，实验室直接使用随版本固定的两个行号清单。

```powershell
Set-Location -LiteralPath 'D:\zhoumeng\Graduate'
git clone --branch independent-capture-lab-v1 --single-branch https://github.com/zhoumeng-creater/tracelight-ids.git tracelight-independent-capture
if ($LASTEXITCODE -ne 0) { throw 'Clone failed' }
Set-Location -LiteralPath 'D:\zhoumeng\Graduate\tracelight-independent-capture'
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\RUN_INDEPENDENT_CAPTURE.ps1
```

已有同一份 UQ 官方 ZIP 或 CSV 时，首次执行可加 `-Source '完整文件路径'`。默认 Python 为 `D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe`。中断后在同一目录、相同参数下重跑最后一条脚本。下载支持断点续传，数据准备、历史重放、两个模型各自的拟合、评分与成本测量均有独立校验记录。`-CheckOnly` 只做合成测试。

结束后把 `local\independent-capture-r20\independent-capture-results-*.zip` 和同次的 `.zip.sha256.json` 放入实验归档。无需重新打包历轮结果。原始 CSV、分片和工作目录留在实验机。

## 为什么重新核对扫描标签

NF-CSE-CIC-IDS2018-v3 与第十九轮的 NF-ToN-IoT-v3 来自不同采集，具备相同的 48 个模型字段和计算历史所需的原始时间、源地址、协议等字段。CIC2018 的内部渗透攻击包含下载、后门通信及 Nmap 扫描。UQ 原始文件用 `Infilteration` 表示一个宽类，落入两个扫描源和规定目标、时间窗的记录又有大量被标为正常。直接拿这个字段区分扫描、其他攻击和正常流，会同时污染训练与误报统计。[UQ 数据发布页](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) · [原采集与 AWS 数据](https://registry.opendata.aws/cse-cic-ids2018/) · [UNB 场景说明](https://www.unb.ca/cic/datasets/ids-2018.html) · [DistriNet 逐类修订说明，第 11.3 节](https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html)

| 扫描证据 | 2 月 28 日 | 3 月 1 日 | 认定依据 |
|---|---:|---:|---|
| 修订流表的已标注扫描 | 19,762 | 5,734 | 有向五元组和首包毫秒在两边唯一，结束毫秒、双向包数一致，修订标签为 Nmap Portscan。 |
| 已登记扫描事件内，经原始报文核对的单包 SYN | 16,418 | 9,028 | 两天指定扫描源与目标、方向、时间和排除 DHCP 的作者规则成立。UQ 原始流与抓包的五元组和首包毫秒唯一，IP 长度及 TCP 窗口也一致。 |

两份清单互不重叠，共 50,942 条。其中 31,931 条原标正常，单包类的 25,446 条全部原标正常。时间和地址只用于核对真值、分组和计算历史。排除其余 61,050 条疑似扫描和 136,779 条未核实的原生 `Infilteration`；其他攻击和正常保留 UQ 标签。额外核对的 267 条原标正常单包可在抓包找到，方向均为发往扫描主机，不能用来证明同一源主机向外发出相似流时的误报率。两种扫描证据在评价时分项报告。[流表核对](capture_validation_20261111/matched_scan_evidence.json) · [报文核对](capture_validation_20261111/packet_scan_evidence.json) · [修订团队标注代码](https://github.com/GintsEngelen/CNS2022_Code/blob/f0ce502818e59e6cd062720ab2286c5ff6f2bdec/Labelling/CICIDS2018_labelling_fixed_CICFlowMeter.ipynb)

## 固定的训练、校准与评价

按流完成时间（UTC）切分，再按无方向端点对及协议将通信组固定分配给一种角色。训练按二元标签等额取样，评价在可判定记录内按固定行号优先级抽样，无攻击族配额。不同角色没有重复行或通信组，主机允许重复。校准和评价的开头 60 秒仅供历史预热。固定的行号、最终族及证据来源摘要在实验室复核后才开始训练。[登记配置](../configs/capture-validation-2018/registration.json) · [实际覆盖和行身份](capture_validation_20261111/cohort_coverage.json)

| 角色 | UTC 完成时间 | 实际样本 |
|---|---|---|
| 训练 P | 2018-02-28 19:00 前 | 4,096 正常。攻击有 DDoS 2,408、BruteForce 1,059、DoS 555、扫描 70、Web Attacks 4。 |
| 正常校准 C | 2 月 28 日 19:00 至 3 月 1 日 14:00 | 10,000 条正常。 |
| 前向评价 E | 3 月 1 日 14:00 至 3 月 3 日 00:00 | 46,817 正常、209 扫描、2,974 Bot。其中 84 条扫描经修订流表互证，125 条单包扫描经原始报文核对。 |

前向扫描全部来自同一台在训练期出现过的主机，它在训练样本中没有充当扫描源。测试期的其他攻击只有 Bot，训练样本中没有 Bot。前向数据中，宽一些的“单个 TCP SYN、无返回包”有 **125 条扫描、1,365 条原标正常**。第十九轮预先定义的 47 字段精确模式为 **0 条扫描、1 条正常**。因此本次可验证新采集中的扫描及单包 SYN 检测，原精确模式跨环境的表现仍不可评价。

## 模型、资源与判断

两臂固定为原始 48 字段 HGB、原 48 字段加四项历史的 HGB。模型均为 200 轮、每棵最多 31 个叶节点、学习率 0.1、相同种子。四项历史是同一源 IP 和协议在此前 60 秒内完成的流数、不同目的 IP 数、不同目的端口数和无返回包比例。输入排除绝对时间、原始 IP、标签和身份字段；同一毫秒先取历史再更新。两臂用同一批记录，在 CIC2018 的训练段重新拟合。

- 默认取最多 24 个物理核重放全源历史。两组 scikit-learn HGB 在两个独立 CPU 进程并行拟合，24 核机器上每组 12 个线程。该固定实现使用 CPU。换成 GPU 树模型会改变被验证的学习器。本轮没有 MLP、TabM 或预训练权重任务。
- 训练结束后依次对两臂做单线程及全线程部署计时，核对 CPU 重放及告警。另测实际维护历史状态的逐流延迟、吞吐和热态进程内存。单键片段的成本与全源准备分开报告，生产导出器的到达延迟另需实测。资源 CSV 记录各阶段实际进程树 CPU、内存和整卡采样。
- 每臂只用 C 的正常分数登记阈值，升序第 9,901 个，严格大于才告警。E 最多容纳 468 个正常误报。沿用第十九轮门槛：实测误报率不超过 1%，相对原 48 字段的误报数不增、总检出不减，扫描检出与 0–1% 误报区间归一化 pAUC 均提高。另报 Bot 检出净变化，若下降，须写明扫描收益牺牲多少其他攻击；两条结论分别判读。
- `last_invocation.json` 记录执行是否完成，`evidence/pair.json` 给出配对结论、分证据扫描检出及部署开销，`evidence/pair-strata.parquet` 与逐行评分用于复核按族、日期和主机的得失。较宽的 SYN 组和精确 47 字段组分别标注是否同时观察到两类。数据、阈值和评价分数保持冻结。新采集若未显示收益，就按本次结果限定论文结论。

本轮来源为 CSE-CIC-IDS2018（Communications Security Establishment 与 Canadian Institute for Cybersecurity），其 UQ NF-v3 导出及 DistriNet 修订标签。关于同源导出与额外时间字段的方法见 [UQ 作者论文](https://arxiv.org/abs/2503.04404)。

标签行号、证据匹配条件及原始分卷的恢复步骤见[标签复算指南](capture_validation_20261111/reproduce/README.md)。本机对真实文件复核了 68,192 条入选记录的 48 个输入字段、三个角色的行号与标签摘要。真实模型的得分、误报和部署成本由本轮实验室运行产生。[真实数据检查](capture_validation_20261111/metadata_validation.json)
