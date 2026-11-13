# 43-1 的来源和采集时间

43-1 是已留出、未评分的历史抓包。公开标签同时包含 20,574,934 条正常流、37,911,674 条扫描、3,498 条纯控制、14 条控制兼下载及 1 条单独下载。全量读取 67,321,809 条流，文件内七类标签计数与作者表完全一致。下载端点直接提供 9,363,261,631 字节的原始 Zeek 文本。登记的 SHA-256 由本地读取完整原文取得。作者 README 所列 SHA-256 标识恶意软件样本。[作者数据目录](https://www.stratosphereips.org/datasets-iot23) · [43-1 原始文件](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-43-1/labeled/conn.log.labeled) · [数据登记](../../configs/raw-workpoint-20261113/source.json)

该抓包晚于 2018 年 10 月 26 日结束的正常校准。对于此前已经打开的第 42、44 场，两者都与 43-1 位于同一实验室、同一网关，同一天采集。三份原始 PCAP 和受感染设备各不相同。根据原始流起始时间及持续时间重建物理流序下的单调时钟，三个场景的时间如下。

| 场景 | 作者登记的设备 | 2019 年 1 月 10 日 UTC 评价时段 | 与 43-1 共享流记录 |
| --- | --- | --- | ---: |
| 42-1 | RPi-04 | 13:34:51.936 至 21:55:58.212 | 0 / 4,426 |
| 43-1 | RPi-05 | 18:22:58.631 至 19:28:24.181 | — |
| 44-1 | RPi-06 | 20:06:30.754 至 22:01:05.569 | 0 / 237 |

42-1 与 43-1 时间重叠，44-1 在 43-1 结束后开始。逐行检查了 43-1 全部原始时间和连接五元组，要求原始时间精确至微秒、源与目的地址、端口和协议均相同。两次比对均无重复键。43-1 可用于检验同实验室一个未评分设备场景上的检测表现，新的外部前向时间验证仍需要独立采集。[42-1 作者说明](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-42-1/README.md) · [43-1 作者说明](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-43-1/README.md) · [44-1 作者说明](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-44-1/README.md) · [完整数字](source-identity.json) · [复核脚本](audit_source_identity.py)

在 i7-13700HX 上，只读 43-1 开头 192 MiB，截取完整的 1,447,538 行做解析对照。第二十三轮串行程序用时 37.13 秒。新程序单进程用时 46.02 秒，启用 8 个实际解析进程后用时 12.81 秒，测得平均并行度 2.19 核。三个版本所得全部观测列、真值列、原行序和时间摘要一致。测试期间另有数据下载，实验室完整耗时以本次运行资源记录为准。
