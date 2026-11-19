# 采用声明 IP 长度表示的测量合同

2026-11-19 更新。**当前采用 `declared-ip-truncation-v1`，不再采用直接丢弃受损包后的剩余计数。** 旧原件及旧完整 raw8 禁用记录均保留；新表示以独立派生文件和契约登记。

## 为什么可以改变用途

原题要求公开数据的机器学习检测与成本比较，没有要求每个包的 payload 完整。[原题第 1 页](<E:/OneDrive - Coventry University/中传/毕业论文/选题/SpecialProject.pdf>)。动作关联可靠和传输正文完整是不同证据：已证动作的网络连接即使缺失部分正文，活动标签仍可成立，不能因此自动禁止所有观测特征用于学习。

CTU-13 作者为正常、恶意及未知背景统一发布仅保留协议头的截断 PCAP，说明完整 payload 不是网络检测研究的一般前提。这一已知隐私截断不等同于本例的原始长度元数据异常，不能借它声称已恢复原包。[作者截断方法](https://www.stratosphereips.org/blog/2015/7/17/new-dataset-ctu-13-extended-now-includes-pcap-files-of-normal-traffic)。Zeek 的 IP 字节字段取自 IP 头声明长度，payload 字节和捕获丢失另有语义。[Zeek 8.0.11 官方字段定义](https://docs.zeek.org/en/v8.0.11/scripts/base/protocols/conn/main.zeek.html)。

## 已验证的新表示

通用规则要求 IPv4 头完整、checksum 有效且无分片，并具有完整的 TCP、UDP 或受支持 ICMP 头。派生 PCAP 的 `orig_len` 低于 `L2偏移 + IPv4 total_length` 时，提升至该声明值；caplen、捕获字节与时间戳不变。它表示部分捕获的声明 IP 数据报，不补齐正文。通用全捕获处理不逐流要求序列号或 ACK 证明。

此前针对 russellmitchell 3 个下载 TCP 的专项验证另有独立序列号、ACK/FIN 证据。该验证改变 28,115 条下载记录的 orig_len；3 个下载 TCP 的捕获记录数及声明 IP 字节累计全部被 Zeek 8.0.11 `-C` 计入，端点、起点、duration 不变。32 个控制候选、38 个 cloud 候选 raw8 不变，两个原件哈希不变。[验证结果](<E:/OneDrive - Coventry University/中传/毕业论文/工作/拟合来源与划分落实_20261119/ait_recovery/declared_length_representation/validation.json>)；[派生记录回执](<E:/OneDrive - Coventry University/中传/毕业论文/工作/拟合来源与划分落实_20261119/ait_recovery/declared_length_representation/representation_receipt.json>)。

此后 raw8 的包数定义为捕获记录数，IP 字节定义为声明 IP 长度累计。卸载或聚合可使一个捕获记录不同于一个物理线上包；不据此声称物理线上包数、完整线上字节、完整文件正文或没有丢包。缺失 payload 仍缺失。也不从成功回执、文件大小或另一捕获点补数值。

## 已完成的装配与训练用途

通用全捕获规范化已完成五件原件，共导出 109,512 条连接。santos intranet/cloud 分别 14,538/17,562 条，fox 分别 21,514/33,973 条，russellmitchell cloud 为 21,925 条；各件均有 normalization_receipt 与 extraction_receipt。处理全部记录，不按标签或动作修改。3 个下载 TCP 的独立序列验证只证明该专项目标，不能概括全背景。

F=santos、V=fox 的同场攻击、正常业务与未标注背景已统一处理。最终模型表含 F32 正例＋439 正常、V17 正例＋430 正常、C25 正常，共 943 条监督记录；F/V 各有明确的攻击下载正例。109,512 条捕获观察形成 88,484 条本机视角历史；另有 43,420 条待评分观察及 31 个正常业务评价单位。[数据摘要](<E:/OneDrive - Coventry University/中传/毕业论文/工作/拟合来源与划分落实_20261119/dataset/summary.json>)。质量/体量/方向与类别的关系作为审查元数据保留，不直接加入模型。缺失正文不再是整体禁训理由；若某种捕获瑕疵只出现在攻击类，成绩不能仅据此解释为跨场景行为识别。实际正常下载及同服务正常浏览用于检验这个解释。

下一步真实拟合使用实验室 RTX4090D 工作站，执行固定 HGB8/HGB27 比较。该估计器使用 CPU；独立 runner 已实现原生 NaN、固定两臂与逐身份变化报告，7 项合成测试通过，真实数据拟合和评分仍为 0。AIT 使用记录的离线 Zeek `replay_export_time`；OpTC 外源采用显式完成流回放，仅作跨源历史开发回测，不作为历史确认或真实在线早检证据。
