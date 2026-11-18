# CESNET-CC25 与 IEU-C2D 的用途

2026-11-18。本轮把两者用于确定方法近邻和补充来源的用途，不扩成新的训练任务。

## CESNET-CC25：可补周期控制，不能补下载动作真值

[作者数据页](https://netmon.fit.cvut.cz/datasets/cesnet-cc25/)和[固定 Zenodo v1](https://zenodo.org/records/16753890)提供真实 botnet 控制流与 ISP 正常背景。正常和恶意来自不同采集点，作者明确指出时间特征可能暴露采集条件；单条流的包长/方向序列还存在跨类别重复。因此它支持研究跨流信息，不足以证明任意上下文模型都会有效。

已完整取得 `botnet.tar.gz`，4,218,619 bytes，MD5 与官方 `4df0c80a20e6d6f74d91f406d7ca4b48` 一致。实读 64 个 CSV，分别为 32 个采集的 train/test；每条有双向包数/字节、起止时间、包长/方向/包时间序列。许可 CC BY 4.0。具体字段、首行样例和 SHA256 保存在 `evidence/cesnet/botnet_inspection.json`，原始包保存在本地同目录。

正常流包约 14.9 GB。本轮只尝试有界读取其前 1 MiB，服务器读取超时，没有取得正常样例，未继续下载全包。不能称已核实该来源的同服务正常反例。各 botnet 捕获的原有时间切分也不等于跨二进制、跨攻击链的独立划分。

[CNSM 2025 方法](https://dl.ifip.org/db/conf/cnsm/cnsm2025/1571190192.pdf)与作者 [EnhancedDeCrypto](https://github.com/koumajos/EnhancedDeCrypto)给出周期特征路线，可作为观察长度足够时的直接对照。这里没有逐命令、下载文件及主机动作关联，故不承担主任务标签。

## IEU-C2D：真实方法近邻，公开数据不能承接当前来源任务

[C2Detector 2026 论文](https://www.sciencedirect.com/science/article/pii/S1389128626000046)引用的 [IEU-C2D 作者仓库](https://github.com/AmyDylen/IEU-C2D/blob/main/README.md)明确说明，公开内容仅展示 C2 交互状态序列。不能把论文叙述的原始采集规模理解为 PCAP、正常业务及逐动作证据已全部公开。

[C2Detector-V2.0](https://github.com/AmyDylen/C2Detector-V2.0/tree/fb206ad3910acba041e3eaf8ff5ee52931bcc7b3)确有提取和训练代码，属于必须比较的近期直接近邻。提取器的包顺序问题、数据过滤及随机切分已记录在 `algorithm_sources.md`。本轮没有运行这些脚本，也没有将作者在线抓包示例接入本机网络。

**取舍：OpTC 继续装配动作—进程—会话原件；AIT 新发现的正常文件动作补开发证据；Unraveled 保留第二来源。CESNET 仅保留周期控制补充用途，IEU-C2D 保留方法近邻用途。**
