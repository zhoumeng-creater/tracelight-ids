# 与直接近邻的机制区别

本次没有获得支持固定 19 维摘要的新方法依据。成熟研究已经采用网络上下文，但上下文的实体、协议状态、可用时刻和监督单位都有明确含义，不能仅凭“历史有用”推导“上一条 raw8 加全前缀 raw8 均值值得训练”。

| 方法 | 输入与状态 | 时钟、监督与输出 | 与本次 19 维的区别 |
|---|---|---|---|
| EarlyCrow，ISC 2022 | PCAP 生成 PairFlow；按端点对保存协议/报文信息，另建 host、destination、URL profile | 固定包窗口；历史 profile 周期更新；有标签恶意/正常分类 | 保存 DNS、重连、连接起始间隔等有含义的状态；并非只平均同主机全部已完成流的 8 个量 |
| CESNET 周期 C2，CNSM 2025 | 同 client—server—service 的多流序列，周期性检验后提取 44 个统计量 | 完整固定窗口，论文比较 300–3,600 秒；C2/正常序列监督，结果再映射到流 | 保留明确端点依赖和重复通信；少点或非周期序列默认正常，不适合当一次性下载的完整解决方案 |
| GESR，2026 | 正常流构造按端点聚合的通信图，结构条件下重建边语义 | 30/60 秒图快照；benign-only 训练，边异常聚合为主机分数；作者明确定位离线分析 | 条件是图邻域结构，评价对象是主机风险；不是逐完整流查询时刻的控制—下载关系模型 |

EarlyCrow 原文 3.2–3.6 节说明 PairFlow、长期 profile 及其更新：数值均值与布尔协议状态分开维护。其长期 profile 可以跨天，所以本次否定依据不是“前缀太长必然无效”，而是固定 raw8 均值没有保留本任务要解释的关系状态。[作者论文](https://www.doc.ic.ac.uk/~maffeis/papers/isc22.pdf)

已保存 EarlyCrow 作者仓库 commit `dd556631d008517fd4f4d859aabf756dca794e55` 的小型源码，未执行：

- [hostProfile.py 300–402](https://github.com/ICL-ml4csec/EarlyCrowAPT/blob/dd556631d008517fd4f4d859aabf756dca794e55/EarlyCrow/profiles/hostProfile.py#L300)：重连次数、按首包时刻计算的新连接间隔、DNS 请求数等。
- [CSU_dataflow.py 80–118](https://github.com/ICL-ml4csec/EarlyCrowAPT/blob/dd556631d008517fd4f4d859aabf756dca794e55/EarlyCrow/CSUpdatingProcess/CSU_dataflow.py#L80)：端点对分组、协议旗标取 OR、数值统计取平均。
- [dataflow.py 13–23](https://github.com/ICL-ml4csec/EarlyCrowAPT/blob/dd556631d008517fd4f4d859aabf756dca794e55/EarlyCrow/dataflow.py#L13)：从整个 PairFlow CSV 生成各类 profile。公开批处理中的分组还包含 `multiple_label`，不能原样当成无需标签、严格按查询时刻截断的部署实现。
- [classification_performance_unseen_malware.py 13–43](https://github.com/ICL-ml4csec/EarlyCrowAPT/blob/dd556631d008517fd4f4d859aabf756dca794e55/experiments/classification_performance_unseen_malware.py#L13)：HTTP/HTTPS 不同字段集、100 树 RF 的恶意/正常监督分类；没有逐 actor 动作的因果关系监督。

CESNET 作者把检测对象先限制为有足够数据点的周期通信。论文明确报告短序列未进入周期集合造成的漏检；这种正反证据只支持重复 C2 的针对性机制，不能支持任意历史组合。作者实现 commit `0c8c1bce8e2a83ea79d6a525020ba94593565023` 的 `new_model.py` 第 443 行跳过不足 10 条流的序列，第 445–479 行进行周期检验和后续统计。[论文](https://dl.ifip.org/db/conf/cnsm/cnsm2025/1571190192.pdf)、[作者代码](https://github.com/koumajos/EnhancedDeCrypto/blob/0c8c1bce8e2a83ea79d6a525020ba94593565023/new_model.py#L443)

GESR 的图快照和主机聚合说明“上下文残差”已有直接近邻。它的输入、正常训练支持和输出单位均不同于当前动作级/流级任务；论文中的实现发布承诺不能当作已在本地复现的基线。本次没有依据将固定 19 维改名为此类机制。[作者论文，第 3 节](https://arxiv.org/html/2605.07536#S3)

已有数据支持的具体事实是：OpTC 控制阶段存在重复端点通信；AIT 能离线核实具体 `wget` 行为与下载顺序，但完整指令连接在下载查询时尚未可用。若要使用指令尚在进行时的报文或请求状态，输入接口与观察时钟必须另行改变，不能声称现有完整流 19 维已经做到。本次没有形成这样的替代算法，也没有运行新窗口、模型或阈值比较。
