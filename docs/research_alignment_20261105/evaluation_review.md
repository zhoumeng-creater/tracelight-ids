# 分组、低误报与资源验收：近期成果核查

2026-11-05。本次查阅论文原文、正式发表记录及作者项目；没有运行这些外部项目。文献用于确定可比较的协议，不把他人的指标并入TraceLight排行榜。

## 1. 广度检索与采用范围

| 成果 | 身份与一手来源 | 对本项目的作用 | 当前不采用的部分 |
|---|---|---|---|
| TabArena | NeurIPS 2025 Datasets and Benchmarks；[正式论文](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf)、[作者项目](https://github.com/autogluon/tabarena) | 区分固定配方、统一调参协议和自行搜索系统；同时记录质量、时间与资源 | 不搬入大规模HPO/集成，不将本轮固定树与多checkpoint MLP称为公平调优排行 |
| TabReD | ICLR 2025；[正式论文PDF](https://openreview.net/pdf?id=L14sqcrUC3)、[作者项目](https://github.com/yandex-research/tabred) | 时间分布变化可能改变方法排名；已有随机切分上的优势不能直接代表部署 | 不把重新切分已使用数据当成全新的独立确认 |
| Beyond IID / BeyondArena | 2026-06预印本；[原文](https://arxiv.org/html/2606.30410v1)、[共享作者实现](https://github.com/autogluon/tabarena) | 外层任务和内层选择都应保留分组/时间结构；优化seed不能替代独立环境 | 不将组hash等同独立采集事件，不以单一环境巨大行数制造显著性 |
| Do Transformers Actually Help Intrusion Detection? | 2026-07修订预印本；[原文](https://arxiv.org/abs/2606.11098)、[作者项目](https://github.com/zachmocz/temporal-ids-bench) | 检查输入构造、填充和拆分是否解释架构收益；先查机制再换网络 | 其任务为真实流序列，本项目为单行流特征；不搬用Transformer或比较其宏F1数值 |
| Multi-Class, Multi-Tier Network Intrusion Detection | 2026-09预印本；[原文](https://arxiv.org/html/2609.36039v1)、[作者项目](https://github.com/YufengXin/network-attack-research) | 标签和流提取器版本必须绑定；需展示逐族支持，不能只报告二分类总体成绩 | 原文§IV使用分层随机70/15/15划分，不能替代本项目的联合分组边界；不因“纠正标签”字样默认其协议解决所有相关性 |

上述论文均有与当前问题直接相关的核查点。搜索中还发现多个只以总体准确率论证复杂融合网络的IDS方案，因未能对应本项目低FPR工作点、字段语义或部署预算，不进入实现候选。

## 2. 原文与作者实现的实查

### TabArena / BeyondArena

原文TabArena §2.1–2.3区分模型配置搜索、数据与评价协议；BeyondArena §4–5及附录F讨论分组/时间任务和内层选择。当前作者仓库的`validation_protocol.py`将协议作为独立记录保存，`BEYONDARENA_VALIDATION_PROTOCOL`启用task-specific validation，`experiment_violations`检查实验是否覆盖任务规定的split structure。核查代码为[作者文件](https://github.com/autogluon/tabarena/blob/main/packages/tabarena/src/tabarena/benchmark/validation_protocol.py)，本次读取的Git blob为`bf1c47dc03b9045c2fdcc1443a2bebed3fb46152`，关键位置204–213、331–361行。当前代码持续更新，不声称它与2025论文发布时逐字节相同。

作者[模型示例](https://github.com/autogluon/tabarena/blob/main/examples/benchmarking/run_quickstart_tabarena_model.py)明确区分接受共享预处理/验证/调参协议的model与自行完成搜索的system，并分别传入`X_val`、时间和CPU/GPU预算。采用的是“公开记录并控制选择机会”这一协议原则，不照搬其8折训练和大规模HPO。

TraceLight最小机制实验应保持输入、初始化、hard标签、更新次数、checkpoint候选数及选择规则一致。若改变输入表示，另计预处理拟合与推理成本；相同网络宽度只控制网络结构，不自动保证完整包体积或墙钟相同。单项机制实验不比较大量新模型。

### 数据集与事件单位

TabReD表明时间拆分可改变相对排名；CIC近期审计类成果进一步要求把流提取、标签和输入构造写清。本项目现有CIC联合组把会话关联与精确特征重复连起来：既不是单一会话，也没有自动获得“独立场景”语义。主结果仍按原始流行计数，同时报告族、完整输入模式和联合组的错误集中度。去掉主导组后的结果只作敏感性分析，不替换主成绩。

已知CIC的2,272条同模式漏报属于已见开发现象。任何针对其设计的修正都须在整个C和其他族上计分；不能添加该hash例外规则，也不能把该模式恢复本身当泛化证据。当前F/R/C分组维持不变，旧V/test不承担新的选择。若未来取得独立来源，需先核字段方向、单位、提取器与标签，随后冻结校准/确认角色。

## 3. 低误报约束与验收应分开

本轮的$\tau=\max(\tau_{R,0},\tau_{C,0})$保证的是有限R/C正常池的经验事件预算。C又用于checkpoint选择，因此C FPR≤1%只是开发工作点，不是总体误报证书。没有新增可交换性、独立性或跨时间稳定性证据时，不引入带总体保证措辞的新guard。

开发阶段保留既有1%工作点与标准化pAUC，主比较预先固定，逐行报告修复FN/新增FN、修复FP/新增FP。下一阶段是否值得继续由冻结的实际效应门槛决定；最终部署达标另在未用于开发的正常来源登记阈值，再在独立确认来源计算FPR和逐族TPR。新独立来源尚未取得，不能把本轮多seed称作完成此阶段。

## 4. 资源口径

原screen在固定首行上重复计时，适合流程开销排查，不代表混合输入的请求p95。下一次正式成本测量使用冻结模型、同设备/线程、共同固定的混合C开发样本清单，随机交错模型执行顺序并保存逐次时长，分别报告单行和batch256；不为测量重训模型。完整包需包括变换器与元数据，模型包不等于整个Python运行环境。本轮只测包字节和CPU管线时间；峰值进程内存另需正式测量，不从包大小推算。

本轮尚不编造目标笔记本的绝对毫秒/内存上限。先以同机的全字段small hard MLP和相同输入small hard MLP作为成本参照，采用预先声明的相对研究门槛；之后再按真实部署条件验收。
