# 成熟产品、开源工程与评测生态审查（2026-11-03）

## 结论与比较边界

本轮覆盖 20 项：6 个网络/主机安全工程、2 个流特征工具、1 个在线异常检测原型、7 个机器学习与评测基础设施、4 个商业 NDR 产品。这里只核官方公开文档和可读源码，没有安装、登录客户环境、运行样本或复现其性能。产品组合、输入层次、检测目标和资源口径不同，不能把厂商检测率、论文分数、链路吞吐量与 TraceLight 的离线二分类或 CPU 模型推理数字放在同一榜单。

成熟成果已经解决了大量需求：采集和协议解析、会话日志、规则告警、历史检索、告警到原始证据的跳转、在线学习评测、自动调参、实验追踪。把这些功能接到一个网页，不构成新的检测算法。TraceLight 可争取的研究贡献应聚焦于：**在压缩与可部署成本约束下，低误报工作点和困难攻击族的识别能否得到可重复保护，改善来自什么机制，失败边界在哪里。** 当前证据尚不足以声称该问题已经解决。

下文中“CPU 可运行”仅说明该组件的执行路径或部署形态，不表示已在本机测得吞吐、低内存或端到端实时能力；“可读源码”也不表示已复现。

## 1. 网络与主机安全工程：已经成熟的能力

| 项目 | 实际输入、检测类型 | 在线/文件、CPU 与证据界面 | 对 TraceLight 的借鉴及不可比之处 |
| --- | --- | --- | --- |
| **Suricata** | 包、流及应用层协议；签名规则、协议/解码异常、可配置 IDS/IPS。这里的 anomaly 不能直接译成学习式异常模型。 | 网卡流量和 PCAP；CPU 多线程引擎；EVE JSON 包含告警、流、协议、文件等事件，通常外接检索界面。[稳定版配置](https://github.com/OISF/suricata/blob/suricata-8.0.8/suricata.yaml.in) | 借鉴事件结构、flow_id、规则/模型来源、丢包与引擎状态。规则命中、阻断 verdict 与监督分类真阳性并非同一概念。[EVE 文档](https://docs.suricata.io/en/latest/output/eve/eve-json-output.html) |
| **Zeek** | 被动包与协议分析，脚本产生连接、DNS、HTTP 等日志及 Notice；基础组件不是一个通用监督分类器。 | 实时接口与离线 PCAP；CPU；结构化日志和脚本生态，核心自身不是完整 SOC 看板。 | 借鉴稳定会话 UID、方向与状态语义。`duration`、字节统计不保证等于 CIC/UNSW 同名字段；不能只重命名列后直接喂模型。[固定版 conn 源码](https://github.com/zeek/zeek/blob/v9.0.1/scripts/base/protocols/conn/main.zeek) |
| **Security Onion** | 组合网络、终端、日志数据和多个检测/检索组件；是 SOC 平台而非单一算法。 | 分布式采集、搜索和存储；服务器 CPU/RAM/磁盘成本；Alerts、Hunt、PCAP、Cases 构成调查流程。 | 借鉴告警筛选、详情、确认/升级、证据跳转；不需照搬整套平台。已读 2.4 Alerts 文档，不能据此保证新 3.x 的接口兼容；部分 AI/Guided Analysis 能力有版本/Pro 条件。[功能文档](https://docs.securityonion.net/en/2.4/alerts.html) |
| **Wazuh** | 代理/日志、解码器与规则关联；还覆盖主机文件完整性等终端需求，不是直接消费任意流 CSV 的同任务分类器。 | 持续代理/日志输入，manager、indexer、dashboard；规则与事件可追溯。 | 借鉴规则 ID、来源、窗口和关联作用域。流表没有终端证据时不能编造主机入侵链；规则频次统计与模型 FPR 不能混写。[规则语法](https://documentation.wazuh.com/current/user-manual/ruleset/ruleset-xml-syntax/rules.html) |
| **Arkime** | 包捕获、会话元数据索引及检索；核心价值是取证，不是统一的学习式攻击识别。 | 接口/PCAP；捕获、索引和存储使用服务器资源；Viewer/API 可从会话取回 PCAP、重组协议内容。 | 借鉴会话到原始证据的联系、保留期和“已过期/不可用”状态。只有 CSV 行号与字段时，不能称为包级溯源。[架构](https://arkime.com/architecture)、[会话 PCAP API](https://arkime.com/apiv3) |
| **ntopng** | 镜像包或收集器提供的流、主机与应用遥测；流量分析、行为检查、告警及 IDS 集成，不等于固定标签监督模型。 | CPU 网络探针；实时和历史 Web UI、REST；包接口与 ZMQ 接口能力不同，部分采集依赖 nProbe 等组件。 | 借鉴实时/历史区分、主机/应用筛选、数据源类型。Community 与 Pro/Enterprise 功能不能混用，更不能套用其链路吞吐量。[接口文档](https://www.ntop.org/guides/ntopng/advanced_features/interfaces/index.html)、[版本许可](https://www.ntop.org/guides/ntopng/versions_and_licensing.html) |

## 2. 流特征、在线学习与研究评测：可复用的基础设施

| 项目 | 实际输入、算法角色 | 在线/文件、CPU 与证据界面 | 对 TraceLight 的借鉴及不可比之处 |
| --- | --- | --- | --- |
| **NFStream** | PCAP、PCAP 列表、接口 → 双向流；DPI、统计和插件框架。插件可以挂 ML，但提取器本身不是已训练的万能 IDS。 | 在线/离线 CPU meter，插件生命周期、DataFrame/CSV 导出、轮转/匿名化；无完整 SOC UI。 | 最适合考察的采集适配候选，但要核字段层次、超时、单位和何时可用。官方已有 ML train/deploy 示例，所以“流上接模型”不是创新。[API/插件文档](https://www.nfstream.org/docs/api) |
| **CICFlowMeter（原作者仓）** | PCAP → 双向流统计表；首包定义方向，TCP/UDP 流终止规则和超时参与定义。不是检测模型。 | 文件提取/Java 工具链、CPU；输出特征，未提供可与本项目直接互换的分类器。 | 固定源码、配置、方向、单位和修复版本再谈新流量兼容；不要把 Python fork、作者旧仓、修正 CIC 数据的生成器视作同一版本。[作者说明](https://github.com/ahlashkari/CICFlowMeter/blob/98a5ebad0df579cc8b43eedd3421b3ae87699901/ReadMe.txt) |
| **Kitsune / KitNET** | 包经 AfterImage 统计后进入在线自编码器集合，输出重构异常分数；无监督，存在特征映射/训练宽限阶段。 | 原型支持包序列处理、CPU 路径；输出分数及示例图，不是成熟多用户取证平台。 | 借鉴预热/训练/执行状态和有界集合；不得把无监督分数称为校准攻击概率，也不能由原论文推断当前数据上的低 FPR。[README](https://github.com/ymirsky/Kitsune-py)、[KitNET 实现](https://github.com/ymirsky/Kitsune-py/blob/7c8ea51e89268d9a20399dca4bf32d663e638eb4/KitNET/KitNET.py) |
| **River** | 逐条字典/特征和可延迟标签；增量监督、异常与漂移等算法/评测库，不解析网络包。 | CPU 增量模型；progressive validation、逐步指标与资源测量，无现成 SOC UI。 | 借鉴先预测后学习、标签延迟、时间与内存记账。随机打乱离线行的实验不能命名为概念漂移验证。[评测 API](https://riverml.xyz/latest/api/evaluate/progressive-val-score/)、[固定版实现](https://github.com/online-ml/river/blob/0.26.1/river/evaluate/progressive_validation.py) |
| **CapyMOA** | Python 数据流接口与 MOA 后端；分类、回归、异常、漂移评测及相关学习器，不是网络流采集器。 | prequential 等流式评测，CPU/MOA 为重要路径，也有需另核依赖的深度模型；指标/绘图接口。 | 适合作为后续真实时间流研究的评测基础，当前不能只调用库便声称新漂移算法；0.x API 与 Java/MOA 依赖需固定。[评测模块](https://capymoa.org/api/modules/capymoa.evaluation.html)、[固定版 README](https://github.com/adaptive-machine-learning/CapyMOA/blob/v0.15.1/README.md) |
| **PyOD** | 数值特征矩阵上的异常模型与统一 score/predict API；当前还提供生命周期/agent 等接口，不能只描述成早期模型列表。 | 传统算法多有 CPU 路径，深度模型依赖另核；统一分数、预测、置信/拒识等接口不等于包级证据。 | 可选少数异常基线，但训练污染率阈值不等于独立正常集 FPR 校准。统一输入/标签可用性和调参预算后才可比较。[当前文档](https://pyod.readthedocs.io/en/latest/)、[固定版 BaseDetector](https://github.com/yzhao062/pyod/blob/v3.6.7/pyod/models/base.py) |
| **ADBench** | 异常检测 benchmark，区分无监督/半监督/监督、标签可用量和若干鲁棒性情景；不是一个网络安全产品。 | 离线矩阵、CPU/GPU 取决于模型；固定 seed、运行结果表和脚本。 | 借鉴负对照、污染与标签预算；默认数据生成可能改变采样/复制样本，不能直接套到本项目冻结分区。[README](https://github.com/Minqi824/ADBench/blob/3dac8221081e190f157d78e93bfa8867f90d0965/README.md)、[run.py](https://github.com/Minqi824/ADBench/blob/3dac8221081e190f157d78e93bfa8867f90d0965/adbench/run.py) |
| **AutoGluon Tabular** | 带标签表格，监督模型选择、调参、bagging/stacking、可选蒸馏/部署处理；不负责包采集与协议语义。 | 离线训练、CPU/GPU 视模型，`predict_proba`/leaderboard；不是 SOC 界面。 | 借鉴合理强基线、时间限制、推理预算。自动集成/重训可能显著增大成本或合并分区，不能无记录地与小 MLP 比较。[fit API](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html)、[部署文档](https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-deployment.html) |
| **TabArena** | NeurIPS 2025 Datasets and Benchmarks Track 的表格评测生态：固定任务、实现、配置与调参/集成结果；不是单个检测模型。 | 离线任务/缓存预测、评测结果；资源随模型与搜索预算变化，无网络流量分析 UI。 | 借鉴版本化协议与共同预算。其 IID 表格排名不能代替 UNSW/CIC 的原始字段约束、困难族、低 FPR 与部署测量。[官方仓](https://github.com/autogluon/tabarena/tree/v0.1.0)、[正式论文](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf) |
| **MLflow** | 代码版本、参数、指标、数据/模型和任意实验产物的跟踪、注册与比较；不改进分类决策本身。 | CPU/服务端追踪，UI/API、runs/artifacts/registry；可与本地训练分离。 | 实验追踪、模型来源展示已有成熟工具。哈希/registry 不能自动证明无泄漏或 test 未被调参；本项目仍需显式 split、阈值来源和确认状态。[Tracking 文档](https://mlflow.org/docs/latest/ml/tracking/) |

## 3. 商业 NDR：公开输入和调查流程已相当完整

以下只核厂商公开技术能力，不采信营销数字作为可复现结果；没有采购/登录/验证模型、告警质量或 CPU 性能。商业产品没有在本轮核到可固定复现的完整内部版本、训练数据、权重和阈值协议，因此不标“已复现”“低误报已证实”或“全面胜过/不如本项目”。公开部署模块的源代码许可，也不代表产品本身免费或开放。

| 产品 | 官方可核的输入/检测与部署 | 可追溯证据/界面及明确限制 | 对本项目的启示 |
| --- | --- | --- | --- |
| **Darktrace / HYBRID NETWORK**（原 `/NETWORK` 路径重定向） | 官网描述从本环境学习行为画像。作者 GCP vSensor 部署仓具体说明 VPC 镜像/osSensor 输入、DPI 提元数据、TLS 上送已有 master、可选 GCS PCAP；需要 Update Key/Push Token。是商业环境学习/响应体系，不是本项目那种公开监督 MLP。[官方产品页](https://www.darktrace.com/products/hybrid-network)、[部署源码说明](https://github.com/darktrace/terraform-gcp-vsensor/blob/main/README.md) | 可核 master 控制台、传感器与 PCAP 管线；公开材料不足以重建其算法/内部校准。虚机 CPU/RAM 和采集带宽有成本，不可称任意笔记本实时能力。 | 借鉴“采集覆盖/传感器健康/证据存储”分离；“自学习、检测未知攻击、AI 调查”不能作为我们的新颖性口号。部署脚本公开不等于取得产品许可。 |
| **Vectra AI Detect for NDR** | Sensor 接收 SPAN/TAP/镜像，提元数据；Brain 分析与关联，物理/虚拟/云部署。[官方技术规格](https://content.vectra.ai/hubfs/downloadable-assets/Datasheet_Appliance-SensorSpecs.pdf)、[网络流量建议](https://docs.vectra.ai/deployment/traffic-engineering-and-validation/network-traffic-recommendations) | 技术文档明确 detection PCAP 来自滚动缓冲且可能不完整/缺失；日志类账户/云检测不必有包。控制台不能把缺证据伪装成完整流量回放。[Detection PCAP 文档](https://docs.vectra.ai/reference/how-detection-pcaps-are-generated) | 借鉴证据不可用状态、数据源类型和检测延迟的影响；机器学习/关联为厂商公开描述，内部模型和实际 FPR 未核。商业订阅形态，合同许可未审查。 |
| **ExtraHop RevealX** | 原始包 → 协议/事务/指标；NDR 行为检测，并可配 IDS 签名和 Packet Forensics。公开 FAQ 区分 sensor、packetstore、recordstore、console 及 SaaS/本地部署。[官方 FAQ](https://www.extrahop.com/products/faq)、[架构指南](https://docs.extrahop.com/9.4/intro-to-eh-system/intro-to-eh-system.pdf) | 具备检测调查、包查询、包存储/取回；模块和访问权限有边界。已读 9.4 架构文档，不把它写成截至本日最新软件版本或现租户兼容保证。 | 借鉴原始包、事务记录、告警三层证据。NDR/IDS/取证的组合不是单一 ML 评分，也不能拿厂商专用设备吞吐对比 CSV 推理。商业交付，合同许可未审查。 |
| **Corelight Open NDR** | Sensors 将 Zeek、Suricata、威胁情报及厂商本地 ML/异常能力组合；硬件、软件、虚拟、云与 flow-log sensor 各有适用输入。[传感器说明](https://corelight.com/platform/sensors) | Smart PCAP 可按流量/事件选择捕获，并在 Zeek 日志用 `spcap.url` 链到相关包；为订阅模块。[Smart PCAP](https://corelight.com/platform/smart-pcap) | 借鉴日志到包的稳定联系和选择性保留。Open NDR 名称不代表整个产品 OSI 开源；上游开源组件与厂商产品许可分开。[厂商 EULA](https://corelight.com/hubfs/legal/eula.pdf) |

这四家已覆盖“环境行为模型/混合检测 → 排序告警 → 调查证据 → 平台集成”的产品需求。本项目不宜以“首次把 AI 用于 NDR”“自动生成攻击链”“替代企业 SOC”定位。有限资源下，更有价值的是对一个受控问题给出厂商材料无法替代的透明实验、可否定结论和部署成本证据。

## 4. 版本、来源与许可核对

日期为公开 release 的发布日期，统一按截至 **2026-11-03** 可见且已发布的记录；没有正式 release 的项目固定 commit。许可只描述所列主仓，依赖、打包组件、数据、规则/模型资产可能另有条款。GitHub API 的 `NOASSERTION` 不当作“无许可”；已对相应原文另核。此表是工程来源核验，不是法律意见或许可兼容审计。

| 项目 | 本轮固定的公开版本 / 日期 | 主仓许可与直接依据 |
| --- | --- | --- |
| Suricata | [suricata-8.0.8](https://github.com/OISF/suricata/releases/tag/suricata-8.0.8)，2026-10-29 | [GPL-2.0](https://github.com/OISF/suricata/blob/suricata-8.0.8/LICENSE)；规则集另核。 |
| Zeek | [v9.0.1](https://github.com/zeek/zeek/releases/tag/v9.0.1)，2026-10-26 | [BSD 三条款式 COPYING](https://github.com/zeek/zeek/blob/v9.0.1/COPYING)，文件自有版权/依赖另核。 |
| Security Onion | [3.4.0-20261102](https://github.com/Security-Onion-Solutions/securityonion/releases/tag/3.4.0-20261102)，2026-11-02 | 主仓 [Elastic License 2.0](https://github.com/Security-Onion-Solutions/securityonion/blob/3.4.0-20261102/LICENSE)，不能笼统称全栈 GPL/OSI 开源。 |
| Wazuh | [v4.14.9](https://github.com/wazuh/wazuh/releases/tag/v4.14.9)，2026-10-15 | [GPLv2 主许可及所列例外](https://github.com/wazuh/wazuh/blob/v4.14.9/LICENSE)，组件另核。 |
| Arkime | [v6.9.1](https://github.com/arkime/arkime/releases/tag/v6.9.1)，2026-10-22 | [Apache-2.0](https://github.com/arkime/arkime/blob/v6.9.1/LICENSE)。 |
| ntopng | [7.0](https://github.com/ntop/ntopng/releases/tag/7.0)，2026-09-22 | Community [GPLv3](https://github.com/ntop/ntopng/blob/7.0/README.md)；Pro/Enterprise [额外商业许可](https://www.ntop.org/guides/ntopng/versions_and_licensing.html)。 |
| NFStream | [v6.6.0](https://github.com/nfstream/nfstream/releases/tag/v6.6.0)，2026-02-15 | [LGPL-3.0](https://github.com/nfstream/nfstream/blob/v6.6.0/LICENSE)。 |
| CICFlowMeter | 无正式 release；[98a5ebad0df5](https://github.com/ahlashkari/CICFlowMeter/commit/98a5ebad0df579cc8b43eedd3421b3ae87699901)，2022-01-05 | 此作者仓的 [LICENSE.txt](https://github.com/ahlashkari/CICFlowMeter/blob/98a5ebad0df579cc8b43eedd3421b3ae87699901/LICENSE.txt) 是 MIT 式授权文本；不能从其他 fork 推断 GPL。 |
| Kitsune-py | 无正式 release；[7c8ea51e8926](https://github.com/ymirsky/Kitsune-py/commit/7c8ea51e89268d9a20399dca4bf32d663e638eb4)，2026-03-24 | [MIT](https://github.com/ymirsky/Kitsune-py/blob/7c8ea51e89268d9a20399dca4bf32d663e638eb4/LICENSE)。 |
| River | [0.26.1](https://github.com/online-ml/river/releases/tag/0.26.1)，2026-08-21 | [BSD-3-Clause](https://github.com/online-ml/river/blob/0.26.1/LICENSE)。 |
| CapyMOA | [v0.15.1](https://github.com/adaptive-machine-learning/CapyMOA/releases/tag/v0.15.1)，2026-09-25 | Python 主仓 [BSD-3-Clause](https://github.com/adaptive-machine-learning/CapyMOA/blob/v0.15.1/LICENSE)；MOA/Java 及其他依赖另核。 |
| PyOD | [v3.6.7](https://github.com/yzhao062/pyod/releases/tag/v3.6.7)，2026-10-12 | [BSD-2-Clause](https://github.com/yzhao062/pyod/blob/v3.6.7/LICENSE)。 |
| ADBench | 无正式 release；[3dac8221081e](https://github.com/Minqi824/ADBench/commit/3dac8221081e190f157d78e93bfa8867f90d0965)，2026-01-08 | [BSD-2-Clause](https://github.com/Minqi824/ADBench/blob/3dac8221081e190f157d78e93bfa8867f90d0965/LICENSE)；数据与调用模型另核。 |
| AutoGluon | [v1.6.3](https://github.com/autogluon/autogluon/releases/tag/v1.6.3)，2026-09-18 | [Apache-2.0](https://github.com/autogluon/autogluon/blob/v1.6.3/LICENSE)。 |
| TabArena | [v0.1.0](https://github.com/autogluon/tabarena/releases/tag/v0.1.0)，2026-09-03 | [Apache-2.0](https://github.com/autogluon/tabarena/blob/v0.1.0/LICENSE)；底层任务数据许可分开。 |
| MLflow | [PyPI 3.17.2](https://pypi.org/project/mlflow/3.17.2/)，2026-10-21 | [Apache-2.0](https://github.com/mlflow/mlflow/blob/v3.17.2/LICENSE.txt)。仓库 `model-catalog/latest` 等 tag 不等于 Python 包版本。 |
| Darktrace / HYBRID NETWORK | 2026-11-03 官方产品页和 GCP 部署仓快照；未核完整产品 build | 商业产品；[公开脚本要求已有产品 Update Key/Push Token](https://github.com/darktrace/terraform-gcp-vsensor/blob/main/README.md)，脚本许可不覆盖产品，合同未审查。 |
| Vectra Detect for NDR | 2026-11-03 官方技术规格和文档快照；未核完整产品 build | [规格说明商业订阅的软件更新](https://content.vectra.ai/hubfs/downloadable-assets/Datasheet_Appliance-SensorSpecs.pdf)；合同未审查。 |
| ExtraHop RevealX | 2026-11-03 官方 FAQ/产品页；架构细节参考 9.4 文档 | 商业模块交付；[官方 FAQ](https://www.extrahop.com/products/faq) 明确基础模块/附加模块，合同未审查。 |
| Corelight | 2026-11-03 Sensors/Smart PCAP 页面快照；未核完整产品 build | [商业 EULA](https://corelight.com/hubfs/legal/eula.pdf)；上游 Zeek/Suricata 等保持各自许可。 |

滚动文档与固定 release 必须分开：Suricata `latest` 文档在本轮为 9.0.0-dev，ntopng 文档头为 7.1.261018，AutoGluon stable 文档头为 1.6.1；本表分别固定 8.0.8、7.0、1.6.3 的发布记录。Security Onion 新发布身份已核，但交互流程来自明确标注的 2.4 文档。若实施集成，须再按目标版本验证接口，不能凭这份阅读报告跳过测试。

## 5. 超出首页宣传的具体核验

以下核验直接影响设计选择，均为文档/源码阅读证据，未执行相应项目。

1. **Suricata：关联键与判定层分开。** EVE 支持 `flow_id`、可选 Community ID 与抓包文件信息，alert action 与引擎最终 verdict 有不同语义。可借鉴为“记录、规则命中、模型分数、阈值、处置”分别存字段；不能只用红色状态块替代证据。[EVE 定义](https://docs.suricata.io/en/latest/output/eve/eve-json-output.html)、[8.0.8 配置](https://github.com/OISF/suricata/blob/suricata-8.0.8/suricata.yaml.in)。
2. **Zeek：字段相同不等于定义相同。** `Conn::Info` 注释指出 duration 不计部分结束处无有效负载的 TCP 包，例如最终 ACK；字节还受序列号等语义影响。把此 duration 或 bytes 直接改名为 CIC 列存在训练/部署不一致风险。[9.0.1 源码 `Info`](https://github.com/zeek/zeek/blob/v9.0.1/scripts/base/protocols/conn/main.zeek)。
3. **Wazuh：关联窗口有范围。** `frequency`、`timeframe`、`same_srcip`、`same_field` 定义具体计数规则；`global_frequency` 文档明确是 manager 范围，不是整个 cluster。若做流告警聚合，也应记录窗口、实体键和作用域，不能把聚合数当独立检测数。[规则语法](https://documentation.wazuh.com/current/user-manual/ruleset/ruleset-xml-syntax/rules.html)。
4. **Arkime：元数据存在不保证包仍在。** API 可按会话 ID/查询导出 PCAP，源码在 SPI 仍存在但 PCAP 文件不可用时返回相应错误，并按原始 packet position 读取/重组。界面必须表现保留期和缺证据状态。[API](https://arkime.com/apiv3)、[6.9.1 `apiSessions.js`](https://github.com/arkime/arkime/blob/v6.9.1/viewer/apiSessions.js)。
5. **NFStream 与 CICFlowMeter：采集契约必须审计。** NFStream 明列 active/idle timeout、记账层次、毫秒时间字段、插件 `on_init/on_update/on_expire` 和早期/流结束统计；CICFlowMeter 说明 duration 为微秒、首包定方向。流结束才能得到的字段不能宣传首包检测；NFStream `application_confidence` 是 DPI 判别语义，不是攻击置信度。[NFStream API](https://www.nfstream.org/docs/api)、[CIC 作者说明](https://github.com/ahlashkari/CICFlowMeter/blob/98a5ebad0df579cc8b43eedd3421b3ae87699901/ReadMe.txt)。
6. **River：在线验证必须约束时序。** `progressive_val_score` 实现先预测，再在相应标签可用时学习；支持 `moment/delay` 和时间/内存测量。没有时间、标签延迟和更新策略的独立定义，仅增加一个流式循环不会形成可信在线学习研究。[固定版实现](https://github.com/online-ml/river/blob/0.26.1/river/evaluate/progressive_validation.py)。
7. **PyOD：异常分数与 FPR 校准是两层。** `BaseDetector` 约定较大 score 更异常，`contamination` 用训练分数的分位数产生阈值；它表达训练异常比例假设，不能替代本项目在独立正常校准样本上的误报预算。当前文档还已有更多生命周期/置信与组合接口，不能把这些普通接口包装为首创。[固定版 BaseDetector](https://github.com/yzhao062/pyod/blob/v3.6.7/pyod/models/base.py)。
8. **ADBench：跑通脚本可能改变任务。** `RunPipeline` 区分异常标签的比例/数量，默认含 `generate_duplicates=True`、小样本阈值 1000 等数据生成选项。本项目不应原样调用其默认数据生成后，把复制样本当新独立样本或与既有完整族支持结果比较。[固定版 `run.py`](https://github.com/Minqi824/ADBench/blob/3dac8221081e190f157d78e93bfa8867f90d0965/adbench/run.py)。
9. **Vectra：商业界面也明确承认证据不完整。** Detection PCAP 由滚动缓冲关联，不保证完整流；检测迟、负载高或数据源为日志时可缺失。我们应同样把“不可用/未采集/已过期/不适用”作为明确状态。[官方 PCAP 文档](https://docs.vectra.ai/reference/how-detection-pcaps-are-generated)。
10. **Corelight：可借鉴的是联系，不是营销吞吐。** Smart PCAP 文档具体给出日志里的 `spcap.url`、按 Zeek/Suricata 事件选择包和订阅模块边界；这说明“告警可追溯到包”已有成熟工程实现，CSV 字段展示不能等价替代。[Smart PCAP](https://corelight.com/platform/smart-pcap)。

## 6. 与当前 TraceLight 的真实差距

这里区分已有代码与应做规划，避免把现有成果抹掉，也避免把源码存在写成已完成产品验收。

| 维度 | 当前代码可核状态 | 成熟生态已解决什么 / 我们的边界 |
| --- | --- | --- |
| 新输入推理 | [`cli.py` 117–147](../../src/ids/cli.py) 有 CPU `predict`，输出 score/阈值/来源/run_id；[`demo.py` 145–191](../../workflows/stage5/demo.py) 可对严格 schema 的新 CSV 跑两个模型。 | 已有真实推理，不能说只有静态图。CSV 必须是已提取且语义兼容的特征表；这不证明从任意 PCAP/网卡采集到检测已打通。 |
| 界面与结果证据 | [`demo.html`](../../workflows/stage5/demo.html) 有联合判定四象限、样本/家族筛选、双分数/阈值、输入字段和来源 SHA；通过 CLI 生成独立 HTML，并非网页上传/后台任务服务。 | 作为离线模型比较界面已有用处；成熟平台还有会话搜索、时间线、证据跳转和案件状态。不能为表格中没有的时间/IP/包内容制造“真实网络现场”。 |
| 模型与数据来源 | demo 固定第四轮封存模型、seed 17 和 validation 阈值；新 CSV 记录 input SHA。 | 需要清楚显示版本与开发/最终确认状态；旧 demo 不自动等于研究 v2 或独立新 test 的结果。哈希证明字节身份，不能单独证明标签正确和无泄漏。 |
| 实时性与成本 | 有模型 CPU benchmark 和离线 CPU 推理路径。 | 采集、解析、流等待、排队、模型、导出/界面是不同耗时；现有模型延迟不能换算为可承载多少 Gbps。完整流特征还可能存在等待流结束的时延。 |
| 检测有效性 | 已有封存实验与机制研究代码，不在本报告重报任何结果。 | 研究贡献取决于独立校准/确认、困难族支持、低 FPR 下可重复收益和资源代价；Suricata 规则、商业 NDR、IID 表格榜单均不能代替同协议实验。 |
| 在线更新与漂移 | 本轮未核到已验收的真实时间流标签反馈、在线更新/回退流程。 | River/CapyMOA 提供基础工具；若没有新时序问题和可审计数据，暂不扩大为“持续学习 NDR”。 |

## 7. 应借鉴的顺序与不应扩大的范围

**P0：先补研究成立所需的契约与比较。** 明确模型输入的原始字段定义、单位、方向、流终止/超时、缺失语义和特征可用时刻；用现有代码与数据证据逐项核，而不是先挑一个提取器接上。参考 TabArena/ADBench 的预算与标签条件管理，把研究 v2 的机制对照、共同训练/选择预算、失败结果和成本完整呈现。真实 v2 训练按实验室 4090D 路线进行；本机继续代码、单测和 synthetic 验证。旧 test 已开，不能把据此优化的新模型再报为独立确认，见 [STAGE6_PROTOCOL](../STAGE6_PROTOCOL.md)。

**P1：完善已有离线系统的可信使用闭环。** 优先补输入不兼容提示、数据/模型/阈值版本、开发/确认状态、完整总量与抽样展示的区别、运行失败与证据缺失状态；用真实模型输出支持比较。此类交付属于可复现性和工程贡献，不能写成新的分类算法。若要宣称端到端可部署，再定义并测量包含读取/解析/预处理/推理的独立成本口径。

**P2：有需求与数据后，只验证一条采集路线。** NFStream 的生命周期契约较清晰，可先做字段可生成性审查；若现有 CIC/UNSW 特征无法忠实生成，就缩小支持范围或重新定义并独立验证任务，不能填零/重命名强行兼容。随后才考虑一条 PCAP → 特征 → 固定模型 → 事件 → 可检查证据的最小闭环。这里的价值是部署验证，不应冒充特征提取或 NDR 产品层面的发明。

**暂不做：** 不安装全部工具、不重建 Security Onion/商业 NDR、不增加没有数据支持的攻击链/拓扑、不自动阻断真实网络、不为“前沿”堆叠在线学习或 agent、不用更多 run 数掩盖研究假设不清。MLflow/完整 SOC 集成应由实际协作需求触发，当前 manifest/哈希/结果表能满足的要求不必立刻增加服务负担。

可提交的贡献边界：新的可检验损失/选择机制及其归因属于**候选方法贡献**；严格低 FPR 与困难族实验、负结果、资源权衡属于**实证贡献**；schema 审计、来源封存、CPU 推理与可信界面属于**工程贡献**。它们都可能有价值，但标题、摘要和答辩应分别说明，不能互相充当证据。

## 核查记录

- 核查日期：**2026-11-03（Asia/Shanghai）**；只使用项目官方文档、作者/维护方仓库、公开发布页/PyPI 和官方论文入口。
- 权限状态：未登录任何商业客户控制台、未申请或使用产品试用权限、未接触客户数据；公开文档不可访问的页面没有当作已验证内容。
- 执行状态：没有安装上述项目、没有抓包/发包、没有训练或基准实测，没有复现论文或商业效果；代码阅读与发布身份核对均不等于运行复现。
- 商业产品：只确认公开能力与部署输入，内部版本、数据、模型、校准和性能不可独立验证；没有比较价格，也没有采纳“零误报/零日领先/专用设备速率”等营销数字。
- 本文仅新增本文件，不更改实验配置、源码、历史结果或封存包。
