# IDS、轻量化、蒸馏与流量预训练：研究定位审查

核查截止：2026-11-03。范围：学术网络入侵检测、特征压缩、知识蒸馏、包级流量表示与严格评估；通用表格基础模型和商业产品由同目录其他报告覆盖。本文是有边界的代表性审查，不声称穷尽所有论文或获得统一排行榜。

**执行状态：本次仅阅读论文、官方会议/期刊资料、作者代码与模型卡；下面 18 项均未由本次审查训练、微调、跑分或部署。代码/权重/数据页面可访问不等于已下载、跑通或复现。** 本文只新增此 Markdown，不改模型或实验协议。

## 1. 定位判断

当前“特征筛选＋树教师/小 MLP 蒸馏＋低成本部署”的大框架，已经被特征选择、选择性蒸馏和轻量 IDS 文献广泛覆盖。加上一个置信 gate、hard-negative 项或前端界面，尚不足以证明新的算法贡献。值得继续研究的是：**在明确的低误报预算下，压缩为何伤害某些攻击/正常组，教师信号和有效类权重各解释多少；针对这些原因的设计，能否在相同学生和总预算下带来可确认的检测—成本收益。** 这是一项待证明的具体问题，不是已确认的文献空白。

不要将“模型更大、更近”自动等同于更适合当前任务。以下工作至少涉及三种不同问题：包级应用/加密流量分类、流/主机级异常检测、已抽取 CSV 的监督二分类。输入信息、样本单位和目标不同，不能互相借用 accuracy/F1 排名。

本项目开发边界以 [STAGE6_RESEARCH_DESIGN.md](../STAGE6_RESEARCH_DESIGN.md) 为准：旧 final test 已被研究者看到，v2 仍处于开发研究；新的确认结论需要真正未用来源。后续真实训练在实验室 RTX 4090D 执行，本机只做工程/合成检查及工件分析。

## 2. 证据口径与输入契约

- **A：全文重点核对。** 已读原论文的方法、相关评估/成本段落，并给出节/图/表定位；不代表复现。
- **B：身份和指定事实已核。** 本次主要依据作者 README、模型卡、原论文摘要/出版页或指定公式；未完整审计 split/实现时明确留空。
- “未核到固定 1% 协议”指本次证据不足以支持同协议比较，不能写成作者一定没有测过。
- 表中 code/model/data 分别写清；没有核到公开入口时写“未核到”，不推断不存在。

| 输入 | 需要保留的信息 | 当前 42/82 字段 CSV 的边界 |
|---|---|---|
| Packet/PCAP | 原始 header/bytes、包长、方向、包时间、流内顺序 | 聚合统计不能反推出包序列；把一行转文本/图片不等于恢复这些信息 |
| Flow/host log 或图 | 流统计加真实端点、时间窗、通信关系 | 当前 UNSW 子集缺可重建真实会话/时间图的时间与 IP；不能把行号当时间 |
| Tabular CSV | 每行固定语义的数值/类别字段 | 适合当前监督学习、字段选择、表格自监督和蒸馏；跨 UNSW/CIC 同名列不自动同义 |

## 3. 18 项代表性成果

### 3.1 轻量检测、特征选择、蒸馏与在线学习

| ID／工作与正式身份 | 输入／任务／方法区别 | split 与指标证据；1% FPR 可比性 | 作者工件及本项目适配状态 | 阅读定位 |
|---|---|---|---|---|
| **R01 Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection，NDSS 2018 主会** | 原始包经 AfterImage 提取带时间衰减的上下文统计，再以多个小 AE 的重建误差检测异常；不是 hard-label MLP | 正常学习期→执行期；Fig.9 报 TPR/FNR@FPR=0.001、AUC/EER。低 FPR 与轻量部署早已有先例；不是当前独立 calibration→锁阈值 test 协议 | [原论文](https://arxiv.org/pdf/1802.09089)、[NDSS 官方版本](https://www.ndss-symposium.org/wp-content/uploads/2018/02/ndss2018_03A-3_Mirsky_paper.pdf)、[作者完整流水线](https://github.com/ymirsky/Kitsune-py)、[KitNET 核心](https://github.com/ymirsky/KitNET-py)。代码与样例可见；原生提取器不能接现有 CSV。可另做“KitNET 在相同 CSV 的适配对照”，须改名说明 | **A**：§IV、§V；Fig.9–11、Table V |
| **R02 Whisper / Realtime Robust Malicious Traffic Detection via Frequency Domain Analysis，CCS 2021 主会** | 把包级特征序列编码并分帧做 DFT，保留顺序模式，以低冗余频域特征检测；强调吞吐、规避鲁棒性 | 42 类攻击；报告 ROC/AUC 与系统吞吐。不是当前逐行 CSV 的同输入任务；本次未核到锁定 1% 的独立校准协议 | [全文](https://arxiv.org/pdf/2106.14707)、[作者代码](https://github.com/fuchuanpu/Whisper)。DPDK/10Gb NIC 原型，仓库已归档；原生需要包序列，不能用 CSV 行顺序替代 | **B**：全文 §3 系统、§5 评估；README Hardware/Usage |
| **R03 FSNID / Feature Selection for Network Intrusion Detection，KDD 2025 主会** | CSV/特征向量；用神经信息估计衡量去掉某字段后的条件信息损失，顺序剔除非信息字段，另有 LSTM/TCN/GRU 版本 | §5.3：有官方 train/test 则沿用，否则 80:20；多类分类，5 runs、95% 区间；报保留字段、accuracy、FPR、F1。**报告 FPR 不等于统一固定 1% 召回** | [作者全文](https://www.machineintelligencelab.ai/papers/kdd25.pdf)、[DOI](https://doi.org/10.1145/3690624.3709339)、[作者代码](https://github.com/c-s-westphal/FSNID)。有数据/选择器/评估入口；现有项目仅审查，未复现；需 train-only 重构与预算核查，不能 MI 冒名 | **A**：§4、§5.3–5.5；Fig.2–5；Appendix C–E、Table 1 |
| **R04 DistillGuard / Transformer-based knowledge distillation for explainable intrusion detection system，Computers & Security 154 (2025), 104417** | 网络特征、Transformer 教师→轻学生；Selective Gradient-Based KD 选择教师隐藏表示，辅以梯度解释。该选择不是本项目 OOF 输出置信 gate | 出版页和方法公式已核；原始时间窗构造、split、低 FPR 阈值和 CPU 全管线证据本次未完整审计，因此不直接引用其成绩作比较 | [出版页/原文入口](https://www.sciencedirect.com/science/article/pii/S0167404825001063)、[DOI](https://doi.org/10.1016/j.cose.2025.104417)。Data availability 为 on request；本次未核到作者 code/model 入口。已覆盖“选择性 KD＋轻量 IDS＋解释”的大概念 | **B**：§1.1、§3.6.2–3.6.3，Eq.33–34；结论/数据声明 |
| **R05 Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Pruning and Knowledge-Distilled Kronecker Networks，CommNet 2025，普通专业会议** | TON-IoT CSV；SHAP 筛字段、大教师、小型 Kronecker 学生、KD，可选量化。直接覆盖“筛特征＋蒸馏＋压缩”组合 | 本次核作者实现说明与正式引用，未精核分组隔离/阈值/量化是否真正执行；不能将 README optional quantization 当实测 | [作者代码](https://github.com/mohammed-jouhari/iot-ids-lightweight)、[DOI](https://doi.org/10.1109/CommNet68224.2025.11288887)。有 notebook 和 MIT 文件；TON-IoT 指向官方来源；本次未运行。不是顶会，仍属于必须回应的直接先例 | **B**：README Key Features/Dataset/Citation |
| **R06 FedKD-IDS: A robust intrusion detection system using knowledge distillation-based semi-supervised federated learning and anti-poisoning attack mechanism，Information Fusion 117 (2025), 102807** | N-BaIoT 统计特征；半监督联邦蒸馏、共享无标签 proxy、logits 投票及恶意协作者筛查，核心问题是 non-IID/投毒 | 出版摘要核到 non-IID 和 label-flipping 情景；未精核全文 split/1% FPR/CPU。不是普通集中式小学生 KD 的直接效率基线 | [原出版页](https://www.sciencedirect.com/science/article/pii/S1566253524005852)、[DOI](https://doi.org/10.1016/j.inffus.2024.102807)。正式卷期 2025，DOI 带 2024 不矛盾；本次未核到作者 code/model。没有联邦威胁模型时不为显新而迁移 | **B**：Abstract/Highlights/卷期元数据 |
| **R07 SAFE: Self-Supervised Anomaly Detection Framework for Intrusion Detection，AAAI 2025 AICS workshop，非 AAAI 主会** | CSV 经 PCA loading 选字段→DeepInsight/t-SNE 映射成 8×8 图→MAE→LOF；正常数据学习，异常检测 | 全文 60/20/20；train-only 选 31 字段；主要 precision/recall/F1。未给当前会话分组与固定 1% 同协议证据 | [全文](https://arxiv.org/html/2502.07119v1)、[作者代码](https://github.com/ElvinLit/SAFE)、[官方 workshop 议程](https://aics.site/AICS2025/AICSProgram2025.pdf)。README “AAAI-25”应按议程纠正。可适配 CSV，但 AE/LOF 及输入转换成本要全计，尚未执行 | **A**：Proposed Framework；Experimental Setup；Overhead Analysis/Table 6 |
| **R08 CARAVAN: Practical Online Learning of In-Network ML Models with Labeling Agents，OSDI 2024 主会** | 流特征；DNN/规则/LLM 作慢速标签源，规则缓存与 accuracy proxy 触发小模型在线更新；不是把 LLM 放实时转发路径 | CIC-IDS2017/UNSW-NB15 等窗口回放；主指标 F1、重训 CPU/GPU 成本；数据流/漂移构造应与真实时间分开。没有本项目固定 1% 证据 | [官方会议页](https://www.usenix.org/conference/osdi24/presentation/zhang-qizheng)、[全文](https://www.usenix.org/system/files/osdi24-zhang-qizheng.pdf)、[作者 artifact](https://github.com/Per-Packet-AI/Caravan-Artifact-OSDI24)。代码入口可见；GPT-4 历史 API 结果不可假定今日逐值复现；硬件端用 Taurus FPGA | **A**：§3–5，特别 §4.1、§5.1；Appendix A |
| **R09 NetVigil: Robust and Low-Cost Anomaly Detection for East-West Data Center Security，NSDI 2024 主会** | 真实端点/时间 flow logs→安全图特征、GNN 对比学习和时间平滑；检测数据中心横向异常 | Yatesbury＋长期生产流；AUC/TPR/FPR；§6.1 阈值最大化 TPR−FPR，**不是固定 1%**；有同机 latency 和运维成本 | [官方全文](https://www.usenix.org/system/files/nsdi24-hsieh.pdf)、[作者基准 Yatesbury](https://github.com/microsoft/Yatesbury)。本文可核公开基准入口，不推断私有生产流可下载。当前 UNSW CSV 缺建图元数据；不能合成假端点补齐 | **A**：§3–4、§6.1/6.5；Table 4、Fig.5 |

### 3.2 包级预训练、基础模型与评估审计

| ID／工作与正式身份 | 输入／任务／方法区别 | split 与指标证据；1% FPR 可比性 | 作者工件及本项目适配状态 | 阅读定位 |
|---|---|---|---|---|
| **R10 nPrint / New Directions in Automated Traffic Analysis，CCS 2021 主会** | 原始包统一位表示，配 AutoML 的 nPrintML；任务包括流量识别，不仅恶意二分类 | 多任务基准，不能跨任务 F1 与本项目排名；本次未精核每个任务的 split/1% 工作点 | [作者全文](https://pschmitt.net/docs/nprint.pdf)、[官方项目/代码与基准入口](https://nprint.github.io/)。表示、AutoML、metadata-PCAP 工具均有作者入口；需要 PCAP，不是现有聚合 CSV 的 drop-in | **B**：论文摘要/方法定位；官方 nPrint/nPrintML/pcapML 说明 |
| **R11 ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification，WWW 2022 主会** | PCAP datagram/burst 字节 token，BERT 上下文预训练后按任务微调；应用/VPN/Tor/TLS 等识别 | 多个加密分类任务，主要 F1；需结合 R15 的同流泄漏审计，不能直接沿用所有原划分或把 F1 当 IDS 低误报成绩 | [原论文](https://arxiv.org/abs/2202.06335)、[作者代码](https://github.com/linwhitehat/ET-BERT)。有 preprocessing/pretraining/fine-tuning/data/model 入口；本次没核全部下载链接。CSV 文本化不构成原生 ET-BERT 输入 | **B**：摘要；README Method/Data/Pretraining/Fine-tuning |
| **R12 YaTC / Yet Another Traffic Classifier: A Masked Autoencoder Based Traffic Transformer with Multi-Level Flow Representation，AAAI 2023 主会** | PCAP flow→Multi-Level Flow Representation 矩阵→MAE 预训练→分类；矩阵仍源于包信息 | 多个真实流量分类集；本次未完整核原划分/固定 1%/CPU；须回应 R15 对流隔离的实证审计 | [AAAI 正文](https://ojs.aaai.org/index.php/AAAI/article/download/25674/25446)、[作者代码](https://github.com/NSSL-SJTU/YaTC)。README 有 checkpoint/data 链接及 PCAP→MFR 脚本；没下载验证。不能把普通 CSV reshape 称 YaTC | **B**：论文摘要；README data_process/pretraining/model |
| **R13 TrafficFormer: An Efficient Pre-trained Model for Traffic Data，IEEE S&P 2025 主会** | PCAP→flow→burst→hex bigram/BPE；细粒度预训练与流量增强；微调涉及 datagram/length/time/direction/message_type | 作者发表页核到六分类集＋协议理解；本次未完整核全文 split/1%/CPU，不能以“efficient”标题推断 CPU 优势 | [作者发表页](https://liuzhuotao.github.io/publications/trafficformer/)、[正式 DOI](https://doi.org/10.1109/SP61157.2025.00102)、[作者代码](https://github.com/IDP-code/TrafficFormer)。有预处理/预训练/微调与模型外链；原始包输入已由 README 核实，未跑通 | **B**：作者发表页；README Pretrain Corpus/Finetuning Data Generation |
| **R14 tFusion / Training with Only 1.0 ‰ Samples: Malicious Traffic Detection via Cross-Modality Feature Fusion，CCS 2025 主会** | 包长/时间序列＋flow/host 统计；跨模态 attention，拓扑对比预训练，再少样本训练轻量下游 | 11 既有数据集回放＋5 天机构部署；§5/Appendix 用 accuracy/AUC/F1 等，不等于统一 1% 预算；attention **训练和执行均在 V100**，不能只数下游轻模型 | [作者全文](https://github.com/fuchuanpu/TFusion/blob/main/tFusion_LongVersion.pdf)、[正式 DOI](https://doi.org/10.1145/3719027.3765143)、[作者代码](https://github.com/fuchuanpu/TFusion)。README 仍有匿名/HotCRP 数据指引；代码有公开入口，完整数据可达性未确认。UNSW CSV 缺所需 packet/time/host 信息 | **A**：§3–4、§5.1/5.5、Fig.2/14/15；原文 pp.4–8/11 |
| **R15 Pcap-Encoder / The Sweet Danger of Sugar: Debunking Representation Learning for Encrypted Traffic Classification，SIGCOMM 2025 主会** | T5 原始包 header/hex 编码，先自编码再协议字段问答；同时审计多个 NFM 和浅层基线 | 原 per-packet vs 正确 per-flow 划分、冻结/解冻 encoder、字段移除；应用/网站识别，非同协议 IDS。论文显示某些高分来自捷径，不证明所有预训练模型普遍无用 | [作者机构全文](https://iris.polito.it/retrieve/handle/11583/3003539/21f8a7c5-c796-4cc1-ab4d-3c6cea41412e/3718958.3750498.pdf)、[DOI](https://doi.org/10.1145/3718958.3750498)、[作者代码](https://github.com/SmartData-Polito/Debunk_Traffic_Representation)。预训练含 **UNSW-NB15 原始包**，现成权重在 UNSW 上不能未经重叠审计称独立教师 | **A**：§3.4、评估设计/结果与字段消融；Fig.1、Table 9 |
| **R16 Demystifying Network Foundation Models，NeurIPS 2025 主会** | 对冻结 packet/flow encoder 做 embedding geometry、领域特征 alignment、受控协议/环境 perturbation；不是另一个 detector | 4 NFM、5 数据环境；重点在表示诊断，不能把改善的下游 F1 直接换算为本项目 1% 召回。受控扰动不等于任意 CSV 加噪就有网络因果解释 | [官方论文](https://openreview.net/pdf/45fe7201d088e047e18f6fe698679d2f2fece0ad.pdf)、[作者代码](https://github.com/maybe-hello-world/demystifying-networks)。有评估实现/数据管线入口；本次未下载/执行，不是当前模型已完成机制解释 | **A**：§3.1–3.3、§4、Appendix A；Fig.1 |
| **R17 Lens: A Knowledge-Guided Foundation Model for Network Traffic，TMLR 2026 期刊，非 ICLR 主会** | 最新 v5：Tshark 包 metadata/header/payload 文本＋网络 BBPE；整语义单位遮盖及知识模板；约 0.25B T5，分类变约束生成；每流前 **34 包** | 6 来源、12 分类与 5 生成任务；全文/附录规定预训练和下游划分。没有当前 1%/CPU 同协议证据；“新类”仍有适配训练，不是零样本攻击检测保证 | [最新全文 v5](https://arxiv.org/html/2402.03646v5)、[TMLR 入口](https://openreview.net/forum?id=cGDwTgnJIR)、[作者代码](https://github.com/V-Enzo/Lens-foundation-model-for-network-traffic)、[模型卡](https://huggingface.co/Charles59/lens-pretrained)。code、weights、processed downstream data 有入口；模型卡说**预训练 corpus 本身未发布**；code Apache2/model-data CC-BY-NC4 | **A**：§3–4，Appendix A/B/D/E；不是旧 v4 的 3 包版本 |
| **R18 netFound: Principled Design for Network Foundation Models，2026-05-12 arXiv v5／netFound-v2 项目；正式会议身份未核到** | 这是 2023 起源工作的实质更新：协议字段 token、操作 metadata、burst-flow 层级注意力，去 payload/IP；从原始 PCAP 或同 tokenizer Arrow 输入。53M–663M 家族 | §5 表示诊断＋5 下游任务；未核到本项目固定 1% 协议。Appendix A 主要 GPU latency/throughput，不能称已证 CPU 优势 | [最新全文](https://arxiv.org/html/2310.17025v5)、[当前作者代码](https://github.com/SNL-UCSB/netFound)、[small 模型卡](https://huggingface.co/snlucsb/netFound-small)、[作者数据 sampler](https://zenodo.org/records/19863446)。实际核到 2026-04-28、15.8GB、60,199,405 flows 的公开索引；全量 4.2B/约1.2TB 是作者发布声明，未下载验证。旧 [640M v1 卡](https://huggingface.co/snlucsb/netFound-640M-base)明确 legacy，不能混用旧 private-450M 与新数据状态 | **A**：v5 §3–6、Appendix A；当前 README、small model card、Zenodo record |

## 4. 全文精读得到的关键区别

以下分析优先解释与项目的关系；不以论文内部的某个高分构造跨论文排名。

### 4.1 FSNID：不是另一种单变量 MI 排名，成本也不只是最终字段数

其核心比较去掉 $X_i$ 前后的标签条件熵差，能够考虑其他字段已提供的信息，动机是冗余与协同；不能把普通 MI 或“多 seeds 取交集”称为等价方法。§5.3 的既有 split/80:20 和多类目标，与当前组隔离二分类的固定误报评价不同。Appendix C–E 给选择耗时及硬件：Table 1 的 UNSW vanilla/LSTM 为 748/71,501 秒，CIC-IDS17 为 13,565/508,093 秒；这是**选择器开销**，不是学生 CPU 延迟。近期对照准入时应先做 train-only 适配和选择预算核算，再保存选集及下游同协议分数。[全文 §4/§5 与附录](https://www.machineintelligencelab.ai/papers/kdd25.pdf)

### 4.2 Kitsune：低误报和廉价硬件本身早已是研究问题

Fig.9 已报告 TPR@FPR=0.001。§V/Table V/Fig.10 研究 Pi3B 与 i7-4790 单逻辑核的处理率；但该图用的是 KitNET 的基准输入，不可直接当原始抓包到告警的全链路性能。作者 Python README 也明确：要达到论文速度需 Cython/C++，普通 Python 版并非速度等价实现。对本项目的启示是，既要测小 MLP，也要测编码/转换/模型调用；仅参数减少、CSV 预测加速，不能宣称在线网络全链路加速。[全文 §V](https://arxiv.org/pdf/1802.09089)、[作者 Implementation Notes](https://github.com/ymirsky/Kitsune-py)

### 4.3 CARAVAN：LLM 离线辅助有强先例，但不是无条件可信教师

§4.1 和 Appendix A 真正使用 GPT-4 对 UNSW 特征行标注，并生成可缓存规则；采用 few-shot 示例及字段解释。因此“LLM 不能读任何 CSV”不成立，“LLM 标注小模型”也非新概念。其作者同时承认公开数据可能进入 LLM 预训练；示例教师质量是先决条件。实验含人为窗口/新攻击到达的漂移设定、Taurus 硬件与 GPU 重训；不能转写为当前 UNSW 有真实时间流或冻结 1% 召回结论。本项目若用 LLM，应冻结版本、提示、示例来源、成本和标签错误审计，先比较规则/普通教师的增量价值。[全文 §4.1/§5/Appendix A](https://www.usenix.org/system/files/osdi24-zhang-qizheng.pdf)

### 4.4 Pcap-Encoder：最该借鉴的是比较设计和预训练来源审计

§3.4 的协议问答和原始 header 输入，与当前统计字段差异根本；预训练含 UNSW 也是本项目不能忽略的重叠来源。论文通过按流划分及移除 IP/TCP 等字段揭示捷径，提示我们将同组重复、端点/时间代理、家族缺席作为检测对象，而不只换分类器。它的应用/网站识别、冻结/解冻 encoder 实验和浅层基线结果，不支持“某预训练模型在本项目不如 RF”的现成结论。[全文](https://iris.polito.it/retrieve/handle/11583/3003539/21f8a7c5-c796-4cc1-ab4d-3c6cea41412e/3718958.3750498.pdf)、[作者实现](https://github.com/SmartData-Polito/Debunk_Traffic_Representation)

### 4.5 SAFE、Lens 与 netFound-v2：语义表示值得借鉴，不能先占新颖性

SAFE 已经在 CSV 上做特征选择、遮盖预训练及紧凑表示；Lens 最新版专门避免切碎端口等语义单位；netFound-v2 更进一步强调 protocol-field token 与去身份输入。因此“CSV 自监督”“整字段遮盖”“融合领域知识”均不是足够具体的新贡献。可以研究**冗余派生字段导致的重建捷径、低 FPR 尾部损失、可测采集成本与可靠蒸馏的结合**，但须补针对这些精确机制的查重和控制。SAFE Table6 的平均 per-sample time 没有建立本项目同 CPU/p95 协议；Lens 的 4×A6000、netFound 的 GPU 成本也不能替代 CPU 测量。[SAFE](https://arxiv.org/html/2502.07119v1)、[Lens v5](https://arxiv.org/html/2402.03646v5)、[netFound v5](https://arxiv.org/html/2310.17025v5)

### 4.6 tFusion 与 NetVigil：真正问题来自缺失信息和环境变化

tFusion §4.1 明确需要每个包的到达时间/长度和五元组聚流，其跨模态优势不能由现有 CSV 反向获得；§5.1/5.5 的 V100 特征模型也说明“轻量下游”不等于轻量全系统。NetVigil 通过 flow-log 图与时间平滑应对正常流量变化，但 §6.1 的阈值目标是最大化 TPR−FPR。可参考其问题意识及成本边界；不能把两者当现有 CSV 同输入、1% FPR、CPU-only 的直接成绩基线。[tFusion 原文 pp.4/7/11](https://github.com/fuchuanpu/TFusion/blob/main/tFusion_LongVersion.pdf)、[NetVigil §6](https://www.usenix.org/system/files/nsdi24-hsieh.pdf)

## 5. 当前方案已经被覆盖的部分与仍需证明的部分

| 项目表述 | 已有覆盖 | 可以继续研究的准确问题 |
|---|---|---|
| “特征选择让 IDS 更轻” | FSNID、SHAP-Kronecker 等直接先例 | 字段成本/冗余机制如何影响低 FPR 与家族；字段数减少是否真的减少采集、编码、CPU 成本 |
| “大教师蒸馏小模型” | DistillGuard、SHAP-Kronecker、联邦 KD 文献 | 树教师受类平衡权重影响的 soft score，与部署误报预算是否失配；同学生下 soft 信息是否超过权重匹配 hard 控制 |
| “只向可靠教师学习” | 选择性 KD 大概念已有，输出/隐藏表示选择各有不同机制 | 本项目 gate 在 binary/OOF/类重加权条件下究竟选择了什么，低分正常组是否受保护；需直接机制对照及更细近期 KD 查重 |
| “加入低 FPR 与难样本损失” | Kitsune 的低 FPR 报告、已有 ranking/pAUC 研究；不由本表宣称首创 | 可微训练代理与最终离散 1% 阈值的关系；困难正常样本是否由少数大组支配；相同 pair/更新预算下是否有增益 |
| “用自监督/基础模型提高泛化” | SAFE、ET-BERT、YaTC、TrafficFormer、Lens、netFound 等 | 数据输入允许什么预训练；收益来自额外未标注数据、架构还是计算；去捷径与新环境检验之后还能保留多少 |
| “用 LLM 辅助 IDS” | CARAVAN 明确先例 | LLM 字段知识是否超过简单规则/词典；标签误差、成本与复现；不能把自然语言解释直接当攻击因果 |
| “加入图/时序信息” | NetVigil、tFusion 等 | 先有真实端点/时间/包序列；缺元数据时不能编造图或从 CSV 行序获得时序创新 |
| “完整可复现与交互系统” | 多个作者 artifact 已有成熟工程 | 工件审计/部署是贡献支撑，不能单独充当新学习算法；界面应显示真实分歧、误报组与成本 |

“本表未发现完全相同公式”不等于新颖性成立。当前 gate、batch ranking 和组合仍是候选设计，需将数学差异、最近相关方法及反例查全；若收益消失，也可将可靠的失效条件作为研究结论，而不是给组合重新命名。

## 6. 真正值得推进的研究问题与可执行边界

### P1：二元蒸馏的低误报目标失配与可控纠正

**问题：**普通 KD 变化来自教师软信息，还是来自有效类别/样本权重变化？类平衡树教师的分数置信是否代表部署尾部的可靠性？哪些正常关联组贡献了误报？

**可执行：**保留现有同初始化/同 batch/同学生控制；普通 hard、精确 matched-hard、KD、gated-matched-hard、不同 gate 机制，逐损失记录。再按训练组检查 gate 错误 mass、teacher score 分布和 top-normal 集中度；已有 validation 只用于开发。若要改教师校准，应在训练组内部留出/交叉拟合，不能利用旧 test 反调。研究不要求新采集；确认性结论仍需未使用来源。不要把“score 离 0.5 很远”未经核验叫高置信概率。

**贡献门槛：**明确预测并检验一个失败机制；改善超过相同逐行权重控制，且低 FPR/家族/CPU 取舍有实际意义。若只有一组均值更高，没有机制和新确认，仍属于开发结果。

### P2：按实际成本压缩字段，同时保护尾部和独立组

**问题：**最少字段与最少成本并不相同；某些便宜代理可能只记住采集环境，某些昂贵字段才保护稀有攻击。普通 MI/稳定性不能保证实用工作点。

**可执行：**先以字段字典区分直接字段、比率、时间/会话统计与派生字段族，明确当前只能测 CSV 预处理/推理成本，不能声称测到真实网络采集成本。在相同可见训练/选择分区，比较 MI、冗余控制和经审计可行的 FSNID；报告检测—编码维度—CPU p95—包大小的 Pareto。只有保留真实提取器并测量后，才能将目标升级为采集预算。若时序元数据不存在，先只评估 FSNID 非时序版本。

**贡献门槛：**证明哪种冗余/代理失效使简单策略不足，以固定其余因素的移除/恢复对照验证；不以“选到更稳定字段”单独作为检测收益。

### P3：语义组自监督与蒸馏，先排除重建捷径

**问题：**字段间确定性派生关系可能使 MAE 只完成算术重建；随机遮盖学到的表示是否保护低 FPR 边界，而不只是重建误差更低？

**可执行：**现有 CSV 上 train-only 正常数据或明确定义的无标签 train；从头训练、相同预算随机遮盖、语义整组遮盖、去掉可直接回算的派生量，比较表示/下游/少标签效率。SAFE 是必须回应的直接先例，Lens/netFound 只提供语义设计动机，不能声称其复现。输入图化、预训练 epochs、额外数据、学生训练总成本均纳入控制。

**贡献门槛：**某个领域约束解决具体可复现问题，并在比简单 MLP/原始特征 novelty detector 更强的控制后仍有用；“把 CSV 变图片”本身不成立。

### P4：包级富信息教师→仅 CSV 学生，条件具备后再做

**问题：**学生压缩能否利用额外包语义，而不把大模型带到部署？

**可执行条件：**有公共原始 PCAP、可靠标签及 packet/flow→CSV 一一映射；固定统一提取器和真实分组；只在 train 生成 OOF 富输入教师，比较相同学生下“CSV 教师 KD”与“包级教师 KD”。可以使用公开 PCAP，**无需自行新抓包**；但需要新数据处理、来源/预训练重叠审计，且当前 42/82 CSV 单独不能完成。netFound-small/v2 相比旧640M更值得先审小规模适配，但预训练 Arrow sampler 本身不提供本项目攻击 ground truth。

**贡献门槛：**增益来自额外信息而非更大的搜索/教师开销；部署学生 CPU 成本确实低；旧 final test 不再包装为未见确认集。这是投入较高、目前条件性路线，不应因为近期模型大而直接替换主线。

## 7. 同协议比较应如何建立

1. **不汇总跨数据集 accuracy 排名。** 每个候选在同一版本/组分区、输入字段、标签定义、训练/选择预算下比较；原论文成绩只用于说明其研究任务。
2. **统一工作点而非仅列 FPR。** 保存连续分数；开发阶段统一 validation 经验规则；确认阶段用独立 calibration 锁阈值再测未开封 test。报告正常数量、有效独立组、TP/FP、召回、每家族支持和区间。1% 是经验预算，不是分布漂移下保证。
3. **同输入基线和富输入方法分表。** 只看 CSV 的学生不能与看 PCAP/真实 host 图的模型直接解释为架构优劣；可另做信息增量研究。
4. **成本至少分四项。** 数据采集/流完结等待、提取/编码、模型 CPU 推理、训练/调参/教师辅助成本。GPU 特征提取＋小 RF 不能叫 CPU 全链路轻量；选择器 runtime 不能冒充推理 runtime。
5. **复现身份逐级升级。** 官方入口已核→锁 commit/checkpoint/许可证→schema 工程试验→不泄漏的训练适配→同协议原始分数/成本→独立复算→新来源确认。任一步没做就保留“未执行/未验证”。

## 8. 优先对照与暂缓项

| 优先级 | 选择 | 理由与下一闸门 |
|---|---|---|
| 必须 | 简单 hard MLP、树强基线、原始 CSV novelty baseline、数学精确匹配控制 | 先确定复杂目标是否提供独立信息；不应只与弱学生比较 |
| 近期直接对照 | FSNID 非时序版本、SAFE 适配；DistillGuard 先补作者工件/完整实现审计 | 与现有输入最接近；前两者也需 split/预算适配；DistillGuard 尚不足以承诺复现 |
| 机制参考 | Kitsune、CARAVAN、Pcap-Encoder/NeurIPS 表示审计 | 低 FPR、慢教师、数据捷径和全链路成本；有助于把问题定义准确 |
| 条件性额外研究 | netFound-v2 small、Lens、tFusion、TrafficFormer | 需要 PCAP/真实元数据、不同输入管线；先做来源与映射可行性，不进入当前 CSV 统一主表 |
| 暂缓 | 为“前沿”而加联邦、伪时间序列、泛 LLM/VLM 数值行分类 | 无对应威胁模型/真实时间/新增信息时不能解决当前问题；CARAVAN 是先例而非自动成功证据 |

## 9. 核查限制及后续补证

- 正式身份分别核到主会、期刊、普通会议、workshop/预印本，未将作者仓库标题视为充分 venue 证明。SAFE 特别按官方 AICS workshop 议程处理；netFound-v5 仍按预印本处理。
- **版本会改变结论。** Lens v5 使用 34 packets；netFound v5 已转 v2 家族并公开 sampler，不能沿用旧稿的模型名、3 packets 或 private-only 概述。反过来，2025 审计的结论也不能不加版本地套到 2026 更新模型。netFound-v5 对其他模型的对比表不代替逐篇最新模型卡。
- TrafficFormer、DistillGuard、SHAP-Kronecker、FedKD-IDS 等 B 项的完整 split/阈值/成本仍需精核，未用搜索片段补写缺失事实。某些原文/下载地址访问不稳定；源链接保留给后续核验，不声称已保存全部论文或工件。
- 当前代码/论文入口不是锁版本清单。真正选入实验时须单独记录 commit、checkpoint revision、数据 SHA、依赖版本和具体修改；本文没有自动安装、批量下载、训练或消费保留分区。
- 本文没有证明任何新方法优于现有系统，也没有完成对所有 gate、pAUC、代价敏感选择方法的查重。它给出已被覆盖的概念、能落地的问题和准入边界，供统一研究设计取舍。
