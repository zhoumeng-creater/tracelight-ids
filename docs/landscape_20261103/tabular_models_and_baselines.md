# 表格模型、基础模型与强基线审查

核查日：2026-11-03。服务于 TraceLight 的统计流特征 CSV 二分类、低误报评价与 CPU 部署。这里的“可适配”只指输入和方法有合理对应，不表示已经安装、训练、复现或获得本项目收益。本轮未运行真实数据训练。

## 1. 当前比较缺什么

现有 LR、RF、sklearn HistGradientBoosting、小 MLP 是必要起点，但不足以代表当前表格学习的强参照。HistGradientBoosting 不能替代经过合理调优的 CatBoost／LightGBM；一个固定 tiny TabM 也不能代表 TabM 的方法上限。应先补强这些对照，再检验蒸馏是否有独立收益。

基础模型有两个角色：离线产生训练信号的教师，或者独立的质量参照。大教师可以使用实验室 GPU，但部署学生必须单独测 CPU；教师权重小、预测无梯度、学生参数少，都不自动等于最终管线低延迟。

## 2. 模型与基准地图

“全文”表示读过方法／协议／局限的相应章节；“代码”表示查看官方入口或文档，并非执行。老方法只保留不可替代的强基线，新增前沿候选优先选 2025／2026 年可核实工作。

| ID | 工作及身份 | 已有核心机制／实际输入 | 与本项目的关系及取舍 | 本轮证据 |
|---|---|---|---|---|
| T01 | CatBoost，成熟强基线 | 有序提升、类别统计；混合类型表格 | 优先补齐。使用原始类别列的规范适配，独立校准；不能只与未调优 RF 比 | [官方算法与参数](https://catboost.ai/docs/en/concepts/algorithm-main-stages_choose-tree-structure)、[参数](https://catboost.ai/docs/en/references/training-parameters/common)；文档 |
| T02 | LightGBM，成熟强基线 | 直方图、逐叶生长、类别分裂 | 优先补齐。限制叶数／深度并记录 CPU 线程；与 sklearn HGB 分列 | [官方机制](https://github.com/lightgbm-org/LightGBM/blob/main/docs/Features.rst)；文档 |
| T03 | RealMLP，NeurIPS 2024；保留为必要旧基线 | 针对表格的 MLP 训练与预调默认值 | 判断小 MLP 的差距来自训练配方还是容量。代码默认、调参和集成三种口径分列 | [全文](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf)、[作者 PyTabKit](https://github.com/dholzmueller/pytabkit)；全文／代码 |
| T04 | TabM，ICLR 2025 | 参数共享的 MLP 集成、可选数值嵌入 | 优先神经参照／教师。当前已有官方包、tiny 结构和工程 pilot；正式 CPU 三轮部署测量与充分调优成绩尚无，新协议早停已修正 | [全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)、[作者实现](https://github.com/yandex-research/tabm)；全文／代码 |
| T05 | ModernNCA，ICLR 2025 | 深度表示与随机化邻域成分学习，预测依赖参考样本 | 用于检验局部近邻结构；不列第一批必跑。完整成本包含参考表示和检索，不能只数神经网络参数 | [全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/ecd294194d5155432aab533f7ce0f6fe-Paper-Conference.pdf)、[作者 TALENT](https://github.com/LAMDA-Tabular/TALENT)；全文 |
| T06 | TabICLv2，ICML 2026 主会 | 表格上下文学习；目标感知表示、可扩展注意力、合成先验与预训练更新 | 优先基础模型候选，先做教师资格和显存检查。上下文、表示与缓存必须折内生成 | [正式记录](https://proceedings.mlr.press/v306/qu26a.html)、[作者代码](https://github.com/soda-inria/tabicl)、[权重](https://huggingface.co/jingang/TabICL)；全文／代码／模型卡 |
| T07 | TabPFN 3.5／Fast，2026-09 技术报告与发布 | 表格 ICL，新 checkpoint 同时支持分类／回归 | 可作当前前沿性能参照；不默认用于可公开 KD。模型许可与输出使用限制需按具体用途处理 | [报告](https://arxiv.org/abs/2609.17895)、[模型卡](https://huggingface.co/Prior-Labs/tabpfn_3_5)、[许可](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE)；报告／卡／许可 |
| T08 | TabDPT，NeurIPS 2025；后续模型 1.3 | 真实表格预训练；当前推理版本与原论文分开锁定 | 与合成先验 TFM 对照。核对预训练数据重叠；先限合法训练上下文，不默认全量 CIC 可放入 24GB | [正式记录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html)、[软件 1.3.2](https://github.com/layer6ai-labs/TabDPT-inference/releases/tag/v1.3.2)、[权重](https://huggingface.co/Layer6/TabDPT)；论文／代码／卡 |
| T09 | Mitra，NeurIPS 2025 主会 | 多种合成任务先验的混合，表格 ICL | 说明教师质量还取决于先验设计；实际候选应进一步核 v2，不能沿用 v1 为“最新” | [正式论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/177d68f4adef163b7b123b5c5adb3c60-Abstract-Conference.html)；全文 |
| T10 | Mitra-v2，2026-09-03 技术报告 | 扩展上下文与先验、优化更新；分类 backbone 约 75.7M；默认结果含微调与 8 折 bagging | 第二位可研究教师，先核单模型／集成成本。不能把默认排行榜成绩说成单次零样本前向的收益 | [全文](https://arxiv.org/html/2609.04540v1)、[分类卡](https://huggingface.co/autogluon/mitra-classifier-2)、[作者代码与结果](https://huggingface.co/autogluon/mitra-finetune)；全文／卡／入口 |
| T11 | ConTextTab，NeurIPS 2025 主会；当前代码重定向 SAP-RPT-1-OSS | 文本、日期、数值和列名语义编码，真实数据预训练，行列注意力 | 用于检验语义信息是否值得成本；UNSW 类别词与计数列不等于自然语义丰富业务表。先核真实数据来源及许可 | [正式记录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/d807e7678ba3afd3a904f4af52819e77-Abstract-Conference.html)、[全文](https://arxiv.org/html/2506.10707v3)、[当前作者入口](https://github.com/SAP-samples/sap-rpt-1-oss)；全文／代码 |
| T12 | LimiX-2M，2026-06 预印本 | 数值局部 RBF 表征、调整样本／特征交互顺序，约 2M 模型 | 是“基础模型也可较小”的参照。与 2026-09 的 LimiX-2 是不同型号，不能混用参数与分数 | [论文](https://arxiv.org/abs/2606.04485)、[作者家族仓库](https://github.com/limix-ldm-ai/LimiX)；论文记录／仓库 |
| T13 | LimiX-2，2026-09-15 预印本 | 联合分布与上下文条件遮盖、细胞级行列表示，支持预测／插补等 | 有意义的前沿备查，暂不作为首选教师。自定义代码和非商业权重许可分别核对，联合建模不等于已有本项目鲁棒性 | [全文](https://arxiv.org/html/2609.17488v1)、[官方代码](https://github.com/limix-ldm-ai/LimiX)、[权重与许可](https://huggingface.co/stable-ai/LimiX-2)；全文／代码／卡 |
| T14 | OrionMoE-Small v1.0，2026-10-14 预印本 | 按数据集上下文和单元表示选择稀疏专家，纯合成预训练 | 新但不宜立即纳入复现。论文列出的 GitHub 本轮 404，HF API 401，官网是申请制 early access；不是已取得可运行公开模型 | [全文](https://arxiv.org/html/2610.20418v1)、[官方状态](https://orionmoe.lexsi.ai/)；全文／入口状态 |
| T15 | Google TabFM v1.0.0，2026 官方模型／项目 | 表格 ICL，JAX／PyTorch 接口，混合类型输入 | 备查参照。代码 Apache-2.0 与非商业预训练权重分开；官方 README 暂无完整技术论文链接，不填顶会身份 | [作者项目](https://github.com/google-research/tabfm)；README／代码入口 |
| B01 | TabArena，NeurIPS 2025 Datasets & Benchmarks | 持续维护的表格模型、调参、集成比较 | 借鉴公平预算和版本锁定，不把 Elo 当本项目低 FPR 证据 | [正式记录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)、[代码](https://github.com/autogluon/tabarena)；全文／代码 |
| B02 | TALENT，JMLR 2025 26(226) | 统一工具箱与大量表格学习方法 | 获取可靠适配入口与比较协议；表格行随机划分不能直接移入关联网络流 | [期刊记录](https://www.jmlr.org/beta/papers/v26/25-0512.html)、[代码](https://github.com/LAMDA-Tabular/TALENT)；记录／代码 |
| B03 | BeyondArena，2026-06 预印本 | 同时关注 IID、分组和时间划分的表格基准 | 直接支持“榜单强不等于组／时间迁移强”的研究设计；正式会议身份本轮未核实 | [全文](https://arxiv.org/html/2606.30410v1)、[官方基准说明](https://huggingface.co/spaces/TabArena/leaderboard/blob/main/website_texts.py)；全文相关协议／结果 |

## 3. 六个需要落实到实验的阅读结论

### 3.1 TabM：不能用一个 tiny 配置代表前沿对照

原文 §3.3／§3.5：训练分别计算成员损失，推理平均概率，早停应依据集成表现。当前适配器已改用集成概率的 weighted BCE 选轮；这属于修复复现偏差。论文附录搜索含 k=32 和更宽网络，现有 k=8、width=32 是低成本配置。需要另列合理调优的质量点，并比较训练较大集成后裁剪成员，成本与结果均实测。当前没有新协议真实训练结果。

### 3.2 ModernNCA：模型文件之外还有参考样本

原文 §3 的预测通过学习后的邻域关系聚合标签，训练可对参考样本随机抽取。它提示：UNSW／CIC 的重复模式、局部密度与异常组值得检查。但检索方法需要保存或产生参考表示，CPU 吞吐、启动内存和支持集规模都必须计入。若仅比较网络参数，容易把存储转移到支持集后误称压缩。此方法先作机制参照，避免第一批增加昂贵全量邻域计算。

### 3.3 TabICLv2：基础模型上下文也是训练资料

目标感知表示使某些预计算结果依赖上下文标签。OOF 不仅要把预测对象移出 fit，还须保证编码、上下文、缓存都来自该折训练侧。百万行、50GB 级论文能力不等于本项目 24GB 显存可用；CIC 约 128 万训练行，需要先做折内上下文规模与攻击支持检查。它是首选候选之一，理由是机制、输入和公开工件有较清楚对应，不是已证实比 HGB 强。

### 3.4 Mitra-v2：较新成绩的成本口径必须拆开

原文 §3.1、§3.6、§5 明确，headline 系统使用短微调及 8 折 bagging；不是零样本、单模型或已证明 Pareto 支配。作者也披露领先组区间重叠、预测时间未单独仪表化，以及多项预训练变化未完全分解。若选为教师，应先比较冻结前向与受限微调，记录所有 bag 子模型、上下文、显存与 OOF 成本；不能直接将其榜单名次作为选教师依据。

### 3.5 ConTextTab／LimiX-2：语义编码与遮盖不是空白

ConTextTab 已利用列名和文本语义；LimiX-2 研究上下文条件遮盖及联合表格建模。因此“给字段加语义”“遮住字段自监督”“加行列注意力”都需要更具体的差异。对网络统计量，应先证明语义分组比随机分组和派生字段泄漏控制更有效。没有这一步，换一个有语义名称的模块仍不足以构成创新。

### 3.6 BeyondArena：迁移检验比换一个榜首更关键

该工作把分组和时间划分纳入基准，强调不能把 IID 表格的结论直接推广到非 IID 任务。适配到本项目，应保留网络会话／采集单元的隔离、重复模式敏感性以及可靠时间元数据。按 CSV 行序构造“未来”不成立；合成组比例变化只能称受控压力测试。基础模型同样需要接受这些检查。

## 4. 2026-11-03 实查的模型身份

以下通过 Hugging Face 官方 API 查询仓库元数据和文件列表；只读元数据，没有下载权重或执行模型。仓库 revision 不等于 checkpoint 文件 SHA-256；运行前仍须下载校验。

| 模型仓库 | 本轮 revision | 对应文件／状态 | 模型许可标记 |
|---|---|---|---|
| autogluon/mitra-classifier-2 | edada0d20759c58ada8c8605c25f22f6e98ea5f0 | model.safetensors | Apache-2.0 |
| jingang/TabICL | 4dcd344ece2c00be9e831fdd35bed57b5ad83e19 | tabicl-classifier-v2-20260212.ckpt | BSD-3-Clause |
| Prior-Labs/tabpfn_3_5 | 06bf2ba35c80a92a3b9abb436b99cf49e7a0365e | 主模型、multiclass、fast 三个 safetensors；本项目尚未选用 | 自定义，见全文 |
| Layer6/TabDPT | a5ca6e01c0fa09ec68c73e958e5199d1932abb3a | tabdpt1_3.safetensors；不是软件 1.3.2 | Apache-2.0 |
| stableai-org/LimiX-2 | de07b679e74a41b50b9de18251a8fa245e537440 | LimiX-2.ckpt；当前文档还使用 stable-ai 别名 | 自定义非商业权重许可 |
| Lexsi/OrionMoE | 未取得 | 公开 API 返回 401；论文链接与官网状态不一致 | 论文／官网为 LSAL，自定义 source-available |

软件、模型发布日期和论文年份不能互相代替。TabDPT 1.3.2 更新解释接口，不代表另发了一个同名模型；ConTextTab 旧仓重定向后，运行接口和论文原版也需分开记录。对 OrionMoE 的访问失败只证明本轮公开入口未取得，不推出模型不存在。

## 5. 最小而有说服力的比较顺序

| 阶段 | 模型／工作 | 入场条件与输出 |
|---|---|---|
| 先补强 | CatBoost、LightGBM、充分调优 TabM，加现有 RF／HGB／小 MLP | 在同一开发分区、同一低 FPR 目标与公开预算下选型；保留类别处理差异与成本 |
| 选一个 TFM | TabICLv2 优先；若质量／资源不合格，再查 Mitra-v2 或 TabDPT 1.3 | 在实验室先做 2–3 个折内上下文规模检查；未胜出不生成全矩阵 OOF |
| 再研究迁移 | 最强合格教师与固定小学生 | 比普通 KD、严格 matched-hard、已发表工作点方法及独立低 FPR 训练基线；教师和机制不能一起替换 |
| 有依据才扩展 | ModernNCA、RealMLP、ConTextTab、LimiX 家族等 | 具体失败诊断需要它时增加；不将模型数量当论文创新 |

调参优先使用可记录的墙钟／算力上限、统一开发划分和停止规则。不同方法每次 trial 成本不同，不能声称相同次数就完全公平。报告默认值质量和预算内调优质量两个口径；不为新方法挑最优 seed，也不以旧 tiny pilot 否定方法。

所有真实训练与正式资源测量均在实验室 RTX4090D；本轮的文献核查不属于新实验结果。下一步的优先级是先回答“强教师是否合格”和“最近邻是否已经解决问题”，再决定是否值得扩展蒸馏矩阵。
