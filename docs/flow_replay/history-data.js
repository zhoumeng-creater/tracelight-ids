/* Editorial summaries of archived evidence. No generated predictions or live metrics. */
"use strict";
window.TRACELIGHT_HISTORY = {
  schema: "tracelight-research-history-v1",
  asOf: "2026-11-19",
  currentBatchExecuted: false,
  algorithmInnovationEstablished: false,
  compilationSources: [
    "E:/OneDrive - Coventry University/中传/毕业论文/工作/研究历程与设计重建_20261119/early_evidence.md",
    "E:/OneDrive - Coventry University/中传/毕业论文/工作/研究历程与设计重建_20261119/later_evidence.md"
  ],
  categories: [
    { id: "lightweight", label: "早期轻量化" },
    { id: "kd", label: "KD" },
    { id: "cache-input", label: "缓存／输入" },
    { id: "history-transfer", label: "历史与迁移" },
    { id: "model-reset", label: "模型重设" },
    { id: "independent-sources", label: "独立来源" }
  ],
  statuses: [
    { id: "completed", label: "完成实验" },
    { id: "analysis", label: "仅分析／装配" },
    { id: "withdrawn", label: "已撤下 · 未运行" },
    { id: "pending", label: "待实验室" }
  ],
  stages: [
    {
      id: "baseline-fields", date: "2026-11-01", dateLabel: "2026.10.27 — 11.01", rounds: "早期 / R1–2",
      categories: ["lightweight"], status: "completed",
      title: "建立强基线，测量少字段的真实代价",
      purpose: "在相同数据划分中比较 LR、RF、HGB、MLP，再检验 8／16／24 字段是否兼顾检测与成本。",
      scope: "UNSW 镜像 · validation 开发工作点 · 三种子；早期本机 pilot 与工程测试另列",
      positive: "全字段 HGB 召回 88.7666%。42→8 字段时，MLP 单条 p50 从 2.205 降至 0.668ms，HGB 从 2.853 降至 1.320ms。",
      limitation: "九个精简配置都未达到当时平均召回损失 ≤2pp 的要求；最接近的 HGB24 仍下降 3.2845pp。字段少也不保证树模型更小。",
      disposition: "保留强简单基线与成本收益，关闭这套 MI 精简候选；不把局部负结果扩大为所有特征选择无效。",
      sources: [
        { label: "首轮结果", href: "../LAB_RESULTS_20261101.md" },
        { label: "第二轮冻结协议", href: "../STAGE2_PROTOCOL.md" },
        { label: "第二轮接收核验与原因分析", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261101_第二轮实验室4090D/核验后报告/接收核验与原因分析.txt" }
      ]
    },
    {
      id: "small-model-success", date: "2026-11-01", dateLabel: "2026.11.01", rounds: "R3",
      categories: ["lightweight", "kd"], status: "completed",
      title: "小网络压缩成立，蒸馏增量没有达标",
      purpose: "把网络缩小的效果与同结构 KD 的效果分开。",
      scope: "UNSW · 42 字段 · 同一 validation · 三种子",
      positive: "small-hard 相对 large-hard 平均召回仅下降 0.03pp，部署包 68,837→39,338B，达到当时检测保留要求。",
      limitation: "KD 对 small-hard 只增加 0.26pp，Fuzzers 下降 1.89pp，未过预定收益条件；约 2.36ms 的单条 p95 基本不变。",
      disposition: "保留小网络作为真实轻量化成果。学生比教师小是结构变化，不能直接记成蒸馏收益。",
      sources: [
        { label: "第三轮实现与验证", href: "../STAGE3_VALIDATION.md" },
        { label: "第三轮完整分析", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/STAGE3-ANALYSIS-2026-11-01.md" }
      ]
    },
    {
      id: "cic-lightweight", date: "2026-11-03", dateLabel: "2026.11.03", rounds: "R4–5",
      categories: ["lightweight"], status: "completed",
      title: "CIC 的轻量化收益兑现，UNSW 校准没有迁移",
      purpose: "比较稳定选择与简单 MI，再用冻结模型和独立正常校准完成测试。",
      scope: "UNSW 与 CIC 分别训练／校准；正式测试三种子；成本沿用相同哈希工件",
      positive: "CIC MI24／全字段小 MLP 测试召回 99.1539%／99.2155%；MI24 包缩小 43.73%，批吞吐为 3.19 倍。",
      limitation: "没有事先注册统计等效界限。UNSW RF／小 MLP 测试误报率为 2.5694%／3.2063%，均超过校准时的 1%。",
      disposition: "保留可量化的质量—成本取舍；稳定选集不等于检测更好，校准经验预算也不等于未来误报保证。",
      sources: [
        { label: "第四轮验证说明", href: "../STAGE4_VALIDATION.md" },
        { label: "第五轮冻结协议", href: "../STAGE5_PROTOCOL.md" },
        { label: "第五轮最终测试复核", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261103_第五轮实验室4090D/复核分析/第五轮最终测试复核.md" }
      ]
    },
    {
      id: "kd-pilot", date: "2026-11-03", dateLabel: "2026.11.03", rounds: "R6 小试",
      categories: ["kd"], status: "completed",
      title: "普通 KD 出现正信号，复杂组合没有自动获益",
      purpose: "用 matched-hard、门控和 ranking 对照，分解软目标与额外权重的作用。",
      scope: "UNSW · train 12,000 / validation 6,000 · 单 seed · 最多 20 epochs",
      positive: "同次开发工作点下，普通 KD 召回 80.1815%，hard 为 78.6608%，matched-hard 为 77.6306%。",
      limitation: "默认门控 KD 为 78.6117%，hard＋ranking 为 76.4778%；多个条件最佳轮贴近上限，不能据此给模型家族排名。",
      disposition: "保留这次正信号及其有限条件；后续完整受控结果决定是否继续，不把 pilot 写成方法已成立。",
      sources: [{ label: "第六轮实际运行与验收", href: "../STAGE6_VALIDATION.md" }]
    },
    {
      id: "risk-kd-design", date: "2026-11-04", dateLabel: "2026.11.03 — 11.04", rounds: "risk-v3 / v4",
      categories: ["kd", "model-reset"], status: "completed",
      title: "发现目标退化、代理间隙与确定性空转",
      purpose: "检查边界蒸馏和受约束升级为何没有形成可用改善。",
      scope: "两环境；v4 计划 48 学生，实际只训练 UNSW 两 seed 的 16 个",
      positive: "定位 CIC 约 99.63% 正常 OOF 目标退化到 0.5；证明三个失败状态存在更高且满足联合预算的阈值。",
      limitation: "v4 raw KD 仅净增 1／29 TP，同时新增 1／3 FP。guard 318 次重复拒绝；CIC 学生未启动，不能计为 KD 阴性。",
      disposition: "修复目标、停止空转，区分实际事件与代理约束；保留原失败判定，撤回未经成立的创新主张。",
      sources: [
        { label: "risk-v3 结果复核", href: "../RISK_V3_RESULTS_20261104.md" },
        { label: "v4 结果与阈值反例", href: "../UPGRADE_V4_RESULTS_20261104.md" },
        { label: "根本设计审查", href: "../RESEARCH_ROOT_CAUSE_REVIEW_20261104.md" }
      ]
    },
    {
      id: "same-input-reference", date: "2026-11-05", dateLabel: "2026.11.04 — 11.05", rounds: "R8–9",
      categories: ["cache-input", "model-reset"], status: "completed",
      title: "同输入强模型能检出，输入裁剪却有交换代价",
      purpose: "分开教师优势、字段裁剪、学生表示和范围外推。",
      scope: "两库固定 screen；R9 复用权重做输入干预，无新拟合",
      positive: "CIC 24 字段 MLP 的 2,293 条 FN 中，2,272 条属于同一模式；同输入 HGB 能检出，裁剪也复现了局部恢复。",
      limitation: "裁剪带来新的 FP／FN，没有一个有 C 检出增益的候选同时达到当时无新增错误条件；screen 没有训练 KD。",
      disposition: "保留局部输入机制及强参照，停止把裁剪作为通用替换。历史无损门不升级成当前永久采用门槛。",
      sources: [
        { label: "输入外推结果", href: "../INPUT_EXTRAPOLATION_RESULTS_20261105.md" },
        { label: "第八轮详细分析", path: "E:/OneDrive - Coventry University/中传/毕业论文/工作/实验结果分析_20261105/本轮实验详细分析_20261105.md" }
      ]
    },
    {
      id: "kd-mechanism-exit", date: "2026-11-06", dateLabel: "2026.11.05 — 11.06", rounds: "R10–12",
      categories: ["kd", "model-reset"], status: "completed",
      title: "优化补救没有承接教师优势，结束固定 KD 分支",
      purpose: "比较 hard／真实教师／置乱／均值软化，再直接测终点、梯度和 mini-batch 补救。",
      scope: "UNSW · 同 full-field small 起点；full 与 mini 方案配对；R11 为零更新诊断",
      positive: "KD full 对 hard 净增 5 条 C TP；测得起点 hard/teacher 梯度余弦 0.998321。HGB 对 H40 在 C 多 488 TP、少 2 FP，p95 只多 0.81467ms。",
      limitation: "full 未过当时收益门槛；mini 的 KD 对 hard 少 30 条 C TP、少 8 FP。优化与表示未唯一分离，不能把软目标熵当不可学习噪声。",
      disposition: "停止原 small 的温度／轮数续试；没有约 35KB 硬上限，就让强树进入实际方案比较。",
      sources: [
        { label: "第十轮四臂结果", href: "../UNSW_KD_RESULTS_20261105.md" },
        { label: "第十一轮终点、梯度与成本", href: "../UNSW_DIAGNOSTIC_RESULTS_20261106.md" },
        { label: "第十二轮机制与实际价值", path: "E:/OneDrive - Coventry University/中传/毕业论文/代码/tracelight-ids/docs/research_alignment_20261106/unsw_minibatch_results/机制与实际价值复核.md" }
      ]
    },
    {
      id: "cache-input-repair", date: "2026-11-08", dateLabel: "2026.11.07 — 11.08", rounds: "R13–14",
      categories: ["cache-input"], status: "completed",
      title: "修复输入取得总体收益，普通 QCOC 也有真实压缩价值",
      purpose: "先修未知类别归并，再在同一新输入上比较完整 KV 与普通 QCOC。",
      scope: "NF-ToN-IoT · 48 字段 · P8,192 / 正常 C5,000 / E20,000",
      positive: "R14 输入修复净增 284 TP、减少 14 FP；混合标签涉及行从 3,699 降至 2。QCOC 再净增 4 TP，FP 不变，完整包缩小 70.90%。",
      limitation: "扫描却从 54 降至 3/1,078。可区分输入不等于可用的低误报边界；这些结果没有验证拟议的新缓存几何。",
      disposition: "保留输入修复与普通压缩基线。先判断完整教师是否完成任务，再讨论压缩新机制。",
      sources: [
        { label: "输入修复合同", href: "../CONTEXT_INPUT_REPAIR_V2.md" },
        { label: "第十四轮实际结果", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261108_第十四轮输入修复/实验分析.md" },
        { label: "第十三轮及 QCOC 来源", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261107_第十三轮上下文缓存/实验分析.md" }
      ]
    },
    {
      id: "port-label-probes", date: "2026-11-08", dateLabel: "2026.11.08", rounds: "R15–16",
      categories: ["cache-input"], status: "completed",
      title: "端口与标签都影响输出，却没有解决扫描—正常混淆",
      purpose: "用端口重编号／遮蔽及真实／伪三类标签检验具体补救。",
      scope: "固定原 P/C/E；真实三类与伪三类支持计数一致",
      positive: "干预确实改变分数。真实三类已识别扫描相对其他攻击的结构，子类排序 AUC 为 0.999984。",
      limitation: "端口两干预均未恢复旧目标 54 条扫描；真实三类扫描仍 3/1,078，净少 47 TP、多 1 FP，主组正常也被指向扫描。",
      disposition: "关闭这两种具体补救；不能说标签语义没有作用，也不靠换映射或伪类 seed 继续试。",
      sources: [
        { label: "三类对照协议", href: "../SCANNING_LABEL_LAB_V1.md" },
        { label: "标签粒度实际结果", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261108_第十六轮标签粒度/实验分析.md" },
        { label: "端口干预实际结果", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261108_第十五轮Scanning机制/实验分析.md" }
      ]
    },
    {
      id: "route-comparison", date: "2026-11-11", dateLabel: "2026.11.09 — 11.11", rounds: "R17–18",
      categories: ["cache-input", "lightweight"], status: "completed",
      title: "得到三档成本取舍，换大模型没有恢复共同盲区",
      purpose: "完整比较 KV、QCOC、TabM，补齐 TabPFN 和缓存构建路径证据。",
      scope: "同源新通信组 E′；TabPFN 只补旧 E；各成本按实际 CPU 路径记录",
      positive: "E′ 上 QCOC 比完整 KV 少 50 TP、少 8 FP，包缩小 70.90%、单行 p95 降低 64.396%。约 318KB 的 TabM 更快，但检出代价更大。",
      limitation: "TabPFN 在旧 E 扫描 0/1,078；固定单 SYN 模式在 E′ 有 544 扫描、80 正常，28 臂均未告警。局部冲突不解释全部漏检。",
      disposition: "保留质量—成本比较，停止无依据扩容量；缓存重放差异已定位，不能继续记作模型失效原因。",
      sources: [
        { label: "完整路线设计", href: "../ROUTE_BATCH_DESIGN_20261108.md" },
        { label: "第十七轮实际结果", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261109_第十七轮路线批次/实验分析.md" },
        { label: "第十八轮实际结果", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261111_第十八轮路线补验/实验分析.md" }
      ]
    },
    {
      id: "history-signal", date: "2026-11-11", dateLabel: "2026.11.11", rounds: "R19–20",
      categories: ["history-transfer"], status: "completed",
      title: "历史有局部扫描信号，排序与登记工作点开始分离",
      purpose: "加入四项严格先前 60 秒历史，检查前向与独立采集迁移。",
      scope: "R19 NF-ToN-IoT；R20 CIC2018；四项历史一起变化",
      positive: "R19 HGB 扫描 37→222/3,267，正常 FP 48→2/22,119。R20 已存分数在事后同误报上界下也有扫描和 Bot 排序增益。",
      limitation: "R19 历史模型其余八族 0/24,614；R20 实际登记点扫描 199→190，Bot 6→0。事后排序上界不能作为可部署阈值。",
      disposition: "保留局部历史信号，不称通用 IDS 成功；把评分能力与阈值迁移分开。",
      sources: [
        { label: "独立采集协议", href: "../INDEPENDENT_CAPTURE_LAB.md" },
        { label: "第十九轮结果", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261111_第十九轮流历史/实验分析.md" },
        { label: "第二十轮结果", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261111_第二十轮独立采集/实验分析.md" }
      ]
    },
    {
      id: "rolling-threshold", date: "2026-11-11", dateLabel: "2026.11.11", rounds: "R21",
      categories: ["history-transfer"], status: "completed",
      title: "校准尾部来源被定位，无标签滚动却不稳定",
      purpose: "解释集中正常通信如何支配阈值，再检查滚动参考在新场景的表现。",
      scope: "旧 CIC2018 回放与新 IoT-23 场景分开报告",
      positive: "旧校准 95 个 FP 中 88 个及全部 15 个边界同分记录来自同一源。开发回放的 Bot 0→858，正常 FP 为 298/46,817。",
      limitation: "新 49-1 扫描 819→17,051，却把 FP 13→497/3,665（13.56%）；C&C 仍全漏，参考池几乎全是攻击。",
      disposition: "关闭近期无标签分数能稳定校准正常尾部的具体方案；保留旧回放改善，不冒充新场景验证。",
      sources: [{ label: "前向阈值最终结果", href: "../FORWARD_THRESHOLD_RESULTS.md" }, { label: "逐流证据回放", href: "index.html#calibration" }]
    },
    {
      id: "endpoint-observation", date: "2026-11-13", dateLabel: "2026.11.12 — 11.13", rounds: "R22–23",
      categories: ["history-transfer", "cache-input"], status: "completed",
      title: "端点历史局部有益，回包门控被直接反例推翻",
      purpose: "检验端点应答历史、当前回包状态和质量编码。",
      scope: "IoT-23、IoT-SFC、TQH-C2；原模型与追加／门控分开",
      positive: "部分 MLP 在 52-1 大幅增加扫描检出、误报不变。原八字段 HGB 在 TQH 的 36 单元全合规，检出 11,627/14,326 条 C2。",
      limitation: "门控挡掉本可检出的 4 条纯 C&C 和 5 条下载；R23 新历史两臂在完整 48 单元的 27,130 条 C2 均未检出。零计数不等于网络真的无回包。",
      disposition: "关闭硬门控与通用追加主张，保留局部收益和观测修正；历史模型全漏不证明原字段没有控制信息。",
      sources: [
        { label: "端点证据研究", href: "../CONTACT_EVIDENCE_RESEARCH.md" },
        { label: "第二十二轮实际结果", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261112_第二十二轮端点历史/实验分析.md" },
        { label: "第二十三轮实际结果", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261113_第二十三轮回包状态/实验分析.md" }
      ]
    },
    {
      id: "port-shortcut", date: "2026-11-14", dateLabel: "2026.11.14", rounds: "R24",
      categories: ["history-transfer", "cache-input"], status: "completed",
      title: "端口捷径有干预证据，下载标签也有循环依据",
      purpose: "对冻结八字段模型只改变端口，检查误判与标签来源。",
      scope: "43-1；同一模型、其他七字段和阈值不变",
      positive: "将正常目的端口 23 改为 52869，原 HGB 误报全部消失。MLP29 的同场追加检出 37,191,051/37,911,674 扫描，FPR 0.42268%。",
      limitation: "扫描与下载作者标签分别被端口／状态、响应字节阈值覆盖，且缺同服务正常反例；另两 seed 的扫描召回仅约 1%。",
      disposition: "保留直接干预和同场收益，关闭通用迁移主张；15/15 已满检下载应讨论保持，不能强求再增 TP。",
      sources: [
        { label: "固定八字段分析入口", href: "../RAW_WORKPOINT_LAB.md" },
        { label: "第二十四轮结果及标签核查", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261114_第二十四轮八字段漏检/实验分析.md" }
      ]
    },
    {
      id: "simple-counts", date: "2026-11-15", dateLabel: "2026.11.15", rounds: "R25 与就绪修复",
      categories: ["history-transfer"], status: "completed",
      title: "简单计数复现了扫描收益，稀有目标仍未解决",
      purpose: "比较简单规则与固定模型，并修复参考数量不足时的阈值登记。",
      scope: "SFC 两个新扫描事件；DEDALE 全日与启动阶段均保留",
      positive: "简单端口规则检出 33,385/33,414 和 1,056,102/1,056,115，两日 FPR 均低于 1%。就绪逻辑从只等一小时改为还需至少 399 条严格先前参考。",
      limitation: "修复后，DEDALE 后续 42 条控制、14 条下载仍无登记策略在合规单元检出；0 FP／0 TP 的空运行不能算有效控制风险。",
      disposition: "保留简单扫描强基线，关闭默认跨来源组合；不靠继续加参考条数或只报启动后掩盖失败。",
      sources: [
        { label: "修复后逐日证据", href: "index.html#latest" },
        { label: "回放来源与导出方法", href: "r25-sources.md" },
        { label: "就绪修复完整分析", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261115_第二十五轮就绪修复/实验分析.md" }
      ]
    },
    {
      id: "representative-supervision", date: "2026-11-15", dateLabel: "2026.11.15", rounds: "R26",
      categories: ["model-reset", "history-transfer"], status: "completed",
      title: "目标监督立即改变检出，混合来源却并非自动有益",
      purpose: "用目标历史监督检验旧八字段模型的能力缺口。",
      scope: "新增 10,502 正常＋180 控制；B 与 D 共用攻击实现，无下载正例",
      positive: "端口置零的 target HGB 和三个 MLP 均检出 B72/72、三单元全合规。mixed HGB 在 D 检出 1,952/1,964。",
      limitation: "D 没有方案全合规；mixed MLP 仅检出 45／46／45，mixed HGB 也仅 2/12 单元合规。样本量、权重、更新数等共同变化。",
      disposition: "保留代表性监督的实质收益，关闭当前 mixed MLP；不能合称混合无用，也不能归因为纯来源多样性。",
      sources: [
        { label: "支持诊断协议", href: "../SUPPORT_DIAGNOSTIC_LAB.md" },
        { label: "第二十六轮实际结果", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261115_第二十六轮历史监督/实验分析.md" }
      ]
    },
    {
      id: "ait-event-evidence", date: "2026-11-16", dateLabel: "2026.11.16", rounds: "R27",
      categories: ["independent-sources"], status: "analysis",
      title: "控制动作和正常业务有了实际连接证据",
      purpose: "从作者日志和原包关联 AIT 控制、下载及同网站正常动作。",
      scope: "动作、连接与整链分开；所有唯一、多义候选保留；没有模型效果评价",
      positive: "32 条 WebShell 请求对应 33 个连接候选；68 次正常网站动作对应 294 个候选；两次下载也收敛到明确候选。",
      limitation: "一条控制连接长达 2,229 秒，完整统计不能倒用于命令起始时刻；约 50 万字节 JPEG 回显可能成为实现特征。",
      disposition: "保留多候选和证据等级，不要求 TLS 明文或一事件一流；正常文件与测量合同随后继续补齐。",
      sources: [
        { label: "独立来源与方法决定", href: "../SOURCE_METHOD_ROUTE_20261118.md" },
        { label: "第二十七轮取证分析", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261116_第二十七轮独立事件取证/实验分析.md" }
      ]
    },
    {
      id: "training-support-audit", date: "2026-11-16", dateLabel: "2026.11.16", rounds: "R28 诊断",
      categories: ["model-reset"], status: "analysis",
      title: "训练支持存在冲突，完整权重诊断未被补造",
      purpose: "区分输入冲突、训练损失与跨场景判决。",
      scope: "已完成输入、轨迹与缓存分析；八权重最终支持评分未核得完整回传",
      positive: "端口置零后旧 source 有 63 组异标签同输入，涉及 3,657 条记录；已有 MLP 几乎相同的训练损失对应不同 D 检出。",
      limitation: "这些精确冲突不覆盖 B72、D1964 攻击，不能直接解释 D 漏检；非零 loss 也不是继续长训的依据。",
      disposition: "只保留已经完成的诊断，不将未回传评分记作成功或失败，也不据此自动补跑。",
      sources: [{ label: "诊断结果及状态", href: "../TRAINING_DIAGNOSIS_DECISION_20261116.md" }, { label: "完成状态定义", href: "../TRAINING_DIAGNOSIS_LAB.md" }]
    },
    {
      id: "ple-model-reset", date: "2026-11-17", dateLabel: "2026.11.17", rounds: "R29",
      categories: ["model-reset"], status: "completed",
      title: "PLE 有明确增益，验证与交付仍未覆盖迁移目标",
      purpose: "比较 HGB、TabM 与有／无 PLE 的固定训练流程。",
      scope: "七条 F/V 轨迹及七次 P 重训；所有 seed 保留",
      positive: "全部新模型 B72/72。有 PLE 的 D 检出为 1,652／1,560／1,657，相对同 seed 无 PLE 增加 1,607／1,560／1,657，未丢对照 TP。",
      limitation: "相对旧强参照仍有丢失且多单元超预算；V 缺所要求的 TCP 正例迁移，P 重训又改变交付函数。",
      disposition: "保留 PLE 消融收益，关闭这套追加轮数／种子方案；不将跨轮多项修改归因于单个组件。",
      sources: [
        { label: "模型与训练设计审查", href: "../MODEL_TRAINING_DESIGN_REVIEW_20261116.md" },
        { label: "第二十九轮完整复核", path: "E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261117_第二十九轮模型训练/实验分析.md" }
      ]
    },
    {
      id: "all-attacks-risk", date: "2026-11-17", dateLabel: "2026.11.17", rounds: "R30",
      categories: ["model-reset"], status: "completed",
      title: "攻击全检是真进展，正常误报仍未达标",
      purpose: "共同调整监督、分层批次、排序选轮和直接交付，比较保留／屏蔽协议。",
      scope: "三个 protocol-blind seed；同一 B72／D1964；十五个正常单元",
      positive: "三 seed 均检出 B72/72、D1,964/1,964，没有丢原 TP；证明该模型／输入确实可以取得很强的攻击区分。",
      limitation: "误报合规只有 8／5／4 个单元，总 FP 154／175／202。保留协议臂已检出 D1,962，屏蔽协议的直接配对增量仅 2 条。",
      disposition: "保留全部正向效果与六模型，采用仍失败；不挑 seed，也不把数百条跨轮增益全部归给去协议。",
      sources: [{ label: "第三十轮逐项结果", href: "../MODEL_TRAINING_REDESIGN_RESULTS_20261117.md" }]
    },
    {
      id: "subgroup-objective", date: "2026-11-18", dateLabel: "2026.11.18", rounds: "R31",
      categories: ["model-reset"], status: "completed",
      title: "代理指标变好，没有形成可用模型更新",
      purpose: "尝试降低正常子群集中度，同时保护攻击身份与排序。",
      scope: "六条轨迹、固定验证与回退合同；最终均回到原 R30 工件",
      positive: "运输臂 48 个验证点的 H 全部改善，目标确实作用于模型；由此定位到代理与最终风险对象的错位。",
      limitation: "没有点同时通过入选条件，最终 TP／FP 增失全为 0；被拒点缺完整分数，不能补出重新校准后的效果。",
      disposition: "关闭固定 R31 候选，不放宽保护重选或延长训练；工程符合规格不证明联合目标必要或可达。",
      sources: [{ label: "R31 最终结果", href: "../NORMAL_SUBGROUP_RESULTS_20261118.md" }, { label: "根因与数学对象", href: "../R31_ROOT_CAUSE_AND_RESOLUTION_20261118.md" }]
    },
    {
      id: "operating-point-reset", date: "2026-11-18", dateLabel: "2026.11.18", rounds: "操作点重设",
      categories: ["model-reset"], status: "analysis",
      title: "冻结分数也有改进空间，取消默认复杂训练",
      purpose: "用固定 C-only NP 参照检验更保守工作点，不再把校准 1% 当作必须花满的预算。",
      scope: "只复用缓存；C5511，α=.01、δ=.05；一基顺位 5457→5469；无模型计算",
      positive: "三 seed 分别移除 34／45／49 个 FP，各自保留 2,036 TP；合规单元从 8／5／4 增至 13／11／6。",
      limitation: "仍无 15/15。已开评价标签的可行区间只作诊断，不能用于部署阈值；标准 NP 也不是新算法。",
      disposition: "固定比较结束，撤下子群／最坏组训练的默认主线；未来完整报告收益和损失，不永久强制所有旧 TP 零损失。",
      sources: [{ label: "操作点重设与边界", href: "../OPERATING_POINT_RESET_20261118.md" }, { label: "固定缓存比较", href: "../operating_point_reset_20261118/comparison.md" }]
    },
    {
      id: "independent-source-assembly", date: "2026-11-19", dateLabel: "2026.11.18 — 11.19", rounds: "独立来源装配",
      categories: ["independent-sources"], status: "analysis",
      title: "真实控制、下载与正常业务接入同一观测合同",
      purpose: "用 OpTC、AIT 与 Unraveled 的作者动作、原包和网络记录建立可消费身份。",
      scope: "完整来源／campaign 不跨角色；关联可多对多；原件与派生测量分别保留",
      positive: "五份 AIT 捕获形成 109,512 条观察。Unraveled 全捕获 47,032 条，目标主机有 12,308 正常和 473 条 C2 连接；正常 ownCloud 下载也已有动作依据。",
      limitation: "473 条连接不是 473 个独立控制实现；新增来源没有已核实恶意下载监督。声明 IP 长度统计不等于恢复完整载荷或真实线速字节。",
      disposition: "保留证据等级、未知标签和全部候选；使用实测可见时刻，不能用未来包倒填控制动作。",
      sources: [{ label: "来源与方法决定", href: "../SOURCE_METHOD_ROUTE_20261118.md" }, { label: "测量合同与划分", href: "../FIT_CONTRACT_20261119.md" }, { label: "当前覆盖结果", href: "../COVERAGE_AND_METHOD_DECISION_20261119.md" }]
    },
    {
      id: "normal-evidence-correction", date: "2026-11-19", dateLabel: "2026.11.19", rounds: "正常证据修正",
      categories: ["independent-sources", "cache-input"], status: "analysis",
      title: "正常身份、输入质量和实验角色分开",
      purpose: "撤销把 SF 与完整活动结束当作正常标签必要条件的额外限制。",
      scope: "russell 整个 run 作候选 C；原 v1 与旧 C25 历史保留",
      positive: "旧 C25 的正常依据全部保留。v2 有 3,475 条合格正常，含 33 次核实下载、188 条文件请求候选连接。",
      limitation: "先前排掉的九条中，六条只是 RSTO/S3，三条是活动未明确结束；这不是正常身份的反证。188 条连接不能冒充独立下载次数。",
      disposition: "按标签、输入、角色三层消费；不再用无依据的额外门槛缩小正常分母，也不把相关流数当总体置信保证。",
      sources: [{ label: "正常来源 v2", href: "../coverage_basis_20261119/normal_calibration/v2/README.md" }, { label: "旧 C25 身份处置", href: "../coverage_basis_20261119/normal_calibration/v2/historical_C25_disposition.json" }]
    },
    {
      id: "withdraw-history-arm", date: "2026-11-19", dateLabel: "2026.11.19", rounds: "HGB8 / HGB27",
      categories: ["history-transfer", "model-reset"], status: "withdrawn",
      title: "历史摘要在拟合前撤下，不等待失败后再解释",
      purpose: "逐事件检查最近槽与全前缀摘要是否包含下载时已可用的相关控制。",
      scope: "已有 5,306 个事件／对象的机制分析；两臂未进行真实拟合",
      positiveLabel: "已完成的依据",
      positive: "OpTC 后 23 条控制确实保留上一控制，重复通信的局部证据存在；检查同时找到了摘要与任务不匹配的位置。",
      limitation: "AIT 四次真正 wget 控制流均未早于下载查询可用；同端点关系又混入代理布局，前缀汇总约四天的 5,275—11,497 条流。",
      disposition: "撤下固定 19 维历史摘要和 HGB27 臂。保留局部证据，不自动替换成窗口、图或蒸馏方案。",
      sources: [{ label: "逐事件机制分析", href: "../coverage_basis_20261119/history_mechanism/README.md" }, { label: "撤下历史臂的决定", href: "../COVERAGE_AND_METHOD_DECISION_20261119.md" }]
    },
    {
      id: "fixed-four-arm", date: "2026-11-19", dateLabel: "2026.11.19", rounds: "固定四臂 · 当前",
      categories: ["model-reset", "independent-sources"], status: "pending",
      title: "只检验监督扩充与端口依赖，四臂结果尚未产生",
      purpose: "同一 HGB 下比较单来源／扩充监督 × 保留／置零端口，主参照预定 mixed_ports_zero。",
      scope: "F_base 471；F_expanded 13,252；C3,475；fox／OpTC 仅开发评价",
      positiveLabel: "已准备的条件",
      positive: "实际输入、来源隔离和固定接口已就绪；先保存全部四模型，再保存全部 C 阈值，最后一次评价，不从结果挑赢家。",
      limitation: "目前没有真实拟合或效果结果。比较包含样本量与来源内容共同变化；控制与下载分报，新增 C2 不能写成下载监督扩充。",
      disposition: "等待实验室固定批次。之后形成基线结论与完整取舍，不因失败自动补数据、调阈值或转向复杂方法。",
      sources: [{ label: "现行方法决定", href: "../COVERAGE_AND_METHOD_DECISION_20261119.md" }, { label: "实验室批次说明", href: "../COVERAGE_BATCH_LAB_20261119.md" }, { label: "冻结注册", href: "../coverage_basis_20261119/batch/registration.json" }]
    }
  ]
};
