# 教师价值与蒸馏任务的根本审查

2026-11-04。阅读本仓源码、v3/v4回传审查及下列原始论文／作者项目；不加载真实joblib、不读取prepared特征、不训练。本报告将“已有证据”“缺失诊断”“未来假设”分开，不以教师更大或年份更新推定知识有效。

## 1. 主要结论：缺少蒸馏必要性的前置证据

目前不能证明教师无用，也不能证明现有KD获得了不可替代的软知识。最初没有先回答：在相同低FPR工作点，教师是否比已训练小MLP发现更多有价值的攻击、是否能纠正学生错误、其错误是否会损害学生已正确检出？这些应先于重新设计蒸馏损失。

一个重要区分：KD学生与hard学生同结构、同字段是必要的公平控制，并非设计错误。HGB大教师→小MLP仍然可以构成模型压缩。但是，hard MLP已经同样小，若KD没有带来额外检出保留，就不能把“小模型本来很小”归功于KD。当前v3/v4主要比较相同 `[32,16]` 全字段学生的更新方式，缺少完整的教师→学生低FPR性能／真实部署成本前沿，也没有检验字段受限条件下的知识迁移。

因此，根本偏差是**先假设蒸馏有价值，再围绕无增益现象补约束**。版本不退化和报警兼容是合理问题，但它们不自动回答原毕设的轻量化收益；若作为主问题，需要独立的应用需求和贡献论证，不能在普通KD负结果后悄悄替换原问题。

## 2. 三个信息环节被混在一起

### 2.1 教师能力没有在正确口径下验证

本仓HGB以balanced sample weight拟合；`ModelBundle.predict_scores`再统一将概率转float32。v3完整F教师的V诊断还曾以float32数组比较float64阈值，CIC FPR=1是已知诊断问题，不能用它证明教师差。OOF训练目标是F上的折外预测，而完整教师只拟合F、用于V诊断；这些都不等于已获得“教师在低FPR上可靠优于学生”的证据。

下一次只读评分应先报告教师原生分数的R校准→V结果、C尾部、每族表现及教师/学生互补。教师总体不优不必然排除KD价值——错误互补或正则化仍可能有效——但微小平均提升更不能反推存在可迁移知识。

### 2.2 分数处理主动抹去了信息并改了监督含义

v3先把float32概率clip至 `[1e-6,1-1e-6]` 再logit；CIC大量正常分数在clip下界重合。随后减去教师R报警阈值取sigmoid，令约99.63%正常目标变为0.5。它们是报警margin目标，不是自然类别后验。把它们送入逐行BCE会把原来深负的正常余量拉向边界，这与希望低误报的目标存在结构性张力。

raw KL恢复普通预测分数拟合，可以消除这个特定的中心化目标；其绝对分数在本任务上能否有效迁移仍待证明。它也不能恢复已经被clip删去的教师置信度差异。原生margin、遗留clip-logit、中心化alarm target必须分别诊断，不能先把raw方案当正确答案。

### 2.3 巨关联组使实际折教师的家族支持发生变化

封存OOF目标的逐行分析又发现：CIC的139,618条F内Portscan全部归入fold2，它们由同一个未见该家族样本的折教师提供目标。三个seed的raw预测概率中位数为0.00168192／0.00165856／0.02893519，按0.5判定的该族TPR均为0；按教师原R工作点判定却均为1。

这不是数据泄漏，也不等于该教师没有二元攻击正类——attack_cat是诊断家族，训练标签是normal/attack。整组OOF仍合法，但此处实际目标发生了“家族未见支持”的泛化条件变化，完整F参考教师无法替该折证明质量。该例也表明绝对概率拟合与有价值的相对排序不能混为一谈，不能凭raw0.5检出0就说教师无用，也不能凭R工作点检出1就说学生必能学会。依据为 `local/upgrade-v4-return-audit-20261104/oof_family_support_audit.json`，不是本次新增模型查询。

### 2.4 “模仿教师”与“改善部署低FPR目标”没有建立因果桥梁

当前蒸馏是F上全体样本的分数拟合，主任务却是在低FPR尾部保留检出；两者不是相同优化目标。hard与soft首步梯度已明显冲突。重新校准可修复全局平移，却不能恢复丢失的排序。后置guard只能筛掉有限C不合格更新，不能让一个无纠错能力的教师变得有价值。停止空转、保存失败模型、精确阈值注册是必要工程，不是新的知识迁移机制。

## 3. 锁版HGB原生评分API：可直接检查，没有必要重训

本机只检查类源码，`sklearn.__version__ == 1.5.1`。`HistGradientBoostingClassifier.decision_function(X)`公开方法返回 `_raw_predict(X)` 的二元原始值；`predict_proba(X)`由同一原始值经link得到概率。官方1.5在线文档目前显示1.5.2，故用它核API语义、用本机1.5.1源码核具体锁版，二者不混称同一版本。[官方API](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier.decision_function)

可执行路线：在同一个已拟合的 `bundle.preprocessor.transform(frame)` 上调用 `bundle.estimator.decision_function`，转float64并校验形状/有限性/`classes_ == [0,1]`；同时调用原生 `predict_proba`，检查其正类概率与 `expit(native_margin)` 一致。遗留分支严格复现 `predict_proba[:,1] -> float32 -> clip(float32) -> scipy.special.logit(float32) -> float64`，不能把float64 clip近似称原实现。两种分数各自在R定自己的阈值，再在V评价，不借用MLP旧阈值。

这只查询 `teachers/s{seed}/reference.joblib`，不需重拟合，也不改变旧OOF目标。完整教师仅用F拟合，学生有F+C硬标签，训练监督量不同；结果是能力／互补诊断，不能当公平方法排行榜。完整教师V能力也不等于三折OOF目标的全部行为。

本项目F约占原train的70%，三折中每个实际供给目标的教师只用约46.7%原train拟合，学生hard使用F+C约90%。各折真实行数应从oof.json记录，而非把软比例当精确计数。完整F参考教师的native V能力可以显示某类纠错空间，却不能替实际使用的折教师证明同样能力；本次仍不额外查询各折native分数，另存现有OOF F-heldout legacy质量及各类target分布以标明这一限制。

## 4. 本次核对的2025/2026原始成果

| 成果与身份 | 本次实际读取及机制 | 对本项目的含义与限制 |
|---|---|---|
| **A Good Teacher Adapts Their Knowledge for Distillation，ICCV2025**，[正式页](https://openaccess.thecvf.com/content/ICCV2025/html/Qian_A_Good_Teacher_Adapts_Their_Knowledge_for_Distillation_ICCV_2025_paper.html) | CVF正式记录与官方PDF搜索索引中的方法／Algorithm1：先用已训练学生引导微调神经教师，再蒸馏给新学生；关注类内分布不匹配。直接PDF打开403，本次未核到作者实现入口，不称完整源码复现。 | “教师准确率高就一定适合学生”不是成立前提。论文为多类图像神经模型，HGB不能直接按其反传微调；不能将二元rawKL改名为AID。 |
| **What Knowledge Transfers in Tabular Anomaly Detection? A Teacher–Student Distillation Analysis，MAKE2026**，[出版方](https://www.mdpi.com/2504-4990/8/3/60) | 出版方索引中的§3/§6及[作者MIT源码](https://github.com/redline-tk/KD-AnomalyNet/tree/1518363d8cf5cc8c62609735254fee766afbc576)：异构无监督异常教师、score/reconstruction双目标，按异常类型分析迁移；源码 `losses.py`、`teacher.py` 已读。 | 研究“什么知识能迁移”已有直接先例。其代码对每次查询的分数重新做百分位/min-max归一，不能直接拿作固定R→V评分器；不是1%FPR二元IDS对照，不搬用其排名／理论口号。 |
| **Multimodal Tabular Reasoning with Privileged Structured Information，NeurIPS2025主会**，[正式记录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/99e1098e58285e076d10e81281fdbb40-Abstract-Conference.html)／[原文](https://proceedings.neurips.cc/paper_files/paper/2025/file/99e1098e58285e076d10e81281fdbb40-Paper-Conference.pdf) | 已读§3–4：训练时结构化表格辅助推理轨迹生成，部署只给表图像；SFT后迭代优化。未核到公开作者代码入口。 | 借鉴的是先明确训练／部署信息差。它是MLLM表图像推理，不是普通表格分类或本项目直接baseline；不能因名称有tabular/privileged就声称同任务复现。 |

KD-AnomalyNet代码固定commit为 `1518363d8cf5cc8c62609735254fee766afbc576`（2026-03-05）；已核MIT许可，未安装／执行。作者仓库只是公开工件，不作为论文性能已复现的证据。以上论文不提供本项目新颖性或效果背书。

## 5. 无训练“教师必要性”诊断应输出什么

现有阈值审计只有MLP目录，不能独立回答教师价值。应另增每seed的 `teacher-value` 诊断，原MLP catalog/计数保持原义。

- 身份：完整教师模型SHA、来源v3完成标记、F-only监督范围、balanced权重事实；学生的F+C预算同时标出。
- 分数：native与legacy clip两套R/C/V数组；逐类分位数、unique数、最大tie比例、clip上下界质量；不得拿不同单位的阈值直接比较大小。
- 工作点：每套分数只在R校阈值，记录R允许/实际FP，再在V记TP/FP、FPR/TPR及每族。C事件只描述，不凭V挑选native或clip。
- 排序：ROC-AUC及 `roc_auc_score(max_fpr=alpha)` 的McClish标准化partial AUC，明确并非raw区间面积；它不替代固定工作点预算。
- 错误互补：对每个既有MLP的R-only判定，分别统计教师纠正漏检、损害原正确检出、修复FP和新增FP；按族分解。用保存的MLP scores.npz复算，不重复forward、不把oracle错误互补当可学习的组合模型。
- 成本：额外查询耗时、保存字节、原教师包大小，和正式重复CPU部署benchmark分开；不以本次审计forward耗时称完整部署加速比。

该诊断只提供是否继续KD主线的证据门槛。若没有稳定纠错空间，就应先改变教师或问题定义；若有纠错空间但KD拿不到，才研究容量／信息限制／目标一致性；若只通过阈值变更改善，则将贡献归于工作点注册。

## 6. 重构研究问题的优先顺序

1. 先写清部署预算：字段数量/可获得性、模型包、参数与CPU时延，以及允许的FPR。以同预算hard学生作必要参照，确认它留下什么真实缺口。
2. 在这个缺口上检查教师的正确性、互补性和学生可表达性。完整字段教师→少字段学生可形成训练期额外信息，但必须证明学生观察范围确实受限、被删信息不能部署、代价有依据；LUPI概念本身已有，不能当新颖性。
3. 只对可迁移信息设计一个可证伪机制。若教师差别来自学生不可观测因素，单纯提高KD权重无法消除信息约束；若教师只在总体指标更好而低FPR尾部无优势，先做直接低FPR控制。
4. 回归保护留为有外部需求时的独立约束或应用要求。它不是给无增益KD附加创新性的装饰；小样本家族保护、重复组与未知分布也不能用训练经验界替代总体保证。

当前值得保留的是低FPR部署目标、预算明确的小学生、完整硬标签控制和可追溯工件。应暂时撤回“现有KD/guard组合构成足够新颖的轻量化算法”这一主张，先把研究问题、教师有效信息和真实部署收益连成证据链。
