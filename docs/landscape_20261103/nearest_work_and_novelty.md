# 近邻工作与创新性红队审查

核查日期：2026-11-03。范围：低 FPR、Neyman-Pearson（NP）、pAUC、工作点与置信蒸馏、表格基础模型蒸馏、预算路由与特征获取、分布迁移及校准。仅使用论文原文、正式会议／期刊页面、作者页面与作者代码。本文件不记录新实验结果；未训练模型、未运行真实数据、未修改源代码或配置。

**审查结论：目前不能把“风险尾部迁移／工作点感知 KD”作为已成立的新方法贡献。** CVPR 2022 EKD 已将目标 FPR、负类分位数、关键关系选择与容量友好的排序蒸馏结合；ICLR 2026 AdaConG 已用校准不确定度加权 KD；2026 年 TFM 蒸馏工作已覆盖 OOF、置信权重、合成查询及二阶段锚定。低 FPR 迁移、预算级联也分别存在直接理论和安全应用近邻。仅更换教师、叠加损失、加上 group 权重或新缩写，都不足以解决创新性问题。

本文标签含义：

- **事实**：能定位到一手原文或代码；论文报告的结果仍属于作者报告，不等于本项目已复现。
- **推断**：本次审查对这些事实及当前实现的数学分析。
- **待验证**：建议后续明确证伪的研究假设，不能写进摘要作为已完成贡献。
- **精读**：已读问题定义、方法公式、算法／假设、主要实验与局限；不表示逐行复核了全部附录证明。

## 1. 16 项代表工作及其覆盖范围

### R01. When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee

Dixian Zhu、Gang Li、Bokun Wang、Xiaodong Wu、Tianbao Yang；**ICML 2022 主会，旧基础工作，精读**。[正式页面](https://proceedings.mlr.press/v162/zhu22g.html)，[全文](https://proceedings.mlr.press/v162/zhu22g/zhu22g.pdf)，[作者团队 LibAUC 实现](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/losses/auc.py)。

事实：通过 CVaR-DRO 或 KL-DRO 构造 OPAUC／TPAUC 目标及随机优化算法。前者对应文中 pAUC surrogate 的精确选择结构，后者平滑但有近似。推断：本项目若只是把普通 pairwise loss 改成 CVaR-pAUC，应表述为采用成熟目标；它是必要强对照，不是新增算法。代码入口已核到 `pAUC_CVaR_Loss`，本轮未运行。

### R02. Asymptotically Unbiased Instance-wise Regularized Partial AUC Optimization: Theory and Algorithm

HuiYang Shao、Qianqian Xu、Zhiyong Yang、Shilong Bao、Qingming Huang；**NeurIPS 2022 主会，旧基础工作，精读**。[正式页面及全文入口](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fc9f83d9925e6885e8f1ae1e17b3c44b-Abstract-Conference.html)，[作者 PAUCI 代码](https://github.com/Shaocr/PAUCI/tree/c08d633d195a9ab1799b283696b02d215b7d29b9)。

事实：将 pAUC 转为 instance-wise 正则化 min-max 优化，使用可控渐近近似。推断：其价值是避免显式全配对及分析近似误差；不能把一个 minibatch hard-pair loss 改名为 PAUCI。官方代码存在，未复现。

### R03. Closing the Approximation Gap of Partial AUC Optimization: A Tale of Two Formulations

Yangbangyan Jiang、Qianqian Xu、Huiyang Shao、Zhiyong Yang、Shilong Bao、Xiaochun Cao、Qingming Huang；**TPAMI 48(3):3482–3498，2026，期刊，精读**；DOI `10.1109/TPAMI.2025.3637063`。[作者全文](https://arxiv.org/html/2512.01213v1)，[DOI](https://doi.org/10.1109/TPAMI.2025.3637063)。

事实：§I 与参考文献 [22] 明确前身为 R02；§VII-C 将 PAUCI 标为 NeurIPS 版本、UPAUCI 标为期刊版本。新形式以更多优化变量处理非光滑选择，区分选择近似与正则化。推断：不能把 2022 PAUCI 仓库当作 2026 UPAUCI 已复现代码；本轮未核得独立新版实现。

这里不能混入另一篇 **Large-scale Optimization of Partial AUC in a Range of False Positive Rates**（Yao Yao、Qihang Lin、Tianbao Yang，NeurIPS 2022，DC／Moreau 路线）；它不是 R03 的前身。[另一篇的正式页面](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ca7998666c2e53cc1e882b7268414d8a-Abstract-Conference.html)

### R04. Neyman-Pearson classification: parametrics and sample size requirement

Xin Tong、Lucy Xia、Jiacheng Wang、Yang Feng；**JMLR 21(12):1–48，2020，旧统计基础，精读阈值与样本量部分**。[正式论文及 R 包入口](https://jmlr.org/papers/v21/18-577.html)，[全文](https://jmlr.org/papers/volume21/18-577/18-577.pdf)。

事实：区分 NP 高概率 Type-I 控制、非参数 umbrella 阈值和带模型假设的参数方法。推断：校准集上恰好 1% 误报不等于总体 FPR≤1% 的高概率保证；更不能把相关 flow 数直接当成独立校准样本数。

### R05. Evaluation-Oriented Knowledge Distillation for Deep Face Recognition（EKD）

Yuge Huang、Jiaxiang Wu、Xingkun Xu、Shouhong Ding；**CVPR 2022 主会，旧但最直接的创新性近邻，精读＋代码核查**。[正式身份](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Evaluation-Oriented_Knowledge_Distillation_for_Deep_Face_Recognition_CVPR_2022_paper.html)，[作者全文](https://arxiv.org/html/2206.02325)，[作者代码](https://github.com/Tencent/TFace/tree/master/recognition/tasks/ekd)。

事实：§3 在指定 FPR 下分别估计师生负关系分位数阈值，选择造成 TPR／FPR 分歧的关系；用相对阈值的平滑排序约束替代绝对特征／相似度匹配，以适应容量较小的学生。已有 hard-negative 与 random-negative 对照。推断：本项目“工作点＋关键尾部样本＋容量友好 KD”概念已被覆盖，必须加入适配版 EKD 对照。人脸关系与 IDS 单行分数不同，是适配工作，不自动构成算法创新。

### R06. Trust the Right Teacher: Quality-Aware Self-Distillation for GUI Grounding

Jingyuan Huang、Zuming Huang、Yucheng Shi、Tianze Yang、Xiaoming Zhai、Wei Chu、Ninghao Liu；**2026-06-16，arXiv:2606.18101v1，预印本，精读方法与消融**。[作者全文](https://arxiv.org/html/2606.18101v1)。

事实：将 soft correctness-aware gate 与 teacher-probability scaling 结合，分别抑制错误监督、调节剩余监督强度，并比较单项与组合。推断：correctness×confidence 的一般组合不能声称首创。其 GUI 坐标 token、privileged context 和 reverse-KL 与本项目不同，不能冒称同一实现或已复现。未核得可运行作者代码。

### R07. Adaptive Conformal Guidance for Learning under Uncertainty（AdaConG）

Rui Liu、Peng Gao、Yu Shen、Ming Lin、Pratap Tokekar；**ICLR 2026 主会，精读**。[正式身份](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0b08d733a5d45a547344c4e9d88bb8bc-Abstract-Conference.html)，[正式全文](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b08d733a5d45a547344c4e9d88bb8bc-Paper-Conference.pdf)。

事实：§3 使用 split CP 预测集合大小构造不确定度，再加权 guidance loss；§4 含噪声／迁移下的 KD 实验。推断：“用校准不确定度决定是否相信教师”已有近期主会近邻。CP 标签覆盖并不等于加权 KD 无负迁移，更不等于学生 class-conditional FPR 控制。本次未核得作者实现入口，不能称已复现。

### R08. Distillation of Tabular Foundation Models into Efficient Predictors

Minho Jeong、Dooho Lee、Jinmo Lee、Jaemin Yoo；**2026-10-01，arXiv:2610.01435v1，预印本，精读监督构造与实验协议**。[作者全文](https://arxiv.org/html/2610.01435v1)。

事实：直接研究 TFM→TabM／XGBoost，比较 full-context（FC）、OOF、硬软监督混合及额外合成查询；其设计集最优配置是 FC＋teacher-only。推断：基础模型教师、轻学生、合成查询、移除部署教师不是空白。其[作者公布代码](https://github.com/nums-ai/TFM_Distillation)本次读取返回 404；没有可运行证据，也未核到主会录用身份。

### R09. Distilling Tabular Foundation Models for Structured Health Data

Aditya Tanna、Nassim Bouarour、Mohamed Bouadi、Vinay Kumar Sankarapu、Pratinav Seth；**2026-05-18，arXiv:2605.18702v1；作者页面标为 ICML 2026 SD4H workshop，非主会，精读**。[全文](https://arxiv.org/html/2605.18702v1)，[作者发表页面](https://pratinavseth.github.io/publications/distilling-tfms-structured-health-data/)。

事实：§3 已有 5 折 OOF 教师标签、hard＋soft loss、逐样本 adaptive temperature 与中等熵处较高的 confidence weight；涵盖 MLP 和树学生。推断：OOF＋TFM＋置信加权同样非新 pipeline。正文实验是小表健康任务，不能直接推出百万行 IDS 或 1% FPR 时有效。作者称依托 TabTune，但本轮未确认对应版本能完整复现。

### R10. GEAR: Generative Expansion and Real Anchoring for Two-Stage Distillation of Tabular Foundation Models

Qi Qin、Jiajie Zhu、Dali Chen、Yuzhao Zhang、Jia-Xing Han、Yu Su、Peng Zhang、Ying Yan、Yifan Sun；**2026-08-19，arXiv:2608.18849v1，预印本，精读**。[作者全文](https://arxiv.org/html/2608.18849v1)。

事实：§2 用合成协变量查询冻结 TFM，再以真实标签＋OOF 教师预测锚定；§3 给生成量与保真度的风险证书。推断：“合成扩充＋OOF＋两阶段＋风险界”已有直接近邻。证书依赖 surrogate-to-target 偏差及分布差异等条件，不等于真实总体风险一定降低，更不是低 FPR 或小学生优势证明。本轮未核得作者代码入口。

### R11. Neyman-Pearson Classification under Both Null and Alternative Distributions Shift

Mohammadreza M. Kalan、Yuyang Deng、Eitan J. Neugut、Samory Kpotufe；**ICLR 2026 主会**；预印本首发 2025-11-10，arXiv:2511.06641；**精读目标、迁移程序和主要保证**。[正式身份](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33870b3e099880cd8e705cd07173ac27-Abstract-Conference.html)，[正式全文](https://proceedings.iclr.cc/paper_files/paper/2026/file/33870b3e099880cd8e705cd07173ac27-Paper-Conference.pdf)，[作者 HTML](https://arxiv.org/html/2511.06641)。

事实：同时允许正常与攻击类条件分布迁移，利用有标签 source／target 数据自适应迁移，分析 source 无用时避免负迁移的条件。推断：“首次在正常分布漂移时迁移低 FPR 能力”不能成立。本文不是 TFM→固定小 MLP 的 KD，亦不覆盖任意相关 flow 的无条件保证。本次未核得作者代码入口。

### R12. A Unified Approach to Routing and Cascading for LLMs

Jasper Dekoninck、Maximilian Baader、Martin Vechev；**ICML 2025 主会**。[正式身份](https://proceedings.mlr.press/v267/dekoninck25a.html)，[作者代码](https://github.com/eth-sri/cascade-routing)，[作者 HTML](https://arxiv.org/html/2410.10347)。

事实：统一预算内质量最大化的 routing／cascading，识别质量估计的重要性。推断：按升级收益／成本路由属于已有决策框架；不能仅换成 IDS 就称新算法。本次公式阅读使用作者公开 **v5（2026-10-26）**；不推断该版本所有新增表述均与 2025 会议版相同。源码入口已核，未运行。

### R13. Active feature acquisition via explainability-driven ranking

Osman Berke Guney、Ketan Suhaas Saichandran、Karim Elzokm、Ziming Zhang、Vijaya B Kolachalama；**ICML 2025 主会；已核摘要、正式身份与作者 README，未完成全文逐公式核查**。[正式论文](https://proceedings.mlr.press/v267/guney25a.html)，[作者代码](https://github.com/vkola-lab/icml2025)。

事实：局部解释产生逐样本特征重要性顺序，decision-transformer policy 顺序获取特征。推断：动态字段预算、逐样本选择特征、用强模型解释指导小策略均非空白；是否能同时满足 IDS 正常类风险与真实采集成本，仍需另行建模。代码可见，未跑通。

### R14. Wasserstein-Regularized Conformal Prediction under General Distribution Shift（WR-CP）

Rui Xu、Chao Chen、Yue Sun、Parvathinathan Venkitasubramaniam、Sihong Xie；**ICLR 2025 主会，精读界、假设与迁移设定**。[正式身份](https://proceedings.iclr.cc/paper_files/paper/2025/hash/283066055b0256ca8e3e0c8c96019357-Abstract-Conference.html)，[正式全文](https://proceedings.iclr.cc/paper_files/paper/2025/file/283066055b0256ca8e3e0c8c96019357-Paper-Conference.pdf)，[作者代码](https://github.com/rxu0112/WR-CP)。

事实：分析联合分布变化下的 conformal score 覆盖差距，区分 covariate／concept shift；应用包括多个 source 的未知混合。推断：分布漂移后校准仍可靠不能直接继承 IID CP；该文也不是任意未知新攻击／相关 flow 的免费保证。代码 README 已核，未复现。

### R15. SAFE-KD: Risk-Controlled Early-Exit Distillation for Vision Backbones

Salim Khazem；**2026-02-03，arXiv:2602.03043v1，预印本，精读风险定义并核作者代码**。[全文](https://arxiv.org/html/2602.03043v1)，[作者仓库](https://github.com/salimkhazem/safe-kd)，[关键代码](https://github.com/salimkhazem/safe-kd/blob/main/src/methods/safe_exit.py)。

事实：提出多出口 KD＋conformal early exit，声称控制早退样本的 selective error。推断：组合想法已存在；但所读实现的 CP 分位数与单标签早退规则**不足以实现论文声称的 selective risk 合约**，见 §3.5。不能据此背书有限样本保证。代码可见不等于正确或跑通；本轮未执行。

### R16. C-GUARD: Context-Adaptive Conformal Gating for Improving Robustness Against Evasive Windows PE Malware

Muhammad Imran、Malik Al-Essa、Felice Franchini、Giuseppe Pirlo；**Future Internet 18(8):431，2026-08-13，期刊，非顶会；精读公开方法与低 FPR 评估部分**。[期刊原文](https://www.mdpi.com/1999-5903/18/8/431)，[版本记录](https://www.mdpi.com/1999-5903/18/8/431/notes)。

事实：OOF 构造 FN-vs-TN rescue，conformal gate 配合新增 FP 预算，只翻转 baseline-negative；另有 validation FP-matched 的单阈值放宽对照。推断：安全任务中的“困难漏报保护＋条件辅助模型＋误报预算”已有直接应用近邻。它报告的是冻结阈值的实际表现，不是总体 FPR≤1% 证明；实验工件需向作者申请，本轮未核得公开实现。

## 2. 哪些候选主张已被覆盖

| 候选说法 | 最强近邻 | 本次判断 |
|---|---|---|
| 目标 FPR 下选择关键样本、相对阈值做排序蒸馏 | R05 EKD | 概念与主要机制已覆盖；固定 0.01、用于 IDS 不足以宣称首创 |
| 正确性与置信度共同门控 KD | R06；R07；R09 | 已有一般机制与近期实例；本项目可以报告有控制实验的适配收益 |
| TFM 教师、OOF、CPU 小学生 | R08–R10 | pipeline 已覆盖；“使用最新模型”是对照质量，不是算法创新 |
| 合成查询、真实锚定、二阶段及风险界 | R10；R08 | 已有直接近邻；不能再以堆生成器作为主要差异 |
| 正常／攻击分布都变化时保持低 Type-I 风险 | R11；R14；R04 | 已有理论主干；本项目尚缺与具体样本依赖、容量约束的可验证连接 |
| 预算约束路由、按困难样本调用强模型 | R12；R15；R16 | 已覆盖一般框架与安全应用；需要真实成本、整体风险、强策略对照 |
| 每个样本动态获取字段 | R13 | 属成熟 AFA 问题；字段数量不能替代采集成本 |
| selected-group 等权就是公平、相关性处理或漂移稳健 | 当前实现的推断，见 §4 | 不成立；这些是不同 estimand、统计假设和约束 |

“未发现一篇同时写完本项目全部组件”的事实不能证明创新。应先排除：成熟方法适配是否已解释全部效果，以及所谓组合收益是否仅来自目标修正、权重变化、额外计算或选择偏差。

## 3. 关键全文与代码精读

### 3.1 EKD 已有“工作点感知迁移”的核心结构

R05 §3.1 的阈值由负关系分位数获得，分别记为 $t_k^T,t_k^S$；关键关系满足

$$
\mathbb{1}\{s_T>t_k^T\}\ne\mathbb{1}\{s_S>t_k^S\}.
$$

§3.2 不要求师生相似度完全相同，而比较相对阈值的平滑排序计数：

$$
\mathcal L_{\mathrm{EKD}}
=\frac1N\sum_n
\left\|\sum_k \sigma((s_{T,n}-t_k^T)/\tau)
-\sum_k \sigma((s_{S,n}-t_k^S)/\tau)\right\|.
$$

事实：作者代码 `ekd.py`（blob `b5ac88c252801a9f708c63ecca965cb31cc57d19`）使用师生各自 EMA anchors、温度 0.01、学生最难负关系 top-2000，最终是排序计数的 MSE。推断：论文的概念选择说明与实际“全部正关系＋hard negatives 上平滑计算”应分开；不能只照摘要写成显式只更新离散分歧集。适配单行 IDS 分数时，应公开关系→行分数的变换、阈值估计和梯度处理，称为 adapted EKD，不能声称原任务复现。[论文 §3](https://arxiv.org/html/2206.02325)，[关键代码](https://github.com/Tencent/TFace/blob/master/recognition/tasks/ekd/distillation/ekd.py)

### 3.2 CVaR／PAUCI 的“精确”究竟指什么

R01 定理 1 的 OPAUC surrogate 可写为

$$
\min_{\theta,u\in\mathbb R^{n_+}}
\frac1{n_+}\sum_{i=1}^{n_+}
\left[u_i+\frac1{\beta n_-}\sum_{j=1}^{n_-}
\big(\ell(f_\theta(x_i^+)-f_\theta(x_j^-))-u_i\big)_+\right].
$$

事实：$u_i$ 是每个正样本的持续状态；SOPA 不是每个 batch 临时挑一批 hardest pairs。推断：应核对样本索引、采样、状态恢复、类条件归一化与 beta；正确实现才对应该目标。“精确”不表示替代损失等于 0-1 pAUC、非凸全局最优，或测试 FPR 自动受控。[R01 定理 1、算法 1](https://proceedings.mlr.press/v162/zhu22g/zhu22g.pdf)

R02／R03 还需要 $a,b,\gamma,s,s'$ 等辅助变量；R03 无选择近似的形式增加逐样本选择变量。推断：不能省略它们后沿用其无偏性或收敛结论。R03 主要实验的 FPR 区间较宽，不能由论文表格直接推出 beta=0.01 的数值稳定性。[R03 §IV–V、VII](https://arxiv.org/html/2512.01213v1)

单点召回与区间面积也不同：优化 $\int_0^{0.01}\mathrm{TPR}(v)\,dv$ 不能单独证明 $\mathrm{TPR}(0.01)$ 提升；应共同报告区间指标及冻结阈值的单点表现。

### 3.3 NP 迁移不等于“阈值迁移＋蒸馏”

R11 的基本问题是

$$
\min_{h\in\mathcal H}R_{\varphi,\mu_{1,T}}(h)
\quad\text{s.t.}\quad R_{\varphi,\mu_{0,T}}(h)\le\alpha.
$$

事实：程序利用四类有标签样本 $(0,S),(1,S),(0,T),(1,T)$，先对齐 source 的可行约束集，再利用 source 攻击数据减少 Type-II 风险；主要界含样本误差项。计算保证还要求凸性、梯度和有界性等条件。推断：不能宣称无 target 标签、任意非凸小 MLP、相关 flow、固定 beta=0.01 下都继承这些保证。其 transfer modulus 也不是可以直接拿 teacher confidence 代替的量。[R11 §3–5](https://proceedings.iclr.cc/paper_files/paper/2026/file/33870b3e099880cd8e705cd07173ac27-Paper-Conference.pdf)

R04 的非参数阈值规则基于独立正常校准分数的次序统计量；在相应假设下，给定目标 beta 和违反概率 delta，最小可行正常样本量为

$$
n_0\ge\left\lceil\frac{\log\delta}{\log(1-\beta)}\right\rceil.
$$

beta=0.01、delta=0.05 时为 299；这是最低可行性，远非良好功效或精确估计所需样本量。多组同时保证需分配 delta；把同一会话的上千条 flow 当成上千个独立样本不满足这个推导。[R04 §2、Proposition 1](https://jmlr.org/papers/volume21/18-577/18-577.pdf)

### 3.4 AdaConG 的不确定度与我们所需风险不是一个量

R07 §3 使用代表 target 的独立校准集构造 $C(x)$，再令

$$
u(x)=g(|C(x)|),\qquad w(x)=\exp(-\kappa u(x)),\qquad
\mathcal L=\lambda_t\mathcal L_t+w(x)\lambda_g\mathcal L_g.
$$

事实：分类实例可用集合大小映射；KD 实验以受噪声影响的 CIFAR-100 教师评价。推断：二分类集合大小的分辨率很低；空集合处理也需明确。集合覆盖是边际事件，不是“教师在正常 1% 尾部正确”，更不是“此样本 KD 梯度必然有益”。因此 CP gate 必须与简单 entropy／confidence、correctness、同平均权重对照，而不能将“有校准”当作算法收益的解释。[R07 §3–4](https://proceedings.iclr.cc/paper_files/paper/2026/file/0b08d733a5d45a547344c4e9d88bb8bc-Paper-Conference.pdf)

R14 进一步说明分布变化可破坏覆盖；其界需 score 分布条件，应用亦有 source-mixture 假设。推断：引入 conformal 一词不会消除依赖、未知 concept shift 或校准集不具代表性的问题。[R14 §3–5](https://proceedings.iclr.cc/paper_files/paper/2025/file/283066055b0256ca8e3e0c8c96019357-Paper-Conference.pdf)

### 3.5 SAFE-KD 是需要红队检查的近邻，不能当保证背书

事实：作者 `safe_exit.py` 的校准函数对 $1-p_y(x)$ 取 split-CP 分位数 $q$；推断时以 $\max_c p_c(x)\ge1-q$ 直接输出 argmax。代码 blob 为 `02048f261d8d6ddbd8746929c6de2c004a606800`。[作者实现](https://github.com/salimkhazem/safe-kd/blob/main/src/methods/safe_exit.py)

**审查反例，不是新实验**：设所有样本 max probability 都是 0.6，其中 90% 预测正确、10% 预测错误，则 $p_y$ 分别为 0.6、0.4。delta=0.05、大校准样本下，$q$ 接近 0.6，所有样本满足早退条件，selective error 约 0.10。CP 的集合覆盖仍可能成立，但单标签早退风险不满足 0.05。

因此，所见代码不能推出文中 $\Pr(\mathrm{error}\mid\mathrm{early\ exit})\le\delta$，更不能推出 $\Pr(\mathrm{alarm}\mid Y=0)\le0.01$。此外每个出口的原始接受集与“前面均未接受后”的条件样本集不同。此文可以反证“KD＋风险早退从无人提出”，但其理论与代码需另行修正，不应直接移植或计作可靠风险基线。

### 3.6 TFM 蒸馏文献存在实际协议分歧

事实：R08 §3.1 实际比较 FC 与 OOF，报告其设计集上 FC＋teacher-only 更好；R09 主张 OOF 避免 self-labeling，并有 adaptive temperature／confidence weight。推断：**FC 使用 train 标签不自动构成 test 泄漏**；它可能是一种有效的训练监督。但如果要根据教师“是否正确／有多可信”决定门控，就不能把教师已在上下文见过该行标签的预测当成样本外可靠性证据。我们的 group-OOF 要求应说明针对后一个识别目标，而非宣称所有 FC 方法无效。[R08 §3.1](https://arxiv.org/html/2610.01435v1)，[R09 §3](https://arxiv.org/html/2605.18702v1)

公平的协议对照应至少在一个固定教师／学生组合上比较 **FC 普通 KD 与 OOF 普通 KD**，保持学生训练预算和 query 数量一致，报告 context 体量与类别覆盖（OOF 可用上下文通常更少）。若另控制相同 context 预算，应说明抽样方式。用独立开发分区判断学生泛化，而不是以 FC 教师在已见行上的正确率证明其门控可靠。该对照只用于识别协议效应，不扩成所有教师／损失的全组合。

R10 的生成保真度项提示更多查询不保证更多真实信息。推断：不能把不可能的网络字段组合当作有效困难攻击；生成查询量和上下文成本必须计入资源预算。[R10 §2–3](https://arxiv.org/html/2608.18849v1)

## 4. 对当前实现的数学定位

本节是当前代码的只读事实与审查推断，不更改现有 pilot。

- `src/ids/research_kd.py:229` 的 correctness 用 0.5 分界，confidence 为 $|2q-1|$；这与正常 99% 分位点没有直接关系。
- `src/ids/research_kd.py:145` 选择 batch 内最高分 normal 的 10%，再受 32 个 normal、64 个 attack、2048 对上限影响；attack 又按最低分选取。实际尾部比例随 batch 组成及上限变化。
- `src/ids/research_kd.py:124` 对**已选样本内** group 赋 $1/(G n_g)$ 权重；两类权重乘积用于 pair loss。它表示选中组等权，不能代替 population per-flow 风险、完整 group-DRO 或相关样本统计处理。
- `configs/research-v2.yaml:5` 的评价目标是 0.01；与训练 10% capped mining 的错配真实存在。把 normal_fraction 改成 0.01 也不足以自动获得全分布分位数或 R01 的目标。
- `gated_matched_hard` 已控制 gate 选择和有效权重，`correct_only`／`confidence_only` 已拆分机制。这些是有价值的识别控制，应保留，不应因为文献推翻首创性而抹去。

因此当前准确名称是：**基于教师正确性与置信度门控的 KD，加有上限的 batch hard-pair 平滑排序正则**。不应叫精确 pAUC、NP 学习、可证明低 FPR 迁移、分布漂移稳健算法。

### 4.1 风险迁移缺少的连接

令冻结决策 $h_T,h_S\in\{0,1\}$，$P_0,P_1$ 为相同评价总体的正常／攻击条件分布。直接集合包含给出

$$
\mathrm{FPR}_S\le\mathrm{FPR}_T+
\underbrace{P_0(h_S=1,h_T=0)}_{d_0},
\qquad
\mathrm{TPR}_S\ge\mathrm{TPR}_T-
\underbrace{P_1(h_S=0,h_T=1)}_{d_1}.
$$

这是**简单分解，不是新定理，也不是创新证明**。若想把它变成有用合约，至少还缺：

1. 教师在目标总体的可靠 FPR 上界；总体 AUC 更强不够。
2. 学生新增正常报警概率 $d_0$ 的可控预算；若教师已耗尽 1%，不能再无预算地允许新增 FP。
3. 从训练 surrogate 到一侧离散分歧的连接，以及有限样本估计误差。
4. 模型、阈值、路由选择之后的独立校准；反复使用同一校准集挑最佳方案会引入选择偏差。
5. 明确的独立统计单位、目标 mixture、未知攻击范围；当前 group loss 不提供这些条件。

R05 已学习相对阈值分歧；R11 已研究 NP 迁移。因此即使补出上述链条，也必须检查它相对成熟受约束学习到底新增了什么，而非把简单上界命名成新框架。

### 4.2 correlated flow、group 和 mixture shift

设正常总体由环境 $g$ 混合：

$$
\mathrm{FPR}_{Q}=\sum_g\pi^0_{Q,g}\,\mathrm{FPR}_g.
$$

推断：训练混合的平均 FPR≤beta 不保证正常环境比例变化后仍≤beta；所有组都≤beta 则对这些组的任意混合成立，但这只是凸组合性质。等权平均各组损失并不约束最差组。纯攻击先验变化、正常条件分布不变时，FPR 本身不变；不能与正常子群混合变化混为一谈。

会话／近重复组、攻击家族、部署环境是三种不同分组：前者用于防泄漏／描述依赖，家族用于覆盖与召回，环境用于迁移风险。三者不能用同一个 group ID 直接替代。分组切分是必要设计步骤，但并不让组内 flow 突然 IID。可以报告 cluster bootstrap 的经验不确定度，不能因此声称分布无关 NP 保证。若改以会话最大分数控制“一个会话是否报警”，控制的已是另一单位的风险。

在正常／攻击条件分布都可任意变化、又没有目标标签或结构假设时，不应承诺固定 FPR 与高召回同时保持；R11 的有标签 target、R14 的 mixture／score 条件必须显式列出。

### 4.3 cascade／feature acquisition 还缺什么

R16 的正常预算是**新增** FP 预算，且 CP compatibility 在 malware 类上构造；这不是对正常总体 FPR 的 NP 保证。作者表 A6 的 nominal 1% 方案，冻结 baseline 测试 FPR 为 1.1807%，C-GUARD 为 1.2111%，说明 nominal／validation 与实际 test 要分开。[R16 表 A6](https://www.mdpi.com/1999-5903/18/8/431)

R16 还先计算 rescue score 再决定其 context gate；若本项目要节省辅助模型成本，不能只在同样结构上把“少翻转样本”当成“少运行强模型”。路由必须在调用强模型前用已可用信息完成。整体级联风险应直接计算，不能因为两个模型各自校准在 1%，就宣称按输入选择后的组合也≤1%。

AFA 的成本还包括共享特征计算依赖、等待完整 flow 的时间和分支管理。CSV 中少几个字段不代表真的少采集／少计算；没有原始流量或可测的提取器时，只能称模拟特征预算。

## 5. 最少关键对照与资源边界

### 5.1 先做诊断，不立即扩展真实训练矩阵

对已有缓存预测可先检查：教师 TPR@低 FPR、师生一侧分歧、各正常环境 FPR、困难攻击家族召回、低尾部教师错误；若教师在目标工作点没有优势，停止把 TFM 强教师当作前提。严格保留最终 test，诊断只使用授权开发分区。

若进入新 pilot，低 FPR 蒸馏最少应有：同容量 hard＋成熟低 FPR 目标、普通 KD、adapted EKD、所提规则。四项共用教师、训练步数、正类权重、数据与相同阈值协议；另加一个与所提规则同样本量／总权重的随机选择控制，排除仅靠 hard mining 或重加权获益。现有 matched-hard 控制可复用其识别思想，不能以换名字省略。

只选一个教师、一个学生、一个环境、三个种子作证伪 pilot；最多约 15 个学生 fit，而不是把教师×损失×字段×门控×seed 变成全组合。只有关键假设成立后才冻结、扩展双环境五种子。这里是预算建议，**本轮未启动这些训练**。

### 5.2 数据要求

| 要回答的问题 | 最少需要的数据／信息 | 当前不能替代它的材料 |
|---|---|---|
| 正常 1% 风险可靠性 | 与评价正常总体对应的独立校准单位、冻结 score／threshold、足够尾部样本 | 训练正确率、全 AUC、相关 flow 行数 |
| 正常 mixture 变化 | 可解释环境标签和各环境正常样本，预先固定 mixture 情景 | 任意 hash group、随机种子划分 |
| 困难攻击保护 | 家族标签及独立开发／评价样本，预设最差家族指标 | 只筛选现有模型漏掉的 test 再报告 |
| TFM 教师有效 | 可追溯 checkpoint／预训练来源、折内 context／预处理／cache、查询成本日志 | 模型最新、官方通用基准更好 |
| 动态字段成本 | 可延迟的真实提取过程、字段依赖和时间／内存测量 | 字段个数或离线文件列数 |
| concept shift 下保证 | 相应 target 标签或明确结构／支持集假设 | 仅无标签漂移检测器或 CP 名称 |

### 5.3 运算和复现要求

- TFM 应离线冻结；固定上下文预算、query batch、集成数和 checkpoint。OOF 每折预处理、标签依赖表示及 KV cache 均需隔离；不能全量标签预计算表示后再分折。
- 小表论文的可运行成本不能外推到百万行 IDS／24GB 显存；先预算完整 OOF 查询时间、峰值 RAM／显存，再决定数据范围。学生仍使用预设训练总体，不能为教师省钱而悄悄删掉困难家族。
- pAUC 求解器的样本级状态和优化器状态必须保存；只存神经网络权重可能无法正确续跑。
- 级联报告端到端 CPU batch=1、吞吐、平均与 p95 延迟、调用比例、预处理与路由成本。训练时间、教师生成时间、学生部署成本分别记录。
- R05、R01 作者代码入口已核；R08 代码链接 404；R07／R10／R11 未核得实现；R16 工件需申请。**这些事实不能合并成“近期方法均已跑通”。** 本轮不自动安装或升级任何依赖。

## 6. 仍值得验证的三个窄问题

### Q1（优先主线）：小容量学生在正常 mixture 改变时，能否保留教师可量化的一侧风险预算？

**待验证假设**：在固定教师查询预算与小学生容量下，显式区分“继承教师漏报”与“新增正常报警”，再针对预设正常环境约束一侧分歧，是否比 EKD 的阈值关系匹配、普通 CVaR-KD 更稳定？

最强近邻：R05 的工作点排序、R11 的双类条件 NP 迁移、R04 的阈值合约；不能称首次风险迁移。可形成差异的部分只能是**固定部署容量、可核依赖结构与具体风险预算如何联合处理**。若只把各组 loss 等权，问题没有被解决。

最小击穿实验：先用同一批开发预测固定若干正常环境的比例变化，仅改变 mixture、不重训，检查 pooled 校准是否失效；再比较 hard-CVaR、普通 KD、adapted EKD、候选方法。在同风险上界／同校准预算下比较召回，而非允许候选多花 FP。若普通 groupwise 校准已经解释全部收益，或候选在 IID 条件下都不胜 EKD，就下调为协议／适配贡献。

数据与失败条件：需要真实环境／会话信息、独立正常单位及足够各组样本；缺少这些时不能承诺保证。若保证仅对已见组成立，应明确对新环境无保证。固定 MLP 并不满足 R11 的全部凸性计算假设，不能直接沿用其定理。

### Q2（资源受限时的窄备选问题）：固定教师查询额度下，尾部目标能否指导“问哪些样本”？

最强近邻：R08 的查询覆盖、R10 的生成量／保真度、R05 的 critical relations；另外仍需继续排除 active distillation 的直接近邻，**本次搜索没有证明此处空白**。

最小击穿实验：同一教师、context、查询额度、训练更新数，比较 uniform random、普通 uncertainty、EKD critical-region 和目标风险导向选择。用完整开发总体评价，禁止只在被选样本上计算优势。先使用真实 train 行，避免同时引入生成器；若只比“问全量教师”省成本，却不胜等预算随机查询，就不保留算法主张。

这是教师成本过高时才启动的资源问题；不与 Q1 同时扩展成第二套大矩阵。即使有效，贡献也应限定为特定预算下的选择规则及证据。

### Q3（部署备线）：只用便宜信息作升级决策，能否在整体正常风险约束下获得真实成本收益？

最强近邻：R12 的质量—成本最优路由、R13 的动态获取、R16 的安全任务 rescue；R15 说明“KD＋风险早退”概念也已有。不能以级联、困难样本、conformal 命名宣称创新。

可能的差异：在调用辅助模型／昂贵字段之前，用当前可得字段预测一侧纠错收益，联合约束**整体**正常 FPR 与实际成本；对正常 mixture 变化明确失效／回退条件。它仍可能只是成熟约束决策的 IDS 适配，需由实验而非措辞决定贡献级别。

最小击穿实验：先在已有模型预测上计算互补性与 oracle 上界，无需新训练；然后比较全量小模型、全量 HGB、同覆盖率随机升级、简单 confidence 升级、收益升级。若存在 rescue，还必须加入 validation FP-matched 的单阈值放宽对照，防止把增加 FP 的收益误认为算法收益。各策略使用统一整体校准、相同信息可用性和实测成本。若扣除 router／字段成本后不胜简单策略，停止。

当前建议：**先用 Q1 的最小反证步骤决定是否存在值得迁移的低 FPR 信息；Q3 只在模型错误互补且可测真实成本时作部署备线。Q2 仅作为教师查询昂贵时的收缩方案。** 这三项都是待验证问题，不是三项已经成立的创新点；新实验证据出现前，应主动下调原有“correctness-confidence＋ranking”首创主张。
