# PCT / MPT：原文、作者工程与 v3 的逐点代码审查

核查日：2026-11-03。范围：PCT（CVPR 2021 基础近邻）、MPT（AAAI 2026 正式近邻）及当前 `risk_distillation.py`。只读作者源码与论文；没有安装作者工程、下载数据/权重或运行真实训练。本文中的“适配”均是建议，不表示已实现或复现。

## 1. 先给结论

1. **保护旧正确样本、避免负翻转、同时允许纠错，以及旧模型加新教师的双源学习都有直接先例。** v3 不能以这些结构或“加入 margin”作为独立新颖性主张。
2. PCT 可作为当前二分类的必要直接对照；MPT 的双源 focal distillation 也应进入机制对照，但其**新类 logit bias**针对 50→100 等类别扩展，不能在本项目固定二分类上随意指定“旧类=正常、新类=攻击”冒充原版复现。
3. 作者工程需要审查后使用：PCT 团队统一仓的固定 KL 路径存在广播偏差；MPT 的温度、类维归一化和任务头与它也不同。不能把代码可见写成“已正确复现”，也不能凭实现问题否定原论文结果。
4. 本轮所读公式/路径中，**未见** v3 那种逐正常分层 FPR 上限、逐攻击族丢失 TP 预算及 projected dual 实现。这个有范围的观察不能证明全领域空白；v3 的修复信用还可改写成 capped hinge，不能单独包装为新的学习原理。

## 2. 一手来源、版本与归属

| 对象 | 本轮证据和固定身份 | 归属与限制 |
| --- | --- | --- |
| PCT 原论文 | Yan et al., *Positive-Congruent Training: Towards Regression-Free Model Updates*，**CVPR 2021**，pp.14299–14308。[CVF 正式入口](https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Positive-Congruent_Training_Towards_Regression-Free_Model_Updates_CVPR_2021_paper.html)、[Amazon 官方记录](https://www.amazon.science/publications/positive-congruent-training-towards-regression-free-model-updates)、[作者全文 v3](https://arxiv.org/html/2011.09161v3)、[Amazon PDF](https://cdn.amazon.science/be/47/881e9215461db4c976773a0ee2df/positive-congruent-training-towards-regression-free-model-updates.pdf) | 方法核查位置：§3.1、§4.1–4.3、Table 1、Eq.2/4/6–9；PDF方法集中在第4页。CVF 本轮直接访问返回403，Amazon CDN PDF和作者全文可读；不称本轮成功取得所有出版附件。 |
| PCT 团队统一工程 | [`amazon-science/regression-constraint-model-upgrade`](https://github.com/amazon-science/regression-constraint-model-upgrade/tree/fb2268c67706834632ec1427945fcf30a7218227)，commit **`fb2268c67706834632ec1427945fcf30a7218227`**，2023-12-08。[README 引用](https://github.com/amazon-science/regression-constraint-model-upgrade/blob/fb2268c67706834632ec1427945fcf30a7218227/README.md#L260-L301)；[Apache-2.0](https://github.com/amazon-science/regression-constraint-model-upgrade/blob/fb2268c67706834632ec1427945fcf30a7218227/LICENSE) | README 同时引用 PCT/ELODI/CNA，贡献声明含 PCT 作者 Sijie Yan、Yuanjun Xiong，以及后续作者 Yue Zhao。因此准确称“作者团队官方统一工程中的 PCT 路径”；**未证实此2023快照就是2021论文逐字节原始 artifact**。 |
| MPT 正式论文 | Ricci et al., *Mitigating Negative Flips via Margin Preserving Training*，**AAAI 2026 主会 Technical Track on Computer Vision VIII**，40(11):8721–8730，2026-03-14。[正式记录/DOI](https://doi.org/10.1609/aaai.v40i11.37825)、[正式 PDF](https://ojs.aaai.org/index.php/AAAI/article/download/37825/41787)、[作者全文 v2](https://arxiv.org/html/2511.08322v2) | 正式出版身份由 AAAI 记录确认；预印本首发2025，v2为2025-11-15，不能因此仍标“仅arXiv2025”。直接出版页/PDF本轮遇访问防护，方法精读用作者 v2，明确不把不同版本默认等同。 |
| MPT 作者工程 | [`miccunifi/negative_flip_MPT`](https://github.com/miccunifi/negative_flip_MPT/tree/2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e)，commit **`2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e`**，2025-11-16。[README](https://github.com/miccunifi/negative_flip_MPT/blob/2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e/README.md) | 作者全文直接链接该仓。递归树未见 LICENSE/COPYING；“公开可读”不等于已取得复制/再发布源码许可。优先依据论文独立实现适配，不直接并入作者工程。 |

搜索到的 `jun-fang/PCT` 自称 PyTorch implementation，但本轮未取得其为原论文作者仓的证据，因此**不把它当作者实现**。上表的 Amazon 团队仓归属依据是明确引用与贡献声明，不是仓库名或共同作者的猜测。

关键固定 Git blob：PCT `ConstrainedUpgrade/losses.py` 为 `1fcc5c0946438e6ae229474f918cd129f6fe0133`，`train.py` 为 `a27384849aba11986596fea01e75c5d0688436f7`；MPT `loss.py` 为 `1d76170e1efef4fe6f1f06d391890c2b9ab72e89`，`main.py` 为 `3dd9b36c846ac40c2d0f06b532af2439ceaff4a3`。以下外部行号均指固定 commit 的原始文本，不指滚动 `main`。

## 3. PCT：公式、旧正确掩码与作者工程的实际差异

记旧模型为 B，当前学生为 S，$c_i=\mathbf1[\hat y_i^B=y_i]$。PCT 原文定义总体 NFR 为“旧正确且新错误”的计数除以全部样本数，而不是仅除旧正确样本数。其方法把 CE 与 congruence 正则相加；Table 1 给出 naive、FD-KL 和 FD-LM，§4.2 也明确讨论 ordinary KD 对正翻转的抑制。[原文 §3.1/§4](https://arxiv.org/html/2011.09161v3#S4)

可对齐的正则结构为

$$L_{FD}=\frac1N\sum_i (\alpha+\beta c_i)D(S_i,B_i).$$

这里按作者代码的**正 KL/MSE 最小化**表示，并统一写样本均值；不是逐字符复制原文：作者 PDF/HTML Eq.7 前有负号，而 Eq.4 的最小化、距离惩罚描述与代码使用正距离。复现应显式固定正则符号/归一化，不盲抄该负号；本轮未取得作者对此记号的解释。

这里正确性是**真实标签与旧模型决策比较**，不是“旧模型置信度超过某阈值”或仅对攻击类判断。$\alpha=0$ 才完全不向旧错误样本蒸馏；一般 $\alpha>0$ 时，旧错误仍受到基础蒸馏拉力。允许正翻转的目标已经存在，不能把它等同于 v3 显式数值修复信用，但也不能把“允许纠错”再说成首次提出。

| 代码位置 | 实际行为 | 对照实现要求 |
| --- | --- | --- |
| [`losses.py` 373–388](https://github.com/amazon-science/regression-constraint-model-upgrade/blob/fb2268c67706834632ec1427945fcf30a7218227/ConstrainedUpgrade/losses.py#L373-L388) | 若师生类数不同，先保留标签在公共类范围内的样本，再切公共输出维度；源码自身留有空集合 TODO。 | 二分类同头时不需这个切片；不能把“空旧类 batch 返回NaN”当风险为零，也不能无记录改变分母。 |
| [`losses.py` 390–406](https://github.com/amazon-science/regression-constraint-model-upgrade/blob/fb2268c67706834632ec1427945fcf30a7218227/ConstrainedUpgrade/losses.py#L390-L406) | 支持 `old_correct`、`neg_flip`、`new_incorrect`、`all_pass`、`likelihood`。前三者分别依赖旧正确、旧正确且新错误、新错误。 | 默认 FD 不能概括为只有一种正确性 gate；`neg_flip` 还读取当前学生硬判定，须与静态旧正确控制分开。 |
| [`losses.py` 409–438](https://github.com/amazon-science/regression-constraint-model-upgrade/blob/fb2268c67706834632ec1427945fcf30a7218227/ConstrainedUpgrade/losses.py#L409-L438) | KL 用 `log_softmax(S/T)` 与 `softmax(B/T)`，按类求和并乘 $T^2$；L2 用逐元素 MSE；最后均值。 | 方向为 $KL(p_B\Vert p_S)$；soft-label BCE 与它在固定教师下仅差不影响学生梯度的教师熵，但类/样本归一化仍需一致。 |
| [`train.py` 1227–1264](https://github.com/amazon-science/regression-constraint-model-upgrade/blob/fb2268c67706834632ec1427945fcf30a7218227/ConstrainedUpgrade/train.py#L1227-L1264) | 在线教师 eval，输出/feature 显式 detach；一般支持 CE＋FD，也支持 `kd_alpha` 混合缩放。 | 不能把整个仓都称固定 hard 系数。拟复现路线必须固定 additive 或 mixed，并报告 effective hard/FD 权重。 |

### 3.1 已定位的 KL 广播偏差：不要直接照抄

固定版 `old_correct` 权重在396行为 `[N,1]`；KL 在412行按类求和后是 `[N]`；438行二者相乘会广播为 `[N,N]`。因此该路径计算的是

$$\operatorname{mean}(w)\operatorname{mean}(KL),$$

而不是论文预期的 $\operatorname{mean}(w_i KL_i)$。例如每行 KL 为 $(1,3)$、权重为 $(6,1)$，广播结果为7，正确逐行加权为4.5；交换两行 KL 后广播结果仍为7，逐行加权变为9.5。这里是 Tensor 形状/代数核查，**不是对原论文实验分数的复现或否定**。L2 的逐元素结果为 `[N,K]`，配 `[N,1]` 不受同一个广播问题影响；`all_pass` 标量权重也不触发该偏差。

后续须区分两种身份：**固定代码路径行为诊断**与**按论文公式、修正逐行权重后的适配基线**。主比较应使用有逐行 mask/梯度单测的正确公式；若保留原样路径，只作实现偏差反例，不能把存在广播偏差的弱对照当成击败 PCT 的证据。

### 3.2 默认参数不是 PCT 复现实验

[`train.py` 286–305/359–369](https://github.com/amazon-science/regression-constraint-model-upgrade/blob/fb2268c67706834632ec1427945fcf30a7218227/ConstrainedUpgrade/train.py#L286-L305) 的 CLI 默认 `kd_loss_weight=0`、`kd_filter=all_pass`、`kd_loss_mode=kl`、温度5；直接运行默认命令没有启用 PCT。类自身默认 `focus_type=old_correct`、`fd_beta=0`，又是另一层默认。README 示例才显式设旧正确 filter、base1、scale5、KL与温度100。复现实验必须固定完整配置，不能只记“用了作者类”。

## 4. MPT：双源、任务头、margin 与温度要分开

原文“Margin-calibrated Softmax Cross-entropy Loss” Eq.1–2 对**新引入类别**加正 logit bias $\Delta_j=k$，旧类别为0；“Double-source Focal Distillation Training” Eq.3–6 再加旧 B 与独立新 T 两项 FD。bias 仅用于训练，推理移除。它不是 v3 的逐样本 $d_i=\max(|a_i|,\kappa)$。[作者全文方法节](https://arxiv.org/html/2511.08322v2)

| 固定作者实现 | 实际行为及证据 |
| --- | --- |
| [旧教师训练](https://github.com/miccunifi/negative_flip_MPT/blob/2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e/train_old_model.py#L34-L80) | CIFAR100 前50类、50维 ResNet18，普通 CE。 |
| [独立新教师训练](https://github.com/miccunifi/negative_flip_MPT/blob/2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e/train_new_model.py#L46-L85) | 全100类、100维 ResNet18，普通 CE；不是旧教师 finetune，也不是一个未经训练的替代来源。 |
| [`main.py` 79–121/133–147](https://github.com/miccunifi/negative_flip_MPT/blob/2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e/main.py#L79-L147) | 学生100维、旧教师50维、新教师100维，两个 checkpoint；教师 eval＋no_grad，学生从头初始化。 |
| [`main.py` 173–195](https://github.com/miccunifi/negative_flip_MPT/blob/2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e/main.py#L173-L195) | MPT先在**未加 bias**的 logits 上算旧/新两项 FD，再对学生50:100维加+b计算 CE。nobias/nodistillation 是明确消融。 |
| [`main.py` 222–245](https://github.com/miccunifi/negative_flip_MPT/blob/2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e/main.py#L222-L245) | 测试直接用未偏移学生输出；旧类负翻转按旧教师正确、学生错误计数，不以新正确样本抵消。 |

`loss.py` 的[FD-KL 18–32](https://github.com/miccunifi/negative_flip_MPT/blob/2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e/loss.py#L18-L32) 是 `[N]` 权重乘 `[N]` KL，逐行加权正确；**没有 $T^2$**。main 使用温度5，因此不能把它与含 $T^2$ 的 v3/PCT 路径用同一数值系数直接比较。其 [FD-LM 35–45](https://github.com/miccunifi/negative_flip_MPT/blob/2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e/loss.py#L35-L45) 为每样本沿类维 mean 的平方差再除2；**LM 是 Logit Matching，不是另一个 margin loss**。与原文平方范数或其他仓库 MSE 的常数差应记入有效正则强度。

MPT 的旧分支将学生切为50维，但没有像 Amazon 统一类那样先剔除所有新类别行：在真实标签≥50的行上，旧正确掩码为0，仍可能留下 $\alpha$ 倍旧分支蒸馏。不能从两个仓库各拿一个默认后，把同名 PCT 对照当作完全相同。

### 4.1 原生工程不能直接进入现有研究流程

- `main.py` 在顶层解析参数、下载 CIFAR、加载 checkpoint 并执行训练；不应直接 import 以“复用一个 loss”。本轮只读取文本，没有执行这些入口。
- 默认方法是 MPT_KL；预置 checkpoint 路径、类别范围、ResNet头、训练/测试流程均为图像类别扩展。作者默认训练新学生的方式不同于 v3 从冻结 B warmstart；公平适配应让全部当前对照共享初始化，另标“受控适配”，不能声称逐值重跑原论文。
- ELODI 分支另有大小写判断、未显式 CLI 参数等需要核的入口问题；本报告不把这些分支算作已可运行基线，也不据此推断本轮未执行的 MPT_KL 结果。

## 5. 正常修复信用、条件预算：已看到什么与未看到什么

**已看到：**PCT 以旧正确掩码降低负翻转，也讨论避免普通 KD 抑制正翻转；MPT 双源 FD 沿用各参考模型正确性，并对旧/新类别施加不同训练偏置。它们都有选择性与类别条件结构，不能笼统称“完全没有 class-conditional 机制”。

**本次未见：**在所读 PCT §4/MPT 方法节及上述固定训练/loss 路径中，未定位到按正常分层规定绝对 FPR 预算、按攻击族规定 lost-TP 质量预算、在完整 C 上计算风险代理并用投影对偶更新的实现；也未定位到 v3 原写法中“新增正常 FP 上界减保守修复下界”的显式预算项。这是对指定材料的查找结果，不是全领域新颖性证明，更不能排除 NP、成本敏感分类、constrained classification、ramp/hinge 或其他 negative-flip 文献的等价机制。

进一步，令 $H_d(m)=\max(0,1+m/d)$，$B=\mathbf1[a\ge0]$。当前启用信用的正常逐行代理可重写为

$$u_0(m,a)=(1-B)H_d(m)+B\min\{1,H_\kappa(m)\},$$

攻击代理是

$$u_1(m,a)=B H_d(-m).$$

所以正常“credit”是既有 hinge/capped-hinge 形式的重表达；$B=1$ 的攻击保护也与“保护旧正确攻击”有直接关系。候选可能有价值的部分只能是具体条件风险目标、分母、约束求解和压缩任务证据的联合设计，且仍需查重/控制，不能把这一代数改写称新理论。

## 6. 对照当前 v3 的具体 API

本轮本地快照：[`risk_distillation.py`](../../src/ids/risk_distillation.py) SHA-256 `19cf30ff7868b238be60e62ab9c4224e87f2dd8b43d84614c4314a146e66881b`；行号对应此快照，后续代码变化须重新核对。

| v3 API / 位置 | 与 PCT/MPT 的关系 | 处理结论 |
| --- | --- | --- |
| `working_margins` 52、`freeze_margin_threshold` 71 | 在 R 正常样本上冻结工作点/尺度再定义报警余量；PCT/MPT 原路径为多类 argmax、原始类别 logits。 | 可研究工作点适配，但“margin”大概念不新。应与工作点对齐 FD 作比较，不能让对照只用0.5原始分类而候选独享低 FPR 阈值。 |
| `BoundaryKDConfig` 407、`boundary_kd_objective` 423–459 | 当前为固定 hard＋单个 T 的工作点 KD，已有 matched_hard；不是 PCT 的旧正确逐行 focal 权重，也未实现 MPT 双 FD。 | **不能称已实现 PCT/MPT。** 可复用 v3 的 tensor/分区/detach 防错，另写独立适配 loss 并核归一化。 |
| `build_risk_reference` 162、`constraint_risks` 237–318 | 冻结 B 在 C 的 margin/身份，正常按 B 阴阳性取不同 hinge/ramp，攻击仅计 B 检出后丢失；分母为指定正常层/攻击族全部行。 | 是候选条件风险目标；与 PCT 的总体 NFR、MPT 旧类 NFR 分母不同，不能横借文献 NFR 数字证明 FPR 改善。 |
| `RiskConstraint` 103、`risk_lagrangian` 320、`ProjectedDual` 337 | 用户声明风险预算、完整 C 风险报告和 bounded dual，与普通固定系数 FD 目标不同。 | 这一区别可做实证问题；投影对偶和约束优化本身不新，也没有由 API 自动得到总体/漂移/收敛保证。 |
| detach/签名/恢复与 [`risk_data`](../../src/ids/risk_data.py) | 角色穷尽互斥、C参与训练、R只作参考；工件身份和数值防错是研究协议保障。 | 可复用工程，不冒充算法创新；作者图像训练脚本不替代本项目 train-only OOF、联合组隔离或独立新来源确认。 |

二分类适配还需锁定三项：

1. 两类 logits 可表示为 $(0,z)$，softmax 的攻击分量为 sigmoid；但原版 `argmax` 在全零 tie 选索引0，而 v3 明确 `margin>=0` 为攻击。旧正确掩码应按声明的工作点与 tie 规则重新定义，并标适配。
2. 原始 logit、工作点 margin、按 R 尺度归一化 margin 三种蒸馏分别命名。LM 对 logit 表示/常数偏移敏感，不能只说“二类相同概率”就默认 LM 等价。
3. 所有学生共享 F∪C hard 标签、同学生结构/初始化、学习率候选/更新预算；FD 在哪一角色应用、B/T 在哪侧拟合、R工作点、OOF/context成本分别固定。修正公式、改变温度缩放和增加第二教师不能隐藏为实现细节。

## 7. 最小可执行取舍与停止条件

**必须先有：**同监督/同容量 hard 对照、普通 T-KD、matched-hard，以及按论文公式逐行实现的 PCT-FD(B)；若候选同时利用 B/T，加入 **FD(B)＋KD(T)** 或明确的 **双源 focal-FD(B,T)**，区分“多一个参考来源”与“条件风险约束”带来的收益。B/T同源、温度/权重匹配控制亦应按归因需要保留；不要先扩大模型数量。

**MPT 适配名称：**当前固定二分类优先称“MPT-style double-source FD, no class-expansion bias”。若声称完整 MPT，必须先有真实类别扩展协议、对应任务头和移除 bias 的推理路径；当前二元 IDS 不满足这个条件。normal/attack 不能冒充论文 old/new 类。

**最有辨识力的反证：**在同工作点和成本下，候选与 PCT-FD、双源 FD、fixed hinge/ramp、无 KD 条件风险控制相比，如果优势消失，就撤销相应知识迁移/自适应代理成功主张；若仅附加 C 标签或 warmstart 带来提升，应归因于监督/初始化。正常修复信用与攻击不抵扣的偏好应分别消融，但不再作为独立发明计数。

**暂不做：**不复制带广播偏差的 KL 路径、不直接运行下载/测试选择混在顶层的作者训练脚本、不为复现 MPT伪造新类别、不用旧 test 选择对照超参，不用图像 ER/NFR 或作者表中数字作本项目成绩。真实对照只有新协议冻结后在实验室执行；本轮仅完成证据审查。

## 8. 证据状态与未完成项

- 已完成：固定两个相关作者仓 commit/blob，精读上述 loss、mask、teacher、head、bias 和训练调用；对照当前 v3 API；记录 PCT KL 的广播偏差、MPT温度/归一化和任务差异。
- 尚未完成：原作者全部配置/附录/历史版本溯源、作者效果复现、IDS适配训练、全面 constrained/negative-flip 查重及新来源确认。代码归属和公式审查不等于这些工作已完成。
- 未见显式预算/信用的结论只限本表所列材料；未给不存在的代码入口补造地址。正式出版入口受防护的访问失败已记录，方法出处使用可访问作者全文，引用正式会议身份另有一手记录支持。
- 本文是本轮唯一新增文件；未修改算法/配置/既有结果，未训练真实数据。后续任何候选改动应先补“来源公式/代码→具体改变→可击穿对照→未解决假设”的记录，再决定实现范围。
