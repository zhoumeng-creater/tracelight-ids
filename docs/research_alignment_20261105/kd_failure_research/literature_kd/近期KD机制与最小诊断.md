# 四臂失效：2025—2026 KD 近邻与机制辨识审查

本报告所述脚本与作者源码保存在原分析目录：E:\OneDrive - Coventry University\中传\毕业论文\工作\KD失效机制研究_20261105\literature_kd。仓库副本保存报告与紧凑证据。


核查日期：2026-11-05。状态：实时文献核查、作者代码静态阅读；没有运行作者训练，没有新模型推理或拟合。目标是区分优化、目标兼容性与任务收益，不为新损失或更大矩阵寻找理由。

## 1. 对当前研究方向的判断

用户已说明没有模型体积、内存或延迟的硬性上限。应先比较现成 HGB 与学生的实际检测效果和成本，不能把“必须压到原 small”当成研究前提。蒸馏是否值得做，取决于它是否带来实际需要的效果—成本收益；教师有优势并不自动意味着应该训练学生。部署已有强模型和提出新的研究贡献也应分开。

四臂实验有效地否定了固定起点、固定目标和40次全F更新这个候选，但没有识别学生容量上限。新发现是：共同warmup的40遍F对应3360次小批更新，四臂续训40遍F仅40次全批更新。同样的数据暴露量不等于同样的优化进展；这个差异不破坏四臂内部公平性，却限制了“该学生学不会”的解释。

优先顺序应为：确认实际压缩价值 → 补齐F端残差与优化信息 → 必要时只检验一种优化方案变化 → 再考虑表示或目标改变。没有证据时，不同时换教师、学生、温度和损失。

## 2. 广度检索与取舍

以下8项均实时核对一手来源的2025/2026正式身份。深读程度单列，不把摘要筛查计为复现。

| 来源 | 本轮读取程度 | 对本项目的作用与决定 |
|---|---|---|
| [Progressive distillation induces an implicit curriculum，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3b4e1336f775c3dba16ebbb8d2afd258-Abstract-Conference.html) | 正式身份；原文§2—3、二分类稀疏奇偶任务；作者parity训练代码固定提交 | 直接支持把可学习性与最终教师精度分开。采用优化机制问题；不直接引入教师checkpoint课程 |
| [In Good GRACES，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/59fa5d049b777d880fdcddba2b24738f-Abstract-Conference.html) | 正式全文§2.1—2.2及作者分数实现固定提交 | 支持学生相关的兼容性诊断。借鉴参数梯度视角，不将GRACE数值直接当本项目低FPR验收门槛 |
| [A Good Teacher Adapts Their Knowledge for Distillation，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Qian_A_Good_Teacher_Adapts_Their_Knowledge_for_Distillation_ICCV_2025_paper.pdf) | CVF原文索引中的§3与Algorithm 1、训练设定；直接PDF请求403；未找到可固定作者实现 | 提醒类内目标分布也会造成难学。AID需反向微调神经教师，再从头训学生，不适合直接套在固定HGB上 |
| [Quantifying Knowledge Distillation using Partial Information Decomposition，AISTATS 2025](https://proceedings.mlr.press/v258/dissanayake25a.html) | 正式摘要和作者RID实现相关分支；未在本轮完整复核理论证明 | 区分任务有关信息和表示保真度。HGB仅提供一个概率，本项目没有其论文需要的对应神经中间表示，暂不引入RID |
| [Distillation Scaling Laws，ICML 2025](https://proceedings.mlr.press/v267/busbridge25a.html) | 正式条目、作者研究页与原文预算设定片段；未复现大规模拟合 | 采用教师训练、学生训练与部署收益分别记账。其语言模型缩放律不用于预测UNSW收益 |
| [TAID，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e664650506f1cf2b4696df892147c06e-Abstract-Conference.html) | 正式摘要及方法概述；未读固定源码 | 动态插值用于语言模型分布差异、mode collapse；目前没有这类机制证据，不增加动态目标模块 |
| [Gap Preserving Distillation，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9a379c1b05793d1c42dc832269834515-Abstract-Conference.html) | 正式摘要、额外动态教师及双向重参数化设定 | 要求额外神经教师和参数共享，改变资源与模型假设；不列入当前最小方案 |
| [Rethinking Self-Distillation，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9d4824d834b5fe8e6b53dcfe42cab8d2-Abstract-Conference.html) | 正式身份实时复核；本轮只读摘要，既有SelfPLL代码审查另见原仓库记录 | 理论针对固定特征、结构化相关性与标签噪声。二分类top-2覆盖全部类别，不能作为新的实例信息蒸馏臂 |

本轮没有把“教师更强”“近年论文有效”或“代码可见”当成本项目改动有效的证据。

## 3. 直接近邻之一：终态教师精度不等于易学监督

Progressive Distillation的正式论文发表于ICLR 2025；可读方法正文见[作者arXiv原文](https://arxiv.org/html/2410.05464v1)。论文在二分类稀疏奇偶任务中区分：

$$
L_{\mathrm{hard}}=\operatorname{KL}(e_y\Vert p_S),\qquad
L_{\mathrm{teacher}}=\operatorname{KL}(p_T(x;\tau)\Vert p_S).
$$

它的机制是中间教师阶段存在易学低阶相关信号，终态高准确率不保证保留同样的优化线索。这里理论依赖具体任务和训练设定，不能直接证明UNSW也有相同课程。对本项目更直接的启示是：必须先排除有限优化路径造成的失效，才能谈函数类限制。

作者仓库固定提交为 **8eb10d1459f6f675dc43122ab8e76cd9c2636a13**：

- [parity/boolean_distill_expts.py](https://github.com/abhishekpanigrahi1996/ProgressiveDistillation/blob/8eb10d1459f6f675dc43122ab8e76cd9c2636a13/parity/boolean_distill_expts.py)，Git blob **d15bda557746590f08f33fcb53ab314e14a391cb**。157—214行切换教师；234—246行处理教师温度；260—280行forward分支是固定教师软目标CE与hard CE混合；282—307行每个batch反传并更新。
- [配置](https://github.com/abhishekpanigrahi1996/ProgressiveDistillation/blob/8eb10d1459f6f675dc43122ab8e76cd9c2636a13/parity/conf/distill_boolean_data.yml)，blob **7b6db33eefe0744ea840c372f2144e33f75f6ba6**：显式区分batch size、n_steps、n_examples、warmup。不是用epoch一个数字代替更新预算。
- 本次只依据forward训练分支判断机制。作者包含其他策略入口，不把配置文件默认值直接称为论文全部实验设置。

**击穿解释的最小对照**：如果同一学生、同一目标、同一数据暴露下，恢复小批更新方案使H/K都同幅改善、K−H仍不达标，那么“KD只是更新太少”的应用解释不成立；如果K−H明显改善，才支持优化方案与教师目标有交互。它仍不证明更大容量一定更好。

## 4. 直接近邻之二：兼容性需要学生梯度，不是教师准确率

[GRACE正式全文§2](https://proceedings.iclr.cc/paper_files/paper/2026/file/59fa5d049b777d880fdcddba2b24738f-Paper-Conference.pdf)定义学生在教师生成数据上的参数梯度，按内部划分比较梯度二阶矩：

$$
\mathrm{GRACE}(D)=\frac{1}{C}\sum_{i=1}^{C}
\operatorname{Tr}\!\left[
\widehat{\Sigma}(D_{-i})^{-1}\Sigma(D_i)
\right].
$$

论文研究语言生成任务；理论连接到特定预条件下单步更新稳定性。它不是多步AdamW后的UNSW低FPR保证，也不说明只看梯度范数即可选择教师。

作者仓库固定提交为 **64fc99a10049f79abe05e2be638a178b448985d0**；[GRACE/GRACE_computation.py](https://github.com/abhishekpanigrahi1996/GRACE/blob/64fc99a10049f79abe05e2be638a178b448985d0/GRACE/GRACE_computation.py)，blob **ca9aa175260ee1b5a244222e5fad03095bab0b12**：

- 136—149行先按prompt把多个response放在同组，再作内部重复随机划分。
- 148—154行训练侧梯度逐行归一化、测试侧保留范数，构造二阶矩。
- 73—76行对谱作收缩平滑；165行取逆谱加权迹。代码的重复随机划分和平滑细节，应与论文理想分区公式分开记录。
- 这份代码没有UNSW语义、低FPR约束、重复流量模式分组或阈值注册。

因此本项目可以借鉴“目标通过学生几何结构才能成为可用更新”的问题，但不应立即增加一个GRACE筛选器。当前缺少F学生逐行分数与参数梯度；不能把已保存q、概率差或全局loss当成已测得兼容性分数。

## 5. 当前二分类KD的直接代数约束

以下是对本项目冻结目标的推导，不是外部论文已给出UNSW结论。

令 $\alpha=0.3/1.3=3/13$，$r_i=(1-\alpha)y_i+\alpha q_i$。在同一个参数点 $\theta$，

$$
\nabla_\theta L_K-\nabla_\theta L_H
=\alpha\sum_i w_i(y_i-q_i)\nabla_\theta z_\theta(x_i).
$$

这说明：

1. q与y的差只给出每行的logit梯度修正系数；真正的参数更新还由学生Jacobian和优化器决定。
2. q接近y的行几乎不改变hard训练，但少数大修正是否被抵消、是否落在学生可利用方向，不能由平均标签移动量判定。
3. 在不同终点比较两个梯度，还混有参数点差异。要识别教师目标的直接作用，应在同一个anchor上比较H/K。
4. C上教师能检出的733条攻击是工作点事件集合，不能拿来代替F上的监督残差、Jacobian或优化可达性证据。
5. KL下降不必然改善低FPR排序；二分类概率匹配、整体BCE和正常尾部工作点是不同对象。

## 6. 为什么暂不堆AID、RID或动态目标

AID原文Algorithm 1先固定预训练学生以反向微调教师，再用适应后的教师训练新学生。它给出“教师目标本身可能不适合某个学生”的反例，但当前HGB没有可照搬的神经反向更新路径。引入这个阶段会同时改变教师、起点、训练预算，不能修补本轮因果解释。作者源码在本轮未取得，不能声称已复现。

RID作者[固定提交25c263c65297f494bb1e3d632077c39c88669f62](https://github.com/pasandissanayake/kd-rid/tree/25c263c65297f494bb1e3d632077c39c88669f62)中，[systems/wrncifar100.py](https://github.com/pasandissanayake/kd-rid/blob/25c263c65297f494bb1e3d632077c39c88669f62/systems/wrncifar100.py)的Git blob为 **ed63f925e366552a592da938d815fcd3d18c38e8**。相关RED分支包含多层表示过滤器、任务头和交替更新，处理的是表示中的任务相关信息，不是对现有一维q简单加一个权重。本轮保留其任务相关性问题，不引入这些模块。

TAID/GPD所需的动态分布、动态神经教师或共享参数，还没有被本项目的失败证据指向。多加组件会让“为什么有效/无效”更难辨识。

## 7. 两个最小诊断建议

### 诊断A：先做0拟合的可用性与必要性核查

第一部分直接使用现成工件比较HGB、H和K的R/C指标、错误集合及当前同口径成本。用户没有硬资源上限，不设“学生必须小于某个虚构阈值”。如果现成HGB的成本可接受且检测明显更好，停止把压缩作为默认主线。

第二部分只在实验室补齐原anchor、H40、K40的F分数；不更新参数、不增教师、不选checkpoint、不读旧V/test。报告同一F上hard BCE、teacher KL、混合目标、分位残差以及按原标签/完整模式的分布。若需要梯度诊断，再对anchor做固定F的H/K参数梯度，分别报告g_H、g_K、修正项及其投影；不能用$p-r$冒充参数梯度。

这一步解释是否存在未学好的F目标及更新抵消，不能独自证明泛化收益或函数类极限。数据不足时应输出缺口，而不是把它改名为“容量瓶颈”。

### 诊断B：仅在仍有压缩价值且A指向优化缺口时，追加H/K两臂

两臂复用同一原anchor和固定q，保持相同40遍F、数据顺序规则、类权重、学习率和终点规则，只将full-F每遍一次更新改为原小批方案。保留已有H/K作为全批条件，形成“目标H/K × 优化方案”对照；不再加入温度、λ、教师、宽度矩阵。

- 比较的是整体优化方案，不把增加更新次数、批噪声与Adam状态演化误称为已单独拆开。
- H与K同预算，完整记样本暴露、优化器更新和训练秒数。
- 固定末端验收，不在C选择中间步。K−H若仍不足原最低实际增益或R方向仍退步，停止该小学生KD方向的继续投入。
- 若出现足够收益，只能说优化方案下出现KD增量。新方案尚未通过对应S/M控制或独立数据确认，不能直接发布完整迁移结论。
- 不把诊断B作为现在立即安排的训练。它由A的证据与实际压缩价值共同决定。

## 8. 交付与证据状态

source目录保存本轮阅读的4份作者源码/配置文本；source_manifest.json记录commit、Git blob、原URL与本地文本SHA。外部代码均未导入执行。既有四臂协议和结果保持原判定；本报告只提出下一步机制辨识顺序，未改变原验收门槛。
