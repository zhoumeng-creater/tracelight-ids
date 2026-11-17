# R30之后：作者实现与训练目标对照

2026-11-17。只读取源码、论文与既有结果；0模型加载、0前向、0训练、0新阈值。本审查直接核对TabM、TabReD和STACO，并纳入主任务本轮活读固定源码后的tapAUC核查；共四个相关对象，不扩展训练矩阵。

**没有发现可以解释R30误报、并能直接修正的TabM通用实现错误。应保留合法的作者主干和工件交付方式。需要重新论证的是项目自行定义的组采样、训练目标与验收之间的关系，而不是把一个开源loss接进去就称设计完善。**

## 1. 固定来源与本地证据

| 对象 | 年份核验 | 固定代码 | 本次读取用途 |
|---|---|---|---|
| TabM | [ICLR 2025正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html) | `yandex-research/tabm@28e47ae301c92ec37787dde1ce923a0793f405b4` | 成员损失、概率聚合、优化器、批次和选轮 |
| TabReD | [ICLR 2025正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html) | `yandex-research/tabred@b5ef15b3749f30da7a1eb8fba21a5b54d706bf32` | 时间迁移、验证指标和最佳工件交付 |
| STACO | [TMLR 09/2025论文](https://openreview.net/pdf/7ae8adaf370a9f67aae912154bf2f6632a96b1e2.pdf)，[2025年9月修订稿](https://arxiv.org/abs/2505.21944) | `Optimization-AI/LibAUC@3c02aa9877e0091114a00f94a8c6367698eb5637` | 两向pAUC尾部目标、采样及状态合同 |
| tapAUC | [2025年v2论文](https://arxiv.org/abs/2502.11570v2) | `ArnaudBougaham/tapAUC@24c2dde9248348e73094a156a64b323a6b41203d`；notebook blob `52f4fc351f9a6848f22f41dba2f291010b85769e` | 主任务活读核对负例截取、零FN阈值与实际工作点 |

固定源码快照及各自许可保存在仓库`docs/r31_design_review_20261117/author_sources`，10个文件约205KB，逐文件SHA见`sources.json`。快照统一LF和一个末尾换行，未导入执行。LibAUC为2026版库中的2025方法，不把库发布日期替代论文年份。

## 2. TabM：哪些地方正确，哪些是我们自己加的

| 项目R30位置 | 作者实际实现 | 本次判断 |
|---|---|---|
| `training.py:99`，四组内部对32个成员BCE取均值，再平均四组 | [example.ipynb L396–415](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb#L396-L415)把`B×K`展开，逐成员交叉熵后取均值 | 成员独立训练原则正确。作者示例用分类CE；单logit BCE是本项目的二分类适配，不能写成逐字复现示例 |
| `route_batch/tabm_training.py:55`，mean member probability的稳定log-odds | [example L440–444](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb#L440-L444)先softmax再沿成员平均；[paper/model.py L535–543](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/paper/bin/model.py#L535-L543)相同 | 聚合语义正确；没有把平均logit错当平均概率 |
| `training.py:250`，AdamW对全部network.parameters施加lr=.002、wd=.0003 | [example L330](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb#L330)使用相同调用 | 全参数衰减是官方合法配方，不能再误报为bug。论文代码L431–433使用parameter groups是另一配方，不能仅据差异就改R30 |
| `sampling.py:49`以后，四个独立循环队列，每组64，32成员共享批 | [example L566–574](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb#L566-L574)支持共享或独立成员排列；当前paper代码L81默认共享 | 共享批合法；A0/A1/B0/B1每组等质量是本项目适配，作者没有为我们的组权重提供验证 |
| `selection.py:31`，最差协议面板pAUC、pooled pAUC、面板均值、最早step | [example L446–452](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb#L446-L452)分类示例选择accuracy；L601–616严格改善后更新checkpoint/patience | 自定义pAUC选轮没有执行错误，但不是作者已证明能控制逐单元1%误报的配方 |
| 选中F checkpoint原样交付，无P重训 | [example L619–620](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb#L619-L620)最终加载最佳checkpoint | 保留，R30哈希链已经验证这一点 |

R30目标可写为每个组的成员BCE平均，再对四组平均。因为每组都取64行，等价于整个`256×32`矩阵的未加权均值；代码没有叠加`pos_weight`。这不等于自然流频率下的经验风险，亦不等于正常高分尾部的最坏风险。每个B正例被重复采样数千次，是固定四组质量和训练长度的结果，不是新增了监督信息。

R30已记录F_B1高度拟合而V_B1存在低分尾部，说明训练拟合与留出置信度有差异。现有证据没有把这种差异唯一归因于采样、容量、weight decay或优化器。没有依据直接切换到自然批、减小成员数、加大衰减或延长训练。

**保留：**作者TabM+PLE、逐成员损失、概率平均、现有优化器和已验证工件链。**需要修订设计论证：**四组目标与正常尾部验收的关系；“面板满分”与“完整支持分离”的区别；是否存在能被训练目标直接作用的尚未学对关系。不要把NLL和pAUC互换当作修复。

## 3. TabReD：支持时间验证，不能替我们保证误报

TabReD论文展示时间切分和特征条件会改变模型排名，支持本项目保留F/V/C/E的时间、采集和角色分工；它不证明更复杂模型能解决当前尾部问题。[论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html)

固定`nn_baselines.py`的[L484–501](https://github.com/yandex-research/tabred/blob/b5ef15b3749f30da7a1eb8fba21a5b54d706bf32/paper/bin/nn_baselines.py#L484-L501)仅在验证score严格增加时保存，结束后载回最佳模型。R30已经采用该交付原则，不能继续把“取消P重训”作为下一轮待修的新点。

该作者实现没有对每个来源单元施加1%误报约束。借鉴范围应是时间验证与工件一致，而不是用其榜单均值支持我们的15单元同时合规。当前没有理由因此增加TabR、Transformer或RFMs训练分支。

## 4. STACO：与尾部目标直接相关，但不适合作为当前即插即用修改

STACO优化低FPR、高TPR区域内的两向pAUC代理目标。它把低分正例与高分正常的选择写成带辅助变量的CVaR/min-max形式；不是在任意256行批中取top1%再做BCE。[2025论文第3节](https://arxiv.org/html/2505.21944v2#S3)

其方法提供两点可借鉴原则：困难样本选择必须对应明确的完整数据目标；若使用随机状态算法，必须保留状态与索引的含义。它没有证明F上代理目标下降就能让C校准后的每个D单元合规。

对本项目的具体采用阻力：

1. **论文与当前库入口不能混称。** 论文Algorithm 1/2要求独立的两份负批，用来解耦dual更新和后续梯度估计。[论文算法](https://arxiv.org/html/2505.21944v2#S3.SS1) 当前固定[`auc.py` L818–853](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/losses/auc.py#L818-L853)只接收一份`y_pred/y_true/index`，同一pair矩阵用于dual更新和模型surrogate；因此不能把论文独立采样保证直接附到该调用。
2. **不是只换loss。** [`staco.py` L163–200](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/optimizers/staco.py#L163-L200)使用逐元素梯度截断并把L2项加进Adam梯度，与R30全局范数裁剪、解耦AdamW衰减不同。若采用作者完整入口，会同时改变优化器、采样、辅助步长和目标。
3. **必须定义32成员如何参与。** 作者loss处理单分数的正负pair矩阵。不能直接把32成员展开成独立流样本，也不能直接改成ensemble loss后仍声称保留TabM的独立成员目标；两种适配都需要重新明确数学合同。
4. **状态交付有额外要求。** [`auc.py` L811–814](https://github.com/Optimization-AI/LibAUC/blob/3c02aa9877e0091114a00f94a8c6367698eb5637/libauc/losses/auc.py#L811-L814)的`u/s1/s2/dual_var`为普通Tensor属性，该类没有注册buffer或自己的state_dict扩展。若训练恢复，只保存网络state_dict不足。静态代码还显示只传正例index时`ids_p`没有赋值，不能照文档示例直接接入；本轮未执行这个入口。
5. **完整F已核实，当前没有未解决的训练跨组逆序。** [input_mechanism_summary.json](input_mechanism_summary.json)确认三个blind的完整F最小正例分数减最大正常分数分别为2.5374、3.5587、8.9514，均严格大于0；F_B分别为16.1458、10.0348、11.2189。先核完整F、不能由F_A/F_B分别分离直接推论整体分离，是必须保留的方法要求；本轮这项核查已经完成。V中的39条低尾交叠仍不能回流F训练。

官方CIFAR10教程使用单输出模型、DualSampler、固定轮次和独立loss状态步长，展示的pAUC区域为FPR≤0.3、TPR≥0.7；它没有验证本项目FPR≤0.01且15单元同时合规的条件。[作者教程](https://docs.libauc.org/examples/tpauc_staco.html)

**当前决定：不引入STACO，不把它列入待跑矩阵。** 它只提供尾部目标和实现审查的对照。只有完整F中确有与目标错误同机制的未解决关系，或另有数据支持的表示约束，才有必要构造一个明确的训练目标候选；随后还需约定配对更新、曝光、成员目标及退出条件。单调放大正确分数间隔也能降低BCE或有限margin损失，不能以loss降低替代原工作点错误的改善。

新缓存审计还显示，完整F最高分147条正常与全F正例之间的logistic pair分数导数绝对值均值分别为`2.18e-7 / 2.73e-7 / 1.13e-9`；这些正常的A/B组成为`130/17、137/10、87/60`。这是既有分数上的解析量，不是网络参数梯度，也不是重新训练的预测。它进一步削弱了“在同一批已分离F上追加成对尾部压力，就会修复D正常尾部”的直接论证，不能用来宣称所有尾部方法都无效。

## 5. tapAUC：固定作者代码与本项目目标不同

主任务本轮活读了[固定notebook](https://github.com/ArnaudBougaham/tapAUC/blob/24c2dde9248348e73094a156a64b323a6b41203d/tapAUC.ipynb)，核对其blob为`52f4fc351f9a6848f22f41dba2f291010b85769e`。以下行号指cell 0代码展平后的Python行号，不是notebook JSON行号：第161行按分数降序排列；第169行取末尾5%负例，因此取到的是低分负例；第112行以正例最低分数构造零FN阈值。不能把改过截取方向的版本称为作者原样复现。

[2025年v2论文](https://arxiv.org/abs/2502.11570v2)报告TPR 92.52%、FPR 20.43%。该工作点不能作为本项目15单元分别FPR≤1%的成功先例，零FN优先目标也不对应原验收。当前不采用tapAUC，不将其与STACO并列为待跑矩阵。因未发现明确许可，本轮不拷贝完整notebook，只保留固定提交、blob与上述核查指针。

## 6. 本次排除的老问题和必须回答的新问题

| 常见跳步 | 这次的处置 |
|---|---|
| 见到FP超标就改alpha或拼接模型 | 原注册和判决保持；不把标定当作已识别根因 |
| 见到高NLL就换回NLL选轮 | 区分训练目标、排序、置信度及工作点；NLL低不保证误报预算 |
| 作者模型成熟，所以这次一定成功 | 成熟性支持底座选择；当前任务的正常支持和验收仍需本地证据 |
| 四组采样保证每组出现，所以正常尾部已受控 | 每组均值目标与组内稀有形态不是同一对象 |
| F_A/F_B分别分离，所以没有训练排序可修 | 本轮已独立核完整F并确认三seed均严格分离；今后仍不能跳过完整支持核查 |
| 添加tail loss却不定义成员、采样、状态和预算 | 不实现、不注册为实验；上述合同必须先完整写出 |
| 新目标需要看一轮结果才知道设计是什么 | 先固定其作用对象、比较能归因到什么、原15单元/TP保护及退出分支 |

本次没有发现必须立即修改的TabM底层代码。完整F严格分离已经确认，后续重点应转向留出正常形态的支持关系及有依据的表示约束；任何新目标都须说明它如何改变具体错误关系，现有开源实现不能替代这一论证。
