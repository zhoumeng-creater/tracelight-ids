# 第三阶段：42 字段小 MLP 与折外蒸馏对照

日期：2026-11-01。状态：本轮真实实验运行前的开发协议。本文规定实验设计，不表示实验已执行或蒸馏已有效。

第二轮结果支持保留 42 字段输入；已有基线是有效开发结果。第三轮保持完整输入，研究减少网络容量，以及教师软目标能否改善同结构小模型。这里不修改第二轮的筛选门槛，不追加字段搜索，不要求蒸馏必须成功。先前结果已用于形成这一研究问题，因此本协议不能称为盲选或正式预注册。

数据、阈值、时间测量的通用约定继续见 [DATA.md](DATA.md)、[PROTOCOL.md](PROTOCOL.md)；具体运行见 [STAGE3_LAB.md](STAGE3_LAB.md)。保留前两轮归档，第三轮使用新的输出目录。

## 1. 比较问题与固定矩阵

**主要比较：small-KD 对 small-hard。** 两者输入、结构、初始化 seed、批次规则、优化器、训练预算及早停规则一致，区别是训练目标是否包含折外教师分数。

**容量比较：small-hard 对 large-hard。** 两者都只用硬标签；用于判断缩小网络的影响。small-KD 对 large-hard 则是容量改变与蒸馏的合并结果，不能单独称为蒸馏效应。

4 种主配置 × seeds 17、29、43，共 **12 个主 run**：

| 名称 | 输入 / 模型 | 训练目标 | 用途 |
|---|---|---|---|
| HGB reference | 全部 42 原始字段；200 次迭代、31 叶、learning_rate=0.1、early_stopping=False | 训练内 balanced 权重的二分类监督 | 全 train 教师能力参考；不是 OOF 软目标本身 |
| large-hard | 全部 42 字段；MLP [64,32] | 原有 weighted hard BCE | 新训练预算下的大 MLP 参照 |
| small-hard | 全部 42 字段；MLP [32,16] | 同一 weighted hard BCE | 小模型硬标签对照 |
| small-KD | 全部 42 字段；MLP [32,16] | 0.5 hard BCE + 0.5 soft BCE | 同结构蒸馏学生 |

三组 MLP 共同使用：epochs 上限 120、patience=12、min_delta=0.00001、batch_size=1024、learning_rate=0.001、weight_decay=0.0001、AdamW。固定现有 ReLU 及单输出 logit；不添加 dropout、重采样、家族权重、额外预训练或蒸馏温度搜索。训练 threads=4；HGB 在 CPU，MLP 使用本次明确指定的 CPU/CUDA 设备。

大 MLP 必须在本轮同预算下重跑。前两轮 60 epoch 的 MLP 只能作为有清楚标注的历史结果，不进入本轮容量或蒸馏配对差值。历史 RF 仍保留，但不增加到这轮主矩阵。

主 run 以同一 prepared 数据清单、同一 train/validation 样本和相同完整字段为输入；数据来源、重复组限制与前两轮相同。不要重新 prepare。实际编码维度由 train 词表决定并保存，不能把 42 原始字段直接写作网络输入维度。

## 2. OOF 教师与软目标的隔离

每个学生 seed 生成一套 3 折 OOF（out-of-fold，折外）软目标，额外执行 3 次 HGB 拟合；三个 seed 合计 **9 个辅助教师拟合**。它们与 12 个主 run 分开计数，总计 21 次模型拟合，但只有 12 个主模型进入最终的 CPU 测量。

折划分只发生在现有 **train** 内：`StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261101)`，以 binary label 分层，以完整 42 字段的 `feature_hash` 分组。固定 fold_seed 与模型 seed 是两个概念；三个模型 seed 使用同一折划分。分组隔离优先于精确比例，不能承诺每折恰好三分之一或各家族比例完全相等。折工件保存行数、组数与身份哈希；二分类计数可从 targets 的 label/折编号重建，家族计数可按 sample_id 关联原 train 重建，不声称存在单独保存的家族计数字段。

每个折的要求如下：

1. 该折 HGB 仅在其余两折拟合；填补值、缩放参数、类别词表和 balanced 样本权重均仅来自该折教师训练部分。不能先在整个 train 上拟合教师预处理器再切折。
2. 教师参数固定为与 HGB reference 相同的 200 次迭代、31 叶、learning_rate=0.1、early_stopping=False 和其余锁定默认值。保留每个折的有效参数与模型 seed，不用 validation 为教师选参数或轮数。
3. 教师只给未参与该教师拟合的那一折生成分数。每个 train `sample_id` 必须恰好得到一个 OOF 分数，不能取多个包含该记录的教师的平均值，也不能混入全 train HGB 对自身训练数据的分数。
4. 教师训练部分与预测部分的 `feature_hash` 交集必须为空，包含冲突标签的完整重复组也不拆散；不静默删除此类记录。每折教师训练部分必须同时含正常与攻击，否则停止并报告，不改用会泄漏的备用生成方式。
5. 软目标表绑定 `sample_id`、折编号、原训练样本身份和数据哈希；以稳定行身份对齐学生输入，不能仅假设两份表当前行序恰好相同。验证分数有限且位于 [0,1]、覆盖完整、无重复和未知 ID。

使用 train 标签进行分层是 train 内监督设计的一部分；折外要求是生成某行分数的教师没有拟合该行及其完整重复组，不是要求训练过程完全不使用标签。组隔离保证的是精确特征组，不能自动扩展为会话或时间独立。

学生预处理器可以在整个 train 上拟合，这是学生合法的训练数据范围；它与各折教师各自的预处理器相互独立。学生不需要复用教师的编码列数或词表。

**全 train HGB reference 与 OOF 教师必须分开解释。** reference 用整个 train；每个 OOF 教师只用约三分之二 train。reference 的 validation 效果不能冒充 OOF 软目标的质量，也不是学生的理论性能上限。折外分数可记录 train 内预测诊断，但不能拿它们直接和 validation 分数作同样本公平排名。

## 3. 损失、权重与早停

令学生输出 logit 为 $z_i$，硬标签为 $y_i$，固定的 OOF HGB 分数为 $q_i$。硬标签正类权重 $p=n_0/n_1$ 只根据完整 train 中正常与攻击的数量计算；不按 batch 或 validation 重新计算。两项均按 batch 样本数取 mean：

$$
L_{hard}=-\frac{1}{B}\sum_i\left[p y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\right]
$$

$$
L_{soft}=-\frac{1}{B}\sum_i\left[q_i\log\sigma(z_i)+(1-q_i)\log(1-\sigma(z_i))\right]
$$

$$
L_{KD}=(1-\alpha)L_{hard}+\alpha L_{soft},\qquad \alpha=0.5,\quad T=1
$$

实现使用 `BCEWithLogitsLoss` 或等价的稳定函数，传入 logits；不先 sigmoid 再作为 logits 输入。soft BCE 不传 `pos_weight`，不重复对教师软目标作正类加权。温度固定为 1，不支持本轮内改用其他温度；此处没有额外的温度平滑。训练历史的 `train_loss` 对 hard 模型表示硬标签损失，对 KD 表示上述混合目标，并由 `training_statistics.json` 的 `training_objective` 标明；`val_loss` 始终表示硬标签 weighted BCE。当前没有分别保存每个 epoch 的 hard/soft 训练分量，不能从总损失图反推两项各自变化。

$q_i$ 是 balanced HGB 的攻击分数，不是经过概率校准的真实攻击概率。两项损失的尺度和梯度不同，alpha=0.5 只表示系数各为一半，不保证梯度贡献各占 50%。本轮评估的是这一明确固定的二元软目标蒸馏方案，不声称它代表所有 KD 设计或必然优于硬标签训练。

所有 MLP 用 **validation 硬标签 weighted BCE** 选择 best_epoch，并使用同一个来自 train 的 $p$。KD 的软目标和训练总损失不用于单独定义更宽松的早停规则，也不生成 validation 软标签用于训练。各组实际训练轮数可以不同，需保存上限、实际 epochs、best_epoch 和完整历史。

small-hard 与 small-KD 每个配对 seed 使用相同初始权重和相同 DataLoader 打乱规则。教师生成结束后，学生训练必须重新设置其 RNG，避免生成教师消耗随机状态改变学生初始化。两组停在不同 epoch 是固定早停规则下的可能结果，不等于保证相同优化步数。

## 4. seed 和数据使用边界

模型 seeds 为 17、29、43，fold_seed 为 20261101；不挑表现最好 seed、不额外找折种子。固定折划分让配对比较可解释，但本轮没有测量 OOF 折划分随机性。

当前 HGB 设置下，不同 seed 可能生成相同预测；若 OOF 分数也相同，照实报告。这不构成错误，但 9 次辅助拟合不能被当成 9 份独立泛化证据。保留完整 seed 账目及实际成本，不假装三套目标必然互不相同。

train 用于教师、学生及各自预处理；validation 用于 MLP 早停、开发阈值和本轮候选筛选。calibration/test 在本轮封存，入口不得执行最终 `freeze`/`evaluate`。重复使用 validation 意味着结果仍是开发证据；完成此轮不等于已经证明真实网络泛化或完成整个毕业设计。

## 5. 主要指标与运行前操作规则

所有主 run 独立使用 validation 正常分数选择经验 FPR 不超过 0.01 的开发阈值，沿用现有 float64 比较与并列分数规则。分别保存阈值、FP/正常分母及实际 FPR，不能以近似同为 1% 隐去实际误报数。最终 test 的 FPR 并无保证。

对全部主配置报告总体 recall、FPR、precision、二分类 Macro-F1、AP、ROC-AUC、混淆计数和每个攻击家族的检出数/总数。重点分析 Exploits 与 Fuzzers，同时保留全部家族结果；不要只展示改善的家族。家族召回表示被检测为任意攻击，不是家族分类准确率。

主要差值在同 seed 内计算，再对三个 seed 汇总均值、样本标准差和范围，并公开三个原始差值。以下是**第三轮新开发操作定义**，在本轮结果产生前落盘；不追溯修改第二轮规则，也不是统计显著性或非劣效证明。

| 判定 | 本轮操作定义 |
|---|---|
| 值得保留的 KD 开发收益 | small-KD 相对 small-hard 的平均总体 recall 提升至少 0.5 pp；至少 2/3 seed 的差值严格大于 0；Exploits、Fuzzers 的平均 recall 各不下降超过 1 pp；所有工件完整且满足经验 FPR 预算 |
| 容量/部署候选 | small-hard 或 small-KD 对本轮新 large-hard 的平均总体 recall 降幅不超过 2 pp；Exploits、Fuzzers 的平均降幅各不超过 3 pp；随后透明比较实际部署资源 |
| 体积挑战目标 | 相对本轮 large-hard，三个同 seed 部署包字节数比值的均值不超过 0.50；未达到就报告未达到，不以参数量或输入字段数替换此指标 |

若 KD 不满足第一行，不能据平均值略高就称蒸馏有效，默认保留 hard 作为更简单的候选；若 small-hard 也不能保留所需效果，可继续保留 large-hard、HGB 或既有基线。即使 KD 有相对 small-hard 的开发收益，也可能仍不满足部署候选条件。未达门槛是明确研究结果，不为成功临时修改 alpha、温度、种子、家族范围或阈值。

模型大小主要由相同的小网络结构决定，KD 不应被描述成相对同结构 hard 学生进一步减少参数。若推理时间不同，按真实测量描述，不把训练方法本身当成保证加速的原因。

## 6. CPU 三轮测量与训练成本

训练全部结束后，12 个主工件在同机 CPU、threads=1 的独占时段测量 **3 个完整轮次**，共 36 个计时记录。`measurement_round` 为 1、2、3，`planned_rounds=3`，各轮 `order_seed` 分别为 20261101、20261102、20261103，保存实际顺序。辅助 OOF 教师不进入部署效率排名。

RF 不在本轮矩阵；BLAS/OpenMP/PyTorch 线程仍须统一限制。每轮沿用 5 次预热、100 个固定单条样本各 30 次观察、固定 256 行 batch 的 30 次观察；所有工件使用相同 validation 输入。计时范围是预处理加预测，不含载入、CSV 解析、特征提取、传输、界面或线程初始化。

每个工件分别保留三轮 p50/p95、批量吞吐与原始计时。先看同 seed、同轮的模型配对，再用每个工件三轮的中位数和范围描述重复性，最后汇总三个训练 seed。不要把 9000 个重复单条观察值当作 9000 个独立样本，或把 3 seed × 3 轮当作 9 份独立泛化实验。三轮方向一致支持本机条件下的复测一致性，不能自动成为统计显著性、跨硬件或线上性能结论。

真实配置不允许降低轮数；只有合成 smoke 可显式单轮。中断后的续跑保留实际时间，跨时段恢复不能标成不间断测量。多轮报告必须按轮次区分并检查覆盖率，不能拿最后一轮覆盖前面文件。

训练成本分列：reference/学生的 `training_seconds`，以及每 seed 的教师 `generation_seconds`、各折 fit/predict/serialization 计时。fit 包含折内预处理和训练行诊断；generation 包含目标生成阶段的准备与工件写入，不是整个工作流墙钟时长。教师 generation 与学生 training 可以相加报告两个已测阶段的合计，但须注明不涵盖后续所有校验、报告和 CPU benchmark，不能声称程序保存了不存在的完整 KD 端到端耗时字段。若工件复用，分别说明原生成成本与本次新增工作，不能把缓存命中后的学生时间写成完整蒸馏成本。GPU 参与 MLP 训练，不改变 HGB 的 CPU 属性。最终推理只加载学生及其预处理器，不依赖 OOF 教师；研究归档仍完整保留教师工件。

## 7. 工件、恢复与完成条件

运行前固定源码、依赖、配置、两份第三轮文档、prepared manifest 及其哈希。主 run 继续保存模型、配置、环境、源码指纹、验证预测/指标/阈值和训练历史。OOF 另保存折划分、sample_id 到折/分数的映射、每折教师模型及预处理、教师配置、时间、数据来源和覆盖/隔离审计。

恢复不得仅检查文件存在：复用教师目标时核对源清单、train 分区、行身份、fold seed、teacher seed/参数、代码指纹与各文件哈希；完成主 run 绑定对应软目标身份；计时绑定 run、模型、数据、源码、环境、轮次和计划。改变任一研究设置应使用新的实验输出，不拼接成原方案。未完成尝试保留，不能让失败目录冒充完成记录。

本阶段应保留：12 个主 run、9 个辅助教师拟合的完整账目与 OOF 校验、36 份覆盖完整的三轮主模型 CPU 测量、配对效果及全部家族表、HTML/JSON 报告、原始计时与可校验打包。自动生成的 `STAGE3_COMPLETE.json` 仅表示工件、报告与计时通过完整性和覆盖检查，不自动按第 5 节门槛判定候选。真实实验带回后，再由人工或分析流程逐条核对门槛并保存候选判定记录；KD 是否达到收益条件不是生成运行完成标记的条件。

实验室入口为 `scripts/run_stage3.py`，默认配置 `configs/distillation.yaml`；合成验证配置为 `configs/distillation-smoke.yaml`。`scripts/package_run.py --work-dir ... --output ...zip` 先核对 `STAGE3_COMPLETE.json` 的文件哈希，再打包新输出、代码快照、协议与 `MANIFEST.sha256`，重新读取 ZIP 核验并生成 `.zip.sha256`；不自动携带原始 prepared 大文件。发布前仍需通过实际 `--help`、dry-run、smoke 和多轮报告覆盖验证，不能把入口存在当作真实实验已完成的证据。

## 8. 一手技术依据与论文边界

- [PyTorch 2.6.0 的 BCEWithLogitsLoss 官方源码和定义](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/loss.py)：直接接收 logits、目标可以在 [0,1]，并区分正类权重与 reduction。本文公式和两项权重的具体组合是本轮方法定义。
- [PyTorch 官方蒸馏教程](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)：支持同初始化对照、固定教师目标和组合监督损失的工程思路；教程使用的网络、数据及多类温度形式不等同于本项目。
- [scikit-learn 1.5 的 StratifiedGroupKFold 文档](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)：分组不交叉并尽量保持类别比例；shuffle 的作用有限，分层比例不保证完全满足。[HGB 1.5 文档](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)用于核对随机状态、早停和树参数。1.5 系列在线文档现标 1.5.2，实验环境仍以锁定版本记录为准。
- Benaddi 等，2025， [Lightweight Intrusion Detection in IoT via SHAP-Guided Feature Pruning and Knowledge-Distilled Kronecker Networks](https://arxiv.org/abs/2512.19488)。作者稿列 CommNet 2025 及 [会议 DOI](https://doi.org/10.1109/CommNet68224.2025.11288887)，并有[作者代码](https://github.com/mohammed-jouhari/iot-ids-lightweight)。它提供 IDS 压缩与蒸馏的研究参照；其 ToN-IoT、SHAP、Kronecker 方法与本文不同，本文不借用其分数、不称为完整复现，也不据此保证蒸馏成功。
