# UNSW 全字段 small 普通 KD：工件与实现审计

日期：2026-11-05。依据原 screen、threshold audit 的归档字节与当前训练源码。本文对应最小四臂、固定第 40 步末端的续训方案；没有执行训练或新推理。

**可以固定一位现成 full-F HGB 教师与一个现成 all-small warmup，只做四次配对续训。现有 runner 不能直接承担这次实验：它的损失尺度、检查点选择、置乱方式和自动实验矩阵都与方案不同。**

## 可复用工件

归档根目录为 `实验结果分析_20261105/received/unpacked/BUDGET_TRANSFER_UNSW/budget-transfer-unsw`。绑定与模型源版本为 `cabebf21cf7591fc14a3fb01c60fb94296fc8084`。逐文件 SHA-256 已与原 `TRANSFER_SCREEN_COMPLETE.json` 核对；详细列表见 `binding.json`。

| 角色 | 固定工件 | 状态与用途 |
| --- | --- | --- |
| 教师 | `teachers/all-s17/model.joblib` | full-F、42 字段、HGB 200 次 boosting、63 最大叶节点、learning rate 0.1、early_stopping=false、random_state=17；训练时使用 balanced sample weights |
| 训练软目标 | `teachers/all-s17/training-scores.npz:fit` | 85,916 个 float64 native margins；取 `scipy.special.expit(z)` 得到 T=1 的 q，不平移、不裁剪、不重估温度 |
| 共同学生起点 | `warmups/all-small-s17/model.joblib` | 42 原字段→189 编码维；Linear(189,32)→ReLU→Linear(32,16)→ReLU→Linear(16,1)，6,625 参数，包大小 34,475B |
| 历史 hard | `students/all-hard-s17` | 只作为既有成绩背景；归档保存的是 continuation step25 的 C 最优状态，不是第 40 步末端 |

共同学生起点由 seed17、40 epoch、batch1024 的 hard warmup 产生，入选 epoch40，共 3,360 次更新。训练过程中评估了初始化与每个 epoch，共 41 次 C 选择。原 continuation 从这一权重重置 AdamW；没有恢复 warmup 动量。新四臂必须同样从这一个权重、各自新建同参数 AdamW，不能将历史 hard 的已续训权重作为其中一臂起点。旧 history 的 step40 为 TP13,960/FP78、标准化 pAUC 0.8895426631；它与归档 model 的 step25 不同，且不替代新方案的同期 hard 端点。

| 数据角色 | 总行数 | Normal | Attack | 不同完整 feature_hash |
| --- | ---: | ---: | ---: | ---: |
| F | 85,916 | 27,440 | 58,476 | 49,507 |
| R | 12,274 | 3,920 | 8,354 | 7,073 |
| C | 24,547 | 7,840 | 16,707 | 14,145 |

三个角色的 sample_id 唯一，彼此没有 sample_id 或 feature_hash 交集。与旧 OOF 归档的对应身份文件逐字节、逐行、逐列相同。NPZ 本身没有 sample_id 数组；分数顺序由原 `_scores` 的输入顺序、共同身份文件及封存文件哈希绑定，不能把一组等长数组视为已经独立证明逐行正确。

预处理采用 F 上拟合的中位数插补、StandardScaler 和 OneHotEncoder(handle_unknown="ignore")，39 数值字段加 proto/service/state，输出 float32 的 189 维矩阵。仅反序列化可信 bundle 的已存元数据与权重，核实教师、warmup、历史 hard 的预处理状态完全一致；没有调用 transform 或 predict。

| 绑定 | SHA-256 |
| --- | --- |
| 教师 model.joblib | `fa4602158bc7c80ef0a0b777f1c05fc0f4edfca31bebd9b36600f24874e731fa` |
| 教师 training-scores.npz | `69630901408bf7262ddc0abb4001c890db71573b023cdd8e19d70d8ce6c53dde` |
| F native margin 连续 little-endian float64 字节 | `02123566eca5ad45b7d3bf2743c36eab8c43063cf2b60618f13b978130e5e423` |
| warmup model.joblib | `2efda2fec3a0623cf34cb228c23f0d2398fd2763fc9677a2effc9613e81d17da` |
| warmup network_state，原 canonical_hash 规则 | `314e0471675e82b46f8f3028adcc2eccab10820c70a07753302d5a96cb2d5274` |
| 预处理语义状态，规则见 binding.json | `487918c52366e7b31598d87d0fd22fa976cda72e75d6fe329b823e600c71f81c` |
| F 身份 parquet | `55d370fc74c449083ccac807aa7da32ca0521fbbf5063693a9f3e0897a1c92b6` |
| R 身份 parquet | `e2d42f8d004c6b6e9293d44f1d2e855a4b90317a00507b4e494d1f6083b01dee` |
| C 身份 parquet | `d7aebbd137986d44c1ef39f2177176ab76796604005b3804ce73a00ef402f93f` |

## 教师软目标非退化，互补错误支持开展受控验证

使用各工件原先冻结的 R+C 工作阈值，重新交叉统计保存的 C 分数。教师阈值 2.307777003139602；历史 all-hard 阈值 1.524279832839966。比较对象明确是 all-small hard，不是 reduced24。

| C 项目 | full-F all-HGB | all-small hard |
| --- | ---: | ---: |
| TP | 14,448 | 13,969 |
| FP | 76 | 78 |
| TPR | 86.4787% | 83.6117% |
| 标准化 pAUC@FPR≤1% | 0.9118371875 | 0.8893908922 |

教师修复 hard 的 730 个漏报，同时丢失 hard 检出的 251 个攻击，净增 479 TP。正常样本方面，修复 54 个 FP、另增 52 个 FP。修复的 730 个攻击中，Fuzzers 278、Reconnaissance 214、Exploits 138；丢失的 251 个攻击中，Fuzzers 174、Exploits 44。不能用净收益掩盖互补错误，或把教师上界直接当作学生必然可达到的收益。

F 上 mean|q−y|=0.0428883264423215，mean entropy=0.088309198018133 nats；63.3083% 行的 |q−y|≤0.001。λ=0.3、T=1 且归一化之后，平均软目标位移为 0.0098973061020742。普通 KD 并非严格退化为 hard，但整体扰动较小。

按真实二分类标签算得均值 q0=0.0428852927131543、q1=0.9571102499731591；类均值控制的有效目标分别为 0.00989660601072791 与 0.9901023653784212。这是从同一教师得到的约 1% 标签软化控制，能区分样本相关信息与一般软化。q=0.5 的统一教师会施加约 11.54% 软化，与这里的教师扰动强度不同。

## 为什么此次只固定 full-F 教师

OOF 身份与当前 F/R/C 完全对齐，因此问题不在行错位。实际 OOF 三折教师分别只见 57,130、57,290、57,412 行；对应 heldout F 的 mean|q−y| 分别为 0.0616130083、0.0661731363、0.0631831955。把这些目标换入会同时改变教师训练样本量、目标生成方式与目标噪声。

seed17/29/43 的 OOF fold membership、native heldout F、各折 R/C 数组逐值完全一致，三个 `oof-scores.npz` 文件 SHA 也相同。这不是三次独立教师重复。最小实验可直接固定当前 full-F teacher，将问题限定为“同一位 F 内教师的样本相关软目标是否优于 hard、类均值软化及破坏对应关系的控制”；没有依据把 OOF 教师再加入本轮矩阵。

## 置乱控制必须保留重复特征模式的一致性

原教师在全部 49,507 个完整 feature_hash 模式内 q 逐值恒定。F 有 41,183 行属于重复模式，其中 89 个模式含两种二分类标签。常见的标签内逐行置乱会使同一个 x 获得不同 teacher q：固定 SeedSequence([17,353]) 下，4,770 个模式、41,175 行出现此问题，占 F 的 47.9247%、重复行的 99.9806%。这个控制额外加入了相同输入无法区分的软目标噪声。

最终固定模式置换如下，整个构造只使用 F，不按 C 结果筛种子：

1. 按 feature_hash 字典序排列完整模式，计算每个模式的 `(n_normal,n_attack)`。
2. 按该计数向量字典序分层。单模式层保持不动。
3. 使用 `rng=np.random.default_rng(np.random.SeedSequence([17,353]))`；每个多模式层只调用一次 `order=rng.permutation(indices)`，再执行 `donor[order]=np.roll(order,1)`。
4. 将 donor 模式的 q 赋给目标模式全部行，全程固定这一个映射。各可置换模式都换 donor；donor 的 q 数值可能恰好相等。

分层保留了每个模式的类别计数。因此置换后的每类 q 多重集合与原教师精确相同，均值、分位数和软化质量随之保持；完整相同 x 的 teacher q 仍完全一致。它破坏的是同计数模式之间的 x→q 对应，不能理解为消除了所有关于重复频次的信息。

| 覆盖指标 | 全 F | Normal | Attack |
| --- | ---: | ---: | ---: |
| 可置换行 | 82,940 | 27,112 | 55,828 |
| 可置换行比例 | 96.5362% | 98.8047% | 95.4716% |
| 可置换软化质量比例 | 98.3163% | 97.9533% | 98.4866% |
| 实际 q 数值变化行 | 82,045 | 26,456 | 55,589 |
| 实际 q 数值变化比例 | 95.4944% | 96.4140% | 95.0629% |

132 层中，共 49,461 个模式可置换，46 个单模式层、2,976 行保留原 q。另有 146 个已换 donor 的模式、895 行因 donor q 相等而数值不变。类平衡平均 |q−qperm|=0.0655330931222410，对应 λ=0.3 的平均有效目标差 0.0151230214897479。

`frozen_pattern_permutation.npz` 保存完整模式名单、类别计数、donor index、原行到模式 index、原 q 与 qperm。重要数组的 SHA：

- donor_group_index 的 little-endian int64：`55a0a11e05da3eeecd300f0ed38de76a15f52fdb77fa593c457621a00c6640c6`。
- qperm 的 little-endian float64：`d8e453b4baa77d8a01eef0c9b51b46bb88c336479356ea48aa17f2504ae1ede0`。

## 当前实现中必须修改的控制条件

| 当前实现 | 对新方案的影响 |
| --- | --- |
| `_backward` 为 hard 与 raw KD 分别遍历 F，hard+λKL 未除以1+λ | KD 与 hard 的目标尺度和前向预算不一致；应统一一次 student forward，以有效目标 BCE 表达归一化 T=1 KD |
| chunk2048 内累积全 F 梯度，全部 chunk 完成后只更新一次 | 40 macrostep 是 40 次 AdamW 更新，不是 40 epoch 的 minibatch 更新；不得将两者混称或在新实现中悄悄改变 |
| 初始化与每个 continuation step 都用 C 选最优，保存 best | 新方案固定第40步；历史 all-hard 保存 step25，不能替代同期重跑控制 |
| `direct_rank_shuffled` 仅在 ranking 分支做标签内逐行置乱 | 它不是普通概率 KD 置乱，且有上面的重复模式噪声问题 |
| `direct_rank_uniform` 表示均匀采样 pair | 它不是 q=0.5 的教师控制，也不是类均值软化控制 |
| 全 runner 按 all/reduced、多种method、teacher/warmup/refit自动展开 | 不能用现有全命令来承载四臂最小方案 |
| `run_transfer_research` 在冻结后加载 validation、输出 V 结果 | 新方案 F/R/C 控制不能直接调用该流程 |
| `_benchmark` 使用 C 前1/256行、25次重复 | 与最近混合 workload 的成本验证口径不同；不能混比前后延迟或重复完整成本矩阵 |

统一目标的精确定义为 $w_i=0.5/N_{y_i}$，hard 臂 $r_i=y_i$，其余三臂 $r_i=(y_i+0.3q_i^{arm})/1.3$，训练 $\sum_iw_i\operatorname{BCEWithLogits}(z_i,r_i)$。T=1 时它与 $(L_{hard}+0.3L_{KL})/1.3$ 的学生梯度相同，标量损失相差一个与学生无关、随 teacher q 改变的熵常数；不要把各臂原始 loss 大小直接当作优劣证据。

四臂相同 AdamW：lr0.001、weight_decay0.0001、默认 betas=(0.9,0.999)、eps1e−8；每臂重置优化器。每步全 F 一次梯度前向、按原行序 chunk2048 累积、全局类计数归一化、一次 clipnorm10、一次 optimizer.step。固定 40 步后再比较。统一浮点计算规则后，新的 hard 浮点轨迹未必逐位等于旧实现，因此历史 step25 指标不能作为新 hard 的替代值。

预算为零新教师、零新 warmup、四次续训、总计 160 次更新、13,746,560 个梯度前向行。若只在第40步对每臂做一次 R/C 输出，则模型评估行数为 147,284。常规诊断记录复用同一次 forward/backward 的量，不另取 F 推理或增加选择次数。

冻结 anchor 后，当前 hard/rawKD 路径没有 minibatch 随机排列、dropout、数据增广或 pair 抽样；Linear/ReLU 网络与全 F 更新均由固定状态和固定目标决定。控制同一硬件、浮点与确定性设置时，只改训练 seed 不会产生新独立轨迹。四臂是一个固定 anchor 的受控实验，不能把伪 seed 重复报告成多次稳定性验证。

## 交付与执行记录

`inspect_unsw_artifacts.py` 是可复跑的零拟合检查脚本，输出 `binding.json`、`target_diagnostics.json`、`preprocessing_state.json` 和冻结置换 NPZ。它不读取原始特征，不调用 model.transform/predict，也不打开保留分区分数。已实际运行并通过身份、哈希、数组长度、组不相交、预处理一致、q 组一致、置换无自映射和类条件多重集合一致的断言。

交互检查时有一次越界：为枚举旧 `scores.npz` 的结构，曾载入其中两个 validation 数组以取得 shape/dtype；没有输出分值、计算指标或用于方案选择。此异常写入 `binding.json`。不能把本次完整人工检查描述为“严格零 V 读取”；所有本文统计与冻结置换均只使用 F/R/C。
