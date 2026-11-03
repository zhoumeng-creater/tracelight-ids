# 风险约束蒸馏候选：v3 最小接入审查

日期：2026-11-03。性质：只读源码审查与建议接口；下列新接口均为**建议，尚未实现或运行**。本轮未加载真实训练数据、未训练、未修改 v2 源码/配置/历史工件。此处 v3 是独立研究版本标识，不表示新颖性或有效性已经成立。

## 1. 结论

建议新增独立的 v3 loss、研究 runner、实验室入口、配置与 synthetic 测试。保留 v2 的配置契约、损失公式、停止规则、完成清单和运行包。复用现有数据验证、预处理、标量 MLP、CPU bundle 和指标底层函数；不要直接复用 v2 总调度或把新模式加入旧损失枚举。

三个优先风险：

1. `constraint-audit` 若用于梯度、对偶变量、阈值、选轮或配置选择，只能称 **constraint-development**；它不再是独立风险证书。真正独立 audit 必须从模型/教师/阈值/选择全部冻结后才读取，若据 audit 继续修改，就要保留探索标记并另找确认来源。
2. CIC 必须按已准备的 **feature/session 联合传递闭包 `group_id`** 划分所有角色与 OOF；旧 `ids.distillation` 仅按 feature hash，不适合本轮直接复用。
3. v2 `--resume` 只复核完整结果，不支持 epoch 中断续训。新约束状态不能只保存部署模型；需另有版本化 training checkpoint，或明确暂不支持中断续训。

## 2. 现有 API 与最小复用点

| API / 位置 | 已有行为 | v3 建议 |
| --- | --- | --- |
| [`runner.verify_development_inputs`](../src/ids/runner.py) 73–87 | 只允许 train/validation，校验 manifest、相应 parquet、audit；不打开旧 calibration/test | 直接复用，不扩大 `splits`。manifest 含旧分区的元数据不等于读取其样本；记录实际读取清单。 |
| [`data.load_partition`](../src/ids/data.py) 342–369 | 按 schema 分发 UNSW/CIC，核对应分区；通用函数本身允许所有正式分区 | v3 runner 只作两个显式调用 `train`、`validation`，再在内存中从 train 划新角色；不要暴露任意 `--split`。 |
| [`research._group_column` / `_identity`](../src/ids/research.py) 91–110 | `group_id`→`group_hash`→`feature_hash`；唯一 sample ID、二元标签、非空身份 | 借鉴规则；CIC 明确要求 group_id，不能悄悄退化。v3 身份表额外保留 session_hash、角色、fold 和可用来源元数据；不把它们送入特征。 |
| [`research.group_budget` / `_sample_record`](../src/ids/research.py) 113–141 | whole-group 预算子样本与家族/标签支持记账 | full 保留全部开发行；pilot 可以复用，但只能在固定 pilot 总体内再划角色，记录被排除大组/家族。 |
| [`selection.select_features`](../src/ids/selection.py)；[`models.FeaturePreprocessor.fit/transform`](../src/ids/models.py) 104–139 | 填补、标准化、类别词表/onehot 均由传入 fit frame 得到，metadata 不进入选定 schema | 学生只在 `fit` 侧选择与拟合；constraint-dev、只读 audit、validation 一律 transform。教师每个 OOF fold 也独立 fit。 |
| [`research.make_oof_teacher`](../src/ids/research.py) 144–193 | 按联合组做 StratifiedGroupKFold；逐折 train-only 预处理/HGB；每行唯一 OOF；另 full reference 看 validation | 可复用结构；新函数显式接收 fit/constraint-dev/validation。full reference 只训练 fit，然后给其他角色评分。不要先对原 train 产 OOF 再切角色。 |
| [`models._network`](../src/ids/models.py) 174；[`ModelBundle`](../src/ids/models.py) 186–251 | 单标量 MLP、CPU sigmoid 分数、预处理和 schema；推理不读标签 | 先保持相同学生结构。约束/教师只影响训练，部署仍是普通 `mlp`，无需改旧模型注册。训练辅助头若必须参与推理，则不能冒称已有 bundle 无缝兼容。 |
| [`models.save_bundle/load_bundle`](../src/ids/models.py) 513–550 | 原子写出、CPU 权重、加载检查、清理 cached network | v3 直接复用。训练 checkpoint 单独存，避免把教师、对偶历史或逐样本状态塞进部署包。 |
| [`evaluation.calibrate_threshold/evaluate_binary`](../src/ids/evaluation.py) 64–123 | normal-only 经验阈值，float64 `score >= threshold`；固定阈值评价 | 可接 v3 合法角色；它不会自动产生统计风险保证。若算法学习阈值，不再像 v2 一样无条件在 validation 重标阈值覆盖训练目标。 |
| [`benchmark.benchmark_predict`](../src/ids/benchmark.py) 46–88 | 接受任意 predict callable，分别测 batch=1 和固定 batch，包含 callable 中预处理 | 可直接给 CPU bundle.predict_scores；正式实测留实验室。不要为复用旧 CLI 而伪造 v2 run manifest。 |

## 3. 不应直接复用的现成调度

- [`research.validate_config`](../src/ids/research.py) 40–88 固定 `schema_version: 1`、`purpose: research_v2_development_only`、精确键集合、HGB 教师和 MLP/TabM 类型。新增约束、audit、checkpoint 配置应进新 validator；不放宽旧 validator。
- [`research.train_research_mlp`](../src/ids/research.py) 196–317 把全部传入 train 用于预处理和类权重，validation weighted-hard-BCE 早停，loss 仅为 batch 内无持久状态调用。不能直接通过把 audit 当 validation 或把状态藏进 config 来实现新算法。
- [`research.run_research`](../src/ids/research.py) 448–552 固定先 OOF 全传入 train、再循环学生、validation 重选阈值。其 `resume`（460–467）先要求 `RESEARCH_COMPLETE.json`，未完成目录不会续训。
- [`research.verify_research`](../src/ids/research.py) 324–356 固定 v2 文件结构与 `(folds+1)×seeds` 教师计数；[`scripts/run_research_lab.py`](../scripts/run_research_lab.py) 56–65 固定载入 v2 profile，127–165 打包前后硬调用 v2 verifier，211–274 硬调用 v2 runner。v3 不能调用这些总入口后只改输出状态名称。
- [`distillation._fold_assignment`](../src/ids/distillation.py) 73–86 仅按 `feature_hash`；`build_oof_targets` 109 起、`verify_oof_targets` 250–255/298 还绑定完整 prepared train，不能原样用于 v3 fit 子集和 CIC 联合组。

## 4. train 内角色与联合分组

建议的显式角色流：

```text
prepared train ─固定联合组划分─→ fit ─→ student fit / fold-local OOF teachers
                           └→ constraint-dev ─→ 指定的约束更新或选型（开发）
prepared validation ──────────────────────→ 预先约定的开发评价/停止
旧 calibration、旧 test ──────────────────→ 不读取、不哈希分区样本
真正新来源 ─冻结方案后────────────────────→ 另立独立确认协议
```

如果本轮必须区分用于风险更新的数据与独立 train 内 audit，则划为 fit / constraint-dev / audit 三个互斥组集，并明确 audit 是冻结后一次性读取的诊断；不得一边每 epoch 看 audit 选轮，一边称其独立。小原型可先只做 fit/constraint-dev，不承诺额外留出集能满足 1% 尾部支持。

建议 `make_training_roles(train, settings) -> roles, split_manifest`：

- 固定 split seed、分配算法、角色比例/最小支持、身份排序与 schema；一次生成、按 seed/variant 共用角色，不根据效果换划分。允许用 train 内标签作预先声明的分层；这不使相关组变成独立环境。
- 角色之间检查 sample_id、feature_hash、group_id（CIC 还可检查 session_hash）均不交叉，保存逐行角色与各角色完整家族/正常支持。整个原 train 都进入已声明角色，不把过大组悄悄丢弃。
- CIC 的单个大 Portscan 联合组可能使比例无法精确，也可能使某角色缺该族。保留整组，报告偏差/缺失；预先声明必须在 fit 的支持条件，无法满足便拒绝协议，不挑效果更好的拆分、不拆组凑行数。
- 只有正常样本的约束角色不必强求攻击支持，但不能据它报告困难攻击保护。若只抽纯正常组，其评价总体发生变化，必须声明混合/冲突组的处理，不能外推为全部正常流风险。
- 所有 `positive_weight`、MI/选择器、填补/缩放/词表仅使用 fit。audit/constraint-dev 改值不能改变 fit 的预处理、OOF 目标和初始化；参与约束更新时可以按设计改变后续模型，但应有清晰的依赖日志。
- 拆出角色后，v3 实际 fit 总体与旧 v2 不同；所有新对照都按相同角色重训，不能把跨版本分数差直接解释为新算法收益。组等权风险、最坏组风险与真实逐行正常 FPR 分列，写清分母；相关组行数不能当独立风险样本数。group/家族可作训练与审计元数据，部署预测不读取它们。

CIC 分组依据已在 [`cicids.py`](../src/ids/cicids.py) 187–201、231–287 实现：规范化双向端点/协议；时间重叠或设定不活跃间隔形成 session，再与完全相同特征关系做 union-find 传递闭包。输出 `group_id == group_hash`（473–475），loader 541–548 要求这些列。它是启发式相关单元，不是物理主机/真正独立会话的证明。UNSW 原 CSV 只有 feature hash 组时，也不能声称已按真实 session 隔离。

已有合成反例可借鉴：[`tests/test_cicids.py`](../tests/test_cicids.py) 57–81 构造 A 与 B 同特征、B 与 C 同反向 session 的链，三行必须一起分配；[`tests/test_research.py`](../tests/test_research.py) 45–54 检查 group_id 优先。v3 的角色划分和 OOF 应各复用这个反例，而不是只测重复行。

## 5. 建议的新接口与最小文件范围

名字可由实现方调整，保持版本隔离和角色显式性：

```python
validate_v3_config(config) -> None
make_training_roles(train, settings) -> (role_frames, split_manifest)
make_v3_oof_teacher(fit, evaluation_roles, teacher_config, *,
                    seed, folds, fold_seed, schema, threads, output)
train_v3_student(fit, constraint_dev, validation, model_config, *,
                 schema, teacher_targets, seed, checkpoint_dir, resume=False)
verify_v3_research(output) -> completion_record
run_v3_research(data_dir, config, output, *, device, dry_run=False, resume=False)
```

最小新增范围：独立 loss 模块、`research_v3.py`、`run_research_v3_lab.py`、新 smoke/dev YAML、对应 loss/runner/lab tests、本审查/协议。先验证候选及能击穿其解释的少数控制，不立即复制 65 次学生矩阵。旧 `research.py/research_kd.py`、v2 YAML、旧 verifier 与旧运行包保持原定义。

v3 lab 可借鉴/复用无训练的 GPU 名称/环境门禁、schema 核查、路径拒绝 symlink、ZIP 清单与 SHA 回读逻辑；打包必须调用 v3 verifier。新 profile 和 workdir 单独绑定，GPU 仍默认 RTX4090D，禁止 CPU 回退，环境串行。保留 UNSW-only bounded pilot 与 CIC full 不截行的边界；不下载、不重新 prepare、不调用 `evaluate_frozen`。

## 6. checkpoint：部署权重与训练状态必须分开

当前 v2 只在内存保留 best 网络权重（[`research.py`](../src/ids/research.py) 300–309），没有 optimizer/loss/RNG 持久化。最小可实现策略是**仅支持完整 epoch 边界恢复**；尚未实现时 CLI 明确拒绝未完成目录，不把重跑称为恢复。

| 必存内容 | 原因/检验 |
| --- | --- |
| schema/algorithm/training protocol version、配置 canonical hash、源码 hash、数据 manifest、角色/身份/fold 顺序 hash、teacher targets hash | 防止错数据、错样本索引、错算法状态恢复。不能只检查 seed 相同。 |
| latest network state、全部 optimizer state、scheduler/AMP scaler（若使用）、已完成 epoch/global step、下一更新阶段 | best 权重不等于当前优化轨迹；Adam 动量和步骤不能重置。 |
| loss/state module 的 state_dict：可学习阈值/分位点、对偶变量、moving averages、样本/组状态、CVaR/pAUC 辅助状态（按实际算法存在项） | 新损失不一定无状态；逐样本状态必须绑定稳定 sample_id/索引表。 |
| best network/loss/threshold 对应快照、best epoch/metric、stale/patience/min_delta、可行性/选轮规则 | 不能将 best 网络与另一个 epoch 的阈值/对偶状态拼装；feasible 与 infeasible 选择顺序事先固定。 |
| Python random、NumPy、torch CPU、全部 CUDA RNG、各 DataLoader generator；约束 sampler/cursor/epoch | fit 与约束流可能消耗不同 RNG；恢复后 batch 顺序和更新次数必须可核。中途 batch 恢复还需 permutation/cursor，初版可不支持。 |
| 历史账本、已累计运行时间、失败/恢复次数、环境/依赖与设备身份 | 重启不能抹去成本；同设备 deterministic 合成续跑可比，不承诺跨 GPU/库版本 bitwise。 |

使用同目录临时文件、原子 replace、SHA sidecar/清单；恢复先校验再反序列化可信工件。每 epoch 写 `latest`，选择成功另写 `best`，完成后将 CPU model bundle 和最终固定阈值导出。完成清单闭合校验所有实际文件，未完成状态不能打成“成功 ZIP”。

部署分数规则还需单独核对：现有 bundle 输出 float32 sigmoid 概率，评价再转 float64；若训练在 logit 阈值上约束，转换/饱和/tie 可能使边界判定改变。应对**实际保存重载后的 CPU 分数与最终比较规则**重新计算开发约束统计，明确 `>=` 与阈值单位；不能只验证 loss 的理论单调性。

## 7. 最小 synthetic 验收与工件

必要测试应验证研究边界，而非只镜像新公式：

1. 用临时 synthetic prepared 数据，删除旧 calibration/test 文件并拦截 loader/哈希调用；v3 仍只读取 train/validation。dry-run 不训练、不读 parquet 样本；真实 lab wrapper 拒绝 synthetic/错误 GPU。
2. 联合传递闭包不跨 fit/constraint-dev/OOF；缺 group_id 的 CIC、重复 ID、非空交叉、巨大单组无法满足支持均明确拒绝或按预先规则报告，不随机补行。
3. 更改 validation/audit 的特征或标签，fit 预处理、OOF fit 目标、初始化不变；teacher 所有拟合 sample IDs 均不含这些角色。恶意打乱 targets 行序不得静默按位置接受，须按 ID+identity 对齐/拒绝。
4. 同一 synthetic 轨迹比较不间断训练和 epoch 边界恢复：网络、optimizer、loss/dual、阈值、batch 顺序、best epoch 及决策一致；故意改角色/config/teacher hash 拒绝恢复。
5. 极端/单类 batch、无可用组/无可行 checkpoint、dual/阈值非有限值明确处理。相同预算的 hard/标签匹配/无约束控制与候选共享 fit、初始化及 batch 计划。
6. CPU save/load 分数与决策一致；推理改标签/家族不变；辅助损失状态不进入部署必需路径；没有真值的输入不生成 FPR/TPR。

v3 应单独生成 `plan.json`、角色身份/fold 表、OOF targets 与 fold 模型、reference 在各开发角色上的预测、student model/固定阈值、training checkpoint/history、逐角色固定阈值预测、运行/失败成本、summary/report、新名称的 COMPLETE 清单与 ZIP/SHA。fit/constraint-dev 上统计均标开发；validation 也因旧 test 已开且存在模型选择，不能写成新独立确认。

## 8. 旧版本指纹和包边界

[`common.code_fingerprint`](../src/ids/common.py) 54–72 会遍历**全部** src/configs/scripts 的指定文本文件。因此即使不改 v2 文件，仅新增 v3 源码/配置也会改变全仓 source SHA，旧 v2 resume 会按既定规则拒绝。这是现有机制的正常结果，不应通过放宽 v2 校验来隐藏。

旧 v2 工件和运行包继续按原封存源码执行/核验；v3 使用新来源快照与 workdir。若未来要按版本做依赖闭包指纹，应单独设计、迁移并留版本标记，本轮不顺手改变所有历史工件的解释。训练期间源码变化仍应拒绝写完成标记。
