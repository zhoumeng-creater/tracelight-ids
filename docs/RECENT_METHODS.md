# 近期方法对照：官方 TabM 接入与 FSNID 复现边界

更新：2026-11-03。本页记录代码、依赖和验证范围，不把接口跑通写成论文结果复现。第五轮已经使用过的测试集不因增加近期模型而重新成为未见数据；后续研究须保留版本边界，不能回看该测试集选择新模型或阈值。

## TabM：已接入官方实现

TabM 的论文为 **TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling，ICLR 2025**。采用作者发布的 `tabm` 包，保留它的参数共享集成层，不用普通 MLP 改名代替。官方仓库核对版本为 [`28e47ae301c92ec37787dde1ce923a0793f405b4`](https://github.com/yandex-research/tabm/tree/28e47ae301c92ec37787dde1ce923a0793f405b4)。[作者说明](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/README.md)和[作者论文](https://arxiv.org/abs/2410.24210)各自对应方法与使用说明。

实际适配器是 `src/ids/recent_baselines.py`。它独立于既有 `ModelBundle`，不修改已冻结模型的加载行为。公共入口为：

```python
from ids.recent_baselines import train_tabm, save_tabm_bundle, load_tabm_bundle

config = {"model": "tabm", "hyperparameters": {
    "epochs": 60, "patience": 8, "batch_size": 1024,
    "learning_rate": 0.001, "weight_decay": 0.0001, "min_delta": 0.00001,
    "k": 8, "n_blocks": 2, "d_block": 32, "dropout": 0.0,
    "num_embeddings": "piecewise_linear", "embedding_dim": 16, "n_bins": 48,
}}
bundle = train_tabm(train_df, validation_df, config, seed=17,
                    device="cpu", num_threads=1, feature_schema=feature_schema)
scores = bundle.predict_scores(validation_df)
save_tabm_bundle(bundle, "new-development-run/tabm.joblib")
restored = load_tabm_bundle("new-development-run/tabm.joblib", device="cpu")
```

示例中的小规模网络仅说明 API；正式参数、候选数量和训练上限必须在实验启动前写入对应配置。它不是作者论文的大规模调参结果，也不能仅凭采用近期架构就声称更强。`learning_rate` 可写作 `lr`，两者不能同时设置。`arch_type` 支持 `tabm` 或 `tabm-mini`。`num_embeddings` 支持 `piecewise_linear` 或 `none`，默认前者；`k`、宽度、嵌入和选择字段都记录在模型中。

### 输入、训练和保存

- `feature_schema` 与本项目现有协议相同，包含 `name/features/categorical/discrete`，因此可显式使用 UNSW 或修正版 CIC 的字段定义。标识、标签和家族不进入预测特征。`selected_features` 可限制训练输入字段；选择方法及其合法拟合范围由调用方负责记录。推理仍要求完整原始 schema，按字段名读取，并忽略额外元数据。
- 数值中位数填补、标准化、类别词表及 PLE 分箱均只拟合 `train_df`。类别缺失使用索引 0，未见类别使用 1，训练类别从 2 开始；官方 TabM 内部执行 one-hot。类别字段不被直接当作连续数值使用。
- PLE 使用官方 `rtdl_num_embeddings.compute_bins` 的无监督训练分位数和 `PiecewiseLinearEmbeddings(version="B", activation=True)`。每列分箱数最多为 `min(n_bins, n_train-1)`；两行训练输入只使用训练最小/最大值。训练常量列在标准化后固定使用 `[-1, 1]`，单独写入 `constant_numeric_fields`。官方包会对单箱列提示其等价于线性缩放；这不是从 validation 借用分箱。
- 官方输出为 `(batch, k, 1)`。训练先对每个成员分别计算二分类 BCE，再平均；正常/攻击权重只由训练计数确定。推理先逐成员 sigmoid，再平均概率。不能先平均 logits 再训练。验证早停使用相同的逐成员加权 BCE，保存最佳轮；验证集不用于拟合预处理、生成训练目标或校准最终阈值。[官方训练与推理定义](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/README.md#training)
- `network_state` 全部落 CPU。序列化清除缓存网络、设备改为 CPU，重新加载时重建官方结构；部署时仍需安装相同可选依赖。joblib 只用于可信的本项目工件。`encoded_dim` 表示嵌入前数值加 one-hot 的宽度；`training_statistics.backbone_input_dim` 才是采用数值嵌入后的输入宽度，两者不能混用。
- 训练函数只接受 train/validation DataFrame，不读取文件路径，不执行 `freeze/evaluate`，不自动安装依赖。单条 CPU 计时应由调用方统一设置线程、预热和测量顺序；缓存初始化、模型加载和线程设置放在计时外。比较中仍须计入该方法的预处理及官方模型预测。

### 固定依赖与隔离安装

最小新增依赖固定为 `tabm==0.0.3`、`rtdl_num_embeddings==0.0.12`、`typing_extensions==4.12.2`，复用项目已安装的 PyTorch 2.6.0 CPU/CUDA 构建。适配器强制核对前两个版本，并记录 torch、numpy、pandas、scikit-learn 实际版本。没有安装时明确报错，基础 CLI 和既有模型不因此新增必选依赖。主环境及主锁文件不被自动升级。

实验室操作沿用 [Stage6 操作指南](STAGE6_LAB.md)中已核验的**绝对解释器路径 `$labPython`**。先用这个解释器确认锁定依赖、torch 版本和 CUDA，再向同一个环境显式安装可选包；不要依据终端是否显示激活标记或 PATH 中的 `python` 推断实际环境。在仓库根目录执行：

```powershell
# $labPython 已按 Stage6 指南指向并核验实验室 python.exe 的绝对路径。
if (-not (Test-Path -LiteralPath $labPython -PathType Leaf)) { throw '请先核验 labPython' }
& $labPython -c "import sys, torch, numpy, pandas, sklearn; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(numpy.__version__, pandas.__version__, sklearn.__version__)"
if ($LASTEXITCODE -ne 0) { throw '既有实验室依赖尚未就绪，先按锁文件和 LAB.md 重建' }

# --no-deps 阻止本次可选包安装自动解析、升级或替换 torch/CUDA 依赖。
& $labPython -m pip install --no-deps tabm==0.0.3 rtdl_num_embeddings==0.0.12 typing_extensions==4.12.2
if ($LASTEXITCODE -ne 0) { throw '可选包安装失败' }
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $labPython -c "from ids.recent_baselines import recent_dependency_versions; print(recent_dependency_versions())"
if ($LASTEXITCODE -ne 0) { throw '可选包版本核验失败' }
& $labPython -m pip check
if ($LASTEXITCODE -ne 0) { throw '依赖一致性检查未通过' }
& $labPython -m pytest tests/test_recent_baselines.py -q
```

版本打印只是检查证据，不能代替 Stage6 的版本及设备准入。上述命令不会自动修复 torch；缺失或版本、CUDA 构建不符时，先按项目锁文件及 [LAB.md](LAB.md) 显式重建并验证环境，随后重新固定 `$labPython`。正式实验保存这个解释器的完整环境清单。

本机已完成的工程验证使用了一个 `--system-site-packages` 临时 venv，复用的是**基础 Python 中已经安装的 torch**，三个新增可选包写入临时 venv。这说明该本机配置可行。若从另一个已激活 venv 内运行 `python -m venv --system-site-packages`，新环境可见的系统包路径并不保证包含原 venv 的包，因此**不能假定会继承原 venv 中的 torch**。需要新隔离环境时，先检查新解释器的实际导入与版本；若缺 torch 或项目依赖，就按锁文件和明确的 CPU/CUDA wheel 安装步骤完整重建，不盲目依赖嵌套继承，也不让可选包安装自动升级 torch/CUDA。

本次对真实官方包运行 **15 项测试，15 通过、0 跳过**：包含 PLE/无嵌入两条实际拟合路径、CPU 保存重载数值一致、确定性、未知类别、常量列、标签不参与预测、训练内预处理/分箱和早停恢复。环境为 Python 3.12.4、torch 2.6.0+cu126、numpy 1.26.4、pandas 2.2.2、scikit-learn 1.5.1；这 15 项使用 CPU。另在本机 RTX 4050 Laptop GPU 上完成一轮合成数据 PLE 冒烟：48 行训练、16 行验证、1 epoch，训练使用 CUDA，保存权重全部位于 CPU，CPU 推理返回 16 个有限分数。此记录不代表实验室 4090D 运行。全局环境未装 tabm 时这些可选真实训练测试会明确跳过，不能把跳过记作已验证。上述验证没有运行真实数据模型，也没有重新打开真实 test；正式研究结果由对应运行记录证明。

包身份可从 [TabM 0.0.3 PyPI](https://pypi.org/project/tabm/0.0.3/) 和 [rtdl_num_embeddings 0.0.12 PyPI](https://pypi.org/project/rtdl-num-embeddings/0.0.12/)核对。平台无关 wheel 的 SHA-256 分别为：

```text
tabm-0.0.3-py3-none-any.whl
6cabed3cb436fef5e8a45f46097fd745f51b3e16be16d555b9af16cffda7f944
rtdl_num_embeddings-0.0.12-py3-none-any.whl
87fd61270118915cf40888f2164f3a1f0353edf702f15fd91e71a147d23c930d
```

TabM 使用 [Apache-2.0](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/LICENSE)，外部包保持独立安装；本仓库没有复制作者实现。适配器及训练设计需引用原论文，项目结果不得借用论文分数。

## FSNID：已审查源码，尚未执行或声称复现

FSNID 对应 **Feature Selection for Network Intrusion Detection，KDD 2025**，DOI [10.1145/3690624.3709339](https://doi.org/10.1145/3690624.3709339)。核对的[作者仓库](https://github.com/c-s-westphal/FSNID/tree/3c23d48984a7b16011cd3cca7f9d5bd49e2f8c85)提交为 `3c23d48984a7b16011cd3cca7f9d5bd49e2f8c85`。不能把已有单变量 MI、稳定 MI 或去冗余 MI 改名为 FSNID。

| 审查项 | 官方实现与本项目之间的差异 | 可靠接入所需工作 |
|---|---|---|
| 数据入口 | `main.py` 读取 `<DATASET>_X.npy` 与 `_Y.npy`，无字段类型、单位、原始字段组清单；只附 BOTIOT 示例 | 建立具名 schema→矩阵→原始字段映射，记录编码后的筛选单位；不可无说明地让 one-hot 列与 24 个原始字段预算比较 |
| 预处理与分区 | 官方入口对整个加载矩阵做 min-max 后选字段，并将相同输入分别作为分类器 train/test 参数 | 只在 train 拟合预处理与选择器；保留本项目分区和组隔离，validation 仅用于开发；这是协议适配，不能宣称原样复现官方入口 |
| 选择目标 | 顺序删除无足够条件信息贡献的字段，输出字段数由规则决定，不接受固定 k | 记录自然输出字段数并补同字段数 MI 基线；不能强截到 24 后仍称原算法复现 |
| 计算成本 | `fsnid_selection` 默认 `num_iterations=10000`；MLP `mine_fa.fit_mlp` 里此数实际控制完整 epoch 循环，每次信息估计含三次重复、每次有输入/空输入两段训练 | 首先只在 train 的固定小样本上测一次完整信息估计，记录实际 epoch/batch、时间、峰值内存及种子；不能把 10000 当作总共 10000 个 minibatch |
| 随机性与设备 | numpy/torch 全局随机数、自动选 CUDA；入口没有当前项目的确定性与可恢复运行绑定 | 独立进程固定随机状态/设备，保存 train 身份、字段顺序、估计轨迹和源码哈希，禁止结果导向重试 |
| 依赖 | 作者锁定 torch 2.2.2、numpy 1.24.4、pandas 2.2.3、scikit-learn 1.5.2 等，与项目主环境不同 | 单独环境验证，不能向主环境直接覆盖安装；升级作者版本时逐项记录差异 |
| 时序版本 | 作者支持 MLP/LSTM/TCN/GRU，但当前 UNSW CSV 不具备足以构造可靠时间序列的元数据 | 第一入口限定 MLP 版本，不拼接相邻 CSV 行冒充时序 |
| 许可 | 此次递归文件树及 README 中未发现 LICENSE/COPYING 或明确代码许可声明 | 保留公开来源和审查记录；需要发布作者代码衍生版本前另行明确许可，不把公开可读等同于已获许可 |

按当前源码静态估算，d 个字段先做一次空模型估计、d 次单字段排序，再做至多 2d 次删除比较，即至多 `3d+1` 次信息估计。若全用默认 MLP 设置，每次包含 `3×2×10000` 个训练 epoch，d=42 时上限量级为 762 万个小网络 epoch，还须乘每个 epoch 的批次数。该估算不等于实际机时；删除路径、样本数、batch 和设备决定实耗，也不保证提前缩减 iterations 后还与论文设置等价。[选择入口](https://github.com/c-s-westphal/FSNID/blob/3c23d48984a7b16011cd3cca7f9d5bd49e2f8c85/feature_selection_methods/fsnid_fs.py)、[信息估计训练](https://github.com/c-s-westphal/FSNID/blob/3c23d48984a7b16011cd3cca7f9d5bd49e2f8c85/mine.py)和[环境清单](https://github.com/c-s-westphal/FSNID/blob/3c23d48984a7b16011cd3cca7f9d5bd49e2f8c85/requirements.txt)是此判断的直接依据。

后续最小入口为独立环境中的 `fsnid_selection(X_train_only, y_train, num_iterations=<事前固定小试预算>, model_type="MLP")`，先审计初始化阶段的空模型成本，再运行 `.run_main()`。将所选字段交回本项目既有 RF/MLP 流程，使用相同训练预算和开发误报规则。小试不足以支持默认参数或论文成绩复现；若无法在约定成本内稳定完成，保留失败和差异记录，不自动增加迭代、字段预算或查看已用测试集寻找优势。本次没有执行这一步，也没有新增 FSNID 预测结果。

## 门控蒸馏与低误报排序：近期先例核查

核查日期：2026-11-03。下列工作按与本项目问题的相关性收录。本文只核查作者全文、出版方记录及明确对应的代码身份，**没有执行或复现这三项方法**，不借用其报告的效果作为本项目结果。本节不改变已启动 pilot 的源码、配置或预算。

### 1. 正确性与置信度联合门控已有明确先例

Huang 等的 **Trust the Right Teacher: Quality-Aware Self-Distillation for GUI Grounding** 为 2026-06-16 发布的 **arXiv 预印本**，编号 `2606.18101v1`；本次未核实其正式会议或期刊接收身份。该文联合使用正确性门控与教师概率缩放，并分别消融两个模块及其组合。正确性由坐标 token 在学生已生成前缀下能否继续完成到真值框判断，未通过者降权，再按教师概率调整蒸馏强度。[作者全文与式 (3)–(6)](https://arxiv.org/html/2606.18101v1)

因此，**不能声称“正确性＋置信度加权蒸馏”这一通用组合由本项目首创**。本项目处理 OOF HGB 的二元攻击分数，按教师在 0.5 两侧与训练标签是否一致判断方向，门控被拒绝时恢复 hard 监督；其 BCE 目标、分组数据和低误报场景与 GUI 自蒸馏不同。这些是需要验证的任务适配与机制差异，尚不足以单独证明算法创新。本次未核实该预印本的作者代码入口。

### 2. IDS 蒸馏的类别先验与 hard 对照值得直接比较

Suhaila Abuowaida 的 **Class-Aware Knowledge Distillation for Detecting Rare Attacks with Tiny IoT Intrusion Detection Models** 发表于 **Journal of Computing & Biomedical Informatics，11(02)，2026-09-01**，DOI `10.56979/1102/2026/1676`。期刊原页报告：普通 KD 可损害稀有攻击，MA-KD 对蒸馏项按类加权并去除教师 logits 中的先验影响；文中也保留了同规模加权 hard 学生不弱于 MA-KD 的负结果。[出版方原页](https://www.jcbi.org/index.php/Main/article/view/1676)

此文用于定位类别先验、有效权重和学生容量的混淆，不以其期刊页面或发表年份作为方法质量、排名或性能优势的背书。本项目已有 matched-hard 和 gated-matched-hard 控制，应据此检验收益是否来自软信息。MA-KD 针对多类别设置；把其思想改用于二元 HGB 分数需重新说明数学目标与先验含义，不能称原样复现。本次未核实可运行的作者代码，也未审计其数据切分和性能报告，故目前只作为相关工作及待评估对照。

### 3. 低 FPR 对照应包含明确区间的 pAUC 目标

Jiang 等的 **Closing the Approximation Gap of Partial AUC Optimization: A Tale of Two Formulations** 刊于 **IEEE Transactions on Pattern Analysis and Machine Intelligence，48(3):3482–3498，2026**，DOI `10.1109/TPAMI.2025.3637063`；作者全文对应 2025-12-01 的 `arXiv:2512.01213v1`。文中区分 PAUCI 与 UPAUCI，以阈值学习、逐样本 minimax 改写和不同平滑处理研究指定 FPR/TPR 区域的优化与近似误差。[作者全文](https://arxiv.org/html/2512.01213v1)、[期刊 DOI](https://doi.org/10.1109/TPAMI.2025.3637063)

本次核实的 [Shaocr/PAUCI 作者仓库](https://github.com/Shaocr/PAUCI/tree/c08d633d195a9ab1799b283696b02d215b7d29b9)明确对应 **NeurIPS 2022 的前身论文**，包含 `losses/SPAUCI.py` 和 `optimizer/MinMax.py`；不能将其标成 2026 年 UPAUCI 官方实现。尚未核实 2026 年新算法的独立代码版本，未运行其中任何方法。

当前 `research_kd.py` 从 batch 内选择最高分的正常记录与最低分攻击，优化带上限的 smooth-margin 成对损失，并对选中组分配质量。配置 `normal_fraction=0.1` 选择 10% 的 batch 正常流，评价目标则是 1% FPR；双端截取、pair 上限及组权重还会改变训练风险。该目标应准确称为**以低误报为动机的 minibatch 困难对 surrogate**，不能称精确优化 `[0, 0.01]` pAUC，也不提供总体 FPR 的 Neyman–Pearson 保证。

### 后续有价值的对照与贡献标准

等当前 pilot 完成，再依据差异决定是否另立实验版本；新增配置、样本边界与预算均先落盘：

1. **同 pair 数随机配对**：沿用同一学生、组质量规则、pair 数和损失系数，只替换困难对的选择方式，检验收益是否来自困难样本选择。必要时再拆分“只选困难正常”“只选困难攻击”与双端选择，不能同时改变多个因素后作单因归因。
2. **明确 FPR 区间的目标**：评估可正确实现、可核对来源的 pAUC 对照，固定 `beta=0.01` 及相同学生和训练预算。算法和优化器未跑通时记录未完成，不能用当前 batch 排序代称已实现 PAUCI/UPAUCI。低 FPR 区域面积与单个冻结阈值的召回分别报告。
3. **解释软信息与权重的区别**：继续完整报告 matched-hard、correctness-only、confidence-only 和 gated-matched-hard；结合 gate 覆盖、家族及组诊断，检查门控是否确实拒绝有害教师，而不是仅改变逐行权重。

研究贡献应落在有控制实验支持的机制、适用条件和部署取舍上。教师可靠性加权、困难负样本挖掘、pairwise margin 和组平衡各有既有基础；给它们的组合命名不能替代相关工作对比。旧 test 的已使用状态保持原记录，新候选不得借已见测试结果继续选择或调阈值。
