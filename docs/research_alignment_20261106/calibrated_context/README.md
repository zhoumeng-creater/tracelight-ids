# 校准裕量敏感的上下文压缩：设计证据

2026-11-06。以[主设计稿](../../CALIBRATION_AWARE_CONTEXT_DESIGN_20261106.md)为准。已完成设计、固定源码审查、数学检查，以及[完整／压缩 KV 最小实现](../../CONTEXT_CACHE_IMPLEMENTATION_20261106.md)的随机官方架构合成核查；未加载真实模型权重、未执行 IDS 方法对照或资源基准。

## 文件

| 文件 | 用途 |
|---|---|
| [objective_and_novelty.md](objective_and_novelty.md) | 逐源状态目标、直接近邻、近似边界及反证 |
| [architecture_and_cost.md](architecture_and_cost.md) | 官方缓存语义、可微适配接口、完整成本和部署依赖 |
| [sources_objective.json](sources_objective.json) | 2025—2026 年方法的来源与采用状态 |
| [sources_architecture.json](sources_architecture.json) | 官方固定 commit、文件定位与工件可用性 |
| [geometry_reference.py](geometry_reference.py) | 六种合并的准则检查、梯度、质量与阈值算术 |
| [geometry_reference_results.json](geometry_reference_results.json) | 准则检查结果 |
| [solver_reference.py](solver_reference.py) | 实际确定性初始化与 15 轮迭代检查 |
| [solver_reference_results.json](solver_reference_results.json) | 迭代结果，包括普通方法也得到正确分区的记录 |
| [implementation_synthetic_report.json](implementation_synthetic_report.json) | 官方小型随机架构、普通 QCOC 编译、M=N 和新进程恢复结果 |
| [implementation_checks.json](implementation_checks.json) | 36 项针对性测试及静态核查记录 |

## 结果如何解释

1. 四组状态压到三槽，普通准则与已撤回的全头共享敏感度准则的穷举最优分区会合并关键状态，校准后 TP=0；逐源状态准则选择背景合并，TP=1000，FP 均为 0。
2. 把确定性欧氏最远点初始化和实际迭代补进来后，普通与逐状态方法都得到背景合并，均为 TP=1000、FP=0。因此该案例支持准则的机制差异，**不支持迭代算法的性能胜出**。
3. 对错误分区，正常尾部与攻击的注意力权重相同；固定 key 后任意调整 value 都无法恢复区分。这是该单头构造中的严格性质，不是关于完整 TabICLv2 的普遍结论。
4. 前两个数学程序未执行 QCOC 最后 V 拟合或真实多层模型。新增适配器已在随机初始化的小型官方架构上执行普通 QCOC 完整编译与冷进程恢复，但没有运行预训练模型或新敏感几何。核心实证问题仍是完整编译后、候选自身重新校准的检出与完整成本。

两个程序只使用 Python 标准库，按路径直接运行可重建同名 JSON。它们不读取数据集、模型、旧 test 或网络，也不触发训练。

## 来源边界

主干为 2026 TabICLv2，官方源码固定 `d83091b845a8c5f7c087bd54964f755a9e560f45`。QCOC 为 2026-10 预印本，本次未确认作者代码；不能把第三方示例称为官方复现。CRT 为 NeurIPS 2025。TACO 为 ICML 2026 的外部结构比较。更早方法不进入新方法推荐。

候选贡献限于逐源状态的校准裕量几何；普通缓存复用、质量、原 N、value 拟合和校准反馈分别归属近期来源。真实结果若显示 QCOC 或同标签 V-only 已足够，应撤回新增几何的必要性。
