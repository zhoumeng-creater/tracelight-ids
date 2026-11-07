# 完整 KV／压缩 KV 适配器与最小编译入口

2026-11-06。已实现 TabICLv2 固定源码的完整 KV 构建、普通 QCOC 编译、加权 KV 推理、独立导出与冷加载。本机验证只使用合成数据和随机初始化的小型官方架构，没有下载预训练权重、读取真实数据或执行正式成本基准。

2026-11-07 后续：[原始 CSV、误报阈值与实验室最小对照](CONTEXT_CACHE_CSV_LAB_20261107.md)已接通，并完成 110 项合成回归和完整／QCOC 四文件包冷加载核查。下文保留 11-06 原生张量阶段的接口与记录；其中“未接预处理／阈值”描述该阶段，不代表后续 CSV 部署包。

## 1. 本轮接口

| 模块 | 已实现内容 |
|---|---|
| [cache.py](../src/ids/context_cache/cache.py) | 独立 `WeightedKVEntry`，保存 K/V、log_mass、原始 N；批切分、拼接、迁移、纯张量保存加载 |
| [attention.py](../src/ids/context_cache/attention.py) | 先 QASSMax(Q,N)，再 1/√d；质量 bias 广播；SDPA 不重复缩放 |
| [compiler.py](../src/ids/context_cache/compiler.py) | QCOC 2026 普通联合 KV 原型、真实质量、分块闭式 V 拟合；M=N 恒等分支 |
| [tabicl_adapter.py](../src/ids/context_cache/tabicl_adapter.py) | 官方支持前向与 φQ 捕获；复用 Stage 1/2，独立执行 Stage 3；包含权重的完整／压缩导出包 |
| [upstream.py](../src/ids/context_cache/upstream.py) | 核对官方模型源码哈希，不把任意安装版本冒称固定版本 |
| [命令行入口](../src/ids/context_cache/__main__.py) | `synthetic-check`、`compile`、`predict` |

本轮先完成完整模型和普通压缩的对照底座。逐状态校准裕量几何、CRT 式 V-only 更新、风险阈值校准和真实模型排行没有接进此入口。

## 2. 输入与部署边界

接口接受**已经预处理的原生数值张量** `X[B,T,F]`。每个 B 是独立上下文，使用相同 N、F；它不是 sklearn 的集成维度。正常标签为 0、攻击为 1，每份支持上下文都包含两类。没有类别置换、特征置换或集成聚合。

完整支持缓存构建使用 FP32。固定官方代码的标签编码器调用 `one_hot.float()`，不支持直接以 FP64 骨干建缓存；接口会提前报错。已经构建的预测器可迁移到 FP32/FP64，独立 cache/compiler 的数学检查也支持两种精度。

原始 CSV 的填补、编码、归一化仍由上游处理。`feature_names` 与 `preprocessing_id` 绑定已经冻结的预处理，查询端必须逐项一致；标识本身不证明训练／评价隔离。当前包没有打包外部预处理器，因此 `size_report` 明确标为原生模型与缓存的张量字节，不称为原始 CSV 到报警的完整系统成本。

导出包包含骨干权重、列 KV、紧凑 Stage 3 KV、质量、原 N、结构配置与来源信息；不含原始支持行、标签数组、完整行表示、拟合查询和 Gram 矩阵。冷进程可以只加载导出包与符合契约的查询张量进行预测，无需原 checkpoint 或支持文件。数值张量预测输出尚未校准，不生成报警判定。

## 3. 固定依赖

使用[官方源码](https://github.com/soda-inria/tabicl/tree/d83091b845a8c5f7c087bd54964f755a9e560f45)的 commit `d83091b845a8c5f7c087bd54964f755a9e560f45`。加载时核对 15 个 `_model` 文件的 LF 规范化 SHA-256，兼容 Windows CRLF 与 Linux 文件换行。没有自动安装和权重下载行为。

仓库本机的源码检出位于被 Git 忽略的 `local/vendor/tabicl`。新环境可安装[独立可选依赖文件](../requirements-context-cache.txt)，或在仓库根目录取得固定源码：

```powershell
git clone --filter=blob:none --no-checkout https://github.com/soda-inria/tabicl.git local/vendor/tabicl
git -C local/vendor/tabicl checkout --detach d83091b845a8c5f7c087bd54964f755a9e560f45
```

已有同名目录时直接核对 commit，不覆盖。环境需要项目开发依赖、PyTorch 与 TabICL 声明的运行依赖；本轮使用现有 `torch==2.6.0+cu126` 的 CPU 路径，不改变历史锁文件。

## 4. 本机合成核查

在仓库根目录执行：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m ids.context_cache --tabicl-source local/vendor/tabicl synthetic-check --output local/context-cache-synthetic-new
```

输出目录必须不存在，避免覆盖已有研究记录。此命令创建随机小型官方模型，关闭 attention/FFN 的零输出初始化，并使 QASSMax 的非线性查询调制非零。它构建 12 行支持缓存、压到每头 3 槽，检查：

- 官方完整缓存与独立 Stage 3 路径的 logits 一致；
- M=N 时恢复完整路径；
- 切分查询批次不改变压缩预测；
- 新进程仅用压缩包和查询文件恢复相同预测。

本轮生成的 `synthetic_report.json` 中，完整路径最大绝对 logit 误差为 **1.4901161193847656e-7**，冷加载误差为 **0**。四项检查通过。随机模型上的 3072→768 字节 Stage 3 K/V 变化仅核对槽数与存储结构，不代表 IDS 模型效果或加速。

针对性测试命令：

```powershell
$env:PYTHONPATH = "$(Resolve-Path src);$(Resolve-Path local/vendor/tabicl/src)"
python -m pytest -q tests/test_weighted_context_cache.py tests/test_context_compiler.py tests/test_tabicl_cache_adapter.py tests/test_context_cache_cli.py
```

本轮合并执行结果为 **36 passed，0 skipped**，Ruff 与 `git diff --check` 通过。测试覆盖分块拟合与稠密参考一致、带锚目标、空簇和零方差、真实质量、原 N、非零模型下缓存确实影响预测、pre/post norm、批切片存储、导出和冷进程、独立 KV 导数、输入契约、随机 checkpoint 的 compile→predict 往返，以及预测器迁移隔离。缺少可选依赖时会显式 skip；skip 不算验证通过。[合成报告](research_alignment_20261106/calibrated_context/implementation_synthetic_report.json)和[检查记录](research_alignment_20261106/calibrated_context/implementation_checks.json)已保存。

## 5. 实验室编译入口

先由支持集上已经冻结的预处理生成 `support.pt`，使用 `torch.save` 保存下列字典。这里给的是结构规格，不是待下载的数据文件：

```python
{
    "format": "tracelight-tabicl-native-input-v1",
    "role": "support",
    "data_kind": "development",  # 合成检查为 synthetic
    "X": support_x,              # float32 [1, N, F]，有限值
    "y": support_y,              # [1, N]，正常 0／攻击 1
    "feature_names": feature_names_in_order,
    "preprocessing_id": frozen_support_preprocessor_hash,
    "label_mapping": {"normal": 0, "attack": 1},
}
```

checkpoint 使用已核实来源的本地 `tabicl-classifier-v2-20260212.ckpt`，格式为官方 `config` 与 `state_dict`。真实权重尚未在本机下载或验证；实验室先核定其 SHA-256，再传入命令。以下占位路径和哈希需要替换：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m ids.context_cache --tabicl-source local/vendor/tabicl compile --checkpoint <local-v2-checkpoint.ckpt> --checkpoint-sha256 <verified-sha256> --support <support.pt> --slots 512 --device cuda --output <new-compile-directory>
```

`slots` 必须不超过支持行数。`--query-chunk-size` 默认 256，只控制中间工作区，不改变原 QCOC 使用全部支持查询拟合的定义。命令只读支持文件，导出 `full.pt`、`compact.pt`、`compile_report.json`；不读取开发查询、校准或最终 test，不训练骨干。

报告按层、批、头保存原 N、M、真实质量范围、迭代与空簇修复、均值分配 C 与质量分配 D 的差异、拟合前后输出误差和导出舍入误差。另记模型／列缓存／Stage 3 K/V／质量的张量字节，以及实际文件字节和哈希。没有把这些数字称为 RSS 或硬件计时。

## 6. 实验室预测入口

`queries.pt` 使用相同格式，`role="query"`，保留与支持集完全相同的 `feature_names`、`preprocessing_id`、`label_mapping`，包含 `X[B,Q,F]`，不含 `y`。评价标签另存。当前入口仅接受 `synthetic` 或 `development`，不承担最终 test 评价。

```powershell
python -m ids.context_cache --tabicl-source local/vendor/tabicl predict --bundle <new-compile-directory/compact.pt> --queries <queries.pt> --output <compact-predictions.pt> --device cpu
python -m ids.context_cache --tabicl-source local/vendor/tabicl predict --bundle <new-compile-directory/full.pt> --queries <queries.pt> --output <full-predictions.pt> --device cpu
```

输出为两类 logits 和 `attack_logit-normal_logit` 分数，并绑定模型与查询文件哈希。此阶段不计算低误报收益；下一阶段需在开发材料上为两个候选分别登记阈值，再分析修复、新增误报与漏检。

## 7. 已处理的实现差异

- 官方缓存容器会在拼接时重建只有 K/V 的原类型，因此压缩 Stage 3 使用独立容器，Stage 1 保留官方容器。
- QCOC 原文第 3.2 节以 C 簇确定中心，再重分配 D 计质量。实现按原文执行，已修正主设计稿此前“最后不重分配”的措辞。空 D 簇采用确定的最大失真非单例 donor 修复，属于本实现明确登记的退化处理。
- CPU 切片即使 `.cpu()` 仍可能引用完整底层存储；导出前 clone，避免把省掉的批次保存进去。
- 完整与压缩预测器初始共享冻结骨干；单独迁移一个预测器到不同设备或精度时先分离骨干，保持另一对照不变。
- QCOC 最后 V 拟合在部署精度的 K 和 log_mass 上构造算子，分块累计并以 FP64 求解；报告实际导出 V 的舍入误差。没有用 M 替换原 N，也没有重复施加 1/√d。

实现依据为[TabICLv2 官方 2026 源码](https://github.com/soda-inria/tabicl/tree/d83091b845a8c5f7c087bd54964f755a9e560f45)和[QCOC 2026 原文](https://arxiv.org/html/2610.11784v1)。QCOC 是按论文独立实现，本轮没有核验作者实现或复现其论文成绩。
