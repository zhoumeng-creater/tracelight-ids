# 算法 v3：设计与原型验收记录

日期：2026-11-03。对应[算法设计稿](ALGORITHM_V3_DESIGN_20261103.md)。本记录证明设计与工程链已经推进，不证明真实 IDS 效果、总体误报保证或首创性。

## 1. 本轮交付

- [风险核心](../src/ids/risk_distillation.py)：完整 C 的错误包络、修复信用、攻击保留、fixed/adaptive 对照、分母与身份绑定、对偶更新／恢复、raw-logit 工作点冻结和 margin KD。
- [数据角色验证](../src/ids/risk_data.py)：F／R／C 穷尽与互斥、validation 隔离、联合组边界、身份／标签一致和支持清单；无磁盘读写或自动分区。
- [完全合成入口](../scripts/check_risk_design.py)：CPU 单线程；只接受新输出路径、seed、steps，不接受 prepared/data/calibration/test 等真实输入路径。
- [近邻审查](ALGORITHM_V3_NEAREST_REVIEW_20261103.md)、[统计审查](ALGORITHM_V3_STATISTICAL_REVIEW_20261103.md)、[正式接入审查](ALGORITHM_V3_INTEGRATION_REVIEW_20261103.md)：三次分工审查及交叉复核，区别已实现与未来接入。

## 2. 单元与回归检查

新增三组测试：风险 100 项、工作点／KD 40 项、角色验证 33 项，共 **173 项通过**。包含独立布尔事件计数、点态界、修复信用、攻击不抵扣、全条件分母、tie、缩放、弱边界 slack、fixed/adaptive 不支配、teacher／anchor／dual detach、reference 原位与 `.data` 变更、恢复原子拒绝、溢出及浮点并列回归。

在含官方 TabM 可选依赖的现有环境执行全项目检查：

```powershell
python -m pytest -q
python -m ruff check src tests scripts
git diff --check
```

最终本地结果：**633 passed、1 skipped、15 warnings，81.65s；Ruff 与 diff 检查通过。** 一个 skip 及 warning 是具体环境／既有测试提示，不算通过项。没有用这些数字代替研究工作量或检测性能。

## 3. 合成训练链

执行命令（已有目录将拒绝覆盖）：

```powershell
python scripts/check_risk_design.py --output local/risk-v3-synthetic-20261103-v2
```

固定 seed=20261103、student steps=20，CPU 单线程。F／R／C／V 分别为 192／112／192／112 个完全人工样本；原 train 为前三者 496 个。标签／家族是 toy 定义，group_id 只是合成隔离单位，不是真实网络环境。

13 项工程检查通过：身份未变、精确 warm-start、锚冻结、reference detach、对偶状态往返、更新数、签名绑定、有限值、经验上界、源码未变、CPU 权重往返、float64 决策及参考计数一致。四个工件 SHA 和三个源码 SHA 已独立复核。所有检查及各 step 原值都保存在实际 `report.json`／`history.json`，没有筛选 seed 或按 validation 挑 checkpoint。

| 项目 | 实际观察 | 不能推出 |
|---|---|---|
| 合成正常预算 | 15%，R 有 64 条正常，floor 允许 9 条；锚与已知函数教师均为 9/64=14.0625% | 不是正式 1% 尾部检验，更不是未知总体保证 |
| 最后一次迭代 | 第 20 步代理与实际预算均满足；完整轨迹中只有第 20 步代理全满足 | 不是已找到最佳模型，也不代表真实网络中有效 |
| 原始锚 | 硬事件预算可满足，但部分弱边界代理未必满足 | 不能减掉初始 slack 使原型显得可行 |
| 教师 | 固定人工函数，无训练／OOF／TFM 声明 | 不能据此声称前沿教师已经适配 |
| 模型保存 | 锚／最后学生及对偶状态保存、CPU 权重与对偶往返一致 | 不支持完整中断续训；模型／optimizer／RNG 等尚未形成正式恢复协议 |

初次合成目录 `local/risk-v3-synthetic-20261103` 保留原源码快照对应状态：它采用 higher quantile，在 R 产生 10/64=15.625%，没有保证参考预算。数值审查后新增保守次序统计及 float64 helper，以同 seed／同 steps 另跑 `-v2`，未改旧输出。这是工程修复，不能选择性将旧轨迹改称新协议结果。

本地两个输出目录均属于 `.gitignore` 的私有工程工件；公共仓提供脚本与验收说明，重跑会生成自己的模型／JSON。正式实验归档与后续实验室包另行生成，不把合成输出混入前五轮。

## 4. 审查修正的实质问题

1. frozen dataclass 不阻止 Tensor 内部变化：每个完整风险报告重算 reference 内容 SHA，`.data` 变更也拒绝；同步／哈希开销需要计时。
2. 恢复状态不能丢失绑定：已更新或非零权重不能以空 signature 恢复，错误预算／reference／设置均在修改状态前拒绝。
3. 正常上界按逐行非负包络求平均，避免完全修复情形中跨均值相减欠界。
4. Lagrangian 单独检查乘积／求和有限，训练脚本单独检查梯度有限。
5. 工作点按整数允许数与并列规则冻结，margin 和硬决策先转 float64，防 nextafter 阈值舍回 float32 边界。
6. F／C 共同 hard 项按人数加权，教师 soft 独立乘 beta；所有正式学生对照共享 C 标签，不能把额外监督说成新机制收益。

## 5. 下一阶段验收边界

尚未完成：真实数据角色分配、严格折内教师 margin adapter、完整规模分块求解、正式 v3 runner／lab wrapper／工件校验、实验室 4090D 新结果和新来源确认。旧 v2 定义／配置／实验归档未修改；新增源码会改变全仓指纹，旧包继续按旧版本执行。

优先按设计稿 G1a 的五条件检验归因，再按证据追加 EKD、PCT、成熟 pAUC 和代理消融。原型通过不等于方法优于近邻，实证未出现前不将候选写为已成立创新。
