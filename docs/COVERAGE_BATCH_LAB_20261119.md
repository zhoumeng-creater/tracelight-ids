# 覆盖与端口依赖：实验室固定四对照

本批次替代已撤下的 HGB8/HGB27 方案，只回答两个问题：完整异源 C2 监督是否改善跨实现检测；效果是否依赖端口。它不是原创算法验证。实际输入与许可由 `coverage_basis_20261119/batch/registration.json` 固定，来源未完成时不生成可运行注册。

运行版本：`coverage-baseline-batch-v1`。实际注册已启用，16 项合成检查和 59,707 条真实评分输入的 InputsOnly 检查通过；真实模型尚未运行。F_expanded 为 13,252 条（12,747 正常、505 控制／下载正例），其中新增的 473 条正例全部是 C2。

| 训练支持 | 保留 raw8 | 两项端口置零，缺失仍保留 |
|---|---|---|
| AIT santos | narrow_raw8 | narrow_ports_zero |
| AIT santos + 完整 Unraveled 的目标主机正常/C2 | mixed_raw8 | mixed_ports_zero |

预先指定 `mixed_ports_zero` 为主参照，另外三臂用于解释监督扩充和端口的作用；不看 fox／OpTC 结果再挑赢家。

共同配置为 HGB 200 次迭代、31 个叶节点、学习率 0.1、固定种子 20261111、关闭提前停止，类别权重使用标准 balanced。其余参数由项目固定的 scikit-learn 1.5.1 决定；数值使用既有 log1p，缺失由 HGB 原生处理，不以零补缺。[官方参数与缺失处理说明](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

F_base 是原 santos 471 条已登记监督。F_expanded 只增加完整 Unraveled 来源中可由作者与原包对应的正常/C2；同一 campaign 不跨角色。C 使用整个 russell 的 3,475 条正常连接，包含 33 次下载动作的业务证据。fox 与 OpTC 开发评价不进入拟合、选轮或阈值设定；harrison 和 Meterpreter 保留材料不读取、不评分。新增 Unraveled 提供 C2 监督，没有提供已核实攻击下载监督。

阈值固定为 C 分数降序的第 `floor(0.01*N)+1` 个值，严格大于才告警，不拆并列。这只是统一的经验工作点，不声称总体 1% 置信保证。每个预先定义正常单元仍单独报告整数预算，不能用总误报率抵消某个单元超标。

## 一次运行

在实验室现有仓库中拉取固定版本并创建独立目录，沿用现有项目 Python 环境（scikit-learn 1.5.1）：

```powershell
git fetch origin tag coverage-baseline-batch-v1
git worktree add --detach '..\tracelight-coverage-v1' coverage-baseline-batch-v1
Set-Location '..\tracelight-coverage-v1'
.\scripts\RUN_COVERAGE_BATCH.ps1 -Output 'D:\Research Work\results\coverage-basis-20261119'
```

路径可替换为空目录。命令完成四次拟合、四次 C 校准和固定开发评价，并生成同名 ZIP 回传包；输出目录非空时不会自动覆盖或重试已有拟合。可先用 `-InputsOnly` 检查输入，这不会构造或调用模型。

全部四个模型先保存，再计算并保存四份 C 阈值，最后评价开发材料。CPU 线程固定为 4；HGB 不使用 GPU。不另开种子、轮数、窗口或阈值扫描，不从四臂中挑一个后重训。四臂都是已开历史材料上的开发比较，不称盲测或时间前向确认。

## 回传与解释

回传整个输出目录：四个模型、逐流分数和判决、事件候选命中上下界、正常单元、控制／下载分别统计、四组配对增失、实际拟合与批量评分成本、输入和代码哈希。缺失或多义事件保留，不挑某个候选作为命中。

先看完整监督支持的两组成对比较，再看相同支持下的端口对照。控制收益、下载收益与正常误报分别判断；一个来源的重复连接数不等于独立攻击实现数。成本是本机实际固定拟合与批量评分耗时，不代替端到端单条告警延迟。

若新增来源没有提高未参与拟合实现的检出，记录这个负结果；它只约束本次 HGB、输入和监督设定，不能推出所有模型或历史信息都无用。若只有保留端口时改善，不能宣称学到了与服务布局无关的行为。即使四臂出现合格结果，普通 HGB、监督扩充和端口置零仍是成熟做法，不能作为已完成的原创算法贡献。
