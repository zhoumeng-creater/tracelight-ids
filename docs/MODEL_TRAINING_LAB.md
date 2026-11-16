# R29 模型训练与开发比较

本轮入口为 `scripts/RUN_MODEL_TRAINING.ps1`，执行版本为 `model-training-r29-v1`。注册文件 `configs/model-training-20261116/registration.json` 绑定设计文件、R26 协议及输入工件哈希。默认输出目录为本仓库的 `local/model-training-r29`，不覆盖 R26 或 R28 输出。

本轮执行 7 条开发训练轨迹和 7 次最终重拟合，总计最多 14 次拟合：同条件 HGB 一组，TabM 三个种子，去掉 PLE 的 TabM 三个种子。没有参数网格、追加种子、源域混合或再搜索阈值。完整训练只在实验室机器运行；本机只运行计划、检查和输入分析。

## 环境与输入

沿用实验室已有 Python：

```text
D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe
```

注册环境包含 torch `2.6.0+cu126`、tabm `0.0.3`、rtdl_num_embeddings `0.0.12`、scikit-learn `1.5.1`；其余精确版本见 registration。入口不会安装、升级或下载任何包。神经模型正式训练使用 `cuda:0`；HGB 使用 CPU。最终 C/E/D 评分统一为 CPU 单线程，成本测试按模型在独立新进程中顺序运行。每个进程先测CPU单流/批量及预处理成本，再对TabM测GPU批量吞吐；GPU分数与CPU判决的变化单列，正式检测资格依据CPU结果。GPU每批512行、8次预热、3次计时，保留原始耗时。

`-R26Run` 指向保留的旧 R26 运行目录，或包含 `protocol.json` 与 `evidence` 的 R26 回传解压目录。它不是本轮输出目录。入口仅检查四个相邻旧工作树位置，并要求协议哈希符合注册；发现零个或多个匹配时，需显式填写 `-R26Run`，不会全盘搜索或任取第一个。

训练数值按以下顺序取得：

1. 旧 R26 的 `local/training/mixed.parquet`。
2. 显式 `-RecoveredTraining` 指定的恢复帧。
3. 本仓库的 `model-training-inputs/recovered-training.parquet`，或仓库父目录的 `input/recovered-training.parquet`；同时存在两个时要求显式指定。
4. `-RawDir` 指定的既有原始档案，经 `scripts/recover_r26_training_rows.py` 恢复登记身份。此步骤不下载、不重新训练 R26。

交付包提供 `input/recovered-training.parquet` 及同名 `.json` 来源收据；两者应保留在同一目录。恢复帧包含已登记的 18,874 条 S∪T 数值，但本轮训练仅使用其中 10,682 条目标域样本，并按 F/V 划分。恢复帧不是原始 parquet 的字节复原，不提供伪造的完整流结束时刻。Python 入口逐项验证身份、标签与收据绑定。

## 一键运行

在本版仓库根目录打开 PowerShell。下例沿用默认 Python；将旧 R26 和交付包路径填写为实验室实际保留位置。`-OutputDir` 可省略，省略时使用 `local/model-training-r29`。

```powershell
.\scripts\RUN_MODEL_TRAINING.ps1 -R26Run 'D:\zhoumeng\Graduate\tracelight-support-diagnostic\local\support-diagnostic-r26' -RecoveredTraining 'D:\zhoumeng\Graduate\model-training-r29-v1\input\recovered-training.parquet' -Threads 4 -Device cuda:0
```

旧 R26 中仍有 `local/training/mixed.parquet` 时，入口优先使用原帧，`-RecoveredTraining` 可省略。若只有既有原始档案，可改用 `-RawDir '<原始档案目录>'`。完整流程自行生成成功或失败回传包，无需重复执行打包命令。

运行前可分别查看计划、检查代码和验证输入：

```powershell
.\scripts\RUN_MODEL_TRAINING.ps1 -PlanOnly
.\scripts\RUN_MODEL_TRAINING.ps1 -CheckOnly -Python 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe'
.\scripts\RUN_MODEL_TRAINING.ps1 -InputsOnly -R26Run 'D:\zhoumeng\Graduate\tracelight-support-diagnostic\local\support-diagnostic-r26' -RecoveredTraining 'D:\zhoumeng\Graduate\model-training-r29-v1\input\recovered-training.parquet'
```

`-PlanOnly` 由 PowerShell 直接读取注册和设计，不要求 Python、R26 或 GPU 存在。`-CheckOnly` 必须显式指定 Python，仅运行 `tests/test_model_training*.py` 与恢复脚本测试。`-InputsOnly` 验证已有工件和输入合同、生成数据及校验清单，不拟合预处理，不构建、训练或调用真实模型。各 `Only` 模式互斥。输入检查与随后正式运行需使用相同的线程、设备及输入参数；默认线程数为4，可设置2–64。

## 固定执行顺序

| 数据角色 | 用途 | 正常 / 控制样本 |
|---|---|---:|
| F：B_i30 的三个单元 | 开发训练 | 5,848 / 90 |
| V：B_i300 的三个单元 | 选择阶段或 epoch | 4,654 / 90 |
| P：F∪V | 7 次最终从头重拟合 | 10,502 / 180 |
| C：B_i1800 的三个单元 | 只用正常样本校准 | 5,511 / 不使用攻击标签 |
| E：B_i3600 的三个单元 | 已打开开发评价 | 5,459 / 72 |
| D：12 个登记单元 | 已打开开发评价 | 9,564 / 1,964 |

每条开发轨迹只在 F 拟合预处理和模型，以 V 的标准化低误报 pAUC 选择阶段；并列时按自然权重 NLL、较早阶段排序。随后在 P 重新拟合预处理，模型从头训练至已选阶段。全部 7 份最终模型冻结后，才进行 C 校准与 E/D 评分。C 每模型产生一个跨单元共用阈值，判定规则为 `score > threshold`。E/D 不用于再选 epoch、调参或改阈值。

输入为五个连续流特征、协议类别和缺失标记；端口只保留用于审计。R26混合训练帧的旧平衡权重另存为`historical_weight`，本轮训练将每行权重设为1，不下采样正常流。完整配置及结果分支以注册绑定的 `configs/model-design-20261116/design.json` 为准。

## 中断、核验与回传

保留本次输出目录。重用同一输出目录时，已完成并通过收据校验的拟合会跳过；有未完成拟合的目录会拒绝自动重试。不得通过删去部分模型、清空状态或换一个输出目录重复训练来绕过 14 次拟合上限。环境或输入错误应先按失败报告修复，再依据已有完成状态继续。

下面两个操作仅需本轮输出目录，不依赖旧 R26 路径或训练帧仍然存在：

```powershell
.\scripts\RUN_MODEL_TRAINING.ps1 -VerifyOnly -OutputDir '.\local\model-training-r29'
.\scripts\RUN_MODEL_TRAINING.ps1 -PackOnly -OutputDir '.\local\model-training-r29'
```

回传入口打印的 ZIP，并保留实验室完整输出目录。成功包用于核对注册、输入身份、阶段选择、最终权重、逐单元指标、增失真阳性身份及成本；失败时回传`failed-evidence.zip`和终端错误。失败包记录原异常与损坏收据，不声明阶段完整或结果有效。运行前的缺路径、输入哈希不符等预检错误会直接阻断；此时回传终端错误。

E/B 与 D 均已打开，本轮可以据此完成开发比较，不能把它们写成新盲测。任务是完整 Zeek 流可用后的历史监督 C2 识别；终点流时长不等于实际日志导出时刻，也不支持早期告警结论。同业务正常对照和独立来源仍不足，换模型不能补齐这两类证据。

三个种子均满足逐单元误报预算与两类参考的检测条件，才能形成开发候选；只提升 V pAUC、只在一个种子获益或依赖 E/D 重新校准，不构成该结论。PLE 无稳定收益时保留负消融；HGB 获胜或 TabM 未达到条件时关闭本轮神经模型分支。若 HGB 也超出误报预算，应报告没有新增合格部署模型。
