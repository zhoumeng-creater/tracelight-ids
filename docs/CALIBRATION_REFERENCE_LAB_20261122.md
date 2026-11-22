# 正常校准参照约束：实验室一次运行

当前版本为 `normal-reference-transfer-v2`。它接替尚无真实回传的 `normal-tail-transfer-v1` 待运行设计。旧包和旧四次 HGB 包保持原样；本入口不重复运行它们。

本次固定比较 A_BCE、G_group_tail、R_reference_envelope，各用种子 17、29、43，共九次拟合，不新增补跑臂。问题、依据、数学约束和各结果出口见[完整设计](calibration_design_repair_20261120/index.md)。

## 拉取后运行

在实验室现有仓库目录执行：

```powershell
git fetch origin tag normal-reference-transfer-v2
git -c core.longpaths=true worktree add --detach '..\tracelight-reference-v2' normal-reference-transfer-v2
Set-Location '..\tracelight-reference-v2'
.\scripts\RUN_CALIBRATION_REFERENCE.ps1 -Output 'D:\Research Work\results\normal-reference-transfer-v2'
```

默认解释器为 `D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe`；位置不同时传 `-Python`。使用既有 RTX4090D、Python 3.11／3.12、PyTorch 2.6.0、TabM 0.0.3、rtdl_num_embeddings 0.0.12、NumPy 1.26.4、pandas 2.2.2、scikit-learn 1.5.1，入口不自动改环境。

输出须为空目录。程序依次保存九份末轮模型、九个 C 阈值、完整开发评价，再生成 ZIP。回传完整 ZIP。发生异常会保留工件与错误记录，不自动重训。加 `-InputsOnly` 并使用另一空目录可只检查代码和 F 输入，不构建模型或评分。

## 固定训练与校准

三臂共用 13,252 条 F：12,747 正常、504 控制、1 下载。未知 123 条、其他攻击 11 条不参与拟合。输入沿用 ports-zero raw8 的类型化编码：5 个连续字段、协议类别、缺失标志，有限端口值不作特征。连续预处理与 PLE 分箱只拟合 F。完整流特征按离线导出时刻解释。

复用作者 TabM＋PLE，2 层、宽 512、32 成员、dropout 0.1；PLE 维度 16、最多 48 箱、version B、无激活。每批 128 正常＋128 攻击，分别按配对种子的固定乱序队列循环抽取。AdamW 学习率 0.002、weight decay 0.0003、梯度范数裁剪 1；每臂固定 15,600 次更新，不选最佳轮、不早停、不 refit。每种子检查初始化和 BCE 抽样哈希一致。

G/R 每步对完整 F 做关闭 dropout 的评分，在 FP64 稳定对数几率上求顺位导数，再按原 1,024 行分块重放活跃块并与 BCE 梯度相加。参数及成员 logits 为 FP32；发现与重放的分数须逐位一致。迁移项权重和攻击间隔均固定为 1。

每份 G/R 训练有 206,731,200 行完整 F 扫描，六份共 1,240,387,200 行，另计活跃反传块。固定更新数沿用既有比较预算，不代表已知收敛；G/R 实际反传成本可能不同，按原始计数、耗时及显存报告。

模型全部冻结后，只使用 russell 整 run 的 3,475 条合格 C 正常。阈值是升序第 `N-floor(0.01*N)` 个分数，严格大于才告警，预算 34；不拆并列、不插值、不搜索阈值。九个阈值冻结后才读取 fox／OpTC 开发数据。已开放材料保持开发身份，harrison／Meterpreter 保留不读。

## 回传包含什么

- 每份末轮模型、预处理、分箱、训练过程与抽样记录；全部代码及输入哈希。
- 同一 F 末轮分数上的 G/R 双目标诊断，18 个参照—业务配对的实际整数 FP、边界并列数、攻击间隔及全正常同分检查。
- 九份 C 分数和阈值；全部逐连接分数、正常单元预算、控制／下载分别检出、事件候选上下界及新增／丢失。
- G−A、R−A、R−G 的每种子结果、跨种子一致性和实际资源计数；完整错误或完成回执。

`evaluation/comparison.md` 为汇总入口；`decision_summary.json` 保存固定解释规则。训练内零损失不是录取门槛，近零损失也不是整数预算通过。任何改善都按实际判决与成本解释，不追加来源、种子、窗口、阈值或模型网格。

固定规格见 [registration.json](calibration_design_repair_20261120/registration.json)，工程检查见 [verification.json](calibration_design_repair_20261120/verification.json)。
