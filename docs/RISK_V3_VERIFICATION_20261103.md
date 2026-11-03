# v3 发布前工程核验

发布标签：`risk-v3-lab-20261103`。本页记录2026-11-03开发机的代码/合成验证，不是实验室实证结果。使用流程见[RISK_V3_LAB.md](RISK_V3_LAB.md)，研究边界见[RISK_V3_PROTOCOL.md](RISK_V3_PROTOCOL.md)。

## 已实际执行

- 完整pytest：**822 passed，2 skipped，15 warnings，158.33秒**。包含现有可选TabM适配测试；两个skip均因当前Windows账户无法创建符号链接，警告来自物理核数检测与合成特征只有一个bin。不跳过数学/近邻/新runner主流程。
- `python -m ruff check src scripts tests` 与 `git diff --check` 通过。
- 三类近邻：逐样本PCT KL、MPT式双源FD、EKD rank-count，检查教师detach、correctness阈值、T²约定、空类/数值/API及原文语义差异。
- 真实 `_student` 的合成梯度核对：β=0时hard/KD/PCT/双源/EKD保存权重逐tensor相同；β=.2时四种边界分支梯度与独立解析目标一致；异构fold阈值和不平衡类别下，不同chunk大小还原完整EKD两类均值。
- 九条件合成端到端：完整OOF教师/锚点/训练/导出/验收；删除生成的calibration/test文件作为误读陷阱，只读train/validation；共享初始化/硬标签/阈值、完整恢复身份、篡改/增文件拒绝、源码变更拒完成均通过。
- 角色分配：全部train行保留、联合组不跨角色、固定seed一次分配、稀有类别支持可行性检查、缺族与巨组记账通过。
- wrapper及打包mock：4090D GPU0门禁、只读dry-run、不完整输出保护、完整恢复重核身份、ZIP逐成员hash及预检原字节封装通过。实际RTX4050 dry-run在数据读取/目录创建前拒绝。
- CPU原始logit入口：复现同序validation子集；`z=1, t=nextafter64(1,+inf)`正确拒绝边界，禁止概率阈值替换、旧test/calibration输入与覆盖输出。

## 尚未发生

没有用本轮代码在实验室4090D训练UNSW/CIC，没有新增真实低FPR增益、家族保护或正式CPU重复计时结果。822项测试证明的是工程契约与已覆盖的合成性质，不证明算法优于近邻、总体FPR受控或创新性足够。

本轮按用户既定方式通过GitHub交付代码与指南，由用户在实验室运行并回传ZIP。收到工件后再核验预检/源码/数据/逐条预测/失败种子/成本与最近邻差异，不把“已发布”改写成“实验完成”。
