# 研究主线审计的工程核验

标签 `research-foundation-audit-20261104`。本页仅记录代码与工件检查，不作为方法增益或新颖性证据；根本判断见[主线复盘](RESEARCH_ROOT_CAUSE_REVIEW_20261104.md)。

- 完整pytest：**1009 passed、2 skipped、15 warnings，113.99秒**。两个skip为Windows符号链接权限，警告来自既有物理核数检测和TabM合成单bin；可选固定TabM环境已纳入。
- 精确阈值solver：43项检查，包含486组小池穷举全部可达阈值，比对有限R/C条件的必要充分性、严格正常下界、含等号TP保留上界、ties、无TP／预算≥TP／空区间／浮点溢出、身份与签名修改拒绝。该次序统计本身不作为新理论。
- guard回归：真实合成pipeline强制全拒，确认只尝试一次完整事务、模型与optimizer恢复、step0保留、计划／实际步数分列。失败Phase-I权重与SHA也被保存；新schema2要求检查点，原v1文件清单仍严格验收，未知schema拒绝。
- 零拟合实验室入口：11项原评分检查，覆盖双环境门禁、metadata-only预检、旧源身份、阈值先由R/C选择再查询V、原GPU基准签名、分数／成员mask封存、ZIP回读与完整恢复。
- 教师信息检查：8项新增检查，覆盖native HGB公开API与predict_proba链接一致、精确旧float32 clip路径及边界、四类错误互补、训练预算差异、实际OOF目标／家族支持和元数据缺失拒绝。含真实合成HGB/MLP工件端到端，audit阶段fit与optimizer.step被陷阱拦住。
- 全仓Ruff与git diff空白检查通过。私有PowerShell交接脚本仅语法解析，未在开发机执行。

另已只读核验真实v4两包SHA／成员、16份V预测、Phase-I和学生历史，并分析真实OOF目标；未在本机读取prepared特征进行真实模型评分或训练。新教师native评分、互补和精确区间仍需实验室只读查询，未取得的结果不提前填表。

工程修复、基础诊断与科学贡献分开：不以1009项检查、更多输出文件或成功打包代替“教师知识确有增量、学生可利用、超过强简单控制、部署取舍改善、与近邻不同”的证据链。
