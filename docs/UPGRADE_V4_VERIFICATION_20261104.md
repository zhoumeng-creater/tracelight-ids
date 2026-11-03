# v4 发布工程检查

2026-11-04；发布标签 `upgrade-v4-lab-20261104`。真实v3结果与新v4工程检查分开记录。

- 完整pytest：**943 passed、2 skipped、15 warnings，97.65秒**。两个skip为Windows符号链接权限；警告来自现有物理核数检测与TabM合成特征单bin。包含固定可选TabM环境，未省略新主流程。
- 新v4专项：121项通过，包含稳定raw Bernoulli KL、全局chunk梯度、post-AdamW整步/缩步/拒绝、optimizer/grad/CPU-CUDA RNG回滚与callback修改检测、八条件合成端到端、无可行起点分支、旧源不改、预算不变、完整恢复与包校验。
- raw KD仅改变目标坐标；balanced为0.5两类均值，不偷加hard权重。测试覆盖教师/标签detach、极端logit与常数熵不改变学生梯度。
- 阈值回归：float32 plateau与float64 nextafter比较修正；CPU oracle不需要V字典且不消耗训练RNG；整数预算避免epsilon放行；原锚点事件mask不改为新候选mask。
- 八条件完整路径的合成fixture为刻意构造的zero-output incumbent，预算明确为toy FPR25%、attack loss100%，用于覆盖step0、loss、选轮和保存接口，**不是1%真实IDS性能试验**。另有严格零允许事件和1%小池边界的单元检查。没有用loose toy条件替换实验室配置1%预算。
- lab wrapper32项mock检查通过；两套真实v3 JSON身份接口已核，未读特征或load模型训练。pilot/full实配置验证通过；实际RTX4050 dry-run在读输入／创建目录前拒绝。固定同一解释器、源码、旧v3完成SHA和prepared身份。
- 私有实验室一键PowerShell交接脚本已通过语法解析，未在本机执行。GitHub通用指南与本机交接默认路径分开，避免把个人机器目录写入公共文档。
- Ruff全仓检查及git diff空白检查通过。

这证明已覆盖的代码与合成契约，不证明v4有效、代理问题解决、总体FPR受控或新颖性成立。真实v4训练仍待用户在实验室运行；新来源确认、成熟pAUC与更强调参基线仍未完成。
