# 2026-11-05 机制重设计：工程验证

本记录验证代码、工件边界和合成行为，不将单元测试数量当成真实检测效果。真实数据工作仅为原归档的只读分层统计和冻结模型诊断，没有新增训练、读取旧test或正式本机资源测量。

## 1. 代码变更

| 位置 | 变更及目的 |
|---|---|
| `src/ids/input_extrapolation.py` | 读取原screen、严格绑定prepared manifest和F/R/C身份；冻结8个seed17 MLP；全部数值列F极值限制；保留同CPU identity、两阈值、配对错误及组模式；可选实验室配对CPU测量；0模型拟合、无候选晋级 |
| `scripts/audit_input_extrapolation.py` | 提供独立的实验室入口；路径须显式给出，不覆盖旧输出 |
| `src/ids/recent_baselines.py` | PLE version B关闭零残差后的ReLU，训练协议2→3；包格式及依赖不变 |
| `src/ids/transfer_screen.py` | CatBoost C结果的阈值来源改为真实使用的R+C规则，未更改其训练或决定 |
| `src/ids/budget_transfer.py` | 首步诊断新增hard／direct pAUC梯度夹角，后续可区分大小与方向；不回写历史run |

诊断候选另存为 `diagnostic-candidate.joblib`，不覆盖原模型。原始／裁剪R/C分数和所有原始耗时均保留，完成标记列出输出文件SHA。正式输入审计明确要求原始manifest字节绑定，本地分析中的语义一致性证明不能绕过此要求。

## 2. 已运行检查

主要测试解释器：

`C:\Users\15995\AppData\Local\Temp\tracelight-tabm-20261103\Scripts\python.exe`

该隔离环境包含项目固定的官方TabM 0.0.3、rtdl_num_embeddings 0.0.12、CatBoost 1.2.10。未改项目默认依赖或实验室环境。

| 检查 | 结果 | 验证内容 |
|---|---|---|
| PLE专项（代理） | 22 passed，0 skipped | 零初始化分段残差收到非零梯度；官方集成损失／概率聚合及旧零残差包预测保持 |
| 六个相关测试文件 | 103 passed | 输入干预、封存screen、预算迁移、native树、实验室流程、近期基线 |
| 全套测试 | 1,176 passed，2 skipped；156.32 s | 本轮核心实现及现有回归 |
| 最后补充计时汇总／8模型范围约束后 | 11 passed，2 skipped；27.53 s | 输入审计及screen端到端复跑；同时核对两处既有跳过原因 |
| 改动Python文件Ruff | All checks passed | 未使用导入、基础语法／名称检查 |
| CLI `--help` | 成功 | 参数与实验室文档一致 |
| `git diff --check` | 通过 | 改动无空白格式错误 |

全套测试在最后补充的局部计时／模型范围检查前运行；之后仅复跑受影响的输入与screen测试，没有把它写成最后每行改动后的第二次全套运行。

两项跳过均为Windows测试账户不能创建符号链接：`test_research_lab.py:240`和`test_risk_lab.py:376`。本轮TabM、CatBoost和输入审计没有因依赖缺失跳过。测试警告涉及Windows物理核心数探测回退及合成单区间PLE的等价性提示；未据合成计时报告部署速度。

## 3. 关键行为证据

- F输入不变、标签改变不影响界限，类别／未知类别、缺失和常量列保持既定变换定义。
- 冻结网络状态逐参数不变，诊断包保存／加载后预测一致。
- 修复FN、新增FN、修复FP、新增FP分开计数，保留原分组列和完整特征模式，不能用净数替代。
- 集成测试物理删除prepared validation/calibration/test后仍完成8模型审计；mock训练入口遇调用即失败，实际未触发。
- 不同prepared manifest在创建输出前被拒绝；模拟CPU重放决定改变后写出失败记录，不生成完成标记。
- 原输出不可覆盖，诊断目录与原模型／数据目录分离。
- 合成计时检查逐次样本、配对顺序、两策略时长与分块结果完整；不对合成绝对时延作性能断言。

## 4. 尚待实验室完成

完整两环境16模型输入审计、同设备混合样本CPU成本、相对全字段small hard的质量—成本前沿，按[已冻结最小协议](MINIMAL_VALIDATION_20261105.md)执行。当前不新增训练，不自动开展KD／PLE。新独立来源尚未取得，局部机制、开发候选和部署确认分别记录。
