# 重构研究入口的工程验收

2026-11-04。当前状态：主线／候选／对照已实现，合成与回归通过；真实实验室研究未运行，新算法效果与创新性未成立。

## 实际检查

- 全仓 `python -m ruff check src scripts tests` 通过。
- 包含官方 TabM 可选依赖的隔离环境运行完整 `pytest -q`：**1149 passed，2 skipped，15 warnings，129.28秒**。两项跳过为现有 Windows symlink 权限用例；警告为已知物理核心查询与旧 TabM toy 单bin情况。
- 新增直接 pAUC 36项、同类关系KD／近期二类适配19项、教师信息诊断28项、完整训练链10项、实验室入口47项，共140项。它们验证实现语义，不是科研工作量或有效性证明。
- PowerShell launcher语法解析通过，**没有在本机执行实验室launcher**。Python `--help` 通过。

## 哪些不是mock

完整链检查实际生成toy CSV和prepared数据，实际产生并核验v3封存源，再执行新F-only教师、四组warmup和全部17个学生条件。检查冻结后才首次加载V、两位教师／四探针也晚读评分、教师输入schema、相同初始化、全局分母、负headroom仍保留对照、固定类内置乱、HGB／MLP CPU流水线分支，以及逐行报警决定。旧calibration／test正文物理删除后仍完成。

直接pAUC对照通过独立有限CVaR次序统计／数值最小化和全枚举梯度检查；关系KD通过两端解析梯度、严格单调变换、ties、全局分块、近期作者二维表达退化检查。输入歧义通过有限重复组穷举，pAUC指标与sklearn独立交叉检查。

实验室GPU／环境预检及双环境大数据打包用mock元数据测试，**不冒充本机有4090D或真实训练完成**。包验证覆盖全部成员SHA、回执原字节、完整条件计数、source/data/interpreter绑定、拒绝部分目录和腐坏包恢复。实验室实际新包仍待生成。

## 保留的科学缺口

当前只承诺固定协议可执行、结果可核。教师在目标区域是否有用、学生实际瓶颈、KD是否超出direct pAUC／raw KL／uniform／shuffle／matched teacher、完整来源泛化和成本Pareto都没有本轮真实结果。ICML2025类别RankLoss的二分类语义适配不等于作者1000类源码复现，也不是足以单独证明击败前沿强KD的对手。

后续按[实验室指南](BUDGET_TRANSFER_LAB.md)运行完整样本pilot，将六个回传文件放回原归档位置。按[科学协议](RESEARCH_RECONSTRUCTION_PROTOCOL_20261104.md)保留全部负结果、逐项撤回不成立的归因；旧test不再用于新模型选择。
