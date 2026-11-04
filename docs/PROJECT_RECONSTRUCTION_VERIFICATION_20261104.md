# 本轮重构验证

2026-11-04。使用独立`--system-site-packages`虚拟环境，复用本机Python3.12、torch2.6.0+cu126及项目固定核心依赖；新增CatBoost1.2.10、TabM0.0.3、rtdl_num_embeddings0.0.12仅安装到该环境。没有真实数据训练或正式硬件性能测量。

## 结果

- 完整`pytest -q`：**1167 passed、2 skipped、15 warnings，180.83秒**。完整输出保存在工作目录`项目重审_20261104/pytest_full.txt`。
- 最后一次screen互补阈值与计时范围修正后，`test_transfer_screen.py`和`test_budget_transfer_lab.py`：**55 passed**。此数包含在项目测试覆盖中，不与完整回归数量相加。
- 官方CatBoost真实合成适配：**10 passed**；实际OOF查询相关审计：**20 passed**。
- 全仓`ruff check src scripts tests`、`git diff --check`及PowerShell解析通过。

两项跳过来自Windows符号链接权限。15条警告为已有joblib物理核查询回退及TabM合成数据单分箱提示；可选TabM/CatBoost的实际拟合测试已执行，没有以缺依赖跳过代替验证。

## 覆盖的实际行为

1. 从900行合成训练源准备封存v3身份，再删除prepared的validation/calibration/test文件，完整训练12个screen模型；包括官方64/256棵CatBoost、共同R/C阈值、模型保存重载和CPU流程计时。拦截parquet读取，确认不解析V身份/预测；原v3字节保持不变。
2. 默认CLI和Python入口计划两环境24次拟合；dry-run不拟合；screen不会分派旧46次矩阵；打包6个文件，完整恢复保持配置/数据/源码绑定。
3. hard/direct pAUC共用对应warmup；保存warmup实际更新次数、处理行数、所选起点及C选择次数。教师互补主结果使用与选择一致的R+C阈值，R-only诊断单列。
4. 原生树使用连续CrossEntropy目标。教师目标等于hard时得到相同预测；软目标未被二值化；原始hard行权重不随软目标变化。未知类别、缺失值、无标签推理、字段子集和完整模型重载均覆盖。
5. OOF审计只查询对应封存fold的heldout及R/C，不fit/step；旧clip目标逐行精确重现；每折native/clip阈值独立，跨折只聚合冻结判定，不混排margin。

以上验证说明实现、数据角色和结果记录符合本轮设计。真实OOF/native教师价值、树学生效果及实验室CPU成本尚待回传，不能用合成分数或测试数量代替。
