# 固定保留通道与增量判决：缓存核查

本入口只执行[已冻结设计](RETAINED_INCREMENT_DESIGN_20261117.md)中的一个公式。模型为三份R29 PLE final，保留通道为旧HGB∪原规则，增量判界固定`max(0, 原C阈值)`。原模型、分数、阈值和归档只读；没有拟合、模型加载、前向、重校准或阈值搜索。

注册为`retained-increment-cached-audit-v1`，文件在`configs/retained-increment-audit-20261117/registration.json`。它绑定设计、设计文档、脚本的规范化文本SHA，以及设计已登记的30份原始工件字节SHA。此注册只允许缓存计算，不启用V2或R29训练入口。

运行前已完成8项合成测试与Ruff检查，并独立审查了身份连接和验收代码。合成测试只证明关键实现行为，不证明候选效果。

在已有Python、NumPy、Pandas和PyArrow环境中，从本工作树运行：

```powershell
python scripts/audit_retained_increment.py `
  --run 'E:/OneDrive - Coventry University/中传/毕业论文/实验归档/20261117_第二十九轮模型训练/回传' `
  --output 'E:/OneDrive - Coventry University/中传/毕业论文/工作/保留增量验证_20261117/primary'
```

输出目录必须尚不存在；程序不覆盖已有结果。三种子分别计算并完整输出，无种子、阈值或floor调参选项。C只用于核对原阈值判决计数，以及报告固定公式的描述性计数；不用于新校准或候选选择。ED保留15个完整单元及17059条原身份。

输出包含：

- `summary.json`：每个种子的B/D计数、全部硬门槛、合规参照退步及整体结论。
- `cells.csv`：45个模型—单元的旧通道/增量/组合/原直接OR计数。
- `reference_comparisons.csv`：三种子各对六个固定参照及同种子原PLE的逐cell比较。
- 三份`*-decisions.parquet`：完整输入身份、原分数、各通道判决及原/新固定判界。
- `identity_changes.parquet`：对每个参照及原直接OR的TP新增/丢失、FP新增/移除身份。
- `input_verification.json`、`registration.json`、`manifest.json`：输入、执行规格和输出哈希，确认原工件未变。

每个种子必须同时满足：15cell原整数误报预算、B72/72、旧TP身份零丢失、所有合规强参照cell的TP不减少，以及D检出严格超过旧通道1694。三种子全部通过才进入组合工件等价与实验室成本核验。任何算法门槛失败均结束此固定候选，不扫floor、不换种子或扩训练；工件连接错误修正后仍只复核原公式。

此核查属于已开开发数据分析；不能由通过或失败得出独立泛化保证。新判决另存，原R29负结果不改判。
