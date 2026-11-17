# R30后续设计审查证据

主决定：[正常子群尾部与下一阶段决定](../R30_NORMAL_TAIL_DESIGN_REVIEW_20261117.md)。本目录名称是设计审查编号，不是新实验回传。

- `input_mechanism.py`：只读取R30冻结输入、分数、原阈值及原判决，复算无参数表示、近邻支持和正常组成；不加载模型。
- `input_mechanism.md`、`input_mechanism_summary.json`：主要结论、六模型摘要与源文件SHA。
- `input_mechanism_original_group_rates.csv`：固定分区在C及各D单元的原数量、分母、FP和组内率。
- `full_evidence_binding.json`：完整逐身份证据位置、大小和SHA。
- `author_methods.md/json`：TabM、TabReD、STACO及tapAUC作者机制与采用判断。
- `author_sources/`：必要的固定源码、对应许可和逐文件SHA；仅阅读，未导入执行。tapAUC仅记录远端固定版本与结论，没有复制其无明确许可的完整notebook。
- `design_contract.md/json`：端点、比较含义、原风险规则与独立确认要求。
- `verification.json`、`manifest.json`：本次交付核对及文件绑定。

在仓库根目录、具备numpy/pandas/parquet读取依赖的现有环境中复算：

```powershell
$taskProject = 'E:\OneDrive - Coventry University\中传\毕业论文'
$taskOutput = Join-Path $taskProject '工作\第三十一轮设计审查_20261117'
python .\docs\r31_design_review_20261117\input_mechanism.py --project $taskProject --output-dir $taskOutput
```

脚本输出完整JSON、摘要JSON和CSV。它会重放原C判决并读取原D判决，既不搜索新阈值，也不生成新候选策略。近邻以log1p时长距离定义；取整Top147分析不作为精确fractional CVaR实现，分数导数不作为网络参数梯度。
