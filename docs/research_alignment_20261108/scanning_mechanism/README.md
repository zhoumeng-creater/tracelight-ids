# Scanning 机制审查证据索引

本目录集中保存第十四轮结果之后的输入语义、支持覆盖、近期成果审查和纯合成干预。主设计见[Scanning最小机制对照](../../SCANNING_MECHANISM_PROTOCOL_20261108.md)。原始逐行结果只引用实验归档中的既有文件，不复制模型或数据总包。

| 内容 | 入口 | 当前证据 |
|---|---|---|
| P/C/E支持覆盖与投影碰撞 | [artifact_findings.md](artifact_findings.md)、[artifact_coverage.json](artifact_coverage.json) | 已读取冻结CSV／分数，0模型调用、0预处理拟合 |
| NF-v3字段及2025—2026数据成果 | [data_semantics.md](data_semantics.md)、[来源](data_semantics_sources.json) | 官方ZIP小型字段表、原论文及版本差异 |
| 原始记录、主机与时间覆盖 | [raw_metadata_audit.py](raw_metadata_audit.py)、[聚合结果](raw_metadata_audit.json) | 流式读完27,520,260条记录，CRC／SHA通过；33,192行的48字段、标签和攻击族全部对齐，无CSV展开或模型调用 |
| 近期TFM与固定作者实现 | [tfm_evidence.md](tfm_evidence.md)、[来源](tfm_sources.json) | 4条主要近邻及排除记录，未运行外部模型 |
| 三臂固定参数 | [design_manifest.json](design_manifest.json)、[正式配置](../../../configs/context-scanning-mechanism-v1.json) | 设计清单与运行配置分列；正式配置由scanning-register登记路径 |
| 输入干预 | [mechanism_reference.py](mechanism_reference.py) | 原数学参考实现；生产实现为ids.context_cache.scanning_inputs，与参考逐位一致 |
| 合成核查 | [mechanism_synthetic_checks.json](mechanism_synthetic_checks.json) | 20项通过；没有对真实P构造映射或重新拟合缩放 |
| 正式入口与完整回归 | [实施验证](implementation_validation.json)、[JUnit](implementation-junit.xml)、[实验室指南](../../SCANNING_MECHANISM_LAB_V1.md) | 288项合成回归通过；真实模型仍只在实验室执行 |

可复现入口为`artifact_coverage.py`、`raw_metadata_audit.py`和`mechanism_synthetic_checks.py`，分别读取已保存工件、流式读取官方ZIP、核查合成数组。真实模型干预与新的独立确认均留在实验室。
