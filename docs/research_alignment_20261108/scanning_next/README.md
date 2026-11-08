# 第十五轮后的Scanning设计证据

当前决定见[下一步设计](../../SCANNING_NEXT_DESIGN_20261108.md)：先做同输入、同支持行的标签粒度三臂，历史观测为条件分支。正式入口已接入，执行以[实验室指南](../../SCANNING_LABEL_LAB_V1.md)及 `scanning-label-lab-v1` 标签为准。

| 文件 | 内容 |
|---|---|
| [tail_findings.md](tail_findings.md)、[tail_audit.json](tail_audit.json) | 保存分数的阈值代价、易混正常流、组内排序、未变查询的上下文响应、P标签分组冲突；14份输入SHA。 |
| [tfm_context_research.md](tfm_context_research.md)、[tfm_context_sources.json](tfm_context_sources.json) | 2025—2026直接近邻、固定作者代码、标签进入两级缓存的位置、检索和重权的取舍。 |
| [network_context_research.md](network_context_research.md)、[network_context_sources.json](network_context_sources.json) | 观察单位、历史信息可用时刻、构窗与向前时间确认；本轮不启动真实历史构造。 |
| [label_reference.py](label_reference.py)、[label_synthetic_checks.json](label_synthetic_checks.json) | 真实三类与整组伪标签的纯函数、攻击联合评分；23项人工数组检查通过。没有构造真实P的标签分配。 |
| [native_multiclass_synthetic.py](native_multiclass_synthetic.py)、[native_multiclass_synthetic.json](native_multiclass_synthetic.json) | 固定官方源码、随机微型模型的2/3类缓存、两级标签响应、统一评分与新模型对象使用外部缓存。没有加载真实权重；不是跨进程部署验收。 |
| [design_manifest.json](design_manifest.json) | 设计决定、数据/代码基准、当前状态和本目录证据摘要。 |
| [implementation_validation.json](implementation_validation.json)、[implementation-junit.xml](implementation-junit.xml) | 正式适配器与三臂入口的完整合成验收。 |

`native_multiclass_synthetic_initial.json`保留正式适配器尚未实现时的首次原生接口检查。当前实现状态以 `implementation_validation.json` 为准；研究稿中“尚未实现”是当时设计阶段记录。

本机复算命令（从仓库根目录执行）：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
D:/python/python.exe docs/research_alignment_20261108/scanning_next/tail_audit.py
D:/python/python.exe docs/research_alignment_20261108/scanning_next/label_synthetic_checks.py
D:/python/python.exe docs/research_alignment_20261108/scanning_next/native_multiclass_synthetic.py
```

第一条只读已有CSV、分数与映射工件；后两条只用人工数组/随机微型模型。所有输出保存在本目录，没有新增大CSV、权重副本或实验ZIP。原始实验室结果与负结果继续保留在既有实验归档中。
