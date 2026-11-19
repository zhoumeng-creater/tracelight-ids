# 本次设计重建使用的近期来源

2026-11-19。此处只记录会改变当前设计取舍的补充材料；已完成的 TabM、TabICL、QCOC、低误报约束与重复 C2 审查继续保留。阅读论文或代码不计作复现，更不构成新增实验指令。

| 一手来源 | 本次核到的条件 | 对当前方案的影响 |
|---|---|---|
| [GraphIDS，论文 v6](https://arxiv.org/html/2509.16625v6) | 通过局部图嵌入和掩码重建学习正常模式。附录 C.2 明确主要利用共现，最终不使用位置编码并打乱顺序 | 不能把它直接当作严格逐查询的控制—下载时序模型；使用图也不自动解决当前信息可用性 |
| [Online IDS GNN，2026](https://journals.sagepub.com/doi/10.1177/1088467X261488186) | 按时间窗口构图，输出窗口类别；文中按每个标签分配最早 80% 窗口训练。该划分不等同于一个全局时间切点 | 不能将窗口 F1 直接与本项目逐流低误报比较；须保留输出单位和划分差异 |
| [REAPER，2025 首发／2026 卷期](https://www.cs.newpaltz.edu/~lik/publications/Dan-Tang-IEEE-ToN-2026b.pdf) | IP-trie 聚合短流，时序嵌入与 VAE，并有 DPDK 实现和吞吐评价 | 输入和系统条件不同于八字段完整流；大吞吐不证明本项目单条 CPU 成本或下载识别收益 |
| [Class-Aware KD，2026](https://jcbi.org/index.php/Main/article/download/1676/931/4447) | 作者报告普通 KD 的稀有类损失，修正后仍未优于良好配置的加权 hard 学生；跨库实验也失败 | 支持必须保留同学生 hard 对照这一设计要求；不能据另一任务成绩为本项目指定新 KD 损失 |

Online IDS GNN 作者源码固定到 commit `3f2bab7c3592be887dfeeb3408273e0c1e350de7`。已读 [processor.py](https://github.com/aron123/Online-IDS-GNN/blob/3f2bab7c3592be887dfeeb3408273e0c1e350de7/processor.py) 与 [dataset.py](https://github.com/aron123/Online-IDS-GNN/blob/3f2bab7c3592be887dfeeb3408273e0c1e350de7/dataset.py)：按 `time_group` 建图，并将窗口标签加入图数据集。未安装、训练或复现该模型；论文在线定位不能替代本项目的实际导出时钟核对。

这些来源没有给出当前新算法的现成处方。其作用是进一步排除不成立的迁移解释：图结构不自动等于因果关系，完整流时序不自动等于早检测，压缩改善也不自动等于教师提供了 hard 标签之外的有效知识。
