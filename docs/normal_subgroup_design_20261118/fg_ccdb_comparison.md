# FG-CCDB作者实现对照

核对日期：2026-11-18。论文为 [ICLR 2026正式版本](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d03791377a31cb3e7357014ba58eb80-Abstract-Conference.html)，作者公开仓库固定到 `MiaoyunZhao/FG_CCDB@6aeb91cfc1abe1685f32ed275f4d5569a0421e42`，许可证MIT。

论文按类别内偏置模式的构成差异设计分布平衡，并通过多阶段训练推断偏置模式。可以借鉴其“粗略总体平均不足以描述子群失衡”的问题分解；本文任务已有F正常形态支持，不需要另训偏置分类器。

实际代码核对如下，未执行作者脚本：

| 固定源码位置 | 实际行为 | 本项目决定 |
|---|---|---|
| `show_tsne.py:353–384`，`group_weight_specific_v4` | 从模式×类别计数构造类别平衡后的分布和样本权重 | 这种重加权作用于样本频率，不能自动等同于正常高分尾部的条件分布约束 |
| `dfr_bias_emerging_reweight_final.py:381–384` | 活动的`if 1`分支以真实组标签覆盖推断组，并用真实组更新`all_place` | 不能把这份脚本原样运行称为无需组标签的论文复现 |
| 同文件`:386`与`show_tsne.py:384` | 调用解包两个返回值，实际函数返回三个 | 原固定脚本存在静态接口不一致，不作为即插即用训练入口 |
| 同文件`:97–107` | 基础逐样本损失再乘权后求均值 | 本项目不照搬权重和指数；当前需要改变已低BCE的正常形态相对分数 |

保留必要源码及许可证供复核：[权重函数](author_sources/FG_CCDB/show_tsne.py)、[调用脚本](author_sources/FG_CCDB/dfr_bias_emerging_reweight_final.py)。本次不引入作者的偏置模型、多阶段重训或重加权参数。

固定线上源码：[权重函数](https://github.com/MiaoyunZhao/FG_CCDB/blob/6aeb91cfc1abe1685f32ed275f4d5569a0421e42/show_tsne.py#L353)、[调用脚本](https://github.com/MiaoyunZhao/FG_CCDB/blob/6aeb91cfc1abe1685f32ed275f4d5569a0421e42/dfr_bias_emerging_reweight_final.py#L381)。
