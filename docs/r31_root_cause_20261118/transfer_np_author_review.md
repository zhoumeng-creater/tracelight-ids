# 迁移 Neyman–Pearson 作者实现核查

2026-11-18，只读核查，未导入或执行作者代码。

**2026 正式论文与代码。** [ICLR 2026 论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33870b3e099880cd8e705cd07173ac27-Abstract-Conference.html)研究正常和攻击两类分布同时变化。其[官方补充包](https://proceedings.iclr.cc/paper_files/paper/2026/file/33870b3e099880cd8e705cd07173ac27-Supplemental-Conference.zip)含 `Nasa.ipynb`、`Climsim.ipynb`，原字节与 SHA 已保存于 `author_sources/np_both_shift_2026`。不是仅引用摘要。

`Nasa.source.txt:53–66` 分开记录 sigmoid 代理风险与真实错误率；`:169–208` 联合更新模型和约束乘子；`:381–407` 以多个风险约束的最大违反量驱动训练。代码需要目标正常、目标攻击及来源数据，示例使用 `alpha=0.10`、样本量容差、乘子截断和参数平均。它不直接支持无目标域训练样本下的本项目 1% 要求。

论文定义要求代理函数在零点取 1，而示例 sigmoid 在零点取 0.5；从该函数本身只能推出 $P(h>0)\le2E[\sigma(kh)]$，不能把代理均值限额等同真实 FPR 限额。不得仅把代码的 0.10 改成 0.01，就声称得到 1% 保证。

可借鉴：把真实风险对象、代理量、约束违反量和原始/对偶更新分别写清。不能照搬：目标域信息前提、容差、零判界、多阶段求解、参数平均以及非凸网络的理论保证。我们的正式 C 分位判决必须进入问题定义。

**2025 仓库是另一篇论文。** [Transfer Neyman-Pearson Algorithm for Outlier Detection](https://arxiv.org/abs/2501.01525)对应 [eneugut 仓库固定版本](https://github.com/eneugut/transfer-learning-neyman-pearson/tree/f048fc973c01adffc8b3cacff8481957d644abe2)，不是上述 2026 论文的实现。读取的 7 份源码及许可见 `author_sources/tlnp_2025/manifest.json`。

`training_utils.py:106–119` 明确拆分目标正常、目标攻击、来源攻击；`point_selection_utils.py:6–21` 按 Type-I、目标 Type-II 过滤后选择；主文件 `:480–498` 对多个来源权重迭代并扩展搜索。它提供风险约束的实际案例，但其权重搜索、多次拟合和固定零判界不作为本项目下一轮流程。
