# 多候选事件监督：本轮不作为主线

2026-11-19，设计探索与现有输入读取；没有拟合或模型评分。

考虑过直接使用动作对应的候选连接集合，避免把未知连接硬贴成正常或攻击。若一个可信恶意动作的候选集合完整，则可写约束 `OR(y_i for i in C_e) = 1`。这是标准多实例学习的正袋语义，不是新算法。“动作只发生在其中一个连接”也不等于“其余候选连接都是正常”。正常动作尤其不能把整个候选集合变成纯正常负袋。

进一步设想：同一连接被多个动作候选集合共享时，使用联合约束的概率，而不是相乘独立袋概率。设连接标签的模型概率为 `p_i`，可行赋值集合为 `Omega`：

`L = -log sum_{y in Omega} product_i p_i^y_i (1-p_i)^(1-y_i)`。

这仍属于已有的约束弱监督／概率模型计数范畴。它与独立 noisy-OR 不同的部分来自共享变量；不能以写出联合分布本身宣称创新。

## 当前训练证据直接否定了这条差异的用途

读取冻结四臂的 `events.jsonl.gz` 和 `rows.jsonl.gz`，没有改变角色。F_base 中只存在以下四个恶意多候选事件：

| 事件 | 候选连接数 |
|---|---:|
| santos webshell:231 | 2 |
| santos webshell:391 | 11 |
| santos webshell:402 | 2 |
| santos download:wphashcrack | 2 |

四个集合共 17 个连接，全部保持未知标签；集合彼此不交，也不与其他 F_base 正动作候选集合相交。因而在独立连接 Bernoulli 模型下，上述联合概率恰好分解为四个 noisy-OR 因子的乘积。所谓利用交叠消除矛盾在这些训练材料上没有作用对象。这四个事件还属于同一条 santos 攻击链，不能称四个独立攻击环境。

F_expansion 的 473 条 C2 连接属于 `all_confirmed`，每条已经有正标签；不能故意降成一个“至少一条为攻击”的弱袋，再把重新学回强标签算新增信息。

普通 MIL 可以作为以后使用这四条弱事件的监督方式，但目前没有理由为它单开算法实验，更没有理由打开保留来源寻找能支撑该设想的交叠。这里结束的是“利用现有交叠约束形成创新”的候选；不是说多候选证据无用，也不是说 MIL 普遍无效。

## 已核一手近邻及证据层级

- [Learning from weak labelers as constraints，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e91fb65c6324a984ea9ef39a5b84af04-Abstract-Conference.html)：读正式页面摘要。它把弱标注器的平均错误界写成约束并交替求软标签与模型。当前材料没有这些错误界；本轮没有采用其实现或保证。
- [Weak to Strong Learning from Aggregate Labels，UAI 2025](https://proceedings.mlr.press/v286/makhija25a.html)：读正式页面摘要。它区分比例标签与 OR 标签，并研究弱到强学习；其结论不能代替当前事件的物理对应。没有把该论文当作本项目已复现的模型。

可重查输入：[冻结事件表](../coverage_basis_20261119/batch/events.jsonl.gz)、[冻结连接表](../coverage_basis_20261119/batch/rows.jsonl.gz)。这里的计数不新增标签，不替代原连接与动作证据。
