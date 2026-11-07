# 部署架构与成本审查

2026-11-06。依据 2026 年 TabICLv2、Localized TabICLv2、QCOC 原文及固定官方源码。本文是实现设计；本次未加载权重或运行训练、推理实验。

本附件负责源码接口与成本审查；主算法、符号及验证协议以主稿 CALIBRATION_AWARE_CONTEXT_DESIGN_20261106.md 为准。

本轮工程实现见[完整／压缩 KV 最小入口](../../CONTEXT_CACHE_IMPLEMENTATION_20261106.md)：保留官方 Stage 1/2，Stage 3 使用独立加权缓存与显式层执行，避免上游容器丢失 mass/N。当前仅普通 QCOC 编译与原生数值输入，36 项合成测试通过；下文敏感几何和真实成本为后续设计。

## 1. 架构判断

主候选采用**一次压缩、所有查询共用的 Stage 3 KV 缓存**。先在完整缓存上累积逐源状态、逐坐标的校准 margin 敏感度 g_{i,d}，即攻击分数与校准阈值的 Jacobian 差的平方；再据此进行相同 M 预算下的原型分配与逐坐标中心计算，最后使用 QCOC 原有的 value 闭式拟合。研究增量是压缩前的源状态特定几何，不是共享 KV 或 value 拟合。QCOC＋CRT 只调整 value 的路线保留为对照。

动态近邻检索不作为第一版主结构。它保留全量表示库，每条查询还需检索、搬运邻居并重算局部上下文。静态支持子集加完整 KV 可作为结构对照：它也能删除全量表示库并支持批推理，但重新编码子集与压缩完整缓存不是同一算子。

官方代码固定为 [d83091b845a8c5f7c087bd54964f755a9e560f45](https://github.com/soda-inria/tabicl/tree/d83091b845a8c5f7c087bd54964f755a9e560f45)，提交时间 2026-10-27。分类器默认 checkpoint 为 tabicl-classifier-v2-20260212.ckpt；默认 8 个 ensemble，默认不启用 KV 缓存。参照必须主动打开完整 KV，而不能沿用默认无缓存配置。[classifier.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_sklearn/classifier.py#L282)

## 2. 三阶段究竟保留什么

| 部件 | 构建时 | 在线查询时 | 压缩后保留 |
|---|---|---|---|
| 预处理 | 支持集拟合编码器、常量列过滤、归一化、排列规则 | 对查询执行同一变换 | 参数、词表、特征顺序及类别映射 |
| Stage 1 列编码 | 每列或特征组的诱导注意力汇总支持集 | 查询只访问诱导状态的缓存 | 全量支持集导出的列 KV；其槽数由诱导点数决定 |
| Stage 2 行交互 | 将每行特征聚合为 512 维表示 | 每条新查询仍要执行行交互 | 骨干权重；无需保存所有支持行表示 |
| Stage 3 ICL | 完整支持集逐层产生 K/V | 查询逐层访问同一套缓存 | 每层每头 M 个 key、value、簇质量及原规模 N |
| 决策 | 独立正常校准集确定阈值 | 比较攻击分数与冻结阈值 | 阈值及分数方向等元数据 |

Stage 1 缓存保存的是 ISAB 第二次注意力的 K/V，对应固定数量的诱导状态，而不是 N 条原始行。因此保留完整列状态，不意味着部署端必须保留 N 行。Stage 2 不能在新查询上省掉；将训练行提前编码只消除了训练行的重复计算。[layers.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/layers.py#L686)

官方提供两种不同缓存：

- repr：保存列 KV 和全部训练行表示；每次预测重跑 Stage 3。
- kv：保存列 KV 和 Stage 3 每层完整 K/V；预测只投影查询 Q，并访问缓存。

两者在时间与空间上有不同取舍，不能都简称“已经缓存”。[tabicl.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/tabicl.py#L615)、[attention.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/attention.py#L237)

## 3. Localized TabICLv2 的证据边界

Localized TabICLv2 是 2026 年工作坊论文，按 Stage 2 相似度选择每条查询的上下文。论文采用 kv_cache='repr'，其 Full 对照也复用前两阶段、重跑第三阶段；报告的约 249 倍单查询加速不能直接解释为相对官方完整 KV 的加速。它仍需要完整训练表示库，且 A100 上的 accuracy/AUC 结果没有回答低 FPR 与 CPU 常驻状态问题。其 k=128 的批推理中位加速低于 1，说明逐查询重复上下文会抵消收益。[论文](https://arxiv.org/html/2608.16429v1)

本次未找到可核验的 Localized 作者实现。官方源码还暴露了检索表示的一个具体问题：

1. Stage 1 建缓存时直接加入支持行的标签嵌入；query 模式不加自身标签。
2. repr 存储前，Stage 3 的标签嵌入又加到训练行表示。
3. 因而直接拿 row_repr 与 query 表示算余弦，不能当成同一 x→z 映射下的距离。只减去第二次标签嵌入也不足以消除第一项不对称。

若实现检索对照，应固定列缓存，将候选支持行重新按 query 模式编码，建立专用检索表示。原有支持行表示仍用于 ICL 上下文。列缓存可以是训练标签条件化的，但检索两侧不得分别使用带自身标签和不带自身标签的直接输入路径。[embedding.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/embedding.py#L773)、[learning.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/learning.py#L551)

## 4. QCOC 可用性与最小适配

QCOC 是 2026-10-08 预印本。原文提供联合 K/V 原型、簇质量、原 N 以及带锚 value 拟合公式，未提供已核验的作者仓库链接。精确题名检索找到第三方 teddytennant/qcoc；其 README 明说只实现普通缩放点积注意力，不包含 TabPFN 的 scalable-softmax，也不运行 OpenML。它不是已核验的作者实现，不能作为复现来源。[QCOC 原文](https://arxiv.org/html/2610.11784v1)、[第三方仓库固定版本](https://github.com/teddytennant/qcoc/tree/bd4ef77039c6e6b615c15e67a5ccfd2b7325e2ee)

自建适配保持以下接口：

~~~python
CompressedEntry:
    key          # [ensemble, heads, M, head_dim]
    value        # 同形状；敏感几何确定原型后，采用 QCOC 原有闭式拟合
    log_mass     # [ensemble, heads, 1, M]
    source_count # 原支持集规模 N，不等于物理槽数 M
~~~

对 Stage 3 每层的查询：

\[
q'=\operatorname{QASSMax}(q,N),\qquad
o=\operatorname{softmax}(q'\widetilde K^\top/\sqrt{d_h}+\log w)\widetilde V.
\]

须明确 QASSMax 与 scaled-dot-product 的缩放分工，避免重复乘温度。Stage 1 继续使用它原来的诱导点长度，不能把全模型所有注意力的 source_count 都改成 N。

官方 attention.py 以 k.size(-2) 调用 QASSMax，直接替换更短 K/V 会把 N 偷换为 M；必须显式传入 source_count。质量项通过加性 attention bias 实现。当前 FlashAttention3 分支不支持自定义 mask，会退出该分支；完整缓存须增加相同后端的受控比较，同时另报告官方最快可用后端的完整缓存结果。前者解释压缩作用，后者判断系统是否真有收益。[attention.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/attention.py#L97)

缓存容器的 to、concat、slice_batch、empty_like、保存与加载都须携带 log_mass 和 source_count。当前 KVCacheEntry 只有 key/value，两项新增数据不能只挂在临时对象上，否则切分 ensemble 或 CPU/GPU 迁移时会丢失。[kv_cache.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/kv_cache.py#L19)

**敏感度计算与对照训练需绕开官方推理包装。** sklearn 调用和 InferenceManager._run_forward 均使用 torch.no_grad。仅把完整 K/V 或对照的压缩 value 设置为 requires_grad 后调用 predict_proba，不会获得需要的梯度。应复用冻结层的权重、Q 投影、归一化、残差、FFN 与 decoder，建立可微 Stage 3 adapter；风险查询的前两阶段表示可以离线固定。主方法只在完整缓存上求 margin Jacobian 并累积 g_{i,d}，不通过这个步骤更新骨干或源 K/V。在线 adapter 切回 eval/no_grad。首版保持所有层相同 M，避免可变长度在 padding 后把实际预算又涨回最大 M。[inference.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/inference.py#L1227)

## 5. 离线构建与部署物件的分离

离线流程：

1. 仅用支持集 P 拟合预处理并构建列缓存和完整 Stage 3 KV，固定类别顺序、ensemble 排列与 checkpoint。
2. 逐层逐头收集完整源状态 K/V 及 QCOC 闭式拟合所需的 query 向量与注意力输出。完整注意力输出可在构建时截取；不应为 value 拟合无意中再跑一次相同 N² 运算。
3. 在完整缓存上，用独立风险查询 Q 累积逐源状态的 margin Jacobian 差平方 g_{i,d}，以该几何执行相同 M 预算下的原型分配与逐坐标中心计算，再采用 QCOC 原有 value 闭式拟合。QCOC＋CRT 的 value-only 更新仅作为对照。风险查询、开发查询和最终正常校准集 C 都不得先进入支持集的列缓存。
4. 固定候选，独立校准阈值，导出紧凑包。
5. 冷进程只加载紧凑包后预测，核对该进程不依赖原数据路径或完整缓存文件。

线上包只保留：

- checkpoint 权重或明确的共享权重依赖、版本和哈希；
- 预处理参数、类别映射、特征排列；
- 列 KV；
- 紧凑 Stage 3 KV、log_mass、source_count；
- 冻结阈值、dtype、ensemble 数和元数据。

可从部署包移除：

- P 的原始特征及标签；
- ensemble_generator_.X_、y_；
- 每个预处理器的 X_transformed_；
- 全量 row_repr、检索索引及完整 Stage 3 KV；
- 压缩器的簇分配、敏感度 g_{i,d}、拟合 Q/Y、Gram 矩阵；
- 风险训练查询表示、优化器状态、开发预测。

官方 save(save_training_data=False, save_kv_cache=True) 已会清空上述 X_、y_ 和 X_transformed_；但这是序列化清理，原有进程内的对象未必因此释放。新格式须保留同等语义，并用冷进程验证。默认 save_model_weights=False 仅使权重从文件中省略，部署仍需它；不能拿这个小文件作为完整模型大小。[base.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_sklearn/base.py#L211)

Stage 1 保存 N 的统计影响，不保存 N 行，这是需要保留的固定状态。它也包含支持标签的影响；删除 Stage 3 某些行不会让这些行自动成为独立校准样本。

## 6. 空间账本

记 E 为总 ensemble 数；N 为支持行数；M 为压缩槽数；F 为预处理后有效特征数；G 为进入列编码的特征组数；C 为预留 CLS 数；I 为诱导点数；d 为列/行维度；D 为 Stage 3 维度；L_c、L_r、L 为列层数、行交互层数和 ICL 层数；H 为 ICL 头数；b 为单个缓存元素字节数。以下为张量字节，不含框架与分配器。

\[
S_{\rm col}=2E L_c(G+C)I d b,\quad
S_{\rm repr}=E N D b,\quad
S_{\rm fullKV}=2E L N D b.
\]

\[
S_{\rm compact}=2E L M D b+E L H M b_{\rm mass}+S_{\rm count}.
\]

\[
S_{\rm deploy}=S_{\rm weights}+S_{\rm preprocessing}+S_{\rm col}
 +S_{\rm compact}+S_{\rm metadata}.
\]

默认 same 特征分组仍产生 G=F 个组，不是 F/3。分类配置为 d=128、D=512、I=128、C=4、L_c=3、L=12、H=8；这些数与论文 A.4 及固定源码一致。[模型配置](https://arxiv.org/html/2602.11139v1#A4)、[分组代码](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/embedding.py#L208)

例如 E=1、FP32、F=42、N=8192、M=512：

| 状态 | 字节换算 |
|---|---:|
| 完整 Stage 3 KV | 384 MiB |
| 紧凑 Stage 3 KV | 24 MiB |
| 紧凑簇质量 | 0.1875 MiB |
| 完整列 KV | 17.25 MiB |
| 官方骨干仅 Stage 3 主要权重矩阵 | 至少 96 MiB |
| 全量行表示，若错误地保留 | 16 MiB |

96 MiB 来自 12 层、每层 attention 的 4D² 与 FFN 的 4D² 参数，未计 bias、QASSMax、列/行模块、decoder；这是源码结构推导的下界，不是已测权重总量。官方缓存推理虽然只用 Q 投影，仍持有完整权重模块。

因此，上例中 Stage 3 KV 缩小 16 倍，**完整常驻张量的缩小比例至多约 3.62 倍**：额外权重、预处理及运行时内存会继续降低这个比例。若 M 再变小，固定骨干与列状态逐渐占主导，不能继续按 N/M 预言系统收益。默认 E=8 会把列缓存和 ICL 缓存大致乘 8，权重则共享；论文单 ensemble 成本不能直接套在默认分类器上。

CPU 若禁用 AMP，官方会把缓存转回 FP32。GPU BF16 缓存不能直接作为 CPU 驻留字节预测。[base.py](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_sklearn/base.py#L178)

官方 cache_size_mb 只数已知缓存张量，既不含骨干和预处理，也不会自动数新增 log_mass；应另做完整 artifact 清单和进程峰值统计。[缓存计数](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_model/kv_cache.py#L251)

## 7. 时间账本与批推理

记 B 为一批查询数，q 为一次部署生命周期内查询总数。以下忽略常数但保留主要矩阵运算：

\[
T_{\rm fullKV}(B)
=T_{\rm preprocessing}(B)+T_{12}(B)+
O\!\left(EL[BND+BD^2]\right).
\]

\[
T_{\rm compact}(B)
=T_{\rm preprocessing}(B)+T_{12}(B)+
O\!\left(EL[BMD+BD^2]\right).
\]

第二项 BD² 包含 Q/输出投影和 FFN；M 小到一定程度后它不会消失。前两阶段在线查询成本大致包含

\[
T_{12}(B)=O\!\left(
E L_c B(G+C)[Id+d^2]
+ E L_r B[(G+C)^2d+(G+C)d^2]\right).
\]

以上用于定位成本，不代替硬件计时。完整 KV 把 N² 上下文计算移到构建阶段；若用 repr 参照，每次调用还会重跑 O(EL[N²D+ND²])。这正是不能用 repr 参照证明完整 KV 加速的原因。

精确全库检索需要 O(EBND) 距离计算，常驻库 O(END)，近似索引还需额外结构。逐查询局部 Stage 3 的主要成本为 O(EBL[k²D+kD²])，同一邻居可能在很多 query 的局部上下文中重复计算。不得将完整上下文的高层 K/V 直接切片称为“局部重算”：那些 K/V 已包含被删上下文的影响。

紧凑共享缓存的 B 条查询均沿同一组 K/V 执行，不按 query 复制缓存。当前官方 attention 的 train_size 路径只允许所有行关注支持行，查询之间没有互相作为 K/V 的需要。实现应将 log_mass 以广播视图加到 [E,H,B,M] logits，避免显式复制 E×H×B×M bias。

权重与列状态固定后，批量增大可摊薄 Python 调用、预处理和矩阵核启动成本，但峰值激活会上升。至少分别报告 B=1 的延迟，以及固定批量的吞吐；不把 1000 条批量总时延换名为单请求延迟。完整缓存和压缩缓存保持相同线程数、dtype、ensemble、预处理与 attention 后端。

QCOC 编译还包括联合聚类、拟合方程和求解。逐头拟合的 Gram 累积含 O(N_cal M²)，求解含 O(M³)，不能只报原型数；可分块累积 PᵀP 与 PᵀY，避免长期保留 N_cal×M 权重矩阵。原文在 adult、N=8192、M=512 的单核编译为 283.40 秒，且从完整缓存已经构建之后开始计时。该数字不是本机预测，但说明离线成本必须单列。[QCOC 附录 H](https://arxiv.org/html/2610.11784v1#A8)

主方法还须单列完整缓存上的 margin Jacobian 计算与 g_{i,d} 累积成本。按查询分块累积平方差，避免保存“全部查询×全部源状态×全部坐标”的 Jacobian；g_{i,d} 本身仍是与完整源状态规模相关的离线物件，不能漏记离线峰值。它不进入在线包，故上述部署空间与在线时间公式不变。

若单查询压缩比完整缓存节省 Δt，额外敏感度计算与编译成本为 ΔC，则盈亏平衡查询数为 q*=ΔC/Δt。Δt≤0 时没有在线耗时的有限盈亏平衡点。完整缓存构建是两者共有成本，可以在这项增量比较中抵消；比较从零开始的系统总成本时仍须全部计入。CPU/GPU/disk offload 改变状态位置和搬运，不代表状态被压缩。

## 8. 实现完成后必须回答的几个问题

- 在 M=N、质量全为 1、source_count=N 时，适配 attention 是否与固定源码完整 KV 数值一致？无压缩应恢复同一算子。
- 完整缓存上的 margin Jacobian 是否有限、能区分源状态，且骨干与源 K/V 的数值保持不变？对 value-only 对照，另检查 value 的有效梯度；这能直接识别 no_grad 包装问题。
- 同一个紧凑包在 B=1 与分批预测中是否给出一致分数，保存加载后是否仍一致？
- 冷进程中是否没有 N 行表示、检索索引、原始 X/y 或完整 KV 依赖？新元数据是否出现在字节清单中？
- 相同字节预算下，源状态特定几何是否优于原始 QCOC、QCOC＋CRT value-only 对照及主稿规定的几何消融？这一比较决定压缩几何的增量。
- 若完整 TabICLv2 在开发材料上没有可保留的低 FPR 检出优势，或紧凑版仍不满足实际 CPU/内存预算，则不以压缩率替代部署价值。最终部署模型与缓存压缩研究结论可以分别成立。
