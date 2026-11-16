# 模型与训练设计的一手来源

核对日期：2026-11-16。以下资料支持设计起点，不提供本项目低误报或跨实现成功保证。

## TabM

- [作者仓库](https://github.com/yandex-research/tabm)、[固定模型源码](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/tabm.py)、[固定示例](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb)。现行包0.0.3；项目已有固定commit和rtdl_num_embeddings0.0.12。
- [至主线差异](https://github.com/yandex-research/tabm/compare/28e47ae301c92ec37787dde1ce923a0793f405b4...b53fd7cf2d9be2d66b918ca6cf66e0365e4ce11c)为非affine LayerNorm转换修复，没有更换当前标准骨干的理由。
- 官方提供共享主参数的MLP集成、连续数值嵌入和类别接口；带嵌入默认2块，不带默认3块，宽512、k32、dropout0.1。训练分别计算成员损失，分类预测平均概率。
- 示例采用PLE48箱/16维/version B/activation=False；AdamW0.002/0.0003、batch256、梯度裁剪1、验证选轮和patience16。
- 本项目适配：log1p＋StandardScaler、显式缺失、pooled低FPR验证指标、200轮上限、禁止逐轮看E/D。示例的accuracy选轮和test展示不照搬。代码Apache-2.0。

## TabICLv2

- [正式论文](https://proceedings.mlr.press/v306/qu26a.html)、[作者仓库](https://github.com/soda-inria/tabicl)、[权重卡](https://huggingface.co/jingang/TabICL)。包2.2.0，分类权重tabicl-classifier-v2-20260212.ckpt；代码/权重BSD-3-Clause。
- 项目commit d83091b845a8c5f7c087bd54964f755a9e560f45；核对主线a626a1b67049b0b1748ebe8c92f7df40aaebd334。[变更](https://github.com/soda-inria/tabicl/blob/a626a1b67049b0b1748ebe8c92f7df40aaebd334/CHANGES.md)针对repr缓存offload设备，不能借此解释旧kv_cache=True的全部数值差异。
- [分类接口](https://github.com/soda-inria/tabicl/blob/a626a1b67049b0b1748ebe8c92f7df40aaebd334/src/tabicl/_sklearn/classifier.py#L423)无普通sample_weight；上下文、预处理和集成构成完整系统，不等同于来源加权监督目标。预训练Muon不直接移植为小MLP优化器。

## TabPFN

- [v10.0.0稳定发布](https://github.com/PriorLabs/TabPFN/releases/tag/v10.0.0)，2026-11-13；[包说明](https://pypi.org/project/tabpfn/10.0.0/)。默认3.5，项目旧9.2.0明确使用2.5。
- [标准3.5权重](https://huggingface.co/Prior-Labs/tabpfn_3_5)：tabpfn-v3.5-20260909.safetensors。[权重许可](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE)独立于代码Apache-2.0，包含非商业/非生产用途及使用者资格条件。
- [v10分类接口](https://github.com/PriorLabs/TabPFN/blob/v10.0.0/src/tabpfn/classifier.py#L1011)并非R26来源加权训练目标。旧2.5单线程CPU成本不代表新版GPU成本；本轮不下载/运行更多变体。

## 树模型与验证工具

- [HGB官方接口](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)：项目固定1.5.1，使用已有逐阶段概率，再以选定max_iter重训；不直接使用新版本外部验证接口。
- [CatBoost fit](https://catboost.ai/docs/en/concepts/python-reference_catboost_fit)：提供类别、权重、eval_set和最佳迭代。现行1.2.10与项目一致；旧无V、类平衡适配不能冒充新任务公平比较。
- [学习曲线说明](https://scikit-learn.org/stable/modules/learning_curve.html)：训练与开发表现可分析拟合/泛化；单个最终检查点不能证明收敛或唯一分解容量与优化。
- [partial AUC](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.roc_auc_score.html)：采用max_fpr=0.01的标准化pAUC，不与原始积分除0.01的数值混写。
- [QuantileTransformer](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.QuantileTransformer.html)：训练范围外数值映射至输出边界，因此本设计不默认使用QT处理跨传输连续量。

## 数据与外推

- [TabReD正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/571799482291411607c54984153190b0-Abstract-Conference.html)：时间划分和特征条件会改变方法排名，MLP类及GBDT仍是强参照；不证明安全任务1%误报迁移。
- [Feature-aware Modulation论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/ddc20f403d0d296c87940393838d5971-Paper-Conference.pdf)、[作者实现](https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation)：针对时间上下文与特征语义变化。B/D还改变传输和流结构，不能直接等同为相同漂移，不据此追加时间模块。
- [TQH作者README v1.2.5](https://zenodo.org/records/23905375/files/README_v125.md?download=1)：身份捷径、同Merlin传输对照。作者RF去身份字段的80/20随机划分F1（B0.992、D1.000）支持场景内可分，不支持B→D逐单元1%预算结论。
- [Zeek字段](https://docs.zeek.org/en/current/reference/logs/conn.html)：ts/duration、方向、IP字节与应用字节的区别；当前文档不能替代原采集实际Zeek版本。
- [安全机器学习方法论](https://www.usenix.org/conference/usenixsecurity22/presentation/arp)：作为评价设计基础，不列为本次新增算法。
