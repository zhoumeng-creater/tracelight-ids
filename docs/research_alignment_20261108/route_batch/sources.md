# 一次路线比较的现成方法与接口依据

核对日期：2026-11-08。这里区分官方方法事实、项目采用的运行配置和待实测效果。新参照使用 2025—2026 年正式方法；包的更新日期不用于给旧算法重新定年。

## TabM：直接监督与容量对照

- 方法：TabM，ICLR 2025。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf)
- 官方实现固定为 `yandex-research/tabm@28e47ae301c92ec37787dde1ce923a0793f405b4`，`tabm.py` 的版本为 `0.0.3`。[模型源码](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/tabm.py)
- `TabM.make` 在不使用数值嵌入时默认 `n_blocks=3, d_block=512, k=32, dropout=0.1, activation='ReLU', arch_type='tabm', start_scaling_init='random-signs'`；使用嵌入时默认层数变为 2。因此不能同时换嵌入、层数、宽度后把差异全部归为参数量。
- 官方教学例使用 AdamW，`lr=0.002, weight_decay=0.0003`，batch 256，梯度范数裁剪 1.0，验证早停 patience 16；训练时分别计算 k 个成员的交叉熵，预测时平均概率。它的近乎无限 epoch 和逐轮打印 test 仅为教学流程，不直接复制进本项目。[官方 notebook](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb)
- 官方示例的 QuantileTransformer 会改变输入表示；当前比较若使用冻结的 48 数值输入，不自动叠加它。固定层数、k、输入和训练配方后改变宽度，可检验这两个容量配置的差异；仍不能证明整个模型家族的容量上限。
- 依赖：`torch>=1.12,<3`，`rtdl_num_embeddings>=0.0.12,<0.1`，`typing_extensions>=4.6,<5`。TabM 无须下载预训练权重。[包声明](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/pyproject.toml)

## TabICLv2：官方整管线参照

- 方法：TabICLv2，ICML 2026。[正式论文](https://proceedings.mlr.press/v306/qu26a.html)
- 沿用项目已冻结的 `soda-inria/tabicl@d83091b845a8c5f7c087bd54964f755a9e560f45`，包版本 `2.2.0`；权重名 `tabicl-classifier-v2-20260212.ckpt`。模型源文件、预处理文件和本次新增使用的 `classifier.py`／`base.py` 均按 LF 规范化 SHA-256 核对。[分类器源码](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_sklearn/classifier.py)
- 官方默认 8 视图，`norm_methods=['none','power']`，列置换 `latin`，类别置换 `shift`，离群裁剪阈值 4.0，temperature 0.9，先平均 logits。`none` 仍包含标准化和离群处理。
- 本次实现接受原始 numeric ndarray，避免将端口类别交给 OrdinalEncoder。1／8 视图由同一官方接口构建，并保存实际视图数；默认预处理会按视图预算选择配置。与现有 protected-ID 单视图相比，还改变了预处理、置换、温度和评分接口，因此是整管线参照，不能把全部差异归于集成。
- 工程设置明确覆盖默认值：`kv_cache=True`；每批处理 1 个 estimator，避免把 8 份完整缓存构建同时推入显存；构建 device 由协议给出，`use_amp=False, use_fa3=False`；构建后使用上游自身设备迁移方法转到 CPU。评分固定 FP32、1 线程、每批最多 128 条查询。缓存及模型权重总成本必须计入；增加视图不等于增加骨干参数。
- 输入传入后仍经过官方裁剪／power 变换，不能继承 protected-ID 表示的身份保持保证。[预处理源码](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_sklearn/preprocessing.py)

## TabPFN-2.5：可选的另一预训练模型

- 方法：2025 技术报告。[原文](https://arxiv.org/html/2511.08667v1)
- 固定运行包 `tabpfn==9.2.0`，对应 commit `f02b3b5941717472eafd078c15ae2043676b3fcf`。包默认模型已是 3.5，必须显式调用 `TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, ...)`。[版本工厂](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/classifier.py#L620)、[发布版本](https://github.com/PriorLabs/TabPFN/releases/tag/v9.2.0)
- 模型库 `Prior-Labs/tabpfn_2_5`，核对 revision `6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b`；文件 `tabpfn-v2.5-classifier-v2.5_default.ckpt`，42,935,499 bytes；SHA-256 `5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12`。这些元数据来自官方 Hugging Face API，未在本机下载或加载真实权重。[模型库](https://huggingface.co/Prior-Labs/tabpfn_2_5)
- 该 default 权重经过真实数据继续训练；`default-2` 才是纯合成训练的分类权重。论文附录 C.1 列出的 43 个数据集未出现 NF-ToN-IoT 名称，但名称未出现不能保证完全无污染，也不能将 default 宣称为纯合成预训练模型。[模型卡与权重区别](https://huggingface.co/Prior-Labs/tabpfn_2_5#different-checkpoints)、[训练数据清单](https://arxiv.org/html/2511.08667v1#A3.SS1)
- 本次固定 8 个 estimator、temperature 0.9、平均概率、`balance_probabilities=False`、不额外下采样、`inference_precision=torch.float32`、`fit_mode='fit_with_cache'`、`kv_cache_precision='auto'`、预处理 1 线程；输入为原始数值矩阵。9.2 的 `None` 会选架构默认缓存精度，可能量化为 int8；显式 `auto` 才保留计算 dtype。官方 `to('cpu')` 会将 executor 迁到 CPU，并由明确的 FP32 设置关闭 autocast。评分还同时限制 Torch 和 BLAS 线程数。这是另一整管线参照，不能独立归因于模型尺寸。[精度与迁移实现](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/base.py)
- 固定代码 V2.5 限制 50,000 支持行、2,000 特征、10 类；当前网站汇总页写 100,000 行，与冻结代码不同，执行遵循冻结代码。P8192 不触及该差异。[固定推理配置](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/inference_config.py#L531)
- 包要求 Python≥3.10、torch≥2.5，现有 Python3.12／Torch2.6 满足这两个下限；另有 pydantic、pydantic-settings、skrub、safetensors、lightgbm 等依赖，须由独立环境安装预检确认，不能只验证顶层包导入。[依赖声明](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/pyproject.toml)
- 权重使用 TABPFN-2.5 License v1.1，含非商业研究用途。包的自动下载路径会先检查 Prior Labs 许可接受状态。官方 Hub 当前 metadata 的 `gated=false` 不表示包没有许可检查。本批次仅接受已准备的本地权重，缺失则记录不可用；不自动登录、下载或接受条款。[许可](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE)、[下载入口](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/model_loading.py#L546)

## 评分与工件语义

两个官方接口统一读取 `predict_proba` 的 class 1 概率，先转 float64，再以 float64 epsilon 裁剪到 `[eps, 1-eps]`，计算 `log(p)-log1p(-p)`。这是概率接口派生分数，不是原生 logits；API 中已饱和的 0／1 会保留并列。每臂应在同一正常校准行上登记自己的阈值，不复用旧完整 KV 阈值。

`FoundationModel.save` 保存可信 joblib 重建配方、原始支持数据和外部权重摘要，不保存权重和已拟合缓存。`load_foundation` 核对权重后重新拟合／构建缓存，记录完整加载重建时间。配方字节数不能单独当作部署包大小；实际部署还需对应权重、依赖及运行时缓存。调用方在 joblib 反序列化前核验登记的工件 SHA。

本机验证只使用 API mock 和随机初始化微型官方 TabICL。真实预训练模型效果、CPU 成本及 TabPFN 实际运行均由实验室返回结果。

## 实验室环境与权重准备

核心环境沿用 `requirements-context-cache-lab.txt` 的 Python 3.12／CUDA Torch 2.6.0+cu126 依赖，另固定 `tabm==0.0.3`、`rtdl_num_embeddings==0.0.12`。安装入口创建新的 venv，再调用已有 context-cache 安装器；不修改历史实验环境。TabPFN 的数值依赖与核心环境一致，元数据解析没有显示二者必须冲突。仍将它放入独立 venv，便于缺少许可或权重时继续核心批次，并在登记的路径补齐后恢复可选臂。

在实验室仓库根目录执行，路径按首次登记的实际位置填写：

```powershell
# 一次准备 core 和可选 TabPFN 环境；两个目录都必须尚不存在。
.\scripts\setup_route_batch_lab.ps1 `
  -Python 'D:\python\python.exe' `
  -CoreVenvDir 'D:\venvs\tracelight-route-core' `
  -WithTabPFN

# 若首次只建立 core，之后仅补同父目录的可选环境：
.\scripts\setup_route_batch_lab.ps1 `
  -Python 'D:\python\python.exe' `
  -CoreVenvDir 'D:\venvs\tracelight-route-core' `
  -TabPFNOnly
```

可选环境默认 `D:\venvs\tracelight-route-tabpfn`；自定位置使用 `-TabPFNVenvDir`，并在首次注册时将对应解释器路径传给 runner 的 `--tabpfn-python`。上述两种安装方式择其适用者执行。安装器记录 pip dry-run、安装 report、`pip check`、完整 freeze、依赖导入和合成测试；不加载或下载预训练权重。当前本机已完成 Windows／Python 3.12 的 fresh binary-only 依赖解析：核心 44 项、TabPFN 60 项（均含 PyPI Torch 2.6.0，用于验证依赖闭包）。实际 CUDA 环境的安装及 `pip check` 由实验室脚本完成。解析报告保存在 `local/route-batch-dependency-audit-20261108/`。

TabPFN 权重准备由使用者完成官方许可流程：在 [Prior Labs 账户](https://ux.priorlabs.ai/) 登录并按 License 页面确认适用条款，再按官方模型库／下载器流程取得上文的固定 checkpoint。官方 `browser_auth.py` 说明了网页登录、已接受许可核验和无界面 token 流程；本脚本不代为接受条款、不读取或输出 token。Hub 文件可公开读取不等于可以绕过许可流程。[官方许可流程实现](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/src/tabpfn/browser_auth.py)

将权重保留为 `tabpfn-v2.5-classifier-v2.5_default.ckpt`，默认放在已登记 TabICL 权重的同一目录；用 `Get-FileHash -Algorithm SHA256` 核对上文摘要。首次注册即保留该未来路径；未准备时该臂记录不可用，其余臂继续。补齐同一路径的解释器与权重后，可以显式恢复开发部分的不可用项，不新增候选或改登记路径。新通信组确认一旦冻结就不再加入后来补齐的模型；要让该臂同时参与开发与确认，应在首次完整运行前准备好权重与环境。
