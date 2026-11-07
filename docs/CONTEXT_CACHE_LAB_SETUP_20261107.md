# 上下文缓存实验室环境安装（2026-11-07）

本轮使用独立的 Windows Python 3.12 64 位虚拟环境、`torch==2.6.0+cu126` 和固定源码 TabICLv2。安装入口为 [setup_context_cache_lab.ps1](../scripts/setup_context_cache_lab.ps1)，直接与间接依赖列在 [requirements-context-cache-lab.txt](../requirements-context-cache-lab.txt)。不调用历史实验安装器，也不修改旧锁文件。

## 1. 安装

在实验室检出本轮代码后，从仓库根目录执行。`-Python` 填已有 Python 3.12 的真实路径，`-VenvDir` 使用本轮专用目录；无需激活虚拟环境。

```powershell
.\scripts\setup_context_cache_lab.ps1 `
  -Python 'C:\Users\lab\AppData\Local\Programs\Python\Python312\python.exe' `
  -VenvDir 'D:\venvs\tracelight-context-cache'
```

脚本需要 `git` 在 PATH 中。NVIDIA 驱动须能运行 CUDA 12.6 wheel；脚本通过实际 CUDA 张量矩阵乘法核验驱动与运行时。若需选择 GPU，先在同一个 PowerShell 会话设置 `$env:CUDA_VISIBLE_DEVICES='0'`，再执行安装与后续实验命令；当前实验入口使用可见的 `cuda:0`。

脚本按顺序完成：

1. 核验 Python 版本，建立或复用不继承系统包的虚拟环境。
2. 若 `local/vendor/tabicl` 不存在，克隆作者仓库并检出 `d83091b845a8c5f7c087bd54964f755a9e560f45`；已有目录须处于该 commit 且没有已跟踪文件修改。
3. 安装固定 pip、完整专用依赖清单、CUDA PyTorch、固定源码 TabICL 和本项目。后三项安装使用 `--no-deps`，其 Windows 运行依赖和构建依赖已在清单中固定安装，随后由 `pip check` 核对。
4. 核对 15 个模型文件和 2 个预处理文件的源码哈希，记录 CUDA 设备、运行时、显存容量及合成矩阵运算结果。
5. 运行 8 个上下文缓存测试文件，以及原生张量和原始 CSV 两个随机微型模型合成入口。

每个外部命令非零退出都会立即中止。脚本不下载 checkpoint 或真实 CSV，不启动 `lab-compare`。合成检查期间启用 Hugging Face 离线模式，结束后恢复调用前的环境变量。

## 2. 报告与复查

默认报告目录为 `local/context-cache-setup/<时间-随机后缀>`；可通过 `-CheckOutput 'D:\reports\context-cache-setup-001'` 指定一个尚不存在的目录。成功时终端输出虚拟环境 Python 路径和报告目录。

| 文件 | 核验内容 |
| --- | --- |
| `environment.json` | Python、Torch/CUDA、可见设备、合成 CUDA 运算、TabICL 安装位置与源码哈希 |
| `pip-install-report.json` | 本次专用依赖安装记录 |
| `pip-freeze.txt` | 包含间接依赖的实际安装版本 |
| `synthetic-tests.xml` | 8 个专用测试文件的通过/失败情况 |
| `native-synthetic/synthetic_report.json` | 完整/压缩 KV、序列化与独立进程预测检查 |
| `csv-synthetic/` | 官方预处理、阈值登记和原始 CSV 独立预测检查 |

直接运行后续命令时固定使用同一解释器：

```powershell
$python = 'D:\venvs\tracelight-context-cache\Scripts\python.exe'
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'pip check failed' }
& $python -m ids.context_cache --help
if ($LASTEXITCODE -ne 0) { throw 'context-cache entry point failed' }
```

真实权重、数据准备、角色登记、预检与正式运行见[实验室完整命令](CONTEXT_CACHE_LAB_RUN_20261107.md)。环境准备成功与真实模型比较完成分别记录。

## 3. 依赖依据与本机核验

[作者固定 commit 的依赖声明](https://raw.githubusercontent.com/soda-inria/tabicl/d83091b845a8c5f7c087bd54964f755a9e560f45/pyproject.toml)要求 Torch ≥2.2、scikit-learn ≥1.3、einops ≥0.7、tqdm ≥4.64 和 tokenizers ≥0.20。本清单保留项目 NumPy 1.26.4、scikit-learn 1.5.1 等已有直接依赖，并固定 einops 0.8.0、huggingface-hub 0.28.1、tokenizers 0.21.0、tqdm 4.66.5。tokenizers 用于满足作者包的依赖声明；当前 CSV 转换不使用文本分词。

PyTorch 安装源与 CUDA 12.6 组合来自 [PyTorch 2.6.0 官方安装命令](https://pytorch.org/get-started/previous-versions/#v260)。本轮不需要 torchvision、torchaudio 或 `tabicl[all]`；作者的可选训练、解释、预测和文档依赖不加入环境。

直接与间接依赖均固定，包括 Torch 在 Windows 上依赖的 sympy、filelock、fsspec、networkx、Jinja2、typing-extensions 和 setuptools；安装后的完整版本仍保存为 `pip-freeze.txt`。运行依赖沿用本机合成检查环境，tokenizers 调整为满足作者要求的 0.21.0；构建工具固定为 Hatchling 1.27.0，并补齐其要求的 packaging 24.2、pathspec 和 trove-classifiers。

本机已通过 PowerShell 脚本解析、内嵌 CUDA 探针的 Python 语法检查，以及 Python 3.12 Windows 的 `pip install --dry-run --ignore-installed --only-binary=:all:` 全部固定依赖解析；项目与固定 TabICL 的构建元数据 dry-run 也通过。本机没有执行此安装脚本，实验室首次执行后才产生该新虚拟环境的安装和 GPU 核验结果。
