# 实验室单卡部署与运行

这份配方面向计划使用的 RTX 4090D 24 GB 实验室机器。**实验室机器尚未执行本配方**；本机已识别为 NVIDIA GeForce RTX 4050 Laptop GPU，PyTorch 报告约 6 GiB 显存。GPU 型号识别、CPU 模型测试、CUDA 冒烟测试和真实数据正式结果是不同证据，只有实际运行产物才能证明对应阶段完成。不要把本机结果写成 4090D 结果。

此项目的小型表格 MLP 不需要为了占满 24 GB 显存而扩大模型。GPU 主要用于 MLP 的重复训练；LR、RF、HGB 仍在 CPU 上训练。所有模型在相同 CPU/线程条件下另测部署延迟，GPU 训练时间不能替代 CPU 推理时间。

## 环境与目录

- 本部署配方固定 Python 3.12；核心依赖声明在 `pyproject.toml`，安装先使用包含开发依赖的 `requirements-lock.txt`，随后 `pip install -e . --no-deps`。
- PyTorch 固定 `2.6.0`，CUDA wheel 使用官方 `https://download.pytorch.org/whl/cu126` 索引；无需为本项目安装 torchvision 或 torchaudio。[PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v2-6-0)
- 机器须已有可用 NVIDIA 驱动。最终判断依据是 `ids doctor --require-cuda` 成功以及一次真实 CUDA 训练成功，不依据目录名或 `nvidia-smi` 标题中的 CUDA 数字。
- 建议代码、虚拟环境、原始数据、准备后的数据与 runs 都放实验室本地 SSD。尤其不要把虚拟环境、正在写入的模型和 Parquet 分区放 OneDrive 同步目录。阶段完成后再备份完整产物。
- 保留原始 CSV、来源地址/下载时间/哈希和 prepared `manifest.json`。不要手工改 prepared 文件；需要换划分时生成一个新的 prepared 目录。

一种目录组织：

```text
research/
  tracelight-ids/       # 本仓库
  venv/                # Python 3.12 独立环境
  data/raw/            # 原始 CSV 和来源记录
  data/prepared-v1/    # prepare 生成的冻结分区
  work/environment/   # 每次启动的机器信息
  work/runs/          # 开发运行产物
  work/reports/       # 验证集交互报告
```

以下路径均为可自行替换的示例，不是已经存在或已经验证过的目录。

## Windows PowerShell

在仓库根目录执行，`-Python` 指向实验室安装的 Python 3.12：

```powershell
.\scripts\setup_lab.ps1 -Python 'C:\Python312\python.exe' -VenvDir 'D:\Research Work\venv' -Device cuda
```

脚本创建或复用指定虚拟环境、安装项目和固定 PyTorch、检查依赖、检查 CUDA、运行测试。已有目录不是虚拟环境时会拒绝，不会删除目录。脚本失败立即停止。它不会下载数据或开始正式训练。若机器策略阻止脚本执行，遵循实验室管理员的脚本策略，不要改变全机策略。

接下来的命令直接调用虚拟环境解释器，不依赖激活脚本：

```powershell
& 'D:\Research Work\venv\Scripts\python.exe' -m ids doctor --require-cuda

& 'D:\Research Work\venv\Scripts\python.exe' -m ids prepare `
  --train-csv 'D:\Research Work\data\raw\UNSW_NB15_training-set.csv' `
  --test-csv 'D:\Research Work\data\raw\UNSW_NB15_testing-set.csv' `
  --output 'D:\Research Work\data\prepared-v1' `
  --data-kind mirror

& 'D:\Research Work\venv\Scripts\python.exe' .\scripts\run_lab.py `
  --data-dir 'D:\Research Work\data\prepared-v1' `
  --work-dir 'D:\Research Work\work' `
  --config .\configs\baselines.yaml `
  --device cuda
```

这里明确使用 `mirror`：若文件来自镜像，就应如实记录为镜像；只有确实来自官方原始分发源并核查后，才使用 `official`。若采用本项目固定版本的第三方镜像，先执行下面的显式下载命令，再执行上面的 prepare：

```powershell
& 'D:\Research Work\venv\Scripts\python.exe' .\scripts\fetch_unsw.py `
  --output 'D:\Research Work\data\raw' `
  --source mirror --accept-unverified-mirror
```

下载器验证训练/测试 CSV 分别为 175341/82332 行及原始 45 列结构，记录下载 URL、固定镜像 commit、字节哈希与类别计数。**这不等于已与官方文件做哈希比对**，`provenance.json` 会明确记录此限制。两个原始 CSV 包含 42 个特征及 `id/attack_cat/label` 三个其他字段。保留 provenance，不覆盖已有原始数据目录。`run_lab.py` 只接收已经 prepare 的目录，不会自行寻找或下载数据。

## Linux

Python 3.12 及 venv 功能先由实验室环境提供。在仓库根目录执行：

```bash
bash scripts/setup_lab.sh --python python3.12 --venv-dir '/data/Research Work/venv' --device cuda

'/data/Research Work/venv/bin/python' scripts/fetch_unsw.py \
  --output '/data/Research Work/data/raw' \
  --source mirror --accept-unverified-mirror

'/data/Research Work/venv/bin/python' -m ids prepare \
  --train-csv '/data/Research Work/data/raw/UNSW_NB15_training-set.csv' \
  --test-csv '/data/Research Work/data/raw/UNSW_NB15_testing-set.csv' \
  --output '/data/Research Work/data/prepared-v1' \
  --data-kind mirror

'/data/Research Work/venv/bin/python' scripts/run_lab.py \
  --data-dir '/data/Research Work/data/prepared-v1' \
  --work-dir '/data/Research Work/work' \
  --config configs/baselines.yaml \
  --device cuda
```

两个 setup 脚本均支持 `cpu`，分别使用 `-Device cpu` / `--device cpu`。这会安装官方 CPU wheel，仍可以运行整个开发流程，只是 MLP 用 CPU 训练。

## 启动器执行什么

`run_lab.py` 使用运行它的 `sys.executable`，以参数列表启动子进程，不做 shell 字符串拼接；带空格路径可用。它设置仓库 `src` 的 `PYTHONPATH`，即使没有 editable 安装，只要依赖已经安装，也能找到本仓库代码。

顺序固定为：

1. `doctor`：记录当前环境；选择 CUDA 时附带 `--require-cuda`，不可用则停止。
2. `suite`：仅训练集拟合、验证集选择/观察；不读取 calibration/test。
3. `report`：从已完成开发运行生成验证集报告。

启动器**不会运行 `freeze` 或 `evaluate`**。开发比较完成后，先依照 `PROTOCOL.md` 锁定研究选择，再单独进入校准与最终测试阶段。报告中的验证集成绩不能称为最终测试成绩。

在实际运行前加 `--dry-run` 可查看三个子进程的精确参数列表；此模式不创建输出、不启动训练。`configs/smoke.yaml` 仅用于检查软件路径，完整基线使用 `configs/baselines.yaml`，特征预算初筛使用 `configs/feature-pilot.yaml`；合成数据只验证工程流程。

## 固定种子、线程与恢复

启动器在任何 Python 子进程启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。模型模块也在导入时设置默认值，早于 runner 收集 CUDA 信息的步骤；MLP 训练开启确定性算法并固定 Python/NumPy/PyTorch 种子。硬件、驱动或依赖版本不同仍可能影响数值与耗时，跨机器一致不等于位级完全相同。

CPU 线程来自 suite YAML 的 `threads`。启动器把它传给 OpenMP/MKL/OpenBLAS 等运行环境，runner 将它用于实际训练配置。RF 的 `n_jobs` 应与 suite `threads` 一致；`num_threads` 由 runner 统一记录。改变线程数属于配置变更，不能把变更前后的耗时直接比较。

相同源码、数据、配置的运行可加 `--resume` 继续套件。已完成的 run 在复用前会校验工件、源码以及 train/validation 数据；损坏的已完成 run 或源码变化会被拒绝，不能关闭检查继续使用。

如果中断 run 尚无 `run.json`，恢复时会将原目录完整保留到 `suite/failed_attempts/<run>-时间戳`，然后按同一 seed 从头重训这个模型。**这是套件级恢复，不是 epoch/checkpoint 级续训**。不要手工把失败目录改为 complete，也不要覆盖原始失败证据；如果修改了源码或配置，则使用新的 `suite_name`。

MLP 模型包保存 CPU 权重，可在实验室 GPU 训练后复制回个人电脑 CPU 推理。`model.joblib` 基于 pickle，只加载自己生成或明确可信的产物。

## 到实验室后的验收

记录以下事实，而不是只记录“安装成功”：

1. `doctor` 中 GPU 完整名称、显存容量、驱动、Python/PyTorch/CUDA 版本、CPU 与 RAM；确认实际是预期 4090D。
2. 合成数据 smoke suite 的 CPU 与 CUDA 路径均成功，报告明确标为合成数据；这些分数不写入论文结果。
3. 真实数据 prepared manifest 的文件哈希、样本数量、重复组处理和四分区大小可核查。
4. 每次正式开发运行有 `status: complete`、配置、代码指纹、数据来源、预测文件、history 和模型包。
5. 挑一份模型包在另一解释器/个人电脑上加载，对相同输入比较分数；浮点误差用事先规定的容差判断。
6. 最后单独测固定 CPU 线程的单条延迟与批量吞吐，记录是否包含预处理；不要拿 4090D 批量推理和其他模型的 CPU 单条延迟比较。

首次部署失败时保留完整错误。CUDA 不可用先看 `doctor` 与驱动，再查 wheel；内存不足先缩小 batch、关闭其他 GPU 程序；`--resume` 拒绝源码变更时创建新 suite，不关闭完整性检查。当前小型 MLP 不需要混合精度、分布式训练或大型预训练模型即可开展主实验。
