# UNSW 零拟合机制诊断：实验室操作指南

本轮只验证冻结的 KD 失效机制，并在同一台机器上比较现有 H40 学生、HGB 教师与 CatBoost 参考模型的 CPU 成本。学生拟合、教师拟合、参数更新均为 **0**。条件式 H/K 两臂 mini-batch 训练不在这个入口内。

实现于2026-11-06，[本地验证记录](UNSW_DIAGNOSTIC_VERIFICATION_20261106.md)已完成；固定标签日期对应2026-11-05冻结规格，真实诊断待实验室运行。

## 1. 拉取固定版本

在实验室 PowerShell 中执行。复用 `D:\zhoumeng\Graduate\venv`，另建代码目录；不覆盖旧代码、数据或实验结果。

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\zhoumeng\Graduate'
git clone --branch lab-unsw-diagnostic-20261105 --single-branch https://github.com/zhoumeng-creater/tracelight-ids.git tracelight-unsw-diagnostic-20261105
if ($LASTEXITCODE -ne 0) { throw '拉取失败，停止。' }
Set-Location -LiteralPath 'D:\zhoumeng\Graduate\tracelight-unsw-diagnostic-20261105'
git rev-parse HEAD
```

若目标目录已存在，先检查其状态；不要删除旧目录或覆盖旧结果。运行入口要求工作区干净，并在结束时再次核对源码。

## 2. 先做零推理预检

```powershell
.\scripts\RUN_UNSW_DIAGNOSTIC.ps1 -PreflightOnly
```

预检核对固定绑定、既有模型和分区身份、依赖版本及原始 `train.parquet` 的物理 SHA。它加载学生工件核对权重和预处理，但不推理、不计算梯度、不计时，也不初始化 CUDA。RTX 4090D / CUDA0 在下一步正式运行时验收。

沿用四臂实验的锁定环境，另外核对已有 `catboost==1.2.10`。若明确提示缺少该依赖，使用同一实验室虚拟环境安装后再做预检：

```powershell
& 'D:\zhoumeng\Graduate\venv\Scripts\python.exe' -m pip install -r .\requirements-native-trees.txt
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败，停止。' }
```

默认从 `D:\zhoumeng\Graduate\experiments` 寻找完成标记 SHA 完全匹配的 screen 与四臂结果，并从 `data` / `experiments` 寻找与冻结 manifest 一致的 prepared 数据。四臂目录指**含 `UNSW_KD_COMPLETE.json` 的 `experiment` 目录**，screen 目录指**含 `TRANSFER_SCREEN_COMPLETE.json` 的 `budget-transfer-unsw` 目录**。

如果提示多个副本或未找到路径，明确传入已保存的原始目录；下面三个路径需要替换为实验室实际路径。预检和正式运行使用同一组参数。

```powershell
$inputs = @{
    ScreenDir = 'D:\zhoumeng\Graduate\experiments\原screen工作目录\budget-transfer-unsw'
    Round10Dir = 'D:\zhoumeng\Graduate\experiments\原四臂工作目录\experiment'
    UnswData = 'D:\zhoumeng\Graduate\data\prepared-v1'
}
.\scripts\RUN_UNSW_DIAGNOSTIC.ps1 @inputs -PreflightOnly
```

只接受仓库中的 [24 工件绑定](research_alignment_20261105/kd_failure_research/mechanism-audit-binding.json)。找不到或哈希不符时停止，不重新训练教师、warmup 或四臂模型，也不重新生成一份“内容相同”的 prepared 数据来替代原文件。

## 3. 正式运行一次

预检成功后，在实验室 RTX 4090D 机器运行：

```powershell
.\scripts\RUN_UNSW_DIAGNOSTIC.ps1
```

如果上一步指定了 `$inputs`，改为：

```powershell
.\scripts\RUN_UNSW_DIAGNOSTIC.ps1 @inputs
```

运行前停止其他会占用 CPU/GPU 的任务。每次调用新建工作目录，程序不覆盖旧结果、不续跑、不自动重试。CPU 成本按冻结的 4 线程方案测量，模型调用顺序由预定调度决定。

本轮预算如下；完整定义见 [冻结规格](research_alignment_20261105/kd_failure_research/最小机制验证规格_20261105.md)。

| 项目 | 固定预算 |
|---|---:|
| 新教师 / 学生拟合、优化器更新 | 0 / 0 / 0 |
| anchor、H40、K40 在 F 上的前向 | 各 85,916 行，共 257,748 行 |
| anchor 的 4 项损失梯度 | 84 批 × 4，共 336 次 backward；复用同一前向图 |
| HGB 在 F 上的新查询 | 0，使用封存缓存 |
| 三模型 CPU 正式计时 | 450 次调用，共 57,825 行 |
| CPU 预热 | 30 次调用，共 3,855 行 |
| 独立 RSS 进程探测 | 每模型 256 行，共 768 行 |
| 原 validation / calibration / test 的新打分 | 0 |

`unsw_diagnostic_complete` 表示输出齐全。机制结论、成本结论以及 `validity_passed` 需要分别检查；完成不等于验证通过。任何预算超限、状态变化、结果无效或计时异常都保留原结果，不重复运行来筛选更好数值。

## 4. 回传 ZIP 和 SHA

预检成功、正式完成或运行阶段报错后，程序自动打包并打印工作目录、ZIP 路径和 SHA256。将正式运行生成的 **`.zip` 与同名 `.zip.sha256`** 一起放入实验归档；如果预检或正式运行失败，也回传失败包和终端报错。

ZIP 包含请求、环境、源码身份、诊断输出或失败记录，以及逐文件 SHA 清单 `UNSW_DIAGNOSTIC_PACKAGE_MANIFEST.sha256`。封包完成后重新核对 ZIP 内容与工作目录；不把上次 ZIP 再装进新 ZIP。

输入发现阶段失败时尚未建立工作目录，回传终端报错即可。若进程被强制关闭或打包失败，保留打印的工作目录，仅补打包：

```powershell
.\scripts\RUN_UNSW_DIAGNOSTIC.ps1 -PackageOnly 'D:\zhoumeng\Graduate\experiments\终端打印的unsw-diagnostic工作目录'
```

`-PackageOnly` 不启动诊断，不检查 GPU，不进行推理或训练，也不覆盖之前的包。不要用再次执行正式入口来代替补打包。
