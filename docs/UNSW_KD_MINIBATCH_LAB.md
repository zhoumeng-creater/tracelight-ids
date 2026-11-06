# UNSW H/K mini-batch：实验室操作指南

本轮只执行已经冻结的 **H/K 两臂 mini-batch 最小受控验证**，检验更换优化方案后，普通 KD 是否获得预设的实际收益。两臂共同使用原 small anchor、完整字段、原预处理和 HGB 教师缓存；只改变全批训练为每遍 84 次更新的 mini-batch 方案。

依据为[最小机制验证规格第 4 节](research_alignment_20261105/kd_failure_research/最小机制验证规格_20261105.md)及[第十一轮零拟合诊断结果](UNSW_DIAGNOSTIC_RESULTS_20261106.md)，本轮执行定义见[冻结协议](UNSW_KD_MINIBATCH_PROTOCOL_20261106.md)和[JSON 配置](../configs/unsw-kd-minibatch-minimal-20261106.json)。本地仅验证实现，真实训练由实验室执行。本轮不增加 S/M、seed、教师、架构或超参数组合。

## 1. 拉取固定版本

在实验室 PowerShell 中执行。另建代码目录，复用原虚拟环境 `D:\zhoumeng\Graduate\venv`，保留旧代码、数据和实验结果。

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'D:\zhoumeng\Graduate'
git clone --branch lab-unsw-kd-minibatch-20261106 --single-branch https://github.com/zhoumeng-creater/tracelight-ids.git tracelight-unsw-kd-minibatch-20261106
if ($LASTEXITCODE -ne 0) { throw '拉取失败，停止。' }
Set-Location -LiteralPath 'D:\zhoumeng\Graduate\tracelight-unsw-kd-minibatch-20261106'
git rev-parse HEAD
```

目标目录已存在时先检查，不删除或覆盖。入口要求干净、已提交的代码，在结束时再次核对源码；不在旧实验目录执行 `git pull` 或修改旧模型的源码指纹。

## 2. 先做零推理预检

```powershell
.\scripts\RUN_UNSW_KD_MINIBATCH.ps1 -PreflightOnly
```

预检核对固定协议、输入工件、F/R/C 身份、原始 `train.parquet` 的物理 SHA、依赖版本和两臂预算，准备并核对共同排列。它不启动模型前向、梯度、参数更新或 CUDA 检测。预检成功状态为 `minibatch_preflight_complete`，`student_fits`、`teacher_fits`、`optimizer_updates` 均为 **0**。

默认在 `D:\zhoumeng\Graduate\experiments` 查找完成标记 SHA 完全匹配的 screen 和四臂结果，在 `data` / `experiments` 查找冻结 manifest 对应的 prepared 数据。需要的是：

| 参数 | 所指目录 |
|---|---|
| `ScreenDir` | 含 `TRANSFER_SCREEN_COMPLETE.json` 的原 `budget-transfer-unsw` 目录 |
| `Round10Dir` | 含 `UNSW_KD_COMPLETE.json` 的原四臂 `experiment` 目录 |
| `UnswData` | 含原 `manifest.json` 和 `train.parquet` 的 prepared 目录 |

无需另传第十一轮目录；已核验的诊断依据已冻结到本轮协议中。若发现多个匹配副本或没有找到目录，明确指定路径。下面三项需替换为实验室实际保存位置：

```powershell
$inputs = @{
    ScreenDir = 'D:\zhoumeng\Graduate\experiments\原screen工作目录\budget-transfer-unsw'
    Round10Dir = 'D:\zhoumeng\Graduate\experiments\原四臂工作目录\experiment'
    UnswData = 'D:\zhoumeng\Graduate\data\prepared-v1'
}
.\scripts\RUN_UNSW_KD_MINIBATCH.ps1 @inputs -PreflightOnly
```

预检和正式运行使用同一组输入。找不到工件、SHA 不符或依赖不符时停止并回传报错，不重新 prepare 数据、补训教师或重建 anchor 来绕过原绑定。

## 3. 正式运行一次

预检成功后，在原 RTX 4090D 实验机执行：

```powershell
.\scripts\RUN_UNSW_KD_MINIBATCH.ps1
```

若预检使用了 `$inputs`，正式运行也使用：

```powershell
.\scripts\RUN_UNSW_KD_MINIBATCH.ps1 @inputs
```

入口检查 CUDA0 和实验室 GPU，随后新建 `experiments\unsw-kd-minibatch-...` 工作目录。没有 CPU 训练回退、resume、自动重试或自动扩展实验。运行前停止其他占用 GPU 的任务。

| 项目 | 冻结预算 |
|---|---:|
| 训练臂 | H：hard BCE；K：原普通 KD，恰好两臂 |
| 教师拟合 / 新教师查询 | 0 / 0，读取原 HGB 的 F 缓存 q |
| 每臂训练 | 40 遍 F，batch=1024，每遍 84 次更新 |
| 每臂带梯度前向行 | 3,436,640 |
| 两臂带梯度前向行 | 6,873,280 |
| 每臂 / 总优化器更新、backward | 3,360 / 6,720 |
| R/C 新评分 | 仅两臂第 40 遍终点，共 73,642 行 |
| 中间 R/C 评分、checkpoint 搜索 | 0 |
| 新 CPU 成本计时、RSS 探针 | 0 |
| 原 validation / calibration / test 新评分 | 0 |

两臂从相同原 anchor 开始，分别重置 AdamW；lr=0.001、weight_decay=0.0001、betas=(0.9,0.999)、eps=1e-8、梯度范数裁剪 10，`foreach=False`、`fused=False`。原 float32 网络、float64 损失、T=1、lambda=0.3、字段、预处理及类权重均保持冻结。两臂共享预先生成并绑定 SHA 的 40 份 F 排列；每批按全 F 固定权重乘 `N/实际 batch 行数` 计算目标。

只保存中间 F 训练过程，第 40 遍为唯一终点。`history` 的每遍损失复用训练时逐 batch 的更新前前向；模型参数在这一遍内持续变化，因此它表示训练路径上的累计损失，不是该遍终点模型的完整 F 损失。程序不增加终点 F 前向来另测损失。

增加更新次数、批噪声、Adam 状态演化和累计 weight decay 同时发生，本轮检验的是整个优化方案与 KD 的交互。

## 4. 如何判定完成与通过

`unsw_kd_minibatch_complete` 表示本轮输出完成；预定科学门槛由结果单独报告。完整执行应记录 `student_fits=2`、`teacher_fits=0`、`optimizer_updates=6720`。预算、工件身份或有效性检查失败时保留原结果。

普通 KD 的最小收益门槛必须全部满足：

- C 上 Kmini−Hmini 的净 TP ≥ 34，标准化 pAUC 不降；R 上净 TP 和标准化 pAUC 不降。pAUC 的数值容差为 1e-12。
- 各臂 R/C 正常样本 FP 分别不超过 39/78，K 相对 H 的 FP 不增加。
- R/C 中每个有支持的攻击族，K 相对 H 的净 TP 均不下降。
- 删除 C 上净正收益最大的 `feature_hash` 组后，K 相对 H 的净 TP 仍大于 0。

“优化方案达到预设的 KD 实际交互增量”另需：

$$
I=(TP_{Kmini}-TP_{Hmini})-(TP_{Kfull}-TP_{Hfull})\ge34.
$$

原 full 差值为 5，因此交互门槛对应 mini 差值至少 39。34 是事前指定的实际收益标准，不是统计显著性。两层判断分别报告：两臂都改善而 KD 差值不足，只支持优化方案的一般作用；未通过时结束本次训练投入，不补 seed、遍数、T 或 lambda。通过也不能借用旧优化方案的 S/M 成绩来宣称新方案已经证明实例知识机制。

正式输出在工作目录的 `experiment` 中：

| 文件 | 内容 |
|---|---|
| `protocol.json`、`binding.json`、`qualification.json`、`preflight.json` | 冻结执行协议、原工件绑定、前轮诊断资格及本次预检记录 |
| `plan.json`、`source/` | 当前源码身份、环境、预算和源码副本 |
| `permutations.npz` | 两臂共用的 40 份排列；整体和逐遍 SHA 记录于协议、预检和每臂报告 |
| `targets.npz`、三份 `*-identity.parquet` | 冻结监督目标、权重、q 和 F/R/C 行身份 |
| `arms/H/`、`arms/K/` | 固定第 40 遍的 `model.joblib`、终点 `scores.npz`、`run.json` 和逐遍 `history.json` / `history.jsonl` |
| `historical-full.json`、`summary.json` | 原 full H/K 的缓存指标，以及新 mini H/K 指标和 `mini_minus_full` 差值，组成目标 × 优化方案的 2×2 对照 |
| `K-H_*.parquet` | R/C 改判事件、攻击族和 C 的 `feature_hash` 收益分组 |
| `REPORT.md`、`UNSW_KD_MINIBATCH_COMPLETE.json` | 可读验收报告、完成记录和输出 SHA 清单 |

`summary.json` 中，`validity_passed` 表示执行、预算和身份检查通过；`quality_acceptance_passed` 表示执行有效且上述全部质量门槛通过；`acceptance_passed` 进一步要求交互量至少 34。完成状态本身不替代这三个判断。

## 5. 打包回传

预检成功、正式完成或运行阶段报错后，程序自动生成 ZIP 和 SHA256，并打印工作目录与包路径。把正式运行的 **`.zip` 与同名 `.zip.sha256`** 一起放入实验归档；门槛未过、结果较差或执行失败，也回传已有结果，不重新运行挑选数值。失败时同时保留终端报错。

ZIP 包含请求、环境、源码身份、实验输出或失败记录，以及逐文件 SHA 清单 `UNSW_KD_MINIBATCH_PACKAGE_MANIFEST.sha256`。程序核对 ZIP 内文件及封包前后的工作目录，不嵌套之前生成的 ZIP。

若进程被强制关闭或打包失败，保留终端打印的工作目录，执行补打包。下面路径替换为实际路径：

```powershell
.\scripts\RUN_UNSW_KD_MINIBATCH.ps1 -PackageOnly 'D:\zhoumeng\Graduate\experiments\终端打印的unsw-kd-minibatch工作目录'
```

`-PackageOnly` 只生成一个新包，不训练、不推理、不检查 GPU，也不覆盖旧包。输入发现阶段报错时还未创建工作目录，回传终端报错即可。
