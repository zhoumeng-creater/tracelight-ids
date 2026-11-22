# 全面开发比较：一次启动与续跑

当前固定版本为 `comprehensive-calibration-v1.1`，修订独立进程调度、GPU 输入驻留与资源日志；72 条科学轨迹和全部评价规则沿用 v1。它接替尚未运行的九次 v2 批次。不要先运行 v2，再运行本批。实验矩阵和固定出口见[完整设计](comprehensive_batch_20261122/index.md)。

72 条 TabM 训练轨迹各保留四个预算点，加原四个 HGB；最多 292 个评价工件。没有固定墙钟上限，具体计算量与实际时间均完整记录。此处一次启动指一个登记好的批次，不承诺不中断，也不在失败时改配方重试。

## 实验室执行

沿用现有 RTX4090D 和项目锁定 Python 环境。在现有仓库目录执行：

```powershell
git fetch origin tag comprehensive-calibration-v1.1
git -c core.longpaths=true worktree add --detach '..\tracelight-comprehensive-v1.1' comprehensive-calibration-v1.1
Set-Location '..\tracelight-comprehensive-v1.1'
.\scripts\RUN_COMPREHENSIVE_BATCH.ps1 -Output 'D:\Research Work\results\comprehensive-calibration-v1.1'
```

默认解释器为 `D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe`；位置不同则加 `-Python`。脚本不安装或升级环境。启动时先用生成数组测量 1／2／4 个 GPU 工作进程的总吞吐，并根据显存、内存和 CPU 余量确定并发；此过程不读取真实 F/C/V，不选择模型参数。结果缓存在 `execution_profile.json`，同版本同机续跑时复用。

若原 `coverage-baseline-batch-v1` 四份 HGB 已完成且保留完整输出，可以在同一命令加 `-CoverageResults '原四臂输出目录'`。程序核对原注册及各模型哈希后只复用模型，不重新训练它们，也不读旧阈值来改变本批。没有该目录时，本批完成四次原配方拟合。

只检查材料和代码可加 `-InputsOnly` 并指定另一个空目录。该模式只解析 F，不构建模型，不评分。

## 已启动旧 v1 时接续现有输出

不要在正在运行的旧 checkout 中 pull、覆盖代码或切标签。先在旧终端按一次 Ctrl+C，等待旧进程退出；保留原输出。然后按上面的命令另建 v1.1 checkout，在新 checkout 执行：

```powershell
.\scripts\RUN_COMPREHENSIVE_BATCH.ps1 -Output 'D:\Research Work\results\comprehensive-calibration-v1' -Resume -AdoptV1Output
```

这里 `-Output` 必须是原来的实际输出目录；原命令若有 `-Python`、`-CoverageResults`，照原值保留。新入口核对旧 v1 注册、相同任务矩阵及已有模型哈希，保留 `source_receipt_v1.json` 与 `execution_adoption.json`，完成的轨迹不重训，中断轨迹继续原优化器、随机和抽样状态。原输出仍有运行进程时会拒绝接续。后续正常续跑只需 `-Resume`。

旧 v1 标签不修改。没有旧输出时直接使用上面的新运行命令，不需要先执行 v1。

## 中断后继续原批次

```powershell
.\scripts\RUN_COMPREHENSIVE_BATCH.ps1 -Output 'D:\Research Work\results\comprehensive-calibration-v1.1' -Resume
```

使用同一固定版本、环境、输出路径和 HGB 复用参数。已完成工件校验后跳过；中断轨迹恢复优化器及随机、抽样状态，最多重做最后未提交的步骤。不要换空目录重新跑整批，也不要用修改超参的方式“续跑”。

数值失败的任务保留失败状态，其他独立任务继续。资源／环境中断先解决提示的问题，再执行上面的续跑命令。新版按一次 Ctrl+C 后，协调器请求工作进程在下一个已保存恢复点停止；请等待退出，不同时开启第二份运行。所有训练任务终结并冻结后才开始校准；此后 `-Resume` 只恢复评分、报告、成本测量和打包。

## 回传与查看

成功完成后返回以下两个文件；有失败轨迹时完成报告也会明确标记：

```text
D:\Research Work\results\comprehensive-calibration-v1.1.zip
D:\Research Work\results\comprehensive-calibration-v1.1.zip.sha256.json
```

只传本轮包，不再另做 `TraceLight-ALL`。包中保留全部规定的推理模型、预算点分数、身份、配置、训练记录、错误、配对判决与成本。初始诊断检查点也保留，但不计入 288 份评价快照。优化器恢复状态留在实验室输出目录，不作为分析回传的重复载荷；实验室仍能精确续训。遵守[统一回传规范](EXPERIMENT_RETURN_POLICY.md)。

先查看 `execution_profile.json` 中选出的并发与吞吐、`resources-*.csv` 中的持续利用率，以及 `worker_logs` 中各任务日志。并发训练耗时包含资源竞争；最后的 CPU 部署计时仍隔离执行。

再查看 `summary.json`、`evaluation/comparison.md` 和 `job_statuses.json`，再查看各训练目录中的 `history.json`、`cost.json`、`snapshots/*/F_diagnostics.json`、分数与曝光数组；`job_invocations` 记录各次执行的墙钟时间与中断状态。固定矩阵为 `plan.json`，代码与输入依据为 `source_receipt.json`，整体冻结阶段见 `models_frozen.json`、`thresholds_frozen.json`。

当前主预算是 15,600，主学习率是 0.002，主权重是 1；其余已登记配置和预算点完整报告，不根据开发结果选择最好的轮次、种子或阈值。

资源证据和执行边界见[资源执行说明](comprehensive_batch_20261122/resource_execution.md)。
