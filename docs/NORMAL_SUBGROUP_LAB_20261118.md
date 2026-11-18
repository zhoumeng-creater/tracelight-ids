# R31 正常子群训练：实验室运行说明

执行版本为`normal-subgroup-r31-v1`，使用独立入口`RUN_NORMAL_SUBGROUP.ps1`。原R30工件是本轮起点，不重跑R30、不重拟合预处理或PLE、不做P重训。

本轮只比较固定分位目标与保持原分数目标，各使用17、29、43三个种子，共六条轨迹。每条恰好2224步、16次CPU验证；全部六个工件冻结后才使用C和完整B/D。模型、目标、权重、种子、训练步数和退出条件已固定。

## 为什么下一阶段需要实验室

本机完成了真实原权重的CPU双冷载、F/V逐位重放、同输入参照一致性、step0完整选择合同，以及微型模型和执行流程核验。正式六轨迹与部署成本沿用此前实验室RTX4090D环境，便于保持硬件和运行口径一致。4050并非理论上不能训练；本轮不另开一套本机真实试训。

实验需要回答两个尚未验证的问题：网络能否在保留F/V排序及来源内尾部保护的条件下学到改善；这一改善能否保留B72、D1964全部TP身份，并使原15单元全部满足误报预算。本机目标数组的改善不能代替这两个结果。

## 取得固定版本

在实验室已有Git仓库执行：

```powershell
git fetch origin tag normal-subgroup-r31-v1
git worktree add '..\tracelight-normal-subgroup-r31' normal-subgroup-r31-v1
Set-Location '..\tracelight-normal-subgroup-r31'
git describe --tags --exact-match
```

使用新目录，保留原实验输出。若目录已存在，进入后核对标签，不强制覆盖。

## 环境和输入

沿用已有Python环境，不安装或升级依赖：

```powershell
$py = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe'
$r30 = 'D:\zhoumeng\Graduate\tracelight-model-training-redesign\local\model-training-redesign-r30'
$out = Join-Path (Get-Location) 'local\normal-subgroup-r31'
```

`$r30`应指向实际保留的R30完整输出，目录内直接含`protocol.json`、`freeze.json`、`inputs`、`models`和`reports`。若实验室移动过目录，只修改这个路径。需要原三份blind权重、F/V/C/E_B/E_D、原三份blind判决及历史参照判决；按固定SHA核验。不需要重新下载原始数据。

版本沿用torch2.6.0+cu126、tabm0.0.3、rtdl_num_embeddings0.0.12、numpy1.26.4、pandas2.2.2、scipy1.14.0、scikit-learn1.5.1、pyarrow21.0.0。正式训练设备核对`NVIDIA GeForce RTX 4090 D`，每次仅一个GPU训练任务。CPU参照、选择和最终评分统一单线程、内部批256。

## 检查与执行

```powershell
.\scripts\RUN_NORMAL_SUBGROUP.ps1 -PlanOnly
.\scripts\RUN_NORMAL_SUBGROUP.ps1 -CheckOnly -Python $py
.\scripts\RUN_NORMAL_SUBGROUP.ps1 -PrepareOnly -Python $py -R30Archive $r30 -OutputDir $out
```

`CheckOnly`使用合成输入和新建微型作者模型；CUDA可用时也验证GPU训练与CPU选择的接线，不读取真实训练数据。`PrepareOnly`生成本机正式CPU参照、目标与选择基线，执行0次更新，不读取C/E。它检查完整F分离、V的排序/面板/并列/来源内尾部合同，并保存源码与哈希收据。

实验室必须在自己的CPU重新生成参照，不能将开发机分数直接当作实验室参照。预检和训练使用相同输出目录及代码；跨机器移动未完成的运行目录会被拒绝。

预检通过后执行一次：

```powershell
.\scripts\RUN_NORMAL_SUBGROUP.ps1 -Run -Python $py -R30Archive $r30 -OutputDir $out -Threads 1 -Device cuda:0
```

`Run`复用已完成且收据一致的预检，按seed17目标/保持、29目标/保持、43目标/保持顺序执行六轨迹。每批A0/A1/B0/B1各64；训练态成员BCE与同批eval态辅助损失分别反传，合并梯度后裁剪并更新一次。CPU选择使用独立模型镜像，不迁移训练模型或改变dropout随机流。

## 查看进度和结果

| 文件 | 含义 |
|---|---|
| `preflight_readiness.json` | 正式参照与step0保护通过，源码/身份/主机绑定 |
| `logs/worker-<id>.log` | 当前轨迹的执行日志 |
| `models/<id>/trajectory/history.json` | 每139步的损失、梯度、采样和全部选择条件 |
| `models/<id>/trajectory/summary.json` | 实际2224更新、16事件、选中步及是否回退 |
| `freeze.json` | 六工件及全部16事件的配对合同通过；之后才允许C/E |
| `evaluation/<id>/` | 唯一C校准、完整E分数及逐身份判决 |
| `evaluation/comparison.json` | 三种子机制、15单元、TP身份及所有历史参照比较 |
| `cost/<id>.json` | 独立CPU进程的端到端/预处理/模型成本，GPU吞吐另列 |
| `outcome.json` | 研究采用条件与部署判决一致性合并后的结论 |
| `status.json` | 执行是否完整；`complete`不等于采用成功 |

没有通过保护且严格改善H的checkpoint时，交付原工件字节并标记step0回退，仍完成固定预算；不把回退当作候选成功。不因早期改善提前停，也不因结果不佳延长训练。

最终要求三个seed分别通过机制条件、B72/D1964及原TP身份保持、全部15单元的`floor(normal_n/100)`预算，以及相对同预算控制的预设增量。双方超额误报均为0时不宣称新增目标有必要；合格的保持目标结果只按已开开发结果报告。

## 失败和回传

已完成阶段按收据复用；失败或非空未完成的拟合不会自动重试。若已开始拟合，不得通过换目录、改seed、追加轮次或调阈值绕过预算。预检/环境错误与算法负结果分开记录。

```powershell
.\scripts\RUN_NORMAL_SUBGROUP.ps1 -VerifyOnly -Python $py -OutputDir $out
.\scripts\RUN_NORMAL_SUBGROUP.ps1 -PackOnly -Python $py -OutputDir $out
```

这两步只验证和打包现有文件，不加载模型。完整运行自动生成`normal-subgroup-r31-v1-results.zip`及`.zip.sha256`；失败证据使用`normal-subgroup-r31-v1-evidence.zip`。回传ZIP、SHA及终端错误，原输出保留。环境或输入在正式目录建立前失败时，保留原错误，不重跑旧实验。

修改依据与本机核验见[实施记录](NORMAL_SUBGROUP_IMPLEMENTATION_20261118.md)，完整目标与v2修订见[设计完善记录](NORMAL_SUBGROUP_REFINEMENT_20261118.md)。
