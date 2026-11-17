# R30 模型与训练设计实现：实验室运行说明

执行版本：`model-training-redesign-r30-v1`。入口为 `scripts/RUN_MODEL_TRAINING_REDESIGN.ps1`，注册为 `configs/model-training-redesign-20261117/registration.json`。它绑定已审定设计和原始工件清单；旧R29脚本不能代跑本轮。

本轮只有两种协议输入模式，各17/29/43三个seed，共六条F/V训练轨迹。原TabM＋PLE主干不变；A/B四组每批各64、每轨迹最多15600次更新。选中的F checkpoint直接交付，0次P重训。全部六份工件及配对合同冻结后，才在CPU单线程进行C和完整B/D评价。执行不下载数据或安装依赖。

## 1. 获取固定版本

在实验室已有Git仓库中拉取固定标签，并建立独立目录，保留旧实验输出：

```powershell
git fetch origin tag model-training-redesign-r30-v1
git worktree add '..\tracelight-model-training-redesign' model-training-redesign-r30-v1
Set-Location '..\tracelight-model-training-redesign'
```

若该目录已存在，进入原目录核对 `git describe --tags --exact-match`，不要用强制覆盖或删除实验输出处理路径冲突。

## 2. 环境与已有输入

沿用实验室Python：

```text
D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe
```

精确依赖版本沿用R29：torch `2.6.0+cu126`、tabm `0.0.3`、rtdl_num_embeddings `0.0.12`、numpy `1.26.4`、pandas `2.2.2`、scipy `1.14.0`、scikit-learn `1.5.1`、pyarrow `21.0.0`。正式训练核对实验室RTX4090D CUDA；合成检查使用CPU，不要求CUDA可用。线程默认4，单GPU按seed依次训练两臂，不并发抢占GPU。

需要两个明确路径：

| 参数 | 指向什么 | 必须保留的内容 |
|---|---|---|
| `-R29Run` | 原R29完整运行目录或回传解压目录，目录内直接有protocol.json | 注册绑定的17份文件：协议、freeze、F/V/P/C/E/D/ED输入、references.json、旧HGB∪规则/R26三MLP/R29三PLE判决 |
| `-RawDir` | 已有原始TQH档案所在目录 | `TQH-C2_labels_v110.zip`与`TQH-C2_labels_D.zip`；两份包内已有原始流及标签，按SHA查验，不联网下载 |

不需要重跑R26，也不需要原R26模型权重。`-R26Run`仅供额外核对旧协议和校准文件，默认省略。输入阶段从已有原始A/B构建监督，核对原B输入逐值一致；C/E/D与原R29保持相同身份、标签、数值及顺序。原UID映射、排除标签及时间信息单独留档。

## 3. 先检查，再运行

以下两条路径需填写为实验室实际保留位置。`$r29`应指向包含`protocol.json`的那一层，`$raw`应指向原始档案目录。

```powershell
$py = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe'
$r29 = 'D:\zhoumeng\Graduate\tracelight-model-training\local\model-training-r29'
$raw = 'D:\zhoumeng\Graduate\tracelight-ids\data\raw\response-evidence-20261112'
$out = Join-Path (Get-Location) 'local\model-training-redesign-r30'

.\scripts\RUN_MODEL_TRAINING_REDESIGN.ps1 -PlanOnly
.\scripts\RUN_MODEL_TRAINING_REDESIGN.ps1 -CheckOnly -Python $py
.\scripts\RUN_MODEL_TRAINING_REDESIGN.ps1 -InputsOnly -Python $py -R29Run $r29 -RawDir $raw -OutputDir $out
```

- `PlanOnly`只读规格，不需要Python、输入或GPU。
- `CheckOnly`要求显式Python，确认固定作者包存在后运行本轮合成测试；其中包含微型官方模型的CPU训练和冷加载检查，不触碰真实实验数据。
- `InputsOnly`只做原始字节、标签、身份、数据角色和旧判决核验，保存本轮输入与源代码快照；不拟合预处理、不构建模型、不生成新分数。
- 各`Only`模式互斥。预检与正式运行使用相同输出目录、输入路径、线程和设备，不能在中途换代码或改参数。

检查通过后，在实验室执行一次完整流程：

```powershell
.\scripts\RUN_MODEL_TRAINING_REDESIGN.ps1 -Python $py -R29Run $r29 -RawDir $raw -OutputDir $out -Threads 4 -Device cuda:0
```

完整流程自动核验并打包。无需另执行旧R29入口、P重训或手动选择checkpoint。

## 4. 固定顺序与核验输出

| 阶段 | 实际行为 | 查看位置 |
|---|---|---|
| 输入 | F19904、V21450、C5511、E_B5531、E_D11528；完整角色与旧判决绑定 | `inputs/input_audit.json`、`inputs/manifest.json`、`protocol.json` |
| 六轨迹 | seed17盲/感知、29盲/感知、43盲/感知；四独立循环队列，每139step验证，末次15600 | `models/<id>/trajectory/history.json`、`sampling.json`、`pairing.json`、worker日志 |
| 选轮 | 最大化A_TCP/B_UDP最差pAUC、全V pAUC、面板平均、较早step；NLL不选轮；patience16；全V严格分离锁最早 | `history.json`的selection及`summary.json` |
| 交付冻结 | best.pt原样复制为model.pt，预处理和PLE同时封存；核对两臂初始参数、拟合输入、分箱及共同更新前缀的采样/dropout随机流 | `freeze.json`、六份trajectory收据 |
| 评价 | 六工件全部冻结后，原C程序得到每模型唯一阈值；完整17059行CPU分数和判决；不拼接告警 | `evidence/<id>/`、`reports/*-decisions.parquet`、`units.csv` |
| 判断 | 三seed分别核验机制、误报、B72、旧合规强参照保护及D增益；旧损失集合逐身份追回/仍漏/新丢失 | `reports/comparison.json` |
| 成本 | 每模型独立进程、CPU单线程：冷加载、端到端、预处理、预变换输入的模型计时；GPU吞吐另列 | `cost/<id>.json`、资源日志 |

训练中的“139step周期”仅是记账单位，不代表每条F恰好使用一次。B正例只有90条，会被循环重复；日志保存实际draw、unique、cycle。每批四组均值只计一次，不再叠类别权重。

报告中的`status=complete`表示执行与归档完整，是否达到研究采用条件由`reports/comparison.json`的`adoption_pass`判定。该字段要求实现合同、三个seed机制条件、三个seed效果条件全部通过；单种子好看、仅V满分或仅D总TP增加均不能替代。

## 5. 中断、继续和回传

保留同一输出目录。再次使用原命令时，已完成且收据通过的阶段会跳过；非空但未完成的拟合拒绝自动重训。不能清空模型目录、换seed或另建输出目录绕过六次拟合预算。若某轨迹已开始后中断，回传失败证据，先核查执行故障。

下面两种操作只读取本轮归档工件，不要求R29目录或原始ZIP仍在原位置，也不会加载模型：

```powershell
.\scripts\RUN_MODEL_TRAINING_REDESIGN.ps1 -VerifyOnly -Python $py -OutputDir $out
.\scripts\RUN_MODEL_TRAINING_REDESIGN.ps1 -PackOnly -Python $py -OutputDir $out
```

成功时回传输出目录中的`model-training-redesign-r30-v1-results.zip`及同名`.zip.sha256`。运行失败时回传`model-training-redesign-r30-v1-failed-evidence.zip`、SHA文件和终端错误，保留完整原输出。输入路径或原始SHA预检失败可能尚未形成正式run，直接保留错误内容；不要重新训练旧轮次修补输入。

完整设计与原因见[模型训练重审](MODEL_TRAINING_REDESIGN_20261117.md)，实现核验记录见[实现与核验](MODEL_TRAINING_REDESIGN_IMPLEMENTATION_20261117.md)。本轮评价材料已用于开发，结果按开发比较报告。
