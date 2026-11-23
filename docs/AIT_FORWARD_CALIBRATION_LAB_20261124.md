# AIT 前向校准比较实验室说明

固定版本 `ait-hgb-forward-calibration-v1`。使用已完成的 `HGB_narrow_ports_zero` 和原 russellmitchell 正常校准分数。两条规则共用评分器、校准来源和确认组分数。本批次为零次拟合、一次 CPU 确认评分，实验室无需安排 GPU 训练。

**当前可运行 Fox 回归与 Harrison 档案核验。Harrison 仍为未准入。** 官方归档核验已确认采集清单为 2022 年 2 月 4–9 日，清单未标时区。两台目标服务各有 5 个非空 Apache 轮转访问日志，包内还有 9 个非空正常角色活动日志。这里只核实了目录和日期清单，逐条动作、正常分母和网络观测时钟需要数据装配。 [AIT 作者数据与归档](https://zenodo.org/records/19483937) · [保留组资格规则](unified_events_20261119/README.md)

## 1. 环境和开放数据回归

沿用项目环境，冻结 HGB 依赖 scikit-learn 1.5.1。使用本批次已有的结果 ZIP，入口逐项核对所用归档成员的已知 SHA-256，再加载模型。运行位置是新工作树根目录。

```powershell
git fetch origin tag ait-hgb-forward-calibration-v1
git -c core.longpaths=true worktree add --detach '..\tracelight-ait-forward-v1' ait-hgb-forward-calibration-v1
Set-Location '..\tracelight-ait-forward-v1'

$ResearchPython = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe'
$FrozenArchive = 'D:\Research Work\results\comprehensive-20261122-results.zip'
& $ResearchPython -m scripts.ait_forward.run verify --archive $FrozenArchive
```

`verify` 只评分已经开放的 fox 661 条，含 213 条未知记录。逐值核对原归档分数，并输出 C 规则和两套 fox 判决。预期两条规则均检出 14/16 条控制和 1/1 条下载，431 条正常流零误报。原池化阈值为 −8.192877421519793，按组共同阈值为 1.3794255278890524。原 C 三组预算为 9／1／23，对应误报由 7／2／14 变为 0／1／7。数值未复现时先检查归档、工作树和 Python 依赖。

需要复核作者档案时，可单独运行以下命令。脚本只读取目录及短日期清单，以 `not_admitted` 记录资格状态。输出路径使用工作树外尚未存在的文件。

```powershell
& $ResearchPython -m scripts.ait_forward.qualification --time-metadata --output 'D:\Research Work\ait-forward\harrison-metadata.json'
```

## 2. 确认数据装配

按已预留的整运行范围核实成功的控制、下载动作，与关联连接及时间一并保留。用正常活动、匹配服务请求和网络记录建立正常业务分母。用已有 AIT 提取语义产生完整流 raw8，记录依据以及可用时钟。日期清单的本地时间需有来源证据才能换算 UTC。无法核实的标签用 `null`，无法核实的终点保留 `unavailable`，并写出原因。装配阶段只判断资格，不用确认输入或结果选模型、字段或规则。

确认包置于工作树外。资格收据和三个 gzip JSON Lines 文件放在同一目录。SHA-256 按压缩文件原字节计算，小写十六进制。资格收据与三文件同时保留。格式如下，尖括号内由合格材料及实际审核记录填写。

```json
{
  "schema": "ait-harrison-qualification-v1",
  "status": "qualified",
  "source_group": "ait:harrison:complete_run",
  "clock_basis": "offline_Zeek_export_network_time_not_detector_receipt",
  "confirmation_window": {"start_utc": "<ISO 8601 ending in Z>", "end_utc": "<ISO 8601 ending in Z>"},
  "normal_evidence": "<真实正常业务与分母依据>",
  "clock_evidence": "<时间、流观测结束及导出时钟依据>",
  "targets": {
    "control": {"status": "qualified", "evidence": "<控制动作与连接证据>"},
    "download": {"status": "qualified", "evidence": "<下载动作与连接证据>"}
  },
  "files": {
    "rows.jsonl.gz": "<sha256>",
    "normal_units.jsonl.gz": "<sha256>",
    "events.jsonl.gz": "<sha256>"
  }
}
```

| 文件 | 每行所需数据 |
|---|---|
| `rows.jsonl.gz` | `flow_id`、`stable_flow_id`、`source_run="harrison"`、`allocation_group="ait:harrison:complete_run"`、`dataset_role="confirmation"`、相同 `clock_basis`，以及 `observed_end` 和 `replay_export_time` 的 Unix 秒。观测结束在已核实时间窗内，导出不早于结束且不早于校准。可选 `actual_available_at` 只作追溯。 |
| 同一行的标签和字段 | `label` 为 0、1 或 null，已知标签填 `label_evidence`。阳性 `target_kind` 为 control 或 download。`raw8` 使用八个命名键 `ip_protocol, orig_port, resp_port, duration_seconds, orig_pkts, resp_pkts, orig_ip_bytes, resp_ip_bytes`，缺失为 null。入口从命名字段取值，复用原模型的有限端口置零处理。 |
| `normal_units.jsonl.gz` | `unit_id`、`flow_ids`。覆盖所有已核实正常流，每组有可计算分母。未知与攻击不进入分母。 |
| `events.jsonl.gz` | `event_id`、`kind`、`candidate_flow_ids`、`association_mode`（all_confirmed 或 one_of）、`candidate_set_complete` 和 `label_evidence`。每个准入终点有已确证事件与本类已知阳性的关联。未知流可以保留在多义候选里。 |

## 3. 冻结并评分

确认组准入后，在评分前确定各可评价终点的**最低检出条数**。命令从已决定的绝对整数读取；若某终点为 `unavailable`，省略对应的 `--control` 或 `--download`，结果只报告声明的部分范围。接受条件包含各登记正常单元的 1% 整数预算。

```powershell
$Qualification = 'D:\Research Work\ait-forward\qualified\qualification.json'
$ForwardOutput = 'D:\Research Work\results\ait-hgb-forward-calibration-v1'
# $ControlFloor 与 $DownloadFloor 为确认评分前已登记的整数，不使用示例值。
& $ResearchPython -m scripts.ait_forward.run freeze --archive $FrozenArchive --qualification $Qualification --output $ForwardOutput --control $ControlFloor --download $DownloadFloor
& $ResearchPython -m scripts.ait_forward.run run --archive $FrozenArchive --qualification $Qualification --output $ForwardOutput
```

- `freeze` 只打开原归档的固定成员及资格收据，生成 `registration.json`。池化规则沿用原 C 经验顺位。按组规则取各 C 正常组第 `floor(n/100)+1` 高分的最大值。严格 `score > threshold` 判告警，同分不告警。两条规则使用共同标量，业务组只用于校准和评价。
- `run` 先核对登记、文件哈希、重复身份、角色、观测时钟、正常分组和事件映射，再评分。两条规则比较同一份分数。中断后以原路径和参数重跑 `run`，原子写入且校验通过的评分记录会直接复用。登记过的最低检出、规则与输入身份用于同一次确认。
- 结果保存原始分数、逐条判决、事件上下界、各业务 FP 和预算、控制／下载检出及得失。排序上限只输出可达检出计数。两条阈值都已固定，报告不会用确认标签挑选新阈值。耗时记录是一次端到端评分调用的时间，包括模型调用与输入准备，不能当成持续服务的 p95。
- 完成结果按规则单独给状态。达标记录此次同一攻击生成体系的跨运行结果。阈值未达标但固定排序仍有余量时，将问题带回开发组。固定排序不足时结束该工件的校准修补。资格未成立的端点没有正式检出结论。确认组继续只用于本次报告。

## 4. 回传

只回传本轮同名结果 ZIP、`.zip.sha256.json` 和必要的新增说明。ZIP 中有冻结登记、模型、评分、逐条判决、评价成员、结果及包内清单。运行中断时同目录保留错误和已生成的本轮工件。早期归档只作为本地输入。 [统一回传规范](EXPERIMENT_RETURN_POLICY.md)

本轮实现只验证评分与校准。只有新的独立开发证据足以确定监督覆盖或新输入的具体对照时，才另行登记重新训练，继续沿用原有 HGB 和 TabM 对照配方。
