# AIT Harrison 前向确认实验室说明

版本 `ait-forward-main-suite-v1`。对同一份已准入的 Harrison 连接记录，一次比较 22 个历史冻结评分器。每个评分器只生成一份确认分数，以自己的历史 C 分数冻结两条阈值规则。本轮拟合次数为零。

**Harrison 的连接级输入已准入，可以启动实验室评分。** 确认包有 413 条流，其中 46 条正常、13 条控制、1 条下载、353 条未知。ownCloud 文件业务有 32 条正常流，Intranet 页面业务有 14 条，按 1% 整数预算，本批两组均须零误报。只对这两个观测分母报告经验误报结果。合格的 Intranet 正常流均采自控制活动前，当前没有同日同服务的合格正常流。 [AIT 作者档案](https://zenodo.org/records/19483937)

已在评分前登记 14 个作者标注控制请求和 2 次下载完成事件。13 个控制和 1 个下载的连接归属已闭合。另 1 个控制的整流证据不足，另 1 个下载在捕获中没有精确的 HTTP 请求匹配。仍逐项报告这两个事件，当前材料不足以判定 16 个事件全部命中。已准入下载由作者完成回执、精确 HTTP 请求和双向传输组成，Zeek 未独立记录该请求的 2xx 与响应体。Harrison 尚未评分。

## 固定比较范围

| 评分器 | 数量 | 固定比较 |
| --- | ---: | --- |
| HGB | 4 | 历史窄来源／混合来源与保留／置零有限端口的 2×2 工件。主评分器为已经登记的 `HGB_narrow_ports_zero`。 |
| TabM+PLE | 18 | A（BCE）、H（参照攻击间隔）、D（正常参照对齐）、G（业务尾部攻击间隔）、R（D+H）、S（D+G）。每臂用种子 17、29、43，学习率 0.002、辅助权重 1、15,600 步的原主检查点。 |

预登记的 31 组跨评分器配对分别报告两条规则的结果。HGB 的来源对照改变的是整套训练来源及样本构成。TabM 对 BCE 的比较按同一种子配对，R 对 H、S 对 G 检验加入对齐项后的变化，R 对 S 比较攻击间隔的参照边界。混合去端口 HGB 与 TabM BCE 作为配方比较，二者仍有类型化预处理、学习算法与采样差异。18 个 TabM 检查点使用同一 Harrison 运行，三个种子说明模型结果的重复性，不能当作三个独立采集。其余预算、学习率和权重使用 11 月 22 日已有曲线解释，主次关系在评分前固定。 [原主配置与训练曲线](comprehensive_batch_20261122/index.md) · [比较清单](../scripts/ait_forward/suite.py)

每个评分器用 russellmitchell 的同一批 3,475 条 C 正常连接，沿用各自已归档的分数与原始严格判决。

- **池化规则**使用 C 分数由高到低第 `floor(n/100)+1` 项。
- **按业务共同阈值**对 C 三个互斥业务组各取同一顺位，再取其中最大值。推断时仍用一个共同标量，不需要在线业务分组。
- 两条规则均为 `score > threshold`。同分不拆开，小于 100 条的业务组允许的误报数为零。每个评分器的共同阈值至少等于自己的池化阈值，因此规则差异只会移除告警。跨评分器配对则可能新增或减少告警。

## 准入和评分前登记

按原有 AIT 观测语义出具完整流 raw8、稳定身份、流结束时间及已核实的离线导出时间。每个已知正常或攻击标签都需有定位到原件的证据。正常组覆盖全部已核实正常连接，未知标签保留为 `null`。控制、下载分别审核，无法建立证据的终点登记为 `unavailable`，并记录原因。确认包必须与历史 F、C、Fox 及外源开发身份隔离。该时钟支持完整流导出后的离线判决，结论不延伸到首包检测或实际入库耗时。

审核完成后，将三份已审数据表、证据文件及其 SHA-256 绑定进 `ait-harrison-reviewed-inputs-v1` 清单。用 `prepare` 输出 `qualification.json`、`rows.jsonl.gz`、`normal_units.jsonl.gz` 与 `events.jsonl.gz`。此步骤不调用模型。若只核实了候选证据，可用 `--diagnostic` 输出覆盖诊断。诊断收据不能进入确认评分。 [装配入口](../scripts/ait_forward/prepare.py) · [档案候选证据](../scripts/ait_forward/harrison_evidence.py)

数据整理从作者归档开始。先用 `harrison_evidence` 生成源材料清单和控制／正常动作锚点，用 `harrison_download_audit` 补下载完成证据。用 `capture` 取得目标服务的原始包并校验成员长度、CRC 和 SHA-256，再用仓库固定的 Zeek 8.0.11 `pcap_extract.py` 导出两处观测。运行 `harrison_project`、`harrison_review` 和 `harrison_download_join` 保留候选及逐项排除依据。用 `assemble` 绑定审核结果和源事件协议，生成本次确认包。Apache 时间秒和端点能形成候选；共享连接的业务或下载对象需要独立归属证据。 [原始捕获入口](../scripts/ait_forward/capture.py) · [Zeek 导出合同](../scripts/fit_contract/pcap_extract.py) · [候选核验](../scripts/ait_forward/harrison_review.py)

审核清单沿用资格收据的 `source_group`、`clock_basis`、`confirmation_window`、`targets`、`normal_evidence`、`clock_evidence` 字段。另用 `inputs` 分别登记 `rows`、`normal_units`、`events`，并用 `evidence_files` 列出原始档案、观测导出及复核收据的清单。每项都含相对审核清单的 `path` 和文件的 `sha256`。 [装配清单可执行示例](../tests/test_ait_forward_prepare.py)

本轮的连接级最低检出预设为每个已准入终点至少 1 条。这个下限回答该类能否在正常预算下被观测到。联合验收另逐一检查源日志登记的成功控制请求和下载完成事件。事件由独立证据给出 ID，未能闭合的必检事件保留未确定。部署所需检出率当前没有业务依据，本轮使用实际检出条数、事件上下界和全部损失报告效果。

```json
{
  "schema": "ait-harrison-event-protocol-v1",
  "required_events": [
    {"event_id": "<已审核的控制事件 ID>", "kind": "control"},
    {"event_id": "<已审核的下载事件 ID>", "kind": "download"}
  ]
}
```

事件候选闭合且所有可能对应连接均告警时，`one_of` 事件才算确定命中。若候选全部已经核实属于该事件，至少一个候选告警即可确定命中。只有可能命中、缺少必检事件、候选未闭合或目标材料不可用时，对应事件保留未确定。闭合候选均不告警时记为漏检。连接数达到下限不能抵消必检事件的漏检或未确定。

## 实验室运行

用原实验室 Python 环境和 `comprehensive-20261122-results.zip`。保留原 scikit-learn 1.5.1、TabM 0.0.3 与 rtdl_num_embeddings 0.0.12。HGB 在 CPU 上评分，TabM 使用与原综合批次相同的 CUDA 设备类型、输入精度及固定评分块。先解压已生成的 `ait-harrison-qualified-inputs-20261124.zip`，目录中已有四份资格输入和 `event-protocol.json`。CUDA、依赖和设备信息随登记保存。

```powershell
git fetch origin tag ait-forward-main-suite-v1
git -c core.longpaths=true worktree add --detach '..\tracelight-ait-forward-suite-v1' ait-forward-main-suite-v1
Set-Location '..\tracelight-ait-forward-suite-v1'

$ResearchPython = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe'
$FrozenArchive = 'D:\Research Work\results\comprehensive-20261122-results.zip'
$QualifiedDirectory = 'D:\Research Work\ait-forward\ait-harrison-qualified-inputs-20261124'
$Qualification = Join-Path $QualifiedDirectory 'qualification.json'
$EventProtocol = Join-Path $QualifiedDirectory 'event-protocol.json'
$ForwardOutput = 'D:\Research Work\results\ait-forward-main-suite-v1'

# 已有合格收据时直接使用。需要从审核清单装配时，先执行这条。
# & $ResearchPython -m scripts.ait_forward.prepare --manifest 'D:\Research Work\ait-forward\reviewed-inputs.json' --output $QualifiedDirectory

$QualifiedTargets = (Get-Content -Raw -LiteralPath $Qualification | ConvertFrom-Json).targets
$TargetArgs = @()
if ($QualifiedTargets.control.status -eq 'qualified') { $TargetArgs += @('--control', '1') }
if ($QualifiedTargets.download.status -eq 'qualified') { $TargetArgs += @('--download', '1') }
& $ResearchPython -m scripts.ait_forward.suite --archive $FrozenArchive --qualification $Qualification --event-protocol $EventProtocol --output $ForwardOutput --tabm-device cuda @TargetArgs
```

入口先核对归档模型、C 分数与原阈值，注册 22 组模型、两条规则、同一资格收据、计数下限与事件协议。随后回归验证已开放的 Fox 数据，记录数值差及严格阈值下的判决变化，才读取 Harrison 确认表。四个 HGB 必须复现已归档的 Fox 分数。所有评分器必须复现已开放 Fox 的严格阈值判决。Fox 回归只检查评分路径，不能替代 Harrison 确认。合格输入和相同登记下，续跑使用同一命令与输出目录，已通过登记和分数哈希核验的确认评分会复用。不能以切换设备、换用检查点或改门槛继续同一次确认。

## 结果判定与回传

- 优先查看 `summary.json` 中固定主 HGB 的连接级状态、联合状态和两规则比较。逐业务列出分母、实际误报、整数预算与相对池化修复的条目。控制、下载分别保留每条新增与丢失的连接和事件上下界。
- 按组规则若修复实际业务超额、同时达到各目标连接下限与所需事件的确定命中，才支持该规则在本次确认中的收益。两条规则告警相同，记录此次等效；误报下降伴随检测损失时，保留得失和联合验收结果。有目标未准入时，只报告证据覆盖的范围。
- 固定分数的排序上限给出满足全部观测正常业务预算时最多能检出的连接数，并核对必检事件在该边界的保守上下界。它帮助区分工作点失配、已有排序不足和事件证据不足，报告保留可达计数、事件状态及上下界。确认边界不作为候选阈值或部署规则。
- 补充配对用于解释来源、端口、模型配方和训练目标的差异，主评分器仍按预登记报告。若工作点未达标且固定排序仍有可证实的余量，将问题带回独立开发；排序不足时结束该工件的校准修补。Harrison 只支持同一攻击生成体系的跨运行、完整流离线结果。
- 本次计时包含模型加载、预处理和一次评分调用。同包保留经校验的历史 CPU 基准、22 份拟合摘要和 18 份主检查点训练诊断。跨设备的批量评分耗时不作为持续服务延迟或单条 p95。

只回传本轮结果 ZIP、旁置 `.zip.sha256.json` 和必要新增说明。包内保留冻结登记、资格和身份绑定、Fox 回归记录、各成功评分器的模型与逐行分数、评价成员、配对结果、计时及本轮错误记录。出现失败时保留缺失状态及已完成结果。原 11 月 22 日归档继续作为输入。 [统一回传规范](EXPERIMENT_RETURN_POLICY.md)
