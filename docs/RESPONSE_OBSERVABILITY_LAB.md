# 第二十三轮实验室运行指南

本轮检查流日志送达时能否辨认包计数的缺口，再检验这种区分能否在每个评价采集正常误报不超过 1% 的条件下，多检扫描、少漏控制和下载。固定标签为 `response-observability-lab-v1`。原始抓包已经核验，真实检测模型由实验室运行。[抓包与日志逐行核对](response-evidence/OBSERVABILITY.md) · [60 个单元的数据与标签](response-evidence/数据.md) · [训练和正常校准集的回应证据](response-evidence/training-observability.json)

## 在 Windows 实验室运行

在实验室 PowerShell 中执行整段。使用原 route-core CUDA 环境，默认按物理核数分配 CPU。独立目录可用于断点恢复。

~~~powershell
$labRoot = 'D:\zhoumeng\Graduate'
$repo = Join-Path $labRoot 'tracelight-response-r23'
$release = 'response-observability-lab-v1'
if (-not (Test-Path -LiteralPath $repo)) {
    git clone --single-branch --branch $release https://github.com/zhoumeng-creater/tracelight-ids.git $repo
    if ($LASTEXITCODE) { throw 'Git clone failed' }
}
Set-Location -LiteralPath $repo
$actual = git rev-parse HEAD
$expected = git rev-list -n 1 $release
if ($LASTEXITCODE -or $actual -ne $expected) { throw 'Checkout differs from the frozen release' }
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\RUN_RESPONSE_OBSERVABILITY.ps1
~~~

入口会核对先前 checkout 中的输入，下载缺少的登记文件，校验 SHA-256，完成构建、拟合、逐流评分、评价和成本核验。脚本默认的 Python 为 D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe。若实际位置不同，在最后一行加 `-Python '完整路径\python.exe'`。数据和工作输出宜放在本地 SSD。只运行合成核查可用 `-CheckOnly`。

离线运行可在能联网的同版本 checkout 中先备好输入，以下两条在仓库根目录执行。把输出目录拷到实验室，运行正式脚本时加 `-SourceDir '离线输入目录'`。离线入口核对投影后的文件哈希，不再要求复制完整 PCAP 或多日志 ZIP。

~~~powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m ids.response_evidence.sources --output .\data\raw\response-lab-offline
~~~

## 如何读这批对照

原始八列及上一轮 16 列端点历史作为基线。将三态回应、两个计数异常方向、六项历史比例放入固定的 11 个新增位置。历史分别按源与协议、源与服务、源与服务及目的主机归组，使用严格早于本行逻辑时钟的 60 秒记录。同一毫秒的记录彼此不可见。训练、校准或评价切换采集时清空状态。模型输入排除了 IP 地址、UID、采集名、文件序号、绝对时间、作者窗口及标签。

| 输入组 | 回答的问题 |
| --- | --- |
| raw、contact | 同一份训练记录、同一校准规则下，原八列和第二十二轮 16 列历史各能做到什么？ |
| zero | 维度与新方法相同，把新增 11 列全部置零后，单纯改变宽度会产生什么变化？ |
| current | 放开本条记录的三态及两项方向标记，本条的观测质量能贡献多少？ |
| prior | 只放开六项过去 60 秒的比例，之前有多少条待定、多少条已知流未见回应，能贡献多少？ |
| quality | 同时使用本条和历史。主比较为普通 HGB 的 contact 对 quality。 |
| counter_masked | 沿用 quality 权重。在出现该方向校验事件、该方向包数又为零时，将该方向为零的包数及 IP 字节数分别改为缺失值 −1。只改有证据的数值，保留观测到的正值。重新在原正常校准集登记阈值。 |

六种输入分别配普通 HGB、按攻击族平衡的 HGB，以及种子 17、29、43 的 CUDA MLP，共 30 组模型、每个学习器七组封存判决。若同级目录 `tracelight-contact-r22/local/contact-r22` 保留了第二十二轮的已完成输出，输入、环境和模型哈希均相符时直接复用 10 组 raw/contact 模型，只新拟合 20 组。旧输出放在其他位置时，可在启动前将 `$env:TRACELIGHT_R22_OUTPUT` 设为其完整目录。找不到可核对的模型时才补做基线。两种路径都会把两条基线在同一批正常记录上的分数、阈值和告警与第二十二轮原件比较。HGB 固定 200 轮，MLP 固定 80 轮。

训练行与上一轮完全相同。4,096 条正常、4,040 条扫描和 56 条纯 C&C 中，56 条控制及 3,953 条扫描均为“发包、未见回应”。训练和 1,826 条正常校准记录中，两项方向性计数缺口都是零。这个分布限制了模型从标签中学习缺口的能力。counter_masked 单独测量将失真零计数按已有缺失表示处理的后果。训练集没有下载正例，下载评价检验跨攻击类别迁移。本机已用十份旧模型和新版历史重放全部 1,826 条正常校准记录，按各自登记阈值，十组都没有改变任何一条告警。[训练覆盖](response-evidence/training-observability.json) · [旧权重实际重放](response-evidence/r22-baseline-replay.json)

## 判读误报和漏检

每种分数只在先前的正常校准集取固定的保守 99% 秩阈值，严格大于阈值才告警。旧 52-1、60-1 单列回放。四个此前未用来评分的 IoT-23 采集均在训练和校准之后，早于已打开的 60-1，单列为历史留出。IoT-SFC 两个后续时段分采集和 UTC 日报告扫描和正常误报。TQH-C2 的独立试验单元另报外部控制通信，保留其 beacon/tasking 和侦察定义。作者标签的细项和数据文件指纹随包保存。[数据登记](../configs/response-evidence-20261112/registration.json)

四个 IoT-23 历史留出合计有 3,386,119 条纯扫描、95 条纯控制及 40 条纯或复合下载。IoT-SFC 采用作者更新的 v7 标签，两个采集共 180,595 条明确正常流，六个 UTC 日均有正常分母。主要扫描终点为 2025 年 4 月 10 日有作者攻击窗口标注的 83,936 条。作者另用行为启发式标了 2,769 条扫描，单列在标签来源附表。后一个采集没有主要扫描样本，可以评价正常误报和其他原生攻击。作者命名的攻击窗没有明确正常分母。外部 C2 试验单元共有 27,130 条信标或任务通信，缺少单独的下载真值。[IoT-SFC 作者版本](https://zenodo.org/records/23371775) · [TQH-C2 逐行核验与范围](response-evidence/TQH_C2.md)

对每个采集分别报告正常条数、允许的最大误报数（正常条数除以 100 后向下取整）、两方法误报数，以及各类原检出、现检出、新增、丢失和净增。纯 C&C、心跳、含 FileDownload 的下载及其他攻击分列。下载通信按作者原始组合标签保留明细。若某类没有样本，则标作无支持。若基线已经全检，报告可挽回漏检为零。同一统计中，净增 1 条仍可能伴随丢失旧检出，需同时看 gained 与 lost。

主结论按事先指定的普通 HGB 比较，不从多个种子或模型中事后挑最好者。单个采集证明三项目标同时改善，要求新旧两臂均符合 1% 且扫描、纯 C&C、下载各自净检出增加。新时段缺某一类时仍分别回答其余有支持的终点。跨采集汇总保留每个采集及 UTC 日的预算结果，不能用总体均值覆盖某一天的超标。1% 表示该时段实测误报率。

## 多核、GPU 和回传

早期源可并行解析。超大采集逐份构建，每份沿逻辑时间切块，带回精确的历史预热区间，向量化计算三种联系范围，再按原物理行号拼回。工作者数量按可用内存和最大窗口收缩。构建后续采集时，CPU 可同时拟合 HGB，CUDA 拟合 MLP。评分也分 CPU、GPU 队列。实验室在独占条件下用 52-1 同一真实前缀对比旧 SQLite、新版单工作者和多工作者，并核对全部八项旧历史、新版全部十四项历史和行号。两份本机实测见 [52-1 前缀](response-history-prefix-validation.json) 和 [60-1 密集前缀](response-history-dense-prefix-validation.json)。

每类评价来源另测一次 HGB 单线程每流判决的中位及 P95 延迟，并与封存分数和告警对照。这项计时从已构建特征开始。解析、历史构建、全量批评分分别计时。资源日志保存进程树 CPU 核占用、内存、GPU 利用率和显存。

完成后只需回传 `local\response-r23\response-results-*.zip` 中最后一份和同名 `.sha256.json`。检查包内 `last_invocation.json` 的状态是否为 complete。`outcome.json` 给出主比较，`evidence/evaluation/` 有逐采集、逐类别和观察状态的配对明细，`evidence/history-benchmark.json` 和资源 CSV 给出实验室实测。若运行中断，同一 checkout、输出目录和参数可接续，已完成阶段会先校验哈希。[Windows 合成预演](response-evidence/lab-rehearsal.json)
