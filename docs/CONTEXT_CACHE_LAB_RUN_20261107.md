# 实验室完整命令：NF-ToN-IoT-v3、完整 KV 与 QCOC

2026-11-07。执行版本固定为 Git 标签 `context-cache-lab-20261107-v1`，开发分支为 `research/context-cache-20261107`。这份指南从空代码目录开始，安装独立环境、取得官方数据和权重、生成冻结配置，再运行两个候选。以下下载与真实数据命令仅在实验室执行；本机交付核查使用合成 CSV 和随机模型。

## 1. 本轮固定选择

| 项目 | 登记值 |
|---|---|
| 数据 | UQ 发布的 [NF-ToN-IoT-v3（2025）](https://doi.org/10.48610/44d7c5e)；[来源记录](research_alignment_20261107/context_csv/dataset_source.json) |
| 输入 | 53 个流字段中保留 48 个；排除两列 IP、两列绝对时间、DNS_QUERY_ID；端口、PROTOCOL、L7_PROTO 为类别 |
| 支持 P | 8,192 行，正常／攻击各 4,096；预处理只在 P 拟合 |
| 校准 C | 5,000 条正常流；双方各用自己的分数登记阈值 |
| 开发评价 E | 20,000 行，抽样不按标签或攻击族分层；记录实际类别组成 |
| 编译 | 同一 TabICLv2 2026 checkpoint；完整 KV 与普通 QCOC，M=512、seed=20261107 |
| 判定 | α=1%，原生 logit 差，严格 `score > threshold` |
| 设备 | 实验室 CUDA 构建／编译，CPU FP32 校准／评价，1 线程、查询批 128、编译查询块 256 |

这是首轮开发预算，未做 N/M/seed 搜索。原始无向端点对 `(IP,port)` 加协议定义通信组；固定哈希将整组分到 P/C/E 候选池（50%／20%／30% 哈希桶）。各池按源文件哈希、行号、seed 的固定优先级抽取。按 P→C→E 顺序排除已被前一角色选中的完整特征模式，再补足当前配额；剔除数量与最终重叠审计写入准备报告。通信组不冒称真实会话，E 是开发比较，不作为时间泛化或最终独立确认。

只运行这一份数据和这两个候选。校准裕量几何与 CRT 更新不在此次入口；旧 test、旧 HGB/KD 阈值和准入门槛不进入本轮。

## 2. 拉取固定代码

在实验室 **Windows PowerShell** 执行。示例使用本地 SSD 的 `D:\TraceLight`，不把 venv 或运行产物放入 OneDrive。需要 Git、64 位 Python 3.12 和可用 NVIDIA 驱动。

```powershell
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path D:\TraceLight | Out-Null
Set-Location D:\TraceLight
git clone --branch research/context-cache-20261107 https://github.com/zhoumeng-creater/tracelight-ids.git tracelight-context-cache
if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }
Set-Location D:\TraceLight\tracelight-context-cache
git checkout --detach context-cache-lab-20261107-v1
if ($LASTEXITCODE -ne 0) { throw 'fixed tag checkout failed' }
git rev-parse HEAD
```

新目录让这轮代码与旧实验工作区分开。若已经有本轮干净检出，可在该目录 `git fetch origin --tags` 后检出同一标签，不覆盖有改动的旧目录。

## 3. 安装和合成核查

先用 `py -3.12` 找到 Python 3.12。安装脚本创建专用 venv，固定 Torch 2.6.0+cu126、项目依赖和 TabICL 源码；检查依赖、CUDA 运算、官方源码哈希，并执行本轮合成测试及冷包核查。详细安装内容见[环境说明](CONTEXT_CACHE_LAB_SETUP_20261107.md)。脚本中任一命令失败即停止，不会自动运行真实实验。

```powershell
$basePython = (py -3.12 -c "import sys; print(sys.executable)").Trim()
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required' }
$runRoot = Join-Path (Get-Location) 'local\context-cache-lab-20261107'
& .\scripts\setup_context_cache_lab.ps1 `
  -Python $basePython `
  -VenvDir D:\TraceLight\venvs\context-cache `
  -CheckOutput (Join-Path $runRoot 'setup')
$py = 'D:\TraceLight\venvs\context-cache\Scripts\python.exe'

function Invoke-LabPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($Arguments -join ' ')" }
}
```

`setup/` 保存依赖解析、`pip freeze`、GPU 身份、源码校验和合成报告。实验室运行前确认其中设备确为预期 GPU。脚本没有修改全机执行策略；若脚本被实验室策略阻止，按该机器已有的脚本许可方式执行。

## 4. 取得固定官方数据和权重

原始 CSV 约 5.3 GB，官方 ZIP 为 417,476,635 字节。UQ 下载 API 返回临时 URL，每次现场获取；不把临时签名保存成固定下载地址。下列命令提取官方 ZIP 中指定的 CSV。准备阶段会核对发布方 BagIt 清单中的 SHA-1 `24854ca3072ab7c2ade9ebfb202345a666a94cfe`，并登记现场计算的 SHA-256。

```powershell
$dataDir = 'D:\TraceLight\data\nf-ton-iot-v3'
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
$zipPath = Join-Path $dataDir 'NF-ToN-IoT-v3-official.zip'
if (-not (Test-Path -LiteralPath $zipPath)) {
    $download = Invoke-RestMethod 'https://api.rdm.uq.edu.au/datasets/343e2e8c-6e6e-4a0c-813d-a46acea1b7f4/download'
    if (-not $download.canDownload -or $download.packagingStatus -ne 0) { throw 'Official dataset is not ready for download' }
    Invoke-WebRequest -UseBasicParsing -Uri $download.url -OutFile $zipPath
}
$member = '02934b58528a226b_NFV3DATA-A11964_A11964/data/NF-ToN-IoT-v3.csv'
$rawCsv = Join-Path $dataDir $member
if (-not (Test-Path -LiteralPath $rawCsv)) {
    Invoke-LabPython -c "import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); z.extract(sys.argv[2],sys.argv[3]); z.close()" $zipPath $member $dataDir
}
```

权重来自作者仓库 `jingang/TabICL`，固定 revision、文件名和 LFS SHA-256 见[权重登记](../configs/context-cache-checkpoint.json)。本机只核对了官方元数据；下面的实验室命令才下载文件。

```powershell
$checkpointMeta = Get-Content .\configs\context-cache-checkpoint.json -Raw | ConvertFrom-Json
$checkpointDir = Join-Path (Get-Location) 'local\checkpoints'
Invoke-LabPython -c "import json,sys; from huggingface_hub import hf_hub_download; m=json.load(open(sys.argv[1],encoding='utf-8')); print(hf_hub_download(repo_id=m['repo_id'],filename=m['filename'],revision=m['revision'],local_dir=sys.argv[2]))" .\configs\context-cache-checkpoint.json $checkpointDir
$checkpoint = Join-Path $checkpointDir $checkpointMeta.filename
if ((Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checkpointMeta.sha256) {
    throw 'Checkpoint SHA-256 mismatch'
}
```

## 5. 生成划分、登记配置与预检

`lab-prepare` 执行固定来源校验和三次流式扫描，不加载模型。原始特征名来自官方特征说明；`Label`／`Attack` 的实际表头和二值／攻击族对应关系在这里检查，不符就报错。不会静默改变标签、抽样配额或 seed。

```powershell
$prepared = Join-Path $runRoot 'prepared'
$protocol = Join-Path $runRoot 'protocol.json'
Invoke-LabPython -m ids.context_cache lab-prepare --raw-csv $rawCsv --output $prepared
Invoke-LabPython -m ids.context_cache lab-register `
  --template .\configs\context-cache-nf-ton-v3.json `
  --support-csv (Join-Path $prepared 'support.csv') `
  --calibration-csv (Join-Path $prepared 'normal_calibration.csv') `
  --evaluation-csv (Join-Path $prepared 'development_evaluation.csv') `
  --checkpoint $checkpoint --checkpoint-sha256 $checkpointMeta.sha256 `
  --output $protocol
Invoke-LabPython -m ids.context_cache lab-preflight --config $protocol --output (Join-Path $runRoot 'preflight.json')
```

`protocol.json` 的路径、哈希、行数来自实际文件；模型配置取自[本轮固定模板](../configs/context-cache-nf-ton-v3.json)。`prepared/preparation.json` 保存官方来源身份、原始 SHA-256、分组／抽样规则、每次扫描计数、双方类别和攻击族组成、重复剔除与最终重叠检查。预检不加载权重、不运行推理。

## 6. 正式运行一次对照

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$comparison = Join-Path $runRoot 'comparison'
Invoke-LabPython -m ids.context_cache lab-compare --config $protocol --output $comparison
```

入口只构建完整 KV 和一个 M=512 的 QCOC。每个模型在 C 上登记自己的阈值，再对同一 E 评分。输出目录必须不存在，失败产物保留；不要通过换 seed、减少行数或改 M 继续沿用本标签的首轮协议。

报告中读取 `evaluation`、`paired.changes`、逐族结果及 `complete_csv_deployment_files`：分别看修复／新增 FP、FN，双方实际 FPR/TPR，以及包含骨干和预处理的完整包字节。`environment.json` 保存实际运行硬件与依赖；一次墙钟时间只作过程记录，不能写成 p95 或 RSS。

## 7. 打包回传

整轮目录含配置、来源准备记录、合成环境核查、校准分数、逐行配对结果和两个部署包。原始 5.3 GB CSV 与原始下载 checkpoint 位于目录之外，不重复打包。

```powershell
$archive = Join-Path (Get-Location) 'local\context-cache-lab-20261107-results.zip'
if (Test-Path -LiteralPath $archive) { throw 'Result archive already exists' }
Invoke-LabPython -c "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]).resolve(); shutil.make_archive(str(pathlib.Path(sys.argv[2]).with_suffix('')),'zip',root_dir=p.parent,base_dir=p.name)" $runRoot $archive
Get-FileHash -LiteralPath $archive -Algorithm SHA256
```

回传 ZIP 及其 SHA-256 后，按冻结配置核对执行完整性，再解释完整 KV 的能力、QCOC 新增和修复的错误、以及实际包体积变化。
