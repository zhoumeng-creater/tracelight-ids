# 第十四轮：输入表示修订与同数据三臂对照

第十三轮发现，P词表外的端口等类别被统一编码为−1，部分Scanning与Benign在进入KV前已成为同输入。本轮先修输入契约和抽样依赖，验证完整模型是否因此改善，再测相同表示下的普通QCOC损失。拟议校准裕量几何和CRT更新不进入本轮。

采用固定[TabICLv2 2026源码](https://github.com/soda-inria/tabicl/blob/d83091b845a8c5f7c087bd54964f755a9e560f45/src/tabicl/_sklearn/preprocessing.py)，骨干与checkpoint保持第十三轮版本。[论文](https://arxiv.org/abs/2602.11139)为2026成果；[NF-ToN-IoT-v3](https://doi.org/10.48610/44d7c5e)为2025发布。下面是任务输入契约修订，不能称为新编码算法或原创贡献。

**输入表示。** 保留登记的48个字段。源/目的端口、PROTOCOL、L7_PROTO使用原生整数身份，按P均值/标准差作无裁剪仿射缩放；P内常量列尺度取1，仍保留。四列不经过类别词表、常量删除、标准分数裁剪或异常值处理。其他44列沿用官方P-only数值链，因此其常量列仍可能被删除。报告实际输出宽度。

端口接受0..65535，PROTOCOL接受0..255。L7_PROTO接受0..65535是本项目的显式输入域，官方说明没有承诺这一上界；实验室准备阶段对原CSV每一行验证非缺失整数和范围，任何越界都停止。每次拟合四列缩放后，穷举登记域检查FP32唯一性及舍入逆恢复。保证保留ID差异，不保证整数邻近就是服务语义，也不预告真实检出收益。

**抽样与误报单位。** `group_disjoint_flow_v2`保持原通信组定义、角色哈希、seed20261107、P8192（两类各4096）、正常C5000、非标签分层E20000。三角色各自在固定组池按哈希优先级独立抽样，取消P→C→E的全部特征排除。不同通信组的相同输入只登记，样本ID/声明通信组交集仍拒绝。误报目标按正常流量行计数，故重复流的频率保留；组数、最大组占比、原始及最终FP32模式数与混合标签情况全部报告。

消除顺序排除并不证明C/E可交换。[CRT 2025 §2](https://proceedings.neurips.cc/paper_files/paper/2025/file/6559542f75b4452ebaaf82094c7defb7-Paper-Conference.pdf#page=3)的保证有可交换性前提。本轮仍为开发校准，不是总体1%误报证书；通信组也不是主机或时间隔离。原第十三轮结果保持原判，新的E与旧E可能重合，最终确认需另行冻结未打开材料。

| 固定臂 | 输入表示 | 缓存 | 回答的问题 |
|---|---|---|---|
| legacy_full | 旧官方单视图词表/裁剪 | 完整KV | 同一v2数据上的表示参照 |
| full | 四ID保真，其余官方数值链 | 完整KV | 输入修订是否改善低误报工作点 |
| qcoc | 与full完全相同 | 普通QCOC，M512 | 新表示下的压缩代价 |

三臂共享相同P/C/E，每臂只用自己的C分数登记阈值（α=1%，严格大于）；E不参与选择。先看`legacy_full_vs_full`的净TP/FP、Scanning修复/新增、其余攻击族回退，再看`full_vs_qcoc`。若新full增加TP且不增加FP，才可报告这份开发比较的检测改善；若只有输入区分信息恢复，仍只能称表示修复。单独报告E实际FPR是否≤1%。压缩收益与检测损失同时报告，不事后放宽阈值或增加seed/M搜索。

准备记录的SHA绑定进入协议；运行前复核源身份、预算、三CSV的实际哈希/行数和v2抽样标记。预检还会拟合两种P-only预处理并审计最终FP32输入，不加载真实模型。旧v1格式和入口保留。

本机合成回归192项通过，0失败、0跳过，覆盖旧版兼容、全域ID身份检查、独立抽样、准备记录绑定、实际FP32输入审计、三臂阈值及三份部署包的独立进程复现；未加载真实预训练权重或拟合真实CSV。验证记录见[validation.json](research_alignment_20261107/context_input_v2/validation.json)。真实检测效果由下述实验室比较确定。

**实验室执行。** 固定代码标签为`context-input-lab-v2`。沿用已经安装的Python3.12、Torch2.6.0+cu126、TabICL2.2.0环境和原官方ZIP/权重，无需重新下载。以下目录来自本轮实验室回传；若移动过目录，只修改前三个路径变量。

```powershell
$repo = 'D:\zhoumeng\Graduate\tracelight-context-cache'
$py = 'D:\zhoumeng\Graduate\venvs\context-cache\Scripts\python.exe'
$rawCsv = 'D:\zhoumeng\Graduate\data\nf-ton-iot-v3\02934b58528a226b_NFV3DATA-A11964_A11964\data\NF-ToN-IoT-v3.csv'
Set-Location $repo
git fetch origin --tags
if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
git checkout --detach context-input-lab-v2
if ($LASTEXITCODE -ne 0) { throw 'fixed tag checkout failed' }
$checkpoint = Join-Path $repo 'local\checkpoints\tabicl-classifier-v2-20260212.ckpt'
$runRoot = Join-Path $repo 'local\context-input-v2'

function Invoke-InputPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $py @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($Arguments -join ' ')" }
}

# 合成回归。这里仅使用随机小模型和合成CSV。
Invoke-InputPython -m pytest -q `
  tests/test_weighted_context_cache.py tests/test_tabicl_cache_adapter.py `
  tests/test_context_compiler.py tests/test_context_cache_cli.py `
  tests/test_context_csv_preprocessing.py tests/test_context_calibration.py `
  tests/test_context_csv_workflow.py tests/test_context_lab_data.py `
  tests/test_context_id_preprocessing.py tests/test_context_lab_data_v2.py `
  tests/test_context_representation_audit.py tests/test_context_input_workflow_v2.py `
  tests/test_context_input_cli_v2.py

# 新建一份v2划分；不使用第十三轮旧prepared目录。
$prepared = Join-Path $runRoot 'prepared'
Invoke-InputPython -m ids.context_cache lab-prepare-v2 --raw-csv $rawCsv --output $prepared
$meta = Get-Content .\configs\context-cache-checkpoint.json -Raw | ConvertFrom-Json
Invoke-InputPython -m ids.context_cache lab-register `
  --template .\configs\context-cache-nf-ton-v3-input-v2.json `
  --support-csv (Join-Path $prepared 'support.csv') `
  --calibration-csv (Join-Path $prepared 'normal_calibration.csv') `
  --evaluation-csv (Join-Path $prepared 'development_evaluation.csv') `
  --checkpoint $checkpoint --checkpoint-sha256 $meta.sha256 `
  --output (Join-Path $runRoot 'protocol.json')
Invoke-InputPython -m ids.context_cache lab-preflight `
  --config (Join-Path $runRoot 'protocol.json') --output (Join-Path $runRoot 'preflight.json')
```

查看`preparation.json`的全源ID验证范围、`preflight.json`的`audit.representations.legacy/new`和组集中度。四ID身份检查通过是工程条件；剩余混合标签模式照实保留，不通过改变seed或删行把审计数字做成零。

```powershell
# 正式三臂。CUDA只用于构建/编译，登记阈值和E评分均为CPU FP32。
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Invoke-InputPython -m ids.context_cache lab-input-compare `
  --config (Join-Path $runRoot 'protocol.json') --output (Join-Path $runRoot 'comparison')

# 只封装这一轮。5.3GB原CSV与原下载checkpoint不在runRoot内，不重复打包。
$archive = "$runRoot-results.zip"
if (Test-Path -LiteralPath $archive) { throw 'Result archive already exists' }
Invoke-InputPython -c "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]).resolve(); shutil.make_archive(str(pathlib.Path(sys.argv[2]).with_suffix('')),'zip',root_dir=p.parent,base_dir=p.name)" $runRoot $archive
Get-FileHash -LiteralPath $archive -Algorithm SHA256
```

回传包需包含三臂部署工件、各自校准分数、`evaluation_scores.csv`、两份配对CSV、两份实际表示审计、准备记录和协议。所有输出路径均须不存在，失败后保留原目录供定位。一次墙钟评分不是p95或RSS，本轮仍不做正式资源基准。
