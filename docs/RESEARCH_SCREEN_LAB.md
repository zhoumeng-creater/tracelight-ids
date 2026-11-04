# 新研究入口：先查已有教师，再筛查学生

2026-11-04。设计和停止条件见[项目重构](PROJECT_RECONSTRUCTION_20261104.md)。本轮代码与旧`budget-transfer-20261104`、`research-foundation-audit-20261104`标签不同。使用本轮完整、干净的独立checkout，记录实际commit；不要更新正在运行的旧目录。

## 1. 解释器与目录

沿用实验室RTX4090D、已有prepared和v3/v4工件。以下设置不执行训练：

```powershell
$labRoot = 'D:\zhoumeng\Graduate'
$repo = Join-Path $labRoot 'tracelight-reconstruction'
$py = Join-Path $labRoot 'venv\Scripts\python.exe'
$v3 = Join-Path $labRoot 'experiments\risk-v3-full-20261103'
$v4 = Join-Path $labRoot 'experiments\upgrade-v4-full-20261104'
$unsw = Join-Path $labRoot 'data\prepared-v1'
$cicids = Join-Path $labRoot 'experiments\stage4-stable-selection\data\cicids-prepared'
Set-Location -LiteralPath $repo
$env:PYTHONPATH = Join-Path $repo 'src'
git status --short
$revision = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw '检查本轮独立checkout' }
$shortRevision = $revision.Substring(0, 8)
& $py -c "import sys, torch; print(sys.executable, torch.__version__, torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw '检查已有实验室解释器' }
```

若交付为本地`.bundle`，先执行`git clone <bundle绝对路径> $repo`。本轮尚未推送时，不用旧GitHub标签替代新代码。

## 2. 第一步查询封存模型，新增拟合为0

使用当前源码执行增强审计。预检与正式查询使用不同输出目录：

```powershell
$auditArgs = @(
  'scripts/audit_upgrade_thresholds_lab.py',
  '--unsw-v3-dir', (Join-Path $v3 'risk-v3-unsw'),
  '--unsw-v4-dir', (Join-Path $v4 'upgrade-v4-unsw'),
  '--unsw-data-dir', $unsw,
  '--cicids-v3-dir', (Join-Path $v3 'risk-v3-cicids'),
  '--cicids-v4-dir', (Join-Path $v4 'upgrade-v4-cicids'),
  '--cicids-data-dir', $cicids
)
$auditCheck = Join-Path $labRoot ('experiments\teacher-audit-check-' + $shortRevision)
$auditWork = Join-Path $labRoot ('experiments\teacher-audit-' + $shortRevision)
& $py @auditArgs --work-dir $auditCheck --dry-run
if ($LASTEXITCODE -ne 0) { throw '预检失败，保留报错' }
& $py @auditArgs --work-dir $auditWork
if ($LASTEXITCODE -ne 0) { throw '审计停止，保留原目录和报错' }
```

正式审计保留原参考教师/学生R/C/V诊断，新增实际OOF教师F-heldout/R/C查询。每折分别登记native/旧clip阈值，核验旧目标精确重现；不跨折混排margin。schema3含F身份、`oof-scores.npz`和每折来源/支持。OOF部分不查V，整个审计仍有原参考模型的V诊断。

带回`$auditWork\packages`中两环境ZIP及SHA、预检JSON及SHA，共6个文件。先核教师信息和分数处理问题，再决定下一节；审计不会自动启动拟合。

## 3. 有必要时执行基础screen

screen新增CatBoost1.2.10。在已有实验室虚拟环境安装，用现有锁文件约束共同依赖：

```powershell
& $py -m pip install -r requirements-native-trees.txt -c requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw '可选树依赖未安装完成' }
& $py -c "import catboost,torch; print(catboost.__version__,torch.__version__); assert catboost.__version__ == '1.2.10'"
if ($LASTEXITCODE -ne 0) { throw '版本核验失败' }
& $py -m pytest tests/test_native_trees.py -q
if ($LASTEXITCODE -ne 0) { throw '合成适配检查失败' }
```

以下是筛查的预检/正式入口。脚本使用自身所在checkout，默认`screen`，不clone旧标签或改分支：

```powershell
.\scripts\RUN_BUDGET_TRANSFER.ps1 -LabRoot $labRoot -DryRun
.\scripts\RUN_BUDGET_TRANSFER.ps1 -LabRoot $labRoot
```

脚本遇到错误会终止；预检失败时不要执行正式命令。路径不同时传`-V3Root`、`-UnswData`、`-CicidsData`。seed17、两环境各12次拟合，只解析F/R/C；CPU计时取C中固定样本，所有模型使用同一管线口径。

C参与经验误报阈值和MLP检查点选择，C≤1%是训练规则的结果。MLP与固定树终态的选择机会不同，本轮是函数类/资源诊断，不是充分调优排行。记录真实选择次数、更新步数和处理行数；树叶数不与神经参数数混用。

带回正式工作目录`packages`的6个文件。旧pilot/full须显式指定，仍是已撤回优先推荐的同类排序对照，不是screen之后的默认下一步。screen完成后不会读取V或运行KD。

`-Resume`只核验已完整封存的同输入结果，不支持中途接续拟合。中断输出保留，先定位错误，不覆盖或重跑全部矩阵。
