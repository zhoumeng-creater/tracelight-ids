# v4后续实验室操作：主线价值与阈值可行性审计

2026-11-04本轮增强入口见[新研究运行指南](RESEARCH_SCREEN_LAB.md)。当前源码新增实际OOF折教师的F-heldout/R/C native查询，工件为schema3；下述旧固定标签保留原schema2行为。需要新增诊断时使用本轮checkout，不能只拉旧标签。

固定标签：`research-foundation-audit-20261104`。先读[根本设计复盘](RESEARCH_ROOT_CAUSE_REVIEW_20261104.md)和[本轮结果复核](UPGRADE_V4_RESULTS_20261104.md)。本次只查询已保存模型，不拟合教师／学生；目的包括教师是否有有效纠错信息、分数处理是否抹去知识、哪些失败来自阈值注册、哪些固定分数确实没有同时满足全部有限池事件预算的阈值。它是研究资格检查，不作为新算法贡献。

## 1. 复用原v3、v4和prepared，另建代码与输出

以下为可替换的布局示例，沿用原实验室根目录和Python；个人路径在本地专用交接脚本中已填，公共指南只用通用示例。v3/v4路径是含各自完整完成标记的环境目录，不能指向packages或只有JSON的回传目录。

```powershell
$labRoot = 'D:\TraceLightLab' # 改为原实验室根目录
$repo = Join-Path $labRoot 'tracelight-research-value-audit'
$py = Join-Path $labRoot 'venv\Scripts\python.exe'
$v3 = Join-Path $labRoot 'experiments\risk-v3-full-20261103'
$v4 = Join-Path $labRoot 'experiments\upgrade-v4-full-20261104'
$unsw = Join-Path $labRoot 'data\prepared-v1'
$cicids = Join-Path $labRoot 'experiments\stage4-stable-selection\data\cicids-prepared'
$check = Join-Path $labRoot 'experiments\research-value-audit-check-20261104'
$work = Join-Path $labRoot 'experiments\research-value-audit-20261104'

git clone --branch research-foundation-audit-20261104 --single-branch https://github.com/zhoumeng-creater/tracelight-ids.git $repo
if ($LASTEXITCODE -ne 0) { throw '拉取失败，保留已有目录，不覆盖' }
Set-Location -LiteralPath $repo
$env:PYTHONPATH = Join-Path $repo 'src'
git rev-parse HEAD
& $py scripts/audit_upgrade_thresholds_lab.py --help
if ($LASTEXITCODE -ne 0) { throw '核对原Python和当前checkout' }
```

沿用锁定环境，不安装新模型包、不重装torch/驱动、不重新prepare。预检要求CUDA0 RTX4090D；原v3基准C事件需按原训练设备重算并核reference signature，候选R/C/V按CPU部署路径评分。新增源码放新checkout，旧结果只读。

## 2. 先预检，再查询已有模型

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
& $py @auditArgs --work-dir $check --dry-run
if ($LASTEXITCODE -ne 0) { throw '预检失败，保留原记录和报错' }

& $py @auditArgs --work-dir $work
if ($LASTEXITCODE -ne 0) { throw '评分审计停止，保留现场，不覆盖重跑' }
```

dry-run只读JSON清单/身份与环境，不load模型、不读parquet。正式审计先验旧v3/v4全部哈希及开发数据；只读取train/validation，再按原身份取F/R/C/V。F不用于新增拟合，阈值只由R/C决定，V在阈值固定后才forward。

查询原v3 anchor/hard及本轮已训练v4选中模型。按当前回包共28个已有MLP工件，含权重相同的回退副本；另查询6个完整F教师，各报告native HGB raw margin与精确旧float32 clip两种分数，不视为新增训练或独立实验。旧v4缺失的Phase-I模型明确记 `missing_phase1_snapshot`；不会靠重新训练伪造旧权重。未来新工件如包含封存的Phase-I模型也会纳入审计。

每模型记录三种口径：R-only阈值、冻结原anchor阈值、精确R+C可行区间。正常R/C预算给阈值下界；原anchor pooled/family TP保留给上界。空区间会标出冲突约束；非空选择最低合法阈值，再描述V结果。C参与阈值选择属于新的训练规则，不是R-only独立校准，更不是总体风险证书；旧v4原判定保留。

教师单列价值诊断：两个分数版本先在R确定自己的阈值，再在V报告低FPR性能、标准化pAUC、逐族表现、相对学生的新增／丢失TP与修复／新增FP；不把MLP旧阈值硬套到HGB分数。完整F教师与实际OOF教师不同，监督量也小于更新学生F+C，所以这是能力／错误互补诊断，不是公平排行榜或可实现的迁移增益。实际OOF的旧分数分布、折拟合支持和家族缺失另行记录；native折预测仍未查询。

## 3. 回传与接收端复算

将 `$work\packages\` 下全部6个文件带回：

- `THRESHOLD_AUDIT_UNSW.zip` 与SHA
- `THRESHOLD_AUDIT_CICIDS.zip` 与SHA
- `THRESHOLD_AUDIT_LAB_PREFLIGHT.json` 与SHA

包内含每模型R/C/V raw logits的 `scores.npz`、教师双分数与互补诊断、三份同序身份、原GPU基准margin／成员mask、精确上下界／事件计数、三口径V指标、模型来源SHA及源码/环境；不重复打包模型权重或prepared特征。接收端可在没有原始特征和GPU的情况下复算阈值与事件。`fits=0`和完成标记表示评分链完整，不表示任何候选改善或统计认证。

如果原GPU重算reference signature与封存v3不一致，停止并带回报错，不静默换基准。所有中断输出保留；`--resume`只接受完整同输入工件，不能续半程。完整结果复核／补打包可在原正式命令后加 `--resume`。

## 4. 结果回来后怎样决定

1. 先判断教师在目标工作点是否有纠错余量、哪些是学生可利用的信息；若没有，不围绕无效目标继续加KD模块。
2. 若旧模型已有非空区间：修正阈值作为可行性变量的注册方式，并用相同规则构造所有方法的共同起点；不把校准收益当成知识迁移贡献。
3. 若区间为空：依据控制下界的正常尾部和控制上界的攻击族，定义有限预算下的排序／表示问题，再做成熟CVaR-pAUC或可行化控制。
4. 结合明确部署预算，先比较同样小的hard与有资格的KD／直接目标；guard全拒即停止，不增加相同空转步数。

新的主问题、训练矩阵及任何新颖性／增益主张，要等这份价值诊断完成后再冻结。当前无需重跑48条件。
