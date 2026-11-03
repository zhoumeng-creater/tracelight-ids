# 第六轮实验室操作：RTX 4090D 真实研究

本轮是TraceLight research v2开发研究。**后续真实数据训练统一在实验室RTX 4090D台式机执行；本机只做代码、单元测试、合成数据验证和回传工件复核。** 已有本机真实pilot保留为工程预验证／探索，不能代替实验室pilot，更不能写成正式五种子结果。

第五轮test已经使用，本入口只访问既有train/validation，不下载、不prepare、不读calibration/test。研究问题和验收见[固定协议](STAGE6_PROTOCOL.md)、[研究设计](STAGE6_RESEARCH_DESIGN.md)及[近期模型身份](RECENT_METHODS.md)。CUDA和RTX 4090D是本轮真实训练的默认门禁；设备不符应停止，不静默退回CPU或本机4050。

## 1. 沿用实验室已有目录与Python

使用已完成前几轮实验的实验室SSD、仓库和Python。下面是通用布局示例；先按实际路径核对变量，不移动数据，也不重新prepare。公开说明不包含个人计算机目录。

~~~powershell
$labRoot = 'D:\TraceLightLab'
$repo = Join-Path $labRoot 'tracelight-ids'
$py = Join-Path $labRoot 'venv\Scripts\python.exe'
$unsw = Join-Path $labRoot 'data\prepared-v1'
$stage4 = Join-Path $labRoot 'stage4-stable-selection'
$cicids = Join-Path $stage4 'data\cicids-prepared'
$stage6Check = Join-Path $labRoot 'stage6-research-v2-preflight'
$stage6Pilot = Join-Path $labRoot 'stage6-research-v2-pilot'
$stage6Full = Join-Path $labRoot 'stage6-research-v2-full'
Set-Location -LiteralPath $repo
$env:PYTHONPATH = Join-Path $repo 'src'

git status --short
git rev-parse HEAD
& $py scripts/run_research_lab.py --help
if ($LASTEXITCODE -ne 0) { throw '本轮入口不可用，请先核对发布版本和Python路径' }
~~~

先按本轮发布提交更新仓库；有本地修改就保留并处理，不用reset/clean丢弃。实验开始后不再pull、切分支或改代码／配置。$py始终指向同一个绝对解释器，不混用PATH里的python。

若实验室布局与示例不同，查看原$stage4/experiment/STAGE4_PLAN.json和$stage4/preflight.json记录的实际prepared路径、清单哈希与设备信息。可在已知第四轮工作目录里定位这两个文件；路径字符串存在不代表数据完整或身份正确。CIC回传ZIP仅有清单和来源，不能拿回传包中的小清单目录代替实验室完整prepared。

分别检查$unsw/manifest.json与$cicids/manifest.json存在，并与旧运行记录核对。新入口不会自动搜索磁盘、下载或重建它们；缺失、schema错误、样本／清单不匹配时先找回正确目录，不能换一份近似CSV继续。

## 2. 核验同一解释器与可选依赖

沿用实验室已有可用的PyTorch 2.6.0 CUDA构建，不重装torch、驱动或CUDA。先核验实际设备：

~~~powershell
& $py -c "import sys,torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
if ($LASTEXITCODE -ne 0) { throw '请先核对原实验室环境' }
& $py -c "from ids.recent_baselines import recent_dependency_versions; print(recent_dependency_versions())"
~~~

如果旧环境尚未安装本轮TabM依赖，使用明确选定的研究解释器一次性安装固定的可选包，再重新核验：

~~~powershell
& $py -m pip install --no-deps -r requirements-research.txt
if ($LASTEXITCODE -ne 0) { throw '可选依赖安装失败，停止实验' }
& $py -c "from ids.recent_baselines import recent_dependency_versions; print(recent_dependency_versions())"
if ($LASTEXITCODE -ne 0) { throw '近期模型依赖核验失败，停止实验' }
~~~

这只增加锁定的TabM等可选包，不自动升级基础依赖。若使用独立研究环境，先恢复并验证同版本基础依赖，再让$py指向该环境。**在已有venv里运行venv --system-site-packages不保证新环境继承旧venv中的torch**，不能仅凭新目录建好就宣布CUDA环境可用。安装与训练必须用相同解释器，保存实际依赖清单。

上方PYTHONPATH只为当前会话显式选定本仓库src；不能误用其他目录中的旧ids包。

依赖、CUDA、GPU型号和数据身份全部通过后才能真实训练。无需在本机重跑真实数据来替实验室“预热”。

## 3. 实验室先dry-run，再UNSW pilot

先对两套正式开发数据做只读预检，不拟合模型、不读保留分区。这里使用独立预检工作目录，避免与真实输出混淆：

~~~powershell
& $py scripts/run_research_lab.py --unsw-data-dir $unsw --cicids-data-dir $cicids --work-dir $stage6Check --profile full --dry-run
if ($LASTEXITCODE -ne 0) { throw '第六轮预检失败，保留输出并核对环境和prepared路径' }
~~~

预检核对发布代码、实际Python／CUDA／GPU、两环境manifest与schema及运行计划。预检成功表示可开始后续步骤，不表示实验已执行。dry-run也会写LAB_PREFLIGHT.json，上述预检、pilot、full采用不同工作目录。若要直接在同一full工作目录从dry-run转入真实运行，必须保持profile、数据、代码和硬件一致，并给真实命令加--resume：

~~~powershell
# 只在stage6Check确实由上方同一full dry-run建立、且未改变绑定时使用。
& $py scripts/run_research_lab.py --unsw-data-dir $unsw --cicids-data-dir $cicids --work-dir $stage6Check --profile full --resume
~~~

这条是复用原full预检目录的可选执行路径，不要再与第4节的新full工作目录重复执行同一研究。pilot与full配置不同，始终使用不同工作目录。

然后只在实验室4090D上做UNSW真实pilot：

~~~powershell
& $py scripts/run_research_lab.py --unsw-data-dir $unsw --work-dir $stage6Pilot --profile pilot
if ($LASTEXITCODE -ne 0) { throw '实验室pilot已停止，请保留现场和错误输出' }
~~~

pilot只允许UNSW；不要传--cicids-data-dir，入口会拒绝。配置为固定seed17、train/validation最多12,000／6,000行、最多20epochs；13个主条件加4个教师拟合属于工程和成本小试，不能判定新方法有效，也不与完整开发研究作同预算比较。

必须检查实际家族支持、组排除、OOF身份、同初始化配对、数值、逐损失记录和耗时。旧本机pilot不能替代这个实验室步骤。UNSW pilot通过后，按实际成本安排完整研究；不因包已生成就自动扩大。

CIC的大Portscan关联组有139,618条训练流，用小行数cap整组采样会排除它。因此不对CIC套用UNSW bounded pilot并称为同任务复验；正式CIC使用完整既有开发分区。

## 4. 一条命令串行运行两环境完整开发研究

确认实验室pilot工件合格、数据和环境未变化后，在同一会话执行：

~~~powershell
& $py scripts/run_research_lab.py --unsw-data-dir $unsw --cicids-data-dir $cicids --work-dir $stage6Full --profile full
if ($LASTEXITCODE -ne 0) { throw '第六轮研究已停止，请保留工作目录和错误输出' }
~~~

入口使用CUDA／RTX 4090D守门，按顺序运行两个数据环境，不同时抢占GPU。每个环境固定五seeds（17、29、43、71、97），13个主条件共65次拟合，加15个OOF教师和5个全train教师参考。两套环境共170次计划拟合，失败／pilot／追加搜索另记。HGB仍在CPU训练，MLP和TabM在CUDA训练，不能把整个工作流全部称为GPU计算。

完整配置使用cohort_policy=complete_development、两个行数cap均为null；不得为省时悄悄删整组或减seed。训练预算通过epochs、batch、选择池和串行排队控制。若实耗不可承担，应停止并公开修订协议，不能静默换低配矩阵。

入口为每个完成的数据环境分别生成可回传ZIP及其SHA-256文件，并打印实际路径；它不自动向外发送或上传数据。真实结果仍属于既有train/validation的v2开发证据，不是新独立test结果。GPU型号与运行记录也不能代替对结果和工件的复核。

## 5. 恢复、回传与完成证据

中断后不更新源码，不改环境／配置，不删错误目录。沿用同一命令和同一工作目录增加--resume：

~~~powershell
& $py scripts/run_research_lab.py --unsw-data-dir $unsw --cicids-data-dir $cicids --work-dir $stage6Full --profile full --resume
if ($LASTEXITCODE -ne 0) { throw '恢复核验未通过，请保留现场并核对绑定信息' }
~~~

完成的研究阶段只在身份与哈希核验通过后复用；不承诺epoch断点续训。未完成子研究若不能恢复，保留原目录和失败记录，按入口错误说明处理或用新的工作目录重跑，不能覆盖成一次不间断成功运行。已完成后再resume用于验证复用，不应重训已封存模型。

完整两环境运行应带回整个packages目录，其中包含research-unsw.zip及其.sha256、research-cicids.zip及其.sha256，以及入口从实际预检原件复制的LAB_PREFLIGHT.json及其.sha256；实际位置以入口打印为准。pilot只会有UNSW一套。保留实验室两套prepared、第四至第五轮原目录和本轮完整工作目录，直到接收端逐项核验完毕。不要只发截图、表格或单个模型。回传内容至少应能核对：

- 实际实验室设备／CUDA／Python／依赖、发布代码与配置、开始结束记录。
- 两环境来源、完整／子样本家族支持、组边界和样本身份。
- 每seed的折内教师、OOF分数、全train教师参考及其区别。
- 学生部署包、逐epoch损失、gate与pair诊断、validation逐条预测／阈值／家族计数。
- 同seed主比较与机制控制、完整矩阵覆盖、逐文件哈希和完成标记。

RESEARCH_COMPLETE.json等标记只表示对应开发链与工件覆盖完整。只有实验室实际运行和接收端复核完成，才能将“待执行”改成“实验室已运行／已复核”。现有本机pilot、合成CUDA冒烟、dry-run都不能替代该证据。

CPU三轮部署基准、视觉系统真实操作验收和新来源的最终确认仍需独立工件；本训练入口完成不自动表示它们完成。界面展示始终标出数据分区与运行版本，不把回放或开发分数写成实时生产检测。
