# UNSW 普通 KD：实验室运行指南

本轮验证：真实教师信息是否让全字段 small 模型优于三个同预算对照。固定 H/K/S/M 四次续训，每次40次全F更新；复用已有教师和warmup。详细设计与验收见[冻结协议](UNSW_KD_MINIMAL_PROTOCOL_20261105.md)。

## 1. 拉取本轮代码

在实验室 PowerShell 执行。默认使用已有目录 D:\zhoumeng\Graduate 和其中的 venv；如果目录迁移，只改 labRoot。代码放到新的目录，实验数据继续复用原位置。

~~~powershell
$ErrorActionPreference = 'Stop'
$labRoot = 'D:\zhoumeng\Graduate'
$repo = Join-Path $labRoot ('tracelight-unsw-kd-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
git clone --branch lab-unsw-kd-20261105 --single-branch https://github.com/zhoumeng-creater/tracelight-ids.git $repo
if ($LASTEXITCODE -ne 0) { throw '拉取失败，停止。' }
Set-Location -LiteralPath $repo
~~~

lab-unsw-kd-20261105 是本轮固定标签。运行期间保持该代码目录不变。

## 2. 先预检，不训练

~~~powershell
.\scripts\RUN_UNSW_KD.ps1 -LabRoot $labRoot -PreflightOnly
~~~

脚本自动寻找原 UNSW screen 和 prepared 数据，核对环境、原始 train 文件、F/R/C 身份、教师、学生起点及四臂目标哈希。预检不训练、不查询新模型分数。

若只有一份匹配工件，会直接通过；如果找到多个副本，终端列出路径。用正确原目录显式指定即可：

~~~powershell
.\scripts\RUN_UNSW_KD.ps1 -LabRoot $labRoot -ScreenDir '原UNSW screen目录' -UnswData '原prepared目录' -PreflightOnly
~~~

原 prepared 路径为 D:\zhoumeng\Graduate\data\prepared-v1。不要重新 prepare 数据或重训教师。预检失败时保留终端提示和生成的失败包。

## 3. 预检通过后运行

~~~powershell
.\scripts\RUN_UNSW_KD.ps1 -LabRoot $labRoot
~~~

若预检使用了 ScreenDir、UnswData，正式命令也传相同参数，去掉 PreflightOnly。

脚本依次完成：

1. H硬标签、K真实KD、S模式置乱、M类别均值四次续训。
2. 固定第40步的R/C评估、三组配对及逐攻击族检查。
3. H/K的CPU成本测量。
4. 生成验收结果并自动打包，核对包内清单和整包SHA。

总计160次AdamW更新、13,746,560行带梯度前向。没有教师重训、额外种子或参数搜索。科研验收未通过也会正常完成，完整回传即可。

## 4. 回传什么

终端会打印工作目录和最终ZIP路径。正式运行目录位于：

    D:\zhoumeng\Graduate\experiments\unsw-kd-时间戳\
        experiment\REPORT.md
        experiment\summary.json
        packages\UNSW_KD_时间戳.zip
        packages\UNSW_KD_时间戳.zip.sha256

把正式运行目录中 packages 的 **ZIP与同名SHA文件** 一起放回归档。REPORT.md是简明结果，summary.json、模型、训练日志、逐行事件与成本明细都已包含在ZIP中。

执行出错时，脚本保留已有工件并尝试生成失败包；回传失败包，不重复开跑。若断电或进程被强制结束，等进程结束后只补包：

~~~powershell
.\scripts\RUN_UNSW_KD.ps1 -LabRoot $labRoot -PackageOnly '刚才打印的工作目录'
~~~

补包不启动训练，也不覆盖已有输出。输入路径发现前的错误尚无工作目录，回传终端错误即可。

## 可选的冻结目标文件

默认从原缓存重建目标。如果环境的浮点末位导致目标SHA不一致，保留失败包供核查；不要跳过校验。入口支持 FrozenTargets 参数读取已经冻结的目标NPZ，整包必须匹配协议中的SHA，不能替换教师或目标。
