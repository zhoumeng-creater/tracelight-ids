# 第六轮运行：蒸馏机制、低误报排序与近期模型

本轮是 research v2 的开发研究。第五轮正式 test 已经使用，新入口不读取 calibration/test，也不提供新的最终测试命令。阅读 [研究设计](STAGE6_RESEARCH_DESIGN.md)、[固定协议](STAGE6_PROTOCOL.md)及[近期模型身份](RECENT_METHODS.md)。旧实验目录和最终成绩保持原样。

## 独立研究环境

先复用已通过基础检查、PyTorch 2.6.0 CPU/CUDA 可用的项目环境。新增的官方 TabM 依赖写入独立 venv，不升级原环境。下面是 Windows 示例；路径由实际实验室布局确定。

```powershell
$researchEnv = Join-Path $env:TEMP 'tracelight-research-v2'
python -m venv --system-site-packages $researchEnv
$researchPython = Join-Path $researchEnv 'Scripts\python.exe'
& $researchPython -m pip install --no-deps -r requirements-research.txt
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $researchPython -c "from ids.recent_baselines import recent_dependency_versions; print(recent_dependency_versions())"
& $researchPython -m pytest -q
```

`--system-site-packages`复用原环境的较大torch包，新增包隔离；完整环境仍需记录。不要在实验运行中更新源代码、依赖或配置。没有TabM依赖时明确失败，不用普通MLP代替。Linux对应使用`python -m venv --system-site-packages`及`bin/python`，其余Python命令相同。

## 三个不同的运行层级

每次使用不存在的新输出目录；完整工件在运行成功后写入`RESEARCH_COMPLETE.json`。示例prepared路径均指已经审计的本机数据，不重新prepare或换分区来寻找更好结果。

```powershell
# 仅展示预算和验证开发文件身份，不拟合，也不读取保留分区。
& $researchPython -m ids.research --data-dir 'D:\TraceLightLab\data\prepared-v1' --config configs/research-v2.yaml --output 'D:\TraceLightLab\research-v2' --device cuda --dry-run

# synthetic prepared：13个学生、4个教师拟合、seed17、最多2epochs；只验证工程。
& $researchPython -m ids.research --data-dir 'D:\TraceLightLab\smoke\prepared' --config configs/research-v2-smoke.yaml --output 'D:\TraceLightLab\research-v2-smoke' --device cpu

# UNSW真实开发小试：最多12000/6000行，整组取样，seed17、最多20epochs。
& $researchPython -m ids.research --data-dir 'D:\TraceLightLab\data\prepared-v1' --config configs/research-v2-pilot.yaml --output 'D:\TraceLightLab\research-v2-pilot' --device cuda

# 完整开发研究：完整train/validation、5seeds、65学生+15OOF+5全train参考拟合。
& $researchPython -m ids.research --data-dir 'D:\TraceLightLab\data\prepared-v1' --config configs/research-v2.yaml --output 'D:\TraceLightLab\research-v2-full' --device cuda
```

两数据环境需要独立运行并保留各自的字段契约；CIC不能直接套用UNSW限量pilot：大关联组可能比整个预算更大，排除它会改变任务。正式配置无行数上限，仍不保证真实网络泛化。三组配置不是同训练预算，不跨层级比较分数。

## 验收与恢复

每个同结构MLP条件共享seed、初始化与批次顺序；配对报告会检查字段、训练预算与初始权重身份。所有MLP统一使用验证hard BCE早停，损失分量、门控质量、选中pair数和逐家族恢复／新增漏检均保留。TabM遵循官方逐成员训练与概率均值推理；本项目固定小预算适配不等于复现作者最优成绩。

输出包括：

- `plan.json`：运行前配置、代码指纹、数据清单、完整/子样本家族支持、身份与环境。
- `teachers/s<seed>/`：折内模型、折分组证据、逐行OOF分数及全train教师参考；参考不生成自己的训练目标。
- `students/<variant>-s<seed>/`：部署包、逐epoch损失、验证逐条分数、阈值与家族指标。
- `summary.json`、`report.html`：同seed控制、恢复检出和新增漏检的双向汇总；报告可离线打开。
- `RESEARCH_COMPLETE.json`：矩阵覆盖和逐文件SHA-256完成清单，只表示开发链完成。

对已完成的同一命令增加`--resume`只核验，不重新训练。当前不支持epoch或部分研究链续训；不完整目录保留，使用新输出目录重新执行，禁止覆盖。完整性检查拒绝源代码、数据、配置、设备或工件变化。

先核验合成链与真实pilot的数值、配对、支持数和成本，再决定完整开发研究的排队安排。性能优劣须由五种子及机制分析回答，单次pilot不判定有效。CPU三轮部署计时、真实应用验收和新来源最终确认仍需各自独立工件，不能凭本轮完成标记自动宣布完成。
