# R30 模型与训练实现及核验

2026-11-17。固定版本`model-training-redesign-r30-v1`。已按[设计](MODEL_TRAINING_REDESIGN_20261117.md)完成独立实现、合成合同核验和真实档案输入预检；真实训练留在实验室。运行入口及完整命令见[实验室说明](MODEL_TRAINING_REDESIGN_LAB.md)。

## 逐项落实

| 设计项 | 实现位置 | 已核验内容 |
|---|---|---|
| A+B监督与完整V | `src/ids/model_training_redesign/data.py` | 30个原始单元SHA、作者UID标签、时间与身份；F19904/V21450；原B输入逐值匹配；C5511与完整E17059保持原值 |
| 五数值PLE及固定协议槽 | `preprocessing.py`、`training.py` | F独占填补、缩放、分箱；两臂只改变显式协议槽；主候选冷载后替换协议码仍输出相同合成分数 |
| 四组均值目标与循环批 | `sampling.py`、`training.py::four_group_loss` | 每批四组等量、周期内无放回/跨周期续接；无额外权重；合成loss值和梯度对应四组成员BCE公式 |
| 配对随机性 | `training.py`、`runner.py::check_pairing` | 两臂初始化、F预处理、PLE、采样种子相同；共同更新前缀draw/dropout状态相同；提前停止长度不同不会被误判 |
| 纯排序选轮 | `selection.py` | NLL下降不换best、不重置patience；面板分离不替代全V分离；首次全V严格分离锁当前点，避免数值指标平局保留旧点 |
| 选中即交付 | `training.py`、`runner.py::freeze` | best.pt与model.pt字节相同；预处理/PLE与权重一起保存；每轨迹fit_calls=1、final_refits=0；冻结时再核对 |
| 全部冻结后评价 | `runner.py` | 六次拟合先完成；训练worker只读F/V；无P路径；全部六工件及配对合同通过后才加载C/E进行新评分 |
| 原标准与身份比较 | `evaluation.py` | 原C序位与严格大于规则；三seed机制/效果/实现AND；15cell、B72、旧合规强参照和配对对照保护；完整新增/丢失TP身份 |
| 独立成本 | `costs.py` | 端到端、仅预处理、预变换输入仅模型三条实际调用路径；冷载与预变换CPU分数一致；GPU结果独立描述 |
| 入口、复用与回传 | `protocol.py`、`runner.py`、PowerShell包装器 | 固定设计/输入哈希；完整阶段复用、半成品拟合拒绝重试；离线Verify/Pack；成功/失败证据包 |

## 验证结果

通过真实PowerShell入口的`CheckOnly`执行本轮完整套件：**96 passed，0 failed，0 skipped**。其中数据13项、评价23项、训练13项、执行编排6项、成本3项、Windows PowerShell/pwsh入口38项。Ruff通过。

模型检查使用合成小数据和微型官方TabM，CPU上执行三个拟合：两臂配对及工件协议策略篡改检查。它验证实际作者包、反传、选轮、随机流与冷加载的接线，不是本任务的实验效果。固定作者依赖为tabm0.0.3、rtdl_num_embeddings0.0.12、torch2.6.0+cu126；本机未执行CUDA训练。

通过同一包装器的`InputsOnly`读入真实归档，生成输入与源代码快照；`VerifyOnly`离线核验210份文件通过。该过程没有拟合预处理、构建网络或计算新分数。保留17份R29来源绑定、30单元原始流与UID审计、63996条UID映射以及7份共119413行原参照判决。

独立审查修正了五处实现细节：配对元数据接口统一；首次严格分离优先于数值排序平局；选中与末次CPU/CUDA RNG实际状态落盘；部署推理去掉训练RNG审计开销；冻结明确核验每轨迹0次refit。当前不需要通过重训旧轮次补齐数据。

机器可读记录见[verification.json](model_training_redesign_implementation_20261117/verification.json)，包含源码哈希、测试结果及输入预检绑定。原R29源码、注册、已有判决和上一轮design.json保持原值；本轮新增独立registration，未把设计阶段的`execution_enabled=false`改写成历史上已可执行。

## 实验室交付边界

本版可按运行说明进入六条固定轨迹。尚未取得R30真实D检出、误报或设备成本结果；这些由固定流程产生并按原设计判断。成功运行会输出完整证据，即使`adoption_pass=false`也保留负结果，不加seed、不选冠军、不换阈值或补P重训。
