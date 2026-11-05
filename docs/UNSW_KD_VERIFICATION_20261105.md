# UNSW 普通 KD：实现与交付核验

2026-11-05。对应[四臂冻结协议](UNSW_KD_MINIMAL_PROTOCOL_20261105.md)。本轮新增专用入口，真实训练和正式成本测量尚未执行。

## 已实现

- F-only目标生成：固定教师原生概率、H/K/S/M、模式计数向量分层循环置换、逐类边际与覆盖检查。
- 同尺度混合BCE，float32网络与float64损失，全F累加后裁剪和AdamW更新，固定第40步。
- 共同step0只评分一次，四臂各评分一次终点；原始logit和float64阈值，不逐步C选点。
- 三组配对、逐族净TP、误报、固定阈值的完整模式集中度、H/K包体积及CPU成本。
- 原归档和prepared绑定、仅train/F/R/C读取、源码快照、训练日志、成功和失败打包。
- 科研验收不通过照常保存完整结果；技术故障保留已有工件并停止后续臂，不自动重试或追加条件。

入口为 scripts/RUN_UNSW_KD.ps1，实验室按[指南](UNSW_KD_LAB.md)操作。

## 已运行检查

本地 Windows，Python 3.12.4、PyTorch 2.6.0+cu126；合成训练使用CPU。

| 检查 | 结果 |
|---|---|
| 全套 pytest | 1,193通过、22跳过；137.75秒 |
| 新增检查 | 37项，包含训练6、验收13、集成4、实验室入口14 |
| Ruff：src、tests、scripts、workflows | 通过 |
| PowerShell入口与简化指南 | 入口及5段命令语法解析通过 |
| 实际网络分块梯度 | 对整批/归一化KL参数梯度，rtol=1e-5、atol=1e-7内一致 |
| 合成端到端 | 四臂、固定起点和终点、预算、目标、配对事件及完成文件清单通过 |
| 故障注入 | K中途失败后保留H工件及K流式日志，S/M未启动，无完成标记 |
| 分区读取检查 | 只验证和读取train；教师NPZ只取fit键；带陷阱的validation键未被访问 |
| 打包 | 科研负结果、技术失败、仅补包、哈希和拒覆盖检查通过 |

全套测试有一条既有loky物理核心数探测警告，随后使用逻辑核心数；没有测试失败。22项跳过没有计入通过数。

## 已保存工件的零拟合复核

读取原UNSW screen清单、F/R/C身份、保存模型元数据与F教师缓存，未调用真实模型transform/predict或训练：

- screen全文件清单、原source commit、binding列出的文件SHA全部一致。
- 原small为6,625参数，网络状态和预处理语义SHA均匹配。
- 教师F margin、q、模式donor和q_perm全部复现冻结SHA。
- 实际改变82,045行目标；类别条件q边际与同模式q一致性保持。
- 原prepared位置为 D:\zhoumeng\Graduate\data\prepared-v1。

本机没有原prepared的train.parquet，完整真实数据预检由实验室入口执行。本轮合成结果验证执行正确性，不代表KD真实收益；收益只按四臂实验回传判定。

## 版本与记录

新配置只更新“入口已实现、等待实验室”的状态字段，教师、数据、四臂、预算与科学验收条件保持原冻结值。原设计指纹保存在工作目录的 protocol_freeze_manifest.json；仓库中的 unsw_kd_protocol_manifest.json 更新为本次交付文件指纹。旧实验与负结果均保留。
