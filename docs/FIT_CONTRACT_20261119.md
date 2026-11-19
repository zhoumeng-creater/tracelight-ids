# 独立拟合来源与固定比较

数据与运行入口已落实，详见[完整交付](fit_contract_20261119/README.md)和[实验室运行说明](FIT_CONTRACT_LAB_20261119.md)。

F 为 santos 的 32 条攻击、439 条条件正常连接；V 为 fox 的 17 条攻击、430 条条件正常连接；C 为 russellmitchell 的 25 条正常连接。943 条监督记录按稳定身份登记，完整链不跨角色拆分。两场均有控制、下载与同场正常业务。五份捕获的 109,512 条观察已进入当前统一表，其中 88,484 条声明本机视角形成完整背景历史；保留多义候选与未知标签。

本次修正了“缺失完整载荷就无法使用包头统计”的前置条件。所有类别统一采用经过验证的声明 IP 长度表示，原始捕获字节不变。旧 AIT 表保存在历史版本；当前消费表不混入旧、新 UID 的重复记录。

唯一实验是固定 HGB8/HGB27 两臂，F-only 拟合，各用 C 最大正常分数固定严格大于阈值，再做 V 与已开 OpTC 开发评价。真实拟合留在实验室工作站；本机已通过真实输入预检，未构造模型、拟合、评分或计算新阈值。这个比较判断历史是否提供额外信息，不构成原创算法验证。OpTC Meterpreter 与 AIT harrison 保留。

[逐身份注册](fit_contract_20261119/fit_contract/registration.json) · [统一表与背景](fit_contract_20261119/dataset/summary.json) · [真实输入检查](fit_contract_20261119/fit_contract/inputs_preflight/inputs_audit.json)
