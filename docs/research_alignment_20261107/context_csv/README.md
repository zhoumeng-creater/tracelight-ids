# 2026-11-07 原始 CSV 与阈值入口核查

实验室交付版见[完整命令](../../CONTEXT_CACHE_LAB_RUN_20261107.md)、[交付核查](delivery_checks.json)与[最终冷进程报告](delivery_synthetic_report.json)：**120 passed、0 skipped**，固定的 41 项 Windows/Python 3.12 依赖解析通过。新增 NF-ToN-IoT-v3 通信组划分、官方摘要核验、自动配置登记和环境安装入口。来源／字段／发布摘要见[数据来源记录](dataset_source.json)，权重身份见[固定元数据](../../../configs/context-cache-checkpoint.json)。

本轮交付见[实验室指南](../../CONTEXT_CACHE_CSV_LAB_20261107.md)和[配置模板](../../../configs/context-cache-lab-minimal.example.json)。实现复用固定 TabICLv2 2026 单视图预处理及普通 QCOC 2026 适配器；本机所有数据和模型权重均为合成／随机初始化。

- [端到端合成报告](synthetic_csv_report.json)：32 条支持、199 条正常校准、80 条开发评价、M=4、α=1%；文本标签映射、支持常量移除和各自阈值绑定通过。完整 KV／QCOC 均在独立进程中，仅靠各自四文件包和无元数据的原始特征 CSV 逐位复现分数、决策。
- [回归记录](checks.json)：110 passed、0 skipped，22.01 秒；Ruff 通过。覆盖预处理与官方单视图一致、support-only 拟合、标签映射幂等、跨角色隔离、有限阈值最小样本边界、严格阈值／并列／正无穷端点、错误配对、哈希绑定，以及前一阶段 KV／编译／冷加载回归。

本机运行目录为 `local/context-csv-synthetic-20261107-b/`，包含输入、完整比较报告、双方部署包及冷进程预测；目录由 Git 忽略。此前 `-a` 运行保留，`-b` 加入原始文本标签并验证最终实现。当前快照含 11 个适配器源码文件的 SHA-256，见合成报告；前一阶段检查记录保持原样。

首轮配置现已固定为 NF-ToN-IoT-v3、P8192／正常 C5000／E20000、M512、seed20261107；实验室命令从官方原始 CSV 自动生成并登记三个文件。上述较早 `-a/-b` 报告保留初始接线阶段，最终交付的本机运行目录为 `local/context-cache-delivery-synthetic-20261107/`。本轮没有真实检测效果、正式延迟或 RSS 结果。
