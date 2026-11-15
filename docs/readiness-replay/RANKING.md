# 已封存分数的排序上界诊断

`ids.readiness_replay.ranking.rank_diagnostics(features, scores, truth, scopes, source, group, output)` 读取已经封存的分数和评价标签，返回四个诊断文件路径。它不训练模型、不改预测文件，也不向上线规则提供阈值。

每次调用只含一个来源、一个学习器。`row_id` 必须唯一，三个表的行顺序必须一致；表的 pandas 索引不承担身份含义。范围掩码是与输入行位置对应的布尔数组。三个模型分数或两个历史计数出现 NaN、正无穷、负无穷时直接报错，不通过剔除这些行改变正常分母。

## 共同阈值与同分

依次对 `score_raw`、`score_contact`、`score_quality`、`hist_dst_ports_60s`、`hist_dst_ips_60s` 单独诊断。每个 `truth.window` 与每个 scope 的交集单独使用正常标签。正常数为 $n$ 时，允许误报数为 $k=\lfloor n/100\rfloor$，将正常分数升序排列为 $z$，最低可行阈值为：

$$t=z_{n-k-1},\qquad \text{告警}\iff s>t.$$

这里下标从零开始。等于阈值的记录全部不报，不拆分同分组；因此即使还有误报额度，也可能无法再纳入一组同分记录。这个阈值对所有攻击类别共同使用，不按类别另选阈值。

对分数为 $x$ 的目标记录，任何能包含它的严格阈值必须小于 $x$，因此至少误报所有分数大于或等于 $x$ 的正常记录。`global_normal_ge` 就是这一最少误报数；只有该数不超过 $k$，目标才能被预算内的某个共同阈值包含。`budget_excess` 是超出的正常记录数。另存 `global_normal_gt` 和 `global_normal_eq`，区分严格排在前面的正常与同分正常。

正常数为零时，计数为零仍可保存，但阈值、FPR、预算资格和目标可包含性留空，状态为 `no_normals`。这不构成零误报结论。

## 阶段与服务条件

调用方提供 `all`、`startup`、`after`、`original_startup`、`original_after`。`startup/after` 使用修复后的就绪边界；`original_*` 用旧首小时边界。模块将这些范围逐一与 `truth.window` 交叉，不跨日期汇总正常池。空交集保留无正常的阈值记录。不同 scope 可能重叠，不相加为独立样本。

目标的同服务参照只匹配 `ip_protocol` 和 `resp_port`。`service_normal_gt/ge/eq` 使用该服务正常记录；`service_cap` 按该服务分母另算 1% 整数预算，用来解释同服务重叠。它不替代全日正常预算。`service_supported=False` 表示没有同服务正常记录，此时服务阈值可包含性与超额留空。

## 输出

| 文件 | 一行的含义 |
|---|---|
| `ranking_thresholds.csv.gz` | 来源、学习器、日期、scope、单个分数的正常分母、预算、严格阈值、误报、边界同分数和未用额度 |
| `ranking_families.csv.gz` | 同一阈值下的 family、原生类别、证据级别、曝光状态与检出数；正常记录的告警计 FP，未决和混合标签不计 TP |
| `ranking_targets.csv.gz` | 每条稀有目标在每个 scope、每个分数下的全局/同服务排序计数；完整保留 UID、作者事件、原生标签、证据、曝光和数值输入/三个模型分数 |
| `ranking_summary.json` | 输入规模、分数及范围列表、诊断用途和解释规则 |

稀有目标按 family 精确纳入 `ExternalC2`、`C&C`、`PureCC`、`C2`、`Download`、`FileDownload`，不把 `OtherAttack` 中的 botnet 名称自动解释为纯控制。所有其他 family 仍进入类别汇总。稀有目标不只取前几条；可用 `(source, group, row_id)` 与评价模块的 `rare_rows.parquet` 联接。数值特征与真值字段重名时，特征加 `feature__` 前缀；上下文保留名与真值重名时，真值加 `truth__` 前缀。

这些阈值依赖评价期正常真值，只说明固定分数的排序还能达到什么程度。五个分数、各学习器和各范围分别保存，不能事后选其中最好的组合当成上线策略性能。
