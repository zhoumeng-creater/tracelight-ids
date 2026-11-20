# 描述性复算与实现检查

从仓库根目录运行。输出放到新的工作目录，保留本次已发布工件。以下命令没有训练循环、阈值选择或候选模型真实评分；不会修改冻结四臂。Python 需要 NumPy，最后一项合成检查需要 PyTorch。

```powershell
python scripts/reference_evidence/coverage.py --output results/reference-evidence-local
python scripts/reference_evidence/necessity.py --output results/reference-evidence-local
python scripts/reference_evidence/geometry_distance.py --geometry docs/packet_interaction_20261119/connection_geometry.jsonl.gz --output results/reference-evidence-local/geometry
python scripts/reference_evidence/gap_structure.py --geometry docs/packet_interaction_20261119/connection_geometry.jsonl.gz --output results/reference-evidence-local/gap_structure_evidence.json
python -m scripts.reference_evidence.check_paired_reference_control --output results/reference-evidence-local/paired_reference_control_checks.json
```

前两项还读取已有动作及标签依据表，以保留可信正常与 `one_of` 多候选；按已开放来源筛选，不使用 C 或保留来源。固定几何项仅在方向序列一致的参照中比较字节比率，不是无缺口 DTW 的真实评分。结构项只检验合法路径存在、精确还原和重复等价组，不计算真实检测分数。

复算 JSON 的连接 ID、计数、关系和数值应一致。部分 gzip 输出包含生成时间，重新压缩后的文件哈希可以不同；发布文件的原字节校验见 `manifest.json`。中文报告由分析整合而成，不由这些脚本重新生成。

`paired_reference_control.py` 是独立普通对照，复用旧 `RelativeCost` 和相同 α、β；没有真实数据装载、训练或实验室批次入口。旧 `stage_design/reference_model.py` 不变，继续记录其缺口退化。
