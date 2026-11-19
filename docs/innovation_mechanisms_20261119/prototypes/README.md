# 机制原型与可复算结果

这三个脚本不导入检测模型，不训练、不评分、不选阈值。时序脚本只用 Python 标准库；另两个使用 NumPy。

从仓库根目录依次执行：

```powershell
python docs/innovation_mechanisms_20261119/prototypes/temporal_prototype.py
python docs/innovation_mechanisms_20261119/prototypes/check_distillation_algebra.py
python docs/innovation_mechanisms_20261119/prototypes/representation_mechanism_checks.py
```

输出 JSON 保存在各脚本旁。时序与蒸馏检查只用人工数组。表示检查另外读取仓库内冻结的 `coverage_basis_20261119/batch/rows.jsonl.gz`，仅统计 F_base/F_expansion 的可信正负记录；输出记录源 SHA。它不使用 C/V/外部开发标签选择条件。

- [时序更新与读取](temporal_prototype.py)：有序二阶统计、同批时钟、方向与支撑分离，以及过时、复制、切分、同形正常等反例。
- [蒸馏代数](check_distillation_algebra.py)：教师区间与 KL 的区别、区间内次序丢失、Jacobian 投影等价性、分位 stop-gradient 差异。
- [表示代数与 F 支持](representation_mechanism_checks.py)：正常 secant 硬投影的曲率和塌缩反例、精确条件配对支持。

脚本中的“通过”表示推导或反例如期出现，不表示检测器有效。它们不构成实验室训练入口。
