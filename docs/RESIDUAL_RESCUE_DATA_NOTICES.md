# TraceLight 补漏实验输入来源

交付包含从公开研究数据提取的连接统计、连接身份与标签、派生核验，以及项目用这些统计训练的模型。包中没有原始抓包、连接载荷或原始主机日志。每个输入的 SHA-256 和归属见包内 data/manifest.json 与 data/data_audit.json。

| 来源 | 使用方式和来源条款 |
|---|---|
| Max Landauer 等，AIT Log Data Set | 使用 Santos 的既有主 HGB、Russell 的正常校准统计和 Fox 的历史开发统计。作者公布的数据条款为 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。AIT 派生统计和关联标注按相同条款提供。 [发布页](https://zenodo.org/records/19483937) · [作者论文](https://doi.org/10.1109/TDSC.2022.3201582) |
| Unraveled 作者团队 | 将作者已标注的一次采集转换为选定主机的完整连接统计，保留标签与观测范围。作者数据说明与代码仓库引用 [GNU GPL v3](https://gitlab.com/asu22/unraveled/-/blob/d2ea90055d82fa448ab20588a13e3ec8bfd74816/LICENSE)。归属于该来源的材料遵循上游条款。 [数据与处理说明](https://gitlab.com/asu22/unraveled/-/blob/d2ea90055d82fa448ab20588a13e3ec8bfd74816/data/README.md) · [作者论文](https://www.sciencedirect.com/science/article/pii/S1389128623001330) |
| DARPA Operationally Transparent Cyber 数据发布 | 使用已公开采集里的连接统计和项目既有证据分组。DARPA 将原始数据发布到公共领域。 [源档案与发布说明](https://github.com/FiveDirections/OpTC-data/blob/master/README.md) |

依赖文件保留各发行方的版权与许可。包内 scripts/residual_lab/wheels-win-cp312.tsv 记录每个文件的版本、下载来源和 SHA-256。项目代码沿用仓库的 LICENSE。各来源材料适用上述条款。
