# 复算扫描行号

实验室入口直接使用本次发布的两份扫描行号。只有核查这些行号来源时，才需要运行本目录的脚本。整套复算使用 Python 3.12、requests、numpy、pandas、pyarrow 和仓库本身的依赖；不拟合分类器。

## 数据位置

默认将派生文件和分段下载写到仓库下的 `local/capture-label-audit/`。将 UQ 的官方压缩包放到 `data/raw/nf-cse-cic-ids2018-v3/NF-CSE-CIC-IDS2018-v3-official.zip`。该 ZIP 为 780,113,687 字节，SHA-256 为 `7399bdef73d29875ab388a53e2ae7e592d1952c4ef07ae6de195ba133823d97f`。若已有实验室下载，设置 `TRACELIGHT_CIC2018_UQ_ZIP` 指向已有 ZIP；设置 `TRACELIGHT_LABEL_AUDIT_DIR` 可更改派生物的工作目录。两项均支持绝对路径。[UQ 数据页](https://rdm.uq.edu.au/files/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22)

## 运行顺序

以下命令从仓库根目录运行。每条命令成功后再运行下一条，使用普通 Python 模式，以启用脚本中的断言。

```powershell
Set-Location -LiteralPath 'docs/capture_validation_20261111/reproduce'

python audit_cic2018_nf.py
python extract_cic2018_native_endpoints.py
python extract_cic2018_packet_probes.py

python cic2018_corrected_range.py
python cic2018_corrected_rows.py
python cic2018_corrected_join.py

python cic2018_scanner_pcap_range.py
python cic2018_scanner_packets.py
python verify_labels.py
```

前三步核验 UQ 全量文件，再保留两个扫描日期的端点与单包 SYN。接着从修订版 ZIP 中读取两份日 CSV 的压缩字节区间，共 2,229,230,952 字节。按有向五元组、首包毫秒要求两侧唯一，再核对末包毫秒及双向包数。仅保留修订版真实标签为 NMAP、且符合作者时间和源目规则的行。[修订团队数据与标注规则](https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html#infiltration) · [标注源码版本](https://github.com/GintsEngelen/CNS2022_Code/tree/f0ce502818e59e6cd062720ab2286c5ff6f2bdec)

PCAP 步骤只取这两天两台扫描源主机的五个分卷，合计 647,267,773 个压缩字节。脚本核对原包唯一身份、IPv4 总长度、SYN 位及 TCP 通告窗口，并将同一作者事先标注的 NMAP 活动范围用于攻击归属。两种证据来自同一次采集的不同工件，不能视为两个采集环境。[UNB 采集说明](https://www.unb.ca/cic/datasets/ids-2018.html) · [AWS 原始包登记](https://registry.opendata.aws/cse-cic-ids2018/)

## 验证输出

`results/cic2018_verified_nmap_rows.npy` 含 25,496 个修订流匹配行，`results/cic2018_packet_nmap_rows.npy` 含 25,446 个逐包核验行。文件均为升序唯一的一基 UQ CSV 数据行号（表头不计），两集合无交集。`verify_labels.py` 同时核对数组的 dtype、形状、顺序、互斥性、归档摘要和实验室入口所用文件。[修订流证据](../matched_scan_evidence.json) · [逐包证据](../packet_scan_evidence.json)

如果已有两份复算结果，将 `python verify_labels.py --results 已有结果目录` 指向它们即可。源数据可按官方地址及登记区间恢复，审计结束后可清理本次下载产生的 `.deflate` 临时文件和 `.received` 进度文件。
