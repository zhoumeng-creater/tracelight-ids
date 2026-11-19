# 固定 Zeek 导出运行环境

2026-11-18 核对官方来源。固定 **Zeek 8.0.11 LTS**，不跟随 latest。

- [官方安装说明](https://docs.zeek.org/en/current/install.html)：官方提供 Linux 包和容器；本轮选择已有 Ubuntu 24.04 amd64 WSL 私有解包，也接受精确同版原生 Zeek。
- [Zeek 官方 OBS 包索引](https://download.opensuse.org/repositories/security:/zeek/xUbuntu_24.04/Packages.gz)：`zeek-8.0-core_8.0.11-0_amd64.deb`，10,883,424 字节，SHA-256 `0b3e180d7677d7009b93fe13091d44079be7fd91e7b1a3cc7d9bc4dd0f337714`。
- Ubuntu 运行库来自 `archive.ubuntu.com`；`runtime-packages.json` 固定每个包的版本、URL、长度和 SHA，依据 Ubuntu 24.04 的 apt 元数据并实际校验下载字节。
- [固定版本 Conn 源码](https://github.com/zeek/zeek/blob/v8.0.11/scripts/base/protocols/conn/main.zeek)：原 IP 字节取自 IP 声明长度；duration 不覆盖全部尾部非生产性 TCP 包；conn 写出发生于 `connection_state_remove`。
- [Zeek 事件文档](https://docs.zeek.org/en/v8.0.11/scripts/base/bif/event.bif.zeek.html)：`Pcap::file_done` 标记输入文件结束；`net_done` 发生于最终事件/状态清理之前。

`export.zeek` 在 Conn 基础处理之后、写日志之前记录 `network_time()`，分别保留 `contract_after_input_eof` 和 `contract_during_shutdown`。这是离线执行的逻辑导出时刻，不是历史检测器送达时间。终止清理行不可冒充原时刻在线告警。

两个固定分支分别正常校验 TCP/IP 校验和及使用 `-C` 忽略校验和，均保留。不存在按检出或数据量挑选分支的逻辑。应用日志仅记录真实解析出的内容；不存在对应协议时明确记录缺失日志，不制造空结果来表示已核验。

环境准备只在项目 `local/ait-zeek-runtime` 下载和解包，不运行 apt 安装、sudo、服务启动或重启。需要已经存在 Ubuntu 24.04 WSL 或精确同版原生 Zeek；缺该基础环境时明确报错，不改动原训练环境。所有运行库缓存校验后可复用。
