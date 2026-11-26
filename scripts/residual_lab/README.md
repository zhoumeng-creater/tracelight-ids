# TraceLight 补漏实验 v1

Windows 10/11 x64，需安装 Python 3.12 x64。实验使用 CPU。ZIP 内有冻结源码、实验输入、四个历史比较模型和 Python 依赖包。运行阶段不需要联网。

1. 将 ZIP 完整解压到可写目录。建议使用路径较短的目录，如 `D:\TraceLightRescue`。
2. 双击 `RUN_RESIDUAL_LAB.cmd`。首次运行从包内安装固定版本的 CPU 依赖，后续运行复用该目录下的环境。
3. 运行完成后，回传窗口列出的 `returns` 目录内本次结果 ZIP 和对应的 `.sha256.json`。若实验报错且生成了带 `-failed` 的 ZIP，回传该 ZIP 和对应校验文件。完整日志保存在 `runs`。

从命令行运行 `RUN_RESIDUAL_LAB.cmd --check-only` 可安装依赖并检查冻结输入。此命令不拟合或评分。若系统检测不到现有 Python，在命令行设置 `TRACELIGHT_PYTHON` 为 64 位 Python 3.12 的 `python.exe` 完整路径，然后重试。

每次正式运行使用新的结果目录。退出后重新双击会从头执行本轮。实验设计和数据署名见 `docs`。包内 `lab-bundle.json` 记录输入与代码的哈希，`scripts/residual_lab/wheels-win-cp312.tsv` 记录每个依赖文件的来源与 SHA-256。
