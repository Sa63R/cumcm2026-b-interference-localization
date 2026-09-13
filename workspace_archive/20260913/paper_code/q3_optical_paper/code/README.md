# 代码入口

从上一级的 [README.md](../README.md) 开始。单例运行入口为 `run_optical.py`，五方法配对批测入口为 `runner.py`。

提前光学的实现为 `vendor/policies/optical_experiment.py` 中的 `ExperimentalOpticalAgent`。`vendor` 保留原始源码以便核对已有实验的 SHA-256；其中历史脚本的 `main()`、原始 README 和旧注释并非本代码包的运行指南。当前入口只加载所需策略定义，使用 `vendor/simulator/jammers_local/core.py` 中的本地复原引擎，不执行历史 ToySimulator 实验。

`runner.py`、`summarize.py`、16 份冻结源码以及已有结果均与原批测副本逐字节一致。`run_optical.py` 是交接时增加的便捷入口。

结果报告：[本地模拟器前五名结果.md](本地模拟器前五名结果.md)。
