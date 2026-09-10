# CUMCM 2026 Problem B: Radio Interference Source Localization and Removal

2026 年全国大学生数学建模竞赛 B 题：无线电干扰源的快速自动定位与清除。

本项目统一管理几何建模、主动选点、全向与定向源搜索策略、模拟器交互、实验结果及论文材料。

## Project structure

| Directory | Purpose |
|---|---|
| `src/geometry/` | 方向约束、区域交集、直径与最小包围圆 |
| `src/localization/` | 定位状态更新与检测点选择 |
| `src/planning/` | 覆盖布点、路径与频道安排 |
| `src/strategies/` | 问题 3、问题 4 的完整策略 |
| `src/simulator_client/` | 模拟器接口、重试、状态与计时 |
| `tests/` | 几何计算与程序行为验证 |
| `experiments/` | 实验配置、运行入口及统计脚本 |
| `results/` | 演练记录、统计表和生成图像 |
| `资料汇总/` | 任务清单、建模思考与参考资料 |
| `论文/` | 正文、插图与最终稿 |
| `支撑材料/` | 正式日志及最终提交材料 |

## Development status

项目处于初始化阶段。算法、模拟器客户端和实验入口尚未实现；运行环境、依赖安装及执行命令随实现补充。

## Reproducibility

- 每次实验记录代码版本、参数配置、运行命令及结果位置。
- 论文图表和统计数值应能追溯到对应实验记录。
- 正式模拟器日志保留原始文件名与内容，并按问题和案例编码归档。
- 本地凭据放在环境变量或被忽略的凭据目录中。
