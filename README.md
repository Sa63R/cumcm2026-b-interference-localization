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

**T01 进行中：客户端、规则文档及本地协议测试已完成，官方模拟器演练待环境就绪。** 当前 30 项测试通过；问题 1—4 的定位与搜索算法尚未开始。

- [任务清单与进度](资料汇总/B题任务清单与重要性权重.md)
- [T01 规则与接口说明](资料汇总/T01规则与接口说明.md)
- [T01 验收记录](资料汇总/T01验收记录.md)

## Quick start

已验证环境：Linux、Python 3.10.12。客户端运行仅使用 Python 标准库；测试依赖版本记录在 `requirements-test.txt`。

在仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e . -r requirements-test.txt
.venv/bin/python -m pytest -q
```

运行本地协议演示，结果保存在新建的 `results/t01/local-时间戳/` 目录：

```bash
.venv/bin/python -m experiments.t01_protocol_demo
```

演示使用两个固定测试源，覆盖全部五种检测/清除结果，并注入一次清除成功后的响应丢失。测试替身只用于协议与程序行为检查；它不是官方模拟器，也不能用于评价搜索算法或代替官方演练。

## Official practice connection

1. 按附件下载官方模拟器，确认运行平台；在模拟器内自行注册、登录并完成联网与时间校验。
2. 在模拟器界面选择 **“问题 3 演练测试”**，等待倒计时结束且接口就绪。
3. 在运行模拟器的同一设备上执行以下命令，将 `YOUR_TEAM_ID` 替换为当前登录参赛队号：

```bash
.venv/bin/python -m simulator_client check
.venv/bin/python -m simulator_client smoke-practice --robot-id YOUR_TEAM_ID
```

`check` 仅检查 TCP 端口，不发送动作。`smoke-practice` 执行附件 2 中的固定动作示例并退出，生成 JSONL 请求日志和 JSON 汇总；它不运行自动搜索。默认地址为 `http://127.0.0.1:2026`，可通过 `--base-url` 修改。

HTTP 接口没有查询或切换演练/正式模式的功能，因此运行前需要在官方界面确认演练模式。官方客户端未就绪时，不要启动正式测试来调试接口。

## Client behavior

- `SimulatorClient` 提供 `enter()`、`measure(position, channel)`、`clear(position, channel)`、`exit()`；位置可以是 `(x, y)` 或 `Position(x, y)`，方法返回官方 JSON 响应。
- 四个接口串行调用；只有通过校验的已接受响应会更新位置、频道、源状态与虚拟时间。`no_signal` 不会把已有目标标为不存在。
- 网络重试保留原请求 ID 和内容。若抛出 `OutcomeUnknown`，保留当前客户端对象，通过 `retry_pending()` 恢复；未确认前禁止发送新动作。若测试已结束，保留日志并在模拟器界面核对。
- 即使后续重试被拒绝，也不会抹去先前动作结果未知的状态。`/enter` 的剩余时限锚定首次发送时间，重试不会延长运行预算。
- 每次运行使用新客户端对象和新日志文件；`close()` 只关闭本地记录，正常结束测试应显式调用 `exit()`。

## Reproducibility

- 每次实验记录代码版本、参数配置、运行命令及结果位置。
- 论文图表和统计数值应能追溯到对应实验记录。
- 正式模拟器日志保留原始文件名与内容，并按问题和案例编码归档。
- 本地凭据放在环境变量或被忽略的凭据目录中。
