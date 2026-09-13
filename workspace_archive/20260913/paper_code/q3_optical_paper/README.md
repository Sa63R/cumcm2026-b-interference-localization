# 第三问：提前光学策略与论文复现代码包

主方法是 **v3 提前光学（`optical`）**，附带另外四种对照方法、可运行的本地复原模拟器核心，以及完整实验记录。论文作者先读 [论文写作说明.md](论文写作说明.md)，查数值时看 [结果报告](code/本地模拟器前五名结果.md) 和 [逐例成绩](code/逐例成绩.md)。

本包结果属于**本地复原仿真**，未完成与官方程序逐输入输出对照，不能标成官方正式测试分数。

## 找代码

| 内容 | 文件及符号 |
|---|---|
| 提前光学主方法 | [optical_experiment.py](code/vendor/policies/optical_experiment.py)，`ExperimentalOpticalAgent` |
| 单案例运行入口 | [run_optical.py](code/run_optical.py)，默认 `optical` |
| 五方法配对批测 | [runner.py](code/runner.py)，`make_agent` / `run_one` / `main` |
| v3 基础策略 | [q3_v3.py](code/vendor/policies/q3_v3.py)，`FastAgent` |
| v2 与可行域细化 | [q3_optimized.py](code/vendor/policies/q3_optimized.py)，`OptimizedAgent` |
| 几何与最小包围圆 | [q3_base.py](code/vendor/policies/q3_base.py)，`clip` / `wedge` / `mec` |
| 位置联合设计 | [new_methods.py](code/vendor/policies/new_methods.py)，`PlanningAgent`，本批 `scenario=False, future_cover=True` |
| 本地场景、反馈和计费 | [core.py](code/vendor/simulator/jammers_local/core.py)，`Scenario` / `Session` |
| 审计及统计 | [summarize.py](code/summarize.py)，`audit` / `ci` / `main` |

`vendor` 中保留了历史文件，供代码核对。**请通过本包的 `run_optical.py` 或 `runner.py` 运行**；直接运行 `optical_experiment.py` 会进入历史 ToySimulator 实验，不能复现这里的 2000 例成绩。当前入口加载原策略类和函数，避开历史 `main()`，所以不需要 pandas。

## 安装与第一次运行

实测环境：Python 3.12.14、NumPy 2.3.5、Numba 0.67.0、llvmlite 0.49.0，macOS ARM64。其他系统的运行时间可能不同；本次交接检查在 macOS 完成。

解压后进入本包根目录。在 macOS/Linux 终端执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -X utf8 verify.py --reproduce
.venv/bin/python -X utf8 code/run_optical.py --output reproduced/optical-case0.json
```

Windows PowerShell 对应命令：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -X utf8 verify.py --reproduce
.venv\Scripts\python.exe -X utf8 code/run_optical.py --output reproduced/optical-case0.json
```

首次运行会编译 Numba 内核，等待时间不等于策略每例计算时间。`-X utf8` 用于正确读取中文文件。程序不需要 UTM、账号、原电脑目录或外部服务；安装依赖需要下载对应 Python 包。

`verify.py` 核对整个交接包的文件摘要。加上 `--reproduce` 后，会重新生成第 0、999、1999 号地图，运行全部五种方法，并核对与存档的请求、业务响应、虚拟微秒总时长是否一致，忽略现实时间戳。

## 运行与复现

以下以已配置依赖的 Python 为例，将 `python` 替换为上面虚拟环境的解释器路径即可。所有新结果放入 `reproduced/`；输出已有内容时程序拒绝覆盖。

```bash
# 换种子；也可通过 --method v3 / v2 / v3_origin20 / future_cover 换方法
python -X utf8 code/run_optical.py --seed my-new-case --output reproduced/new-case.json.gz

# 先跑 5 个共同案例，每例比较全部五种方法
python -X utf8 code/runner.py --cases 5 --workers 1 --seed-prefix paper-smoke- --out reproduced/smoke5

# 复现原 2000 例主批：10000 次策略运行
python -X utf8 code/runner.py --cases 2000 --workers 2 --seed-prefix q3-top5-holdout-20260912- --out reproduced/paired2000

# 复现串行计时：50 例，每法重复 2 次，共 500 次
python -X utf8 code/runner.py --cases 50 --workers 1 --repeats 2 --seed-prefix q3-top5-timing-20260912- --out reproduced/timing50

# 审计已有存档并重新生成 code/ 下的统计报告
python -X utf8 code/summarize.py
```

`summarize.py` 针对 2000 例主批及 50 例计时批，不能用于上面的 5 例小批。可传 `--main reproduced/paired2000 --timing reproduced/timing50` 汇总完整重跑结果，但它会重新写入 `code/statistics.json`、两份 Markdown 报告和输入批次的 `validation.json`；需保留原包时请在副本操作。新机器的现实运行时间正常会变化，变化后的报告也不会再通过原交接摘要核对。

## 数据与可引用结果

- `code/results/paired2000/results.jsonl`：10000 条主批逐轮指标；2000 张共同地图，每张五种方法。
- `code/results/paired2000/runs/*.json.gz`：每轮完整地图、请求响应、结束状态及评价指标。
- `code/results/timing50/`：另外 50 张共同地图 × 5 方法 × 2 次串行计时的 500 轮记录。
- `code/results/smoke5/`：原始 25 轮接入试跑，未合并到主结果。
- `code/statistics.json`：均值、区间、分位数、相对 v3 的配对差值、源数分层统计。
- 各结果目录的 `metadata.json`：种子前缀、环境、并发数、命令和原源码摘要；`validation.json`：批次审计结果。
- `code/source_manifest.json`、`code/delivery_manifest.json`：原实验源码与交付记录摘要。里面的绝对路径仅是历史来源，运行只用包内相对路径。
- `code/selection.json`：五方法的候选筛选来源。此前 184 例官方演练**未同图配对**，此文件只解释为何选这五种，不作为本批排名依据。
- `code/validation_setup.json`：原实验的模拟器测试和协议适配测试摘要；原工作区的官方接入测试脚本未纳入本包。交接时新增检查见 `HANDOFF_VALIDATION.json`。
- `MANIFEST.json`：本包所有交付文件的 SHA-256 和大小（清单本身除外）。

五方法均在主批 2000 例全部清除。提前光学的案例等权平均为 **239.261 秒/源**，v3 为 **240.892 秒/源**；同图平均差值 **−1.631 秒/源**，95% 配对 bootstrap 区间 **[−1.715, −1.546]**。提前光学主批有 **1616 次失败光学尝试**，这些失败及其费用均已计入，不能删去。

上述“秒/源”是行动虚拟时间。提前光学另测的本机平均执行时间为 **0.0567 秒/例**，两种时间不要混用。
