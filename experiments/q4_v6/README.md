# 第四问 V6：可运行源码与复原模拟器实测

本目录发布经过本机配对测试的保护版 V6，包括求解器、两组已训练树权重、复原模拟器、设备适配器及完整证据。在线求解只用 Python 标准库，无需 GPU、模型 API 或训练依赖。

在 300 个新随机场景上，各场完整任务时间除以源数后取平均：

| 方法 | 秒/源 | 相对 V4 的时间减少 |
|---|---:|---:|
| V4 | 456.959 | — |
| V5 | 452.892 | 0.890% |
| V6 | 450.967 | 1.311% |

V6 对 V4 的配对提速 95% 自助法区间为 0.88%～1.75%；204 场更快、96 场更慢，最差慢 36.94%。另有 70 场压力测试，合计 370 个不同场景、1,110 次完整运行全部全清。源码、模型和计划在实验前冻结，没有根据结果调整参数。

这是本地静态复原模拟器的结果。V6 尚无官方演练或正式成绩。另做的 20 场串行计算测试中，预载入后的平均墙钟为 V4 0.0229 秒、V6 0.4862 秒；计算成本与机器人虚拟任务时间是不同指标。详见[实测报告](V6_本地复原模拟器实测报告.md)。

## 运行一局新的本地测试

从仓库根目录执行，输出目录必须不存在或为空：

```sh
python3 experiments/q4_v6/run_local.py --method v6 --seed my-new-case --output results/q4/v6/my-new-v6
python3 experiments/q4_v6/run_local.py --method v4 --seed my-new-case --output results/q4/v6/my-new-v4
```

也支持 `--method v5`。同一个 seed 生成相同地图，但每条命令创建独立会话。输出包括全部动作、反馈、场景、真实清除数及完成证书。无需启动 UTM 或登录任何账号。

本地评测入口要求 **macOS/Linux、Python 3.10+**，超时保护使用 `SIGALRM`。纯 V6 求解器仅依赖标准库；本次没有验证 Windows 上的发布入口。

默认 V6 保留原模型、前 4 个固定站保护和最多 16 次沿途停测。评测适配器将 V4/V5/V6 的保守几何误差界统一设为 1.005°，与已有 V4 接口适配一致；物理检测和微秒计时由复原模拟器处理。原始源码文件没有修改。

## 接入设备

策略入口位于 `source_v6/q4_v6.py`：

```python
import sys
from pathlib import Path

root = Path('experiments/q4_v6').resolve()
sys.path.insert(0, str(root / 'source_v6'))
from q4_v6 import solve_v6, default_config

# device 只暴露 position、channel、move(point)、detect(channel)、clear(channel)。
# detect 返回 q4_baseline.Observation；clear 返回真实清除成功与否。
report = solve_v6(device, default_config())
```

`benchmark.Device` 已把这套设备接口接到本地四个业务接口。它将 move 与下一次 measure/clear 合并，clear 不切换 RF 频道。已有真实 `SimulatorClient` 也通过本地回环 HTTP 验证。纯策略不能接收真值、实际源数、噪声种子或场景内部对象。参考模型权重会相对源码目录解析。

V4 与 V6 历史代码使用同名顶层模块，应用中请在独立进程运行不同版本，避免导入缓存相互影响。仓库新增回归测试也使用子进程隔离。

## 验证

从仓库根目录运行：

```sh
# 包内原有 78 项测试，无第三方依赖
python3 -m unittest discover -s experiments/q4_v6/source_v6 -v

# 发布入口、已发布 V4 的动作一致性、冻结案例复现
python3 -m unittest discover -s tests -p test_q4_v6_release.py -v

# 从压缩包直接重放全部 1,110 个会话，不解压，不覆盖历史记录
python3 experiments/q4_v6/verify_results.py replay --output results/q4/v6/replay-check

# 使用现有客户端，在临时本机回环端口分别运行 V4/V5/V6
python3 experiments/q4_v6/verify_http.py --output results/q4/v6/http-check

# 单进程重跑同样前 20 场，校验动作与冻结数据一致
python3 experiments/q4_v6/verify_results.py serial --output results/q4/v6/serial-check
```

输出目录非空时拒绝覆盖。需要重跑同一检查时请另选目录。HTTP 检查只绑定本机随机端口，完成后关闭服务，不连接官方服务。

需要重现原始 370 场批量实验时，先复制本目录，在副本中将 `main/` 改名保留，然后运行 `python3 benchmark.py --phase main --workers 4`。原实验的 `benchmark.py`、`plan.json`、`source_manifest.json` 均按原字节保存；再次运行会核对冻结哈希。复用已公布地图属于复现，不是新的独立评测。

## 文件

| 路径 | 内容 |
|---|---|
| `source_v6/` | 原包运行与测试源码、默认与测试树权重；不含上游训练数据和研究附录 |
| `simulator/` | 本次评测使用的复原模拟器原始快照及历史说明 |
| `benchmark.py` / `run_local.py` | 冻结评测器与新场景入口 |
| `plan.json` / `source_manifest.json` | 全部冻结场景、评测约定及原始源码哈希 |
| `summary.json` / `paired_records.csv` | 各组统计、置信区间、逐场结果 |
| `main/sessions.zip` | 1,110 份原始会话 JSON，逐文件无损压缩 |
| `serial_runtime/` / `bound_sensitivity/` | 额外串行、误差界敏感性检查记录及压缩会话 |
| `http_validation/` | 原测试的回环 HTTP 动作、客户端日志及一致性结果 |
| `replay_verification.json` | 全部 315,961 条业务动作与微秒计时重放检查 |
| `worst_case_audit.json` | 最差退步例及路线数据 |
| `PUBLICATION_MANIFEST.json` | 发布整理时保持不变的文件及少量路径、压缩日志读取适配 |

`source_v6/README.md`、`source_v6/V6_REPORT.md` 和模拟器目录内文档按原包保存，属于上游历史说明，可能引用未随本目录发布的研究附录或原机器路径。使用本页命令运行。主实测报告中的 `main/sessions/` 在 Git 中对应 `main/sessions.zip`；会话 JSON 内容不变。

压缩日志全部来自本地复原环境，机器人标识是 `local-test`。本次发布不包含登录账号、密码或官方会话日志。
