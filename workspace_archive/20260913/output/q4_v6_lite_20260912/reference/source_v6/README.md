# B题第四问 V6：可运行默认版与完整实验

默认入口：**带前期搜索保护的V6**。不保护版在最终600例上平均更快，但出现超过100%的个例退步，作为研究选项保留。默认版也没有逐例优于V5的保证。完整报告在 `V6_REPORT.md`。

## 运行

需要 Python 3.10+。已附训练好的回归树权重，在线运行仅用标准库，不用GPU、语言模型API或训练环境。

```bash
python run_v6.py --mode v6 --scenario 1 --seed 241200000 --out LOCAL_ONLY_v6.json
python run_v6.py --mode v5 --scenario 1 --seed 241200000 --out LOCAL_ONLY_v5.json
python run_v6.py --mode unprotected --scenario 1 --seed 241200000 --out LOCAL_ONLY_unprotected.json
```

每个命令生成一个全新的本地环境。同一个官方会话不能先运行V5清掉目标，再拿剩下的环境运行V6作比较。

## 接入入口

```python
from q4_v6 import solve_v6, default_config

# device必须按附件1/2的真实协议实现，语义沿用原V5设备抽象。
# 只需暴露position、channel、move(point)、detect(channel)、clear(channel)。
report = solve_v6(device, default_config())
```

本包没有官方HTTP适配器，因为当前对话缺少附件1/2。没有猜测端点或字段。光学失败按3秒计时是本地假设，必须按官方协议核对。本地 `LOCAL_ONLY` 记录不能作为三次正式测试日志。

## 复现

```bash
python -m unittest discover -v
python certify_default.py --out regenerated_coverage.json

# 保留已有results_final，不覆盖已冻结的原始结果。
python validate_final.py --freeze --out reproduced
python validate_final.py --out reproduced --workers 5
python summarize_final.py --out reproduced
```

复用已公布种子是复现，不是新的独立测试。使用新种子基数时，freeze和运行均传入相同的 `--seed-base`；规模也必须与冻结计划一致。验证程序在开始、结束核对模型/源代码哈希，拒绝静默覆盖现有结果。

`minimum_stations=4` 是当前默认的保护阈值（包含原点）。设为0会恢复无保护组合；修改阈值、树权重或其他配置后，已有最终成绩不再直接适用。模型只影响动作选择，不删除未认证站点、不宣布概率很小的频道不存在。

## 文件位置

- `q4_v6.py`、`q4_guarded_solver.py`：默认求解器及保护条件。
- `q4_transit_critic.py`、`q4_route_critic.py`：可观测特征、沿途停测、树推理。
- `route_critic_extra.json`、`transit_critic_big_extra.json`：选定模型权重，已训练好。
- `q4_v6_unprotected.py`：首轮未保护版本，明确作为对照。
- `q4_v5.py`及原33个Python文件：不修改的基准。
- `results_final/`：最终600例、三策略1800次运行、冻结计划、逐例CSV、摘要与本地动作日志。
- `coverage_certificate.json`：21站的连续方格覆盖证书。
- `baseline_source_audit.json`：原V5源文件核对。
- `research/round1/`：初轮开发、全部离线训练数据/权重候选、布局失败实验、首次780例测试和冻结代码快照。
- `research/followup/`：后续补扫描的负结果、2至6站保护阈值开发对照，以及首轮极端例诊断。

首轮780例在发现严重退步后被用于诊断，因此不能再称为最终保护版的独立验证。最终600例另行冻结、另取新种子。附加分布也出现在离线训练中，不冒称未见分布泛化。详见报告。

研究附录中的训练脚本可选安装 NumPy、scikit-learn、LightGBM；部分布局搜索使用SciPy。它们不是运行已训练求解器的依赖。训练脚本请在 `research/round1/` 内运行，使相对数据路径一致；重训会改变权重，不能沿用旧冻结评测成绩。
