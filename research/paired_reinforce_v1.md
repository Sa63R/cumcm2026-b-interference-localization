# 同场景贪心基准的 REINFORCE 对照

此实验属于独立强化学习分支，不调用状态搜索。当前仅支持 v1/v2 的相同观测特征、候选动作与集合网络；改变的是训练估计量。动机是检查 PPO 的剩余成本 critic 是否因部分可观测性和场景难度差异，难以提供足够清楚的微小改进信号。它不是已经验证的解释，也不预先宣称新训练法更好。

每批固定当前参数。对每个训练场景分别执行一条随机策略轨迹与一条独立的当前参数贪心轨迹，得到包含所有兜底成本的总耗时 `C` 和 `b`。不清除完整则在真实成本之上额外加 360000 秒，行政时间中断则整对丢弃。两次执行从独立模拟器开始，基准不会接收随机轨迹的动作或历史；基准随机数也独立重置。任何真实场景信息都不输入网络，事后成功判定只影响奖励。

令 `A=(b-C)/1000`，按**每条轨迹**平均优化 `-A sum_t log pi(a_t|h_t)`。基准可以依赖初始隐藏场景，因为给定该场景与本批固定参数后，它不依赖待求梯度的采样动作；`E[b grad log P(trajectory)]=0`。所以在熵系数为零时，这是最小化期望总成本的无偏策略梯度估计量。实现不通过 `b`、采样概率或模拟器求导，不以不同轨迹的决策数重新加权。网络 critic 参数不参与价值回归。每批仅进行一次累积梯度更新，绝不对同一 on-policy 轨迹无修正地进行多轮 REINFORCE 更新。

默认另加 0.005 的平均每轨迹熵正则，作为探索启发式；不能将含此项的实际梯度写成纯耗时目标的严格无偏估计。不给 `A` 做当前批标准差归一化，避免将随机缩放与严格公式混为一谈。以小批次累积显存内梯度，但最终除以 episode 数。专门测试以三叶、变长度、可解析决策树核对期望梯度，再检查分块与整批参数更新一致。

这里的基准是每批当前策略的贪心版本，不是 Kool 等作者实现中用统计检验定期升级的冻结挑战者；两者应明确区分。同场景成对执行约增加一倍环境采样工作，训练日志分别报告采样与基准局数、采样/优化墙钟时间和完整成本。不能只报告 GPU 优化时间。

检查点沿用旧控制器的 `algorithm`/特征兼容标签，但 `state.trainer` 明确为 `paired-greedy-reinforce-v1`，它没有进行 PPO 更新。共同评估须使用 `research_rl.train_paired:run_reinforce_search`，返回结果会单列控制器兼容标签并正确标注训练方法。支持新目录显式迁移与同源码恢复；不覆盖其他训练目录。

```bash
PYTHONPATH=src python -m research_rl.train_paired --output results/rl/paired-v2-001 --initialize-from /path/to/frozen-v2.pt --feature-version v2 --device cuda --updates 1000 --pairs-per-update 32 --workers 4 --lr 0.0001 --max-wall-s 1800 --checkpoint-seconds 600 --deadline-utc 2026-09-11T06:00:00+00:00
```

评估仍只使用共同协议的验证集选型，最终测试保持封存。PPO、此训练法及无教师热启动的对照均需报告实际经历的采样量和计算预算，不能把算法名当作性能证据。

## 核实的原始来源

- Kool、van Hoof、Welling，ICLR 2019：[Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)。借鉴同实例贪心基准降低策略梯度方差的训练思路；论文处理的主要是已知节点组合优化，与本题未知源部分观测有区别。
- [作者 train.py](https://github.com/wouterkool/attention-learn-to-route/blob/master/train.py) 和 [reinforce_baselines.py](https://github.com/wouterkool/attention-learn-to-route/blob/master/reinforce_baselines.py)：核对 `(cost-baseline)*log_likelihood` 的符号、完整序列似然和基准更新规则。这里独立实现，未复制源码，不声称复刻论文架构。
- Engstrom 等，ICLR 2020：[Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729)：支持对实际训练实现做控制变量核查；其连续控制实验本身不能证明本题的 PPO 瓶颈或 REINFORCE 优越性。
