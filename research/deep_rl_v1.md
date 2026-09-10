# 第三问深度强化学习首版：实现与复现说明

本文件是实验设计及代码说明，不是论文正文。实现版本 `q3-candidate-ppo-v1`。
主分支、第四问、官方模拟器均未修改或调用。性能结论必须以实际训练及冻结评测为准。

## 实际学习什么

神经网络从观测历史的几何摘要和当前合法候选集合直接输出类别策略，控制：

1. 下一个覆盖扫描点，以及扫描与已发现源定位之间的交替；
2. 下一个处理的已发现频道，因此也决定源间移动和清除顺序；
3. 主动测量位置：已有中心/交叉方位探测点、中心两侧偏移、当前位置至中心的中点、未重复测量的当前位置；
4. 已经有几何清除证书的目标何时清除。

位置仍是有限候选集合；每个覆盖点内的整轮频道扫描固定为当前频道优先，然后按频道编号排序。最初原点扫描固定。几何外包集合、负观测约束、19.9 m 安全清除边缘和最多 16 源的终止规则沿用已验证模块。每源主动探测次数上限或总决策上限触发已有定位/光学兜底，完整计入奖励，并记录触发原因、动作数和虚拟耗时。网络没有连续任意坐标控制，也没有学习覆盖点内未知频道的排列。这些是首版清楚保留的动作抽象，不应写成完全端到端连续控制。

策略只接收 `client.state` 和合法 measure/clear 反馈。不读取场景种子、源总数、真实坐标、真实接收半径或引擎私有属性。训练端在 episode 结束后检查是否全部清除，检查结果仅用于失败惩罚。测试使用 `ObservationOnlyClient` 明确拒绝非观测接口。

## 网络与真实 PPO 更新

每个候选有 24 维特征（动作种类、位置、相对距离、外包半径/面积、正负观测数、当前频道、最近选中源、几何邻近关系等），全局状态为 12 维。共享 MLP 编码候选，通过 masked mean/max 聚合剩余任务集合，actor 对每个候选输出 logit，critic 预测剩余回报。默认隐藏宽度 96。候选排列改变时策略 logits 同步排列，价值预测不变。

当前摘要不是经过证明的充分 belief state。原始完整历史仍由控制器保存用于合法几何，网络使用有损摘要；首版不声称实现精确 POMDP 信念状态、POMCP、AlphaGo 或论文中的完整 Attention Model。后续可将循环记忆或注意力作为独立改动评估。

每个策略动作奖励为 `r_t = -实际新增虚拟秒数 / 1000`。覆盖扫描是变长宏动作，定位/清除是单次动作，因此固定 `gamma=1`，不按动作数折扣。默认 GAE `lambda=1`，于是回报为负的剩余总虚拟时间；可以显式调整 lambda 并记录。固定原点扫描成本没有可学习动作，对最优策略只差同场景常数；日志单列它以便还原总用时。若决策上限后调用完整基线收尾，整个收尾成本归入最后一个学习动作的奖励。失败额外惩罚 360000 秒，不能通过提前退出获得高奖励。由于训练 wallclock 截止造成的行政中断不用于 PPO 更新，单独记录。

PPO 使用新策略与行为策略的概率比、clipped surrogate、价值 MSE、熵正则、全批优势标准化、梯度裁剪及可选 KL 提前停止。每批所有 episode 使用更新前同一份网络权重，之后在 GPU 进行 minibatch 优化；环境与单步推理默认在 CPU 工作进程进行。这里确实存在基于交互回报的反向传播和参数更新，不是把蒙特卡洛前瞻评分称为强化学习。

可选 BC 热启动以当前 efficient 的宏行为拆成逐次动作生成教师标签，并同时拟合教师回报作为初始 critic。标签没有进入 actor 输入；实际 PPO 阶段默认辅助 BC 权重为零。必须保留 `random.pt`、`bc_only.pt`、PPO 检查点分别评测，分别回答“模仿了多少”和“强化学习额外改进了多少”。BC 教师是用于起步的辅助，不足以证明 PPO 有收益。

## 复现命令

Python 3.10+，训练额外依赖 numpy、PyTorch；远端既有 Python 3.11/PyTorch 2.9.1/CUDA 12.8 可直接使用。不要为了复现这段命令替换用户已有 CUDA 环境。

```bash
export PYTHONPATH=src
python -m pytest tests/test_deep_rl_controller.py tests/test_deep_rl_training.py -q
python -m research_rl.train --output results/rl/smoke --device cuda --bc-episodes 8 --bc-epochs 2 --updates 2 --episodes-per-update 4 --workers 2 --epochs 2 --max-wall-s 180 --deadline-utc 2026-09-11T06:00:00+00:00
python -m research_rl.train --output results/rl/run01 --device cuda --bc-episodes 128 --bc-epochs 20 --updates 1000 --episodes-per-update 32 --workers 4 --num-threads 1 --max-wall-s 1800 --checkpoint-seconds 1200 --deadline-utc 2026-09-11T06:00:00+00:00
python -m research_rl.train --output results/rl/run01 --resume results/rl/run01/latest.pt --device cuda --updates 2000 --workers 4 --max-wall-s 1800 --deadline-utc 2026-09-11T06:00:00+00:00
```

默认训练场景从 100001 开始；训练入口拒绝验证和最终测试种子。父任务的 `research/v1_protocol.json` 为共同划分权威来源。本入口不执行验证，避免训练端窥视封存测试。训练/验证进程应各自使用不同输出目录，不并发覆盖同一 checkpoint。`--resume` 恢复模型、优化器、计数器、场景序列及 Python/NumPy/Torch RNG；可显式调整学习率和墙钟预算，配置随运行保存。调用者需维持架构 hidden 一致。

运行产生：随机、BC、最新及阶段检查点，`config.json`，逐批 `training.jsonl`（含每局种子、成功性、总用时、失败惩罚、动作类型、兜底贡献、采样/优化分时）。检查点含算法版本、Git commit、Torch 版本和 UTC 时间，使用临时文件原子替换。checkpoint 使用 PyTorch 序列化，只加载本任务自己生成且可信的文件。训练目录默认 gitignore，最终选中权重和哈希应由交付流程明确归档。

若通过 `git archive` 部署而不带 `.git`，设置 `Q3_SOURCE_COMMIT` 为归档的真实 commit；训练配置和检查点还会独立记录 `src` 下 Python 源码、项目配置、共同协议的逐文件 SHA256 和合并摘要。代码改变后应启动新进程以避免来源缓存混淆。

共同评测入口：

```python
from research_rl import run_rl_search
result = run_rl_search(client, problem=3, checkpoint="results/rl/run01/latest.pt",
                       device="cpu", deterministic=True, max_actions=10000)
```

函数返回 `SearchResult` 子类；`as_dict()` 包含标准报告与 `learning` 统计。checkpoint 按路径/mtime 在进程内缓存，避免每局重载。评测应固定线程数（例如 PyTorch 1 线程），报告现实运行时间，不能把 GPU 或规划开销藏起来。

## 参考资料及采用范围

以下资料已核实来源；实现为按本题接口独立编写，不直接复制第三方项目。

- Schulman 等，2017，[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：采用 on-policy clipped surrogate 思路，不将其误称全局最优算法。
- Schulman 等，[Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：优势估计的偏差/方差权衡；首版为不折扣总时间使用 gamma=1、默认 lambda=1。
- Kool、van Hoof、Welling，[Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475) 及[作者实现](https://github.com/wouterkool/attention-learn-to-route)：参考“对可变任务节点产生策略分布、用完整路线成本学习”的思路。本实现采用紧凑集合网络而非复刻其 attention 架构，其已知 TSP 图假设不能直接移植成未知干扰源真值输入。
- Ni、Eysenbach、Salakhutdinov，ICML 2022，[Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://proceedings.mlr.press/v162/ni22a.html)：作为后续循环记忆及部分可观测性研究依据；首版尚未实现循环网络。
- [PyTorch A2C/PPO/ACKTR 研究实现](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)：作为实现核对参考；本题自定义的是变长动作集合、法律观测和 undiscounted virtual-time 口径。

## 第一轮应检查的失败机制

随机策略可能在测点间往返；BC 若仅学会最近点，可能遗漏整条覆盖路线的后续成本；PPO 大幅偏离 BC 时可能忘记连续完成同一源，增加切换与折返；过强 BC 约束则可能阻止超过教师。需分别查看回报、动作占比、probe 次数和安全兜底耗时，不能把“全部成功”自动解释为网络学会可靠清除。若 BC-only 胜过 PPO，应保留这个失败结论，并根据采样量、熵、KL、critic误差诊断，不做选择性汇报。
