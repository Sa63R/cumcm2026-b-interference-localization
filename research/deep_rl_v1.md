# 第三问深度强化学习首版：实现与复现说明

本文件是实验设计及代码说明，不是论文正文。初始实现版本为 `q3-candidate-ppo-v1`；当前代码默认 `q3-candidate-ppo-v2`，保留 v1 推理兼容。前面的首版说明描述 v1；v2 的增量、实验和迁移见文末。
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
- Zaheer 等，[Deep Sets](https://arxiv.org/abs/1703.06114)：参考共享元素编码与对称集合聚合，当前 actor 的排列等变性及 critic 的排列不变性由结构和测试保证；有限网络容量不等同于逼近完整 POMDP 最优策略。

## 第一轮应检查的失败机制

随机策略可能在测点间往返；BC 若仅学会最近点，可能遗漏整条覆盖路线的后续成本；PPO 大幅偏离 BC 时可能忘记连续完成同一源，增加切换与折返；过强 BC 约束则可能阻止超过教师。需分别查看回报、动作占比、probe 次数和安全兜底耗时，不能把“全部成功”自动解释为网络学会可靠清除。若 BC-only 胜过 PPO，应保留这个失败结论，并根据采样量、熵、KL、critic误差诊断，不做选择性汇报。

## v2：观测几何表达消融

动机：v1 的半径和面积不足以表达细长方位条带的方向，也没有明确告诉策略哪个候选点保证能收到信号、哪个测点能够产生新的交会方向。v2 **仅扩展观测特征，不改变候选动作集合、策略架构、奖励、教师或几何清除规则**，不调用状态搜索分支的规划器。因而可进行较清楚的特征表达消融，而不能先假定它一定改善总时间。

24 维旧候选特征完整保留为前缀，新增 20 维，总共 44 维；全局仍为 12 维：

- 多边形顶点散布矩阵的主轴/次轴支撑宽度、细长度、主轴双角方向、包围矩形填充率；顶点散布用来描述形状，不宣称它是后验协方差。
- 相对测点的径向/横向支撑宽度、到顶点质心的距离、沿主轴/次轴的偏移、是否位于保守外包多边形内。
- 到所有顶点的最大距离及相对 1000 m 的接收裕量。**凸多边形上距离函数的最大值由顶点决定**，因此最大值不超过 1000 m 是真实源必可接收的保守证书；不超过 5 m 则为保守 near 证书。证书只作为输入，清除安全判定仍使用原来的外包圆。
- 与首次观测方向的交会角、角误差条带相对横向跨度、线性秩一更新得到的迹收缩与面积收缩代理。

对于最后两项，令 `S` 为顶点散布矩阵，`n` 为“候选测点到顶点质心”方向的法向量，`d` 为该距离，`tau² = (d tan(1.005°))² / 3`。计算代理矩阵 `S' = S - S n nᵀ S / (nᵀ S n + tau²)`，输入 `trace(S') / trace(S)` 和 `sqrt(tau² / (nᵀ S n + tau²))`，均裁剪到 `[0,1]`。这只是便宜的局部线性信息代理，**不是实际观测的期望后验、不是真实收缩上界，也不是新下界**。候选点恰等于质心时方向未定义，直接使用收缩比 1，避免虚构零噪声和完美定位。覆盖宏动作对所有已发现未清除源的这些特征取均值；例如接收证书维表示此轮扫描可保证接收到的已发现源比例。

信息来源始终是保守外包多边形、已有方位和候选测点。控制器按每个频道的不可变顶点对象缓存形状，新增 `feature_wall_time_s` 分项用于检查特征成本。

### CPU 成本与正确性检查

2026-09-11，本地 Windows/Python 3.12，训练种子 `100101..100132`，3 次重复，v1/v2 交替执行，相同 teacher 策略以隔离特征计算开销，不含 Torch。每个版本共 96 局：

| 指标 | v1 | v2 |
|---|---:|---:|
| 完整清除 | 96/96 | 96/96 |
| 平均单局现实耗时 | 0.080594 s | 0.095995 s |
| 平均特征计算耗时 | 0.010029 s | 0.025730 s |
| 平均虚拟耗时 | 3425.745990 s | 3425.745990 s |

单局增加约 15.4 ms、总控制器开销约 19.1%；由于父任务并发研究会影响共享 CPU，这只是本机单次基准，不是 GPU 训练吞吐的保证。同轨迹虚拟耗时最大差为 0，符合“仅改特征”的预期。新增独立几何测试检查接收证书、凸组合内部点距离界、长窄区域交会角代理、旋转不变性、退化方向保守处理及 v1 前缀/teacher 轨迹完全一致；随机和困难场景继续使用拒绝真值访问的客户端。

可在部署后的 Linux 快速复现上述基准（只用训练种子，不加载权重）：

```bash
PYTHONPATH=src python - <<'PY'
import statistics, time
from simulation import LocalResearchSimulator, random_scenario
from research_rl import run_rl_search
rows = {v: [] for v in ("v1", "v2")}
for repeat in range(3):
    for seed in range(100101, 100133):
        for version in (("v1", "v2") if repeat % 2 == 0 else ("v2", "v1")):
            simulator = LocalResearchSimulator(random_scenario(3, seed))
            started = time.perf_counter()
            report = run_rl_search(simulator.client(), policy=lambda f,c,t: t, feature_version=version)
            rows[version].append((time.perf_counter()-started,
                report.learning["feature_wall_time_s"], report.virtual_time_s))
for version, values in rows.items():
    print(version, [statistics.mean(column) for column in zip(*values)])
print("max_virtual_difference", max(abs(a[2]-b[2]) for a,b in zip(rows["v1"], rows["v2"])))
PY
```

### 检查点兼容与可控迁移

`run_rl_search(checkpoint=...)` 从算法版本自动选择 v1 的 24 维或 v2 的 44 维控制器。旧 `3315abf` 的 v1 checkpoint 可以直接推理；v2 checkpoint 除算法名外还验证特征语义摘要。

`--resume` 现在同时验证算法、hidden、特征语义及逐文件源码摘要；不同代码不能默默接着相同训练编号运行。若需要在新代码或新特征上继续利用旧权重，使用**全新的输出目录**和显式 `--initialize-from`。它重置优化器、RNG 与本轮计数器，保留来源 checkpoint 的哈希、算法、源文件摘要、Git commit 和旧训练计数；新增输入列初始化为零，旧输入列和其他参数拷贝。因此在 v1 前缀保持一致时，迁移后未更新的 logits/价值应与原网络相等，随后才通过 PPO 学习新特征；有专门 Torch 测试验证这一点。迁移自动跳过 BC，并保存 `initialized.pt` 作为消融起点。

```bash
Q3_SOURCE_COMMIT=<new-snapshot-commit> PYTHONPATH=src python -m research_rl.train --feature-version v2 --initialize-from /path/to/v1-selected.pt --output results/rl/geometry-transfer01 --scenario-start 140001 --device cuda --updates 2000 --episodes-per-update 32 --workers 4 --max-wall-s 1800 --checkpoint-seconds 1200 --deadline-utc 2026-09-11T06:00:00+00:00
```

从头训练 v2 沿用上文命令，另选目录。要控制“新增训练量”这个混杂因素，可用同一 v1 checkpoint 显式迁移到 `--feature-version v1` 和 `v2`，然后给予同样的新场景/更新预算进行比较。

运行记录修正：`optimizer_steps` 记录真实梯度更新次数（含 BC）；只有实际执行至少一次 optimizer.step 的 PPO 批次才增加 `update`，截止发生于首次梯度步前不会虚报完成一轮训练。非空输出目录在未传 `--resume` 时会拒绝覆盖。当前不实现跨架构或输入列重排迁移；这些必须另行定义新语义版本和转换代码。
