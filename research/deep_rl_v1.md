# 第三问深度强化学习首版：实现与复现说明

本文件是实验设计及代码说明，不是论文正文。初始实现版本为 `q3-candidate-ppo-v1`；当前代码默认 `q3-candidate-ppo-v2`，保留 v1 推理兼容。另提供显式选择的 `q3-joint-scan-ppo-v3` 原型。前面的首版说明描述 v1；v2、v3 的增量、实验和迁移见文末。
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

运行记录修正：`optimizer_steps` 记录真实梯度更新次数（含 BC）；只有实际执行至少一次 optimizer.step 的 PPO 批次才增加 `update`，截止发生于首次梯度步前不会虚报完成一轮训练。非空输出目录在未传 `--resume` 时会拒绝覆盖。输入列重排仍不支持；下面的注意力实验新增了显式、版本化且可保持初始输出的架构扩展。

## v3 原型：未知频道与可中断扫描

新增独立 `research_rl.joint_scan.JointScanRLSearch`，公共接口显式选择 `feature_version="v3"`，训练参数为 `--feature-version v3`。v1/v2 模块和 checkpoint 仍按原动作语义运行；默认训练版本仍为 v2，不会悄悄扩大正在进行的实验。

v3 取消固定原点扫描和不可中断覆盖宏动作。初始候选是 7 个保证覆盖点与 20 个频道的笛卡尔积，共 140 个单次测量；扫描一对点/频道后移除该候选，已清除频道在所有覆盖点的剩余候选同步移除。出现合法正观测后，再加入已有的主动定位测点、几何清除和必要兜底候选。网络每次直接选择一个“测点＋频道＋动作”，可立刻定位刚发现的源、改变扫描频道、离开尚未扫完的点，或随后再回来补扫。未知频道的测量位置仍限定于这 7 个覆盖点，尚未允许在任意连续位置搜索未知源。

每个候选共 60 维：前面的状态/几何特征加 16 维扫描特征，包括实际频道编号、该频道是否已经发现、当前点/候选点还欠多少频道、该频道还欠多少覆盖点、是否停在原地、是否对应最近一次扫描点、该频道覆盖点负观测比例，以及明确的 **7 位逐点待测向量**。同一点不同频道不再只有相同的位置输入，网络能看见它们不同的观测账本。单次扫描的成本特征改为移动/切换/检测实际成本量级，覆盖候选几何使用当前选择频道的外包集而不是全频道均值，因此 v3 与 v2 不具有可保持语义的参数前缀关系。

### 完备性、成本和边界

对覆盖点 `p` 保存真正已经测过的频道集合 `M_p`，对已清除频道保存集合 `C`。待办集合严格为 `{1,...,20} - M_p - C`。只有实际接受的测量才写入账本；恰在覆盖点进行的其他定位测量也可写入，近似坐标不替代真实点。所有待办集合为空时，任何尚存在的源都应在原 7 点覆盖保证下被检测；再结合检测源已清除、公示的源数下界检查，才能给出终止证书。成功清除 16 个源仍可提前终止，不必证明其他 4 个频道为空。网络没有“相信没有源所以结束”的自由动作。

每次覆盖动作消除一个尚欠的点/频道对，最多 140 次；每个已发现源最多 6 个学习定位测量，随后一次几何清除或完整兜底。因此在合法静态模型下，学习决策次数有 `140 + 16×(6+1) = 252` 的粗上限，默认 256 决策预算足以覆盖这个有限动作抽象。光学兜底可能包含多条物理指令，其完整时间和指令数照常单列并计费；有限成功测试不等于证明所有光学尝试都会一次成功。更小的决策预算会触发确定性收尾，只补尚欠点/频道，避免重复扫描。新边界是“7 个点都还未完成”，旧的覆盖路线例程只接受最多 6 点；v3 收尾先完成最近一个待办点，再调用原来的有界例程，不修改旧路由模块。

初始观测现在全由策略选择，`initial_scan_virtual_time_s=0`，每一秒都进入策略回报。行政训练截止、物理动作预算耗尽和坏接口引起的全负反馈不会被写成成功；仍通过失败记录和完整评测统计处理。

### 能力验证与成本

在 7 个困难场景和 6 个训练随机场景，teacher/random 两种控制器均通过真值隔离、完整清除、账本及时间检查；另有 10 个训练场景逐条验证 v3 teacher 与原 efficient 的 `(action, channel, position, result)` 完全一致。这使后续训练差异可与动作实现差异区分。

一个明确的动作能力例子：16 个源均在原点 5 m 内，旧固定原点扫描要先测 20 个频道，再清除，合计 **199 秒**。v3 可以每发现一个便清除，测完第 16 个源后凭数量上界结束，总共 **175 秒**，少测 4 个空频道。**这是测试中的显式观测策略，不是已经训练出来的模型性能结论。** 它证明新增动作空间确实能表达旧宏动作空间中无法实现的路线。

相反，随机策略频繁打断扫描会造成大量折返。首轮基准使用训练种子 `100301..100316`、2 次重复、v2/v3 交替 teacher，虚拟时间完全一致；平均决策数从 48.375 增至 145.5，候选总评估数从 1079.625 增至 12074.188。未做缓存时 v3 约 450.7 ms/局；缓存同状态中的重复面积、点间距离和待测集合后约 267.7 ms/局（同时 v2 约 97.4 ms/局）。这些是本地 Python 控制器开销，不含神经网络；远端训练需实测吞吐，不能预设更大动作空间一定值得新增计算。

### 首轮训练和消融

v3 **禁止**从 v1/v2 隐式或零扩列迁移，因为覆盖动作的含义已变化。应先用新输出目录进行新的 BC 热启动，再真实 PPO 更新，并分别保留 random、BC、PPO 权重。父任务负责安排 GPU 资源、验证与是否继续训练；本模块没有连接官方模拟器。

向后兼容检查还直接加载了冻结 `acb447d` 控制器：v1/v2 各在训练种子 `100251..100255` 运行，比较每次送入 actor 的完整候选特征、全局特征、教师索引与最终虚拟时间，共 10 局全部逐值相同；两版特征语义摘要也保持相同。新的缓存改变计算成本，没有改变已有策略输入语义。源码摘要仍然不同，旧运行应继续使用自己的冻结快照恢复。

```bash
Q3_SOURCE_COMMIT=<snapshot-commit> PYTHONPATH=src python -m research_rl.train --feature-version v3 --output results/rl/joint_scan01 --scenario-start 150001 --device cuda --hidden 96 --bc-episodes 128 --bc-epochs 20 --updates 1000 --episodes-per-update 32 --workers 4 --max-wall-s 1800 --checkpoint-seconds 1200 --deadline-utc 2026-09-11T06:00:00+00:00
```

建议同时检查现实吞吐、完整清除率、动作总数、切换次数、扫描站点切换、未完成扫描被打断次数及兜底贡献；不能只比较训练批的平均回报。报告字段 `scan_measurements`、`scan_site_changes`、`interrupted_scans` 和逐点 `coverage_ledger` 支持这项诊断。`interrupted_scans` 是离开仍欠频道的扫描意图次数，可能包括同一未完成扫描期间的多次源动作，不应解释为不同站点数量。

另修正采样可复现性：worker 在构造/加载模型后重置 Torch、NumPy、Python 的 action seed，避免首次模型初始化消耗随机数，导致同一任务在新 worker 与复用 worker 中抽到不同轨迹。测试检查两种生命周期的动作、输入、概率和奖励一致；这不回溯改变旧训练结果，也不宣称旧快照可逐位恢复到未中断的采样序列。

## v3 精确缓存：不改变策略的 CPU 加速

先对冻结 `fe6cf25` 的 teacher 执行进行 cProfile：16 个训练场景的被测调用约 10.68 s，其中特征构建 6.28 s（约 59%）；几何特征累计调用约 19.7 万次、1.60 s，多边形面积约 3.3 万次、1.31 s。公共观测与裁剪引擎另约 3.1 s，本次不修改它。源候选构造也在同一点连续扫描不同频道时重复计算。

据此只做三种精确缓存：

- 按频道及不可变 `vertices` 对象缓存面积；顶点对象变化时重算。
- 按顶点对象、首次方位、角误差参数和候选位置缓存 20 维几何特征，返回不可变内部元组；构造 actor 行时复制数值，外部输入修改不能污染缓存。
- v3 按机器人位置、该源外包顶点、near 位置、已观测坐标集合、探测次数、首次方位及探测上限缓存源候选。其他频道的观测不使这一源的候选失效；机器人移动或该源新观测会正常失效。

动作顺序、特征含义、网络、训练目标和超参数完全没有改动，不需要新的动作/特征语义版本；源码摘要仍更新，因此继续使用独立冻结训练快照或显式新 trial。

### 等值审计与性能结果

`research/benchmark_joint_cache.py` 可以加载冻结 commit 或独立快照，通过只读观测客户端进行审计。训练种子 `100501..100532`，每局同时检查 teacher 与固定 RNG 的随机策略，共 **64 对执行**。前后候选顺序与字段、完整 60 维候选输入、全局输入、teacher 索引、全部行动日志及虚拟时间逐值一致，所有执行均完整清除。另设缓存失效单元测试检查新的正/负观测、角误差/方位变化、机器人移动、重复坐标、探测上限和 near 反馈。

独立计时不保留大型输入快照，采用相同 32 个场景、2 次重复并交替先后顺序，每版本 64 局；Windows/Python 3.12，不含 Torch：

| 指标 | 冻结 fe6cf25 | 精确缓存 |
|---|---:|---:|
| 平均现实时间/局 | 273.123 ms | 193.198 ms |
| 平均特征时间/局 | 175.730 ms | 104.095 ms |
| 平均虚拟时间/局 | 3501.656395 s | 3501.656395 s |
| 平均学习决策数 | 145.53125 | 145.53125 |
| 平均候选总数 | 12155.40625 | 12155.40625 |

在这组同平台测试中现实时间降低 **29.26%**，控制器吞吐约提高 **1.414 倍**。8 局前后 cProfile 中，实际几何重算次数从 36193 降至 5364，底层源候选重算从 4165 降至 1605。剩余成本包含不可省略的网络输入组装和公共观测几何运算；不将本机控制器速度提升直接当作远端训练或解题虚拟时间改善。

逐局计时、64 对等值结果、源文件哈希和 Python/平台信息保存在 `research/joint_cache_results/audit.json`，文本热点在同目录。复现：

```bash
PYTHONPATH=src python research/benchmark_joint_cache.py --reference-commit fe6cf25 --count 32 --repeats 2 --profile-count 8
# 无 .git 的远端可改用 --reference-root /path/to/frozen-fe6cf25-snapshot
```

## v3 动作不变的注意力架构消融

### 动机与可检验假设

当前网络先独立编码各个候选，再用所有候选的逐维 mean/max 汇总供 actor/critic 使用。这种有限维、有限深的实现，可能难以直接表示“这个候选与哪个同频道候选配合”或“不同频道候选集中在哪个方向”等关系；这是**表示瓶颈假设**，尚不是既有结果不够好的已证实原因。[Deep Sets 原论文](https://arxiv.org/abs/1703.06114)讨论集合函数的表达，不应被误读为所有池化架构必然不能表示这些关系。初始 140 个扫描候选只有 7 个不同坐标，频道任务数会改变这些坐标在 mean 中的权重；注意力本身也有 token 数量敏感性，所以不能把它说成自动消除了重复测点。

受 [Kool 等人的原始图编码器](https://github.com/wouterkool/attention-learn-to-route/blob/master/nets/graph_encoder.py)启发，在原候选 encoder 后加入 1 层（可选 2 层）内容自注意力，让一个候选在打分之前能按学习到的权重读取其他候选的表示。仍使用 v3 原来的 60 维候选输入、全局输入和完整合法候选序列；没有增加真值、场景种子、teacher 标签、规划搜索结果或新的几何特征。该试验只改变架构，不改变动作空间或回报。

每层为 LayerNorm → 4 头 masked self-attention → 标量门控残差，加 LayerNorm → 两层 Tanh 前馈网络（内部宽度 2h）→ 标量门控残差。没有按候选序号的位置嵌入，也没有新增成对距离偏置；候选的顺序置换应只置换 actor 输出，critic 保持不变。padding 只作为无效 token，不参与注意力键或最终池化；每个真实输入必须至少有一个合法动作。最终仍接原 mean/max、context encoder、actor 和 critic。

两个残差门控都初始化为零，借鉴 [ReZero 原论文](https://arxiv.org/abs/2003.04887)的恒等初始化思想，但本实现保留 LayerNorm，不声称逐行复现其架构。它允许将同一 v3 MLP checkpoint 显式迁移后，未训练时 logits 和 value 逐值相同。首次反向传播先更新门控，门控非零后内部注意力权重获得梯度；测试明确检查这两个阶段，防止“名义加了注意力但永远没有学习”。也支持从头 BC 的独立冷启动对照。

### 版本、训练与代价

checkpoint 新增独立 `architecture` 字段，MLP 为 `{version: 1, name: "mlp"}`；注意力记录层数、头数、零门控、无位置编码、零 dropout。控制器算法仍是 `q3-joint-scan-ppo-v3`，特征语义不变。旧 checkpoint 缺少 architecture 时按 MLP 读取，带注意力参数却缺少或矛盾的元数据会拒绝。旧 paired 训练的 Namespace 没有新参数时仍默认 MLP；其训练法没有在这里顺带更改。

`--resume` 必须匹配架构、动作/特征语义和源码摘要。`--initialize-from` 是新 trial，可以进行 MLP→注意力、相同头数的 1→2 层扩展；已训练层完整复制，新增层保持零门控，不允许无声明地丢层、更改头数或跨 v2/v3 动作语义转移。来源和目标架构都写入初始化记录。对应的 `initialized.pt` 是相同初始策略的对照，不应与后续 PPO 增益混淆。

隐藏宽度 96、140 个候选时，每个注意力块的主要矩阵计算约为 `8Nh² + 2N²h = 14.09×10^6` 次乘加，新增参数 74786 个。实际 CPU 时间取决于 Torch 内核、线程和候选数，不能用这个运算量推算整局提速；注意力**可能因采样开销变大而降低相同墙钟预算的效果**。提供独立 CPU 合成输入基准，交替测试 MLP/单层/双层、16/32/80/140/220 个候选，分离预打包 forward 和打包加 forward 的耗时，不接触任何场景或测试种子。对最终训练决策还应看完整局吞吐，以及同起点同新增场景数、同墙钟两种口径的验证表现。

```bash
PYTHONPATH=src python -m pytest tests/test_deep_rl_attention.py tests/test_deep_rl_training.py tests/test_paired_reinforce.py -q
PYTHONPATH=src python research/benchmark_attention.py --output results/rl/attention_cpu_benchmark.json --iterations 100 --repeats 3 --num-threads 1
# 与 MLP 对照使用相同 source checkpoint、新场景范围、更新/采样预算，各用新目录
PYTHONPATH=src python -m research_rl.train --feature-version v3 --architecture attention --attention-layers 1 --attention-heads 4 --initialize-from /path/to/v3-mlp.pt --output results/rl/joint_attention01 --scenario-start 160001 --device cuda --hidden 96 --updates 1000 --episodes-per-update 32 --workers 4 --max-wall-s 1800 --checkpoint-seconds 1200 --deadline-utc 2026-09-11T06:00:00+00:00
```

本机没有 Torch；此次本地控制器/缓存回归为 79 项通过、3 个 Torch 模块跳过，并通过 Python 编译检查。新增 Torch 测试覆盖掩码、排列等变、旧模型加载、严格迁移、门控梯度、真实环境采样/PPO 参数更新和恢复；其执行结果、远端 CPU 耗时及训练效益必须由训练主机验证后另行记录，不能将“测试已编写”当作“训练已优于旧方法”。
