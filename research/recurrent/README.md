# 候选集策略的跨步骤 GRU 记忆实验

本分支实现可采样、训练、恢复和独立评估的轻量记忆 PPO。当前只完成机制验证，没有把少量训练样例当作性能证明，也没有宣称 GRU 优于已训练 MLP。

## 为什么选择这一结构

PPO 采用固定一批交互数据、再进行多轮概率比裁剪优化的方式；循环策略应在保存的同一历史上重算当前策略概率，不能把旧隐藏向量加在随机打乱的单步上冒充循环训练。[Schulman 等，PPO 原论文](https://arxiv.org/abs/1707.06347)

SB3 的 RecurrentPPO 文档明确要求在预测时传递循环状态并提供 `episode_start`，其实现也对补齐后的序列损失使用掩码。本实现选择更简单、可直接核验的整局展开：每个优化小批次都从零状态重新计算整个案例，避免截断边界的过时隐藏状态与 burn-in 选择。[官方文档](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)、[官方实现](https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/ppo_recurrent/ppo_recurrent.html)

POPGym 用多种部分可观测任务比较了多种记忆模型，说明应把记忆能力单独检验，不能从架构名称推导收益。本任务现有候选特征是历史压缩而非已证明充分的信念状态，因此 GRU 是可检验的候选路线；是否减少测量仍需冻结后的同预算对照。[Morad 等，POPGym](https://arxiv.org/abs/2303.01859)

上述来源支持循环状态管理与序列训练方法；下面的 GRU64 选择、残差温启动及整局小批次是本分支针对 CPU 预算的工程选择，不是论文已经证明的最优配置。

## 网络与观测边界

保持原 `v3/base/flat` 动作集合、60 维候选特征、12 维 context、MLP96 编码器及原 actor/critic。对候选编码做有掩码的均值/最大池化，再与 context 编码为当前决策表示 `z_t`。

```
h_t = GRU(z_t, h_(t-1))       # 一层，默认隐藏宽度 64
z'_t = z_t + W_memory h_t    # W_memory 初始全零、无偏置
actor_i = old_actor(candidate_i, z'_t)
value = old_critic(z'_t)
```

GRU 记住的是此前**决策边界可见的候选/上下文表示**，不接收隐藏源坐标、真实接收半径、最终源数或未来反馈。没有额外读取真实采集数据库。当前没有逐频道记忆槽或原始反馈专用编码，池化也不保证保留全部历史信息，这是该轻量路线的明确局限。

从同宽度 v3/base/flat MLP 温启动时，复用全部原参数，GRU 新建，记忆读出置零；在相同输入历史上，初始化的 logits 和 value 与原 MLP 严格一致。首个梯度更新先打开读出，随后 GRU 参数得到跨步梯度。此性质不保证跨平台的物理轨迹逐位一致，也不保证训练后的策略保持原输出。

## 训练规则与恢复

- 只打乱整局顺序；每个案例从零隐藏状态展开，默认每优化小批次 4 局，最多 256 个策略决策（硬性支持范围 1–512）。所有真实步骤参加 BPTT。
- 候选项补齐与时间步补齐分别有掩码；GRU 通过 packed sequence 跳过时间补齐，PPO/价值/熵损失只计算真实步骤。
- 每局单独计算 `gamma=1, lambda=0.95` 的 GAE。终态值为零。达到策略决策上限后的既定完成策略全部耗时计入最后一个策略动作。
- 行政截止导致未完成的整个案例被丢弃，既不伪造失败标签，也不把片段当作完整轨迹。失败但已经正常终止的案例保留完整耗时与 360000 秒失败罚项。
- GPU 构建与非 CPU device 均拒绝。采样进程各使用 1 个 Torch 线程，优化线程数独立设置；不自动启动远端工作。
- 新模型在采样前原子保存 `latest.pt`；每批派发前先预留并保存 `attempted_episodes/next_seed`。强杀批次的预留预算不补领，恢复不会重用场景。
- checkpoint 保存独立算法版本、memory/feature/action/training schema、源码哈希、优化器、全部 CPU RNG、计数与来源。恢复拒绝架构、随机种子、训练语义或源码变化；禁止扩大已保存的场景与尝试上限。
- 不在 checkpoint 中续接半局隐藏状态；恢复从下一个独立案例的零状态开始。软中断期间已经完成的优化步计数与优化器一起保存。

**小批次口径必须报告。** 4 个完整案例通常包含多于旧训练器的 128 个单步样本，优化步频率有所不同。它保持动作/观测的单因素对照条件，但不能声称优化调度与旧 MLP 逐步小批次完全一致。正式比较应固定总预算、完整记录批量口径，也可给两种网络采用相同整局批量。这里不调参选择最佳记忆宽度或序列长度。

## 接口

采样与优化入口是 `research_rl.train_recurrent`。示例只描述服务器应如何启动，不代表本地已执行 50 进程训练：

```powershell
$env:PYTHONPATH = 'src'
python -m research_rl.train_recurrent --output results/rl/recurrent_trial_1 --initialize-from models/parent.pt --workers 50 --num-threads 16 --episodes-per-update 128 --episodes-per-minibatch 4 --hidden 96 --memory-hidden 64 --scenario-start 1600001 --scenario-end 1799999 --max-attempted-episodes 199936 --max-wall-s 3600
```

恢复时保留原训练参数，将 `--initialize-from` 替换成 `--resume results/rl/recurrent_trial_1/latest.pt`。`--max-wall-s` 是跨恢复累计训练墙钟上限；可另加带时区的 `--deadline-utc`。

服务器统一入口可使用等价拼写 `--max-wall-seconds`、`--checkpoint-interval`（秒），以及 `--scenario-range START END` 替代起止两个参数。`--updates 0` 只初始化可恢复 checkpoint，不采样。训练分区引用公共 `train.TRAINING_SEED_RANGES`，不在循环实现中另建范围；必须先完成源码集成，再生成初始化 checkpoint。

独立评估使用已有公共评估器的 callback：

```json
{
  "name": "recurrent_gru64_v3_base",
  "entrypoint": "research_rl.recurrent:run_recurrent_search",
  "kwargs": {
    "checkpoint": "results/rl/recurrent_trial_1/latest.pt",
    "device": "cpu",
    "num_threads": 1,
    "deterministic": true
  }
}
```

每次 callback 调用都重置隐藏状态，结束或异常时再次清空。不要交给旧 `run_rl_search` 读取此分支的循环 checkpoint。正式最终场景 2200001 起的分区保持未打开；初步对照由主任务使用允许的开发案例执行。

任何性能报告均附原先的理论下界：先用缓存/审计得到清除圆路线图下界 `L`，物理口径为 `L/5 + 5N`；旧 RL 口径在 `N<16` 时另加 `30(20-N)`。汇总比值使用 `sum(T)/sum(LB)`，不能把训练机制测试的耗时冒充性能结果。

## 已有机制测试

`tests/test_recurrent_ppo.py` 覆盖完整序列与逐步执行一致、隐藏状态重置、两类补齐掩码、候选置换、相同当前观测而不同过去历史导致策略概率不同、梯度到达早期步骤、温启动输出严格一致、GAE 边界、实际完整场景更新、完成策略成本归属、进程采样与串行采样的权重/RNG一致、连续训练与恢复训练逐张量一致、软中断计数保存。测试使用少量合法合成训练案例，未调用官方接口或读取真实验证数据库。

采样中断测试在第一个 worker 任务返回前直接读取磁盘检查点，确认整个批次预算已保存；恢复只使用尚未预留的场景，直到声明上限。验证命令：`python -m pytest tests/test_recurrent_ppo.py tests/test_deep_rl_training.py tests/test_cpu_training_budget.py -q`。其中循环实现 14 项，加旧训练与预算回归 26 项，共 40 项；此处是机制验收，不产生策略性能排名。
