# 固定的第四问强化学习对照

source.zip及checkpoint.pt来自另一研究任务已归档的Pair-v2宏PPO512；reference.json记录实际SHA与原配置，不使用其当前仍在改动的工作树代码。模型是512步学习决策后带完整兜底的CPU候选网络，不是无约束端到端策略。选择依据和既有512场景比较在q4-round2的RL_COMPARISON_HANDOFF.md。本轮不训练、不换权重、不依据618的表现选择模型。

新旧算法同一场景比较：先运行冻结的状态搜索矩阵，再执行：

```powershell
python experiments/run_q4_frozen_rl_compare.py --state-results results/q4_joint_visibility/development --output results/q4_joint_visibility/development-rl --python ../q3-deep-rl/.venv-win/Scripts/python.exe --workers 3
```

前一个python只需本仓库原环境；--python指定有PyTorch2.9.1+cpu、NumPy2.2.6的独立解释器。模型进程加载已校验的源压缩包；环境配置由外层评估器传入，模型只得到ObservationOnlyClient。结果逐局检查case SHA、历史LB、真实动作费用及独立逐频道全覆盖证据。不要把不同生成器的相同seed当成同一场景。

compatibility-check.json保留一次旧RL场景的跨解释器完全一致重放，以及24局R8跨源码重放的行级一致性。它们只验证评估适配，没有给新候选增加独立效果证据。运行目录results/tmp中的解压包可由prepare-only重新创建；原模型及源码均留在此目录以便复现。
