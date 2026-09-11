# Q4 G1 微动作训练入口与功能验证

本模块为 `q4-micro-g1-v1` 控制器提供独立 CPU PPO 训练，不修改原 `network.py`、`train.py` 或冻结 R8。新增 `src/q4_rl/micro_network.py`、`src/q4_rl/micro_train.py` 和专属训练测试。当前只有一次新合成训练场景的功能烟测，没有训练出经过独立验证的推荐模型。

## 模型与策略接口

- 控制器：`q4_rl.micro_controller:run_q4_micro`，公共 callback 仍为 `(global_features, candidate_features) -> index`。
- 全局特征 13 维、候选特征 50 维，schema 精确匹配全部特征名称及顺序；候选第 5 维为 `immediate_cost_upper`（零起始索引），不是 v1 的最低即时费用。G1 的移动/检测/清除动作及其安全门由独立控制器定义。
- `MicroCandidateActorCritic(hidden=64)` 继承原 `CandidateActorCritic(13,50,64)` 的 MLP 运算，复用 `TorchPolicy`、掩码与打包实现；没有隐式权重扩维、拷贝或宏模型迁移。
- checkpoint 版本 `q4-micro-ppo-cpu-v1`，架构 `q4-micro-candidate-mlp-v1`，明确保存控制器入口、完整 schema、CPU 和 gamma=lambda=1 的目标契约。micro/v1 的加载器均拒绝对方 checkpoint；同维度但不同特征语义也拒绝。
- 独立加载入口：`q4_rl.micro_network:load_policy`，默认确定性选择，可作为统一冻结评估器中的 `policy_factory.entrypoint`。

默认从随机参数开始，`warmstart_episodes=0`，最多 512 次策略决策；如显式启用 BC，也只使用新的合成训练场景与本版本微动作启发式，不加载宏模型或历史验证轨迹。初始模型 `random.pt` 永久保留。

## 计费、数据与资源

PPO、可选 BC、回报构造、固定训练种子映射、统计和原始批写入函数均直接导入原 `q4_rl.train`；模型候选打包直接复用原 `q4_rl.network`，不复制维护另一套统计/PPO 实现。

每个 worker 将 OMP/MKL/OpenBLAS/NumExpr 及 PyTorch intra/inter-op 线程限制为 1，隐藏 GPU；并发检查要求 `workers+1 <= cpu_budget <= 60`。单 worker 在主进程顺序采样、更新，实际为一条计算线程；额外保留 learner 配额是保守预算。训练场景严格限 `8000000..8099999`，不会访问验证 SQLite、官方模拟器、对象存储或远程服务器。

每条策略记录保留完整公共特征、候选、选择、动作种类、实际 `cost_s`、末步 `terminal` 和 `fallback_cost_s`；PPO 另附选择时的 `log_prob`、`value`。先逐条比对 callback 缓存与控制器记录，再要求全部 cost 和等于整局回执费用，最后构造未折扣回报。失败罚费采用 `max(实际用时,360000)`，单独记录末端调整，不篡改实际费用。完整兜底/终止尾费保留在最后一条策略记录。

真实源数据只在策略终止后由评估端取得；实际回报完成后才计算已有 `q4-common-source-edge-v1` 下界。下界只用于事后日志，不加入特征、奖励或归一化。每个原始批保存策略记录、实际动作和原始回执、控制器微动作/网格账本、错误信息及事后环境评估。失败和光学未命中都保留，不能按单局结果选择性更新。

## 事务与恢复

每批先将训练种子、独立动作随机种子、当前模型/优化器和 RNG 保留到 `latest.pt`，再执行采样。每次尝试使用不同文件名 `batch-BBBBBB-attempt-AAAAAA.json.gz`。初版存完整前缀列表；后续修复改为小型 `q4-training-episode-index-v1` 索引，每个返回局另写一次不可覆盖的 `-episode-NNNN.json.gz`，索引记录逐文件 SHA256。`q4_rl.training_journal.read_batch` 同时读取旧列表和新索引，返回相同的工作进程结果。恢复整批重跑同一预留种子，不覆盖旧尝试，不跳过失败场景。

更新中收到行政停止或异常时回滚到最近完整事务，包含模型、优化器、种子位置及随机流；一次更新完整成功后才推进 episodes/batches，并将进度及原始尝试文件名随模型提交。进度写到 `progress.json` 和每批独立的 `progress-NNNNNN.json`，没有覆盖式丢弃历史。`latest.pt` 内的 `last_progress` 可在恢复后重建最新进度。

`--resume` 只允许本输出目录的 `latest.pt`，且冻结训练配置必须一致；从旧 checkpoint 分叉实验需另行明确实现/安排，不能默默覆盖现有进度。改变管理截止时间、允许批数和并发预算不会变更模型训练配置，但同一复现比较仍应保持资源一致。

工作进程在当前局尚未返回前若被系统强杀，尚在其内存的未返回轨迹无法由本驱动补写；已返回局和预留任务完整保留，恢复会重跑整批。未完成行政批不作挑选后部分更新。检查点可恢复不等于保证硬故障前每一条内存记录都已落盘。

## 运行方法

仓库根目录，已安装 torch 的 Python，Windows 先设置 `$env:PYTHONPATH='src;.'`，Linux 设置 `PYTHONPATH=src:.`。示例是可复制的配置，本文未启动长训练：

```text
python -B -m q4_rl.micro_train --output results/q4_rl/micro-run --workers 1 --cpu-budget 2 --hidden 64 --warmstart-episodes 0 --max-decisions 512 --batch-episodes 16 --epochs 3 --scenario-start 8005001 --scenario-end 8099999 --max-wall-seconds 1800 --deadline 2026-09-12T10:00:00+08:00
```

恢复时重复同一配置并追加 `--resume results/q4_rl/micro-run/latest.pt`。截止时间已过则拒绝启动，需用户/任务已有授权给出新的明确截止时间，不会自动延长。评估独立进行，使用原统一场景/资源/下界定义；训练日志不能当独立确认。

## 已完成验证

专属测试 `tests/test_q4_rl_micro_training.py`：11 项通过（5.24 秒），CPU 单线程。覆盖 13/50 特征与候选掩码、宏/微 checkpoint 双向拒绝及语义伪装拒绝、保存不推进 RNG、真实 PPO 更新改变参数、下一次更新恢复一致，以及完整驱动在“梯度已改后中断”和“第二局 worker 异常”下的回滚、原始尝试保留及同批重放。统计/奖励测试直接确认公共函数身份相同，防止后续复制漂移。

真实训练功能烟测只用新训练 seed **8005000**，固定课程 `random/all_directional`，实际 N=16。配置 hidden=64、32 次微决策、batch=1、epochs=1、minibatch=16、warmstart=0、CPU 单线程；有意触发完整兜底以检查费用。实际命令其余参数与上例相同，输出 `results/q4_rl/micro-training-smoke`，起止 seed 均 8005000，`--max-batches 1 --max-wall-seconds 180`。

| 功能烟测指标 | 结果 |
|---|---:|
| 全清 | 16/16（单局） |
| 实际 T | 22861.472938 秒 |
| 原统一 L | 2008.034401 秒 |
| T/L | 11.38500064 |
| 移动 / 切频 / 检测 / 光学 / 清除 | 20427.472938 / 255 / 1445 / 702 / 32 秒 |
| 失败清除 | 218 次，全部保留 |
| 策略决策 | 32，均选择 measure |
| 兜底费用 | 10338.663706 秒，完整计入末步 |
| 首状态回报 | −22.861472938（固定 1000 秒单位） |
| worker wall / CPU | 1.879 / 1.859 秒，含终局下界审计 |
| 其中下界 wall / CPU | 1.016 / 1.016 秒 |
| PPO 更新 wall / CPU | 0.186 / 0.188 秒，2 个 minibatch |
| 随一次 PPO 变化的参数张量 | 14 个 |

独立事后物理回执与完成审计通过（16 次实际成功清除）；32 个策略动作逐个均对应一条 accepted 请求，全部 transition 成本和等于 T，兜底等于末条所记费用，训练后 checkpoint 可重新加载并执行选择。原始模型、训练模型、原始批、进度和 `posthoc-audit.json` 全部保存。

该单局随机初始策略很差，且选择前缀中没有 learned clear/grid 行为，因此只能证明训练/保存/计费链路正常，不能验证已学会全部联合动作、不能判断相对 R8 的性能，也不能拿一局自举得到的退化点区间当统计可靠性。本批没有配对基线、不属于开发/确认；P95 等于该单局 T 是描述性输出，效果判断必须另开冻结配对开发批。
