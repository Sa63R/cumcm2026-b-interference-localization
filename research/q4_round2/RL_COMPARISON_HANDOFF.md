# Q4 强化学习固定对照交接：Pair-v2 宏 PPO512

核对日期：2026-09-12。仅读取已提交开发结果、源码 ZIP、模型字节与摘要；未加载模型推理、训练、生成新场景、调用 SSH/模拟器或编辑 RL 工作树。

## 固定对照及证据范围

当前建议固定 **Pair-v2 宏 PPO512**：它是原训练作业的最后完整终点，在 Pair-v2 已比较的学习臂中最快，但未胜过 R8，不能称为所有后续模型的已证最优者，也不是根据本轮 618 或独立测试结果挑选的检查点。

`q4-deep-rl@ac7b0929` 是更早的 PPO128 研究；其旧模型的 512 步外推消融仅有 16 场景，均值 7931.04 秒，对应 R8 6941.16 秒，不能与不同场景均值直接排名。`q4-rl-micro-attention@c8b9a3c0` 已保存四个新训练终点及比较计划，核对时未发现已提交的完整新比较结果。`q4-rl-negative-memory@92a16b39` 与同 HEAD 的 scan-bundles 树有新作业部署说明，单训练场景 smoke 很差，不能据此升级推荐。

## 可直接复用的 Pair-v2 结果

预登记 development seed **8101000–8101031**，每个 seed 对应 8 个 family × mixed/all_directional 两种源模式，共 **512 场景、32 个 seed 簇**。family 为 random、minimum_radius、boundary_outward、cluster、positive_error、negative_error、alternating_error、narrow_strip。全部 7 臂共 3584 局已由原任务完成归档审核。

| 此面板方法 | 平均实际 T，秒 | 平均共同 LB，秒 | ΣT/ΣLB | 各自 P95(T)，秒 | 完整清除局数 |
|---|---:|---:|---:|---:|---:|
| compact_baseline | 未运行 | — | — | — | — |
| compact_combo | 未运行 | — | — | — | — |
| R8 / center_once | 7651.591837 | 1790.822234 | 4.272670 | 11938.489835 | 512/512 |
| 宏规则 macro_rule512 | 7726.708396 | 1790.822234 | 4.314615 | 11880.674677 | 512/512 |
| 宏 PPO512 / checkpoint-000055 | 7998.673062 | 1790.822234 | 4.466481 | 12325.294856 | 512/512 |

**宏规则不是 compact_combo**：它是在 RL 控制器内用规则选动作。不能改名为原基线，也不能把 617 场景中的 combo/R8 数字拼进本表。512/512 表示每局清掉其全部 10–16 个源，不表示每局都有 16 个源。R8/宏 PPO 的光学未命中分别为 29297/22855 次，全部计费；未命中少不等于用时短。

宏 PPO 相对 R8 平均节省 **−347.081226 秒**，95% 簇 bootstrap 区间 **[−453.521317, −238.952240] 秒**；相对节省 **−4.5361%**，区间 **[−5.9523%, −3.1267%]**。138 胜、374 负，最坏回退 2880.115182 秒。P95 比值 **1.032400** 是两组各自 P95 之比，不是逐局比值的 P95。

区间采用 **5000 次配对 seed-cluster bootstrap**、随机流 4260911；单位是 32 个 seed 簇，不能当作 512 个独立样本。这是已查看的开发面板和探索性比较，不覆盖官方分布差异。

| 分层 | 局数 | 平均 LB，秒 | R8 平均 T / ΣT÷ΣLB | 宏 PPO 平均 T / ΣT÷ΣLB |
|---|---:|---:|---:|---:|
| random，两种源模式 | 64 | 2103.163641 | 7028.433608 / 3.341839 | 7529.138574 / 3.579911 |
| 七类压力，两种源模式 | 448 | 1746.202033 | 7740.614441 / 4.432829 | 8065.749418 / 4.619024 |

总表压力占 7/8，不能替代 random 层。T/L 均为总和之比；例如 R8 的逐局比值均值另为 5.498138，不能与 4.272670 混用。

## 已核验的身份与实际路径

以下路径均相对 `D:/jwt/2026数模国赛/`：

- 原始结果：`q4-rl-micro-actions/results/q4_rl/server-pair-v2-evaluation-001/`。
- 原独立审查：`q4-rl-micro-actions/research/q4_rl/PAIR_V2_EVALUATION_REVIEW.md`；训练说明为同目录 `PAIR_V2_TRAINING.md`。核对时工作树 HEAD 为 `d3cafa78`。
- 冻结推理/审核源码为结果目录内 `source.zip`，52 文件，实际 ZIP SHA256：`86580efae901756eccaeeb48edb6045ca2a9bb8ac9b436ca60cf17fa9179229d`。
- manifest canonical SHA256 与 freeze 一致：`0982b845e2b55fb612f77799a1ed6c330a703359c3bcbffac88dc376f460d1c5`。
- summary 实际字节 SHA256：`f51e0ebc97afcbdf271e14e8aaf2ec26a3584d85797a72df5bdecee5f4e4314f`。
- 模型实际路径：`q4-rl-micro-actions/results/q4_rl/server-pair-v2-complete-001/macro512/training/checkpoint-000055.pt`，348523 字节；实算 SHA256：`3e830b070d5247e3573b3e59b7dfc32d54da101c5aab05cd4effb0dec2b354e8`。

ZIP 中除新增 RL 文件/协议以外的 **42 个共有文件**，与 `git show dc8651b7:<path>` 全部字节相同，包括 R8 父策略、几何、引擎、下界及共有审核器。因此这里的 R8 确为本方向合格版本；spec 为 `config="center_once", max_expansions=200`，全局 `max_actions=20000, max_active_probes=6`。

共同下界文件 `experiments/q4_comparison_bounds.py` SHA256 为 `973da2a119eaad526e643aaebc6cf99b9a624af53f850d4dfe85abf1705e8804`，同 dc8651b7 完全一致：`q4-common-source-edge-v1`、相同 N=16 分支和微秒核界。本表未混入其他信息论或观测路线下界。

本次还逐项核验 52 个 ZIP 文件的 manifest SHA；从摘要独立确认 512 对 R8/宏 PPO 的 case_sha256、共同 LB、成功/全清/审核标志一致，无缺失/重复。没有重跑昂贵几何审核，完整动作审核沿用原已提交独立审查。当前 micro-actions 的 `micro_train.py` 已有后续修改，因此以 ZIP 为冻结身份，不能把整个当前 HEAD 冒充原实验快照。

## 新场景隔离评估 API

模型为 CPU 候选 MLP，hidden=64，10 维全局/16 维候选特征；schema `q4-joint-scan-service-v1`，架构 `q4-candidate-mlp-v1`，检查点版本 `q4-ppo-cpu-v1`。终点含 64 局 BC 和 816 局 PPO。512 是决策上限；后续固定兜底的全部费用须保留。它仍使用手工候选与继承的安全定位/清除，不是无约束端到端机器人策略。

在新工具目录解压原 ZIP、复制并验固定模型；新进程只从该目录导入 frozen src/experiments，不能在已经加载当前树同名模块的进程内切换 sys.path。原 RL 工作树不改。新适配器、输入 case 清单、ZIP、模型及参数在首局前另冻结。

核心调用示意（本文没有执行）：

```python
from q4_rl.network import load_policy
from q4_rl.controller import run_q4_rl
policy = load_policy(checkpoint=absolute_checkpoint_path, deterministic=True)
report = run_q4_rl(
    observation_only_client, policy=policy, problem=4,
    max_actions=20000, max_active_probes=6, max_expansions=200,
    max_decisions=512, record_transitions=False,
)
```

加载器在 frozen `src/q4_rl/network.py:139`，使用 `weights_only=True, map_location="cpu"` 并校验 schema；入口在 `src/q4_rl/controller.py:364`。原 Linux 环境已验证 Torch 2.9.1+cpu / NumPy 2.2.6，数值库每进程 1 线程、GPU 禁用。新平台差异应记录，不能将原 Linux 墙钟与新 Windows 墙钟直接比较。

原 `ObservationOnlyClient` 仅暴露 `state`、`remaining_real_time_s`、`pending_request`、`enter/measure/clear/exit`，策略接受此鸭子接口，不能收到 case、seed、真值、LB 或环境引用；模型回调只看数值特征。

**不要直接给旧 CLI 传入 618 seed。** `q4_rl.scenarios.build_case` 使用自己的 800/810/820/830 万命名空间，生成器也不同；同 seed 不保证同场景。新外层适配器应读取公共生成器产生的完整配置，保留 case_id、seed、error_mode、description 和全部 Source 浮点参数，在隔离进程里直接构造 frozen `Q4Scenario`（构造器允许普通整数 seed），再交给相同 LocalResearchSimulator。不能重新调用 RL build_case。策略退出后才读 evaluation/truth，核对各臂 case_sha256 与 LB 相等，所有失败照旧保留处罚。

RL 覆盖顺序可变，不能强套固定链完成检查。沿用 frozen `experiments/q4_rl_evaluate.py:audit_record`：物理 wire 审核加真实逐频道负测覆盖，或 16 次实际成功清除证书。新适配器还须保留错误/退出/完整物理记录与总费用核验。该文件第 162 行的旧 `run_one` 可作为接口参考，但内部调用 RL 生成器，不能原样用于共同 618 case。

现有 512 对可支持当前并列表，无需为表格重跑 617。后续固定该模型为额外臂；新模型完整比较与资格证据到齐后，先冻结替换规则再用于新场景，不得逐局挑较好的 RL 结果。
