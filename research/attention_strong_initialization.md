# 强策略起点的 attention 有界对照

这是一次尚未完成收益评估的架构对照准备，不新增动作、特征或奖励。使用已有 `v3 / base / flat` 的 60 维输入、96 隐藏单元和一层四头零门控 attention；初始化来自 `cold-finetune-ppo-002/u384`，其继承链累计完成 40960 条纯 RL 采样轨迹。它不与 route-debt v4 或 axis 候选扩展混合。

## 为什么不是重复旧试验

已归档 [训练审计](rl_training_audit/audit.json) 的 12 个完整训练试验中，attention 只有 `joint-attention-matched-001`。配置明确从 `joint-ppo-001/ppo_000344.pt` 初始化：seed `9112031`、场景 `150001..158192`、256×32=8192 条新增轨迹、一层四头。该起点属于早期 BC 继承链；初始化后没有再次 BC。旧固定 48 局结果为 attention 3432.241 s、同起点 MLP 3413.818 s，没有显示 attention 优势。0605 的后续 base/axis 试验均使用 MLP，现有记录未包含 best002→attention。

此次假设是：较强策略状态分布下，候选之间的全局关系表示可能有用。旧结果不能排除这一可能，也不能证明旧差效由 BC 引起。新旧两轮还存在训练预算、随机种子等差别；即使新试验成功，也只说明 attention 在本次初始化与预算下有效，不能单独归因于初始化强弱。

## 最小修改与 RNG

原实现已把 relation 层放在完整旧 MLP 构造之后，保留旧 MLP 参数；但 relation 的随机权重初始化仍额外消费全局 Torch RNG，导致同 seed 下后续首轮 minibatch 排列不同。

现仅用 `torch.random.fork_rng(devices=[])` 包住 relation 构造：层内随机参数仍取原有序列、零门控不变，退出后恢复外部 CPU RNG。MLP 路径完全不进入该上下文。训练器先在 CPU 构造网络再 `.to(device)`，因此此修复覆盖当前 CPU 采样/GPU 优化入口；不声称覆盖外部调用者自行改变默认设备、在 GPU 上直接构造参数的其他用法。

架构元数据、参数名和前向公式都没有改变。已有 attention 权重可直接加载推理；恢复训练依旧遵守原有源码清单核对规则，同一部署快照内恢复会在网络构造后恢复保存的 RNG/优化器。不能把新源码与旧训练快照直接混用并宣称逐位续训。

新增 5 项测试检查：一层/两层关系权重与旧初始化方法逐值相同；初始化后 CPU RNG 与 MLP 一致；真实 CLI 保存的 `initialized.pt` RNG 一致；首轮排列一致；非零门控旧 attention 的加载预测一致；同 action_seed 的完整随机动作/回报轨迹一致。原有注意力测试继续检验 padding、排列等变、零门控的首步梯度、后续分支参数更新和真实 PPO 恢复。全部 deep_rl、paired、portable 相关测试共 192 项通过（46.26 s）。

## 真实 best002 迁移核验

[audit_attention_initialization.py](audit_attention_initialization.py) 使用便携 best002 检查点，在训练场景 `110601..110608` 分别跑贪心和随机策略，每种模式比较 MLP 与 attention，共 16 对完整轨迹。全部全清、零失败清除，完整动作、候选特征/context、log_prob、value、成本和整批 logits/value 在本机 CPU float32 下逐值一致。CPU 构造及迁移前后的随机状态也一致。

[initialization.json](attention_strong_evidence/initialization.json) 保存检查点、模型张量、源码及脚本 SHA256。平均 CPU 单局 MLP 0.32731 s、attention 0.41740 s，增加约 27.52%；包含相同记录和诊断开销，不是 GPU 训练吞吐预测。实际 GPU 数值与公共验证初始轨迹由根代理独立核验。

```bash
PYTHONPATH=src:. python research/audit_attention_initialization.py \
  --checkpoint <portable-best002-u384.pt>
```

## 冻结的训练安排

根代理使用同当前 v3 对照的 best002 起点、seed `9112037`、新场景从 `180001` 开始、512×32 条采样、学习率 `1e-4`、entropy `0.005`，无 BC。只改 attention 架构；最多 3600 s，并记录实际更新数和墙钟时间。现有 ea1a35d v3 对照的 MLP 构造/采样/更新路径未改变，RNG 修复使其首轮排列可比较。若新训练截止未达到同样更新数，须对齐相同更新数端点，不能把不同预算包装为严格架构消融。

```bash
PYTHONPATH=src:. Q3_SOURCE_COMMIT=<new-commit> python -m research_rl.train \
  --feature-version v3 --initialize-from <best002-u384.pt> \
  --architecture attention --attention-layers 1 --attention-heads 4 \
  --hidden 96 --probe-candidates base --group-alpha 0 \
  --seed 9112037 --scenario-start 180001 --bc-episodes 0 \
  --updates 512 --episodes-per-update 32 --workers 4 --num-threads 1 \
  --lr 1e-4 --entropy-coef .005 --max-wall-s 3600 \
  --checkpoint-seconds 600 --output <fresh-attention-trial>
```

只利用既有开发验证比较训练结果。若没有可靠优势，保存负结果并停止此方向；不因增加 attention 而自动推荐它。网络为何能表达集合关系、零门控与有限候选的局限仍沿用 [算法依据](rl_algorithm_basis.md) 及原有实现说明。
