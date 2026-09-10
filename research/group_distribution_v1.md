# Q3 分组概率校正消融（保持 v3 动作不变）

## 可检验问题与修改范围

既有 greedy 验证中，BC 初始化后的 v3/paired 策略约99%的主动测量选择中心，v3很少中断扫描。其原因尚未由这些轨迹单独确定。新的无BC冷启动已由主任务启动；抽查其6002/6036/6003三个既有验证轨迹，中心占比分别15/32、18/31、18/30，中断次数36、26、25，表明初始化/训练路径确实可以对应很不同的部署行为。这是三局诊断，不是完整48局统计或BC因果证明。

本模块只检验**平坦 softmax 是否受各任务候选数量影响**。v3 所有物理动作、候选顺序、60维特征、MLP参数量、奖励、清除证书和兜底不变。没有新增状态搜索，没有加入额外熵奖励，不增加神经网络层。新增的每源条件熵和中心概率仅用于日志。

## 概率公式

令原始网络分数为 `z_i`。同一个覆盖点的不同频道属于一组；同一个已发现源的定位/清除/兜底候选属于一组，覆盖组与源组互不相交。对合法候选 `i`，定义

`z'_i = z_i - alpha * log(|g(i)|)`，`pi(i|o) = softmax(z')_i`。

仅支持 `alpha=0` 和 `alpha=1`。前者逐值返回原logits，保持旧路径；后者使组概率与组内 `exp(z_i)` 的平均值成正比。其随机采样等价于先按 `logsumexp(z_g)-log|g|` 选择组，再按组内softmax选具体动作。**greedy推理仍取完整联合分布的最大概率动作**；它不等同于“先选最可能组，再选组内最大动作”，代码没有混用这两种解码。

例如三个同分扫描频道与两个同分定位候选：原来扫描组概率3/5、源组2/5；alpha1后两组各1/2，扫描各1/6、定位各1/4。概率校正对同一源内的相对概率没有直接作用，因此不能单靠它解释或保证中心探测会变少。它也可能过度偏向小组、促成不合适的提前定位，需要实际对照验证。

分组完全从既有v3输入中的动作类别、覆盖点坐标和频道编号构造；padding不计数，候选序号不参与。直接在 `CandidateActorCritic.forward` 返回之前校正logits，所以行为采样、PPO旧/新log_prob、BC交叉熵、熵、greedy推理和checkpoint加载均使用同一个分布。没有只改训练采样而在优化时仍用旧概率的错误。

## 源内条件熵是实际行为分布诊断

每次PPO采样，对至少有两个probe候选的源计算 `H(A|channel, probe)`，并除以 `log(候选数)` 形成归一化熵。日志保存各状态/源出现次数、熵总和、归一化熵总和；合并批次时按出现次数加权，不能把不同局简单平均后当成同一分布。

中心概率由当前合法候选与该源外包集的最小包围圆圆心比较（1e-6米容差），**不把 option0 无条件认作中心**。这些标志在准备日志元数据时生成，没有传入actor/critic。只对中心候选存在且有多个probe选项的源统计条件概率；同时记录实际选中的probe/中心probe次数。BC阶段动作由teacher强制，所以BC和PPO日志必须分开解释。

每局 `sampling_probe_diagnostics` 与训练批同名字段提供：`source_groups`、`conditional_entropy`、`normalized_conditional_entropy`、`center_available_groups`、`center_conditional_probability`、`selected_probes`、`selected_center_probes`。无符合条件的组时均值为null。它们来自实际行为策略的forward，不是更新后网络重算，也不是greedy验证代理。

## 兼容和消融入口

checkpoint独立记录 `action_distribution`：旧文件缺少字段时默认 `{version:1,name:"flat"}`；组校正包含alpha和分组语义版本。与已有architecture字段分开，MLP/attention结构元数据不会被概率变化伪装。v1/v2只能用flat；同checkpoint直接resume必须匹配概率分布、架构、特征和源码摘要。

改变alpha须显式新trial。PPO `--initialize-from` 允许复制权重并改变概率，保存 `source_distribution`、`target_distribution` 和 `preserves_initial_probabilities=false`，以及单独的 `initialized.pt`。必须评估这个起点，不能把初始化重加权的收益记成PPO训练收益。其他旧caller（包括当前paired）调用 `initialize_from` 时默认不允许分布迁移，因而会明确拒绝把非flat策略静默当作flat使用。旧Namespace缺少group_alpha时默认0。

```bash
PYTHONPATH=src python -m pytest tests/test_deep_rl_distributions.py tests/test_deep_rl_attention.py tests/test_deep_rl_training.py tests/test_paired_reinforce.py -q
PYTHONPATH=src python research/benchmark_group_distribution.py --output results/rl/group_probability_cpu.json
# 以同一checkpoint分别做alpha0/alpha1，使用两个新目录和完全相同新增场景/预算
PYTHONPATH=src python -m research_rl.train --feature-version v3 --architecture mlp --group-alpha 1 --initialize-from /path/to/v3-source.pt --output results/rl/group-alpha1-001 --scenario-start 170001 --seed 9112033 --device cuda --hidden 96 --workers 4 --episodes-per-update 32 --updates 256 --max-wall-s 1800 --checkpoint-seconds 600 --deadline-utc 2026-09-11T06:00:00+00:00
```

分组计数当前用简单的候选两两比较，以减少难以审计的映射规则；140候选约19600个比较，需用附带CPU基准及完整训练采样耗时实测。诊断在CPU采样worker计算，推理加载默认不计算条件熵/中心元数据。两个alpha对照都启用相同诊断，避免日志开销只加在一边。本机无Torch，局部控制器回归79项通过；概率解析梯度、mask/排列、checkpoint迁移、真实PPO/恢复测试已编写，须由训练主机执行后再报告结果。

## 原始方法依据及区别

[Hierarchical Approaches for Reinforcement Learning in Parameterized Action Space](https://arxiv.org/abs/1810.09656)提供先选离散动作、再选条件参数的设计类比；本题这里全是有限离散合法动作，没有声称复现其连续参数架构。[Efficient Entropy for Policy Gradient with Multidimensional Action Space](https://arxiv.org/abs/1806.00589)说明多维动作下应区分策略结构与熵的计算；本模块候选总数有限，直接精确计算源内条件熵，没有使用或冒称其无偏估计器。上述公式是本次候选组消融的明确实现定义，并不由文献保证能减少本题时间。

## Regression provenance

`tests/fixtures/rl_network_d33074b.py` is the exact LF-normalized network source from commit `d33074b`, SHA256 `090b25830dedcf28102a7b61c99202472cdea9e76df909457bdf1346d7d36796`. The regression compares old/new flat logits and values exactly, then compares complete sampled v3 action histories under identical weights and RNG with the new training diagnostics enabled.

The training seed guard and recorded ranges now start at `100001`, matching the frozen protocol. Seed `100000` is rejected; historical training snapshots and completed trials are unchanged.

2026-09-11 remote verification reported by the training coordinator: the `f912dc8` snapshot passed 76 Torch tests in 7.97 s. Matched cold alpha0/alpha1 trials have started with the same diagnostics overhead and a shared 512-update target. No performance conclusion is available from those active trials yet.
