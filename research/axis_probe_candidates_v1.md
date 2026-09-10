# Q3 RL 轴向探点候选扩展

这是独立的候选动作集对照，入口为 `--feature-version v3 --probe-candidates axis_quantiles`。它不增加状态规划器，不用几何评分替代actor选择。已有v1/v2/v3默认动作集仍是 `base`；60维输入没有改成一个同维数的假“v4”，检查点另行记录严格的 `action_schema`。

## 为什么值得尝试，又为什么不能预设收益

旧候选包含baseline点、最小包围圆中心、按首次方位横向偏移的两个点、当前位置与中心的中点、当前位置。它没有沿**当前**观测外包区域长轴的分位位置；第一次测向之后区域方向变化时，首次方位横向点也未必对应当前几何结构。

本次用110001—110032共32个训练场景，仅在teacher原轨迹上检查候选机会。6232个源状态都存在新增且保守保证可接收的位置，旧候选均4.743个，平均可增加10.509个（最多14个）；旧集只覆盖1.174%的合法轴向模板。原候选峰值169，扩展候选峰值271。缓存模板生成均2.09毫秒/局，额外20维几何特征均11.00毫秒/局；这一局部计时不包含完整网络、基础特征、优化padding和概率校正。

随后在同32场景、2次交替重复中，用相同MLP权重执行teacher动作并计算实际网络前向与诊断，共64组完整配对。旧候选均0.41274秒/局，新增候选均0.54580秒/局，**CPU整局开销增加32.24%**；其中前向由0.1562增至0.1908秒、完整特征由0.1329增至0.2181秒。两边物理轨迹逐值相同。这说明新增点构造本身很便宜，但把全部新增点交给特征/网络处理有实质成本，不能用前述13毫秒局部计时宣传整局仅增加13毫秒。训练时的采样轨迹与GPU优化成本还需主任务实测。

机会审计32局的所有原teacher物理动作及总虚拟时间逐值相同，**没有执行新增候选**，因此这里只说明旧动作集确实缺位置，不是新策略性能提升的证据。主任务在既有最佳PPO384验证轨迹中发现主动probe约59%选择当前位置、约16%选择中心；其主要收益可能是就地跨频道测量与路由选择，不能把旧BC模型偏爱中心的诊断直接套到这个更好的模型上。

## 候选的明确数学定义

在当前凸外包多边形顶点中，取欧氏距离最大的顶点对，其单位方向为 `u`，垂向为 `v`。令最小包围圆中心为 `c`，顶点对 `u` 的投影范围为 `[l,h]`，`w=h-l`，`s=min(200,max(20,w/8))`。

- 取 `q∈{1/4,3/8,5/8,3/4}`、`t∈{-s,0,s}`，产生 `c+(l+qw)u+t v`，共12个点。
- 再产生 `c-sv`、`c+sv` 两个点。
- 只保留对全部外包顶点最大距离不超过 `1000-1e-7` 米的点；再排除已测位置和已有候选的六位小数坐标重复。

该接收过滤是保守条件：若未知真源在凸外包多边形中，则它与测点的距离不大于各顶点距离的最大值；Q3接收半径的已知下限为1000米。没有读取实际源位置、实际半径、训练种子或模拟器真值。此过滤只应用于新增点，**不删除或重新排序旧候选**。

新增点借鉴的是状态分支已研究的轴向几何候选族，RL实现独立写在 `src/research_rl/axis_probes.py`，只生成坐标，不导入 `choose_radius_probe`、代理代价或搜索模块。也没有假想测量、蒙特卡洛未来世界或额外信息奖励。

## 特征、概率与迁移

旧候选每一行60维特征保持原值。所有新增点共享 `option=6`，第18列仍按 `option/6` 编码，故新增族值为1；它表示“轴向候选族”，并非候选顺序的任意6—19编号。坐标、移动代价、外包几何等已有特征区分族内位置。

新增候选会改变mean/max集合编码，也会改变softmax分母，即使复制所有旧权重，旧动作的概率也通常不同。迁移使用新输出目录与显式 `--initialize-from`，保存 `source_action_schema`、`target_action_schema`、`preserves_initial_probabilities=false` 和 `initialized.pt`；后者必须单独验证，不能把初始化时动作集变化的收益归为训练收益。`resume`不允许换动作集。

旧文件没有action_schema时默认base。旧paired及其他caller的 `initialize_from` 默认拒绝跨动作集，防止将扩展模型权重静默装进base控制器。推理工厂按元数据选择控制器，显式请求的候选族与检查点不符会报错。alpha0/1与动作集是两个独立维度，本次应先保持alpha0和MLP，避免同时更换概率校正或注意力。

## 完备性与成本边界

新增候选只出现在旧策略仍允许主动probe时。每源最多6次主动probe、近点/包围圆清除证书、七点频道账本、16源上界、失败后的保守兜底均继承且未改。即使actor持续选择新点，学习决策的原上界 `140+16×7=252` 仍成立；没有因为候选增多而提高物理试错上限。

每源原始候选至多6个，现在至多20个；一个保守的全候选数量上界是 `140+16×20=460`。实际同时可行状态通常较小，但不能继续假定最大只有140。MLP前向开销随候选数增加；若以后结合alpha1，两两分组计数还会随候选数平方增长。当前只量出局部构造开销和teacher完整前向开销，真实PPO优化与训练收益由主任务另做对照。

`axis_probe_measurements` 和 `base_probe_measurements` 统计被记录的真实主动测量，新增点动作带 `rl_probe_family=axis_quantiles`。它们不把teacher标签或仅被提供的候选计作执行。继承的 `interrupted_scans` 是覆盖点尚未扫完时执行源动作的状态计数，不是独立中断事件数量。

## 复现和验证

```powershell
$env:PYTHONPATH='src;.'
.venv-win\Scripts\python.exe research/audit_axis_probe_candidates.py --seed-start 110001 --count 32 --output results/rl/axis_candidate_audit_110001.json
.venv-win\Scripts\python.exe research/benchmark_axis_candidates.py --seed-start 110001 --count 32 --repeats 2
.venv-win\Scripts\python.exe -m pytest tests/test_deep_rl_axis_probes.py tests/test_deep_rl_distributions.py tests/test_deep_rl_training.py tests/test_paired_reinforce.py -q
```

首轮本地Torch验证74项通过；随后包括controller/joint-scan/cache/distribution/attention/training/paired/portable-checkpoint在内的完整相关回归175项通过（34.85秒）。验证包含新点接收保证、旧teacher动作及旧特征逐值保持、强制选择新点仍全清、真实动作标签、检查点加载/迁移拒绝、实际PPO更新和CLI恢复。这里尚无新动作集的固定验证成绩，也没有启动远端训练。

原始机会审计与64组计时保存在 `axis_candidate_evidence/opportunity_110001.json`、`axis_candidate_evidence/teacher_cpu_64pairs.json`，包含训练种子、每局记录、脚本SHA256和网络权重摘要。它们是CPU实验数据，不是官方演练或冻结验证成绩。
