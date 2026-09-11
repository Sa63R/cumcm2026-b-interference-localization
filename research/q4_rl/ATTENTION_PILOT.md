# 同输入的 induced attention 消融与 CPU pilot

本分支基于 `d9a7a93`，只扩展网络架构及训练配置。控制器、10/16 维输入、动作集、gamma=1 完整计费回报均保持原样。MLP 仍为默认，旧 MLP checkpoint 已实际加载；23 项新旧训练测试通过，覆盖置换等变、padding、CPU 限制、优化器/RNG 与下一次更新恢复。本文中的训练审阅不等于已验证模型有效。

## 已完成的微基准

使用 CPU PyTorch 2.9.1、1 个数值线程、440 个候选、相同固定随机特征；每配置在独立进程运行。B32 预热5次、记录30次，B128预热5次、记录10次。更新包含 PPO 张量损失、backward、梯度裁剪和 Adam，不包含 Python 轨迹打包、场景生成或模拟器。峰值为进程工作集，包含解释器与 PyTorch，不是模型精确分配量。

| 网络 | B32 前向中位/ms | B32 更新中位/ms | B32 峰值/MB | B128 更新中位/ms | B128 峰值/MB |
|---|---:|---:|---:|---:|---:|
| 原 MLP64，26690参数 | 1.294 | 53.210 | 307.4 | 236.172 | 429.6 |
| attention，内部64，norm+FF，78146参数 | 3.929 | 192.468 | 355.1 | 未测 | 未测 |
| attention，内部16，norm+FF，34242参数 | 3.083 | 139.551 | 342.8 | 未测 | 未测 |
| **attention，内部16，residual，31858参数** | **2.884** | **112.371** | **322.3** | **512.423** | **484.2** |

因此将最初版本缩小为16 latent、2 heads、1 block、内部宽度16、外层宽度64、residual。相对初版，B32更新下降约41.6%；相对MLP仍约2.1倍，不能假定整体训练更高效。完整宽度/norm+FF版本仍由超参数保留。上述是计算基准，没有实际策略动作，不定义虚拟计费时间或任务T/LB。

原始逐次计时与配置在 `results/q4_rl/attention-microbench/`，重建配置见 `attention_configs.json`。最早两种原始JSON没有后来加入的显式维度/细化字段：`induced16-b32`隐含内部64/norm_ff；`induced16-dim16-b32`隐含内部16/norm_ff。其当时源码哈希保留；当前实现仍支持这两种结构，不能把不同时刻计时视为精确硬件常数。

复现微基准（项目根目录，设置 `PYTHONPATH=src`）：

```text
python tests/test_q4_rl_attention.py --benchmark mlp --batch-size 32 --repetitions 30 --output results/q4_rl/attention-microbench/new-mlp.json
python tests/test_q4_rl_attention.py --benchmark induced --attention-dim 16 --attention-refinement residual --batch-size 32 --repetitions 30 --output results/q4_rl/attention-microbench/new-induced.json
```

## 训练信号审阅

当前 `train.py` 的采样策略只接收公开特征，worker外层在退出后读取全清结果和LB。`records`只由实际学习决策构成，训练器逐条核对其特征、动作、旧概率与controller记录一致；隐藏场景/LB只在独立诊断字段中，没有传给actor/critic或按LB缩放回报。实际费用经 `attach_returns()`核对完整总和，再计算不折扣剩余成本，具备可学习的真实成本信号，但尚无证据证明信噪比足够。

**128决策的语义。** 达到上限后整个可靠兜底费用加到最后一条transition，再经gamma=1传播至所有此前return。这没有漏算或只奖励短前缀，但优化对象是“最多128次学习决策+固定尾部”。服务宏动作可以消耗多次实际动作；128也不等于128次HTTP动作。若尾部占比很高，网络主要学习如何给固定尾部安排状态，不能声称已经独立控制全程。所有步骤共享大幅尾部费用还会增加优势相关性和critic误差。

critic没有显式剩余决策额度，可能混合不同剩余horizon的相似特征状态。当前评估默认controller上限2048，而拟议训练上限128；pilot的所有学习/启发式对照必须在spec中显式固定 `max_decisions=128`。之后128→256/512属于新的horizon实验，不能把收益全部归于attention。本分支不为解决此问题增加新输入。

**warmstart语义。** 模仿期使用同一动作集中的 `_heuristic`，不是逐动作模仿冻结R8完整调度；两者只有源处理模块和安全兜底部分共用。教师用最小频道编号打破部分同分，而输入没有数值频道ID：完全相同特征的多个候选必有相同logit。one-hot BC存在不可消除的tie损失，原始CE不应要求接近0。应补充“与教师特征完全相同候选组”的总概率/命中率作诊断；不将不可区分标签当实现失败，也不偷偷给attention补频道ID。模仿的critic也只拟合该教师+128尾部的return，PPO切换后需重新校准。

## 建议的小规模验收

这是待root安排的pilot，尚未执行。可先在新训练seed各做32局暖启动+128局PPO，两个架构使用同场景、同动作与horizon、相同CPU上限；记录额外实际CPU秒。另列相同CPU预算下的结果，避免忽略attention约2倍更新成本。开发请求只取尚未用于模型选择的新810xxxx子范围；确认和最终集继续封存。

1. **正确性必过**：每条轨迹费用总和误差≤2e-5秒；失败原始记录及惩罚保留；全清与覆盖/物理审计全部通过；无NaN、无越界动作、无未解释的中断。checkpoint恢复后固定诊断输入概率及下一更新可复现。
2. **学习信号诊断**：相同特征候选组的teacher概率/准确率、train/dev完整cost、critic的MAE和explained variance、优势标准差、策略熵/KL、学习决策数、service频率、fallback费用占比与最坏值。均值只降loss、不降完整cost不算有效。
3. **配对开发主表**：R8、启发式128、MLP128、attention128同场景配对，报告全清率、失败清除、均值/p95/最大值、计算开销、统一LB及两种T/L聚合。冻结warmstart及pilot末checkpoint一并报告，不能只挑最有利checkpoint。小样本CI仅作诊断，不作推荐版本晋级。
4. **是否扩展**：attention需在相同输入与完整时间口径下相对MLP显示重复可见的开发收益，并解释额外CPU。若只有teacher拟合更好或已知尾部更重，先改善训练/控制信号；若两个预登记资源档位均无收益且更贵，暂停该配置，保留记录。不得因本pilot无收益宣布attention或Q4强化学习无效。

128上限、有限摘要和固定service仍留下可改进空间。该分支只是关系表示消融，不构成全局最优性或完整联合控制的结论。
