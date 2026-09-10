# 几何分支候选入口与冻结参考

本索引只组织已经实现和归档的配置，不改变主分支、共享协议、Q4 或任一策略默认值。所有配置均在 `research/` 下，使用共同 `experiments.research_v1_eval` 入口；最终策略选择须依据相同平台、相同场景的公共验证，再由根代理统一冻结。不同实验批次的百分比不能相加，Windows 与 Linux 生成器 case hash 不一致时不能混成配对样本。

|配置文件|修改内容|独立几何实验状态|建议用途|
|---|---|---|---|
|`v1_baseline_efficient.json`|最初几何基线|共享冻结配置|基线对照|
|`v1_baseline_rollout.json`|原有浅层前瞻，使用真实冻结参数|共享主要基线|主要基线对照|
|`v1_geometric_clear_only.json`|只在成功清除后择机跨频道测量|103001–103080 完成|机制消融|
|`v1_geometric_joint.json`|增加主动探测点的择机跨频道测量|64 确认相对 rollout 平均省 53.118 s，CI [14.390,90.756]|几何共享基座及对照|
|`v1_geometric_probe_single.json`|joint 加首次主动探点的完整单源费用代理|107017–107080 相对 joint 平均省 61.326 s，CI [39.140,81.830]|优先公共验证候选，计算较省|
|`v1_geometric_probe_cross.json`|single 加一个邻源的增量共享信用|同 64 场景相对 joint 省 66.297 s；相对 single 省 4.972 s，CI [−7.766,19.541]|保留公共验证，跨源额外优势未确立|
|`v1_geometric_centroid_first.json`|首探 MEC 改为保证接收域内的安全质心|16 pilot 平均慢 97.159 s，未扩到 64|负结果档案，不推荐|
|`v1_geometric_route_clear.json`|原安全清除圆内考虑后继点的局部凸几何优化|64 确认平均省 0.636 s，CI [−0.898,1.533]|小规模可证局部优化/全局弱结果，不作为主候选|
|`v1_geometric_probe_radius_width.json`|single 的假设费用按兼容 R 区间宽度加权|112001–112016 平均慢 3.530 s，CI [−11.846,2.535]；按协议停止|负结果档案，不推荐|
|`v1_geometric_probe_preempt.json`|不确定源定位中途按固定目的地路线选择性中断|115001–115016 平均慢 26.682 s，0 胜2负14平；停止扩样|负结果档案，不推荐|
|`v1_geometric_probe_relocation.json`|single + 全域覆盖凸可行域内连续移站，原2-opt排序|116017–116080 平均省 17.412 s，CI [−8.798,42.400]；45胜19负、最坏慢320.046s|保留公共验证，额外优势未确立；不改默认|

上述实验全部使用本地研究仿真，涉及的几何研究组均全清且零失败清除。这是所列样本上的安全记录；覆盖/清除的数学认证另有模型前提，不把有限测试记录解释为无条件全场景证明。

详尽依据、费用分解、异常局及来源身份分别见：

- `geometric_joint_design.md`、`geometric_joint_confirmation_v1.md`：联合观测及 64 场景证据。
- `geometric_probe_cost_v1.md`：首次主动探点单源/cross 设计、代理局限、纠错记录及 64 确认。
- `geometric_centroid_first_pilot_v1.md`：为什么更接近源的代理不等于从当前位置出发总路线更短。
- `geometric_route_clear_v1.md`：安全圆内局部路径问题的最优性检验及下游信息分叉。
- `geometric_probe_radius_width_v1.md`：R 区间积分、非精确后验声明及停止理由。
- `geometric_preempt_v1.md`：受限任务中断的理论前提、两条负例及实际尾策略与固定路线代理的差异。
- `geometric_relocation_v1.md`：连续覆盖可行域、纯几何2-opt费用代理、逐站实际执行审计及配对证据。

`results/geometric_joint/` 保存各批全部压缩记录。候选源码是普通 Python，无 Torch、SSH、网络或隐藏真值读取依赖；所有实际几何更新均来自可观测接口。主要推荐候选共同使用 `strategies.geometric_probe_cost:run_probe_cost_search`，分别指定 `probe_mode=single/cross`；不指定 `hypothesis_weighting` 即原 equal 方式。
