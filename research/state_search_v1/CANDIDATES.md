# 状态搜索分支候选清单与停止建议

该清单供共同评测选择，不是最终排名。主分支冻结原方法仍由父 agent 管理；本分支所有方法只用于 Q3。各候选必须用其完整 spec，不能省略配置后以函数默认值替代。下列路径均相对仓库根目录。

## 优先核验

| 候选 | 独立 spec（均在 experiments/） | 作用与当前证据 |
|---|---|---|
| **axis + inferred** | `state_search_candidate_inferred_silence_v1.json` | 当前保守优选待验：继承 axis，省略已知频道严格必然无信号的物理测量。新64训练平均省13.250 s，95%区间[9.750,16.969]，37胜0负27平，全部全清；剩余动作与axis对应删除后的记录完全一致。仍须Linux公共及独立测试。 |
| **axis 原版** | `state_search_candidate_axis_quantile_v1.json` | 保留主参考候选：有限任务掩码搜索、精确代理剪枝、至多23个几何主动测点、清除后共享。64训练相对首版省40.81 s且区间为正；父agent Linux公共48验证相对旧rollout平均快7.68548%，45胜3负。不是原题最优证明。 |
| mean_point + inferred | `state_search_candidate_inferred_mean_point_v1.json` | 薄组合备选；新16训练相对axis+inferred平均慢3.180 s，区间跨零，7胜9负。全清及推断账通过，暂不推荐替换。 |
| mean_point | `state_search_candidate_region_mean_point_v1.json` | 单独调度消融：12面积节点同半径质量，使用均值点旅行矩阵。64训练平均省23.074 s、Linux48相对axis省18.410 s，两区间均跨零，存在大单局退化。 |

axis+inferred 的单独证据强于“叠加更多模块即会更好”的假设。主统计应等待同平台共同评测；不要把训练集、公共验证、扩展验证和最终测试混在一张平均值里，也不要将不同平台相同 seed 直接配对。

## 保留的其他独立 spec

| 候选 | spec 文件名（experiments/） | 用途 |
|---|---|---|
| 首版主动九点 | `state_search_candidate_v1.json` | 原始有限状态+九点几何代理基线；记录了相对原方法的主要收益来源。 |
| 精确剪枝首版 | `state_search_candidate_pruned_v1.json` | 与首版代理评分及行为等价的加速消融；不能把减少计算时间当作减少虚拟移动时间。 |
| 局部格点细化 | `state_search_candidate_local_refine_v1.json` | 保留旧九点并在旧最优附近细化；新64训练改善33.73 s，但本机计算比axis更贵，主要作为动作构造消融。 |
| 完整期望距离 | `state_search_candidate_region_expected_distance_v1.json` | 与mean_point对照区域尺度；64训练和Linux48相对axis的区间均跨零，不优先重复扩展。 |
| active共享 | `state_search_candidate_active_sharing_v1.json` | 阴性对照：16训练平均慢5.948 s，额外检测抵消移动和原after-clear共享；按停止条件未做64。 |
| 只扫未知频道 | `state_search_candidate_unknown_only_v1.json` | 阴性对照：16训练平均慢90.362 s，少测虽省78.688 s，却多走169.050 s；不能把已知覆盖读数统统视为冗余。 |

## 有代码与日志、没有推荐为候选的研究路线

- 更精细的有限源/噪声反馈树及分箱观测：发现离散假设的观测退化和过度确定性；增加深度/节点未稳定改进完整局。见 `FINITE_PROBE_TREE.md`。
- 环相位和半径自适应布局：有全圆盘解析覆盖证明，有限评分选环仍未改善实际时间。见 `LAYOUT_AND_EXACT_PRUNING.md`。
- 未知源质量与未来路线插入费用：同一个R交集、源数先验组合因素和未知频道质量均显式建模；16训练只省6.45 s且区间跨零、单局有退化，未继续64。见 `UNKNOWN_INSERTION_RECOURSE.md`。
- 原子扫频中断诊断：首次发现后剩余约61 s并不是可直接省去的成本，剩下仍有其他未知频道；先清除再返回可能新增更大路程。见 `INTERRUPTIBLE_SCAN_AUDIT.md` 和 `UNKNOWN_ONLY_COVERAGE.md`。

这些日志保留了负结果及原因，不能将它们删除后只展示最佳小样本。也不能因多条尝试失败就宣布达到不可改进极限。

## 可以证明与仍未解决的部分

冻结点/非负对称矩阵任务模型中，`(visited_mask,last)`支配、MST加入口界和预算终止时上下界成立；有限候选的非负评分界可安全剪枝。实际轨迹显示当前有限任务问题已接近完全求解，因此继续提高A*预算没有明显依据。

这些界没有展开未知源发现、未来连续反馈和定位路径，**不是Q3整体最优值的上下界**。覆盖及清除使用真实历史的保守几何；推断删测只是逻辑确定反馈的局部费用删除。其余启发式收益必须由独立完整局评估，而不能由局部代理分值下降推出。

截至该清单，状态方向暂停新策略扩展，等待父agent统一验证各冻结候选。建议先核验axis+inferred，并保留axis作为可回退参考；mean_point组合只作有限备选，不做无根据参数扫。

后续只做了[冻结训练压力可靠性评估](TRAINING_STRESS_RELIABILITY.md)：独立7类×4场景、四份源码快照，rollout/axis+inferred/几何single/RL002u384全部28/28全清、零失败清除，112条历史及费用账通过。axis+inferred本组平均3047.572s，较rollout3584.967s低14.99%，但这组全R=1000的指定压力分布不能替代正式最后测试或证明普遍占优。没有据此更改策略。
