# 有限条件观测树：冻结前记录

本次只研究问题3，不写论文，不调用模拟器或读取验证SQLite。受保护v1提交为8c624d0，独立候选分支research/q3-r2-observation-tree提交a876e595。源码、配置、模型门控和新案例范围均在首次运行新案例前归档。

## 待检验假设

原基准已经会在真实反馈后重规划。本次检验：在行动前考虑“做完一个完整任务后得到不同反馈，再据此选择下一个任务”，是否能改善发现和清除的整局调度。三组依次为原v1、同模型的根节点MC、两层条件MC。两层版本没有实现完整POMCPOW、UCB或渐进扩展，不称为强化学习。

两新组使用同候选生成、条件模型、完整v1尾策略和计算上限。每次最多48次尾评估启动、40000条内部动作、90秒；整局最多3次/270秒。根节点组每代表用8个尾估值；两层组用2个世界选后续动作，再用独立2个世界计价。两组估值精度不同，必须同时比较两新组与原v1、两层与根节点，不能将小样本低估归为策略收益。

## 实现门控

61项实际模块/集成测试通过；旧开发案例200114的三次当前源码检查全部完整清除、零失败，均3361.326128秒，旧LB为1921.819589203秒，T/LB为1.749033128。禁用新模块完整128条物理历史与已归档v1一致。两新组完成48尾；两层实际产生6次第二层比较、2次内部改动作，最终根选择仍为v1。此为实现检查，不能加入独立性能样本。

原60秒计算门槛下，两层在44次尾启动后中断回退。为完成相同预定比较，首次新评估前将两新组同时调整为90秒/270秒，保持10秒接纳门槛。原记录全部保留。depth2-v0记录运行时实现尚在整理，只属未冻结开发证据；当前门控仅绑定带源码前后hash的90秒检查。

机器门控在OBSERVATION_TREE_MODEL_GATE.json，独立审核在diagnoses/MODEL_GATE_REVIEW.json。runner在创建新案例目录前核对证据、源码和配置hash；归档manifest也保留门控hash。任何源码改动均须重新检查、登记新实验，不能沿用旧门控。

## 案例与停止规则

固定pilot为212101..212116，每案三组，共48局。新的独立confirmation为212201..212264；final为214001..214256；stress为215001..215028。已被certified_tail打开的210xxx/211xxx不再当作此实验的独立案例。不给策略实际seed、编号或隐藏真值；runner只在结束后读取真值作审计及旧LB。

沿用round2的预先固定配对门槛，pilot无足够收益就不开confirmation，不因单局获胜追加案例。两层未通过时，不以“回退安全”宣称树有效；应报告成功计算比例、回退原因和实际机制覆盖。未过性能门槛不等于POMCPOW理论无效。

整局指标包含完整清除/可靠退出、failed clear、平均/P95/最大虚拟用时、配对差值与区间、CPU/墙钟以及旧T/LB。旧LB是先知物理松弛，不能把比值当可达在线最优差距。当前模型是声明先验及有限条件池近似，不是官方已知后验；K=2是有限积分，也不是充分统计保证。

## 复现

先在候选树运行61项模块测试；在archive树运行experiments/observation_tree_model_gate.py生成一次性门控，然后执行：

```text
python experiments/round2_runner.py prepare --trial observation_tree --stage pilot --spec experiments/state_search_candidate_observation_tree_r2.json --output results/round2/observation_tree/pilot
python experiments/round2_runner.py run --output results/round2/observation_tree/pilot --workers 3
python experiments/round2_posthoc_audit.py --batches results/round2/observation_tree/pilot --output research/round2/observation_tree_pilot_audit.json --report research/round2/observation_tree_pilot_audit.md
```

已有输出不要覆盖。源zip和runner副本随manifest保留；冻结后只读诊断，不修改运行代码。官方演练由用户采集脚本管理，本批不占用该接口。
