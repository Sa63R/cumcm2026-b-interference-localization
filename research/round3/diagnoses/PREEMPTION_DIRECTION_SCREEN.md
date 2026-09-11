# 每次测向后重新选任务：重复方向筛查

结论：本轮不为泛化的“解除单源锁定、只执行一个微步、然后重新排序”另开分支。廉价固定路线版本已试过；当前 RL 也已具备细粒度跨源选择。尚未同样验证的条件完整续跑版本需要新的估值可靠性依据，不能只换一个名称重试。此次仅核对既有源码、报告及已完成旧例记录；没有新代码、测试、世界、SQLite 或官方接口调用。

| 已有方向 | 已覆盖什么 | 与建议的关系 |
| --- | --- | --- |
| geometric preempt | 在真实 active/share 反馈后，A 仍未持证时，释放强制 A 首任务；比较完整固定 MEC/cover 路线，B 可是另一个源或覆盖点；每源最多中断一次，累计 probe 预算保留 | 已覆盖“固定宏路线作余程代理、反馈后跨源抢占”的主要廉价机制。负例的预计回访 A 未兑现，新发现及后续重排使 A 拖至末段。 |
| round2 feedback | 改成 B 覆盖扫描后立即恢复 A 的承诺，真实预算和清除正常计费 | 已排除“只需加恢复承诺就能修好”的简单推断。完整 64 对仍不支持收益：基准/候选均时 3102.767310/3107.152259 秒，平均旧 LB 1729.099752 秒，逐局均 T/LB 1.811424/1.813709；4 胜4负56平。局部领先仍可被后续路径反转。 |
| observation_tree / coupled_tree | 在 `_next_task` 宏边界，用条件假想世界比较首宏任务和完整 v1 尾；`V1Continuation.step` 的 source 执行整个 `_resolve`，cover 执行整个 `_scan` | 没有测试单次 bearing 后暂停。但完整条件尾模型已真正运行而未取得可靠改善，不能假定进一步细化天然解决后验近似、小支持和根择优问题。 |
| 当前 RL reference | `JointScanRLSearch._execute_plan` 每次真实单频道 coverage/probe/clear 后重新生成全部候选；保留预算兜底与覆盖 ledger | 微步动作空间已经存在于 RL，不能描述为尚未采用。已绑定旧 200114 日志有155次决定：79 coverage、60 probe、15 clear、1 fallback；这只是执行粒度证据，不是性能判断。 |

真正尚未以同样合同测试的是：在 A 的**真实新 bearing** 后，对每个候选微动作做合法条件观测生成，再运行完整原 v1 尾，以配对完整费用比较“继续 A”与切换。它与原宏树有明确动作粒度差异，但需要正确保留微步快照、每源累计 probe 预算、防饥饿/恢复约束、部分覆盖 ledger，并隔离训练选择与计价世界。若尾估值仍只是固定 MEC 路线，旧负例揭示的未知发现、后续重排和出口漂移缺口仍在；若换条件世界，则继承旧树的近似后验、有效支持和高计算开销。目前没有支持可靠统一误差界或收益方向的新证据。

建议先看当前 directed 完成成本代理的独立诊断。只有找到明确的公开反馈事件，并对“完整余程增量”建立比旧固定点代理更可靠的依据时，才登记有限新机制；不降门槛反复刷旧抢占案例，不把发现机会当作可兑现节省。这是停止重复低价值实现的依据，不是所有微步控制不可改善或现方法最优的证明。

证据路径均相对工作区 `D:\jwt\2026数模国赛`；SHA-256 绑定本次读到的版本：

| 文件或代码 | SHA-256 |
| --- | --- |
| `q3-geometric/research/geometric_preempt_v1.md` | `6ad622664b48a5e935b3a503b4ee3b04bee66e2ae79807c97683c1a7f92432c7` |
| `q3-round2/research/round2/diagnoses/feedback_coverage_dev_diagnosis.md` | `3ff1cde338ce2259e1563f87f1d45295b5e1062fcc58bd5519e8525130b3818e` |
| `q3-round2/research/round2/diagnoses/OBSERVATION_TREE_LIMITS.md` | `29e16243cde76dc7c1b63745150394109ecb070c28015aebd431b88b7c1c277d` |
| `q3-round2/research/round2/OBSERVATION_TREE_PILOT.md` | `1f91a5b6248e4de7a75b5509e8bff5e5bf1be363924ceaf51c5dd766333561c4` |
| `q3-round2/research/round2/COUPLED_TREE_PILOT.md` | `c3e22ae3ce3e58e4801532f94527e2eecc2793b1e025198736545506de0a62e5` |
| `q3-r3-rl-reference/src/research_rl/joint_scan.py` | `190a61032cedfb55d170a591f8b4d53efd8aa9138a78ad2790c396db5504fbd1` |
| `q3-r3-rl-reference/src/research_rl/controller.py` | `f0103793cff8da2ebd2140837b11362f3542b834aa56574ddd296064305f2a9c` |
| `q3-round3/results/round3/reference/model_smoke/records/rl-200114.json.gz` | `c5f7a49e7e082e6fb00d01fd08af65e3d7d7f4a72ef7e3ca31967dc2350ccaea` |

宏步源码另已直接核对 Git 对象 `a876e595:src/strategies/observation_continuation.py` 与 `149c612:src/strategies/observation_tree_state_search.py`；完整实现由 round2 的 `branch_archives/observation_tree.bundle`、`coupled_tree.bundle` 保留，无需恢复失败工作树。旧物理 LB 仅作事后参照，不是在线可达最优值。
