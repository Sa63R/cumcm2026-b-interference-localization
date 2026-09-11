# 观测树单次仪器剖析：重复快照与完整几何尾策略

本次只运行一次已打开的开发案例 **200114**，使用冻结候选 `a876e595d14fdf067852ddf54078aa99292de9ca` 和现有 `experiments/observation_tree_model_smoke.py`，`depth=2`、harness 固定 `max_searches=1`。`--seconds 120` 仅容纳 cProfile 开销，不是新的算法性能配置。没有修改候选、读取 SQLite、新 pilot 或运行其他案例。

主要结论：本次仪器运行的完整条件采样累计仅 **0.822 秒**；大量工作发生在重复运行完整 v1 尾策略，其中深复制累计 **43.405 秒**，覆盖 oracle 累计 **27.170 秒**。最明确的可消除工作是：尾策略的内部循环每完成一个宏动作都建立完整快照，尽管该循环丢弃返回的 `MacroResult`。这比继续减少 optional 的固定4096提议更值得先检查。

累计时间有嵌套，以上数字不能相加组成总时长或收益预算。本次因仪器开销达到120秒规划限制，只完成部分比较；不能直接标定无仪器的完整48尾成本。

## 1. 运行范围、结果和复现

运行前核对下列输出不存在，再执行一次：

```powershell
& '..\cumcm2026-b-interference-localization\.venv-win\Scripts\python.exe' -m cProfile `
  -o 'results\round2\observation_tree\model_smoke\profile.stats' `
  'experiments\observation_tree_model_smoke.py' --depth 2 --seconds 120 `
  --output 'results\round2\observation_tree\model_smoke\depth2-profile-120s.json.gz' `
  > 'results\round2\observation_tree\model_smoke\depth2-profile-120s-console.json'
```

工作目录是 `q3-round2`。只占用一个仿真进程；该进程退出后才读取 pstats 生成摘要，没有重跑。原harness标签仍为开发烟测；本次候选实际冻结版本与53个源码文件的前后SHA256均已核对，`source_unchanged_during_smoke=true`。

| 项目 | 本次结果 |
| --- | ---: |
| 规划墙钟 | 120.126929秒 |
| 整个harness运行墙钟 | 123.722684秒 |
| 启动尾评估 | 33次，其中前32次完成，最后1次预算中断 |
| 内部生成measure/clear | 3306条 |
| 条件节点 | 5个 |
| 已完整完成第二层比较 | 4个，2次子动作改变 |
| 最终规划状态 | fallback：`incomplete_simulated_tail:continuation_compute_budget` |
| 真实执行结果 | 16/16全清，0失败清除，合法退出 |
| 真实虚拟时间T | 3361.326128秒 |
| 既有物理先知下界LB | 1921.819589203秒 |
| T/LB | 1.749033128 |

LB来自已有200114审计，harness仅核对该旧开发场景的事后哈希；不是新推导的可达信息下界。规划最后回退到v1，因此这里的T/LB也不是观测树获得的新性能。动作费用审计无错误，账本重建T与报告一致。

证据文件：

- [原始运行记录](../../../results/round2/observation_tree/model_smoke/depth2-profile-120s.json.gz)
- [cProfile二进制数据](../../../results/round2/observation_tree/model_smoke/profile.stats)
- [累计/独占时间前25项](../../../results/round2/observation_tree/model_smoke/profile-top.txt)
- [结构化摘要及源码哈希](../../../results/round2/observation_tree/model_smoke/profile-summary.json)
- [harness控制台记录](../../../results/round2/observation_tree/model_smoke/depth2-profile-120s-console.json)

`profile.stats` SHA256：`51575f7f36b684ee62ee9d74e9a18941c697f2f043cbb1deaed4b24c489863cf`。

原始压缩记录SHA256：`76fb7badb1faa05a3a345a29581dc43d9956ace692123575f549fae606bc4885`。

## 2. 函数累计时间与独占时间

累计时间包含被调用函数；独占时间仅为该函数自身执行。下面按函数列示，**不是互斥分区**。表中的调用次数含递归调用时另作说明。

| 函数 | 调用次数 | 累计/秒 | 函数自身独占/秒 | 含义 |
| --- | ---: | ---: | ---: | --- |
| `observation_tree_state_search.evaluate_tail` | 33 | 118.020970 | 0.002992 | 包含分支初始化、强制首任务及完整/中断尾 |
| `V1Continuation.finish` | 33 | 113.585277 | 0.000222 | 首任务后的v1收尾调用 |
| `V1Continuation.step` | 601 | 117.496686 | 0.018950 | 与finish/evaluate_tail大量嵌套，含几何行动和快照 |
| `copy.deepcopy` | 30775021总调用 / 2068非递归入口 | 43.404972 | 18.608328 | 深复制的总代价，不能再加其helper累计 |
| `capture_v1` | 607 | 42.420368 | 0.063387 | 几乎全部时间进入深复制 |
| `V1Continuation.__init__` | 38 | 0.857001 | 0.002628 | 构建新分支时的必要状态复制 |
| `V1Snapshot.history` | 139 | 0.089309 | 0.000404 | 历史副本；在本次不是复制主因 |
| `RelocatingStateSearch._next_task` | 589 | 29.557353 | 0.064383 | 包含覆盖重定位、证书与路线搜索 |
| `feasible_ray_point` | 5346 | 27.812995 | 0.239078 | 包含多次覆盖oracle调用 |
| `CoverageOracle.__call__` | 117352 | 27.170064 | 0.648354 | 包含缓存命中和实际几何核验 |
| `disk_cover_radius` | 71864 | 25.636866 | 3.786271 | 原完整覆盖几何，包含大量内层距离求值 |
| `sample_conditioned_worlds` | 11 | 0.821927 | 0.005112 | 整个条件世界采样调用 |
| `_conditional_pool` | 81 | 0.757564 | 0.112843 | 包含所有提议、R约束和量化似然 |

以文件为组直接相加函数**独占**时间，可以得到不重复包含子函数的另一种统计：`copy.py`为33.787091秒，`q3_observation_likelihood.py`为0.252289秒，`q3_belief.py`为0.158920秒，`coverage_relocation.py`为1.314715秒，`disk_cover.py`为6.103998秒。它们不包含在其他文件或内建函数中执行的工作，不能替代上表的功能累计时间。

深复制的`_deepcopy_tuple`本身被调用9567349次，独占11.600831秒。覆盖核内的`min`、生成器和距离运算也分散在其他函数，故仅看`disk_cover_radius`的3.786秒自身时间会低估完整证书代价。

还有一块重要的原v1工作：`choose_radius_probe`调用1004次、累计44.969786秒；`_next_probe`累计45.038257秒。这些属于源定位宏动作及完整尾，而不是新的条件采样。其几何裁剪函数也可能被其他组件调用，不能把这些累计值与整条尾或覆盖/复制累计简单求和。

原pstats共记录407102365次函数调用，平坦独占汇总112.716894秒；harness墙钟123.722684秒。两种计时口径在本次仪器运行下不相同，本报告不把任何一组累计时间归一化成“总墙钟百分比”，也不以其差额推断某个未测组件的耗时。

## 3. 能定位到代码的重复工作

`observation_continuation.py:229`的内部`_execute_plan`循环调用`self.step()`，但不使用返回值。`step`在第224行附近无条件建立：

```python
observations = tuple(deepcopy(self.report.action_history[before:]))
return MacroResult(self.snapshot(), chosen, observations, ...)
```

`self.snapshot()`再调用`capture_v1`，深复制整个政策状态、报告、历史、区域和覆盖oracle等对象。本次601次step和607次capture相符；直接调用者统计显示，`capture_v1 → deepcopy`累计42.347239秒，而新分支初始化时的`__init__ → deepcopy`约0.835115秒。问题主要是每一步都创建结果快照，不能笼统归咎于构造33条分支本身。

`evaluate_tail`对强制首任务也只执行`continuation.step(task)`而不消费MacroResult；根宏执行虽赋给`first`，后续使用的是`prepare_decision()`返回的另一个快照。生产路径里存在明确的无消费快照，但公开step API和人工等价测试仍需要默认完整结果。

因此下一项有依据的性能改动是：仅对明确不消费结果的尾执行路径，增加不建立返回快照的内部执行模式；真正进入观测树子节点时仍完整capture，保持sibling隔离、pending task、route/oracle预算和行为账本不变。这应在另一个版本中做逐动作等价与无污染验证，不能在正在运行的冻结pilot上热改。消除本次42秒累计工作的理论上限也不是无仪器可兑现的42秒收益，须另外验证。

覆盖oracle和主动测点几何确实是剩余的大块工作，但它们目前承担原v1的合法行为与完整证书。不能为了提速直接禁用其调用，再声称仍用相同v1尾；需要保留相同结果的缓存/计算优化及相应等价检查。

## 4. 本次证据能够支持的决策

1. 不应把当前几十秒开销首先归因于固定4096条件提议。本次含11次条件采样仍只累计0.822秒，削弱证据估计规则的收益空间很小。
2. 可以优先验证“去掉无人消费的中间快照”这一实现层优化；它不需要改变后验、K、候选、门槛或真实任务策略。
3. 若只看冻结pilot的回退率，须区分模型支持不足、实际预算不足与仪器造成的中断。本次是后者，没有生成新的独立退步案例。
4. 当前profile止于第5个条件节点的第1条训练尾，未覆盖完整48尾以及第2/3次真实规划。不同任务族、历史长度和机器负载会改变比例；本次不能作为全任务吞吐标定，也不能与未加仪器的pilot墙钟直接比较。

本次只做了上述一轮剖析，未修改任何冻结策略文件，未为获得更好profile结果追加运行。
