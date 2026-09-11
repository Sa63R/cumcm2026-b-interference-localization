# 续跑内部丢弃快照：单机制优化设计与等价边界

结论：值得做一个独立、可关闭的计算优化试验。`V1Continuation.finish()` 内部循环不消费 `step()` 的返回值，每次仍复制完整状态。可以只省去这些返回值的物化，保留原宏动作执行和所有边界检查。**有限真实时间下不能无条件声称整局逐动作等同，也不能声称所有日志数值完全相同**：更快可能跨过原先的规划截止边界，运行耗时字段本来就应变化。

本文件是只读设计，没有修改候选、建立分支、运行场景、访问 SQLite、调用官方接口或读取进行中的 pilot。未来实现/测试需要另行登记版本；不热改当前冻结 `149c612`。

## 1. 源码与已有证据

检查源为 `q3-r2-coupled-tree/src/strategies/observation_continuation.py`，SHA256：
`d413c473fd572c9036af4987e508ec3b4041398caf959bdaa09003ace700d7d0`。
它与旧 profile 记录的该文件完全一致，虽然外层评估模式已经更换。

- [既有仪器报告](OBSERVATION_TREE_PROFILE.md)：已打开开发 200114 的一次 cProfile 运行，规划墙钟 120.126929 秒，33 次尾评估启动，最后一次中断。不能当作新的性能批次。
- [profile 摘要](../../../results/round2/observation_tree/model_smoke/profile-summary.json)，SHA256 `05eb922b7eeb6eba3d0308396a22f1a9b88fcd61acd743fffabe89884031cad8`。
- 只读该 `profile.stats` 的调用者关系得到：`step` 共 601 次，其中内部 `_execute_plan` 调用 **563 次**，外层 `evaluate_tail` 强制首任务 33 次，根宏动作 5 次；`capture_v1` 共 607 次，其中 `step→snapshot` 601 次，`prepare_decision→snapshot` 5 次，根捕获 1 次。

若保持这条已发生的语义轨迹，第一版只省内部循环，完整 capture 次数应从 **607 降为 44**，而非降为零。这个数字是可核验的调用次数目标，**不是节省时间比例**。profile 的 `step→snapshot` 累计 42.288567 秒；pstats 的聚合调用者关系不能再按内部/外部调用数量准确拆分这段时间。总 capture 累计 42.420368 秒和总 deepcopy 累计 43.404972 秒互相包含，不能相加，亦不能许诺无仪器运行省 42 秒。

源码路径：

1. `_execute_plan` 循环调用 `self.step()`，丢弃 `MacroResult`。
2. `step` 先执行原 `_next_task/_scan/_resolve`，更新真实到该分支的状态和日志。
3. 尾部 `deepcopy(action_history[before:])` 构造 observations，再 `self.snapshot()`。
4. `capture_v1` 将完整 `V1_FIELDS` 一起深复制，包含增长的历史、几何状态、日志及覆盖 oracle cache。
5. 下一轮只读取续跑对象本身；没有读取上次 `MacroResult`。

因此移除的工作不是必要的分支隔离。构造新 sibling 的 `__init__` 深复制、树节点需要的完整 capture、公开 `snapshot.history` 的隔离仍要保留。当前源码没有定义复制/序列化/析构自定义钩子；正常对象图上的复制未发现会更新活动对象或随机数的副作用。这个结论限定在冻结对象图，不外推到未来任意子类。

## 2. 第一版实现范围

建议只新增一个明确实现开关，例如 `V1Continuation(..., elide_finish_results=False)`。默认保持旧路径；候选新 spec 只改变对应开关。开关是执行器实现配置，不混入后验、候选、K、门槛或 v1 几何参数。构造 sibling 时必须显式传递当前试验开关，不能依赖父分支被复制的临时属性。

将现有 `step` 的动作执行体移到一个共享内部方法，例如 `_advance_macro(task=None, *, materialize_result=True)`：

- 公开 `step(task)` 原样调用 `materialize_result=True`，仍返回完整 `MacroResult`；API、默认行为和暂停语义不变。
- `finish` 的 `_execute_plan`：开关关闭仍调用 `self.step()`；开启才调用共享方法且不物化结果。
- `_advance_macro` 的任务校验、时钟检查、pending 清空、选源/cover、移除 cover、blocked 更新、coverage 标记、三类异常处理全部保持原相对顺序，不复制第二份调度循环。
- 开启路径在原返回值物化位置执行同等廉价边界校验，随后返回 `None`。省掉的是 observations 深复制、完整 snapshot 深复制及 `MacroResult` 构造，不是逻辑状态更新。
- 保留 `_execute_plan` 的下一轮 stop 检查与 `super().run()` 的完成分类、显式 `/exit`、finally 报告回填。不要把停止直接当成功，也不要新增 enter。

最小伪代码仅表达分工，不是当前已实现代码：

```python
def step(self, task=None):
    return self._advance_macro(task, materialize_result=True)

def _execute_plan(self):
    while not self._continuation_done:
        if self._continuation_stop:
            raise _StopSearch(self._continuation_stop)
        if self._elide_finish_results:
            self._advance_macro(materialize_result=False)
        else:
            self.step()
```

第一版**不改** `evaluate_tail` 中强制首任务 `step(task)`，也不改根 `step(task)` 后紧接 `prepare_decision()` 的重复 capture。两处确实还有丢弃结果，但仅 38 次外部 step，先隔离占多数的内部循环。后续若要一并去掉，应作为额外实施范围明确登记，不能偷偷扩大同一首版。

## 3. 容易遗漏的正确性检查

`capture_v1` 不只是 deepcopy，还验证：Q3 类型、session active、无 uncertain pending request、策略 actions 与公开 accepted_actions 一致；它还读取全部 `V1_FIELDS`、规范化 remaining、校验 pending task。因此不能简单在 `step` 尾部提前 return，顺手把原来会拒绝的无效边界放过去。

建议抽取供两条路径共用的**无复制 capture 输入准备/校验**，复用原检查、字段读取及 `Position.coerce`；公开 capture 再深复制。快路径也执行这些检查和 pending task 校验，不能添加 `_next_task`、oracle 调用、计费或时钟查询。正常冻结对象下等价；不声称还能复现本应来自内存耗尽、自定义 `__deepcopy__` 等异常的行为。

具体风险及约束：

| 风险 | 保留要求 |
| --- | --- |
| 原选择会重定位 remaining cover 并消耗扩展/oracle 预算 | 共享原执行体；不能额外 prepare，不能改 cover 列表别名 |
| `_scan` 中途 budget stop | 未完成站点不 remove、不增加完成覆盖数；stop 仍传给 run |
| 第 16 次成功 clear 之后抛 `_StopSearch` | 保留实际 clear 日志与集合更新；父 run/清除证书分类不改；只退出一次 |
| protocol error 或未确定动作 | 与旧 capture 一样检查 pending/state/counter；不以快路径掩盖原边界异常 |
| inferred silence、shared cross bearing | 不增/删物理记录，不把推断静默变成测量；共用 `_resolve/_scan` |
| 父 snapshot / sibling 污染 | constructor deepcopy 与需要返回的 capture 仍完整；report log aliases、oracle cache 不跨分支共享 |
| 外部子类覆盖 `step` | 开启内部快路径可能绕过覆盖；首版范围限原 `V1Continuation`，必要时对自定义 step override 拒绝开启或退回原路径，不能声称支持任意派生类 |
| 计时日志字节不一致 | 保留字段含义，真实性能字段应记录新的真实耗时，不能填回旧耗时营造全日志一致 |

完整副本删除不改变算法状态的归纳依据：在相同宏起点、同一固定假想世界与反馈锚点、同一时钟分支结果下，共享执行体产生相同动作和后继活动状态；被丢弃的纯副本不进入下一次决策。逐宏归纳得到相同终止、计费和非计时日志。需要测试对象图确实纯净、校验/异常路径没有变化；这不是对一切 Python 对象的形式化证明。

## 4. “完全相同”应如何定义

不能直接比较完整 `report.as_dict()` 的字节：`program_runtime_s`、route planning 的 `runtime_s`、relocation 的 `runtime_s` 本来测量真实时间；外层还有 `planning_runtime_s`。这些日志的**语义和格式不变，数值不要求不变**。比较程序应只排除提前列明的计时字段和新增实现配置标识，不能泛化删除所有不同键。

其余要求精确一致，包括：完整物理 action_history（每个 action/channel/position/phase/result/bearing/time）、微秒 virtual_time、time_breakdown、接受动作/测量/清除次数、清除集合、完成原因及证书、remaining/visited/coverage 标志、regions/负历史、near/first bearings/observed positions、blocked、inferred、所有 route 选择/界/扩展数、oracle 调用数与 cache 数值、probe/relocation 选择日志。公开 step 的 MacroResult 也按同样规则验证，包括 observation 副本隔离。

有限时钟下需要分开两种结论：

1. **执行器等价**：假想世界、候选和样本固定，时钟不绑定，或在相同语义检查点人为注入同一截止结果，则删副本不改变工作内容、动作和成本。
2. **同预算程序效果**：真实时钟照常计时。更快可能完成旧版会中断的尾，允许更多完整候选/搜索，改变 fallback、估值和实际根选择，继而改变整局动作。此时算法采样/候选规则虽相同，实际执行的规划工作量已经不同，需要当作预算受限程序另行评价。更多完成规划不保证选得更好，模型偏差和小样本根择优仍可能造成退步。

即使单条尾无显式 continuation deadline，公开 client 还有 real_deadline/remaining_real_time，外层有 total_seconds、decision_seconds、实际 session 预留。因此“关掉一个 deadline”不自动构成时钟无关实验。

## 5. 必要测试与解耦验证

先做实现验证，不能用新性能随机案例代替。

1. **关闭开关回归**：现有 continuation 21 项及 coupled/旧模式相关 76 项保持通过；公开 step 返回类型、snapshot 行为、prepare 的幂等性不变。
2. **开关两态逐动作配对**：复用已有人工 near/inferred 场景和已打开 114005 中预选的 pre/post selection 边界（原测试 0、2、5、末尾）；同一世界分别续跑，验证上述完整状态及非计时日志。开/关父 snapshot 序列化内容均不得被污染，两个 sibling 互不污染。
3. **暂停组合**：`step(source/cover/shared) → finish`、`step → snapshot → 新 clone.finish` 与原连续 finish 等价；开关仅影响 finish 内部，公开 step 仍得到真副本。forced task 覆盖 pending 后不能多做 route search。
4. **停止与协议边界**：cap16；action/max_actions/virtual 预算；已到期真实 deadline；在相同第 k 个 `_planning_budget/_check_budget` 语义检查点注入截止；coverage 中断；拒绝动作；uncertain request、session/counter mismatch。比较同样的返回/抛错、stop、exit 次数及已执行前缀。不要按总函数调用次数模拟时钟，因为删副本正会改变总调用数。
5. **确认省掉的工作**：在 continuation 已构造后，用只读计数 spy 标记 `snapshot/capture` 的调用位置。开启 finish 内部应为零次物化，必要边界验证仍每轮存在；公开 step、prepare、显式 snapshot 仍物化。不要只测总速度，然后猜测优化命中。
6. **异常/边界不吞错**：快路径保留 capture 检查的失败类型与退出行为；如不适合支持异常客户端，须显式限定范围并拒绝，不能悄悄继续。

随后才做两层独立测量：

- **固定工作量 CPU 测量**：预先固定一小组已打开合法边界和假想世界；两态都完成同一份尾清单，没有墙钟绑定，保持原动作/扩展预算。先验固定重复次数和 AB/BA 顺序，不并发，使用 `process_time` 与 `perf_counter` 无 cProfile 记录 CPU/墙钟/捕获次数；功能等价比较放在计时区外。不修改正式 spec 的时间上限来把它冒充实际部署收益。
- **真实预算评价**：另冻相同候选、K、seed规则、阈值、每决策/全局时间和 tail/action cap，只变物化开关；根统一安排未打开的独立配对。报告 CPU/墙钟、生成动作、开始/完成尾、完成比较、fallback 原因、实际选择差异，以及成功率/失败清除/整局虚拟均值与尾部。没有必要继续一个已确认策略质量差且无可靠用途的算法，仅因计算更快而追求更多案例。

不使用“快了所以强了”的结论。若固定工作量 CPU 明显减少而实际预算轨迹不变，结论是同策略计算提速；若预算下规划工作增加并改变路线，必须另看完整清除和时间证据；若加速后仍劣于 v1，应保留通用执行器优化的证据，但不将劣势决策器提升为默认方案。

## 6. 建议的进入门槛

首选范围是内部 563 次类工作，单独分支、默认关闭，先证明共享执行体和 snapshot 校验没有删错，再完成固定工作量验证。暂不改 coverage oracle、active probe、snapshot 结构共享/持久化、条件采样或候选集；那些都不是“省一个被丢弃返回值”的同一变化。

该设计的价值已有明确调用证据，但收益幅度尚未实测；在固定非绑定时钟下的语义等价可强验证，在真实时间预算下的全局不退步不能由此证明。
