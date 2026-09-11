# R9：仅在已调度定位过程使用联合可见域

本轮是两个固定配置的独立实验，继承已合格的 R8 `center_once` 控制器，不表示已经获得整局时间改进。旧前缀诊断证明了一些几何区域可以缩小；它不能给出改变行动后的时间反事实。几何论证见 [MODEL.md](MODEL.md)、[THEORY.md](THEORY.md)，旧开发诊断见 [DIAGNOSTIC.md](DIAGNOSTIC.md)。

## 固定接口与对照

入口为 `strategies.q4_joint_visibility:run_q4_joint_visibility`，固定 `problem=4`，默认 `max_actions=20000`、`max_active_probes=6`、`max_expansions=200`。配置仅有：

| config | 局部探测 | 光学兜底 |
|---|---|---|
| `probe` | 在原探测决策入口用辅助区域产生原五点候选 | 完整原 canonical 区域网格 |
| `probe_optical` | 与 `probe` 相同 | 先遍历辅助区域完整网格；异常耗尽后保留原 canonical 网格 |

两者都以 R8 为对照。覆盖站点、覆盖扫描、远距推断删测、已持证频道删测、全局 `_ready/_target`、链调度、沿途 60 秒服务片及 16 个真实清除的停止条件均继承原方法。没有改变任何既有基类文件。

## 信息来源及两个区域

每次已经被调度的 `_resolve(channel)` 入口，检查源仍存活、已经检测、没有 near 回包、canonical 最小包围圆半径大于 19.9 m 且已有真实 `no_signal`。满足这些条件才调用一次 `joint_visibility_outer`；其输入只有该频道清除前接受的真实测量坐标与 canonical 正方位区域顶点。负测量可以发生在首次检测之前。光学清除失败不当作无线电负测量，清除后的静默不进入输入。

辅助区域 `aux` 从 canonical 的独立副本生成，只替换安全外包顶点并清除圆缓存。原对象始终由原测量更新逻辑管理。辅助区域依靠“同一个未知半径、同一个未知朝向”的联合约束保留所有兼容真实位置；它既不把未知类型当作已知，也不把无信号单独解释为全向距离下界。

仅当 helper 排除了至少一个旧多边形边界顶点时才启用 aux。若 `status=unchanged`，或 `old_vertices_excluded` 为空，记录 `no_boundary_reduction` 并继续原控制器。这直接使用 helper 的精确 Fraction 包含判定，不引入面积阈值或场景调参。只删内部格但凸包填回原边界，并未给本策略新增位置约束；不能让外舍入造成的微小圆心变化无故关闭 R8 的原中心精确匹配或改变实际探点。

入口之后，每个接受的真实 `direction` 同时按同一方位误差模型裁剪 canonical 和 aux。`no_signal` 仅记入真实观测账本，本次定位不重新执行几何删格；下一次被调度才可使用新增负测量。near 继续由原父循环优先处理。预算未执行、拒绝请求、光学 miss 均不能制造方位反馈。

## 原有决策结构上的局部变化

父循环先尝试 canonical 已持证中心和真实 near 点。只有随后调用 `_next_probe` 时，才使用 aux：

1. 若其包围圆半径不超过 19.9 m，在该圆心发出真实 `joint_visibility_clear`。只有收到 `success` 才结束定位；专用局部异常穿过原 R8 和 Q4State 的 `finally` 后被本层捕获，不伪造一次测量。
2. 否则保持原五点结构：辅助圆心，以及首方位的正负平行、正负垂直偏移，偏移尺度为 `min(180, max(25, radius/2))` m。
3. 仍按六位小数的真实已测位置去重，优先尚未测过的中心，再取距当前位置最近的新候选。六次主动测量预算不扩大。

若最后一次允许的测量后 aux 才持证，而 canonical 仍未持证，父循环准备第一个光学网格点时，两配置都先用这个辅助圆心执行一次真实清除。这个终端检查不增加测量、不重置预算，单独记录 `terminal_clear` 事件；失败或预算不足仍分别保留矛盾或中断状态。只有没有可用辅助证书时，才进入对应的完整网格。

R8 的清除先于测量规则仍严格检查 **canonical** 半径 `19.9 < r <= 40` 及原 canonical 中心。辅助区域改变探点后，原 R8 尝试可能不再触发；没有扩大 40 m 阈值，也不以 aux 中心冒充原中心。候选与 R8 的收益不能假定简单相加。

## 光学完整性与执行边界

`probe_optical` 在父循环第一次准备 `guaranteed_clearance` 时，先调用独立 `_run_joint_optical_once`。它沿原首方位旋转 aux，生成整个旋转外接矩形的 28 m 完整网格，再用原确定性次序遍历。覆盖依据是每个 28 m 方格半对角 `14√2 < 20` m，必须覆盖整个多边形，不能只验证有限采样点。

日志存储全部计划网格与实际执行前缀。真实 success 允许提前停止；服务片、动作数、虚拟时间或真实时间不足时直接沿继承路径抛出中断，不伪装成已遍历。若完整辅助网格全部真实 miss，记录 `exhausted_without_success` 模型矛盾，再允许原 canonical 完整网格兜底。这个矛盾不能因为最终清除成功而从资格审计中删除。辅助区域被真实方位裁空、辅助持证清除失败同样保留矛盾记录。

这里只保证有限几何证书与真实动作约束。探点启发式和光学次序不是全局最优，辅助区域较小也不保证更好的可接收方位、较少移动或较短实际完成时间。

## 可重放日志

`report.strategy_parameters` 新增下列数组。所有动作索引是 `action_history` 中已接受 `measure/clear` 数量的半开区间，不含 enter/exit；全局预算仍使用父控制器含 enter/exit 的计数。

| 字段 | 主要证据 |
|---|---|
| `joint_visibility_resolver_log` | 入口与结束索引、频道、跳过理由、完整 helper 证据、初始 aux 顶点；后续每个真实 direction 的点/方位/前缀及 aux 顶点 |
| `joint_visibility_probe_log` | resolver ID、探测轮次、决策前后索引、canonical 原 proposal、aux 圆/顶点、完整五点、fresh 索引、实际选点及结果 |
| `joint_visibility_terminal_clear_log` | resolver ID、原 canonical 网格首点、aux 圆/顶点、真实动作起止索引及清除或中断状态 |
| `joint_visibility_grid_log` | resolver ID、开始点/首方位/aux 顶点、完整 28 m 网格、实际动作数量及结束状态 |
| `joint_visibility_model_contradictions` | 全网格 miss、持证 clear 失败或新真实方位裁空，保留真实前缀 |

探测事件区分真正测量、R8 先清除后测量、R8 成功从而没有测量、以及被预算或拒绝请求中断。`decision_wall_s` 只统计本地决策，不将请求往返时间冒充几何计算时间。独立 prefix 审计绑定真实 wire 前缀与 helper 证据，重建 aux 更新、候选、持证清除和完整网格；它不需要读取源真值。

## 本轮验证范围

新增策略控制测试 28 项，与冻结 R8 的 24 项测试合计 **52 passed**。命令：

```text
python -m pytest tests/test_q4_joint_visibility_strategy.py tests/test_q4_clear_before_probe.py -q
```

覆盖副本隔离、真实正负反馈、canonical 清除优先、R8 成功异常、探测预算耗尽后的辅助持证清除、光学完整计划与真实前缀、失败后的原网格、预算和请求拒绝、继承方法以及 16 个实际清除停止。无边界缩小的两个回归对照逐项核对实际 R8 清除、原探点测量和随后清除轨迹与原控制器完全一致。终端清除正例包含真实 helper 证书的脚本观测构造。部分测试注入明确标为 test-only 的辅助多边形以隔离控制流程，不能作为几何或实局性能证据；实际 helper 及独立几何审计另有构造测试。此提交不运行新完整场景、官方演练或正式测试。
