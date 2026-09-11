# R26：固定区域块内收集互补观测

本轮从合格R12 `81aa6e1a` 独立继承。只改变原22个认证覆盖站的访问顺序，不移动、增加或删除任何站点。希望同一区域的内外圈观测更早齐备，减少后续定位、清除的折返。这个机制与R5根据已ready源重排剩余站点、R16旋转反射整条原路线不同；并未假设相邻站必然收到同一源。

## 固定点与块模型

`planning.sector_cover_route.sector_cover_route(m)` 从原 `certified_cover_points('compact_22')` 读取原点、7个970米内圈点和14个外圈点。坐标对象直接按原索引重排，没有重新计算三角函数。锚方向是原路线第一个内圈点的方向，顺时针将每个内圈点与对应两个外圈点组成三点扇区，连续m个扇区构成一块，尾块保留剩余扇区。先访问原点，各块须依次完整访问，块内顺序自由；不要求返回原点。

生产入口 `strategies.q4_sector_service:run_q4_sector_service` 只接受 `sector_1`、`sector_3`，默认前者，`max_expansions=200`。m=2只允许调用纯几何函数，不是运行候选。运行参数保留原R12的动作、探测及200/60000规划预算。

给第b块内访问集合M和最后点j建立状态。块内转移为

`D_b[M∪{k},k] = min_j (D_b[M,j] + distance(j,k))`。

初始化同时考虑上一块**每一个**可能出口e：`D_b[{j},j] = min_e (F_(b−1)[e] + distance(e,j))`；块完整后每个j均保留到下一块。第一个块从原点、零代价开始，最后取所有出口最小值。未来代价只依赖已完成块、当前块访问集合与末点，因此同状态较贵前缀可舍弃。逐块只选一个最近出口不满足这个递推，本轮有专门反例测试。

这是固定分块约束、固定欧氏边长、加性纯移动模型的动态规划；不是对未知源、定位观测、清除服务与连续路径的完整穷举。实数递推有标准最优子结构；实现使用float、普通加法和确定性索引打破相等计算值，没有区间算术证明。独立小图测试枚举全部块内排列，用独立路径求和核对，容许求和顺序造成的微小舍入差异。对于块大小n_b≤9，块内时间 `O(Σ n_b²·2^n_b)`，状态 `O(n_b·2^n_b)`；本实现为确定性平局保留完整前缀，另有至多22长度的路径存储/复制。跨块初始化保留前后出口组合，不用额外A*预算。

|纯覆盖路线|点数|开放路程（米）|相对原路线的纯移动时间差（秒，距离/5）|
|---|---:|---:|---:|
|原compact_22|22|17612.419400|0|
|sector_1|22|18451.731|约167.862|
|m=2，仅几何|22|18394.34|约156.38|
|sector_3|22|18171.960|约111.908|

这些是无定位、无清除绕行的纯覆盖代价，不是整局增量上界。真实动作对移动时间取微秒舍入，表内距离/5仅为纯几何比较。路线变长可能完全抵消观测先后的收益，也可能改变发现第16个源的时机。

## 执行与证据边界

新类仅实现构造函数，`run/_execute_plan/_scan/_resolve/_next_probe/_perform/_clear/_early_service/_check_budget` 均直接继承R12。构造时核验22点精确一一对应、起点原点、原覆盖证书station hash，替换 `self.points`、`report.coverage_points`。没有合成观测、提前增加visited或清除状态。

原方向覆盖几何证书对同一站点集合仍有效，全部站点是否已经实际扫描仍由原逐动作账本判断。已知16与实际清除16的区别、持证已知频道跳测、远距推断跳测、ready源插入、60秒early服务、R8原中心试清、R9/R12辅助区域更新和完整光学兜底都保留。任何源真值仅由原runner在退出后评分，不传入策略。

`strategy_parameters.sector_service_config` 保存配置；`sector_cover_route` 保存扇区、块、原索引排列、实际新纯路程、原路线长度、DP状态数、原station hash及原覆盖模块SHA。父 `directional_cover_certificate` 原样保留：其中 `route_length_m` 仍描述**原compact_22顺序**；另有明确route_scope提示，不能拿它冒充实际新顺序长度。新路线元数据每次深复制，调用者修改报告不会污染缓存。`sector_route_setup_runtime_s` 是本次构造墙钟时间，可能命中缓存，不假称每局重算DP时间。

## 构造测试与已打开案例烟测

以下命令118项通过（1.44秒）：

```powershell
$env:PYTHONPATH='src'
& '../cumcm2026-b-interference-localization/.venv-win/Scripts/python.exe' -B -m pytest -q tests/test_sector_cover_route.py tests/test_q4_sector_service.py tests/test_q4_joint_continuation.py tests/test_q4_clear_before_probe.py tests/test_q4_r2_scheduling.py
```

其中新54项涵盖独立全排列、小块出口贪心反例、重复坐标保留任务身份、空任务/非法模型、原22点与分块次序、缓存隔离，以及真实脚本反馈的440次unknown扫描顺序、原R8成功清除/父finally、实际16清除停止和入口预算。最初测试的直接浮点等号及未捕获父StopSearch属于测试预期问题，已分别用数值容差与原终止契约修正；生产机制未因此改变。

仅对**已打开的621001**各运行一次集成QA，保留完整 `old-smoke/records/*.json.gz`、source.zip、输入源码SHA、summary与独立审计。两个配置caseSHA相同，13个源都清完、失败clear均0，root的 `audit_full`（通用物理/整覆盖/LB、独立冻结22索引、R12/R8/range/scheduling）均通过。

|旧621001烟测|总时间T（秒）|T/N（秒/源）|T/LB|
|---|---:|---:|---:|
|sector_1|7014.651181|539.588552|3.165823|
|sector_3|6940.472122|533.882471|3.132345|

这两条仅证明接口闭环，不作选型或泛化收益证据。尚未运行632开发或独立数据；后续正式研究采用冻结PROTOCOL的两个固定候选、完整源数分层及所有失败保留规则。本文件不是论文正文。
