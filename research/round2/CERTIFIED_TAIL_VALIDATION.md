# 持证清除尾段候选：冻结验证与使用边界

2026-09-11。结论：本地完整随机案例上有稳定但极小的虚拟时间改善，可以保留为可恢复的独立候选；并未解决问题3的主要耗时差距，尚未做官方演练兼容性确认。

基准为保护分支`research/q3-state-search`的`8c624d08c61c0e411f93345d63a932b01ead4868`。候选分支`research/q3-r2-certified-tail`，冻结提交`eb0b5a822a61f9077bbc403f87549b148349b5bd`，入口`strategies.certified_tail_state_search:run_certified_tail_state_search`，完整配置`experiments/state_search_candidate_certified_tail_v2.json`。原基准及其他原有核心分支保持不变。

## 结果及证据

|批次|每组局数|基准均时秒|候选均时秒|平均节省95%配对区间秒|胜/平/负|逐局均T/LB：基准→候选|
|---|---:|---:|---:|---|---|---|
|开发206101..206132|32|3155.426927|3155.329083|[0.008957,0.208741]|4/28/0|1.778712→1.778663|
|独立确认206133..206196|64|3141.621273|3141.574517|[0.017354,0.083950]|11/53/0|1.793287→1.793260|
|独立最终210001..210256|256|3083.748585|3083.703140|[0.027055,0.066123]|26/230/0|1.799075→1.799049|
|独立压力211001..211028|28|3190.606598|3190.538455|[0,0.202378]|2/26/0|3.750836→3.750801|

总计380对、760条完整运行记录，双方均100%完整清除、明确退出、0失败清除。最终随机集候选平均省0.045445秒，约0.001474%；P95双方均3516.524421秒，最大值均3786.354168秒。压力集P95双方均4338.376083秒，最大值均4666.100145秒；仅2局改变，压力集单独不足以证明平均优势。

最终随机集程序墙钟均值0.993709→0.995463秒，CPU均值0.984314→0.985779秒。完整日志保留P95/max及每局值；这是并发独立进程下的计时，毫秒级差别不应解读成精确的硬件性能结论。所有策略源、配置、物理引擎和评估接口按字节冻结；真实隐藏状态仅由评估器在策略退出后读取。

每批都完成两类独立审核：`certified_tail_*_audit.{json,md}`核对物理反馈、微秒费用、几何安全和完备覆盖；`diagnoses/certified_tail_*_execution.{json,md}`直接对照配对的真实基准日志，核对接受前相同前缀、完整原尾段、候选实际执行点、费用和整局节省。760条记录均通过，无接受后取消/中断。

当前候选33项实现测试通过；尾段核对器8项正常/篡改fixture通过，能识别前缀变化、伪造基准后缀、取消却声称completed等错误。独立源码评审见`diagnoses/certified_tail_review.md`。

## 实际改动

只有当发现阶段在当时已合法完成（覆盖完毕或合法已知16源）、无blocked源，且剩余2..4源全部near或MEC半径≤19.9米时，才考虑整条清除尾段。其他情况沿原v1。

先按原程序真实规则重现完整基准：每清一个源都重新按频道排序、调用原状态路径求解器、累计相同展开预算，并用原安全圆投影得到实际清除点。保留这条可执行路线为对照。新方法最多枚举24个顺序，在各源安全圆内做有界坐标优化，以逐段微秒舍入后的整段真实费用比较，至少省10毫秒且逐点有安全证书才接受。

接受后必须执行整条已比较的路线，并核对每步公开状态；不能只执行便宜的第一步后重新走另一条昂贵路线。费用或状态不符、失败、中断会撤销整段不退步主张。低实际时间余额及不足以完成尾段的动作/虚拟预算直接保留原行为。

最终随机及压力批的28次改变均未改变源顺序，实际收益来自清除点微调。因而这些结果没有证明排列枚举本身贡献了收益，也没有证据支持继续增加排列深度来取得大幅提速。正式的固定顺序算法若要单独替代当前候选，仍须重新冻结评估，不能把这个事后观察当作另一个未经测试版本的性能。

## 下界解释

沿用旧口径：事后已知源中心，每个源用20米清除圆盘代替，起点/圆盘之间采用最短距离边权，子集DP精确求最短开放图路径L，LB=L/5+5N。它省略发现、定位、换频和无遗漏证明，也允许相邻边在同一圆盘选不一致端点，因此是先知松弛下界，不能解释成可达在线最优。

最终随机集平均LB为1732.957082秒；逐局均T/LB从1.799075降至1.799049，而总T/总LB为1.779472→1.779446。两种汇总不同，均有完整每局数据。压力集包含聚集源和极端误差，平均逐局比值大于总量比；不应把其3.75混作随机分布下的平均表现。

## 复现

在归档工作树`q3-round2`运行，使用项目已有Python环境；需要保护基准工作树`../q3-state-search`与候选工作树`../q3-r2-certified-tail`分别处于上述冻结版本。若原树有用户工作，另建工作树，不做reset。`prepare`输出必须用新目录，旧原始证据不可覆盖。

```powershell
& ..\cumcm2026-b-interference-localization\.venv-win\Scripts\python.exe experiments\round2_runner.py prepare --trial certified_tail --stage pilot --spec experiments/state_search_candidate_certified_tail_v2.json --output results/round2/certified_tail/reproduction-pilot
& ..\cumcm2026-b-interference-localization\.venv-win\Scripts\python.exe experiments\round2_runner.py run --output results/round2/certified_tail/reproduction-pilot --workers 3
```

把stage换成confirmation/final/stress并使用各自新输出目录，可在原有已验证阶段的门槛记录下重跑对应已公开案例；重跑属于复现，不能再次称为新独立验证。运行器按源码/spec/helper哈希锁定策略。已有批次同时存有当时的runner.py、manifest、完整源码zip和每条gzip记录，可直接离线复查，不必重新仿真。物理审计依赖保护工作树`q3-geometric`与`q3-state-search/research/theory_v1`中的审核器，依赖哈希已记录。

```powershell
& ..\cumcm2026-b-interference-localization\.venv-win\Scripts\python.exe -m experiments.round2_certified_tail_diagnosis --batches results/round2/certified_tail/final --physical-audits research/round2/certified_tail_final_audit.json --output research/round2/reproduction-tail-execution.json --report research/round2/reproduction-tail-execution.md
```

若源码所在目录不同，请在新复现记录中明确路径变化，保留旧manifest不动。推送的分支和本地目录包含完整实现；研究使用的临时路径不是策略的环境真值来源。

## 官方验证与下一阶段

本轮没有触碰正式测试，也没有打断其他任务的官方演练采集。SQLite全库只作严格前缀验证，未用来拟合分布；它不能为新清除点提供官方反事实反馈。该候选的逐步精确微秒签名可能受官方序列化/舍入差异影响，尚不能宣称官方环境整局不退步。因此保留原v1作为已有官方证据支持的版本，候选保留独立分支，待演练验证后再决定实际使用。

当前研究未结束。五个早先无可靠收益方向已完成归档和清理；本候选获得了可验证的小收益。用户新增文献对应的多目标观测条件树尚未经过新实现实验，应先解决完整宏动作观测似然、条件重采样及v1续跑等价，再做独立三组对照。见`LITERATURE_ASSESSMENT.md`和`diagnoses/NEXT_OBSERVATION_TREE_DESIGN.md`。这不是预算耗尽或理论不可改善的结论。
