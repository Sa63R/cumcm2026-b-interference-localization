# 旧开发记录的联合可见性几何诊断

本诊断只使用旧 `compact_combo` 已接受的真实测量和已发生的 resolver 入口，不执行策略、不生成新场景、不读取源真值。记录中的 `evaluation`、`row` 等非观察子树采用词法跳过，不解码；读取字段仅为 `summary/history/spec`。输入为旧开发随机 610001–610024、旧开发压力 610031–610044，不能把这两组已打开的数据重新称为独立验证集。

**结论：联合负信息确实能缩小一部分原正观测外包，但直接达到清除阈值的入口只有两个。** 不能将半径缩小的米数换算成整局节省秒数，本阶段没有候选任务 T，也没有可计算的候选 T/LB。

| 项目 | 随机24局 | 压力14局 | 合计 |
|---|---:|---:|---:|
| 真实 resolver 入口 | 308 | 179 | 487 |
| 已 ready，跳过 | 252 | 131 | 383 |
| 无负反馈，跳过 | 2 | 0 | 2 |
| 几何调用且独审通过 | 54 | 48 | 102 |
| 删除至少一个相交格 | 12 | 18 | 30 |
| 排除至少一个原顶点 | 10 | 18 | 28 |
| MEC半径减少超过1e−6米 | 10 | 17 | 27 |
| 从不 ready 变成 ready | 1 | 1 | 2 |
| 审计失败 / 回退 | 0 / 0 | 0 / 0 | 0 / 0 |

102 个调用覆盖 31 个场景、101 个不同的场景—频道；27 个缩圆入口分布在18个场景、27个不同的场景—频道。半径改善的中位数为零，含未改善调用的平均减少为60.606511米，最大减少401.965115米。重复服务入口没有被冒充独立场景。

两处变 ready：

- 随机610013、频道9、实际动作前缀283、原链调度入口：半径20.613943→17.053875米。
- 压力610033、频道15、实际动作前缀125、原 early-service 入口：半径20.259758→4.708295米。

最大缩圆例是随机610024、频道17、前缀310：721.009147→319.044032米，仍远大于可清除半径。因此仅增加“新区域已经 ready 时立即清除”的机制，无法兑现大部分缩圆潜力。若将新外包用于主动探点或光学覆盖，需另行实现真实反馈更新、连续覆盖证书和新的独立配对实验。

## CPU与证据边界

先用两个旧场景的四个有效入口做小试：几何平均0.032117秒、最大0.039154秒；独审平均0.169658秒，4/4通过。随后才处理全部38条旧记录。

全量102次几何总计3.300777秒，平均0.032361秒，最大0.090824秒；独立审计总计16.664099秒，平均0.163374秒，最大0.371380秒。这是本机顺序执行的诊断成本，包含日志构造的几何调用时间；不是候选策略整局程序耗时，也没有将离线审计成本算入虚拟移动时间。

每个 resolver 入口来自原 `chain_route_log` 的 source 宏任务或 `early_service_log`。CandidateRegion 只按该前缀之前、该频道尚未清除时的真实 direction 更新；near 计入真实正点并直接跳过已经 ready 的入口；no_signal 保留真实坐标，clear miss 不当无线电负反馈。每份诊断保存旧记录SHA、所有实际动作前缀SHA、同频道真实测量前缀、几何输入、完整256格证据、独立审计与统计。原记录和原 canonical C 均不修改。

必要复现与证据：

- 脚本：[diagnose_q4_joint_visibility.py](../../experiments/diagnose_q4_joint_visibility.py)
- 小试：[freeze](../../results/q4_joint_visibility/prefix-pilot/freeze.json)、[summary](../../results/q4_joint_visibility/prefix-pilot/summary.json)
- 全量：[freeze](../../results/q4_joint_visibility/prefix-full/freeze.json)、[summary及逐入口证据链接](../../results/q4_joint_visibility/prefix-full/summary.json)
- 102份完整前缀、几何证据与审计：`results/q4_joint_visibility/prefix-full/prefixes/`。

```powershell
& 'D:/jwt/2026数模国赛/cumcm2026-b-interference-localization/.venv-win/Scripts/python.exe' experiments/diagnose_q4_joint_visibility.py --input 'D:/jwt/2026数模国赛/q4-round2/results/q4_round2/combination-development' --input 'D:/jwt/2026数模国赛/q4-round2/results/q4_round2/combination-development-stress' --output results/q4_joint_visibility/prefix-reproduction
```

输出目录必须不存在。脚本只允许这两个已打开开发集及其准确种子和原combo配置；重建使用的geometry/localization源码须与旧manifest哈希一致，helper/独审/诊断脚本的实际哈希在运行前后核对。全量结果 `complete_requested_records=true`、`all_audits_passed=true`。

下一步宜在已经调度到某源的 resolver 内建立独立辅助区域，原 scan、range 与 canonical C 保持原职责。新辅助区域若用于探点或光学网格，必须独立记录其连续外包证书并按真实正反馈收紧；不能将本诊断的未来观察或旧场景结果作为在线先验。后续性能比较应使用届时已资格通过的最佳 Q4 基线，而不是借用这里旧combo的几何变化宣布升级。
