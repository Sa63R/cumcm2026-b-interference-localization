# R15 原地未知频道补扫：开发结果（未晋级）

固定选择器结论：`passed=false`、`selected=null`、`report_complete=true`。两候选相对 R12 在随机和困难开发集均更慢；停止本轮，不开启预留独立集。R12 可靠版本不变。本结论是两个已固定补扫规则的阴性结果，不证明所有利用空闲位置发现新源的方法都无效。

## 完整对照

随机24局：624001–624024；困难14局：624031–624044。每个案例五状态臂和同配置固定 RL，共228条真实本地运行记录；均全清，通用及各专项审计全部通过。没有补抽或隐藏退化。表中倍率是**均T/均LB**，不是逐局倍率平均；LB沿用完整同场景事后理论下界，包含原题信息限制尚未付出的理想化条件，不是可实现最短时间。

|方法|随机均T/s|随机均T/均LB|困难均T/s|困难均T/均LB|
|---|---:|---:|---:|---:|
|原22站基线|7292.459211|3.524954|7277.426262|4.354675|
|R9 probe|6614.958789|3.197471|6857.713704|4.103527|
|R12 当前参照|6566.926477|3.174253|6569.173920|3.930870|
|R15 cap|6581.459129|3.181278|6580.011282|3.937355|
|R15 bounded|6652.614163|3.215672|6659.082710|3.984670|
|固定宏 PPO512|7225.580447|3.492627|7075.256074|4.233700|

共同均 LB：随机 2068.809914 秒，困难 1671.175580 秒。RL 使用协议冻结的源包与 checkpoint，未训练或重选。

|候选/数据|相对R12平均节省/s（负为退化）|配对95%区间/s|胜/负/平|P95时间比|
|---|---:|---|---:|---:|
|cap/随机|-14.532652|[-22.916667, -4.981971]|1/12/11|1.004701|
|cap/困难|-10.837362|[-19.460438, -3.123076]|0/5/9|1.000000|
|bounded/随机|-85.687686|[-101.656699, -61.593939]|1/23/0|1.013013|
|bounded/困难|-89.908790|[-101.500000, -73.357143]|0/13/1|1.012519|

全部退化案例及逐局T/LB保存在 [development-decision.json](development-decision.json) 的 `comparisons_vs_incumbent.*.*.loss_rows`；更完整的逐局费用和动作统计在 [development-mechanism.json](development-mechanism.json)。冻结规则、Bootstrap种子和门槛没有更改。

## 为什么没有改善

两规则真实付费补扫，未获免费覆盖信用。cap 仅在已知14或15源后试一次；bounded允许更早最多两批。所有新增批次均没有通过本次补扫达到已知16源上限，因此本轮未兑现“立刻结束剩余发现义务”的关键收益。少数新阳性确有价值，但总体抵不过测量和换频。

|候选/数据|批数|实际补测数|新阳性数|每局直接补扫/s|每局移动变化/s|每局测量+换频变化/s|
|---|---:|---:|---:|---:|---:|---:|
|cap/随机|13|75|1|18.458333|-3.092348|17.625000|
|bounded/随机|38|387|7|95.458333|-7.103981|92.791667|
|cap/困难|5|30|1|12.714286|1.480219|9.357143|
|bounded/困难|22|217|2|92.000000|1.480219|88.428571|

上述变化均为候选减R12；光学与移除费用在四组中全部相同。直接补扫费用与整局费用差不同，因为后续换频、少扫的频道以及路径都可能变化。决策日志墙钟合计不到0.1秒（76个候选运行），性能瓶颈是虚拟行动开支，不是补扫决策CPU。并发运行的整局墙钟不用于跨方法速度排名。

有用反例和失败尾部：

- 唯一获益随机案例624002：cap补6次发现1个新频道，移动省74.216343秒，额外测量/换频16秒，净省58.216343秒，T/LB由3.791662降为3.759565；bounded净省138.265492秒，T/LB为3.715430。说明机制偶有实际收益，不能只用平均值抹去。
- cap最坏退化37秒（624001、624044）：补6次无新阳性，直接批费36秒，后续额外换频使整局多37秒。因此36秒直接批预算不是整局退化保证。
- bounded最坏随机案例624003：补测13次发现1个新频道，但实际移动反增75.580441秒，加60秒测量和11秒换频，总退步146.580441秒；T/LB由2.821338升至2.882694。120秒直接预算同样不能限制策略分叉后的总退化。
- bounded困难集最坏624035：20次补扫全阴性，移动不变，测量+换频净增117秒。

未根据这些场景调阈值、重选补扫子集或重跑。未来若重开此方向，需要独立的行动价值依据；不能把这一轮发现率反向拟合为当前成功规则。

## 身份、审计与复现

- 冻结生产源码：`eef86b6c7608d42f1c6e6ce8509e48d3464b553e`；开发freeze SHA256：`4c14eabd942b82dc7baea5f108d2e9968e1a0ca0b638db3aadc1d69d65de2947`。
- 随机阶段执行HEAD `ca6c9af45ae2e6d036c998496616d0012e708e97`。stress放行文件原先把 `reserved_seeds` 写成列表，但已冻结通用runner要求 `{"stress":[...]}`；执行stress前只修该元数据并提交 `bb4f71eafacddb4f0793d9430e58814c75a14531`，修正SHA `27b8b8a63a24b3e047c5728ad1e812ee68565aa8dc406ea6255bf203a41e4968`。原文件保存在Git历史中，未用其生成stress场景；算法、审计、选择器、协议和参数全未改。
- 两组各五份审计：`independent_audit.json` 覆盖120/70条；`joint_visibility_prefix_audit.json`、`joint_continuation_prefix_audit.json` 和两份 `opportunistic_discovery_{cap,bounded}_audit.json` 各24/14条，全部passed，候选专项串联R8/R12前缀。固定RL另有24/14完整审计和全部case SHA配对。
- 所有原始gz、源包、manifest/freeze、summary、各audit和console在 `results/q4_opportunistic_discovery/`，两组RL在对应 `*-rl`。没有真实模拟器/数据库/SSH调用。

从相同冻结工作树复现到**新输出目录**，勿覆盖本次记录：

```powershell
python -m experiments.run_q4_round2 --specs research/q4_opportunistic_discovery/development-specs.json --stage pilot --start 624001 --count 24 --workers 3 --output <new-random-directory>
python -m experiments.run_q4_round2 --specs research/q4_opportunistic_discovery/development-specs.json --stage stress --start 624031 --count 14 --workers 3 --selection research/q4_opportunistic_discovery/development-stress-release.json --output <new-stress-directory>
```

每目录先执行 `run_q4_round2 --output DIR --audit`，再执行 `audit_q4_joint_visibility_prefix_batch`、`audit_q4_joint_continuation_batch` 的 `--input DIR`，以及 `audit_q4_opportunistic_discovery_batch --input DIR --label compact_opportunistic_cap` / `compact_opportunistic_bounded`。RL需同固定源包和对应完整case。正式已执行的决定命令是 `python -m experiments.evaluate_q4_opportunistic_discovery --phase development`；它会拒绝覆盖已有决定，不应为重现而覆盖本次证据。

本轮未创建独立selection；预留624101–624228、624301–624384均未开启。后续归档应保留本次完整源码与证据。
