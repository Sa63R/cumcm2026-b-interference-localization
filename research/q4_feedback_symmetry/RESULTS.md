# R16 首轮反馈选择覆盖方向：开发结果

本文仅报告625开发证据。root已完整执行冻结 evaluator，`passed=true`、`selected=compact_feedback_centers`、`report_complete=true`；进入独立验证的唯一候选为 early_centers。本报告不宣告独立资格通过，不改变当前R12可靠版本。形成此开发报告时，预留625101–625228及625301–625384尚未开启。

## 完整开发对照

随机24局625001–625024，困难14局625031–625044。五状态臂共190条，加同环境固定RL38条，总228条均全清。每组generic及R9/R12/mean/centers五份审计全部通过；RL全部通过。保持冻结源码、配置、协议、统计门槛，未重抽或丢弃退化。下表倍率为**均T/均LB**，不是逐局倍率均值；LB沿用同一完整场景的事后理论下界，不代表在线可实现的最短时间。

|方法|随机均T/s|随机均T/均LB|困难均T/s|困难均T/均LB|
|---|---:|---:|---:|---:|
|原22站基线|6874.286208|3.326031|7440.058259|4.330999|
|R9 probe|6432.280754|3.112173|6839.335786|3.981307|
|R12 当前参照|6430.643667|3.111381|6609.508038|3.847520|
|R16 bearing_mean|6417.654161|3.105096|6532.524873|3.802706|
|R16 early_centers|6370.811304|3.082431|6495.844224|3.781354|
|固定宏 PPO512|7328.572412|3.545831|7698.024884|4.481166|

共同均LB：随机 2066.813606 秒，困难 1717.861970 秒。固定RL源包、权重与worker/runner沿协议，不训练或重选。

|候选/数据|相对R12平均节省/s|配对95%区间/s|平均改善|胜/负/平|P95时间比|
|---|---:|---|---:|---:|---:|
|mean/随机|12.989507|[-46.284757, 78.703853]|0.2020%|4/4/16|0.994744|
|mean/困难|76.983164|[-0.944068, 167.223946]|1.1647%|5/1/8|1.000000|
|centers/随机|59.832363|[-85.585343, 219.673417]|0.9304%|11/7/6|1.002269|
|centers/困难|113.663814|[1.811880, 232.975092]|1.7197%|5/3/6|1.000000|

两个随机区间都跨零，不能声称稳定提升或把开发点估计当独立结论。centers随机均时比mean低46.842857秒，超过协议5秒简化优先阈值；完整选择器已确认均值/P95/审计开发门槛并按原规则选择centers，独立运行仍须root另行放行。

## 实际变化机制

R16只利用原点已经支付费用的20次真实测量选择一次D7等长链。逐条对照确认：全部76个候选运行的首20动作与R12相同，所有选择identity的病例之后完整 `action_history` 也与R12逐项相同。没有新增探测、免费覆盖或坐标旋转；非identity病例首次实际分叉出现在第21个动作。

|候选/数据|改链次数/局数|每局移动变化/s|每局检测变化/s|每局换频变化/s|
|---|---:|---:|---:|---:|
|mean/随机|8/24|-8.614507|-3.541667|-0.833333|
|centers/随机|18/24|-36.915697|-19.166667|-3.750000|
|mean/困难|6/14|-57.483164|-16.428571|-3.071429|
|centers/困难|8/14|-64.663814|-40.714286|-8.285714|

变化为候选减R12，负值表示减少。四组光学与移除费用均未变。mean的集中度门控保留了较多原链；centers更频繁改变方向，收益和退化尾部也更大。两者76局选路决策墙钟合计约0.0264秒；这是日志中的局部决策时间，不把并发整局墙钟作跨算法公平速度结论。

完整22站纯覆盖路线仍等长。实际收益来自观测先后顺序影响可定位/可提前服务的源、后续绕行，以及何时观测到16个不同频道而结束剩余发现任务。仅把已知中心放到前三站附近是代理评分；尚未见源、定向背侧、交会角和反馈变化都不在这一分数里，因此评分下降不保证整局省时。

## 尾部与局限

全部退化逐局T/LB、费用差、选向ID、首分叉、16源首次正向发现时刻以及动作数量变化，在 [development-mechanism.json](development-mechanism.json) 的 `stages.*.*.case_rows` 与 `comparison.loss_rows`，没有删除。

- centers随机最大收益625016：净省1351.321140秒，T/LB由3.243328降为2.608943。首次实际观测到16个不同频道从5703.711203秒提前至3546.756332秒，实际覆盖站数21→14；移动省1013.321140秒，测量和换频省338秒。选择本身没有缩短完整固定22站路线，而是实际更早满足了发现停止条件。
- centers随机最大退化625001：反慢740.378497秒，T/LB由2.951281升为3.352373。首次16频道发现由4351.503647延迟至5065.973014秒，实际覆盖16→20；移动多642.378497秒，检测和换频多98秒。首轮代理分数6058.577→4853.973米虽变小，却推迟了后续发现。
- mean随机最坏625009：集中度约0.99885仍反慢296.665949秒，T/LB由3.636011升为3.807182，其中移动增加254.665949秒。方向集中不是整局改善证书。
- 困难集两候选最坏同为625042：反慢192.600633秒，T/LB由2.970941升为3.060732；移动增加234.600633秒，测量/换频减少42秒。不能只统计测量减少。

这些都是开发后解释，不据此调ρ、前三站数、阈值或剔除场景。centers随机总净收益受625016单个大收益影响明显，后续完整独立验证必要。也不将旧621六局闭环QA混为新的性能证据。

## 身份与复现

生产源码提交 `a6e42b4d992905ce691c4a7de0c51d49a29419b2`，开发冻结HEAD `75f4788196ba279dc40ce090525ca29aff06eda6`，development-freeze SHA256 `334cd4e6cab4da5757a4a5fd7ff6aa3dd80fdced2a94a47da258fd7fb4051d2c`。root在冻结前修复了通用runner接受已选四臂 `selected_specs` 的衔接，保留旧协议兼容；本轮执行期间未修改任何源或审计。

输出位于 `results/q4_feedback_symmetry/{development,development-stress}`，各自保留manifest/freeze/source.zip、全部raw、summary、五审及同级console；固定RL在对应`*-rl`。所有行为为本地生成器，不访问官方模拟器、DB、SSH或GPU。

实际运行命令（重现必须改为尚不存在的新输出目录）：

```text
python -m experiments.run_q4_round2 --specs research/q4_feedback_symmetry/development-specs.json --stage pilot --start 625001 --count 24 --workers 3 --output results/q4_feedback_symmetry/development
python -m experiments.run_q4_round2 --specs research/q4_feedback_symmetry/development-specs.json --stage stress --start 625031 --count 14 --workers 3 --selection research/q4_feedback_symmetry/development-stress-release.json --output results/q4_feedback_symmetry/development-stress
```

每目录先 `run_q4_round2 --output DIR --audit`，再 `audit_q4_joint_visibility_prefix_batch --input DIR`、`audit_q4_joint_continuation_batch --input DIR`，以及 `audit_q4_feedback_symmetry_batch --input DIR --label compact_feedback_mean` / `compact_feedback_centers`。候选专项独立串联R8/R12及锁定覆盖链校验。完整选择器已由root执行 `python -m experiments.evaluate_q4_feedback_symmetry --phase development`，输出保存在研究目录的 `development-evaluation-console.txt`；它拒绝覆盖已有决定。
