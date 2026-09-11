# R12 仅演练入口

唯一固定方法为`strategies.q4_joint_continuation:run_q4_joint_continuation`，配置`after_active_miss_optical`、200次原A*扩展上限、6次主动测量上限。独立源码冻结bbbaf159；128随机与84压力配对及四层审计和冻结PPO对照全部完成后才放行。

从本工作树执行，机器人队号通过命令参数或本地环境变量提供，不写进源码或公开材料：

```powershell
python -B experiments/run_q4_joint_continuation_practice.py --preflight-only
python -B experiments/run_q4_joint_continuation_practice.py --robot-id '<本机队号>' --simulator-dir '<模拟器目录>' --repeat 1
```

必须保留兄弟`q4-r9-joint-visibility`树中的固定RL源码、权重及adapter。入口重新检查全部20份配对证据、真正的候选晋级门槛、60份源文件/zip内容、选择和evaluator依赖SHA；51项禁止真实联网的离线测试通过，已对本次真实证据preflight成功。80份源与证据的Git对象字节也已逐项核对SHA一致；本分支属性文件保留其原始换行，保证新检出可复核。

入口先取得共享`controller.lock`再访问模拟器，只有明确空闲才新建问题4演练，不接管已有案例、未知状态或采集会话。没有正式测试选项；`triangular`只是旧控制器传输标签，实际注入上述唯一R12策略。每次运行前后重核冻结证据，登记结束后才计算历史条件包含圆下界；失败也保留全部本地记录并停止后续重复。

原始输出在忽略的`results/practice_batches/q4-r12-<UTC时间>/`，含本地身份字段，不提交。可用`experiments/verify_q4_continuation_practice_record.py --input <run-001目录> --output <新公开目录>`核对实际wire前缀、R8及递归证书、登记全清数和T，然后导出白名单观测记录和去标识结果。该观测记录不能为未测坐标提供反事实反馈。

2026-09-12 03:30已完成官方问题4演练QXJ9-TKNG-68F2-3ZGG：10/10全清，T=6644.712802秒，历史条件LB=1902.646119223秒，T/LB=3.492353483，程序墙钟22.594秒。实际R8与递归前缀检查、登记源数及T核对均通过。该局没有触发辅助区刷新，验证了新入口和继承流程的实际运行，不能说它单独检验了所有新机制。

原始完整身份材料保留本地`results/practice_batches/q4-r12-20260911T193020230013Z/`。公开`practice-validation/{result.json,observed-record.json.gz}`只含实际观测与去标识结果，已检查队号/robot/arena/team/access token字段不存在。单局不能用于证明平均优于旧策略或PPO，公平比较见RESULTS.md的独立同场景对照。官方历史包含圆下界与本地终止后真值共同下界信息不同，分别沿用原口径。
