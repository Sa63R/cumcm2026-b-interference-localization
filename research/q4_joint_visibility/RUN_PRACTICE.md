# 合格R9的演练入口

入口 `experiments/run_q4_joint_visibility_practice.py` 固定调用 `run_q4_joint_visibility(config="probe", max_expansions=200)`。没有正式测试选项。

```powershell
& ..\cumcm2026-b-interference-localization\.venv-win\Scripts\python.exe -B experiments/run_q4_joint_visibility_practice.py --preflight-only
```

真实离线预检通过：检查冻结策略、选择器/协议/RL身份、18份资格证据、128/84场景×4状态臂完整配对矩阵、两份源码ZIP完整成员、全部审计计数和重算晋级门槛。42项独立离线测试通过，测试中禁用真实socket。

实际运行使用CLI的 `--robot-id`（或环境变量 `CUMCM_ROBOT_ID`）及 `--simulator-dir` 指向包含模拟器EXE的目录，可用 `--repeat 1`。先取得模拟器目录的共享controller.lock，再确认当前状态明确空闲；有正式场次、其它演练、残留案例或采集器占锁就退出。程序不终止既有场次，也不恢复采集器。

2026-09-12 02:54本地首次官方演练已完成，13/13全清；T=6673.890281秒，历史口径条件下界=2085.3740759251605秒，T/LB=3.2003324286263504。官方终局登记核验和实际wire前缀的R8/联合可见性审计全部通过。未使用正式测试。原始记录保存在本地被忽略的 `results/practice_batches/q4-r9-20260911T185419099166Z`。

`practice-validation/result.json` 保存数值、专项审计和原始文件SHA；`observed-record.json.gz`仅导出实际观测及策略日志，剔除了账号、队号、arena和request标识。`verify_q4_joint_practice_record.py` 可从原始记录复核。它不能补出未到达位置的反馈，不能当成可任意重跑同场景的完整模拟器。

此单局用于官方运行可行性检查。没有同一官方案例的RL对照，不把其它案例的均值拼接成配对收益；独立本地的同场景RL比较见RESULTS.md。历史官方下界使用清除位置对应的包含圆放松，本地共同LB使用环境退出后真实配置，均沿用各自既有口径，不能把两个分母说成完全相同的观测信息。

入口提交e57b016d，入口文件SHA256 `ae1887aa014bd00b3acbb6912c91f6bf10f8d9a1dd78e334769f4762a7e3ccd9`；独立算法冻结e3c6cda0，两组源码ZIP均为 `efc2eacf50b1cb23658502cfa4f63546e63c2d13d1192fc3b8888f1d42e70b7a`。后续分析/文档提交不改变这些策略字节。
