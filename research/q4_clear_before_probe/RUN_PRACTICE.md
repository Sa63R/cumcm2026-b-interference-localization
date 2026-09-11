# R8 合格方法演练入口

仅运行 `compact_clear_before_probe`：
`strategies.q4_clear_before_probe:run_q4_clear_before_probe`，冻结 kwargs 为
`{"max_expansions": 200}`，默认 `config=center_once`。算法源码冻结于
`6d7ab88c133531946d7af832ee804008046b3508`；本交接入口的后续提交不改变该算法。
资格来自本目录 `selection.json`、`qualification.json` 的独立确认 64 局与压力 42 局。

在 `D:\jwt\2026数模国赛\q4-r8-clear-before-probe` 的 PowerShell 执行离线预检：

```powershell
& ..\cumcm2026-b-interference-localization\.venv-win\Scripts\python.exe -B experiments\run_q4_clear_before_probe_practice.py --preflight-only
```

此次实际预检通过，未连接模拟器。输出包含源码、配置、资格、入口、下界工具及两个源码归档的 SHA256：

| 文件 | SHA256 |
|---|---|
| selection.json | `5cfe90775fcaa0db122b4214badbb49b1bf1d2acc077ff87c46079802f1e12cc` |
| qualification.json | `329c95ee7973f22eeca7827d90d8d0e84fc0d453682c22cba11112fc7d92e357` |
| 两组 source.zip，各自相同 | `bfec28f977c89f7a1be9165a292fbc753c0c3f3d54e1d94ebbc1925229360ed4` |
| run_q4_clear_before_probe_practice.py | `3c5484cb855bc3b71c8393f632ecd275805b7dc3eedc4deba0b2f9adb98d1332` |

需要实际演练时使用下面命令（本次交付没有执行）：

```powershell
& ..\cumcm2026-b-interference-localization\.venv-win\Scripts\python.exe -B experiments\run_q4_clear_before_probe_practice.py --robot-id 202627001104 --simulator-dir 'D:\jwt\2026数模国赛\模拟器\数模2026\CUMCM2026B\Jammers-simulator-full' --repeat 1
```

路径必须是实际包含 `jammers-simulator-full.exe` 的目录；调试端口默认 19226。
入口要求模拟器明确空闲，并在接触桥接器前取得该目录下
`.practice-control/controller.lock` 共享锁。采集器占用锁、已有演练、正式测试、状态不明或残留案例编码都会拒绝；不会终止既有场次、绕开采集器或接管正在运行的演练。
入口通过原有 `run_once` 只创建 Q4 演练，原有按案例校验的客户端仍在每次请求前确认所属场次。
没有正式测试模式参数。控制器内部 `triangular` 是兼容传输标签，实际求解器始终是上面的固定 R8 方法。

预检检查本分支源码与冻结清单、获选配置、确认/压力场景编号、冻结清单摘要、全部物理与专项审计计数、资格所列 10 项文件摘要，以及两份源码 ZIP 的完整成员和内容。运行前后重复校验，并将 ZIP 原始字节摘要绑定到本次批次。
这些检查用于发现版本和证据漂移，不是对可同时修改代码及证据者的防攻击签名。

每局原始摘要、登记结果和下界写入新的 `results/practice_batches/q4-r8-…` 目录；已有输出不会覆盖。
复用原有登记后下界工具，只有已退出且匹配官方终局登记、确认全清的场次才给出
`time_to_conditional_lower_bound_ratio`，分母是旧口径
`conditional_guaranteed_all_clear_lower_s`。未完成或未验证全清不报告成功比值，也不继续下一局。
本入口用于单方法演练，不把不同演练案例当作与 Q4 RL 的配对比较；后续本地配对实验应在同批场景并列当时冻结的最佳 Q4 RL。

离线定向测试：

```powershell
& ..\cumcm2026-b-interference-localization\.venv-win\Scripts\python.exe -B -m pytest tests\test_q4_clear_before_probe_practice.py -q
```

44 项通过：网络接口全部禁用，覆盖真实预检逻辑、资格/归档漂移拒绝、锁先于桥接器、空闲限定、仅注入 R8、动作预算固定 20000、登记后计算下界、未完成停止及输出保留。
