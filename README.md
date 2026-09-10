# CUMCM 2026 B题 无线电干扰源自动定位与清除

四问模型、保守几何定位、全向与混合源搜索、官方接口客户端、本地成对试验、论文与材料生成工具。

截至2026-09-10：**288项自动化测试通过；第三问新版在100个独立随机场景均全清，较原策略平均节省31.16%总虚拟耗时，7个困难场景也全部全清。** 原第三问官方演练已核验13/13；新版尚待官方演练，第四问官方总源数尚待GUI核验，六次正式记录仍为空。论文暂停更新。本地研究不代替官方成绩。

- [完成情况与手动清单](资料汇总/完成情况与手动清单.md)
- [论文PDF](论文/B题论文.pdf) / [可编辑正文](论文/B题论文.md)
- [本地实验报告](资料汇总/T06本地实验报告.md)
- [第三问新版策略、独立验证与下界对照](资料汇总/第三问清除策略改进与独立验证.md)
- [官方操作指南](资料汇总/手动操作与正式测试指南.md)

## 在本机运行

Windows独立环境 `.venv-win` 已建立，旧Linux `.venv` 保留。先在模拟器界面选择正确题号和演练模式，开始一局并等接口就绪，再双击“启动问题3演练.cmd”或“启动问题4演练.cmd”，按提示输入当前登录队号。正式入口在完成官方演练后使用。

也可以在项目根目录的PowerShell执行：

```powershell
.\scripts\run_session.ps1 -Problem 3 -Mode practice
.\scripts\run_session.ps1 -Problem 4 -Mode practice
```

脚本不需要账号密码。它不能代替GUI切换模式；`-Mode`记录用户声明，HTTP没有模式查询接口。默认地址为 `http://127.0.0.1:2026`。问题3默认 `efficient + center`（1150米覆盖与联合路线），问题4默认 `triangular + center`；问题3旧版用 `-Variant adaptive`。

在新Windows设备初始化：

```powershell
.\scripts\setup.ps1
# 测试、作图和PDF生成依赖：
.\scripts\setup.ps1 -WithTests -WithReports
```

若PowerShell策略阻止脚本，可用仓库中的双击CMD入口，或一次性执行 `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1`；该参数仅作用于当前进程。

## 本地验证与复现

最新第三问研究位于 `results/q3_optimization`，复现命令见[新版报告](资料汇总/第三问清除策略改进与独立验证.md)。独立样本种子3000—3099与开发的2000—2029分开；冻结参数、逐局轨迹、成对统计和共同下界均保留。平均耗时5044.02→3472.45秒；平均耗时/平均下界2.627→1.809，这不是全局近似比。7个困难场景中的边界例变慢15.20%，不能保证逐场提速。

以下 `results/study` 的749次运行和论文对应优化前版本，保留作历史研究，不代表新版求解器。论文和支撑包本轮未重新生成。

核心求解器依赖Python标准库；完整报告环境在Python3.12上验证，依赖见 `requirements-test.txt` 和 `requirements-report.txt`。从项目根目录运行：

```powershell
$env:PYTHONPATH = 'src'
.\.venv-win\Scripts\python.exe -m pytest -q
.\.venv-win\Scripts\python.exe -m experiments.run_study --output results/reproduced --random-cases 100 --start-seed 1000
.\.venv-win\Scripts\python.exe -m experiments.make_figures --study results/study
.\.venv-win\Scripts\python.exe -m experiments.build_paper
.\.venv-win\Scripts\python.exe -m experiments.package_support
```

新实验输出目录必须尚不存在。论文默认使用已归档 `results/study`；若切换实验目录，给 `make_figures` 和 `build_paper` 同时传入 `--study results/reproduced`。Linux使用同样的模块命令，先建立虚拟环境并 `pip install -e '.[test,report]'`。

| 历史本地随机案例，每问100局 | 全清局数 | 各局T/K平均值 | 相对基准平均总虚拟时间 |
|---|---:|---:|---:|
| 问题3 即时中心定位 | 100/100 | 384.14 s/源 | 降低12.60% |
| 问题4 三角网格延迟定位 | 100/100 | 962.92 s/源 | 降低12.65% |

两问另各有7个困难案例，各对照策略也全部全清。三角策略在部分困难例上慢于基准，不能认为每个案例都会提速。记录包含种子、代码摘要、动作轨迹、真值和五类虚拟耗时；真值只在会话退出后交给评估器。

## 模型与模块

| 路径 | 内容 |
|---|---|
| `src/geometry` | 半平面交、凸包、旋转卡壳直径、最小包围圆 |
| `src/localization` | 外包候选集、保证再接收域、第二点规则 |
| `src/planning` | 七点全向覆盖、方格/三角定向覆盖、28m光学网格、开路2-opt |
| `src/strategies` | 发现、局部定位、清除与终止证书 |
| `src/simulator_client` | 串行HTTP、幂等重试、状态和计时 |
| `src/simulation` | 独立本地物理引擎和场景生成器 |
| `src/workflow` | 正式结果登记、原日志归档、审计与表格导出 |
| `experiments` | 实验、图表、论文PDF和支撑ZIP生成 |

证明见 [T02/T03模型](资料汇总/T02与T03数学模型.md) 与 [T04/T05覆盖证明](资料汇总/T04与T05覆盖及终止证明.md)。单源直径不超过40m不充分；清除采用外包区域最小圆半径不超过19.9m，或近场响应，或有限光学网格兜底。

## 正式材料

六次正式测试须来自官方GUI启动的两问各三次案例。运行后按 [登记与审计说明](支撑材料/登记与审计说明.md) 登记案例编码、GUI现实时间、原始加密日志和上传状态，再执行：

```powershell
.\scripts\materials.ps1 -Action export-tables
.\scripts\materials.ps1 -Action audit
```

现有六槽为空，审计返回未完成是正确结果。明文会话可能包含登录队号，保存在本地，不纳入公开Git提交或匿名支撑包；官方加密日志保留原名原内容。

客户端遇到结果未知时不发送新动作；未完成、超时、中断或退出未确认均返回非零退出码。若GUI已关闭本局，不能复位后续跑同一案例。

本地题面建议北京时间2026年9月13日15:30前完成正式测试；17:30后不能开始新测试。论文与支撑压缩包分别不超过20MB，单份官方加密日志不超过2MB。
