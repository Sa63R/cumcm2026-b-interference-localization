# 第三问官方演练接口

本目录将 `q3_comparison` 的九种策略接入附件 2 HTTP 协议。2026-09-12 已在 UTM 内的 Windows 官方模拟器完成九轮第三问演练，113/113 个源全部清除，单轮实际执行 1.334–3.299 秒。详情见 [官方演练结果](官方演练结果.md)。

用户追加的大样本测试正在单独运行：九种方法各 100 例，共 900 例。当前进度和操作说明见 [BULK_RUN.md](BULK_RUN.md)，阶段性统计见 [大样本结果](results/bulk/q3_900_20260912/大样本结果.md)。它与上面的九轮试跑分别归档。

## 环境与操作

- UTM 虚拟机：`Quartus9-Windows11-ARM`，UUID `755E0E00-7B0E-4845-A9C3-7345C9BC2320`。
- 官方程序：`C:\Users\baiwc\Downloads\Q4Practice\Jammers-simulator\jammers-simulator.exe`。
- 已核验 EXE SHA-256：`2373b9e7af83735a04309e2983eb433ec46faf7e0b8494410ce7fded2a297c27`，与用户下载的 7z 内原文件相同。
- 本次策略与独立依赖：`C:\Users\baiwc\Downloads\Q3Practice`。
- 复用已有 Python：`C:\Users\baiwc\Downloads\Q4Practice\python\python.exe`，3.13.15 ARM64。
- NumPy 2.5.3 官方 PyPI Windows ARM64 wheel，放在 Q3Practice/lib，没有修改 Q4 依赖。
- Windows 不启用 Numba。原 v3 已有无 JIT 后备实现；新方法复用同一个装饰器。计算公式及策略参数保持一致。

先在界面进入**问题 3 演练测试**，记下案例编号并等待倒计时结束，再在 Windows 运行；总源数在完成后的界面核对：

```powershell
& C:\Users\baiwc\Downloads\Q4Practice\python\python.exe C:\Users\baiwc\Downloads\Q3Practice\bootstrap.py --method v3 --robot-id <界面队号> --case-label <界面案例编号> --practice-confirmed --output C:\Users\baiwc\Downloads\Q3Practice\practice_results
```

支持 phased、joint、v2、v3、v3_origin20、optical、scenario、future_cover、scenario_future。

`--practice-confirmed` 是操作者已检查界面的声明，接口自身无法查询当前题号或测试模式。每次只执行一轮，并正常 `/exit`。程序不会选择或启动正式测试。

## 不占用 Mac 鼠标键盘的方式

Windows 内的 `desktop_server.ps1` 通过临时计划任务运行在已登录用户会话中。它只操作该虚拟机内的官方模拟器窗口；Mac 侧通过 `utmctl` 传输命令文件和截图，无需将 UTM 窗口保持在 Mac 前台。虚拟机须继续运行，Windows 登录会话须可用。本次已验证后五种策略通过该方式完成。

在本目录启动：

```bash
python3 launch_desktop.py desktop_server.ps1
python3 guest_desktop.py snapshot
```

读取返回的截图，确认当前页面后，使用 `guest_desktop.py click --x <横坐标> --y <纵坐标>` 在 **Windows 原生截图坐标** 内点击。不要套用 Mac 窗口坐标；分辨率变化后需重新确认位置。

确认进入第三问演练并取得新案例编号后：

```bash
python3 dispatch_run.py --method v3 --robot-id <界面队号> --case <案例编号> --practice-confirmed
python3 read_latest.py
```

`dispatch_run.py` 返回仅表示已派发；以 `read_latest.py` 返回的 `policy_completed_and_exited` 及官方界面为完成依据。运行结束用 `python3 guest_desktop.py quit` 停止工作进程。本次创建的临时任务均已删除，模拟器保留运行；核验记录见 [cleanup_status.json](results/cleanup_status.json)。后续启动会创建新的临时任务，其名字记录于 `results/created_tasks.txt`，结束后仅清理本次记录的对应任务。

## 计时与验证

- result.json：官方累计虚拟时间、成功清除数、每个成功清除的平均虚拟秒数、进入到退出的实际墙钟时间、策略部分墙钟时间、HTTP 请求墙钟时间，以及二者之差。
- requests.jsonl：客户端完整请求、响应和已确认状态。通信结果未知时不发送另一种新操作。
- 移动在下一次 measure/clear 中提交；clear 不改变测向仪频率；清除成功 5 秒、失败 3 秒。每次接受响应都复核计时。
- 示向度以度转弧度，几何界限含 0.005 度取整余量，共 1.005 度。
- 策略只能访问当前位置、测向频率与公开观测。终止要求 20 个频道全部已清除或已证明不存在。
- optical 加载原文件中未经修改的函数和类，排除仅用于原 CSV 报表的 pandas 导入和 main。
- 本地 `test_adapter.py` 验证九种方法与本地对应案例的虚拟时间精确一致，并验证 clear 不换频及失败清除计时。两项测试通过（第一项包括九个策略子用例）。

## 当前证据边界

`results/windows_dependency_probe.json` 仅为 Windows 内运行自建场景的依赖/执行检查，九种策略均清除同一例全部 11 个源。它不是官方模拟器结果，不能代替官方性能比较。首次 phased 包含首次加载开销，也不能据此比较策略计算速度。

本次九轮官方演练每种方法一例，案例互不相同，因此可验证接口、清除完成情况和本批实际执行速度，不能据此作公平的策略效率排名。配对效率比较仍见 [400 组本地结果](../q3_comparison/本地速度对比.md)。optical 有一次失败光学尝试，已按 3 秒计入虚拟时间，最终全部清除。

原始请求/响应、结果和官方行为日志在 [q3_evidence.zip](results/q3_evidence.zip)，解包后位于 `results/evidence`。`python3 summarize.py` 会复核文件大小与哈希、请求接受状态、清除数、完成证书、逐步计时及日志对应关系，并生成报告与 `results/summary.json`。未进入接口即结束的案例 `9G4U-6EVR-ZN9J-CCDN` 没有对应策略，单独保留且不计入比较。
