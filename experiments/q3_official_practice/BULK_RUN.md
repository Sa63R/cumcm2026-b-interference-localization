# 900 轮第三问官方演练运行说明

用户要求“多跑一些样本，跑好多好多样本”。本批固定为 **九种方法各 100 例，共 900 个新增官方第三问演练案例**，原先九轮试跑保持独立。用户已明确授权 Windows 内部自动化，要求不占用 Mac 鼠标键盘。不要使用 CUA 操作 Mac，不进入任何正式测试或第四问。

## 运行状态

- 批次：`q3_900_20260912`。
- Windows 根目录：`C:\Users\baiwc\Downloads\Q3Practice`；批次位于 `batches\q3_900_20260912`。
- Mac 脚本目录：`/Users/zephyrr/竞赛/26国赛/数模/code/experiments/q3_official_practice`。
- 本地结果：`results/bulk/q3_900_20260912`。
- 固定计划 `bulk_config.json`：每组九种方法各一次，随机种子 20260912 打乱组内顺序；100 组。配置中的 `run_limit=0` 表示连续到计划末尾。
- 首组调试后已完成 11 例，149 个源全部清除。随后已于 2026-09-12 15:43 左右从第 12 例接续完整计划；**此段是历史启动说明，当前进度必须重新读取**。
- 最近启动的 Windows 计划任务：`Codex-Q3Practice-73e7d2f584`，从第 185 例接续批次，使用 guest SendInput 与返回列表等待。后续若重启会生成新名字，以 `results/created_tasks.txt` 和现场进程为准。
- 当前对话的跟进 automation ID：`900`，每 10 分钟检查。任务完成后暂停/停止该跟进。

在工作区根目录执行：

```bash
python3 code/experiments/q3_official_practice/bulk_status.py
python3 code/experiments/q3_official_practice/bulk_summary.py
```

第一条通过 UTM guest agent 获取新状态和逐例记录，UTM 命令在当前宿主环境需要 `require_escalated`。不要将陈旧本地文件的时间当作新进度。状态的 `updated_at` 使用 Windows -07:00 时区，Mac 使用 +08:00；实际耗时使用单调时钟。

`bulk_status.py` 保留 guest 完整 `status.json`，终端只显示简版。完整文件包括工作 PID。`completed.jsonl` 每轮一个登记，包含方法、案例、源数、耗时和官方日志哈希。OCR 的案例编号可能有字母错误，**正式关联以官方导出日志文件名中的案例编号为准**，OCR 原读数单独保留。

## 备份和校验

```bash
python3 code/experiments/q3_official_practice/backup_bulk.py
python3 code/experiments/q3_official_practice/validate_bulk.py
python3 code/experiments/q3_official_practice/bulk_summary.py
```

`backup_bulk.py` 调用 guest 的 `collect_bulk.py`，只打包已登记且策略进程已结束的案例以及调试证据，下载成 `evidence_NNNN.zip`，校验 ZIP CRC 和每个文件的 SHA-256 后解包。900 轮结束时必须做一次全量备份、`validate_bulk.py` 校验，并确认校验覆盖 900 例而非旧归档。程序结果、原始请求/响应和官方加密 `.jlog` 均须保留，不解密 `.jlog`。

`bulk_summary.py` 输出 `大样本结果.md` 和 `statistics.json`：每例先算 T/N，再案例等权平均；独立 bootstrap 10000 次给出均值 95% 区间；中位数和 P95；源数分组及按合并源数分布标准化的均值；实际执行与非请求部分耗时。样本不齐时明确标记阶段性结果。各方法案例不同，不称为同一场景配对实验。光学失败尝试不能删去，费用已计入模拟器虚拟时间。

## 断点恢复

先核对 Windows 工作 PID 是否还在，不启动第二个并发批次。状态停留但 PID 存活时，读取当前阶段的截图/OCR 文本和运行日志，区分正常计算、界面等待和真正异常。

`bulk_run.ps1` 以临时计划任务在已登录的 guest 用户会话运行；截图、OCR 和点击均在 Windows 内。当前已核验宽度 1710，窗口左上角 (-8,-8)，高度至少 880。屏幕宽度变化或窗口不匹配会停止，需按新的 guest 截图调整后再恢复。不要操纵 Mac 的鼠标键盘。

- `paused_after_limit` 且界面已返回列表：确认配置 `run_limit=0`、无 `resume_ready_index`，推送配置后再启动脚本。通常只用于最初的有界验证组。
- `error` 且当前策略已正常退出、尚未登记：修复识别问题后重启，脚本会读取已有 `result.json`，只补做界面/日志核验，不重复运行策略。
- `error` 且当前仅创建了案例、尚未建立 run 文件夹：确认 guest 正处于该轮第三问演练且尚未进入接口；可以临时将 `resume_ready_index` 设为该计划索引并推送，再启动脚本。成功后移除该配置。
- 当前策略已有失败结果：脚本会保留并停止。检查原因，不把它删掉后当作成功率的分母未发生。未知请求结果只能按原请求编号重试，不能用新编号盲目重复操作。
- 标记停止：在 guest 批目录创建名为 `STOP` 的文件，会在两轮之间停下。恢复前在确认用户仍希望继续时移除该标记。

启动方式（工作目录必须是脚本目录）：

```bash
python3 launch_desktop.py bulk_run.ps1
```

推送配置可在工作区根目录执行：

```bash
python3 code/experiments/q3_official_practice/utm_guest.py push code/experiments/q3_official_practice/bulk_config.json 'C:\Users\baiwc\Downloads\Q3Practice\bulk_config.json'
```

临时单步 UI 工作进程 `desktop_server.ps1` 与批量进程不要同时操作界面。必要时只在批量脚本已停止后使用。`guest_desktop.py snapshot` 返回 guest PNG，可以用 `view_image` 查看。

## 已处理的调试事件

最初自动化的倒计时 OCR 漏字导致提前发送 `/enter`，接口尚未开放，客户端保存了未知请求；未运行任何策略动作。按原请求编号恢复进入并正常退出，所有原始记录及官方日志保存在 `setup_attempts/countdown_enter`。随后从新案例开始固定 900 轮计划。该事件是自动化调试失败，与算法样本分开统计，报告中已说明。

当前就绪检查截取 y=210..355 区域，倒计时横幅存在时，该区域不包含“尚未进入”；另有固定等待，避免再次提前连接。结果核对容忍标题中的“测试”漏字，但必须同时认出“问题3演练…完成”、`/exit`、正常结束和日志已保存。总源数从总数或全向+定向之和读取；若两者均有效必须一致，并与成功清除数一致，否则停下保留现场。

第 176 例曾在启动就绪等待阶段超时。恢复时核实旧批测进程已经退出、第 176 例 run 目录不存在，Windows 实时截图仍为“可以开始”的第三问演练列表，最新日志仍为第 175 例。错误状态、OCR 截图、进程探针和恢复截图保存在 `automation_events/0176_start_guard_timeout`，备份程序包含该目录。从第 176 例重新启动固定计划；没有运行策略，因此该自动化事件不计为算法案例。

第 183 例策略正常退出并全清 10 个源，官方完成弹窗核验通过，但 `Add-Content` 写入成绩时被并发读文件的 Windows 共享锁阻止。原结果、日志、完成弹窗 OCR 和恢复时的完成页面均保留在 `automation_events/0183_ledger_sharing_lock`。`Append-Ledger` 现在只对打开文件时的共享锁冲突重试，最长 30 秒；写入一旦开始便不自动重试，避免未知写入结果造成重复登记。`test_ledger_lock.ps1` 在独立文件上通过实际四秒占用测试，确认 UTF-8、解锁后写入及无重复记录，并确认其他 IO 错误仍会停止。恢复时若完成弹窗已关闭，可用原始已核验弹窗 OCR 加上当前第三问已结束页面进行补登记，保存 `done_page` 截图；不重跑策略。

第 184 例已完成登记后，返回按钮未生效，下一例在列表检查处停止，185 尚未执行。现场保留于 `automation_events/0185_return_not_effective`。点击改用 guest 内一次 `SendInput` 插入移动、按下、松开三个事件，并在返回后等待列表真正出现；这两项需要以恢复后的连续新案例验证。恢复期间还看到 Windows 更新的计划重启提示，点击“别的时间”后提示消失，随后手动通过 guest 助手回到第三问演练列表；只读重启探针没有返回明确安排，不能宣称已确认取消重启。后续注意 Windows 更新弹窗或重启，不关闭更新服务、不改正式测试。

## 完成后

**最近检查的未解决状态（必须重新读取现场）：**184 例已完整备份并验证，185 的 SendInput 启动已切换到演练页面，但页面一直显示 `XXXX-XXXX-XXXX-XXXX`、`00:00`、`尚未进入`、`等待测试状态`，约 110 秒后就绪检查超时。未派发策略、未创建 run 目录，不能把它计入成绩，也不能当作仍在列表盲目点开始。现场保存在 `automation_events/0185_waiting_test_state`；先确认旧工作进程已停止、读取 live guest 页面，再判断服务端是否已分配案例以及是否适合 `resume_ready_index=185`。当前 SendInput 只证实切换页面，尚未通过恢复后的连续新案例验证。Windows 更新曾提示当地 03:32 重启，点击“别的时间”后提示消失，但尚未确认新的重启安排。

确认 900 例、每法 100 例、全量归档校验完成后，交付报告和证据包，解释统计波动及各方法使用不同随机场景的限制。

只清理本次记录在 `results/created_tasks.txt` 中的临时计划任务；确认它们已停止后注销。保留模拟器、Q4 环境和全部结果文件。不要用宽泛进程名杀掉用户其他程序。

Mac 临时防空闲睡眠进程记录于 `results/bulk/q3_900_20260912/power_guard.json`，初始 PID 12260，命令 `/usr/bin/caffeinate -i -t 21600`。它允许屏幕关闭，最长六小时。结束时先核对 PID 当前仍对应这条命令再发送 TERM，以提前释放临时断言，保留记录；Windows 脚本退出时会自动释放自己的临时执行状态。

完成后暂停 automation `900`，只在完成、异常或需要用户操作时通知；正常推进时更新本地文件并保持安静。
