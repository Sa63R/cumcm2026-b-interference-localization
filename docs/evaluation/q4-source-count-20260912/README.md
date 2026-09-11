# 问题 4 演练用时统计

本目录记录 2026-09-12 官方模拟器的演练结果，未使用正式测试。方法固定为已通过独立验证的 R12 `compact_joint_continuation`，配置 `after_active_miss_optical`，A* 扩展上限 200。策略资格对应源码提交 `bbbaf15944e73e27460a1c937675cfa38e79ddf4`；演练入口 SHA256 见 `summary.json`。

本次汇总保留此前同一冻结策略批次的全部 19 局，并加入随后新跑的全部演练。源数只能在每局结束后由模拟器结果确认；不按用时好坏筛选。补充目标是 10～16 每组至少 3 局，达到目标后完成正在进行的本局再停止。报告中的旧、新标签区分两批来源。

- `REPORT.md`：按源数分组的均值、样本最小值/最大值和标准差。
- `PER_RUN.md`：每局总用时、源数、每源用时、该局每源下界、倍率及测量次数。
- `summary.json`：未四舍五入的数值、分组统计及原始证据摘要哈希。

“每源用时”为整局计费虚拟时间除以该局干扰源总数，包括移动、频道切换、测量、光学定位和清除；它不是单次检测读数所需的固定 5 秒，也不是程序的现实运行时间。每源下界同样由该局下界除以源数。下界沿用历史条件性全清包含区域下界，不能把它解释为已求得实际最优路线时间。组内倍率为总用时除以总下界。范围仅描述已完成样本，标准差使用样本分母（局数减一）。

统计脚本逐局核对原始摘要、下界及登记文件的哈希、演练模式、问题号、全清数量、实际计费时间与策略入口。公开输出只保留白名单数值和证据哈希；带身份信息的原始通信日志留在被 Git 忽略的本地演练目录。

在项目根目录刷新：

```powershell
& '.venv-win/Scripts/python.exe' -B scripts/summarize_q4_source_count_practice.py --baseline results/practice_batches/q4-source-count-r12-20260912 --new-batch results/practice_batches/q4-source-count-r12-followup-20260912 --output docs/evaluation/q4-source-count-20260912
```
