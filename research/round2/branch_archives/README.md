# 可恢复的试验分支归档

本目录的Git bundle只包含对应试验在共同基准8c624d08c61c0e411f93345d63a932b01ead4868之后的提交及所需新对象。该基准属于受保护核心分支，必须先存在。两个bundle均通过git bundle verify；原始策略配置和每局完整结果另在results/round2中，源码zip不依赖仍保留工作树。

| 文件 | 归档分支最终提交 | 冻结实验源码提交 |
| --- | --- | --- |
| feedback.bundle | a19d9b69ba0cafbc3a90932f8273b9a949b6e300 | 7b755aa9ce8ebd9c4da1a2c8741d460165f5135d |
| clear_lens.bundle | 2d045bed0b6107b35134ce4a8a61a03a7f03a985 | dbc0ea552510fd72ccd42b4f82fe1eb34511bd63 |

从仓库根恢复（恢复到新名称，避免覆盖现有分支）：

```powershell
git bundle verify research/round2/branch_archives/feedback.bundle
git fetch research/round2/branch_archives/feedback.bundle refs/heads/research/q3-r2-feedback:refs/heads/restored/q3-feedback-r2
git worktree add ../restored-q3-feedback-r2 restored/q3-feedback-r2
```

clear_lens同理替换bundle文件名和源ref。旧pilot使用的运行器原文在research/round2/frozen_harnesses/pilot_runner.py；追加批有独立runner.py。manifest中的绝对source_root是当时执行位置，原始记录不能直接编辑来伪装换机复现；换机重跑应先恢复源码，再prepare到新的输出目录并保留新manifest，比较同一场景的动作与虚拟费用。完整审核可直接对已归档原始记录运行round2_posthoc_audit.py，不需要恢复试验工作树。
