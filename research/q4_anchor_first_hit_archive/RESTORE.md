# 恢复 R37 完整研究

永久标签 `archive/q4-r37-anchor-first-hit-rejected-20260912` 指向 `1970354e22b01cf3bbb6451423ba603fad6e7aae`，已推送，禁止删除或改指向。本目录仅为读者副本；完整源码、原始研究/结果、首次审计与诊断都在标签中。公开摘要采用workspace相对路径，不改原证据字节。

已从新空bare直接从远端depth-1 fetch该标签：目标对象事前不存在；恢复完整tree、182个变更blob、49个基础源码、169个本轮原研究/结果文件逐字节相同。恢复仓库 `q3-v1-artifacts/archive-checks-20260912/anchor-first-hit-tag-r37.git` 保留。冻结运行文件数量为69；详细raw/QA/proof计数和SHA见MANIFEST、ORIGINAL_EVIDENCE、RESTORED_BLOBS。

在确认新恢复目录不存在后：

```text
git fetch origin refs/tags/archive/q4-r37-anchor-first-hit-rejected-20260912:refs/tags/archive/q4-r37-anchor-first-hit-rejected-20260912
git -c core.autocrlf=false worktree add --detach ../q4-r37-anchor-first-hit-restored archive/q4-r37-anchor-first-hit-rejected-20260912
git -C ../q4-r37-anchor-first-hit-restored rev-parse HEAD
```

上述命令不运行模拟器。原路径分析脚本应从恢复树执行。当前状态为 archived_and_removed。core 归档 `e642e3dd1e3db07ce09a3c023b17cecd132c26b3` 推送完成后，已用不带 force 的 git worktree remove 移除原树，并按精确 HEAD lease 删除该实验分支；永久标签及独立恢复仓库保留。删除窗口其他 refs 零变化，10 个核心保护分支不变；证据见 MANIFEST 与 CLEANUP。未通过开发/几何门槛的研究不能写成独立验证失败或全体方法不可能，见RESULTS。
