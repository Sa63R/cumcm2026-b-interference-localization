# 恢复 R15 原地未知频道补扫研究

完整归档 HEAD `89d1399b48cdc1c6a7f486b1cccb0cafc55287c2`，bundle `../q4_round2/archives/opportunistic-discovery-rejected.bundle`，SHA256 `5a5fb9a0058bafd7d8c302406122fcb5557dbe6f75563a4fd528241db6ef25fe`。

这是增量包，需要已保留的 R12 基础提交 `81aa6e1a1231a2358cf098fbba3fdd0557b540ef`。已在只预置该基础提交的新 bare 仓库实际恢复；导入前目标 HEAD 不存在，导入后全部 364 个源码、控制、模型和开发证据哈希吻合。源代码四个提交及未使用的 stress release 元数据修正历史均保留。

在 q4-round2 目录执行，恢复分支和目录须尚不存在：

```text
git bundle verify research/q4_round2/archives/opportunistic-discovery-rejected.bundle
git fetch research/q4_round2/archives/opportunistic-discovery-rejected.bundle refs/heads/experiment/q4-r15-opportunistic-discovery:refs/heads/restored/q4-r15-opportunistic-discovery
git -c core.autocrlf=false worktree add ../q4-r15-opportunistic-discovery-restored restored/q4-r15-opportunistic-discovery
git -C ../q4-r15-opportunistic-discovery-restored rev-parse HEAD
```

`core.autocrlf=false` 保留冻结字节。恢复后按 `ORIGINAL_EVIDENCE.json` 核对；此目录是可读副本，报告中的相对源码/结果路径应在恢复工作树解析。全部228条开发记录、各审计、固定RL原档及源码/模型身份均可恢复。未开独立集，恢复命令不运行策略或模拟器，也不调用网络。选择器的历史RL身份来自仍受保护的R9 sibling树；包内保留相同五项源包/模型/worker/runner字节，不需要重新训练。
