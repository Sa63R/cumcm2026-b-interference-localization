# 恢复 R33 顺路增量预算研究

固定归档标签 `archive/q4-r33-transit-budget-rejected-20260912` 指向 `524345e3a0fc10d5c54fdc3afe67767f9bb148e4`，已推送。禁止删除或改指向该标签。完整生产源码、119 条开发记录、两局旧 QA 的原始记录与首次/最终合同审计、源码包、67 文件冻结身份、计划和结果均在标签中；本目录只提供精简读者副本，原文件 SHA 见 MANIFEST 与 ORIGINAL_EVIDENCE。

已从全新空 bare 仓库直接向 origin 作 depth-1 标签 fetch：事前目标不存在，恢复后的完整 tree、178 项变更 blob、67 运行文件、49 基础源码及 168 项原始研究/结果文件均逐字节核验。恢复 bare 保留。归档已随 core 提交 `a241ef19b2f6fe227658c9b737a3dd19834b31c5` 推送，之后只清理了 R33 工作树及本地、远程实验分支，状态为 `archived_and_removed`。归档标签、独立恢复 bare 与全部读者副本保留。CLEANUP 及引用快照记录精确 lease 删除：10 个保护分支及其他引用未变。

在原仓库中确认新恢复目录尚不存在后执行：

```text
git fetch origin refs/tags/archive/q4-r33-transit-budget-rejected-20260912:refs/tags/archive/q4-r33-transit-budget-rejected-20260912
git -c core.autocrlf=false worktree add --detach ../q4-r33-transit-budget-restored archive/q4-r33-transit-budget-rejected-20260912
git -C ../q4-r33-transit-budget-restored rev-parse HEAD
```

恢复不调用模拟器。分析脚本应从恢复树的原研究路径执行。119 局全清且首审全部通过，48 局发生真实新增服务；困难组 523.892313 秒/源未过 500 门槛，选择器正常拒绝晋级，6342001/6344001 独立集未打开。随机组 499.149259 秒/源不代表达到 460，更不构成同场景改善证据；完整 T/LB 与作用范围见 RESULTS。
