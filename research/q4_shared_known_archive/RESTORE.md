# 恢复 R35 清除后共享观测研究

固定归档标签 `archive/q4-r35-shared-known-rejected-20260912` 指向 `01dda5cb5e99710be03c25e8f95f8e53d20a0d6e`，已推送。不得删除或改指向该标签。完整源码、119条开发记录、2局旧QA原始记录及首次审计、冻结清单/源码包/计划、阴性选择和全部首审证据保留于标签；本目录只放原字节读者副本，119个gz不重复复制。

从全新空bare直接向origin执行depth-1单标签fetch，事前目标HEAD不存在；恢复后完整tree、183个变更blob、70项冻结runtime、49个base源码文件和168项原始research/results文件均逐字节核验。恢复仓库保留于 `q3-v1-artifacts/archive-checks-20260912/shared-known-tag-r35.git`。核验明细见RESTORED_BLOBS与ORIGINAL_EVIDENCE。归档状态及后续清理记录以MANIFEST/CLEANUP为准。

恢复到尚不存在的新目录（从主仓库执行）：

```text
git fetch origin refs/tags/archive/q4-r35-shared-known-rejected-20260912:refs/tags/archive/q4-r35-shared-known-rejected-20260912
git -c core.autocrlf=false worktree add --detach ../q4-r35-shared-known-restored archive/q4-r35-shared-known-rejected-20260912
git -C ../q4-r35-shared-known-restored rev-parse HEAD
```

恢复不调用模拟器。分析脚本从恢复树原路径以 `python -m research.q4_shared_known.analyze_development --output <新输出文件>` 执行；原输出拒覆盖。119局全清且首次审计全通过，但随机542.348736/压力535.880992秒每源均超过冻结500门槛；107局实际触发共享观测，仍未晋级，预留独立未开启。T、N、T/N、LB及meanT/meanLB、真实费用与代理遗漏边界见RESULTS；不将本单臂结果当不同种子算法的因果差。

归档报告已随core提交 `9bb620d0c01994d8dbbc57bb15bef18e20135eca` 推送后，才用无强制的 `git worktree remove` 删除本轮R35工作树，再以精确expected-HEAD lease删除远程实验分支和本地同名实验分支。归档标签与fresh bare恢复副本保留；原10保护分支及全部既有foreign refs在即时清理窗口内均未变，并行新增引用若有则逐项记录于CLEANUP，不归因本任务。
