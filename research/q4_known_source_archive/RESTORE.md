# 恢复 R34 已知源联合调度研究

固定归档标签 `archive/q4-r34-known-source-rejected-20260912` 指向 `eb95af2d29d332e2fe5a9cd5c94d8215497e239f`，已推送。不得删除或改指向该标签。完整源码、119条开发记录、2局旧QA原始记录及首次审计、冻结清单/源码包/计划、阴性选择和全部首审证据保留于标签；本目录只放原字节读者副本，119个gz不重复复制。

从全新空bare直接向origin执行depth-1单标签fetch，事前目标HEAD不存在；恢复后完整tree、181个变更blob、68项冻结runtime、49个base源码文件和169项原始research/results文件均逐字节核验。恢复仓库保留于 `q3-v1-artifacts/archive-checks-20260912/known-source-tag-r34.git`。核验明细见RESTORED_BLOBS与ORIGINAL_EVIDENCE。归档状态及后续清理记录以MANIFEST/CLEANUP为准。

恢复到尚不存在的新目录（从主仓库执行）：

```text
git fetch origin refs/tags/archive/q4-r34-known-source-rejected-20260912:refs/tags/archive/q4-r34-known-source-rejected-20260912
git -c core.autocrlf=false worktree add --detach ../q4-r34-known-source-restored archive/q4-r34-known-source-rejected-20260912
git -C ../q4-r34-known-source-restored rev-parse HEAD
```

恢复不调用模拟器。分析脚本从恢复树原路径以 `python -m research.q4_known_source.analyze_development --output <新输出文件>` 执行；原输出拒覆盖。119局全清且首次审计全通过，但随机535.820651/压力538.340582秒每源均超过冻结500门槛；全部119局触发宽源真实服务，仍未晋级，预留独立未开启。T、N、T/N、LB及meanT/meanLB、真实费用与代理遗漏边界见RESULTS；不将本单臂结果当不同种子算法的因果差。
