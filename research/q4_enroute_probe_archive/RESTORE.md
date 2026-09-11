# 恢复R25沿覆盖边补测研究

完整HEAD `07318d2b701d3a7456081ac61e508ade1dffa972`；唯一前置提交是可靠R12 `81aa6e1a1231a2358cf098fbba3fdd0557b540ef`。完整增量包为 `../q4_round2/archives/enroute-probe-rejected.bundle`，SHA256 `9b933c253d6a9e8a070c5fdeb9d50396f63910dd8d59455452fdd42fa2099066`，共50778061字节。归档准备时状态为 **archived_pending_cleanup**：工作树、本地和远程分支均保留，待root复核并提交推送归档后，再执行用户已授权的无效新分支清理。最新生命周期见MANIFEST与archives/enroute-probe-rejected.json。

已新建独立bare仓库，仅通过本地depth-1 fetch取得R12，先确认R25目标HEAD不存在，再由本bundle导入。恢复HEAD/tree一致，164个新增或修改blob、67冻结文件、224个严格原始文件全部逐字节通过；49个R12原src（44个.py与5个.gitkeep）均保持原样。results的全部133个文件、old-smoke的全部6个文件均单独核过。详见MANIFEST、RESTORATION_CHECK、RESTORED_BLOBS、ORIGINAL_EVIDENCE及BUNDLE_VERIFY。

在已含R12提交的q4-round2中，确认恢复分支和目录均不存在后执行：

```text
git bundle verify research/q4_round2/archives/enroute-probe-rejected.bundle
git fetch research/q4_round2/archives/enroute-probe-rejected.bundle refs/heads/experiment/q4-r25-enroute-probe:refs/heads/restored/q4-r25-enroute-probe
git -c core.autocrlf=false worktree add ../q4-r25-enroute-probe-restored restored/q4-r25-enroute-probe
git -C ../q4-r25-enroute-probe-restored rev-parse HEAD
```

以上为纯本地Git命令，不连接模拟器、不运行案例。`core.autocrlf=false`及原仓库文本属性保护冻结字节；恢复后应核MANIFEST与全部67文件哈希。完整bundle含119局原始gzip、每局失败请求、首次审计、源码zip、计划及控制哈希，也含冻结前3个旧621烟测。可读副本不重复每局原始大文件；复制的分析脚本应放回完整恢复仓库的原research路径执行才能找到原始records。

两组mean(T/N)为512.261449、519.138386秒/源，119局全清且审计通过，但未过500开发继续线；116局有实际新增补测，独立6312001/6314001未打开，460目标未完成。单臂没有同场景R12性能反事实，不能编造相对退化。保留失败原因与完整方法；完成归档核验与推送后，只清理本轮新建的这条失败分支，不动原核心分支。
