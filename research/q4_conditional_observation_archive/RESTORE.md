# 恢复 R17 连续条件观测研究

完整提交 `27e8c71c4d701787a735bfcb073ad3661fa54867`；相对可靠 R12 `81aa6e1a1231a2358cf098fbba3fdd0557b540ef` 的增量包为 `../q4_round2/archives/conditional-observation-rejected.bundle`，SHA256 `dac705471e62fac78c515bea5de1600d9eb20904f7c91789e1a0bf2ae0d11131`（33046481 字节）。R17 工作树、本地和远程分支目前均保留；core 归档尚待 root 审查提交，本文件不授权删除。

已实际新建 bare 仓库，只预置深度1的 R12 基提交，先确认 R17 HEAD 不存在，再从本包导入。170 个新增或修改 Git blob、66 个冻结运行/审计源文件及 204 个原始冻结/控制/实验文件全部逐字节匹配；无删除路径。校验明细见 `RESTORED_BLOBS.json`、`ORIGINAL_EVIDENCE.json` 和 `MANIFEST.json`。

在已含 R12 基提交的 q4-round2 中执行以下本地命令，目标分支和目录必须不存在：

```text
git bundle verify research/q4_round2/archives/conditional-observation-rejected.bundle
git fetch research/q4_round2/archives/conditional-observation-rejected.bundle refs/heads/experiment/q4-r17-conditional-observation:refs/heads/restored/q4-r17-conditional-observation
git -c core.autocrlf=false worktree add ../q4-r17-conditional-observation-restored restored/q4-r17-conditional-observation
git -C ../q4-r17-conditional-observation-restored rev-parse HEAD
```

这些命令不运行策略、不连接模拟器，也不调用网络。用 `core.autocrlf=false` 保持冻结字节。包中保留原始119局、计划/源码归档/完整审计、失败门槛、全部源码与测试，以及冻结前旧621集成失败和修正记录；独立140+98局未打开。完整模型与方法证明保留在包和可读副本。

`reusable/` 额外保存单臂分层 runner、完整审核入口和其纯测试的原字节，需在完整恢复仓库解析依赖。未来采用这些工具须更换新未观察种子并重新冻结，不能把原626开发作为独立集。单臂负结果只能说明本方案未达预设500探索线，不能编造未运行旧方法的同场景因果比较。
