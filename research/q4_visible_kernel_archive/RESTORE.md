# 恢复 q4-r11-visible-kernel 研究路线

失败路线完整归档 HEAD：`108bec55bd628cccfaf9407e8b4645d1d1d965e1`。增量 bundle 为 `../q4_round2/archives/visible-kernel-rejected.bundle`，SHA256 `3a18fc3e0f986f9a5c6bb32a90519e665ceccb2b816279188b27796db526ae48`，16101298 字节。

需要仓库中已有 prerequisite `1188fd553b91f855f6374185e15821ca2ac5bcd3`；现有受保护核心分支保留该提交。本包不是独立于 prerequisite 的完整仓库。已在只预置该基础提交的新 bare 仓库中实际导入，恢复 HEAD 及本目录所有材料字节核对通过。

在 q4-round2 目录执行以下命令（恢复分支和目录必须尚不存在）：

```text
git bundle verify research/q4_round2/archives/visible-kernel-rejected.bundle
git fetch research/q4_round2/archives/visible-kernel-rejected.bundle refs/heads/experiment/q4-r11-visible-kernel:refs/heads/restored/q4-r11-visible-kernel
git -c core.autocrlf=false worktree add ../q4-r11-visible-kernel-restored restored/q4-r11-visible-kernel
git -C ../q4-r11-visible-kernel-restored rev-parse HEAD
```

明确使用 `core.autocrlf=false`，避免恢复时改变源代码与证据的换行字节。恢复后核对 HEAD，再按原研究目录的身份清单核 SHA。这里的可读副本不是完整工作树，原报告中的相对源码/记录路径在恢复工作树中解析。正式测试不属于恢复操作；恢复不启动策略或模拟器。

R11 的纯几何理论、精确证书、独立审计器、构造测试、旧前缀诊断和完整五臂开发证据全部保留在 bundle 内；本目录另提供理论与失败结论的可读副本。
