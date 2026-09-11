# 恢复 R13 换站诊断

归档 HEAD：`a455270c3b5396e15bfe2e70ecd160916b06366f`，包含原始未压缩 JSON 的追加提交；原研究提交为 `164f5b673b5cb02abbc13cb3788b86755910ed17`。增量 bundle 为 `../q4_round2/archives/cover-replacement-rejected.bundle`，SHA256 `9a5a093057a18b1c1cd8d22a5a008dae694c837b163e8601a3009442360abe37`，105872 字节。

需要已有 prerequisite `2def11d4b357181bfdf2d312530ddb39ff4cc6aa`；现有受保护 R9 核心分支保留该提交。本包不是独立于 prerequisite 的完整仓库。已在只预置该基础提交的新 bare 仓库实际导入，恢复 HEAD 与原始证据字节核对通过。

在 q4-round2 目录执行（恢复分支和目录须不存在）：

```text
git bundle verify research/q4_round2/archives/cover-replacement-rejected.bundle
git fetch research/q4_round2/archives/cover-replacement-rejected.bundle refs/heads/experiment/q4-r13-cover-replacement:refs/heads/restored/q4-r13-cover-replacement
git -c core.autocrlf=false worktree add ../q4-r13-cover-replacement-restored restored/q4-r13-cover-replacement
git -C ../q4-r13-cover-replacement-restored rev-parse HEAD
```

使用 `core.autocrlf=false` 保持已提交源及证据字节。完整源码、原始结果、压缩副本均在 bundle 内；本目录仅提供可读摘录，原相对路径在恢复工作树解析。不会启动策略、模拟器或网络测试。

`evidence.json` 是原 HEAD 的 LF 字节；`evidence-working-copy.json` 另存原工作树 CRLF 字节。二者只有换行差异，JSON 值相同；`MANIFEST.json` 同时记录两个哈希。两个未压缩原始 JSON 已逐字节对照既有 gzip 和 evidence SHA 后提交，恢复核验通过，不受摘要换行差异影响。
