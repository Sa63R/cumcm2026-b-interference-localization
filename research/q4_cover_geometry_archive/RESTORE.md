# R10 归档恢复

方法、失败原因与完整T/LB比较见RESULTS.md。此目录另保存完整22点覆盖证书及其独立重放结果。纯几何结果仍有效，整局候选没有通过开发门槛；尚未运行独立场景，不计为一轮独立失败。

全部源码、267个公开几何候选记录、190条开发轨迹及RL对照在 ../q4_round2/archives/cover-geometry-rejected.bundle 中。恢复需要已有1188fd553b91f855f6374185e15821ca2ac5bcd3祖先，本仓库保留该R8核心分支。

```text
git bundle verify research/q4_round2/archives/cover-geometry-rejected.bundle
git fetch research/q4_round2/archives/cover-geometry-rejected.bundle refs/heads/experiment/q4-r10-cover-geometry:refs/heads/restored-q4-r10
```

bundle的SHA256为79001c4a97aa4b857043c67d6f4a3f967a32b4d155e36bf29a701a7a2872ef05。恢复后RESULTS中的原相对结果/源码路径均有效；本目录不是整个实验工作树的复制。
