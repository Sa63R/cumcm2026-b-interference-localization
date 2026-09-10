# 首次主动探测的安全质心：保留的负结果

只修改单源首次主动探测位置：从原 MEC 中心沿到观测多边形面积质心的线段移动，直接检查所有外包多边形顶点距探点不超过 `1000−1e-6` m。保证接收集合是凸集，沿线段二分保留可行端点；这不是全域欧氏投影。调度代表点仍为 MEC，后续探测、覆盖路线和实际清除证书都保持原 efficient 行为。该变体不加入共享测量，以隔离一个变量。

在已用于几何 pilot 的 train 103001–103016 上重新配对运行冻结 efficient 与该变体，共 32 次仿真。所有局全清、正常退出，零失败清除。仅使用本地仿真，未打开保留终测场景。完整身份、实际动作、终止后真值和汇总保存在 `results/geometric_joint/centroid_first_pilot_v1/`；实测总程序时间 2.163 s。

|指标|efficient_frozen|geometric_centroid_first|
|---|---:|---:|
|平均虚拟时间 / s|3512.990|3610.149|
|P95 / s|3889.539|4226.236|
|平均移动时间 / s|2690.240|2795.212|
|平均检测时间 / s|643.750|635.938|
|平均换频时间 / s|109.625|109.625|

变体平均**慢 97.159 s（2.766%）**，5 胜、11 负，配对平均节省的 bootstrap 95% 区间为 **[−180.208, −22.656] s**。最差回退 535.971 s，发生在 seed 103006。平均检测节省 7.813 s，而移动增加 104.972 s；光学、移除和换频平均成本不变。

因此不扩展到 64 局确认，也不将它合并到当前 joint 候选。代码、配置、测试和记录保留用于复现失败方向。已有轨迹上的“质心距真实源更近”是事后位置精度指标，不能推出从当前机器人位置开始的总路程更短；本次整局成本分解恰好显示移动费用抵消了测量收益。近源过冲是可解释的机制，但这里没有额外诊断证明全部回退都来自它。

验证：`python -m pytest tests/test_geometric_centroid.py tests/test_geometric_joint.py -q`，16 项通过。检查包括面积质心、接收安全线段截取、保持原调度点/后续探测、仅观测接口、实际完整费用和 Q3 范围限制。

复现：

```powershell
python -m experiments.geometric_joint_pilot --phase pilot --output results/geometric_joint/centroid_first_pilot_v1 --spec research/v1_baseline_efficient.json --spec research/v1_geometric_centroid_first.json
```

使用新的输出目录复跑，或者在来源身份完全一致时加 `--resume` 读取已有记录。结论仅覆盖本次训练场景；它足以拒绝本轮候选，不是证明所有质心类方法均无效。
