# 第三问前五名：本地复原模拟器配对评测

用户要求使用现有本地模拟器测试之前官方演练暂列前五的方法：v3 原点扫描、位置联合设计、提前光学、v2、v3。

使用工作区 `local-jammers-simulator` 的当前 `Session`、`Engine` 和场景生成器，版本与策略副本固定在 `vendor/`，逐文件源路径、大小和 SHA-256 见 `source_manifest.json`。代码未调参。此模拟器基于静态复原，不称为官方实测，不沿用此前 ToySimulator 的自建误差模型。

已完成：2000 个新固定种子第三问地图，每种方法在完全相同的地图和噪声场上执行，共 10000 轮；全部全清、逐轮重放与独立审计通过。另外用独立的 50 个地图、每方法两遍，完成 500 轮串行计时。原 UTM 跟进保持 PAUSED，已有 184 例官方演练记录保留。

[结果报告](本地模拟器前五名结果.md) · [2000 个案例逐例成绩](逐例成绩.md) · [完整统计](statistics.json) · [主批审计](results/paired2000/validation.json)

提前光学的平均成绩最低，为 239.261 秒/源；v3 为 240.892 秒/源。提前光学在 1736/2000 个配对案例中更快，平均节省 1.631 秒/源。位置联合设计与 v3 的均值差为 +0.032 秒/源，配对 95% 区间包含零。本机预热后串行执行以 v3 最快，均值 0.0492 秒/例。

运行 Python：`../q3_comparison/.venv/bin/python`（在此目录执行）。

```bash
../q3_comparison/.venv/bin/python test_adapter.py
../q3_comparison/.venv/bin/python runner.py --cases 2000 --workers 2 --out results/paired2000
../q3_comparison/.venv/bin/python runner.py --cases 50 --workers 1 --repeats 2 --seed-prefix q3-top5-timing-20260912- --out results/timing50
```

策略只接收公开观测；真值由评测器在策略退出后读取，用于核对全清。逐轮保存完整场景和动作响应的 `.json.gz`，每轮均重放并检查响应及微秒计时。失败案例和失败光学尝试保留，费用计入成绩。脚本不会访问 Windows、既有网页服务或官方接口。

计时包括 enter、策略创建、规划及进程内模拟接口、exit；排除场景生成、模块导入/JIT 预热、导出和重放。并行批测的实际耗时受竞争影响，以独立串行计时结果比较计算速度。
