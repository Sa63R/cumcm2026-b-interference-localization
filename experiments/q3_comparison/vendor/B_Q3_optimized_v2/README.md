# B题第三问：优化版算法与自建测试

推荐配置：`combined`。本包不是官方模拟器客户端，不包含官方正式测试成绩或加密日志。

## 运行

要求 Python 3.10 或更高版本、NumPy。进入本目录后执行：

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_optimized.py
python q3_optimized.py --cases 20 --modes baseline combined --out results/my_comparison.csv
```

多进程批测（并行的是互相独立的自建案例，不是同一只机器狗的动作）：

```bash
python benchmark_parallel.py --start 2000 --cases 200 --workers 4 --modes baseline negative_info lateral_only opportunity_only full combined combined_no_lateral combined_no_info --out results/my_validation.csv
python benchmark_parallel.py --start 20000 --cases 20 --workers 4 --modes combined --stress --noises hash smooth plus minus alternating --out results/my_boundary_stress.csv
```

## 核心文件

- `q3_base.py`：上一版算法及自建模拟器，冻结保存，供配对比较。
- `q3_optimized.py`：负观测约束、候选测向点评分、动态覆盖、覆盖点移动、多起点路径优化、精确安全清除区域。
- `benchmark_parallel.py`：配对批测工具。
- `test_optimized.py`：14项几何、数值回归、覆盖、强信号与接口隔离测试。
- `optimization_report.md`：模型、推导、局限与结果说明。
- `results/`：逐案例数据、均值表、统计汇总与自建动作日志。

`combined` 配置在代码中是：

```python
agent = OptimizedAgent(
    backend,
    info=True,
    lateral=True,
    dynamic=True,
    opportunity=True,
    relocate=True,
    better_route=True,
    exact_clear=True,
    opportunity_gain=30,
)
agent.run()
```

## 结果口径

开发使用种子0–39，共18种配置；原200组回归使用种子0–199。最终新增验证使用种子2000–2199；不使用这批验证数据继续调参。早期几何调试使用过种子1041，它不在最终新增验证集中。

所有数据是**自建模拟**：源数均匀取10–16，位置在1800米圆域内按面积均匀取样，频道无放回抽样，接收半径均匀取1000–1500米；常规误差为由种子、频道和位置确定的有界空间函数。这些分布不是题面指定的官方生成分布。

自建环境沿用上一版动作计费：移动、测向、切频、光学定位及清除逐项累计；面向频道的清除在需要时也计一次切频。实际HTTP附件尚未提供，这一接口层约定不能当成官方协议。

原200组：平均总虚拟耗时3415.35秒 → 3162.52秒，减少7.40%。
新增200组：3409.20秒 → 3148.54秒，减少7.65%，187组更快、13组更慢。
原200组、新增200组与110组压力案例均全部清除；最终测试记录中失败清除次数为0。14项单元测试通过。

新增验证集中，题目定义的“每案例总时间/清除源数”再对案例取平均为268.18秒/源 → 248.36秒/源。不要混淆该指标、平均总时间以及跨案例合并后的总时间/总源数。

## 官方环境接入

`Backend` 只向策略提供当前位置、当前频道，以及 `move`、`detect`、`clear`。应根据真实附件1、附件2实现适配器；当前位置、频道及动作成功状态必须以服务器响应为准。不要自行猜测URL、JSON字段、角度单位、失败收费或重试语义。

模拟器的进入/退出、剩余程序时间、HTTP错误处理及正式日志导出需在官方适配层完成。串行执行同一案例的设备动作；不要把本地多案例并行测试误用为同一设备的并发操作。

如果观测使定位区域为空、认证测向点接收不到信号，或认证清除失败，程序会报错，而不是悄悄丢弃约束或宣布任务完成。
