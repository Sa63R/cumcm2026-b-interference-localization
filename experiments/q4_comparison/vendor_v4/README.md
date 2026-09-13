# B题第四问 V4：每轮测向后重新规划

Python 3.10 及以上，只依赖标准库。本文和所有结果均为自建模拟，不是官方演练、正式测试或加密日志。

## 运行

```bash
python -m unittest -v test_optimized test_v3 test_v4
python run_v4_validation.py --cases 100 --workers 4
python benchmark_routing.py
```

完整复现包含2400个不同案例：前三版的1800例与新的600例。V3和完整V4都运行2400例，另两个消融版本各运行新的600例，共6000次策略运行。

仅复现新的600例：

```bash
python run_v4_validation.py --fresh-only --cases 100 --workers 4
```

## 接入

沿用原有Device接口，只暴露position、channel、move、detect、clear，不读取目标数量、坐标、半径或朝向。
具体HTTP端点和字段必须依据题目附件1、2实现；本包不猜测官方协议。

```python
from q4_v4_solver import solve_v4, V4Config
from q4_v4_local import LocalConfig

# 完整V4：RF成对测向后重新规划，光学覆盖最多12点。
report = solve_v4(device, V4Config())

# 改动较少：仍使用V3的6点光学覆盖，只改为每轮RF测向后重新规划。
report = solve_v4(device, V4Config(local=LocalConfig(optical_cover_limit=6)))

# 只加速本地计算，不改变默认V3的动作路线或虚拟耗时。
from q4_v3_cached import solve_v3_cached
report = solve_v3_cached(device)
```

三个入口每次应使用独立、刚初始化的测试会话，不能在同一已运行会话中连续执行三遍。

光学未命中反馈必须可靠。题面没有给出附件中的失败返回字段。本地模型把失败光学操作计为3秒，命中再加2秒。没有核对真实失败语义时，可设置`optical_cover_limit=0`禁用多点光学覆盖；本轮未报告该设置的速度收益。

## 文件

- `q4_v4_solver.py`：全局循环、共享观测、21点覆盖和完成判据。
- `q4_v4_local.py`：可中断的成对测向；轮数持久化，最多12轮后进入原保底算法。
- `q4_route_cached.py`：缓存距离的多起点2-opt，保持原计算顺序及破平局规则。
- `q4_v3_cached.py`：默认V3的只计算加速入口。
- `q4_v3_*.py`、其他基础模块：保留的参考实现与基础几何，支持直接对照。
- `results_v4/paired_cases.csv`：6000次运行的逐例指标。
- `results_v4/validation_summary.json`：分组、配对改进、置信区间、失败成本敏感性、运行代码哈希。
- `results_v4/LOCAL_ONLY_*.json`：6份本地动作记录，不是官方日志。
- `results_v4/coverage_certificate.json`：沿用21点布局的4228个方格证书。
- `development/`：两批开发实验和未采用方向的记录，不属于新留出测试。

## 本轮结论

新的600例：V3 476.5602秒/源，完整V4 470.1980秒/源，下降1.3350%；仅RF重规划版470.7488秒/源，下降1.2194%。完整V4有402例更快、185例更慢、13例持平，最差个例变慢38.47%。旧1800例加新600例共31060个目标，V3和完整V4都全部清除。

没有逐例占优保证，没有全局最优保证，也没有完成官方测试。详见《V4优化报告.md》。
