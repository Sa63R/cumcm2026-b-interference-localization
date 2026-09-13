# B题第三问：v3 平均动作时间优化版

这是策略代码、自建模拟器和可复现实验，**不是已经接入官方模拟器的程序，也不是正式成绩或官方日志**。

## 运行

建议先创建独立的 Python 环境；程序使用 Python 3.10+ 的语法。

```bash
python -m pip install -r requirements.txt
python -c "from q3_v3 import warmup; warmup()"
python -m unittest -v test_optimized test_v3
python bench_v3.py --start 5000 --cases 400 --modes v2 v3 --out results/my_holdout.csv
```

`numpy` 是必需依赖；`numba` 用于编译数值计算循环。没有 Numba 时仍有纯 Python 回退，但计算速度会下降。第一次编译或读取缓存的时间，不应与本文预热后的计算时间混为一谈。

`v2` 精确保留上一版 `combined`；`v1` 是更早的七点联合策略；`v3` 是本次冻结的默认版本。所有动作仍串行执行，停止后才检测，不能并行发送机器狗动作来绕过计费。`--workers` 只并行自建的独立测试案例。

## 默认配置和边界

v3 默认不在原点做全频道扫描，使用经过连续覆盖认证的七个未来搜索点，采用有前后偏移的测向候选点、数值编译的多起点路线搜索，以及考虑下一段路径的安全清除点。每个源最多4次启发式选点机会，随后回退到有收缩保证的圆心递进。

这是一种平均时间优先策略，**不是逐案例都比v2快**。新400组中257组更快、143组更慢。250组边界/最小接收半径压力测试虽然全部清除，但平均虚拟时间比v2慢1.89%。对源分布靠近原点或边界等情况，不应假定删除原点扫描一定有利。

保留原点全扫描的模式：

```bash
python bench_v3.py --start 2000 --cases 200 --modes v2 v3 v3_origin20 --out results/origin_comparison.csv
```

其他消融：`v3_no_probe`、`v3_no_route`、`v3_no_through`。`v3_prior` 额外使用自建模拟器的均匀接收半径先验，不在默认版本中启用；题目没有规定这个分布。

## 官方接入

读取真实附件1、附件2后，实现 `q3_base.Backend` 的 `pos`、`channel`、`move(target)`、`detect(channel)`、`clear(channel)`，再运行：

```python
from q3_v3 import FastAgent, PublicBackend, warmup

warmup()  # 无环境动作，可在进入官方测试之前执行
# backend = YourVerifiedOfficialBackend(...)
# agent = FastAgent(PublicBackend(backend))
# agent.run()
```

`detect` 必须转换成 `Observation('none')`、`Observation('strong')` 或 `Observation('bearing', angle_in_radians)`；坐标单位为米。适配器必须根据实际响应同步位置和频道，处理初始化/退出、剩余时间和错误。不可猜测HTTP路径、JSON字段或对收费动作无条件重试。官方光学、清除及切频语义需要按附件核对。

自建环境约定：移动5米/秒、切频1秒、检测5秒；频道定向清除前按需切频，光学3秒、成功清除2秒。默认策略没有失败光学尝试；环境虽定义失败光学3秒，这个约定没有用于获取默认策略收益。

## 实验组织

* 开发：0–79号种子（部分筛选只用0–39），记录74种配置、4360次案例—配置运行。重复运行同一案例不等于独立样本。
* 回归：2000–2199，200组；上一版v2时间与已交付CSV最大差异约4.55e-13秒。
* 冻结后新验证：5000–5399，400组，只比较预先冻结的v2和v3。
* 压力：91000–91049的50个边界案例×5种误差，另20个特殊布局×5种误差，共350组；两版本均运行。
* 计算时间：新验证集前50组，两次串行重复并交换版本顺序；排除首次JIT编译、HTTP和环境创建，包含策略初始化和执行。

自建随机分布：N均匀取10–16；频道无放回；源位置按圆域面积均匀分布；固定有效半径均匀取1000–1500米。空间哈希误差由案例种子、频道和坐标确定，同点重复误差不变。压力误差另外覆盖固定±1度、按频道交替±1度和平滑空间场。

这些概率分布是自建测试选择，**不是题面或官方模拟器的已知分布**。不同位置的哈希误差可能使不同平台上的极小坐标舍入差异改变后续读数；完整原始结果及本次软件版本已保留。

## 文件

`q3_v3.py` 是默认策略入口；`probe_score.py` 和 `route_search.py` 是可选编译的数值核；`q3_optimized.py`、`q3_base.py` 保留前两版算法和自建环境。

`bench_v3.py`、`stress_v3.py`、`runtime_check.py` 分别执行随机配对、压力、预热计算时间测试。`summarize_results.py` 从原始CSV重新计算均值及配对bootstrap。`test_v3.py`、`test_optimized.py` 包含28项单元及审计测试。

`q3_experiments.py` 与 `bench_development.py` 保存开发分支，**不是推荐把所有开关一起打开**。其中保留了预条件会拒绝的配置，例如删除原点扫描却把全部七点放在1050米外圈，会遗漏中心；这种不满足覆盖证书的配置不计入74种成功配置的CSV统计。

报告见 `optimization_report_v3.md`。`results/` 的所有日志与CSV均为自建结果，不应改名冒充官方正式日志。
