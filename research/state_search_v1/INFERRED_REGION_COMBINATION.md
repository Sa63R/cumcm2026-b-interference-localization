# 均值点调度与确定无信号删测：薄组合检查

## 组合方式

新增 `InferredMeanPointSearch(InferredSilenceSearch)`，直接绑定冻结的 `RegionStateSearch._next_task`，初始化同样的 12 节点均值点调度状态和日志。`_scan` 原样继承确定无信号删测，`_next_probe` 原样继承 axis_quantile；未修改两个已有候选的代码或默认配置，也没有加入未知源插入、active-sharing 或其他新机制。

推断负位置进入同一半径的区间约束是合法的：在证书成立的整个外包区域内，其到每个节点的距离都超过 1500，因此在精确算术下不会使 `min(1500, min_negative_distance)` 再下降。保留它主要用于几何更新语义的一致性。节点仍是面积求积代理，不因组合而变成官方精确后验。

## 先验证继承和记录，再做小样本比较

15 个针对性测试通过（5.61 s）：方法对象身份检验确认没有复制第二套扫描、局部探测或矩阵逻辑；禁用推断在两个场景的实际动作与原 mean_point 完全一致；困难场景与新训练场景使用只能访问合法观测的客户端，独立重建每次推断前的区域，验证未清源、严格距离证书、真实测量计数、未知频道七点实际无信号记录和清除区域证书。

冻结组合后，仅比较 111001..111016 的两种策略。自动附带的 efficient 仅是开发工具基线，下面只报告预先指定的配对：

| 16 局策略 | 平均虚拟时间/s | 移动/s | 检测/s | 切频/s | 推断次数 |
|---|---:|---:|---:|---:|---:|
| axis + inferred | 3237.783357 | 2479.408357 | 582.187500 | 107.437500 | 64 |
| mean_point + inferred | 3240.963234 | 2471.463234 | 590.937500 | 109.812500 | 48 |

两者全部 16 局全清、零失败清除。组合平均**慢 3.179877 s（0.0982%）**；以节省为正的配对 95% bootstrap 区间为 [-45.186563,37.010107] s，7 胜9负，最大退化 146.036678 s，P95 配对时间比 1.043320。平均实际执行墙钟 0.465587→0.432667 s，这一机器时差不代表稳定速度优势。

组合平均移动减少 7.945123 s，但检测和切频增加 11.125 s。16 局的任务序列和实际成功清除顺序全部改变，推断机会也由 64 次变为 48 次，说明“均值点收益+删测收益”不可简单相加。独立删测在自己的固定排序下有小幅可靠证据，并不能使区域代理的排序效果也变得可靠。

## 逐时审计

`summarize_inferred_region.py` 从这 32 条轨迹仅提取实际动作与明确标记的推断，不向审计函数传入归档的场景真值。112 次推断全部从此前合法记录重新算出严格距离证书；实际测量数与检测费用逐条一致；未知频道真实覆盖以及每次认证清除再次检查通过。

将推断虚拟地插回各自冻结动作序列，仅计检测和真实相邻频道切换，可分别得到 24.250 s/局、18.0625 s/局的局部服务费用差。这是该序列的记账审计，**不是又运行了一条原 mean_point 策略，更不是可相加的整局收益估计**。

原始数据在 `results/state_search/inferred_region_pilot/`，含冻结源码哈希、两候选完整真实及推断记录、逐局费用和配对汇总。本轮使用 Windows 合成训练环境；没有查看扩展验证或最终测试，也未运行官方模拟器。

## 结论与接口

组合安全性和继承语义通过本轮检查，但小样本没有显示超越 axis+inferred 的效果，之前独立 mean_point 的 64 局训练确认与 Linux 48 局公共验证收益区间也跨零。将组合作为待统一核验的备选保留，**不推荐根据这轮均值替换 axis+inferred，不追加参数搜索或新训练批次**。

入口为 `strategies.inferred_region_state_search:run_inferred_mean_point_state_search`，独立 spec 为 `experiments/state_search_candidate_inferred_mean_point_v1.json`；`enabled=false` 禁用推断而保留原 mean_point 调度。

```
python -m pytest tests/test_inferred_region_search.py -q
python experiments/run_state_search_development.py --configs experiments/state_search_inferred_region_configs.json --seed 111001 --count 16 --traces --output <新目录>
python experiments/summarize_inferred_region.py <新目录>
```
