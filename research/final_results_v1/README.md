# 首版独立最终统计报告

已冻结的四种方法均完成 **256 个随机案例＋28 个压力案例**，共 1136 次评估；全部成功完成、失败清除为 0。状态搜索和强化学习通过预先声明的共同统计门槛；几何方法有正向改善，但未达到随机均值改善至少 5% 的门槛。

| 方法 | 随机平均时间（秒） | 相对基线改善 | 配对节省 95% 区间（秒） | 随机赢／输 | 随机最坏退化（秒） |
|---|---:|---:|---:|---:|---:|
| 冻结 rollout | 3361.655233 | — | — | — | — |
| 状态搜索＋未来覆盖移站 | 3081.408646 | 8.336565% | [259.474685, 301.173847] | 241／15 | 125.081450 |
| 强化学习 GAE=.95 / u512 | 3149.787235 | 6.302490% | [187.035923, 237.121621] | 216／40 | 365.446723 |
| 几何法＋未来覆盖移站 | 3214.959620 | 4.363791% | [125.789058, 167.347664] | 204／52 | 267.734124 |

三种方法的随机及压力 P95 比均低于 1.05；压力集上状态搜索 28 赢 0 输、强化学习 25 赢 3 输、几何法 26 赢 2 输。各压力家族仅四局，描述结果与随机集分开报告，不将其混入随机均值或置信区间。

几何方法“未通过”指未达到预设的实用改善幅度，不表示没有效果：其相对基线的节省区间仍完全为正。所有方向和不利案例都保留，没有根据最终结果更换候选。

## 文件入口

- [首版研究结论](FIRST_VERSION_REPORT.md)：三个方法、理论保证范围、下界及下一版方向。
- [独立串行运行成本](runtime/README.md)：整局程序实际时间及监督工具拒绝记录。
- [总交付包复现与演练说明](PACKAGE_REPRODUCTION.md)：在总包布局下执行的离线恢复、报告复算与人工环节。
- [中文统计解释与七家族表](final-acceptance/statistical-interpretation.md)：门槛解释、P95、最不利案例与每家族结果。
- [随机集共同报告](report-random/comparison.md)／[完整精度 JSON](report-random/comparison.json)。
- [压力集共同报告](report-stress/comparison.md)／[完整精度 JSON](report-stress/comparison.json)。
- [机器可读最终验收](final-acceptance/final_acceptance.json)。
- [压力家族描述统计](final-acceptance/stress-families.json)：保留全部 28 个配对案例。
- [来源与冻结身份](statistics-provenance.json)：原 registry、selection、identity audit 摘要、实际源提交、spec/checkpoint 哈希、平台及阈值。
- [报告执行记录](statistics-status.json)：独立工具 `d5c082b`、各步骤退出码和 11 个原始输出文件字节哈希。
- [只读报告生成脚本](tools/await_final_statistics.py)：等待最终身份审计通过后才生成报告。

两套图分别为 [随机 PNG](report-random/comparison.png)／[SVG](report-random/comparison.svg) 和 [压力 PNG](report-stress/comparison.png)／[SVG](report-stress/comparison.svg)，已目视核验。曲线横坐标是每个方法各自排序后的案例名次，不是相同场景沿横轴直接逐方法比较。

本目录不重复存放 1136 份原始行为档案；远端原始记录位于 `/home/volleyball/q3-research-v1/v1-selection/evaluations/`，由总交付归档保留。下载后已重新核验全部 11 个报告输出文件、生成脚本和来源记录的字节摘要。

## 结论范围

这是固定本地研究分布上的独立最终证据，未使用正式模拟器次数。置信区间是各预声明方向相对基线的配对对比，不能当作事后挑选全体最佳方法的同时置信保证，也不证明全局最优或普遍可靠性。物理／几何／下界深审以及公开案例的串行程序计时，由独立工序单独归档；本目录中的并发 wall time 不用于算法计算速度排名。
