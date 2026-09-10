# 可选的 cited_six_disk 下界审计 v1

这是原批量审计结果的一份**独立补充**，不会修改 `certify_bounds.py`、`audit_eval_bounds.py`、物理 DP 缓存、评估策略或 `v1_protocol.json`。原审计默认输出和默认下界值保留。

使用方式（Windows / Linux 相同）：

```text
python research/theory_v1/audit_cited_six_disk.py research/theory_v1/results/validation_bounds_hot.json --output research/theory_v1/results/validation_cited_six_disk.json
```

输入必须是 `audit_eval_bounds.py` 已完成的批量 JSON，版本 `q3-physical-disk-dp-and-information-v2`。脚本仅打开显式指定的这个文件，不扫描目录、不打开原始 eval.gz、不访问模拟器或隐藏场景、不执行 DP。原始逐动作账本及事后真值核验由原审计完成；本入口复用其事实字段，并核对源数、占用频道、原下界分项、原舍入、记录身份和版本的一致性。

默认仅接受 validation 6000–6047。未来对明确获准且已经冻结的其他评估批次，可显式传 `--allow-heldout`；本次没有使用该开关，没有读取 sealed / final / extended 数据。

输出版本为 `q3-optional-cited-six-disk-v1`。每行原字段原样保留，新界放在独立的 `cited_six_disk` 对象下，同时记录输入批量文件 SHA256、增强脚本 SHA256、外部定理来源、证明查阅层级和适用前提。拒绝把增强结果再次输入，以免重复加费；命令行也拒绝覆盖原输入文件。

## 公式与前提

引用依据见 [六圆定理补录](SIX_DISK_COVER_THEOREM_AUDIT.md)。这里仅采用允许失败清除的**混合策略共同界**，不使用观察到的零失败次数去推断全局零失败策略保证。

记旧连续认证界为 B、原按实际物理动作数计算的舍入修正为 δ，真实源数为 N，E=20−N。增强式为

\[
\Delta=\begin{cases}3E,&N<16,\\0,&N=16,\end{cases}
\qquad B_{\rm cited}=B+\Delta,\qquad
L_{\rm cited}=\max\{0,B_{\rm cited}-\delta\}.
\]

这等价于把旧空频道动作及首次切换项 `31E−1[频道 1 为空]` 替换为 `34E−1[频道 1 为空]`（仅 N<16）；成功清除费与移动项不变，舍入修正保留原值且仅扣一次。N=16 的空频道/空间认证项仍为零。增强脚本没有采用 35/36 秒的零失败策略类别界。

**条件仍是策略对所有合法隐藏场景保证完成认证**，以及 N<16 时能够添加一个与既有历史不可区分的合法隐藏源。单次或若干次全清成功，不能证明该策略条件；此脚本也不声称完成其证明。成功行显示的是这个条件下的时间比，失败行保留但不计算全清比。源位置 DP 仍只是连续清除路线的下界，引用六圆定理不会使它变成精确在线最优解。

## 已有 validation 的复算例子

仅使用 `results/validation_bounds_hot.json` 的 48 场景 × 2 基线，共 96 条已经审计的记录。两组各含 6 条 N=16 记录，均不加认证增量；全部 96 条通过增强式一致性核验。耗时约 **0.0064 秒**，DP 调用与 eval.gz 打开数均为零。

|基线|原平均认证界 / s|增强平均认证界 / s|平均实际耗时 / s|原均值之比|增强均值之比|
|---|---:|---:|---:|---:|---:|
|efficient_frozen|1944.326367|1964.638867|3413.024728|1.755376|1.737228|
|rollout_frozen|1944.326368|1964.638868|3373.252971|1.734921|1.716984|

逐局下界增量范围 **0–30 秒**，平均 **20.3125 秒**。表中比值为“成功行平均实际耗时 / 成功行平均下界”，不是逐局比值的平均。策略实际耗时未改变，不能把比值下降称为算法提速，也不代表已经知道可实现的最短在线时间。

复算输出：`results/validation_cited_six_disk.json`。独立增强测试覆盖 N=10/15/16、初始频道是否为空、保留舍入、零失败观察不改变共同界、失败行资格、原数据不变、不执行 DP、拒绝重复增强与损坏字段、默认种子范围及成功轨迹违反条件界时的报错。与六圆证书测试合计 **16 项通过**：

```text
python -m pytest research/theory_v1/test_audit_cited_six_disk.py research/theory_v1/test_six_disk_certificate.py -q
```
