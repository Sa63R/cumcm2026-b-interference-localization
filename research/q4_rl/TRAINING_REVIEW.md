# 本地训练批次审阅工具

`scripts/q4_analyze_training.py` 只读取命令中逐一明确列出的本地批次，以及可选的已下载 checkpoint 和事后下界文件。它不发现其他案例、不访问数据库或网络、不启动模拟器、采样或训练，也不修改策略。输出必须采用新文件名；输入文件名、SHA-256、审阅脚本 SHA-256 和 checkpoint SHA-256 随报告保存。

```powershell
python scripts/q4_analyze_training.py `
  --batch results/q4_rl/server-training-pilot-001/batch-000000.json.gz `
  --bc-batch results/q4_rl/server-training-pilot-001/batch-000000.json.gz `
  --checkpoint results/q4_rl/server-training-pilot-001/checkpoint-000001.pt `
  --output results/q4_rl/server-training-pilot-001/review-new.json
```

省略 `--checkpoint` 即只做账单和轨迹审计，无需 PyTorch。checkpoint 模式需要现有 CPU PyTorch 环境，固定线程数为 1、禁用 CUDA。多批次使用重复 `--batch`；整个文件的所有局均纳入。只有明确属于启发式暖启动的数据才能重复指定 `--bc-batch`，它必须也是一个已包含的 `--batch`；对 PPO 数据指定 BC 会报错，不会把学习动作冒充启发式标签。

审阅内容：

- 每局所有 transition 费用等于最终实际账单，fallback 完整记入，失败总费用为 `max(actual, 360000)`，逐步 return 为无折扣剩余总费用的负值除以固定 `1000`。
- 从保存的真实物理请求逐项重算移动、切频、测量、光学扫描和拆除费用，验证每一时刻账单与策略动作日志一致，并用终止后的真值核对实际清除结果、剩余源、失败清除次数。真值仅用于离线审核，不输入网络；这里不声称重新证明连续覆盖证书。
- 检查统一下界版本、组成、场景 hash 和费用口径；不重新求解下界 DP。早期 smoke 缺少物理日志或下界时明确显示缺证据，不重新生成案例。已有 `posthoc_bounds.json` 可以通过显式 `--bounds` 引用，按 seed、可用分层字段和实际时间匹配，歧义报错。
- 汇总全清与失败、平均/P95/最大实际用时、含失败惩罚用时、下界及总时间/总下界、fallback 占比、决策动作类别、候选移动距离与实际移动距离、切频和各物理费用。计算开销是逐局 worker 累计时间，不能当成整批并行墙钟或 PPO 更新费用。
- 可选 checkpoint 在已保存的公开状态上计算精确策略熵、归一化熵和 critic explained variance。它是明确提供的 checkpoint，可能与轨迹生成时的行为策略不同。行为策略单个已选动作的 `log_prob` 只足以报告 sampled surprisal，不足以恢复精确熵。
- BC 同时报告精确候选索引命中，以及全部 16 维公开候选特征转为 float32 后完全相等的保守等价类命中；不使用频道编号定义等价。该等价不意味着不同后验或不同公开特征的两个候选同样好。

所有失败和解析错误保留。出现任意审计错误或无解释的缺局时，整批平均值/比值停止输出，避免只对成功解析的子集给出有利统计。缺少旧版物理证据的完整账单仍可描述，但独立物理全清记为未验证，不能升级为可靠版本。训练批次随策略改变、内部相关，报告为训练诊断，不提供独立泛化或配对改进结论。

## 首批真实远程数据核验

完整读取 `server-training-pilot-001/batch-000000.json.gz` 的 16 局、2048 个决策，所有物理/return/阶段账单核验通过：实际全清 16/16，平均实际时间 `7729.949832 s`，P95 `8977.897035 s`，最大 `9560.913230 s`；平均统一下界 `1746.586238 s`，总时间/总下界 `4.425748`。575 次失败清除全部保留，其中 555 次为覆盖网格的合法未命中、20 次为推测性先清除未命中；near/certified 清除失败 0。

fallback 占总实际费用 `70.893983%`。主决策为 1989 次 measure、59 次 service；最终实际包括 4568 次测量和 788 次光学清除。移动费用合计 `93948.197306 s`，切频 `4101 s`、测量 `22840 s`、光学 `2364 s`、拆除 `426 s`。逐局 worker CPU 合计 `15.963525 s`，worker 墙钟之和 `16.163757 s`，不是并行 makespan。

`checkpoint-000001.pt` 是本批更新后权重，在本批状态上的平均熵 `5.716317 nats`，相对 `log(合法候选数)` 归一化熵 `0.999785`；BC 精确和公开特征等价命中均为 `56.396484%`，1571 个标签状态存在公开特征等价候选；critic EV `0.025047`。高熵说明暖启动后的分布仍接近均匀，BC 的 argmax 命中不能解释为已获得同等强度的随机行为策略。该批来自启发式暖启动，不能与开发基线不同场景的均值直接比较。

完整机器报告为 `results/q4_rl/server-training-pilot-001/review-batch-000000-v2.json`。`review-batch-000000-first.json`、`review-batch-000000-v1.json`、`review-batch-000000.json` 是开发诊断工具时产生的中间报告，原始输入未改动；以带最终审阅脚本 hash 的 v2 报告为准。早期 smoke 的实际原文件也已核验，`training-smoke/review-legacy-v2.json` 如实记录 2 局都缺少物理证据，独立实际全清比值为 null。6 项小型单测覆盖失败保留、缺证据、解析错误阻止子集汇总、费用/return 篡改、float32 等价和常数目标 EV，没有运行新案例或训练。
