# 第四问本地五组方法对照

已完成：60 个新案例 × 5 方法 = 300 次完整任务，全部清除成功（每种方法均涉及 761 个源）。六类场景各 10 例，使用相同场景、种子与位置误差函数配对运行。结果为 LOCAL_ONLY，不是官方测试成绩。

| 方法 | 任务秒/源 | 相对 V4 平均降低 | 计算秒/例 |
|---|---:|---:|---:|
| 原 V4 | 478.74 | 基准 | 0.027 |
| 仅解析概率 | 478.42 | 0.07% | 0.032 |
| 加一步 rollout | 478.39 | 0.07% | 2.751 |
| 再加共享测向点 | 478.07 | 0.14% | 3.124 |
| 再加动态覆盖替代 | 478.38 | 0.08% | 4.888 |

最佳均值只快 0.14%，三种复杂前瞻的收益区间均跨过零。这一版没有兑现明显提速；不能把这次有限候选的一步前瞻结果推广为所有 POMDP、动态覆盖或更深搜索的结论。

- [完整实测报告](results/holdout60/实测报告.md)
- [逐例结果及决策记录](results/holdout60/runs.jsonl)
- [机器可读汇总](results/holdout60/summary.json)
- [完整配对和源码校验](results/holdout60/audit.json)
- [实验协议与实现范围](EXPERIMENT_PROTOCOL.md)
- [源压缩包路径与 SHA-256](inputs.json)

新增代码：`online.py` 保存/恢复 V4 状态并接入解析概率；`posterior.py` 采样与历史观测一致的世界；`planner.py` 完成前瞻、独立复核、共享候选及动态覆盖替代；`benchmark.py` 串行配对计时。

原 V4 与概率原型在 `vendor_v4/`、`vendor_inference/`，均保持下载包原文。原 V4 的 20 项测试、概率原型的 16 项测试和新增 8 项测试通过；新增测试包括 18 个案例的逐动作等价检查。三条旧存档示例中有一条在本机出现行为差异，原因未定；新表使用同一本机运行环境的重新实测基线，见 `archive_replay_audit.json`。

复现（Python 3.10 以上，无第三方依赖）：

```sh
python3.13 -m unittest -v test_online test_planner
python3.13 benchmark.py --cases 10 --out results/reproduced60 --seed-base 108000000 --scenarios 8 --decisions 8 --decision-budget 5
python3.13 audit_results.py results/reproduced60 --require-complete
python3.13 make_report.py results/reproduced60
```

本机 Python 完整路径：`/opt/homebrew/Caskroom/miniforge/base/bin/python3.13`。

开发预试在 `results/pilot/` 与 `results/pilot_expanded/`；它们不计入最终均值。原讨论正文存档为 `reference_conversation.md`，其中旧成绩不能视为本轮复测。
