# 第三轮离线复现

在解压后的项目根目录运行。Python 3.10 以上，测试另需 pytest；策略、审计和分析只依赖标准库。所有命令均使用离线研究模拟器，不调用正式测试或官方演练接口。

包内包含全部 Python 测试文件，因此 `tests.test_strategy` 等被导入的辅助模块也在包内。主报告中的相关测试范围是以下命令；未声称运行其他任务的全部测试。

```sh
python -m pytest tests/test_q3_fresh.py tests/test_q3_fresh_experiment.py tests/test_q3_fresh_stepper.py tests/test_q3_movable_tail.py tests/test_q3_confirmation_bound.py tests/test_q3_belief.py tests/test_q3_branch.py tests/test_q3_fresh_belief.py tests/test_q3_fresh_rollout.py tests/test_q3_fresh_round2.py tests/test_q3_fresh_round2_validation.py tests/test_analyze_q3_fresh_round2.py tests/test_audit_q3_fresh_choices.py tests/test_q3_round3.py tests/test_q3_round3_journals.py tests/test_q3_round3_analysis.py tests/test_q3_round3_exploration.py -q
```

先审计已保存的结果，比重新运行带时间截止的规划器更适合核对报告数字：

```sh
python -m experiments.analyze_q3_fresh_round3 --batches reference_development stage1_g1 stage1_g12 --out reproduced-stage1
```

新运行必须指定一个尚不存在的输出目录，示例：

```sh
python -m experiments.run_q3_fresh_round3 --suite development --configs G12 --out reproduced-g12
```

每次规划保存 `branches-案例序号/策略/decision-序号.json.gz`，其中有公共历史、控制器任务栈、全部候选、名义世界、筛选与验收分组、分支费用和退出状态。基线与选优候选另含完整未来动作；未完成分支的费用为 null。风险见证单独编号，不计入名义均值或标准误。

`experiments.replay_q3_round3` 用保存的决策逐步重放，可加 `--risks` 对跨区域动作离线生成风险见证。它要求历史摘要、任务栈及动作反馈一致，不会通过坐标取整来掩盖差异。模拟器以浮点坐标精确值缓存同位置误差；Windows 与 Linux 数学函数可能相差一个浮点最小单位，随后产生不同的缓存键。因此，逐步重放应在原实验 Linux 环境运行；保存日志的计费及兼容性审计可跨平台运行。任何重放失败均应保留并报告，不能当作通过。

`PACKAGE_MANIFEST.json` 列出每个打包文件的 SHA-256。原始验证数据库、连接配置、对象存储凭据和本地交换脚本不在包内。开发参考 B/CR 来自旧轮次不可变轨迹，复用来源摘要保存在 `reference_development/results.json`；它们没有新增分支日志，也不计为本轮新运行或新留出测试。
