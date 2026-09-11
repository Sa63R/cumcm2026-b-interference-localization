# 第二轮复现入口

先读 GPT_PRO_HANDOFF.md，再读 IMPLEMENTATION_NOTES.md 和 RESULT_TABLES.md。analysis.json 和 all_runs.csv 是完整统计；results/q3_fresh_round2 下保留每次运行的结果与压缩轨迹，包含退步、失败清除和回退日志。

运行目录为项目根目录。Python≥3.10即可执行策略、证书验证、实验和分析；pytest用于测试。重新生成数值证书需要SciPy，但独立验证已保存的720个证书只用标准库。图表生成另需matplotlib。本轮所有复现命令均为离线研究，不使用正式或官方演练接口。

```text
python -m experiments.q3_confirmation_bound
python -m experiments.run_q3_fresh_round2 --suite development --configs B C R CR --out reproduce-development
python -m experiments.run_q3_fresh_round2 --suite development --configs CRA --out reproduce-attempt
python -m experiments.run_q3_fresh_round2 --suite nominal --configs B CR --out reproduce-nominal
python -m experiments.run_q3_fresh_round2 --suite pressure --configs B CR --out reproduce-pressure
python -m experiments.run_q3_fresh_round2 --cases research/q3_fresh_round2/validation_input.json --configs B CR --out reproduce-validation
```

每个--out必须是尚不存在的目录。不要同时启动超过两个CPU实验进程；无需GPU。默认名义种子980000起64个、压力种子970000起32个。开发与测试定义、全部参数和阈值已经在preregistered_protocol.json及FROZEN_CANDIDATE.json登记。本轮报告里的未接触测试在使用一次后就不能再用于下一轮“独立新测试”的宣传。

无需重新运行策略也可复核已保存的原始结果：

```text
python -m experiments.analyze_q3_fresh_round2 --batches development_bc development_r development_cr development_cra holdout_nominal holdout_pressure validation --out research/q3_fresh_round2/rechecked-analysis.json
python -m experiments.report_q3_fresh_round2 --analysis research/q3_fresh_round2/rechecked-analysis.json --out research/q3_fresh_round2/rechecked-tables.md
python -m pytest tests/test_q3_fresh.py tests/test_q3_fresh_experiment.py tests/test_q3_fresh_stepper.py tests/test_q3_movable_tail.py tests/test_q3_confirmation_bound.py tests/test_q3_belief.py tests/test_q3_branch.py tests/test_q3_fresh_belief.py tests/test_q3_fresh_rollout.py tests/test_q3_fresh_round2.py tests/test_q3_fresh_round2_validation.py tests/test_analyze_q3_fresh_round2.py tests/test_audit_q3_fresh_choices.py -q
```

计时截断由墙钟决定，因此不同机器、系统负载或并行任务可能改变个别采样/延续回退，完整策略输出不保证跨机器逐位相同。种子、代码哈希、已发生的决策和真实动作均已保留。独立审计直接验证当时的轨迹；离线改动作诊断能重放已选决策，从而分离计费正确性与重新搜索的随机/截止差异。

validation_input.json是只读数据库导出的固定兼容重建，包含复现所需的历史锚点；它不包含原数据库、不用于拟合，也不等于官方地图真值。VALIDATION_SELECTION.json记录436个合格组、排除12旧组、424个候选中哈希选取12新组及输入SHA256。未来重新对持续增长的数据库执行导出，可能选出另一批案例，不能冒称重现本轮固定样本。

不在分享包中放对象存储凭据、SSH配置或本机交换助手。文件交换与进程管理是运行设施，不是策略复现依赖。完整证据与策略代码已在分支 research/q3-from-scratch-v0-v2 保存。
