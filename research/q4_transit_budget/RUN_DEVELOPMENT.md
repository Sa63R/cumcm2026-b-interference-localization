# R33待放行命令

仅在root完成提交、推送、核67项Git HEAD字节并明确放行后运行。本次准备没有执行下列新案例命令。工作目录为`D:/jwt/2026数模国赛/q4-r33-transit-budget`；两批并行时各3worker，全部119局与失败保留，不中途改源或计划。

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$py='D:/jwt/2026数模国赛/cumcm2026-b-interference-localization/.venv-win/Scripts/python.exe'
& $py -B -m experiments.run_q4_per_source run --plan research/q4_transit_budget/plans/development.json --plan-sha256 04956a4611327c0ea123ae7dd40079c95d281e5f50402fdb136a3d3ff5b70da0 --output results/q4_transit_budget/development --workers 3 *> research/q4_transit_budget/development-console.txt
& $py -B -m experiments.run_q4_per_source run --plan research/q4_transit_budget/plans/development-stress.json --plan-sha256 30b01aa02bba61a4b7a0e2fff1a7c63c43c6708478f17f16fa7125162764aac0 --output results/q4_transit_budget/development-stress --workers 3 *> research/q4_transit_budget/development-stress-console.txt
```

每批完整结束后执行首次审计，原结果目录拒覆盖。两批原档和首审全部齐全后统一调用selector；exit1表示投资门槛未过时保留该结果，不补抽。

```powershell
& $py -B -m experiments.audit_q4_per_source --input results/q4_transit_budget/development *> research/q4_transit_budget/development-audit-console.txt
& $py -B -m experiments.audit_q4_per_source --input results/q4_transit_budget/development-stress *> research/q4_transit_budget/development-stress-audit-console.txt
& $py -B -m experiments.q4_transit_budget_release select --development results/q4_transit_budget/development --development-stress results/q4_transit_budget/development-stress --output research/q4_transit_budget/selection.json *> research/q4_transit_budget/development-selection-console.txt
```

完整测试命令：

```text
python -B -m pytest -q tests/test_q4_transit_budget.py tests/test_audit_q4_transit_budget.py tests/test_q4_per_source.py tests/test_q4_transit_budget_release.py tests/test_q4_joint_continuation.py tests/test_q4_clear_before_probe.py tests/test_q4_r2_scheduling.py
```

独立6342001/6344001区间仅在协议中预留，开发通过也不得自动启动，须另获root确认及完整证据绑定的release。
