# R29冻结开发执行记录

root明确放行的HEAD为`ff5106417559e69d9720ea66e42409aa1db577a1`。起跑前69项运行文件SHA与四个开发plan原字节全部匹配。只执行两个候选各70随机+49压力，共238局；独立计划不执行。随机两条并行各3worker，完成后压力两条并行各3worker，保留全部返回码、记录和首次审计。没有中途改源码、协议、计划或筛选规则。

环境：Windows，本树目录；Python为`../cumcm2026-b-interference-localization/.venv-win/Scripts/python.exe`，`PYTHONPATH=src`、`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`。下列`python`代表该解释器。

```text
python -B -m experiments.run_q4_per_source run --plan research/q4_observation_cover/plans/ring_28-development.json --plan-sha256 ef9e73305335a7d5121ea28bbafffe8f8579c22b75f730fbdeb04716df2b6cb9 --output results/q4_observation_cover/ring_28/development --workers 3
python -B -m experiments.run_q4_per_source run --plan research/q4_observation_cover/plans/ring_31-development.json --plan-sha256 c6a4cd75f14cdb0bd3d0fef6087c7fcc3000bd1f03973f1040982e099214151c --output results/q4_observation_cover/ring_31/development --workers 3
python -B -m experiments.run_q4_per_source run --plan research/q4_observation_cover/plans/ring_28-development-stress.json --plan-sha256 cf2ad62e5b15b1f909e2029d8289995fa9de73b419e292d3d1d2eb912b14b44b --output results/q4_observation_cover/ring_28/development-stress --workers 3
python -B -m experiments.run_q4_per_source run --plan research/q4_observation_cover/plans/ring_31-development-stress.json --plan-sha256 82c08626cca44f52fa5c6caa4526d8e6dc1115ce47366a83b594fcedb68d6e3f --output results/q4_observation_cover/ring_31/development-stress --workers 3
```

每条输出原样重定向至本研究目录`<config>-<split>-console.txt`，原目录拒覆盖。每批记录完整后执行以下首审，输出保存`<config>-<split>-audit-console.txt`及该结果目录`independent_audit.json`：

```text
python -B -m experiments.audit_q4_per_source --input results/q4_observation_cover/ring_28/development
python -B -m experiments.audit_q4_per_source --input results/q4_observation_cover/ring_31/development
python -B -m experiments.audit_q4_per_source --input results/q4_observation_cover/ring_28/development-stress
python -B -m experiments.audit_q4_per_source --input results/q4_observation_cover/ring_31/development-stress
```

四批记录及首审齐全后统一执行冻结选择程序；exit1可以表示未过开发门槛，不据此补抽或重跑：

```text
python -B -m experiments.q4_observation_cover_release select --ring-28-development results/q4_observation_cover/ring_28/development --ring-28-development-stress results/q4_observation_cover/ring_28/development-stress --ring-31-development results/q4_observation_cover/ring_31/development --ring-31-development-stress results/q4_observation_cover/ring_31/development-stress --output research/q4_observation_cover/selection.json
```

实际完整返回码、阶段指标和选择结果见最终RESULTS及原console。本说明列出完整已授权流程，不是独立验证release；即使开发通过也须root另行确认。
