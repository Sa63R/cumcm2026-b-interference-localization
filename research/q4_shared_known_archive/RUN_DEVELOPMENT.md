# R35 固定开发运行记录

运行提交 `3e6e3158ef34a187589789438874b2df9c43764b`。运行前核验 local/origin 一致、工作树 clean、70项 runtime 的 working/HEAD/freeze 字节一致、原49项base源码不变、两份计划与所有旧QA文件的原SHA一致；旧QA源码zip逐项符合原manifest。记录见 `postpublication-preflight.json`。没有重跑旧QA或打开预留独立集。

实际两批同时各2 workers，合计最多4。工作目录为本仓库，PowerShell命令如下。输出目录、首次审计及selection拒绝覆盖，不应重复执行已完成案例。

```powershell
$py = 'D:/jwt/2026数模国赛/cumcm2026-b-interference-localization/.venv-win/Scripts/python.exe'
$env:PYTHONPATH = 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $py -B -m experiments.run_q4_per_source run --plan research/q4_shared_known/plans/development.json --plan-sha256 54eab5652a265d93bfa210a68b184210ce8c1370e9aa527d658a0e3eaf255c8a --output results/q4_shared_known/development --workers 2 *> research/q4_shared_known/development-console.txt
& $py -B -m experiments.run_q4_per_source run --plan research/q4_shared_known/plans/development-stress.json --plan-sha256 53593063e0fffe7925545d6fb6c7310c8f55dd640b697e243c183841cbf4b64b --output results/q4_shared_known/development-stress --workers 2 *> research/q4_shared_known/development-stress-console.txt
& $py -B -m experiments.audit_q4_per_source --input results/q4_shared_known/development *> research/q4_shared_known/development-audit-console.txt
& $py -B -m experiments.audit_q4_per_source --input results/q4_shared_known/development-stress *> research/q4_shared_known/development-stress-audit-console.txt
& $py -B -m experiments.q4_shared_known_release select --development results/q4_shared_known/development --development-stress results/q4_shared_known/development-stress --output research/q4_shared_known/selection.json *> research/q4_shared_known/selection-console.txt
& $py -B -m research.q4_shared_known.analyze_development --output research/q4_shared_known/development-analysis.json *> research/q4_shared_known/development-analysis-console.txt
```

分析脚本从R34已归档的逐动作费用分析扩展，仅新增研究脚本，未改任何冻结代码。仅读结束记录的row、summary、action_history及真实共享事件，不读evaluation/隐藏源，不构造反事实回包。每条动作累计时钟和五项费用均与原row相符，分享动作另外检查零移动、真实回包、实际区间和直接费用。

首次审计完成后才执行冻结selector。退出码1若对应正式 `passed=false`，表示原门槛未过，不修改规则或补抽种子。500仅为本地继续投入门槛，用户后续官方演练约束见 `USER_STEERING.md`。
