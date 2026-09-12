# R34 固定开发运行记录

运行提交 `e5cb36b42b2570c0a4dfda32a551f3c04bf7da60`，发布后 local/remote、68 项运行文件的 working/HEAD/freeze 字节及两份计划均核验通过，见 `postpublication-preflight.json`。没有打开独立批次，没有追加横比或 RL。

工作目录为本仓库。PowerShell 环境和实际命令如下；输出目录及首次审计、selection 均拒绝覆盖，不应重复执行已完成案例。

```powershell
$py = 'D:/jwt/2026数模国赛/cumcm2026-b-interference-localization/.venv-win/Scripts/python.exe'
$env:PYTHONPATH = 'src'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $py -B -m experiments.run_q4_per_source run --plan research/q4_known_source/plans/development.json --plan-sha256 68be87421ce365cce07ea9eeb411f8bfba216d3c3083cbb9cc8739e74e098dac --output results/q4_known_source/development --workers 3 *> research/q4_known_source/development-console.txt
& $py -B -m experiments.run_q4_per_source run --plan research/q4_known_source/plans/development-stress.json --plan-sha256 5fe48eaf96748664261af97f4a7fce488eee0f27b34c13c68ba2fa4916e4fefc --output results/q4_known_source/development-stress --workers 3 *> research/q4_known_source/development-stress-console.txt
& $py -B -m experiments.audit_q4_per_source --input results/q4_known_source/development *> research/q4_known_source/development-audit-console.txt
& $py -B -m experiments.audit_q4_per_source --input results/q4_known_source/development-stress *> research/q4_known_source/development-stress-audit-console.txt
& $py -B -m experiments.q4_known_source_release select --development results/q4_known_source/development --development-stress results/q4_known_source/development-stress --output research/q4_known_source/selection.json *> research/q4_known_source/selection-console.txt
& $py -B -m research.q4_known_source.analyze_development --output research/q4_known_source/development-analysis.json *> research/q4_known_source/development-analysis-module-console.txt
```

实际两批各 3 workers 并行，合计最多 6；两批全部结束后启动两份首次完整审计。选择未过时原 selector 返回 1，这是冻结门槛失败，不作为程序异常修正或重跑。

费用分析仅读取已结束记录的 row、summary、真实 action_history；不读取 evaluation/源真值。分析程序第一次直接按文件路径调用时因缺少仓库模块搜索路径失败，没有生成分析结果；原错误 console 保留。改用上列 `-m research.q4_known_source.analyze_development` 调用成功，未重跑策略或修改运行源码。

冻结前最终合同测试 191 项通过，旧 621003/621013 各一次接口 QA 均全清、首次完整审计通过。冻结检查第一次只因保留 CRLF 的 console 被 `diff --check` 识别行尾空白而失败；按授权在 research/results 属性补 `whitespace=cr-at-eol` 后以独立临时 index 验证，原 console、首失败证据全部保留，未改 68 项运行源码或计划。最终 94 路径暂存字节和 working 字节一致。
