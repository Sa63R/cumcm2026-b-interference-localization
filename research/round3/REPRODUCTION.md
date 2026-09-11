# 第三轮复核与运行入口

本轮共同评估使用本地 Python 3.12.14、torch 2.9.1+cpu、numpy 2.2.6，三个独立单线程 CPU 进程；没有训练或调用官方接口。策略本身与评估器分离。历史运行环境、源文件、配置、权重和每局世界哈希均在各批 `manifest.json`。原始案例复跑属于复现，不能重新计为独立验证。

## 使用已验证的状态搜索

保留 `research/q3-r2-derived-silence` 的冻结提交 `760af8230a8e5bda6050663b19d871a9701c9829`。其配置为 `experiments/state_search_candidate_derived_silence_combined_v1.json`，入口为 `strategies.derived_silence_state_search:run_derived_silence_state_search`。不要通过其他旧README中的默认efficient入口误选版本。该策略只依赖Python标准库；本輪共同比较需要上述统一环境来执行RL参考。

在现有 `q3-r2-derived-silence` 工作树可验证入口，以下命令不发送模拟器请求：

```powershell
..\q3-deep-rl\.venv-win\Scripts\python.exe -B experiments/run_q3_practice.py --spec experiments/state_search_candidate_derived_silence_combined_v1.json --dry-run
```

真实接入应等待现有采集任务释放接口，并确认当前GUI确为问题3演练；该HTTP协议不能自行证明演练/正式模式。本轮没有新增官方兼容性成绩。旧入口有冻结的首版时间限制，后续实际演练需显式给出新的带时区 `--action-deadline`。不要为了通过入口检查而伪造GUI确认，也不要中断采集任务。

## 重新审核原始记录

在 `q3-round3` 工作树运行。输出必须使用新文件名，审核只读取归档，不重启策略、训练或模拟器：

```powershell
..\q3-deep-rl\.venv-win\Scripts\python.exe -B experiments/round3_posthoc_audit.py --batches results/round3/directed_localization/pilot --output results/reproduction/directed-audit.json --report results/reproduction/directed-audit.md --seed-cache results/round3/directed_localization/pilot/audit.json
```

所有比值使用同一个旧物理下界：起点到首个20米清除圆盘的最短距离，加圆盘间独立最短距离的开放Hamilton路线，除以5米/秒，再加每源5秒清除费。图上的子集DP精确；圆盘前后接触点可以不一致，位置又是免费已知，故它只是先知松弛。`mean(T/LB)`与`sum(T)/sum(LB)`分别统计；二者不能混写成在线最优近似比。

## 复跑冻结的配对试验

先恢复所需策略工作树并使HEAD等于recipe的提交；各角色的源码不能有未提交修改。已删除的本轮负结果工作树按 `closed_trials/README.md` 从Git bundle恢复。原状态搜索、增强版、RL参考均保留。推理权重为 `rl_reference/trial1-inference.pt`，SHA-256为 `9a1761c8474335ca373b66a5cf85120e1410bd5adbab378405df015a82458e69`；仅删除训练元数据，14个张量与验证权重完全相同。

在 `q3-round3` 中，以新目录复跑有向修复的32局五组试验：

```powershell
..\q3-deep-rl\.venv-win\Scripts\python.exe -B experiments/round3_runner.py prepare --trial directed_incumbent --stage pilot --recipes research/round3/recipes/directed_incumbent.json --output results/reproduction/incumbent-pilot
..\q3-deep-rl\.venv-win\Scripts\python.exe -B experiments/round3_runner.py run --output results/reproduction/incumbent-pilot --workers 3
```

完成后对这个新目录执行上述审核器。冻结runner拒绝改动的源码、模型、输入spec、公共物理代码或运行时；已有输出不能覆盖。后续独立确认和最终阶段还有真实整局门控及完整审计前置条件，不能用开发局的局部评分替代。

## RL比较边界

本轮参考是开始时已有48局同平台配对支持的trial1，验证源码 `4e2ba11`；`2953d0e`只加入共同评估辅助文件，没有改变策略源码。比较结论限于该冻结参考。

截至2026-09-12 03:01的本地并行研究快照，`q3-rl-autonomy/research/autonomy/range_r1_final_snapshot.json`（提交 `aef8a94b`）出现新的64局开发收益，但两个端点的提升区间均跨零，尚无独立确认。其他新RL模块只有训练或预检记录。本轮没有中途替换对照，也不能将不同平台、不同案例的均值直接排名；新的可靠RL版本应另行冻结后配对比较。
