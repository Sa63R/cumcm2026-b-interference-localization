# 首版评估与计时编排

这里的脚本只负责编排已冻结的策略，不改变原策略、选择规则或测试场景。

`restore_sources.py` 是总交付包的离线恢复入口，应在包含 `source-identities.json`、`checksums.json` 与 `sources/source-history.bundle` 的交付包根目录运行；本源码副本用于保存实现。它拒绝覆盖目录，按原冻结 Git 身份恢复四算法、两个工具及原配置和权重，不执行模拟器。总包布局见最终结果中的 `PACKAGE_REPRODUCTION.md`。

- `run_registered_evaluations.py`：以固定 registry 运行全部扩展选择集，按预定规则写出 selected；随后用相同身份运行两个最终分区并核验档案。它检查真实解释器和安装包、保留失败与原案例、为独立子进程设置截止时间并保存逐任务记录。选择工具使用 `experiments/research_v1_selection.py` 的冻结副本。
- `serial_runtime_benchmark.py`：单个策略的计时工作进程，只用已经打开的 6000–6015，预热一局后重复两轮，逐局保存记录。指标是整局策略及模拟器接口耗时，排除 Python 启动，不能称为纯网络推理延迟，也不属于独立最终性能证据。
- `run_serial_benchmarks.py`：外层串行监督器，先核对所选身份和训练进程已经结束，再依次运行基线、状态搜索、强化学习和几何策略。它监督计时进程的硬截止，保留异常与不完整记录，不终止训练进程。其它评估或高负载程序也应在计时前结束。

生产编排使用同一 Linux 主机和实际登记的 Python 环境。各脚本拒绝禁用断言的解释器模式。Git HEAD、源码、配置、实际模型字节、规则和选择记录均须保持与 registry 一致；不能在首次打开选择集后改名单。完整流程与输入 schema 见 [FINAL_EVALUATION_PREFLIGHT.md](../state_search_v1/FINAL_EVALUATION_PREFLIGHT.md)。

```bash
python run_registered_evaluations.py --phase extended --tool /fixed/tool/research_v1_selection.py --registry /run/registry.json --run-root /run --workers 3
python run_registered_evaluations.py --phase final --tool /fixed/tool/research_v1_selection.py --registry /run/registry.json --selection /run/selected.json --run-root /run --workers 3
python run_serial_benchmarks.py --tool /fixed/tool/research_v1_selection.py --registry /run/registry.json --selection /run/selected.json --worker-script /fixed/serial_runtime_benchmark.py --run-root /run/serial --training-process-record /logs/rl-trial-process.json
```

以上是命令格式；应使用首版交付记录中的真实路径。`--training-process-record` 可重复传入，运行时至少提供一份，缺失或格式不完整会拒绝开始。计时监督器还检查研究目录下其它当前训练进程。输出拒绝覆盖既有计时证据；评估续跑只读取同身份目录中的原案例，不重新抽样或悄悄重试失败局。

准备阶段已完成三个入口的 CLI 检查与独立代码复核；计时监督器另通过正常退出、超时、启动失败、截止不启动和训练 PID 识别等九项构造检查。Windows 构造测试注入对子进程的清理方法，不冒充实际 Linux `killpg` 验证；Linux 部署与真实批次的执行记录另行保存。
