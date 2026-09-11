# R11：可接收内核救援的预定开发与验证协议

四个固定状态搜索臂为22站 `compact_baseline`、R8 `compact_clear_before_probe`、已合格 R9 `compact_joint_probe` 与唯一新候选 `compact_visible_kernel`。新候选配置固定 `after_silence_once`，每次 resolver 至多一次实际内核救援，候选上限24、认证裕量1e−5 m、生成裕量2e−5 m、原主动预算6、A*预算200均不扫描参数。完整规格见 `development-specs.json`。主比较对象为R9，旧基线与R8保留背景数据。

所有场景沿原 `run_q4_round2.py` / `make_case` 生成方式；策略仅取得观测接口，退出后才允许评估器读取真值计算历史放松下界。不得从旧610前缀代理得分推断闭环用时，不使用官方采集日志拟合或补造反事实。

开发预定随机 **620001–620024（24局）**，压力 **620031–620044（14局）**。压力使用原 `make_case(seed,'stress')`，七个既有家族各2局；这是开发压力，不冒充独立验证。源码、独立审计器、协议、规格与工具全部提交且稳定后，才由root显式执行 `freeze`。冻结前不生成这些场景。

开发通过条件：候选与R9在两个完整配对集都全部清除；所有四臂的物理、覆盖与相应R8/R9/R11前缀审计通过；随机平均节省严格大于0，压力平均节省不小于0，两组各自 `P95(T候选)/P95(TR9) <= 1.05`。只有一个候选，不存在事后配置择优或5秒平手调换。所有合法光学失败记入真实费用；未完成或错误保留，惩罚360000秒，不能删掉失败局再计算均值。

仅在开发通过并写出固定 `selection.json` 后，root才可单独允许打开 **620101–620228随机128局**、**620301–620384压力84局**。压力每族12局。运行命令还要求显式 `--allow-independent`，不会在开发结束后自动运行，独立样本不参与更改策略或参数。

独立晋级条件保持与R9相同：候选与R9全部清除且全部审计通过；随机平均节省的10000次配对bootstrap 95%区间下端严格大于0，随机平均改善至少0.5%；压力平均不劣；两组P95比不超过1.05。bootstrap随机种子沿原比较器固定610941。P95比是两组各自分位数之比，另保留逐局比值，不能混用两种口径。没有通过时如实保留失败和退化。

每个阶段是一次明确的独立命令，已有目录、freeze、决策或选择文件均拒绝覆盖。执行收据保存全部记录SHA，部分执行失败也保留已产出的记录；重试不得重新抽样。普通源变化、控制文件变化、规格变化、案例/下界不配对、矩阵不完整、审计输入哈希变化均阻止评估。所有阶段必须能复算统计表与原记录一致。

最优已冻结RL臂由root使用R9固定适配器在**相同案例配置**上另行运行，输出各阶段的 `<stage>-rl`。R11状态选择器只做上述事前固定状态门槛，不能训练、更换或挑选RL权重；状态决策明确 `report_complete=false`，完整交付须待同场景RL比较及其身份审计补齐。RL结果不改变R11相对R9的预定门槛。

所有性能表包含T、历史LB、均T/均LB；单局T/LB另保留。历史LB使用终止后真值的放松问题，不表示未知源时必然可达。虚拟费用与程序CPU分开，不把本机并行负载解释为算法性能。仅本地合成研究，不接模拟器/数据库/SSH，不开启正式测试，不写论文正文。

## 显式执行步骤

以下命令只是复现说明；本协议准备阶段不执行数据命令。

```text
python experiments/run_q4_visible_kernel_study.py freeze
python experiments/run_q4_visible_kernel_study.py run --stage development --workers 3
python experiments/run_q4_visible_kernel_study.py run --stage development-stress --workers 3
```

每阶段完成后依次运行通用物理/覆盖审计、R8批审、R9批审和R11批审：

```text
python experiments/run_q4_round2.py --audit --output results/q4_visible_kernel/<stage>
python experiments/audit_q4_clear_before_probe_batch.py --input results/q4_visible_kernel/<stage>
python experiments/audit_q4_joint_visibility_prefix_batch.py --input results/q4_visible_kernel/<stage>
python experiments/audit_q4_visible_kernel_prefix_batch.py --input results/q4_visible_kernel/<stage>
python experiments/evaluate_q4_visible_kernel.py --phase development
```

只在root后续明确允许时，分别执行 `run --stage confirmation --allow-independent` 与 `run --stage stress --allow-independent`，两者仍各需完整审计，再执行 `evaluate --phase confirmation`。任何开发失败都不会产生可放行的独立选择。
