# R14 单候选配对协议——文本定稿，待全部源码与审计冻结放行

唯一候选 `strategies.q4_exact_frontier:run_q4_exact_frontier`，kwargs 为 `config="truncated_dag8",max_expansions=200`，共用最大20000动作/本次resolver最多6次主动探测。主参照为 R12 `after_active_miss_optical`；背景为 R9 `probe` 与原22站 compact baseline。同批另加此前冻结的Q4宏PPO512，通过独立源码/模型/worker SHA 绑定，不训练或重选网络。每臂同完整生成配置及case SHA，不能只凭相同seed声称配对。

固定四臂规格见 development-specs.json：`compact_baseline`、`compact_joint_probe`、主参照 `compact_joint_continuation`、唯一候选 `compact_exact_frontier`。开发与独立均保留这四个状态臂，不额外放入 R8 standalone 或 R9 optical 臂，不设多候选平手选择。独立另加固定 RL，因此完整性能表为五臂。

开发预留：623001–623024随机24局；623031–623044压力14局，沿用621的原pilot/stress生成口径。独立预留：623101–623228随机128局；623301–623384压力84局。全部生成仍用原 run_q4_round2.py / make_case；不增加生成器或自动连续运行入口。root完成公共入口、专项独立审计、固定spec、源归档、RL身份及本协议最终冻结后才显式放行开发。只有开发通过且写出 selection.json 后，root才可分别放行独立两组；预留种子不代表执行授权。

候选策略只接收 ObservationOnlyClient。必须检查固定链与源点来自决策前真实观测，原 A* 预算计数没有被 DP 污染，只有原exact=False/源≤8才计算，完整DP成本与同代理独立重算一致，严格改善才替换；其他情况保持原路线。再串联既有物理、覆盖、范围、调度、R8、R12辅助区和光学完整性审计。负向/失败案例均保留。

DP仅处理不超过22个有序覆盖站；固定状态数上限60000，合法最大8源/22站的宽松状态数界为52992。每次原A*仍受单次200及整局60000次扩展限制；新增DP的expanded/generated/runtime另记，不覆盖或扣减该A*总账。改善必须严格超过1e−9秒，平局保留原incumbent及首动作。浮点DAG闭合仅对冻结点任务代理成立，不是实数区间算术证书或原Q4最优性证明。

固定门槛：开发候选和主参照 R12 全清、审计全过，随机平均节省>0、压力平均节省≥0，两组各自 P95(T候选)/P95(TR12)≤1.05；单候选不通过就停止，不另调参数。独立晋级要求随机平均改善≥0.5%且配对Bootstrap95%节省区间下端>0，压力平均不劣、两组P95比≤1.05，全清全审计通过。Bootstrap10000次、固定seed610941。这里的P95比不是逐局倍率的P95。失败惩罚360000秒，合法失败光学正常计费，不把其直接视为整局失败；失败、部分执行和退步档案不能丢弃、补抽或只选成功局。

冻结入口仅 `evaluate_q4_exact_frontier.development_identity()`，其本身只读、不生成场景。root在所有 source/audit/tests 提交稳定后将返回值写入 development-freeze.json：包括 run_q4_round2.hashes() 源与全部 audit_q4 文件、evaluator自身、被调用 evaluate_q4_round2.py 依赖、本协议、spec字节及结构、RL源包/权重/reference/worker/外层执行器SHA。须核每个运行源与Git blob字节相同；新 research/results 使用仓库 .gitattributes 的 -text，不依赖本机 info 属性。

每个阶段保存 manifest/freeze/source.zip/summary/全部原始记录。选择器复算汇总，严格检查四臂×完整种子矩阵、重复或多余记录、逐案例SHA/共同LB、失败惩罚及原始row/spec；不能仅检查N个均值。开发压力若原runner要求selection参数，root只建立绑定开发freeze、这14个种子及四臂的明确 development-stress-release，不作为独立放行。独立两组的 manifest 必须绑定正式 selection SHA；selection同时绑定原development decision、freeze、完整四臂、未变的源码/RL/协议/依赖和128/84预留列表。

每组依次执行四个审计层：`independent_audit.json` 覆盖全部4N条；`joint_visibility_prefix_audit.json` 只覆盖R9的N条；`joint_continuation_prefix_audit.json` 覆盖R12的N条；`exact_frontier_audit.json` 覆盖候选N条且包含R8/R12继承核验。没有R8 standalone臂，不要求虚构的 clear_before_probe_audit.json。各专项必须提供唯一(strategy,seed)、完整 records/passed_records、input_sha256及逐条passed；输入哈希绑定原始记录与前置审计。独立随机要求512条通用+三个128条专项，压力要求336条通用+三个84条专项。

RL固定使用兄弟R9树 research/q4_joint_visibility/rl_reference：source.zip SHA86580efae901756eccaeeb48edb6045ca2a9bb8ac9b436ca60cf17fa9179229d，checkpoint SHA3e830b070d5247e3573b3e59b7dfc32d54da101c5aab05cd4effb0dec2b354e8，以及该树已固定 q4_frozen_rl_worker.py / run_q4_frozen_rl_compare.py。root负责追加 `<stage>-rl`；不训练或重选网络。完整选择报告必须等RL结束，核reference内源包/模型SHA、冻结spec、外层/worker身份、全部case与LB、原始RL档案与comparison汇总一致及每条独立审计。RL不改变R14相对R12的预定门槛。失败/未齐则保留证据、不得提前资格通过。

完成阶段后的入口为 `python experiments/evaluate_q4_exact_frontier.py --phase development`；独立两组另经root放行且完整后才调用 `--phase confirmation`。已有决定、选择和资格拒绝覆盖；没有通过开发时不写selection，不运行独立。本轮不连接官方模拟器、数据库或SSH，不修改已合格R9/R12源，不写论文正文。

每轮报告实际虚拟T、旧口径共同LB、均T/均LB、逐局T/LB、P95、失败率、退步案例及CPU代价。代理精确/代理节省与实际T/LB分开，不以旧27个相关前缀的代理改善替代完整实验。实际结果出来前不宣称收益或最优。
