# R12 预登记开发与独立配对规则

当前合格参照为R9 `probe`；来源为独立R9研究的已冻结源码。本轮两个固定候选：原样复用R9 `probe_optical`，以及继承它的`after_active_miss_optical`动态辅助区刷新。后者只在真实主动测量no_signal后，用完整真实P/N和当前已证明外包继续调用旧16×16联合可见性helper。半径/朝向消元、数值余量、网格、探点族、R8条件、清除证书和全局调度均不改。每resolver最多6次、整局最多120次额外helper，每次原有65536工作量上限；CPU耗时只记录，不作策略门控。全部变更见MODEL.md。

固定五个状态策略臂见development-specs.json：原22站baseline、R8、主参照R9 probe、静态optical、动态continuation。全部最大20000动作、6次主动探测、max_expansions200，失败惩罚360000秒。合法光学失败付完整成本，不当作整局失败。策略只得ObservationOnlyClient；真实源只在策略终止后用于物理核验和历史共同下界。

开发：随机621001–621024共24局；压力621031–621044共14局，旧make_case(seed,"pilot"/"stress")原规则。每臂严格相同完整环境配置及case SHA，不能仅用同名seed跨生成器。独立预留：随机621101–621228共128局，压力621301–621384共84局；开发选择之前绝不生成。任何开放的独立结果不得用于设置本轮参数或选择候选。

进入开发前冻结：run_q4_round2.hashes()全部源与audit、本协议SHA、spec文件SHA及完整spec、新evaluator自身SHA、固定RL源码/权重/adapter身份。每组保存manifest/freeze/source.zip、全部原始轨迹、summary。执行复用原run_q4_round2，不引入新环境生成逻辑。以root审核完成的development-freeze.json作为来源，源码与独立prefix审计完整后才运行；最多3个本地worker，不SSH/GPU/官方模拟器/数据库。

开发必要条件：候选与R9全部清除，物理/覆盖/原R8/静态R9或动态递归前缀审计全部通过；随机平均节省>0、压力平均节省≥0，两组各自P95(T候选)/P95(TR9)≤1.05。两候选均通过时选随机平均时间较小者；两者差严格小于5秒时选更简单的静态optical。失败和退步全部保留，不能改用另一分区或比值挑胜者。相同规则复用comparison()，按每局有放回Bootstrap10000次，seed610941。

开发选定后才冻结唯一候选，与baseline/R8/R9同开独立集。通用晋级要求随机平均节省Bootstrap95%区间下端>0且平均改善≥0.5%，压力均值不劣，两组各自P95比≤1.05，全清与全部审计通过。显著但低于门槛的改善如实保留，不把它硬算作最优或无效。独立验证不更改选定源码、参数、审计或协议。

每轮同时追加当前固定最佳已评估Q4学习臂宏PPO512。依赖读取兄弟树`../q4-r9-joint-visibility/research/q4_joint_visibility/rl_reference/`：source.zip SHA86580efae901756eccaeeb48edb6045ca2a9bb8ac9b436ca60cf17fa9179229d，checkpoint SHA3e830b070d5247e3573b3e59b7dfc32d54da101c5aab05cd4effb0dec2b354e8。外层执行器及worker也从该R9树按SHA冻结，独立解释器加载zip，确定性CPU单数值线程、512学习决策并保留原兜底。本轮不训练或重选模型。外层从已退出的状态策略记录读取完整环境配置重建环境，仍仅将观测client交给网络；逐例case SHA、LB、费用及完整覆盖一致后，保存到本树对应`<stage>-rl`。RL补齐审计前不发布完整比较，但它不改变对R9的预定候选选择规则。

报告各臂真实虚拟T、共同LB、T/LB、全清率、P95、退步及CPU时间。均T/均LB与逐局T/LB均值分开；下界为退出后真值的放松必要成本，不称未知源条件下可达最优。动态prefix按实际动作序号重建递归外包，禁止把输入aux直接当canonical正区域。旧R8独立batch只审它自己的入口；静态R9与动态batch各自还须审其继承的R8真实前缀，不能遗漏新入口。
