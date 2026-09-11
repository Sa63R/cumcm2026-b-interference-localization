# 可逐观测中断的扫描 option：可实施设计，尚无性能结论

本次只读代码与两篇原始论文，没有修改生产代码、训练、测试或传输。
建议作为独立动作表示试验保留，先做接口与计费一致性，再决定是否训练。
保留当前宏 PPO 推荐版本及当前最佳状态搜索作外部冻结对照；本设计不能
解释为已经超过它们。

## 证据与适用边界

`q4-rl-scan-bundles/research/q4_rl/SCAN_BUNDLES.md` 的两个预先指定 TRAIN
案例中，固定组把约 270 个上层决策降到 37，但平均实际时间从宏规则的
9140.49 秒增至 12185.68 秒；对应统一下界的 ratio-of-sums 为 4.31095 →
5.74716，R8 为 8825.89 秒 / 4.16257。主要损失来自移动与光学搜索，少量
切频节省无法抵消。这里同时删除了组内服务插入和已知频道/当前点补测，
所以两例只能证伪“少决策就快”，不能单独判定分组、训练或中断的贡献。

Options 可包含原子动作，也可以在执行中根据新状态终止；原论文的中断
改进结论依赖正确的价值比较，不能直接套到这里的近似神经网络和部分可观测
状态。[Sutton、Precup、Singh，1999，第 2、4 节](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)
Option-Critic 提供学习组内策略及终止条件的梯度方法，也观察到 option
趋向退化为一步，并用终止正则缓解；这不是本任务必须加正则的理由。
这里先不引入“停止罚分”或“坚持奖励”，以免改变实际费用目标。
[Bacon、Harb、Precup，AAAI 2017，第 3–4 节](https://arxiv.org/html/1609.05140v2)

## 最小控制接口：中断后立刻拥有全部选择

新增独立 `option_controller.py:run_q4_options`，使用独立 schema
`q4-interruptible-scan-options-v1`；不继承固定 bundle 的候选删减。

| 层次 | 具体行为 |
|---|---|
| 公共动作底座 | 完整保留 `Q4MicroSearch._build_candidates()` 的固定站/当前点/定位测量、安全清除和光学网格动作；另加入每个已发现未清源的有界 service 宏。先让无 option 的对照使用同一底座。 |
| 上层选择 μ | 从全部底座动作及 `start_scan_option(p)` 中选择。初版 p 为现有 22 站及实际当前位置去重；存在至少一个未知且合法未测频道才可启动。 |
| 组内选择 π | 只从锚点 p 当前仍未知且合法的真实单频道测量中选择下一项，使用学习的 masked channel head；新发现立即从该集合移除，已知频道仍可经 stop 后选择微动作补测，不预记整个组的信用。 |
| 终止选择 β | 每次测量回执进入 ledger 后，学习 `continue / stop`。stop 仅结束这个 option，立即让 μ 从全部底座动作选择，因而可以插入 service、去另一个位置或改选任意合法频道；不是结束任务。 |
| 强制边界 | 无合法组内动作、真实发现证书足够而扫描仅用于发现、预算/行政截止或真实全清时，按各自物理语义终止。强制终止不伪装成网络选择或训练标签。 |

每次 start 至少执行一个真实请求，stop 后同一状态最多重新选择一次并执行
下一真实请求，防止无费用的 stop/start 循环。每个真实回执后均允许 stop，
不能等到固定数量频道完成才放行。service 仍是已有有界 resolver，一旦选中
其内部暂不拆分；可自主选择微动作替代 service，但不能声称已学会 service
内部的必要性判断。

纯 Python 策略接口分别为 `choose_manager(public_features, candidates,
option_context)->index`、`choose_channel(public_features, legal_channels,
option_context)->index` 和 `terminate(public_features, option_context,
alternatives_summary)->bool`。Torch adapter 另存真实条件采样概率；这些接口
均只能访问当前公开数据，不能读候选动作执行后的未知反馈。

复用 `micro_controller.py:267` 的候选生成、`:409` 的单请求执行、`:184`
的清除证明和已启动完整网格状态；`controller.py:120` 的实际回执消费、
`:148` 的发现证书、`:162` 的真实全清门和 `:287` 的兜底保持原义。
service 必须使用明确的独立执行分支，正确保留 R8 拦截及已启动网格续行；
不能直接给 MicroCandidate 填 `service` 然后调用不支持它的单请求执行器。

新增公开 option context：是否活跃、锚点相对坐标、已执行请求数、剩余合法
频道数、下一测量费用、最新实际观测类别，以及可用 service/微动作摘要。
可以复用 G3 阴性兼容性分数，但需另作 G1/G3 消融；这些分数不是校准概率或
全清证明。有限特征未必充分表达历史，理论上不能因此宣称已得到 Markov 状态。
option 实例编号只用于日志索引，不作输入；不加入真值、seed、案例 ID 或下界。

## 先采用容易核对的 PPO 接法，准确界定“缩短信用链”

最小实现保留真实执行序列，增加 `active_option_before/after`、边界、候选
mask、条件决策及其 old log-prob/value。用扩展状态 `公开状态 + active option`
运行分解策略：有活跃 option 时先抽 β；continue 再抽组内频道；stop 则抽 μ，
若 μ 选择新 option 再抽首频道。一次实际执行之前所有**真正抽过的条件概率**
相乘，其 log-prob 相加；强制分支概率为 1。PPO 必须重新计算同一路径的联合
log-prob，不能只保存上层选点的概率而漏掉停止/频道决策，也不能把多步行为
当成一次旧 `TorchPolicy` 的单候选采样。新建薄的 option policy/update 适配，
不直接谎报给现有 `micro_train.rollout()` 的一对一 callback 校验。

这版是扩展状态上的分解 PPO：全局多候选选择次数可能减少，组内只有二元
停止和至多 20 个频道选择；但仍有逐观测的学习决策，完整终局回报仍然很长。
它主要检验决策分解、持续目标及价值表示，**不能仅凭上层次数下降宣称信用
链已经缩短**。若每次都 stop，或 continue 后仍完整重算相同全局候选，它
可能只是原策略的冗余重参数化，且推理更贵。

真正的上层 SMDP 学习可作为第二个明确版本：仅在 μ 的边界建立 manager
记录，option 内保留 β/π 的真实学习事件；manager 的价值/优势在 option
边界估计。它需要分层 actor/critic 与对应似然检查，不能仅把已有微动作日志
压缩后继续调用原 PPO。任何条件概率项只参与指定的一份 actor 损失，不能
同时在微动作损失和 option 损失里重复加入上层 log-prob。首轮先冻结下层，
检验 manager 的 SMDP 更新；再用新的真实 on-policy 批次训练下层，避免同时
改变 option 行为却把旧组轨迹当作固定转移核。

## 实际费用、终局与 GAE

物理账本唯一拥有费用。option j 从 τ_j 执行到 τ_{j+1}，其奖励是
`R_j = -sum(actual_cost_s[t]) / COST_UNIT_S`，不是组数、请求数或预估费用。
manager 视图通过账本区间引用聚合，不能将微动作费与 option 费再次相加。
移动、切频、扫描、光学、失败清除、完整兜底与 exit 尾费全部保留；失败的
终局调整仍仅为 `max(actual, failure_penalty) - actual`，只加一次。

始终 **gamma = 1**。先固定 **lambda = 1** 做动作/层次消融，critic 目标
仍为完整未折扣 MC 费用，终局 bootstrap=0。option 的 stop 不等于 episode
terminal；下一 option 的未来成本必须接上。行政中断不造 terminal=0：保留
真实前缀与整个 pending reservation，不用未完成批次更新。

若以后加入 lambda<1，上层可预先声明物理请求时钟：
`delta_j = R_j + V_H(next_boundary) - V_H(boundary)`，
`A_j = delta_j + lambda**k_j * A_{j+1}`，k_j 为实际请求数。它是另一个
duration-aware 估计器，**不等价于逐微动作 GAE**，因为内部 TD 项的权重不同。
不能沿用 `advantages.attach_actor_advantages()` 的“每行乘一次 lambda”却
让行从微动作变成 option，或在跨 option 时清空 trace。零费用元决策若单列，
物理时钟 k=0，不凭空衰减；首轮分解 PPO 合并同一实际动作前的条件选择可
避免这类歧义。metadata 必须绑定记录单位、GAE 时钟、lambda 和 MC critic。

## 权重与最小因果消融

原 G1 13/50 encoder、或原 G3 13/58 encoder 可以通过明确的初始化映射和
SHA 绑定复用；保留原特征块与语义。新 context/manager/termination heads
须新训，旧 critic 不包含 active option 不能直接视为校准价值。宏 10/16、
固定 bundle 10/18 与新头不兼容，不可按 tensor 形状强装。可保留旧宏 PPO
作为外部参照，或在新训练场景收集合法教师标签；不把旧轨迹中未执行的 stop
或插入服务结果编造成反事实反馈。新实验使用新 Adam/RNG，禁止跨 schema resume。

最小比较是共同底座三臂：A 为 flat 微动作+service；B 为同接口 option 但
每次回执强制结束；C 为学习 β 与频道 π 的可中断 option。A/B 查实现/状态
表示开销，B/C 查持续 option 的贡献。额外“只允许自然终止”的消融仍保留同一
底座，在 option 外可选全部微动作；它只用于区分中断贡献，不能取代 C。
原固定 bundle 的候选删减不同，只作历史失败参照。纯 SMDP manager 更新及
lambda<1 都是后续独立消融，不与第一轮同时改变。

所有臂使用同一合法合成训练划分、初始化来源、真实样本/CPU预算和独立确认
集。512 个旧 `decisions` 不能直接解释为 512 个新 option，否则会改变触发
兜底前的探索额度；新三臂需统一实际请求预算并分别记录上层、gate、频道和
物理请求次数。service 内请求也计入。外部冻结宏版本保留原配置并明确差异。
评价同时给当前最佳状态搜索/冻结宏的 T、同一下界 L、T/L、全清率、失败
清除、均值/尾部/配对不确定性及采样/推理/更新 CPU，不因 option 数少而换口径。

只有当 C 在配对真实 T 上有收益，且日志表明它在观测后改变了扫描/服务
选择，才值得扩大。长期总选 continue、全部退化一步、只减少网络调用而 T
不降、训练分支增多造成样本效率更差，都是预先接受的无收益结果。22 站本身
不是物理必需，但本试验仍保留原发现证书：stop 不撤销缺失频道的覆盖义务，
因此它不能单独解决 N<16 时的全域排除成本，也不证明某个未做动作永远不必要。
