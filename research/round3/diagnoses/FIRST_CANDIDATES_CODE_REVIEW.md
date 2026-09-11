# 首批两个候选的独立只读审核

结论：截至下列源码冻结版本，未发现阻止旧例实现检查和随后预登记小批实验的致命问题。这是代码与数学范围审核，不是性能证明。本审核未调用策略、生成世界、读取 SQLite 或官方接口；作者已报告的人工测试通过结果不冒充本审核独立运行结果。

| 候选 | 冻结提交 |
| --- | --- |
| directed localization | `0d16cc38e66a036e7b6628e1aec1cf4ff04e7c06` |
| feedback envelope | `14946fd9f3b476a23ec872b909bbd48bee0aed47` |

有向定位模型的固定任务出口，使 `(mask,last)` 足以表示**本次冻结矩阵**的剩余费用。任意有向剩余路径去掉方向后是生成树，每条边不低于 `min(Dij,Dji)`，所以该无向包络的 MST 加上当前点真实出边的最小费用及剩余固定服务费，是可容许的有限模型下界；删除额外覆盖扫描罚项也只会降低下界。2-opt 重新计算整条有向路径，未误用对称反转公式。定位入边为原半径代理分数加首次测量 5 秒，清除 5 秒仅放在任务服务项一次，没有再加一次中心间行程。父调度先完成一次合法覆盖点重定位，子模型读取更新后的 remaining；假想区域仅在副本计算，执行仍交给父策略。

反馈包络将完整角度圆分成 720 个区间，每个区间用误差上限加半区间宽度构造外包。省略将来正测接收半径及正负距离比较约束只会扩大该外包。每个非空分支都有清除圆证明；near 单独在测点清除，no_signal 由整个原区域的保守接收距离条件排除。实际执行中父 `_perform` 先更新真实区域，之后和清除前各做全顶点校验；父 `_resolve` 的 near/MEC 分支抵达 `_clear`，子类用计价的分支清除点替换父目标。新的 phase 绕开父 MEC 安全圆二次投影，clear 不改变频道，首测换频一次、两段移动分别取微秒、成功清除 5 秒均与实际接口一致。当前继承链没有测量后插入 active-sharing；清除之后的正常共享位于该局部完成段之外。cap16 的成功清除即使抛终止异常，finally 仍检查真实末动作。

必须保留的限定：有向图的出口、未来观测及换频仍是代理，其 gap 不是整局物理 gap；额外几何计算消耗 CPU，并与父调度共享总展开预算。包络的完整最坏费用小于父 nominal 分数，是跨模型启发式筛选，不是期望改善或不退步证书。几何继承基准 binary64 及 padding，未宣称形式化区间算术；如实际区域数值结果未让父策略立即清除、状态变化或预算中断，须按日志认定未兑现/上界失效，不能把 selected 数量当成完成数量。disabled 均保留 fully enabled derived-silence 父版本。收益、触发频率和实际计算开销仍需冻结整局配对与独立物理/公共前缀审计。

源码 SHA-256：

| 文件（各候选树内） | SHA-256 |
| --- | --- |
| directed: `src/planning/directed_state_route.py` | `2856db86f4c46d3f9df047e1e755ddd741e20a5deddd938c690b0437e87be0c4` |
| directed: `src/planning/localization_transition.py` | `93fed39079a7958702f0057c227f6654300e8a20e2fad97db03b069008238bfd` |
| directed: `src/strategies/directed_localization_state_search.py` | `6c50687399de881ce757ad4ba95006d04d7683d3da76015935bc63b0f48c4cbb` |
| envelope: `src/planning/feedback_envelope.py` | `2a3f8028e0344e9b0abe53b9d66f8078ebd85e2a62df5d029dd31b671d3a0fa4` |
| envelope: `src/strategies/feedback_envelope_state_search.py` | `feffd35b0e55086b8da13cd37ad065b5a4b5c89b865eee28d486d8d1ab46f34e` |
