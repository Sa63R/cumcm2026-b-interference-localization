# 推断静默四组实验诊断

仅分析明确指定的完整归档；已核对manifest、runner、源zip/逐文件、记录哈希、独立物理与前缀证书审核。未启动仿真或读取数据库。
旧LB是先知物理松弛，不是可达在线最优值。逐局均比与总量比使用相同审核合格局，分别列出。

|批次/策略|全清/局|失败清除|平均T|P95 T|最大T|平均CPU|均T/LB|总T/总LB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|derived_silence/pilot:baseline|16/16|0|3168.009223|3571.398289|3808.600885|1.125977|1.744681|1.732744|
|derived_silence/pilot:cap_only|16/16|0|3165.634223|3561.648289|3808.600885|1.111328|1.743481|1.731445|
|derived_silence/pilot:derived_silence|16/16|0|3155.884223|3531.648289|3796.600885|1.111328|1.737997|1.726112|
|derived_silence/pilot:relative_only|16/16|0|3158.259223|3541.398289|3796.600885|1.140625|1.739197|1.727411|

|批次/策略|相对静默省略|数量上限省略|旧静默省略|平均实际测量|平均换频秒|平均移动秒|平均检测秒|平均光学秒|平均移除秒|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|derived_silence/pilot:baseline|0|0|25|119.875000|109.625000|2394.946723|599.375000|38.437500|25.625000|
|derived_silence/pilot:cap_only|0|6|25|119.500000|109.125000|2394.946723|597.500000|38.437500|25.625000|
|derived_silence/pilot:derived_silence|26|6|25|117.875000|107.500000|2394.946723|589.375000|38.437500|25.625000|
|derived_silence/pilot:relative_only|26|0|25|118.250000|108.000000|2394.946723|591.250000|38.437500|25.625000|

clear不换接收频道；移动按每段微秒取整重建。省略数×5秒只是局部检测收费，不代替整局节省。

|批/案例|候选|实际整局节省秒|实际少测|仅删除有证负查询|物理子序列|首个不能延续的候选索引|
|---|---|---:|---:|---|---|---:|
|derived_silence/pilot/216101|relative_only|6.000000|1|True|True|无|
|derived_silence/pilot/216101|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216101|derived_silence|6.000000|1|True|True|无|
|derived_silence/pilot/216102|relative_only|0.000000|0|True|True|无|
|derived_silence/pilot/216102|cap_only|25.000000|4|True|True|无|
|derived_silence/pilot/216102|derived_silence|25.000000|4|True|True|无|
|derived_silence/pilot/216103|relative_only|12.000000|2|True|True|无|
|derived_silence/pilot/216103|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216103|derived_silence|12.000000|2|True|True|无|
|derived_silence/pilot/216104|relative_only|6.000000|1|True|True|无|
|derived_silence/pilot/216104|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216104|derived_silence|6.000000|1|True|True|无|
|derived_silence/pilot/216105|relative_only|36.000000|6|True|True|无|
|derived_silence/pilot/216105|cap_only|13.000000|2|True|True|无|
|derived_silence/pilot/216105|derived_silence|49.000000|8|True|True|无|
|derived_silence/pilot/216106|relative_only|18.000000|3|True|True|无|
|derived_silence/pilot/216106|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216106|derived_silence|18.000000|3|True|True|无|
|derived_silence/pilot/216107|relative_only|0.000000|0|True|True|无|
|derived_silence/pilot/216107|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216107|derived_silence|0.000000|0|True|True|无|
|derived_silence/pilot/216108|relative_only|0.000000|0|True|True|无|
|derived_silence/pilot/216108|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216108|derived_silence|0.000000|0|True|True|无|
|derived_silence/pilot/216109|relative_only|18.000000|3|True|True|无|
|derived_silence/pilot/216109|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216109|derived_silence|18.000000|3|True|True|无|
|derived_silence/pilot/216110|relative_only|0.000000|0|True|True|无|
|derived_silence/pilot/216110|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216110|derived_silence|0.000000|0|True|True|无|
|derived_silence/pilot/216111|relative_only|0.000000|0|True|True|无|
|derived_silence/pilot/216111|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216111|derived_silence|0.000000|0|True|True|无|
|derived_silence/pilot/216112|relative_only|6.000000|1|True|True|无|
|derived_silence/pilot/216112|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216112|derived_silence|6.000000|1|True|True|无|
|derived_silence/pilot/216113|relative_only|0.000000|0|True|True|无|
|derived_silence/pilot/216113|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216113|derived_silence|0.000000|0|True|True|无|
|derived_silence/pilot/216114|relative_only|18.000000|3|True|True|无|
|derived_silence/pilot/216114|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216114|derived_silence|18.000000|3|True|True|无|
|derived_silence/pilot/216115|relative_only|18.000000|3|True|True|无|
|derived_silence/pilot/216115|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216115|derived_silence|18.000000|3|True|True|无|
|derived_silence/pilot/216116|relative_only|18.000000|3|True|True|无|
|derived_silence/pilot/216116|cap_only|0.000000|0|True|True|无|
|derived_silence/pilot/216116|derived_silence|18.000000|3|True|True|无|

索引从0起。物理子序列比较不要求累计时间相同，但要求动作、坐标、频道及实际反馈一致。若仅证书对齐失败，记为无法按现有前缀日志证明，不能当作策略违法。
JSON保留首个动作分歧、最初不能延续的子序列/证书前缀、逐项删除见证、逐动作收费和整局分项差额；改变后的全局路线不得用局部查询潜力解释为必然收益。

全部输入与依赖SHA256见JSON input_sha256，所有输出以新文件保存。
