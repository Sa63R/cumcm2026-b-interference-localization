# R33实现与冻结合同

唯一入口为`strategies.q4_transit_budget:run_q4_transit_budget`，固定`config=incremental_60, max_expansions=200`。覆盖仍为R12原22站，定位、清除证书、R8投机清除、辅助可行域、range跳测和原early服务保持继承。本轮新增文件只在父调度已经选中并进入`_scan(q)`时接入服务；全49个原src文件逐字节保留。

候选是当时已知未清、未ready、未near、未blocked且未被原early/本轮新增服务用过的频道。canonical MEC半径在`(19.9,40]`，中心在当前到q线段的投影比例为`[.1,.9]`，经中心的绕路至多100米，原首次接近费用超过60秒，且已知频道不足16。原early存在候选时优先，不插入新增服务。新增每源一次、每局最多4次，按绕路、接近距离、频道稳定排序。

每个真实动作经`_check_budget`检查微秒不变式：已消费服务费用+拟动作上界+返回q移动上界+首测6秒−原p到q移动下界≤60秒。未来移动向上取整，原入边向下取整；clear按5秒上界，measure按5秒及实际换频计。R8失败clear和随后原同点measure需以`8+switch`秒共同预留，避免只执行半个动作组合。另留完整20频道扫描、返回及退出的动作/虚拟额度；真实全局期限仍由原路径核验。

这里的60秒只约束实际服务后回到q完成首测时相对原入边的局部费用。它不是服务总时长、整站扫描费差或整局四次最多损失240秒的保证。真实新反馈和清除会改变后续调度，整局收益必须实测。

服务正常结束或本地预算中断后，恢复所有resolver上下文，再执行父类原完整扫描。全局停止、请求拒绝和期限异常传播，不在finally里强发动作；仅父扫描正常返回后，父调度才能消费覆盖站。日志每个scan宏都记录，含零动作尝试与跳过。新增服务真实范围为`[resolver_start_action_count, service_end_action_count)`；返回覆盖扫描从`scan_start_action_count`开始。选择器逐真实accepted wire绑定范围，再与独审的`prefix.transit_service_actions`相等比较；零动作尝试及返回扫描不计入“至少5个实际触发案例”。

开发仅70随机+49困难，主指标各局等权mean(T/N)，两组均≤500且119全清全审、至少5局真实触发才具备进一步投资资格。最终460目标与独立晋级规则见[PROTOCOL.md](PROTOCOL.md)，不能以开发通过代替。失败保留360000秒惩罚，实际未知耗时保持null。独立release重算两批全部原始证据、source、plan、spec与审计，不只信selection中的passed字段。

旧621003、621013只做冻结前接口QA，每例只运行一次，原raw/source.zip及首次审计保存在[old-smoke](old-smoke)。最终审计收紧只重放相同raw另存final-contract审计，不补跑场景；原首次证据不覆盖。计划只执行公开生成器的首抽N和家族分层，尚未构造634案例。完整测试输出、最终源码映射、两开发plan SHA及隔离Git index字节验证见`final-contract-tests-console.txt`、`source-freeze.json`、`freeze-preflight.json`；此预检之后还需要root提交后逐Git HEAD核验。
