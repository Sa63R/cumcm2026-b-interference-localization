# R12：在真实主动无信号后继续更新辅助可见区域

本轮以已晋级R9 `probe`为主比较对象。固定两个候选：旧R9 `probe_optical`原样复用；新`Q4JointContinuation`继承同一个`probe_optical`，只增加“真实主动测量返回no_signal后，再调用一次联合可见性helper”。不改变覆盖坐标/顺序、全局调度、R8同点先清后测条件、主动探测次数、光学网格生成器或独立几何helper。

## 状态与递归安全依据

canonical C仍只由真实正方位形成，供旧调度、扫描删测和R8谓词使用。当前resolver的aux是独立对象。入口aux由原R9规则建立；实际新bearing同时更新canonical与aux，原`aux_updates`只记录这类真实bearing。实际no_signal不伪装成bearing，不进入canonical；真实光学失败也不作为radio负观测。

一次accepted `active_localization/measure/no_signal`完全返回后，R9的`joint_radio`已记录该反馈，pending probe日志已完成。动态helper输入选当前持证aux；若入口未形成aux或已回退，输入当前canonical。传入该频道从真实历史开始至当前prefix的全部positive/negative测点；不依赖隐藏位置、半径、类型、朝向或假想观测。

归纳证明：假定当前输入外包A包含所有与此前真实观测相容的位置；新真实负测加入后，相容集合是旧集合的子集。原helper对A中每个整格只在所有未知类型/半径/朝向都被排除时删格，输出仍外包新相容集合。因此以前helper得到的合法aux可以作为下一次helper的输入；中间实际正方位按原保守方位带再相交，同样保真。每次证书的`canonical_vertices`字段在helper接口里仅表示本次输入，**不能据字段名误认它一定是策略canonical正区域**；新独审必须验证递归来源。普通浮点三角函数外扩仍不是区间算术形式证明；有向外roundoff时也不把输出严格嵌套于旧aux当数值定理。

仅在新helper `passed`、`outer_refined`且实际排除旧边界顶点时替换aux；fallback或无边界改善保留原aux，原canonical始终不变。异常记录后原样抛出，不把实现错误隐藏为成功。aux持证清除、末尾aux完整28米网格及其耗尽后的canonical完整网格均原样继承；真实服务/请求/总时限异常也继承，不额外发动作或重置主动探测预算。上下文在resolver退出后仍由R9恢复，下一resolver从它的真实入口历史重建。

## 事件与独审接口

新增`strategy_parameters.joint_visibility_continuation_log`，每个适用的真实主动miss一个事件，含：

- `id/resolver_id/channel`、触发真实动作索引、`after_actual_action_count`（该measure之后的prefix）；helper不发动作，end等于after。
- 该prefix的canonical、旧aux、本次输入、输入来源`auxiliary/canonical`、全部真实positive/negative测点和完整`helper_evidence`。
- `basis`：resolver入口prefix、上次已应用更新事件ID/prefix、之后真实bearing更新prefix列表。新审计按时间顺序交织旧`aux_updates`和此日志，逐次重建；不能直接将旧静态R9 prefix审计套在动态臂。
- 是否调用helper、状态、替换输出、每resolver/全局调用计数前后、CPU耗时。丢弃、提前到达清除证书、计算预算用尽都如实记录。

旧报告`joint_visibility_config`保持`probe_optical`，另有`joint_visibility_continuation_config=after_active_miss_optical`明确动态入口。审计对spec配置需按两个字段分别绑定，不将新config字符串误送旧静态审计。原R8额外真实清除前缀审计仍必须对新入口调用；独立共同物理、wire、覆盖、清除和LB审计继续保留。

## 确定性预算与实验边界

每个真实主动miss最多一次helper；每resolver调用不超过`max_active_probes`（默认6），整局不超过`20*max_active_probes`（默认120）；每次helper原有65536 constraint work上限保持。只有调用helper才增加计数，预算用尽保留先前aux。CPU时间只记录，不用耗时/并发决定算法开关；官方整体真实截止保护仍由原控制器负责。新方法没有新增物理探测请求，下一次真实动作仍由原resolver及其预算选择。

这些约束只保证合法性与有限工作，不保证aux每次都收紧、光学期望成本降低或Q4全局最优。凹排除区域取凸包可能恢复全部原边界；新探点也可能落在定向背面，进而改变后继任务时间。静态光学与动态刷新须作为两臂在新开发数据独立比较，不能利用新独立结果改网格、阈值或预算。

拟开发621001–621024随机、621031–621044压力；独立预留621101–621228随机、621301–621384压力。主参照R9 probe，背景baseline/R8保留；固定静态optical和动态候选，近5秒优先简单静态候选，门槛沿用事先规则。根任务另追加冻结Q4宏PPO512。**目前只实现与构造测试，未生成任何621场景；新prefix独审完成并冻结后才可进入开发。**
