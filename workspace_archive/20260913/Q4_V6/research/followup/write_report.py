from pathlib import Path
import json,statistics
r=Path(__file__).parent;s=json.loads((r/'results_final/summary.json').read_text());a=s['all'];m=s['main'];x=s['additional_distributions'];u=s['unprotected_vs_v5']
head=f'''# 第四问 V6：实际改进、两轮验证与完整复现

## 最终结论

本轮找到一个能降低平均虚拟任务时间的版本，默认采用“前4个固定站保持V5＋离线任务代价评分＋沿途停测”。最终600个全新案例（480主例、120附加分布例）中，V5为{a['baseline_seconds_per_source']:.6f}秒/源，保护V6为{a['variant_seconds_per_source']:.6f}秒/源，降低{a['improvement_pct']:.6f}%。两版及未保护版本在所有600例、{s['source_total']}个源上均全部清除；1800次运行无错误。

主测试降低{m['improvement_pct']:.6f}%；附加分布基本持平，略慢{-x['improvement_pct']:.6f}%。主要收益集中在圆周附近朝外发射这一组（7.518697%）；普通混合240例仅降低0.372440%。因此不是所有分布都出现大幅提升，也不是已经达到SOTA或证明最优。

保护版600例中，{a['faster']}例更快、{a['slower']}例更慢、{a['tied']}例持平；最差个例比V5慢{a['worst_relative_slowdown_pct']:.4f}%。全部案例每源配对节约均值为{a['mean_paired_saving_seconds']:.6f}秒，按场景组分层的3000次自助法95%区间为[{a['stratified_bootstrap_95_CI'][0]:.6f}, {a['stratified_bootstrap_95_CI'][1]:.6f}]秒。

未保护版本的均值更低（{u['variant_seconds_per_source']:.6f}秒/源，比V5快{u['improvement_pct']:.6f}%），但最差个例慢{u['worst_relative_slowdown_pct']:.4f}%。保护默认牺牲约{a['variant_seconds_per_source']-u['variant_seconds_per_source']:.6f}秒/源的本批平均时间以减轻极端退步；它不是本批均值最低者，更不是逐例支配未保护版本。两者均随包提供，不隐去这个权衡。

## 冻结后的最终同案例对照

|集合|案例数|V5秒/源|保护V6秒/源|平均时间降低|
|---|---:|---:|---:|---:|
'''
for label,key in [('主测试','main'),('附加分布','additional_distributions'),('整体','all')]:
 t=s[key];head+=f"|{label}|{t['cases']}|{t['baseline_seconds_per_source']:.4f}|{t['variant_seconds_per_source']:.4f}|{t['improvement_pct']:.4f}%|\n"
head+='\n时间先按每个案例完整T/N计算，再取平均。包含最后一个源清除后确认没有遗漏的搜索，不以清除最后一个源的隐藏时刻提前结束。\n\n### 场景分组\n\n|场景|例数|V5秒/源|保护V6秒/源|降低|\n|---|---:|---:|---:|---:|\n'
names={'mixed25':'约25%定向','mixed50':'约50%定向','mixed75':'约75%定向','boundary':'圆周附近朝外、R=1000','plus':'固定+1°误差','minus':'固定−1°误差','cluster':'聚簇','all_radius_1000':'全部R=1000','smooth_error':'平滑空间误差'}
for k in ['mixed25','mixed50','mixed75','boundary','plus','minus','cluster','all_radius_1000','smooth_error']:
 t=s['by_scenario'][k];head+=f"|{names[k]}|{t['cases']}|{t['baseline_seconds_per_source']:.4f}|{t['variant_seconds_per_source']:.4f}|{t['improvement_pct']:.4f}%|\n"
head+='\n附加分布也存在于离线训练环境中，不能称为未见分布泛化。聚簇及平滑误差组稍慢，不予省略。\n\n### 时间到底省在哪里\n\n|每案例平均，全部600例|V5|保护V6|\n|---|---:|---:|\n'
for label,k in [('移动距离/米','distance_m'),('无线电检测次数','detections'),('频道切换次数','switches'),('光学未命中次数','failed_clears'),('最后清除后确认时间/秒','tail_after_last_clear'),('完整任务时间/秒','virtual_seconds')]:
 t=a['mean_metrics'][k];head+=f"|{label}|{t['v5']:.4f}|{t['candidate']:.4f}|\n"
head+='\n平均每例插入7.9567个停测点、在其中进行了10.8417次RF检测，但总RF净增加2.0467次：部分原来后续必须做的检测被替代。平均少走402.7558米，完整任务省82.2045秒。固定站点仍为21个，没有把学到的评分当成免检证书。\n'
head+='\n|每源时间分解/秒|V5|保护V6|\n|---|---:|---:|\n'
for label,k in [('移动','movement'),('无线电与切换','rf_and_switching'),('光学与清除','optical_and_clear')]:
 t=a['mean_time_components_per_source'];head+=f"|{label}|{t['v5'][k]:.6f}|{t['candidate'][k]:.6f}|\n"
head+='\n这次不是仅优化光学操作：RF成本略增加，但节省的移动和后续光学搜索超过新增检测成本。并行测试中的平均本地CPU时间0.9174→0.7711秒、墙钟1.1699→0.9910秒仅作运行记录，不是受控硬件速度基准，也不含官方接口耗时。\n'
head+='''
## 第一轮不能被隐藏

先冻结的未保护版本在780个新例上平均451.122907→445.976581秒/源，快1.140781%；主600例快1.500004%，附加180例慢0.171768%。最差个例慢150.3178%。这说明只报均值或者只报主测试会掩盖严重风险。

随后使用这些失败例诊断早期路线变化与16源数量上界的关系，原780例不再作为保护版的独立验证。最终保护参数只在既有90例开发集合上对照，并用另一批600例测试。第一轮数据和原冻结代码完整留在research/round1/results_v6，最终数据在results_final。不能把两批合并来声称“保护版在1380个独立例上快多少”。

第一轮预先指定的180例消融：V5 453.9792，只有沿途停测448.1861，只有任务重排455.9008，两者组合449.3694，使用较便宜V4局部控制451.0174。单独任务重排并没有获益，组合也不是每个子集都强于只有停测；本轮不声称所有模块独立显著有效。最终600例全量比较V5、未保护组合、保护组合，没有另做一套最终独立组件消融。

## 验证与证书

78项新旧单元测试全部通过，其中包括原V5恢复状态一致性、真实设备反馈隔离、学习特征不改变真实状态、无信号不冒充清除、失败清除报错、新增停点预算、保护前缀逐动作一致、阈值0复现首轮、生产重构复现开发版本、树模型推理确定性。所有600例检查完整清除、时间恒等式和100小时虚拟上限；这不等于证明官方网络和墙钟限制永不触发。

原V5顶层33个Python文件哈希核对无修改。新旧模型、代码与最终计划冻结时哈希均保留，评测开始/结束校验一致。连续覆盖重新验证接受4228个方格，源布局没有改成未经证明的新20点方案。

---

'''
method=(r/'report_method.md').read_text();method=method[method.index('## 研究范围'):]
(r/'V6_REPORT.md').write_text(head+method)
print('reportchars',len(head+method))
