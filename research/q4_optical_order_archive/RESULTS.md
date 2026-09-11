# 光学首次命中顺序开发结果

结论：两候选均未通过事先登记的开发筛选，不进入独立验证，不替换通用compact_combo。全部38个新开发场景、152份轨迹真实清除完毕；通用审计152/152、光学顺序附加审计76/76通过。614101–614164与614201–614242预留场景未生成。

|集合|方法|平均T（秒）|平均历史LB（秒）|平均T/平均LB|
|---|---|---:|---:|---:|
|development|compact_baseline|7204.567716|2089.747405|3.447578|
|development|compact_combo|6760.557286|2089.747405|3.235107|
|development|compact_optical_uniform|6794.230081|2089.747405|3.251221|
|development|compact_optical_weighted|6798.515481|2089.747405|3.253271|
|development-stress|compact_baseline|7896.247673|1743.009610|4.530238|
|development-stress|compact_combo|7481.050173|1743.009610|4.292030|
|development-stress|compact_optical_uniform|7812.729554|1743.009610|4.482322|
|development-stress|compact_optical_weighted|7578.194463|1743.009610|4.347764|

|候选|集合|平均节省秒（负数为退化）|配对95%区间|赢/输|p95时间比|
|---|---|---:|---|---|---:|
|compact_optical_uniform|development|-33.672795|[-101.01838562500006, 0.0]|0/1|1.000000|
|compact_optical_uniform|development-stress|-331.679381|[-588.1739415714287, -98.03383164285722]|1/7|1.061968|
|compact_optical_weighted|development|-37.958195|[-113.87458549999997, 0.0]|0/1|1.000000|
|compact_optical_weighted|development-stress|-97.144290|[-390.3001560714283, 177.3644648571429]|5/4|1.043281|

p95时间比为两个集合各自95分位时间之比，不是逐局比值的95分位。统计区间是bootstrap近似。全部性能来自本地混合类型Q4合成场景，未使用官方演练或真实数据库作为反事实成绩。

本轮保持完整原光学网格，只重排顺序。有限位置求积先验不保证覆盖真实目标的概率质量；完整几何网格则始终覆盖真实C。这两个不同性质不可混同。实际失败使有限模型耗尽后恢复剩余原序，保住完备性却仍可能为此前远距离跳转付费。详见DEVELOPMENT_ANALYSIS.md。

58项有针对性的模型、执行、审计和回归检查通过。38局中随机集仅1局改变总时，不应把均值退化描述为每局普遍退化。源码冻结ec9ab75，结果决定文件development-decision.json。
