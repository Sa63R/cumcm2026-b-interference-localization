# R29实现与冻结前检查

两个固定配置是`ring_28`（原点+9×999米内环+18×1920米外环）和`ring_31`（原点+10×999米内环+20×1900米外环）。几何选择依据、六个公开候选的完整证据及适用范围见GEOMETRY_PROTOCOL、GEOMETRY_RESULTS、PROTOCOL。本文件仅记录实现和冻结前检查，不是论文正文或性能筛选结果。

## 实现范围

`planning.observation_cover_route.observation_cover_route(config)` 返回固定Position元组、新覆盖证书摘要及路线元数据。角0起点，内圈逆时针，随后从最后内点同径外站开始顺时针走完外圈；顺序与公开几何阶段一致，不调用NN、2-opt或反馈选向。每进程首次构造均按固定预算执行完整四叉树覆盖认证，再重放每个叶证据和完整分区。任何未知、缺叶或失败都拒绝作为合格覆盖，不能只靠“passed”标签。

摘要绑定新站点SHA、完整叶集合SHA、场域1800/接收半径1000、距离余量1e−5米、方向余量1e−7米、max_depth16/max_cells200000。原实数凸性证明与保守浮点实现有明确区分，不声称形式化区间算术证明。指定径向向外边界模型的最少2个接收站只是该条件下的解析计数；不声称所有源和所有朝向都得到2个观测，也不保证可直接定位或清除。

`strategies.q4_observation_cover.Q4ObservationCover`仅覆盖构造函数。它继承R12的全部真实行动、定位、试清、顺路服务与调度方法，替换`self.points`、`report.coverage_points/total`和真正对应新站点的`directional_cover_certificate`。`q4_compact_profile`改为`observation_ring_28/31`，不会留下伪装旧compact_22证书的元数据。入口为`run_q4_observation_cover`，labels为`compact_ring_28/31`，config为`ring_28/31`，原200/60000规划、20000动作、6次active探测等预算不变。

返回证书和嵌套元数据均为深复制，避免调用者修改报告污染缓存。证书的`runtime_s`是首次生成成本，本次构造耗时另记`observation_cover_setup_runtime_s`，不能将缓存中的旧耗时冒充每局实际计算。

## 测试

- 生产、纯几何及继承组：**95通过，1.94秒**。其中31项新测试，64项R12/R8/scheduling回归。用公开完整压缩proof独立重放新实际点、核叶SHA/固定索引/闭式纯路长；同时检查错误分区、未知证书拒绝、无NN/2-opt、缓存隔离、完整560/620次脚本unknown测量顺序、真实R8清除/finally、16实际清除退出和非法预算。
- 独立几何/前缀审计组：**60通过**（theory agent完成并释放）。独立从k、半径及固定索引重建新坐标，重放新覆盖叶，强绑定49个原src与2个新增src，再检查原R12/R8/range/scheduling真实前缀。
- 单臂分层runner、双候选选择/release及批审计组：**85通过**（deep agent完成并释放）。使用纯行、元数据、合成封套和已有旧案例合同，不构造633场景。

生产/继承组准确命令：

```powershell
$env:PYTHONPATH='src'
& '../cumcm2026-b-interference-localization/.venv-win/Scripts/python.exe' -B -m pytest -q tests/test_observation_cover_route.py tests/test_q4_observation_cover.py tests/test_q4_joint_continuation.py tests/test_q4_clear_before_probe.py tests/test_q4_r2_scheduling.py
```

三组测试范围有目的地区分，不把数量之和称为未经核对的去重测试数。冻结准备只做必要字节/计划核查，不重复跑无关测试。

## 仅旧621001接口QA

最终实际仅对**已经打开的621001**各配置运行一次，共两条；没有运行621002。两条同caseSHA、均13/13全部清除，失败clear为0、退出成功，第一次`audit_full`均通过新整覆盖/物理/LB、51源文件契约、原R12/R8/range/scheduling检查。

|旧621001配置|T（秒）|T/N（秒/源）|T/LB|测量次数|失败clear|
|---|---:|---:|---:|---:|---:|
|ring_28|6703.847446|515.680573|3.025553|301|0|
|ring_31|6880.345644|529.257357|3.105209|323|0|

完整记录、首次审计、source.zip及输入SHA保留在`old-smoke/`，脚本`run_old_smoke.py`拒绝覆盖。不能据此挑候选或声称泛化收益；两个固定候选仍须完整执行冻结后的开发规模。

## 冻结范围

源文件字节核查明确区分：R12原src全部**49个文件**（44个.py、5个.gitkeep）不变；R29新增2个.py，src共51个文件。runtime哈希map另包含评估/审计、协议和两个spec。新目录与文件受仓库`.gitattributes`的`-text`保护，避免Windows换行改变证书/计划绑定。

8个plan是2配置×4splits，只按公开首抽N、固定家族及每层配额选择最早种子，完整保留1000种子的取舍轨迹/hash，未构造任何633案例。两配置同split选种子完全相同，四段不相交；开发每配置70+49，独立预留140+98。独立plan存在不代表已获运行许可，仍必须另有绑定完整开发证据的release。

`source-freeze.json`及`freeze-preflight.json`记录源码、协议/spec、八份plan哈希与必要字节检查。真正提交/推送和新场景放行由root在核验后执行。本阶段不提交、不运行新性能案例。
