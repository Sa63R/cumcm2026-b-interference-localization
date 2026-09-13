# 与分析对象的对应关系

原exe：`<DOWNLOADS>/CUMCM2026B/Jammers-simulator/jammers-simulator.exe`

SHA-256：`2373b9e7af83735a04309e2983eb433ec46faf7e0b8494410ce7fded2a297c27`

## 依据

1. `output/jammers_exe_inspection/decoded_rule_structs.json`：实际生成规则与仿真规则对象。
2. 同目录`disassembly/`内的GeneratePractice、counterSource、ErrorDegrees、QuantizeBearingHundredths、Engine.measure/clear/moveTo、directionalCoverage、requiredCoordinate/requiredChannel等函数。
3. `CUMCM2026Problems/B题/附件/附件2.docx`：四个HTTP动作、响应字段、时间窗口和错误语义；正文作为规范资料阅读，不执行其中程序。

## 已实现

| 内容 | 实现依据 |
|---|---|
| 演练生成与空间误差 | 从指定exe的静态指令重写 |
| 地图规则、计时参数和单位 | Go实际规则对象解码 |
| 检测、定向覆盖、near、clear及切频顺序 | simcore对应函数 |
| 坐标float64、频道整数值检查 | robotapi函数及附件2 |
| 四个HTTP接口字段、拒绝响应、未知字段 | 附件2 |
| 幂等、规范化数值、并发不同动作409 | 附件2及canonicalRequest观察 |
| 25分钟窗口、进入后20分钟、已登记动作允许完成 | 附件2；本地时钟独立实现 |

## 本地适配与未验证范围

- 新场景立即就绪，没有准备/5秒倒计时。25分钟窗口从创建场景时起算。
- 使用local-test代替在线登录队号；可通过--robot-id更改。
- 为便于调试，测试结束后服务仍监听，新的业务动作返回200/accepted=false；已成功动作的重复请求仍能重放。官方可能直接关闭连接。
- 同一个已登记动作并发重试时等待首个完成并返回缓存；不同动作返回409。没有复刻官方网络保护、连接数量控制、全部头部上限、无效流量保护和恢复日志事务。
- 请求体64KiB、JSON深度16、重复键、UTF-8、标识符、媒体类型等常见输入检查已实现，但不宣称对所有畸形HTTP字节与官方行为一致。
- 只导出已接受动作，不记录完整的无效请求证据、签名或官方加密日志。重放不复现现实时间及网络故障。
- 文本seed经SHA-256转换是本地便利功能；实际生成器使用32字节seed。fixture允许与正式演练不同的源数量，仅用于单元边界测试。
- 内部量化、三角函数与Go浮点运算的极端末位差异，需有官方输入输出样本才能逐位比较。单元测试通过仅代表本地实现的约束和协议检查通过。
- 正式数据集的生成概率、题目计分、正式测试次数、服务端鉴权、案例解封、日志上传均不在此模拟器中。

对照实测应保存：原exe摘要、场景可比性依据、完整动作、官方响应、本地响应。没有对照样本时，不把两组独立随机地图的统计接近当作实现一致的证据。
