# 可靠 R12 的独立每源用时测量

本树是评估工具和结果记录树，不是新算法。唯一方法为 `compact_joint_continuation / after_active_miss_optical / max_expansions=200`；全部49个 `src/**` 文件逐字节保持 `81aa6e1a1231a2358cf098fbba3fdd0557b540ef`，其中44个Python文件和5个.gitkeep均核验。不能更改覆盖站、R8/R12规则、源调度、策略参数或模型；不运行RL或官方模拟器。

本次主问题是新口径下的可靠R12实际水平，不作多算法横比，不复制旧结果充当新样本。旧资格文件 `research/q4_joint_continuation/qualification.json` 的passed表示此前128随机+84压力独立实验通过原资格条件；它不表示本轮每源目标已达到，也不表示通过了新开发门槛。

## 固定样本与指标

主指标是各局等权 `mean(T_i/N_i)`，不得替换成 `sum(T_i)/sum(N_i)`。两者分别列出；保留各N分层、P95、费用分量、程序运行时间，以及原口径 `mean(T)/mean(LB)` 作为辅助。失败、退出不合格或未完整清除的局不删除，主指标将其T置为360000 s；实际原始费用另存。基础设施失败也保留固定seed行和同样罚值，不能标记为物理审计通过。

仅开启以下两组，分别报告，不合并不同压力构成掩盖差异：

| split | 固定扫描起点 | 分层 | 每层数 | 总数 |
|---|---:|---|---:|---:|
| confirmation | 6302001 | 源数10…16 | 20 | 140 |
| stress | 6304001 | 源数10…16 × 既有7压力家族 | 2 | 98 |

计划器在每个起点之后固定1000个整数seed中，只读取公开生成器的第一个源数随机抽取，按每层最早seed填满配额；不会构造场景、读取布局或反馈。保留完整1000行选取轨迹和哈希。`development=6300001`、`development-stress=6301001`仅保留元数据设计，**本次release不允许这两组执行**。源数和压力家族的实际完整记录还需独立复核；不能通过选择高N“凑”平均每源达标。

使用固定 bootstrap seed `630941`、10000次，confirmation按N、stress按N×家族在层内重采样；先按seed排序以消除并行完成顺序的影响。每压力层仅2局，区间仍是有限经验分布的不确定性估计，不代表任意压力场景保证。若观测均值低于460 s/source，只能如实说明这个固定分层样本的结果，必须同时报告全清情况、区间和分层值。

## 放行与证据

`runtime-81aa6e1a.json` 从原Git提交归档生成，固定全部源码、原selection和qualification的SHA。新 `q4-qualified-reference-release-v1` 绑定精确新plan文件SHA、评估工具源码、唯一spec及完整seed集合，放行依据明示为**既有合格R12的新口径测量**。不创建或修改原selection，不杜撰本轮development passed。

生成release前、run预检以及最终批审都会核对原资格、原selection和资格文件绑定的20项旧证据，核对全部策略源码与资格源码一致；这些哈希用于防止误配和确保可复现，不是对抗恶意修改者的数字签名。每次worker前后仍执行原源码冻结检查。新plan包含新增评估/审计工具及本协议和spec的哈希；归档source.zip，保留输入plan和release原始字节，不覆盖旧输出。

最终审计重算完整配额和汇总，验证每条记录的真实动作/费用、全覆盖、旧下界，再调用原R12辅助区域、R8试清、提前服务调度与range跳测审计。任何审计失败保留错误，不能只依据summary自报成功。策略侧只有原ObservationOnly客户端输入；真值仅在动作结束后的原物理/下界审计使用。

## 审核后执行

以下命令使用工作树根目录。`plan`只产生元数据；首次执行任何630场景必须由主代理审核源与测试后明确放行。

```powershell
$python = '../cumcm2026-b-interference-localization/.venv-win/Scripts/python.exe'
# 如果交接包已有plan则不要重复创建；plan命令拒绝覆盖。
& $python -B experiments/run_q4_per_source.py plan --split confirmation --spec research/q4_per_source_reference/spec.json --output research/q4_per_source_reference/confirmation-plan.json
# 审核通过后，读取完整plan原始字节的SHA；stress独立重复同样步骤。
$planSha = (Get-FileHash -Algorithm SHA256 research/q4_per_source_reference/confirmation-plan.json).Hash.ToLower()
& $python -B experiments/run_q4_per_source.py reference-release --plan research/q4_per_source_reference/confirmation-plan.json --plan-sha256 $planSha --authorize-reference --output research/q4_per_source_reference/confirmation-release.json
& $python -B experiments/run_q4_per_source.py run --plan research/q4_per_source_reference/confirmation-plan.json --plan-sha256 $planSha --release research/q4_per_source_reference/confirmation-release.json --workers 3 --output results/q4_per_source_reference/confirmation
& $python -B experiments/audit_q4_per_source.py --input results/q4_per_source_reference/confirmation
```

新工具构造/篡改测试使用合成封装与此前已打开的少量621构造，禁止调用任何630实际场景或网络。冻结就绪不等于性能验证完成。
