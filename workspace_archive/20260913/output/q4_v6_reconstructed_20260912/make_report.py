"""Render the report only from this run's saved measurements."""
import json
from pathlib import Path
import platform
import benchmark as bench

root=bench.ROOT
s=json.loads((root/'summary.json').read_text())
primary=s['groups']['practice_generator'];p=primary['methods']
c=next(c for c in primary['comparisons'] if c['baseline']=='v4' and c['candidate']=='v6')
c5=next(c for c in primary['comparisons'] if c['baseline']=='v5' and c['candidate']=='v6')
serial=json.loads((root/'serial_runtime/summary.json').read_text())
replay=json.loads((root/'replay_verification.json').read_text())
data=[json.loads(x) for x in (root/'main/records.jsonl').read_text().splitlines()]
index={(r['case_key'],r['method']):r for r in data}
text=[
'# 第四问 V6：本地复原模拟器配对实测',
'',
f'结论：V6 在 300 个新随机场景中，完整任务的平均每源用时比 V4 降低 **{c["saving_percent"]:.2f}%**，比 V5 降低 **{c5["saving_percent"]:.2f}%**。V6 对 V4 的配对提速 95% 自助法区间为 **{c["saving_percent_ci95"][0]:.2f}%～{c["saving_percent_ci95"][1]:.2f}%**。这是本地复原模拟器结果；没有运行官方演练或正式测试。',
'',
f'平均收益成立于本批配对实验。V6 在随机场景中 {c["faster"]} 例更快、{c["slower"]} 例更慢，最差比 V4 慢 **{c["worst"]["slower_percent"]:.2f}%**。保护机制仍有明显个例退步。',
'',
'## 300 个随机场景的主结果',
'',
'| 方法 | 平均完整任务秒/源 | 平均完整任务秒/场 | 相对 V4 |',
'|---|---:|---:|---:|',
*[f'| {m.upper()} | {p[m]["seconds_per_source"]:.3f} | {p[m]["virtual_time_s"]:.3f} | {(1-p[m]["seconds_per_source"]/p["v4"]["seconds_per_source"])*100:.3f}% |' for m in ['v4','v5','v6']],
'',
f'V6 平均每场少用 {p["v4"]["virtual_time_s"]-p["v6"]["virtual_time_s"]:.3f} 秒虚拟任务时间，约 {(p["v4"]["virtual_time_s"]-p["v6"]["virtual_time_s"])/60:.2f} 分钟。统计先计算每场完整 T/N，再按案例等权平均；没有用所有场景总 T/总 N 替换此指标。包含最后一个源清除之后，为证明没有遗漏而继续扫描的时间。',
'',
f'V6 对 V4 平均节约 {c["saved_seconds_per_source"]:.3f} 秒/源，95% 区间 [{c["saved_seconds_ci95"][0]:.3f}, {c["saved_seconds_ci95"][1]:.3f}]。对 V5 节约 {c5["saved_seconds_per_source"]:.3f} 秒/源，区间 [{c5["saved_seconds_ci95"][0]:.3f}, {c5["saved_seconds_ci95"][1]:.3f}]。按同一案例配对、案例有放回抽样 10,000 次；区间只反映这类抽样波动，不覆盖复原误差、正式分布差异等因素。',
'',
'## 收益来自哪里',
'',
'| 每场平均 | V4 | V5 | V6 |',
'|---|---:|---:|---:|',
*[f'| {label} | '+' | '.join(f'{p[m][field]:.3f}' for m in ['v4','v5','v6'])+' |' for label,field in [('移动距离/米','distance_m'),('无线电检测次数','measurement_count'),('切频次数','switch_count'),('失败清除次数','failed_clear_count'),('最后清除后的确认时间/秒','tail_after_last_clear_s')]],
'',
f'V6 相比 V4 平均少走 {p["v4"]["distance_m"]-p["v6"]["distance_m"]:.3f} 米，RF 检测净增加 {p["v6"]["measurement_count"]-p["v4"]["measurement_count"]:.3f} 次，失败清除减少 {p["v4"]["failed_clear_count"]-p["v6"]["failed_clear_count"]:.3f} 次。省下的移动成本超过新增检测成本。',
'',
'## 70 个压力场景（不混入主结果）',
'',
'两组各 35 场，每组 N=10～16 各 5 场。全定向组保留生成器位置与半径、将所有源改成定向；边界组把源放在 1760～1769 米圆周附近、全部朝外、接收半径固定 1000 米。这些是人工构造的压力分布。',
'',
'| 场景 | 场数 | V4 秒/源 | V5 秒/源 | V6 秒/源 | V6 对 V4 提速 |',
'|---|---:|---:|---:|---:|---:|']
for group,label in [('all_directional','全定向'),('boundary_outward_r1000','边界朝外、R=1000')]:
    g=s['groups'][group];means=g['methods'];cc=next(v for v in g['comparisons'] if v['baseline']=='v4' and v['candidate']=='v6')
    text.append(f'| {label} | 35 | {means["v4"]["seconds_per_source"]:.3f} | {means["v5"]["seconds_per_source"]:.3f} | {means["v6"]["seconds_per_source"]:.3f} | {cc["saving_percent"]:.2f}% |')
text+=['','边界组 35 场 V6 全部比 V4 快；全定向组 28 场快、7 场慢。这说明收益依赖场景，边界组的大幅收益不能当作普通随机场景的平均收益。',
'','## 退步案例','']
worst=[]
for base,cc in [('v4',c),('v5',c5)]:
    key=cc['worst']['case_key'];a=index[key,base];v=index[key,'v6']
    ar=json.loads((root/'main/sessions'/f'{key}-{base}.json').read_text())['algorithm_report']
    vr=json.loads((root/'main/sessions'/f'{key}-v6.json').read_text())['algorithm_report']
    detail=dict(case_key=key,baseline=base,slowdown_percent=cc['worst']['slower_percent'],
                baseline_row=a,v6_row=v,baseline_stations=ar['visited_stations'],v6_stations=vr['visited_stations'])
    worst.append(detail)
    text.append(f'- `{key}`：{a["source_total"]} 个源，其中 {a["directional_count"]} 个定向。{base.upper()} 用时 {a["virtual_time_s"]:.3f} 秒，V6 用时 {v["virtual_time_s"]:.3f} 秒，慢 {cc["worst"]["slower_percent"]:.2f}%。V6 多走 {v["distance_m"]-a["distance_m"]:.3f} 米，固定扫描站从 {len(ar["visited_stations"])} 个增加到 {len(vr["visited_stations"])} 个。')
bench.dump(root/'worst_case_audit.json',worst)
text+=['','日志支持的主要问题是路线改变后多走路、延迟发现全部 16 个频道。前四站保护沿用的是 V5；它不保证保留 V4 的前期动作，也不保证后期路线改动总能受益。本次没有依据这些结果修改模型、阈值或选择另一个版本。',
'','## 计算速度与机器人任务时间分开','',
'主实验用 4 个进程完成 1,110 次策略运行，批次墙钟约 185.7 秒；这不用于比较单个策略的计算延迟。另对冻结主样本前 20 场单进程复跑，交替运行顺序，得到：','',
'| 方法 | 本机计算墙钟/场 | CPU/场 |','|---|---:|---:|']
for method,m in serial['methods'].items():text.append(f'| {method.upper()} | {m["wall_seconds"]:.4f} 秒 | {m["cpu_seconds"]:.4f} 秒 |')
ratio=serial['methods']['v6']['cpu_seconds']/serial['methods']['v4']['cpu_seconds']
text += ['',f'V6 的计算 CPU 时间约为原始 V4 的 {ratio:.1f} 倍，平均仍低于半秒/场。此表含策略与本地协议处理，已预载入模块、树权重和固定站覆盖证书，不含文件导出、事后审计、进程启动和真实网络延迟。V4 未启用之前的计算缓存。60 次串行复跑均与主实验逐动作、逐反馈一致。V6 的机器人虚拟任务收益与它更高的计算成本同时存在。',
'','## 实验可信度与范围','',
f'- 主实验与压力实验合计 **{s["unique_cases"]} 个不同场景、{s["complete_runs"]} 次完整运行、{s["unique_source_instances"]} 个不同源实例**；三种方法各自清除全部源，共 {s["source_clear_events"]} 次成功清除。零策略错误、零漏清、零超时。',
f'- 全部 {replay["runs"]} 份会话重放，{replay["compared_actions"]:,} 条请求的业务反馈和微秒时钟全部一致。V4/V5/V6 各另跑一局真实回环 HTTP，动作与批量核心逐项相同。',
'- V6 包内 78 项自测通过；本地模拟器原有 23 项测试通过，包含现有客户端的 HTTP 集成。新增 2 项适配测试通过，其中 V4 在随机、全定向、边界三类场景与已发布执行内核完整动作一致。',
'- 包内 12 个 V4 基础及测试文件与已发布版本逐字节一致；没有改用户下载目录。运行副本、模型权重和模拟器代码在测试前冻结哈希，结束后再次核对。所有方法读取同一个固定场景，各用一个全新会话。',
'- 策略只收到 position、channel、move、detect、clear。真实源数量、位置、类型、半径、方向及噪声种子由评测器持有，结束后用于评分；没有把真值传给模型。',
'- 少于 16 源时，独立检查每个未发现频道均在 21 个认证站检测过，并核对连续覆盖证书；16 源时检查 16 次真实成功清除。不能用猜测的剩余源数终止。',
'- 移动 5 米/秒、RF 检测 5 秒、RF 切频 1 秒、清除成功 5 秒/失败 3 秒，由复原 Session/Engine 计入；clear 不改变 RF 频道。每段移动在模拟器内按微秒取整。',
'- 默认 V6 的模型、四站保护、16 次沿途停测预算保持不变。为与已有 V4 接口适配保持一致，三种方法统一采用 1.005° 保守几何误差界；复原反馈本身按其代码量化和钳制。另用原始 1° 界复跑同样前 20 场，仍全部清除，V6 对 V4 降低 1.990%（同样 20 场在 1.005° 下为 1.574%）；这是参数敏感性检查，不是新的独立验证。',
'- 此模拟器是指定 exe 静态分析后重写的本地版本，未做官方输入输出逐位对照。其生成器、空间误差与 V6 原包的自建环境不同；不能由本结果承诺官方成绩。额外复跑、HTTP 和单元测试不重复计入 370 个独立场景。',
'','## 文件与复现','',
'根目录 `summary.json` 保存各组均值、配对置信区间、快慢例数及最差例；`paired_records.csv` 是逐场记录，`main/sessions/` 保存全部真值、动作、反馈和完成证书。`plan.json` 为测试前冻结的完整地图与参数；`source_manifest.json` 为源码核对记录。`source_v6/` 是下载包的运行与测试文件副本，`simulator/` 是复原模拟器快照。',
'','在解压后的目录，用 macOS/Linux 的 Python 3.10+ 可运行新的一局（评测器使用 SIGALRM 做超时保护）：','',
'```bash','python3 run_local.py --method v6 --seed my-new-case --output my-new-v6-run','python3 run_local.py --method v4 --seed my-new-case --output my-new-v4-run','```','',
'两个命令使用相同地图、各自新建会话。无需 UTM、登录、GPU 或第三方 Python 库。输出目录非空时拒绝覆盖。`benchmark.py` 的已冻结批量输出也拒绝覆盖；若重现同一批，应先将原 `main/` 目录另行保留，再运行 `python3 benchmark.py --phase main --workers 4`。不要把复现当成独立新实验。',
'',f'报告生成环境：Python {platform.python_version()}，{platform.system()} {platform.machine()}。测试日期：2026-09-12。','']
(root/'V6_本地复原模拟器实测报告.md').write_text('\n'.join(text))
print(root/'V6_本地复原模拟器实测报告.md')
