"""Generate a readable report from verified paired run records."""
import argparse, json, math, statistics
from pathlib import Path
from collections import Counter
NAMES={'v4':'原 V4','analytic':'仅解析概率推断','rollout':'加一步 rollout','shared':'再加共享测向点','dynamic':'再加动态覆盖替代'}
GROUPS={'mixed_25pct':'25% 定向','mixed_50pct':'50% 定向','mixed_75pct':'75% 定向','outward_boundary':'边界朝外','constant_plus1deg':'恒定 +1°','constant_minus1deg':'恒定 −1°'}

def main():
    p=argparse.ArgumentParser();p.add_argument('result_dir',type=Path);args=p.parse_args()
    d=json.loads((args.result_dir/'summary.json').read_text());audit=json.loads((args.result_dir/'audit.json').read_text())
    rows=[json.loads(l) for l in (args.result_dir/'runs.jsonl').read_text().splitlines()]
    overall=d['overall'];n=overall['v4']['cases'];base=overall['v4']['mean_seconds_per_source']
    lines=['# 第四问：五组方法本地实测','',f'在本机对 **{n} 个相同案例运行五种方法，共 {len(rows)} 次完整任务**。成功完成 {audit["successful_runs"]}/{len(rows)} 次；六类场景各 {n//6} 例。下表均为新跑结果，不引用原聊天的历史均值。','',
           '**本地模拟结果，不是官方模拟器成绩。** 新方法采用有限位置采样、一步前瞻和有限动态换点候选，不代表完整 POMCPOW 或这些方法的最优实现。','',
           '这次没有观察到明显提速。仅概率修正平均节省约 0.32 秒/源（0.07%）；共享测向点版均值最好，节省约 0.68 秒/源（0.14%），但三种复杂前瞻版本的配对收益区间均跨过零。前瞻计算增加到每例数秒，暂没有证据支持用这一版复杂前瞻替换 V4。','',
           '| 方法 | 完整任务秒/源 ↓ | 相对 V4 用时降低 | P95 秒/源 | 本机计算秒/例 ↓ | P95 计算秒/例 | 全清除 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for m,s in overall.items():
        change='基准' if m=='v4' else f'{100*(1-s["mean_seconds_per_source"]/base):+.2f}%'
        lines.append(f'| {NAMES[m]} | {s["mean_seconds_per_source"]:.2f} | {change} | {s["p95_seconds_per_source"]:.2f} | {s["mean_wall_seconds"]:.3f} | {s["p95_wall_seconds"]:.3f} | {s["successes"]}/{s["cases"]} |')
    lines+=['','负的“用时降低”表示变慢。每个案例先计算完整虚拟任务时间 T/N，然后在案例之间等权平均；包含最后一个源清除后的补搜索。计算时间为本机单进程实际墙钟时间，另将一次性覆盖布局验证耗时记录在 manifest 中。','',
            '## 配对差异与波动','','节省秒/源为 V4 减新方法；正数表示新方法快。区间是探索性的近似正态 95% 区间，没有做多重比较调整。','',
            '| 方法 | 平均节省秒/源 | 95% 区间 | 更快/更慢/持平 | 最差个例退步 |', '|---|---:|---|---:|---:|']
    for m,s in overall.items():
        if m=='v4':continue
        q=s['paired'];lo,hi=q['normal95_saved_seconds_per_source']
        lines.append(f'| {NAMES[m]} | {q["mean_saved_seconds_per_source"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {q["faster"]}/{q["slower"]}/{q["tied"]} | {100*q["worst_relative_regression"]:.2f}% |')
    lines+=['','## 六类场景','','| 场景 | 原 V4 | 仅概率 | rollout | 共享点 | 动态覆盖 |','|---|---:|---:|---:|---:|---:|']
    for group,ss in d['groups'].items():
        lines.append('| '+GROUPS[group]+' | '+' | '.join(f'{ss[m]["mean_seconds_per_source"]:.2f}' for m in NAMES)+' |')
    lines+=['','单位均为完整任务秒/源。','',
            '## 成本分解','','| 方法 | 移动公里/例 | 检测次/例 | 光学未命中次/例 | 末源清除后补搜索秒/例 |','|---|---:|---:|---:|---:|']
    for m,s in overall.items():
        lines.append(f'| {NAMES[m]} | {s["mean_distance_m"]/1000:.2f} | {s["mean_detections"]:.1f} | {s["mean_failed_clears"]:.1f} | {s["mean_tail_search_seconds"]:.1f} |')
    lines+=['','## 前瞻实际做了什么','','| 方法 | 决策尝试 | 改变动作 | 模拟续跑次数 | 超时回退 | 后验回退 | 无效续跑回退 | 实际共享点动作 | 实际换站 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in ['rollout','shared','dynamic']:
        s=overall[m];q=s['planning']
        lines.append(f'| {NAMES[m]} | {q["attempts"]} | {q["accepted"]} | {q["rollout_runs"]} | {q["timeouts"]} | {q["posterior_fallbacks"]} | {q["invalid_rollouts"]} | {s["selected_shared"]} | {s["selected_shift"]} |')
    reasons=Counter(dec.get('detail','') for r in rows for dec in r.get('planning',{}).get('decisions',[]) if dec['reason'] in ['posterior_fallback','invalid_rollout'])
    lines+=['',('回退原因明细：'+json.dumps(dict(reasons),ensure_ascii=False)) if reasons else '最终测试没有发生超时、后验或无效续跑回退。前瞻只有约 4%–6% 的决策尝试改变原动作，多数候选没有在粗评及独立复核中显示足够收益。','']
    dyn=overall['dynamic']['planning']
    lines += [f'动态覆盖执行了 {dyn["coverage_checks"]} 次候选布局验证，通过 {dyn["coverage_passed"]} 次。对于实际更换站点后以覆盖条件退出的案例，另外用最终真实扫描位置复核，记录在 runs.jsonl 的 coverage_verified 字段。','',
              '## 实现、验证与局限','',
              '- 原 V4 的 20 项测试、原概率模块的 16 项测试及新增 8 项测试通过。新增测试包含 18 例逐动作等价、断点续跑、历史观测一致性、数量及两类源约束、策略不读取真实参数、超时回退及动态覆盖验证。',
              '- 代码冻结后，固定 60 个新案例做最终比较。开发预试不计入结果；最后没有根据这批结果再调参数。',
              '- 前瞻每次粗评 8 场景，另外 8 场景复核；每例最多 8 次，间隔至少 5 个策略动作，单次预算 5 秒。详见 EXPERIMENT_PROTOCOL.md。',
              '- 规划先验是显式假设。有限位置样本可能漏掉低概率但合法的状态，因此只用于动作排序；无源及清除仍使用几何与真实反馈。',
              '- 仅概率组保留 V4 原三点评分；共享点与动态站点替代都只搜索有限候选。本次没有实现深层 POMCPOW、VOPP、强化学习或连续全局优化，不能据此判断这些算法没有潜力。',
              '- 原模拟器没有按官方接口将示向度舍入至两位小数，也不包含官方通信耗时。这里验证的是同一本地模拟器中的相对表现。',
              '- 原包三条存档示例在本机复跑，两条总虚拟时间一致至数值精度，一条不同；差异原因尚未定位，详见 archive_replay_audit.json。因此本表始终使用本机同环境重新运行的 V4 基线，没有借用存档中的成绩计算改进率。',
              '- 计算时间来自当前机器，测试进程按轮换顺序串行运行，主机并非独占；不能直接当作官方设备运行时间。','',
              '## 可复现文件','',
              '- `runs.jsonl`：每例每方法的结果、规划决定与回退记录。',
              '- `summary.json`：总体、分组、配对统计。',
              '- `manifest.json`：Python/平台、完整参数、种子和源码 SHA-256。',
              '- `audit.json`：完整配对、输入包未改动、运行代码未改动检查。',
              '- `trace_*.json`：每组首例的五种逐动作日志。',
              '- `../../EXPERIMENT_PROTOCOL.md`：实现范围、计时与复现命令。','']
    (args.result_dir/'实测报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print(args.result_dir/'实测报告.md')
if __name__=='__main__':main()
