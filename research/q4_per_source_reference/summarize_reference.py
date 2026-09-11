"""Derive the reference report from completed, independently audited outputs.

This does not construct cases, run a policy, or alter frozen inputs.
"""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.run_q4_per_source import stratified_interval, validate_plan

OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = {
        'schema': 'q4-qualified-reference-measurement-report-v1',
        'method': 'compact_joint_continuation',
        'runtime_commit': '81aa6e1a1231a2358cf098fbba3fdd0557b540ef',
        'purpose': 'New count-stratified measurement of the unchanged historically qualified R12; not promotion of a new algorithm.',
        'primary_metric': 'Equal-run mean(T_i/N_i), failed T_i=360000 s',
        'target_s_per_source': 460,
        'bound_metric': 'mean(T)/mean(common_lower_bound_s); not mean individual ratios',
        'bound_definition': 'Original Q4 physical open-path relaxation plus 50*(20-N) empty-channel action bound for N<16; no empty-channel term at the public cap N=16.',
        'interval_method': 'Frozen seed 630941, 10000 empirical bootstrap replicates within N (confirmation) or N x family (stress); each displayed N interval uses that same rule restricted to that N.',
        'limitations': [
            'Local synthetic generator and fixed balanced source-count mixture, not official practice or a population-independent guarantee.',
            'Stress has only two cases per N x family stratum; percentile intervals describe this finite empirical distribution.',
            'N-specific intervals are descriptive, not simultaneous confidence bands or additional selection gates.',
            'Both complete batches are retained; no new development gate or candidate selection is inferred.',
            'Program runtime was measured locally with three case workers and is hardware/load dependent; virtual time is the objective.',
        ],
        'report_script_sha256': sha(Path(__file__)),
        'splits': {},
    }
    for split, count in [('confirmation', 140), ('stress', 98)]:
        folder = ROOT / 'results/q4_per_source_reference' / split
        summary = read(folder / 'summary.json')
        audit = read(folder / 'independent_audit.json')
        plan = read(folder / 'plan.json')
        validate_plan(plan)
        if not (summary['complete'] and summary['source_unchanged'] and
                summary['runs'] == count and not summary['infrastructure_errors'] and
                audit['all_passed'] and audit['records'] == audit['passed_records'] == count and
                not audit['errors']):
            raise ValueError(f'{split}: incomplete or unaudited evidence')
        for name, expected in audit['input_sha256'].items():
            if sha(folder / name) != expected:
                raise ValueError(f'{split}: audited input changed: {name}')
        if audit['plan_sha256'] != sha(folder / 'plan.json'):
            raise ValueError(f'{split}: audit plan mismatch')
        rows = summary['rows']
        item = {k: v for k, v in summary.items() if k != 'rows'}
        item['all_cleared_and_accepted_exit'] = all(r['all_cleared'] and r['completion_certified'] and r['accepted_exit'] for r in rows)
        item['observed_target_met'] = summary['mean_time_per_source_s'] < 460
        item['gap_above_target_s_per_source'] = summary['mean_time_per_source_s'] - 460
        item['required_reduction_fraction_from_current'] = 1 - 460 / summary['mean_time_per_source_s']
        item['cases_below_460'] = sum(r['penalized_time_s'] / r['source_total'] < 460 for r in rows)
        item['mean_measurement_count'] = statistics.mean(r['measurement_count'] for r in rows)
        item['mean_failed_clear_count'] = statistics.mean(r['failed_clear_count'] for r in rows)
        item['mean_action_count'] = statistics.mean(r['action_count'] for r in rows)
        item['max_time_per_source_s'] = max(r['penalized_time_s'] / r['source_total'] for r in rows)
        item['min_time_per_source_s'] = min(r['penalized_time_s'] / r['source_total'] for r in rows)
        for n, stratum in item['by_source_count'].items():
            subset = [r for r in rows if r['source_total'] == int(n)]
            stratum['stratified_bootstrap'] = stratified_interval(subset, split)
            stratum['cases_below_460'] = sum(r['penalized_time_s'] / r['source_total'] < 460 for r in subset)
        item['evidence_sha256'] = {
            (folder / name).relative_to(ROOT).as_posix(): sha(folder / name)
            for name in ('plan.json', 'release.json', 'manifest.json', 'freeze.json',
                         'source.zip', 'summary.json', 'independent_audit.json')
        }
        item['plan_sha256'] = sha(folder / 'plan.json')
        item['source_sha256'] = plan['source_sha256']
        item['record_file_sha256'] = {
            k: v for k, v in audit['input_sha256'].items()
            if k.endswith('.json.gz')
        }
        item['audit'] = {k: audit[k] for k in ('all_passed', 'records', 'passed_records', 'all_clear', 'errors', 'scope')}
        item['execution_git_commit'] = read(folder / 'freeze.json')['git_commit']
        report['splits'][split] = item
    report['total_runs'] = sum(v['runs'] for v in report['splits'].values())
    report['total_successful'] = sum(v['successful'] for v in report['splits'].values())
    (OUT / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    lines = [
        '# 可靠 R12：按源数分层的新口径基准', '',
        '238 局全部完成、全部清除并接受退出，独立物理和前缀审计全部通过。随机集平均 **511.26 秒/源**，压力集 **519.78 秒/源**，两组均未达到 460 秒/源。', '',
        '这次只测量既有合格 R12，没有更换算法或参数，也没有让新候选晋级。所有 `src/**` 保持 `81aa6e1a1231a2358cf098fbba3fdd0557b540ef`；执行工具和 release 提交为 `391a1841ad84699db681852b084c79d36c1e862e`。两组按冻结计划完整执行，没有提前停止、补抽或删除失败行。', '',
        '主指标为各局等权的 `mean(T_i/N_i)`。下表 T、LB 均以秒计，T/LB 指 `mean(T)/mean(LB)`，不是逐局比值的平均。LB 继续使用旧 Q4 共同下界：全知清除路程松弛，加 N<16 时的空频道排除成本；仅在终局计算，不进入策略。', '',
        '| 数据组 | 全清/局数 | mean(T/N) [95% CI] | P95(T/N) | mean(T) | mean(LB) | T/LB |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for split, name in [('confirmation', '随机确认'), ('stress', '压力')]:
        s = report['splits'][split]; lo, hi = s['stratified_bootstrap']['ci95_s']
        lines.append(f"| {name} | {s['successful']}/{s['runs']} | {s['mean_time_per_source_s']:.3f} [{lo:.3f}, {hi:.3f}] | {s['p95_time_per_source_s']:.3f} | {s['mean_time_s']:.3f} | {s['mean_lower_bound_s']:.3f} | {s['mean_time_over_mean_lower_bound']:.6f} |")
    lines += ['', '合并总时间再除以总源数的数值分别为 496.616、505.669 秒/源，它们不是主指标。相对当前观测均值，达到 460 还需分别降低约 10.03%、11.50%；这只是目标差距，不是任何下一种方法能实现的收益预测。', '',
              'N=10…16 每层随机20局、压力14局（七个家族各2局）。随机/压力总体分别报告，不混合成一个有利均值。各 N 下的完整费用分量、家族分层和区间见 `metrics.json`。', '']
    for split, name in [('confirmation', '随机确认'), ('stress', '压力')]:
        lines += [f'## {name}：源数分层', '', '| N | 全清/局数 | mean(T/N) [95% CI] | mean(T) | mean(LB) | T/LB |', '|---:|---:|---:|---:|---:|---:|']
        for n, s in report['splits'][split]['by_source_count'].items():
            lo, hi = s['stratified_bootstrap']['ci95_s']
            lines.append(f"| {n} | {s['successful']}/{s['runs']} | {s['mean_time_per_source_s']:.3f} [{lo:.3f}, {hi:.3f}] | {s['mean_time_s']:.3f} | {s['mean_lower_bound_s']:.3f} | {s['mean_time_over_mean_lower_bound']:.6f} |")
        lines.append('')
    lines += ['只有 N=15、16 的两个分层在两组中都出现均值低于460；N=10…14 仍高于目标。因此不能只挑高源数案例宣称方法总体达标。固定覆盖与空频道排除费用会被更多源分摊；N=16 还触发公开源数上限的停止发现规则，但本表本身不是这些机制各自贡献的因果消融。', '',
              '## 实际费用与计算成本', '', '| 数据组 | 移动 | 检测 | 换频 | 光学 | 清除 | mean(T) | mean(LB) | T/LB |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split, name in [('confirmation', '随机确认'), ('stress', '压力')]:
        s = report['splits'][split]; c = s['mean_components_s']
        lines.append(f"| {name} | {c['movement_s']:.3f} | {c['detection_s']:.3f} | {c['switching_s']:.3f} | {c['optical_s']:.3f} | {c['removal_s']:.3f} | {s['mean_time_s']:.3f} | {s['mean_lower_bound_s']:.3f} | {s['mean_time_over_mean_lower_bound']:.6f} |")
    lines += ['', '移动分别占总虚拟时间约75.76%、73.81%。本地三进程下平均策略计算耗时为0.211、0.634秒/局；它受机器和并发负载影响，不能当作虚拟任务时间。光学失败是允许并已计费的动作，全部清除不表示每次光学检查都成功。', '',
              '## 审计、复核与边界', '',
              '两组独立审计分别通过140/140和98/98，核验完整种子配额、原始动作费用、实际源数/清除/退出、全覆盖与旧下界、R12辅助区域、R8试清、提前服务调度、range跳测，以及原合格源和20项旧资格证据。没有基础设施失败。', '',
              '95%区间使用冻结 seed=630941、10000次层内bootstrap，随机按N、压力按N×家族。各N区间只限制到该N后重复同一规则；这些是描述性区间，不是同时置信带。压力每细层仅2局，不能外推为任意场景保证。它们说明本地固定生成分布的测量结果，不是官方演练成绩。', '',
              '所有原始记录与审计保持在 `results/q4_per_source_reference/{confirmation,stress}/`。`metrics.json` 保存完整源/plan/release/source.zip/summary/audit SHA、逐记录SHA、完整分层指标与区间。复核命令（仓库根目录，仅读现有结果）：', '',
              '```powershell',
              "& ../cumcm2026-b-interference-localization/.venv-win/Scripts/python.exe -B research/q4_per_source_reference/summarize_reference.py",
              '```', '',
              '派生报告脚本不会生成场景、运行策略或修改冻结计划。', '',
              '图示：[按源数分层的均值与P95](figures/per_source_by_count.png)、[总体每源费用构成](figures/per_source_components.png)。前者的P95不是置信区间；后者各费用先逐局除以N再平均，460虚线只表示总体目标。两图已人工检查标签和布局；对应PDF及输入/绘图脚本SHA保存在同一 figures 目录。绘图使用 core 的 `research/q4_round2/plot_per_source_reference.py`，输入仅本轮两份 summary.json，未修改 core。', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'runs': report['total_runs'], 'successful': report['total_successful'],
                      'outputs': ['research/q4_per_source_reference/metrics.json', 'research/q4_per_source_reference/RESULTS.md']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
