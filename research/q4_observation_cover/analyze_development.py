"""Post-exit R29 costs only; never read evaluation or construct scenarios."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / 'research/q4_observation_cover'
COMPONENTS = ('movement_s', 'detection_s', 'switching_s', 'optical_s', 'removal_s')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    output = dict(scope='Completed development records only. No evaluation/hidden source data or new simulation. Phase movement is charged to its destination action; it is not a causal marginal cost.', input_sha256={}, batches={}, paired={})
    rows_by_batch = {}
    for config in ('ring_28', 'ring_31'):
        for split in ('development', 'development-stress'):
            directory = ROOT / 'results/q4_observation_cover' / config / split
            summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
            phases = defaultdict(Counter)
            totals = Counter()
            per_case = []
            for path in sorted((directory / 'records').glob('*.json.gz')):
                record = json.loads(gzip.decompress(path.read_bytes()))
                row, report = record['row'], record['summary']
                actions = report['action_history']
                source_channels = set(report['cleared_channels'])
                assert row['successful'] and len(source_channels) == row['source_total']
                totals_one = Counter()
                source_measures = Counter()
                position, channel, time = (0.0, 0.0), 1, 0.0
                for action in actions:
                    phase = action['phase']
                    target = action['position']
                    costs = Counter(movement_s=round(math.dist(position, target) / 5 * 1e6) / 1e6)
                    if action['action'] == 'measure':
                        costs.update(detection_s=5, switching_s=int(channel != action['channel']))
                        channel = action['channel']
                        phases[phase]['measure_count'] += 1
                        phases[phase]['measure_' + action['result']] += 1
                        totals_one['measure_count'] += 1
                        if channel in source_channels:
                            source_measures[channel] += 1
                            totals_one['source_channel_measures'] += 1
                        else:
                            totals_one['empty_channel_measures'] += 1
                    elif action['action'] == 'clear':
                        costs.update(optical_s=3, removal_s=2 * int(action['result'] == 'success'))
                        phases[phase]['clear_count'] += 1
                        phases[phase]['clear_' + action['result']] += 1
                    else:
                        raise ValueError('Unexpected policy action')
                    assert abs(sum(costs.values()) - (action['virtual_time_s'] - time)) < 3e-6
                    phases[phase].update(costs)
                    totals_one.update(costs)
                    position, time = target, action['virtual_time_s']
                for key in COMPONENTS:
                    assert abs(totals_one[key] - row[key]) < 1e-4
                assert abs(time - row['virtual_time_s']) < 1e-5
                n = row['source_total']
                per_case.append(dict(seed=row['seed'], source_total=n,
                    all_measures_per_source=totals_one['measure_count'] / n,
                    actual_source_channel_measures_per_source=totals_one['source_channel_measures'] / n,
                    empty_channel_measures_per_source=totals_one['empty_channel_measures'] / n,
                    source_channel_measure_counts=dict(sorted(source_measures.items())),
                    coverage_points_visited=report['coverage_points_visited']))
                totals.update(totals_one)
                output['input_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
            count = summary['runs']
            assert count == len(per_case)
            key = config + '/' + split
            output['batches'][key] = dict(runs=count, successful=summary['successful'],
                mean_components_s=summary['mean_components_s'],
                mean_phase_counts_and_costs={p: {k:v/count for k,v in sorted(c.items())} for p,c in sorted(phases.items())},
                mean_all_measures_per_source=statistics.mean(r['all_measures_per_source'] for r in per_case),
                mean_actual_source_channel_measures_per_source=statistics.mean(r['actual_source_channel_measures_per_source'] for r in per_case),
                mean_empty_channel_measures_per_source=statistics.mean(r['empty_channel_measures_per_source'] for r in per_case),
                mean_coverage_points_visited=statistics.mean(r['coverage_points_visited'] for r in per_case),
                mean_measures=totals['measure_count']/count, per_case=per_case)
            rows_by_batch[key] = {r['seed']:r for r in summary['rows']}
            for name in ('summary.json', 'independent_audit.json'):
                path = directory / name
                output['input_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    for split in ('development', 'development-stress'):
        old, new = (rows_by_batch[c + '/' + split] for c in ('ring_28', 'ring_31'))
        assert old.keys() == new.keys()
        differences = []
        for seed in sorted(old):
            a,b = old[seed],new[seed]
            assert a['case_sha256'] == b['case_sha256'] and a['source_total'] == b['source_total']
            assert a['common_lower_bound_s'] == b['common_lower_bound_s']
            delta = b['virtual_time_s'] - a['virtual_time_s']
            differences.append(dict(seed=seed, source_total=a['source_total'],
                delta_time_s_31_minus_28=delta,
                delta_time_per_source_s_31_minus_28=delta/a['source_total'],
                delta_components_s={k:b[k]-a[k] for k in COMPONENTS},
                ring_28_time_over_lower_bound=a['time_over_lower_bound'],
                ring_31_time_over_lower_bound=b['time_over_lower_bound']))
        output['paired'][split] = dict(scope='Same frozen cases; positive delta means ring_31 costs more. No R12 rerun or R12 causal comparison.',
            ring_31_faster=sum(r['delta_time_s_31_minus_28'] < -1e-6 for r in differences),
            ring_31_slower=sum(r['delta_time_s_31_minus_28'] > 1e-6 for r in differences),
            equal=sum(abs(r['delta_time_s_31_minus_28']) <= 1e-6 for r in differences),
            mean_delta_time_s=statistics.mean(r['delta_time_s_31_minus_28'] for r in differences),
            mean_delta_time_per_source_s=statistics.mean(r['delta_time_per_source_s_31_minus_28'] for r in differences),
            mean_delta_components_s={k:statistics.mean(r['delta_components_s'][k] for r in differences) for k in COMPONENTS},
            all_paired_rows=differences)
    output['input_sha256']['research/q4_observation_cover/selection.json'] = sha(RESEARCH / 'selection.json')
    path = RESEARCH / 'development-analysis.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k:{n:v for n,v in data.items() if n not in ('per_case', 'mean_phase_counts_and_costs')} for k,data in output['batches'].items()}, ensure_ascii=False))
    print(json.dumps({k:{n:v for n,v in data.items() if n != 'all_paired_rows'} for k,data in output['paired'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    analyze()
