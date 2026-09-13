"""One prespecified held-out Lite/full comparison, without variant selection."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REPLICATES = 50000
METHODS = ['v4', 'full', 'lite']


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def group_summary(rows):
    keys = sorted({r['case_key'] for r in rows})
    index = {(r['case_key'], r['method']): r for r in rows}
    assert len(index) == len(rows) == len(keys) * 3
    values = np.array([[index[key, method]['seconds_per_source'] for method in METHODS] for key in keys])
    means = values.mean(axis=0)
    rng = np.random.default_rng(912675412026)
    bootstrap = np.empty((REPLICATES, 3))
    for start in range(0, REPLICATES, 250):
        size = min(250, REPLICATES - start)
        chosen = rng.integers(0, len(keys), size=(size, len(keys)))
        bootstrap[start:start + size] = values[chosen].mean(axis=1)
    delta = values[:, 2] - values[:, 1]
    relative = 100 * (values[:, 2] / values[:, 1] - 1)
    summary = dict(cases=len(keys), methods={}, lite_vs_full=dict(
        change_seconds_per_source=float(delta.mean()),
        change_ci95=interval(bootstrap[:, 2] - bootstrap[:, 1]),
        saving_percent=float(100 * (1 - means[2] / means[1])),
        saving_percent_ci95=interval(100 * (1 - bootstrap[:, 2] / bootstrap[:, 1])),
        faster=int(np.sum(delta < -1e-6)), slower=int(np.sum(delta > 1e-6)),
        equal=int(np.sum(np.abs(delta) <= 1e-6)),
        worst_slowdown_percent=float(relative.max()),
        worst_case=keys[int(relative.argmax())],
        best_saving_percent=float(-relative.min()), best_case=keys[int(relative.argmin())],
        slowdown_p95_percent=float(np.quantile(relative, .95)),
        median_change_seconds_per_source=float(np.median(delta))))
    for j, method in enumerate(METHODS):
        rr = [index[key, method] for key in keys]
        vs_v4 = 100 * (values[:, j] / values[:, 0] - 1)
        data = dict(mean_seconds_per_source=float(means[j]),
                    saving_vs_v4_percent=float(100 * (1 - means[j] / means[0])),
                    saving_vs_v4_ci95=interval(100 * (1 - bootstrap[:, j] / bootstrap[:, 0])),
                    worst_slowdown_vs_v4_percent=float(vs_v4.max()),
                    worst_vs_v4_case=keys[int(vs_v4.argmax())],
                    p95_slowdown_vs_v4_percent=float(np.quantile(vs_v4, .95)))
        for field in ['virtual_time_s', 'distance_m', 'measurement_count', 'switch_count',
                      'failed_clear_count', 'transit_stops', 'transit_detections',
                      'planner_calls', 'planner_accepts', 'shared_detections',
                      'visited_station_count', 'last_discovery_s', 'tail_after_last_clear_s']:
            data['mean_' + field] = float(np.mean([r[field] for r in rr]))
        data['time_per_source_breakdown'] = {
            field: float(np.mean([r['time_breakdown_s'][field] / r['source_total'] for r in rr]))
            for field in rr[0]['time_breakdown_s']}
        summary['methods'][method] = data
    summary['by_source_count'] = {}
    for n in range(10, 17):
        chosen = [key for key in keys if index[key, 'full']['source_total'] == n]
        if chosen:
            summary['by_source_count'][str(n)] = dict(cases=len(chosen), methods={method: {
                field: float(np.mean([index[key, method][field] for key in chosen]))
                for field in ['seconds_per_source', 'last_discovery_s', 'visited_station_count']}
                for method in METHODS})
    summary['largest_regressions'] = [{
        'case_key': keys[i], 'source_total': index[keys[i], 'full']['source_total'],
        'slowdown_vs_full_percent': float(relative[i]),
        'task_seconds': {method: index[keys[i], method]['virtual_time_s'] for method in METHODS}}
        for i in np.argsort(relative)[-5:][::-1]]
    return summary


def main():
    plan = json.loads((ROOT / 'plan.json').read_text())
    rows = [json.loads(line) for line in (ROOT / 'main/records.jsonl').read_text().splitlines()]
    expected = {(case['key'], method) for case in plan['cases'] for method in METHODS}
    assert len(rows) == len(expected) and {(r['case_key'], r['method']) for r in rows} == expected
    failures = [r for r in rows if r['error'] or not r['all_cleared']]
    if failures:
        (ROOT / 'failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2))
        raise RuntimeError('Failures retained; refusing a success-only mean')
    result = dict(local_reconstructed_only=True, runs=len(rows), unique_new_cases=len(plan['cases']),
                  all_cleared=True, planner_wall_budget_reached=sum(bool(r['planner_budget_reached']) for r in rows),
                  source_instances=sum(r['source_total'] for r in rows if r['method'] == 'full'),
                  successful_clear_events=sum(r['source_total'] for r in rows), bootstrap_replicates=REPLICATES,
                  primary_contrast='Lite minus original V6, complete task seconds/source on 1000 new random cases',
                  groups={})
    for group in dict.fromkeys(case['group'] for case in plan['cases']):
        result['groups'][group] = group_summary([r for r in rows if r['group'] == group])
        print(group, json.dumps(result['groups'][group]['lite_vs_full']), flush=True)
    (ROOT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    simple = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
    with (ROOT / 'paired_records.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(simple[0]))
        writer.writeheader()
        writer.writerows(simple)


if __name__ == '__main__':
    main()
