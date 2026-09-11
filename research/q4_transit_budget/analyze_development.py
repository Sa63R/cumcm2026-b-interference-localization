"""Post-exit R33 mechanism/fees, without evaluation data or any counterfactual run."""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from experiments.q4_transit_budget_release import actual_service_count
from experiments.run_q4_per_source import read, sha, write_new, family_from_seed, percentile

RESEARCH = ROOT / 'research/q4_transit_budget'
COMPONENTS = ('movement_s', 'detection_s', 'switching_s', 'optical_s', 'removal_s')


def group(rows):
    times = [r['T_s'] for r in rows]
    lowers = [r['LB_s'] for r in rows]
    ratios = [r['T_per_N_s'] for r in rows]
    return dict(runs=len(rows),successful=sum(r['successful'] for r in rows),
        source_counts=sorted(set(r['N'] for r in rows)),mean_N=statistics.mean(r['N'] for r in rows),
        mean_T_s=statistics.mean(times),mean_T_per_N_s=statistics.mean(ratios),
        pooled_T_per_N_s=sum(times)/sum(r['N'] for r in rows),p95_T_per_N_s=percentile(ratios,.95),
        mean_LB_s=statistics.mean(lowers),mean_T_over_mean_LB=sum(times)/sum(lowers),
        triggered_cases=sum(r['actual_service_actions']>0 for r in rows),
        mean_components_s={k:statistics.mean(r['components_s'][k] for r in rows) for k in COMPONENTS})


def main():
    selection = read(RESEARCH / 'selection.json')
    result = dict(scope='Completed development summary/action_history and audited service epochs only. No evaluation/hidden-source fields accessed. Observed service/phase costs are not whole-run causal effects.',
        selection_sha256=sha(RESEARCH/'selection.json'), input_sha256={}, batches={})
    for split in ('development', 'development-stress'):
        directory = ROOT/'results/q4_transit_budget'/split
        summary, audit = read(directory/'summary.json'), read(directory/'independent_audit.json')
        assert audit['all_passed'] is True
        audited = {item['seed']:item['transit_budget']['prefix'] for item in audit['audits']}
        phases, totals, rows = defaultdict(Counter), Counter(), []
        for path in sorted((directory/'records').glob('*.json.gz')):
            record = json.loads(gzip.decompress(path.read_bytes()))
            row, report = record['row'], record['summary']
            actions, events = report['action_history'], report['strategy_parameters']['transit_service_log']
            n, t, lb = row['source_total'], row['virtual_time_s'], row['common_lower_bound_s']
            actual_count = actual_service_count(record)
            assert actual_count == audited[row['seed']]['transit_service_actions']
            per_case, skips, stops, outcomes, services = Counter(), Counter(), Counter(), Counter(), []
            previous, channel = 0., 1
            for action in actions:
                fees = Counter()
                if action['action'] == 'measure':
                    fees.update(detection_s=5., switching_s=float(action['channel'] != channel))
                    channel = action['channel']
                    phases[action['phase']]['measure_'+action['result']] += 1
                else:
                    assert action['action'] == 'clear'
                    fees.update(optical_s=3., removal_s=2.*int(action['result']=='success'))
                    phases[action['phase']]['clear_'+action['result']] += 1
                movement = action['virtual_time_s'] - previous - sum(fees.values())
                assert movement >= -1e-6
                fees['movement_s'] = movement
                per_case.update(fees); phases[action['phase']].update(fees)
                previous = action['virtual_time_s']
            for key in COMPONENTS:
                assert abs(per_case[key]-row[key]) < 1e-5
            assert abs(previous-t) < 1e-5
            for event in events:
                if event['selected'] is None:
                    skips[event['skip_reason']] += 1
                    continue
                start, end = event['resolver_start_action_count'], event['service_end_action_count']
                actual = actions[start:end]
                outcomes[event['service_status']] += 1
                for reason in event.get('stopped_reasons',[]): stops[reason] += 1
                counts = Counter(a['action']+'_'+a['result'] for a in actual)
                item = dict(channel=event['selected']['channel'], start=start, end=end,
                    actual_actions=len(actual), outcome=event['service_status'], counts=dict(counts),
                    service_elapsed_s=(event['service_end_virtual_us']-event['start_virtual_us'])/1e6,
                    direct_transit_credit_s=event['baseline_floor_us']/1e6,
                    incremental_through_first_measure_s=event.get('actual_incremental_through_first_measure_us',0)/1e6,
                    first_coverage_action_index=event.get('first_coverage_action_index'),
                    stopped_reasons=event.get('stopped_reasons',[]))
                assert event.get('actual_incremental_through_first_measure_us') is not None
                assert item['incremental_through_first_measure_s'] <= 60.+1e-6
                services.append(item)
                for key,count in counts.items(): totals['service_'+key] += count
            assert sum(s['actual_actions'] for s in services) == actual_count
            totals['selected_services'] += len(services)
            totals['zero_action_services'] += sum(s['actual_actions']==0 for s in services)
            totals['triggered_cases'] += int(actual_count > 0)
            totals['service_actions'] += actual_count
            rows.append(dict(seed=row['seed'], N=n, T_s=t, T_per_N_s=t/n,
                family=family_from_seed(row['seed']) if split=='development-stress' else None,
                LB_s=lb, T_over_LB=t/lb, successful=row['successful'],
                components_s={k:row[k] for k in COMPONENTS},
                actual_service_actions=actual_count, services=services,
                skip_reasons=dict(skips), stop_reasons=dict(stops), service_outcomes=dict(outcomes)))
            result['input_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
        flat = [s for row in rows for s in row['services']]
        assert len(rows) == summary['runs']
        result['batches'][split] = dict(runs=len(rows), totals=dict(totals),
            metrics=group(rows),
            by_source_count={str(n):group([r for r in rows if r['N']==n]) for n in sorted(set(r['N'] for r in rows))},
            by_family={name:group([r for r in rows if r['family']==name]) for name in sorted(set(r['family'] for r in rows))} if split=='development-stress' else {},
            mean_components_s=summary['mean_components_s'],
            mean_phase_counts_and_costs={k:{n:v/len(rows) for n,v in sorted(c.items())} for k,c in sorted(phases.items())},
            mean_service_elapsed_s=statistics.mean(s['service_elapsed_s'] for s in flat) if flat else None,
            mean_incremental_through_first_measure_s=statistics.mean(s['incremental_through_first_measure_s'] for s in flat) if flat else None,
            max_incremental_through_first_measure_s=max((s['incremental_through_first_measure_s'] for s in flat),default=None),
            service_outcomes=dict(sum((Counter(row['service_outcomes']) for row in rows), Counter())),
            skip_reasons=dict(sum((Counter(row['skip_reasons']) for row in rows), Counter())),
            stop_reasons=dict(sum((Counter(row['stop_reasons']) for row in rows), Counter())),
            highest_T_per_N_rows=sorted(rows,key=lambda row:(-row['T_per_N_s'],row['seed']))[:5],
            all_rows=rows)
        for name in ('summary.json','independent_audit.json'):
            path=directory/name;result['input_sha256'][path.relative_to(ROOT).as_posix()] = sha(path)
    write_new(RESEARCH/'development-analysis.json',result)
    print(json.dumps({k:{n:v for n,v in b.items() if n not in ('all_rows','highest_T_per_N_rows','mean_phase_counts_and_costs')} for k,b in result['batches'].items()}))


if __name__ == '__main__':
    main()
