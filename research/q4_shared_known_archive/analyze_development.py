"""Post-exit R35 accounting; reads rows and actual histories, never evaluation/truth.

Run with PYTHONPATH=src from this tree. This is descriptive single-arm evidence,
not a simulation or a counterfactual comparison with a different seed set.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics as st

from experiments.run_q4_per_source import family_from_seed, percentile

ROOT = Path(__file__).resolve().parents[2]
PARTS = ('movement_s', 'switching_s', 'detection_s', 'optical_s', 'removal_s')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return dict(count=len(values), minimum=min(values), median=st.median(values),
                mean=st.mean(values), p95=percentile(values, .95), maximum=max(values)) if values else dict(count=0)


def costs(actions):
    result = []
    position, tuned, elapsed = (0., 0.), 1, 0
    for action in actions:
        part = dict.fromkeys(PARTS, 0.)
        part['movement_s'] = round(math.dist(position, action['position']) / 5 * 1_000_000) / 1_000_000
        if action['action'] == 'measure':
            part['detection_s'] = 5.
            part['switching_s'] = float(tuned != action['channel'])
            tuned = action['channel']
        else:
            assert action['action'] == 'clear'
            part['optical_s'] = 3.
            part['removal_s'] = 2. * (action['result'] == 'success')
        elapsed += round(sum(part.values()) * 1_000_000)
        assert abs(elapsed / 1_000_000 - action['virtual_time_s']) < 2e-6
        result.append(part)
        position = action['position']
    return result


def add_cost(parts):
    return {name: sum(item[name] for item in parts) for name in PARTS}


def action_stats(actions, parts):
    return dict(actions=len(actions), components_s=add_cost(parts),
                actual_time_s=sum(sum(p.values()) for p in parts),
                results=dict(Counter(a['action'] + ':' + a['result'] for a in actions)),
                phases=dict(Counter(a['phase'] for a in actions)))


def summarize(records):
    n = len(records)
    shares = [event for r in records for event in r['sharing']]
    measured = [event for event in shares if event['accepted_actions']]
    services = [s for r in records for s in r['services']]
    broad = [s for s in services if s['wide_before_cover']]
    precover = [s for s in services if s['remaining_covers']]
    notready = [s for s in precover if not s['ready_before']]
    phases = sorted({p for r in records for p in r['phase_costs']})
    service_components = {k: sum(s['components_s'][k] for s in broad) / n for k in PARTS}
    return dict(runs=n, all_clear=all(r['successful'] for r in records),
        mean_T_s=st.mean(r['T_s'] for r in records), mean_N=st.mean(r['N'] for r in records),
        mean_T_per_N_s=st.mean(r['T_per_N_s'] for r in records),
        pooled_T_per_N_s=sum(r['T_s'] for r in records)/sum(r['N'] for r in records),
        p95_T_per_N_s=percentile([r['T_per_N_s'] for r in records], .95),
        mean_LB_s=st.mean(r['LB_s'] for r in records),
        mean_T_over_mean_LB=sum(r['T_s'] for r in records)/sum(r['LB_s'] for r in records),
        mean_components_s={k:st.mean(r['components_s'][k] for r in records) for k in PARTS},
        mean_phase_components_s={p:{k:sum(r['phase_costs'].get(p, {}).get(k, 0.) for r in records)/n for k in PARTS} for p in phases},
        total_phase_actions={p:sum(r['phase_actions'].get(p, 0) for r in records) for p in phases},
        total_phase_measures={p:sum(r['phase_measures'].get(p, 0) for r in records) for p in phases},
        mean_measurements_per_source=st.mean(r['measurement_count']/r['N'] for r in records),
        mean_coverage_measurements=st.mean(r['phase_measures'].get('coverage', 0) for r in records),
        mean_failed_clears=st.mean(r['failed_clear_count'] for r in records),
        mean_program_runtime_s=st.mean(r['program_runtime_s'] for r in records),
        sharing_opportunity_events=len(shares),
        sharing_no_candidate_events=sum(e['status']=='no_candidate' for e in shares),
        sharing_status=dict(Counter(e['status'] for e in shares)),
        sharing_candidate_reasons=dict(sum((Counter(r['sharing_candidate_reasons']) for r in records), Counter())),
        sharing_nominal_eligible_candidates=sum(e['eligible_candidates'] for e in shares),
        sharing_measured_cases=sum(any(e['accepted_actions'] for e in r['sharing']) for r in records),
        sharing_measurements=len(measured),
        sharing_actual_feedback=dict(Counter(e['actual_result'] for e in measured)),
        sharing_actual_ready=sum(e['actual_ready_after'] is True for e in measured),
        sharing_actual_ready_feedback=dict(Counter(e['actual_result'] for e in measured if e['actual_ready_after'] is True)),
        sharing_actual_nonready=sum(e['actual_ready_after'] is False for e in measured),
        sharing_direct_cost_s=sum(e['actual_cost_s'] for e in measured),
        mean_sharing_direct_cost_s=sum(e['actual_cost_s'] for e in measured)/n,
        sharing_total_components_s={k:sum(e['components_s'][k] for e in measured) for k in PARTS},
        sharing_before_radius_m=distribution([e['actual_radius_before_m'] for e in measured]),
        sharing_predicted_radius_m=distribution([e['predicted_radius_m'] for e in measured]),
        sharing_actual_after_radius_m=distribution([e['actual_radius_after_m'] for e in measured if e['actual_radius_after_m'] is not None]),
        sharing_next_target_action=dict(Counter(e['next_target_action'] or 'none' for e in measured)),
        wide_service_cases=sum(any(s['wide_before_cover'] and s['actions'] for s in r['services']) for r in records),
        wide_service_macros=len(broad), wide_service_actions=sum(s['actions'] for s in broad),
        wide_service_cleared=sum(s['cleared'] for s in broad),
        wide_service_zero_actions=sum(s['actions']==0 for s in broad),
        wide_service_status=dict(Counter(s['status'] for s in broad)),
        wide_service_action_results=dict(sum((Counter(s['results']) for s in broad), Counter())),
        mean_wide_service_components_s=service_components,
        mean_wide_service_total_s=sum(service_components.values()),
        wide_service_cost_s=distribution([s['actual_time_s'] for s in broad]),
        precover_nonready_radius_m=distribution([s['radius_m'] for s in notready]),
        precover_nonready_radius_bins=dict(Counter('19.9-40' if s['radius_m']<=40 else '40-120' if s['radius_m']<=120 else '120-500' if s['radius_m']<=500 else '>500' for s in notready)),
        precover_nonready_positive_count=dict(Counter(str(s['positive_observations']) for s in notready)),
        precover_nonready_remaining_covers=distribution([s['remaining_covers'] for s in notready]),
        wide_service_exit_to_proxy_m=distribution([s['exit_to_proxy_m'] for s in broad]),
        wide_service_entry_to_proxy_m=distribution([s['entry_to_proxy_m'] for s in broad]),
        wide_service_actual_movement_m=distribution([s['actual_movement_m'] for s in broad]),
        wide_service_movement_minus_entry_proxy_m=distribution([s['movement_minus_entry_proxy_m'] for s in broad]),
        early_service_macros=sum(r['early_service_macros'] for r in records),
        mean_early_service_s=st.mean(r['early_service_s'] for r in records),
        planned_decisions=sum(r['planned_decisions'] for r in records),
        closed_proxy_decisions=sum(r['closed_proxy_decisions'] for r in records),
        mean_planning_wall_s=st.mean(r['planning_wall_s'] for r in records),
        first_clear_time_s=distribution([r['first_clear_time_s'] for r in records if r['first_clear_time_s'] is not None]),
        mean_after_last_clear_s=st.mean(r['after_last_clear_s'] for r in records),
        top_five_seeds_by_T_per_N=[r['seed'] for r in sorted(records, key=lambda x:x['T_per_N_s'], reverse=True)[:5]])


def analyze(root=ROOT):
    inputs, splits = {}, {}
    for split in ('development', 'development-stress'):
        folder = root/'results/q4_shared_known'/split
        saved = json.loads((folder/'summary.json').read_bytes())
        records = []
        for path in sorted((folder/'records').glob('*.json.gz')):
            inputs[path.relative_to(root).as_posix()] = sha(path)
            record = json.loads(gzip.decompress(path.read_bytes()))
            row, summary = record['row'], record['summary']
            actions, parameters = summary['action_history'], summary['strategy_parameters']
            parts = costs(actions)
            total = add_cost(parts)
            assert all(abs(total[k]-row[k]) < 2e-6 for k in PARTS)
            plans = {p['id']:p for p in parameters['known_source_plan_log']}
            services = []
            for event in parameters['known_source_service_log']:
                start, end = event['resolver_start_action_count'], event['service_end_action_count']
                selected = event['selected']; channel = selected['channel']
                plan = plans[event['decision_id']]
                evidence = next(s for s in plan['source_evidence'] if s['channel']==channel)
                stat = action_stats(actions[start:end], parts[start:end])
                assert abs(stat['actual_time_s']-event['actual_cost_s']) < 2e-6
                services.append(dict(decision_id=event['decision_id'], channel=channel, start=start, end=end,
                    ready_before=selected['ready_before'], radius_m=selected['radius_m'],
                    positive_observations=evidence['positive_observation_count'],
                    remaining_covers=len(event['remaining_covers_before']),
                    wide_before_cover=(not selected['ready_before'] and selected['radius_m']>40 and bool(event['remaining_covers_before'])),
                    start_time_s=event['start_virtual_time_s'], status=event['status'], cleared=event['cleared'],
                    proxy=evidence['target'], exit=event['end_position'],
                    exit_to_proxy_m=math.dist(event['end_position'], evidence['target']),
                    entry_to_proxy_m=math.dist(event['start_position'], evidence['target']),
                    actual_movement_m=stat['components_s']['movement_s']*5,
                    movement_minus_entry_proxy_m=stat['components_s']['movement_s']*5-math.dist(event['start_position'], evidence['target']), **stat))
            phase_parts, phase_actions, phase_measures = defaultdict(list), Counter(), Counter()
            for action, part in zip(actions, parts):
                phase_parts[action['phase']].append(part); phase_actions[action['phase']]+=1
                phase_measures[action['phase']]+=action['action']=='measure'
            clear_times = [a['virtual_time_s'] for a in actions if a['action']=='clear' and a['result']=='success']
            sharing, reason_counts = [], Counter()
            for event in parameters['shared_known_observation_log']:
                start,end=event['after_actual_action_count'],event['end_actual_action_count']
                reason_counts.update(candidate['reason'] for candidate in event['candidates'])
                selected=next((x for x in event['candidates'] if x['channel']==event['selected_channel']), None)
                stat=action_stats(actions[start:end],parts[start:end])
                assert abs(stat['actual_time_s']-event['actual_cost_s'])<2e-6
                if end>start:
                    assert end==start+1 and actions[start]['phase']=='shared_known_observation'
                    assert stat['components_s']['movement_s']==0
                    assert event['actual_result']==actions[start]['result']
                future=next((x for x in actions[end:] if x['channel']==event['selected_channel']),None) if selected else None
                sharing.append(dict(id=event['id'],trigger_channel=event['trigger_channel'],
                    trigger_clear_action_index=event['trigger_clear_action_index'],start=start,end=end,
                    target_channel=event['selected_channel'],status=event['status'],
                    candidate_channels=len(event['candidates']),eligible_candidates=sum(x['eligible'] for x in event['candidates']),
                    accepted_actions=end-start,actual_result=event['actual_result'],
                    actual_ready_after=event['actual_ready_after'],actual_cost_s=event['actual_cost_s'],
                    actual_radius_before_m=event['actual_radius_before_m'],actual_radius_after_m=event['actual_radius_after_m'],
                    predicted_radius_m=selected['predicted_radius_m'] if selected else None,
                    original_positive_count=selected['positive_observation_count'] if selected else None,
                    next_target_action=(future['action']+':'+future['result']) if future else None,
                    components_s=stat['components_s']))
            T, N, LB = row['penalized_time_s'], row['source_total'], row['common_lower_bound_s']
            early = parameters['early_service_log']
            records.append(dict(seed=row['seed'], case_sha256=row['case_sha256'], successful=row['successful'],
                family=family_from_seed(row['seed']) if split.endswith('stress') else None,
                T_s=T, actual_T_s=row['virtual_time_s'], N=N, T_per_N_s=T/N, LB_s=LB, T_over_LB=T/LB,
                components_s=total, measurement_count=row['measurement_count'], failed_clear_count=row['failed_clear_count'],
                program_runtime_s=row['program_runtime_s'],
                phase_costs={p:add_cost(v) for p,v in phase_parts.items()}, phase_actions=dict(phase_actions),
                phase_measures=dict(phase_measures), services=services,sharing=sharing,sharing_candidate_reasons=dict(reason_counts),
                early_service_macros=len(early), early_service_s=sum(e['actual_cost_s'] for e in early),
                coverage_visited=summary['coverage_points_visited'],
                planned_decisions=len(plans), closed_proxy_decisions=sum(p['result']['exact'] for p in plans.values()),
                planning_wall_s=sum(p['runtime_s'] for p in plans.values()),
                first_clear_time_s=min(clear_times) if clear_times else None,
                after_last_clear_s=row['virtual_time_s']-max(clear_times) if clear_times else row['virtual_time_s']))
        assert len(records)==saved['runs']
        splits[split] = dict(overall=summarize(records),
            by_N={str(n):summarize([r for r in records if r['N']==n]) for n in range(10,17)},
            by_family={f:summarize([r for r in records if r['family']==f]) for f in sorted({r['family'] for r in records if r['family']})},
            records=records)
        for name in ('summary.json','independent_audit.json','manifest.json','freeze.json','source.zip','plan.json'):
            p=folder/name
            if p.exists():inputs[p.relative_to(root).as_posix()]=sha(p)
    return dict(scope='Post-exit single-arm descriptive accounting. No evaluation/truth read; no counterfactual savings claim.',
        source_freeze_sha256=sha(root/'research/q4_shared_known/source-freeze.json'),
        analysis_script_sha256=sha(Path(__file__)), input_sha256=inputs, splits=splits)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=analyze()
    with args.output.open('x',encoding='utf-8',newline='\n') as out:
        json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')
    for split,s in result['splits'].items():
        print(split,json.dumps({k:s['overall'][k] for k in ('runs','mean_T_per_N_s','wide_service_cases','wide_service_macros','wide_service_actions','mean_wide_service_total_s')}))
