"""Describe all frozen development outcomes, never rerun a policy or choose actions."""
from pathlib import Path
import collections
import gzip
import json
import math
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from experiments import run_q4_per_source as runner
from experiments.q4_anchor_first_hit_release import actual_service_count


def mean(values):
    return statistics.mean(values) if values else None


def describe(split):
    directory=ROOT/'results/q4_anchor_first_hit'/split
    summary=runner.read(directory/'summary.json');plan=runner.read(directory/'plan.json')
    audit=runner.read(directory/'independent_audit.json')
    assert summary['all_clear'] and summary['complete'] and summary['source_unchanged'] and audit['all_passed']
    counts=collections.Counter();reasons=collections.Counter();responses=collections.Counter()
    phase_cost=collections.Counter();phase_move=collections.Counter();changed=[];cases=[];inputs={}
    for seed in plan['seed_selection']['seeds']:
        path=directory/'records'/f"{plan['label']}-{seed}.json.gz"
        inputs[path.relative_to(ROOT).as_posix()]=runner.sha(path)
        record=json.loads(gzip.decompress(path.read_bytes()))
        row=record['row'];actions=record['summary']['action_history'];params=record['summary']['strategy_parameters']
        assert row['successful'] and row['seed']==seed
        events=params['anchor_first_hit_log'];epochs={e['id']:e for e in params['joint_visibility_resolver_log']}
        costs=[];old_time=0.;tuned=1;total_move=0.
        for a in actions:
            cost=a['virtual_time_s']-old_time
            radio=5.+int(a['channel']!=tuned) if a['action']=='measure' else 0.
            optical=3.+2.*(a['result']=='success') if a['action']=='clear' else 0.
            movement=cost-radio-optical
            assert movement>=-1e-6
            total_move+=movement;costs.append(cost)
            phase_cost[a['phase']]+=cost;phase_move[a['phase']]+=movement
            old_time=a['virtual_time_s'];tuned=a['channel'] if a['action']=='measure' else tuned
        assert math.isclose(sum(costs),row['virtual_time_s'],abs_tol=1e-5)
        assert math.isclose(total_move,row['movement_s'],abs_tol=1e-5)
        counts.update(e['status'] for e in events)
        reasons.update(e.get('unavailable_reason') for e in events if e['status']=='model_unavailable')
        actual=[e for e in events if e['executed_measure']]
        assert len(actual)==actual_service_count(record)
        cases.append(dict(seed=seed,source_total=row['source_total'],time_s=row['virtual_time_s'],
            time_per_source_s=row['virtual_time_s']/row['source_total'],lower_s=row['common_lower_bound_s'],
            time_over_lower_bound=row['time_over_lower_bound'],new_probes=len(actual)))
        for e in actual:
            start,end=e['after_actual_action_count'],e['end_actual_action_count']
            assert end==start+1
            epoch=epochs[e['resolver_id']];tail=epoch['end_actual_action_count']
            model=e['model'];responses[e['actual_result']]+=1
            changed.append(dict(seed=seed,channel=e['channel'],event_id=e['id'],result=e['actual_result'],
                canonical_radius_m=e['canonical_radius_m'],model_region_kind=e['model_region_kind'],
                first_action_s=costs[start],remaining_actual_resolver_s=sum(costs[start:tail]),
                remaining_actual_measures=sum(a['action']=='measure' for a in actions[start:tail]),
                remaining_actual_optical_checks=sum(a['action']=='clear' for a in actions[start:tail]),
                model_original_s=model['original_expected_cost_s'],model_selected_s=model['selected_expected_cost_s'],
                predicted_gain_s=model['original_expected_cost_s']-model['selected_expected_cost_s'],
                work_units=model['work_used']))
    n=len(cases)
    result={k:summary[k] for k in ('runs','successful','all_clear','mean_time_s','mean_time_per_source_s',
        'pooled_time_per_source_s','p95_time_per_source_s','mean_lower_bound_s','mean_time_over_mean_lower_bound',
        'mean_components_s','by_source_count','stratified_bootstrap')}
    result.update(first_audit_all_passed=True,source_unchanged=True,
        decision_status_counts=dict(counts),model_unavailable_reasons=dict(reasons),actual_probe_results=dict(responses),
        actual_new_probes=len(changed),triggered_cases=sum(c['new_probes']>0 for c in cases),
        actual_new_probe_fee_mean_per_case=sum(e['first_action_s'] for e in changed)/n,
        mean_changed_resolver_remaining_s=mean([e['remaining_actual_resolver_s'] for e in changed]),
        mean_changed_predicted_cost_s=mean([e['model_selected_s'] for e in changed]),
        mean_predicted_gain_s=mean([e['predicted_gain_s'] for e in changed]),
        mean_cost_by_phase_s={k:v/n for k,v in phase_cost.items()},
        mean_movement_by_phase_s={k:v/n for k,v in phase_move.items()},
        worst_ten_cases=sorted(cases,key=lambda c:c['time_per_source_s'],reverse=True)[:10],
        cases=cases,changed_probe_details=changed)
    for name in ('plan.json','summary.json','independent_audit.json','manifest.json','freeze.json','source.zip'):
        inputs[(directory/name).relative_to(ROOT).as_posix()]=runner.sha(directory/name)
    return result,inputs


def main():
    freeze=runner.read(ROOT/'research/q4_anchor_first_hit/source-freeze.json')
    assert runner.source_hashes()==freeze['source_sha256']
    output=dict(scope='All 119 fixed local development cases. Model predictions are not counterfactual realized savings; different batches are not paired improvements.',
        selection_sha256=runner.sha(ROOT/'research/q4_anchor_first_hit/development-selection.json'),
        source_sha256=freeze['source_sha256'],splits={},input_sha256={})
    for split in ('development','development-stress'):
        result,inputs=describe(split);output['splits'][split]=result;output['input_sha256'].update(inputs)
    assert runner.source_hashes()==freeze['source_sha256']
    runner.write_new(ROOT/'research/q4_anchor_first_hit/development-analysis.json',output)
    for split,result in output['splits'].items():
        print(json.dumps(dict(split=split,**{k:result[k] for k in ('runs','mean_time_per_source_s','mean_time_s','mean_lower_bound_s',
            'mean_time_over_mean_lower_bound','actual_new_probes','triggered_cases','actual_probe_results',
            'mean_changed_resolver_remaining_s','mean_changed_predicted_cost_s','mean_predicted_gain_s')})))


if __name__=='__main__':main()
