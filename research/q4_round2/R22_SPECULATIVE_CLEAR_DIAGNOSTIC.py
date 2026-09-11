"""Replay only the already opened 38 R12 development observation prefixes."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

CORE = Path(__file__).resolve().parents[2]
SOURCE = CORE.parent/'q4-r12-joint-continuation'
sys.path[:0] = [str(SOURCE/'src'), str(SOURCE)]
from localization import CandidateRegion
from experiments.diagnose_q4_joint_visibility import observation_record
from experiments.audit_q4_joint_continuation import wire_prefix

STAGES = {'development': list(range(621001,621025)),
          'development-stress': list(range(621031,621045))}
LABEL = 'compact_joint_continuation'
SPEC = dict(entrypoint='strategies.q4_joint_continuation:run_q4_joint_continuation',
            kwargs=dict(config='after_active_miss_optical', max_expansions=200))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def later_r8(events, history):
    # Post-hoc execution analysis only; never part of prefix eligibility.
    for event in events:
        event['later_original_r8_actions'] = [dict(prefix=j,result=a['result'])
            for j,a in enumerate(history) if j>event['prefix'] and
            a['channel']==event['channel'] and a['phase']=='speculative_clear_before_probe']


def scan(record, seed, stage):
    assert record['spec'] == SPEC
    assert record['summary']['completion_certified_under_model'] is True
    h, before = wire_prefix(record)
    params = record['summary']['strategy_parameters']
    resolvers = params['joint_visibility_resolver_log']
    services = params['early_service_log']
    regions, observed = {}, defaultdict(set)
    known, near, cleared, attempted = set(), set(), set(), set()
    counts, events, close_only = Counter(), [], []
    for n, action in enumerate(h):
        channel, point = action['channel'], tuple(action['position'])
        region = regions.get(channel)
        if action['action']=='measure' and action['phase']=='active_localization':
            counts['actual_active_measurements'] += 1
            assert channel in known-cleared-near and region is not None and region.vertices
            disk = region.enclosing_disk()
            delta = math.dist(point, disk.center)
            if point == tuple(disk.center):
                counts['exact_canonical_center_active_measurements'] += 1
            if 40. < disk.radius <= 120.:
                counts['active_with_canonical_radius_40_120'] += 1
                if point == tuple(disk.center):
                    epochs = [e for e in resolvers if e['channel']==channel and
                              e['after_actual_action_count'] <= n < e['end_actual_action_count']]
                    assert len(epochs)==1
                    early = [e for e in services if e['channel']==channel and
                             e['after_actual_action_count'] <= n < e['end_actual_action_count']]
                    assert len(early)<=1
                    # Eligibility is fixed from the prefix. Current measurement
                    # feedback is appended only as an explicitly post-hoc label.
                    events.append(dict(seed=seed, stage=stage, prefix=n, channel=channel,
                        position=list(point), canonical_center=list(disk.center), radius_m=disk.radius,
                        positive_count=len(region.observations),
                        previously_observed_exact=point in observed[channel],
                        previously_observed_round6=tuple(round(v,6) for v in point) in
                            {tuple(round(v,6) for v in p) for p in observed[channel]},
                        previously_attempted_r8=channel in attempted,
                        resolver_id=epochs[0]['id'], early_service=bool(early),
                        original_incoming_distance_m=math.dist(before[n][0],point),
                        actual_measure_fee_s=5.+int(before[n][1]!=channel),
                        actual_observed_feedback=action['result'],
                        actual_bearing_deg=action.get('bearing_deg'),
                        hypothetical_clear_result=None))
                elif delta <= 1e-7:
                    close_only.append(dict(prefix=n,channel=channel,radius_m=disk.radius,
                                           center_offset_m=delta))
        if action['action']=='clear':
            if action['phase']=='speculative_clear_before_probe':
                attempted.add(channel)
            if action['result']=='success':
                assert channel not in cleared
                cleared.add(channel);known.add(channel)
        else:
            observed[channel].add(point)
            if action['result']=='direction':
                regions.setdefault(channel,CandidateRegion()).observe(point,action['bearing_deg'])
                known.add(channel)
            elif action['result']=='near':
                near.add(channel);known.add(channel)
    counts['eligible_exact_center_radius_40_120'] = len(events)
    counts['near_equal_but_not_exact_excluded'] = len(close_only)
    later_r8(events,h)
    return dict(seed=seed,stage=stage,actual_successful_clears=len(cleared),
                counts=dict(counts),events=events,near_equal_exclusions=close_only)


def group(cases):
    events = [e for c in cases for e in c['events']]
    totals = Counter()
    for c in cases: totals.update(c['counts'])
    by_threshold = {}
    for threshold in (60.,80.,120.):
        all_events = [e for e in events if e['radius_m']<=threshold]
        eligible = [e for e in all_events if not e['early_service'] and
                    not e['previously_attempted_r8'] and not e['previously_observed_round6']]
        first = {}
        for e in eligible:
            first.setdefault((e['seed'],e['channel']),e)
        chosen = list(first.values())
        per_case = [sum(e['seed']==c['seed'] for e in chosen) for c in cases]
        by_threshold[str(int(threshold))] = dict(
            exact_center_events=len(all_events),feedback=dict(Counter(e['actual_observed_feedback'] for e in all_events)),
            cases_with_any=sum(any(e['seed']==c['seed'] for e in all_events) for c in cases),
            already_r8_attempted=sum(e['previously_attempted_r8'] for e in all_events),
            already_observed_round6=sum(e['previously_observed_round6'] for e in all_events),
            early_service_events=sum(e['early_service'] for e in all_events),
            first_fresh_formal_unattempted_source_events=len(chosen),
            first_event_feedback=dict(Counter(e['actual_observed_feedback'] for e in chosen)),
            first_events_with_later_original_r8=sum(bool(e['later_original_r8_actions']) for e in chosen),
            later_original_r8_results=dict(Counter(a['result'] for e in chosen for a in e['later_original_r8_actions'])),
            first_event_prefixes=[[e['seed'],e['channel'],e['prefix']] for e in chosen],
            max_first_events_in_one_case=max(per_case,default=0),
            frozen_trace_direct_all_fail_added_s=3.*len(chosen),
            frozen_trace_mean_direct_all_fail_added_s=3.*len(chosen)/len(cases),
            frozen_trace_mean_direct_all_fail_added_s_per_observed_clear=statistics.mean(
                3.*n/c['actual_successful_clears'] for n,c in zip(per_case,cases)),
            warning='Opportunity counts on the old trace only; added time can change scheduling and later observations. Not an overall regret bound.')
    return dict(cases=len(cases),counts=dict(totals),
        feedback=dict(Counter(e['actual_observed_feedback'] for e in events)),thresholds=by_threshold)


def main():
    output=Path(__file__).with_suffix('.json')
    if output.exists():raise ValueError('Preserve original diagnostic output')
    began=time.perf_counter();cases=[];inputs={}
    for stage,seeds in STAGES.items():
        directory=SOURCE/'results/q4_joint_continuation'/stage
        manifest=json.loads((directory/'manifest.json').read_bytes())
        assert manifest['seeds']==seeds and manifest['specs'][LABEL]==SPEC
        assert json.loads((directory/'independent_audit.json').read_bytes())['all_passed'] is True
        for name in ('src/geometry/__init__.py','src/localization/__init__.py'):
            assert sha(SOURCE/name)==manifest['source_sha256'][name]
        for name in ('manifest.json','freeze.json','source.zip','independent_audit.json'):
            inputs[f'{stage}/{name}']=sha(directory/name)
        for seed in seeds:
            path=directory/'records'/f'{LABEL}-{seed}.json.gz'
            record=observation_record(path)  # Decoder skips evaluation/row/truth entirely.
            case=scan(record,seed,stage);case['record_sha256']=sha(path)
            cases.append(case)
    result=dict(checked_local=datetime.now().astimezone().isoformat(),
        source_tree=str(SOURCE),input_sha256=inputs,script_sha256=sha(__file__),
        source_sha256={n:sha(SOURCE/n) for n in ('src/geometry/__init__.py',
            'src/localization/__init__.py','src/strategies/q4_clear_before_probe.py',
            'experiments/diagnose_q4_joint_visibility.py','experiments/audit_q4_joint_continuation.py')},
        all=group(cases),by_stage={s:group([c for c in cases if c['stage']==s]) for s in STAGES},
        cases=cases,elapsed_s=time.perf_counter()-began,policy_runs=0,new_scenario_seeds=[],
        performance_comparison=None,
        scope='Actual prefix opportunities and post-hoc radio categories only. No hypothetical optical outcomes, source truth, or replayed new policy.')
    with output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(all=result['all'],elapsed_s=result['elapsed_s']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
