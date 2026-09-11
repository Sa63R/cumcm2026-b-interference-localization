"""Read already-completed development records; no counterfactual simulation."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def mean(xs):
    return statistics.mean(xs) if xs else None


def analyze(split):
    directory = ROOT / 'results/q4_enroute_probe' / split
    cases, events, statuses, inputs = [], [], Counter(), {}
    for path in sorted((directory/'records').glob('*.json.gz')):
        blob = path.read_bytes()
        inputs[path.relative_to(ROOT).as_posix()] = hashlib.sha256(blob).hexdigest()
        record = json.loads(gzip.decompress(blob))
        # Never inspect evaluation/source truth: only actual actions, logged
        # decisions and the already-post-exit scored row's source count.
        row = record['row']
        summary = record['summary']
        history = summary['action_history']
        actual = []
        for event in summary['strategy_parameters']['enroute_probe_log']:
            statuses[event['status']] += 1
            if not event['executed_measure']:
                continue
            selected = event['selected']
            channel = selected['channel']
            start, end = event['after_actual_action_count'], event['end_actual_action_count']
            assert end == start+1 and history[start]['phase'] == 'enroute_localization'
            assert history[start]['channel'] == channel and history[start]['result'] == event['actual_feedback']
            next_coverage = next((i for i in range(end,len(history)) if history[i]['phase']=='coverage'), None)
            own_clear = next((i for i in range(end,len(history)) if history[i]['action']=='clear'
                             and history[i]['channel']==channel and history[i]['result']=='success'), None)
            later_same_measures = [a for a in history[end:own_clear] if a['action']=='measure' and a['channel']==channel]
            previous = next(x for x in event['regions'] if x['channel']==channel)
            item = dict(seed=row['seed'], event_id=event['id'], channel=channel,
                actual_action_index=start, branch=event['selection_branch'],
                feedback=event['actual_feedback'], ready_after=event['actual_ready_after'],
                old_radius_m=previous['radius_m'], nominal_radius_m=selected['nominal_radius_m'],
                direct_radio_s=5.+int(channel!=event['current_channel']),
                same_channel_measures_before_clear=len(later_same_measures),
                same_channel_active_before_clear=sum(a['phase']=='active_localization' for a in later_same_measures),
                own_successful_clear_index=own_clear, next_coverage_action_index=next_coverage,
                cleared_before_next_coverage=own_clear is not None and (next_coverage is None or own_clear < next_coverage),
                next_action_is_own_clear=end < len(history) and history[end]['action']=='clear' and history[end]['channel']==channel,
                virtual_delay_until_own_clear_s=None if own_clear is None else history[own_clear]['virtual_time_s']-history[start]['virtual_time_s'])
            actual.append(item)
            events.append(item)
        cases.append(dict(seed=row['seed'],source_total=row['source_total'],successful=row['successful'],
            time_s=row['virtual_time_s'],time_per_source_s=row['penalized_time_s']/row['source_total'],
            time_over_lower_bound=row['penalized_time_over_lower_bound'],
            movement_s=row['movement_s'],detection_s=row['detection_s'],switching_s=row['switching_s'],
            optical_s=row['optical_s'],extra_measures=len(actual),
            extra_silent=sum(e['feedback']=='no_signal' for e in actual),
            extra_ready=sum(e['ready_after'] for e in actual),record=path.relative_to(ROOT).as_posix()))
    return dict(cases=len(cases),triggered_cases=sum(c['extra_measures']>0 for c in cases),
        cap_four_cases=sum(c['extra_measures']==4 for c in cases),
        accepted_extra_measures=len(events),statuses=dict(statuses),
        actual_feedback=dict(Counter(e['feedback'] for e in events)),
        branch_counts=dict(Counter(e['branch'] for e in events)),
        actually_ready=sum(e['ready_after'] for e in events),
        not_actually_ready=sum(not e['ready_after'] for e in events),
        immediately_next_own_clear=sum(e['next_action_is_own_clear'] for e in events),
        cleared_before_next_coverage=sum(e['cleared_before_next_coverage'] for e in events),
        ready_but_not_cleared_before_next_coverage=sum(e['ready_after'] and not e['cleared_before_next_coverage'] for e in events),
        total_direct_radio_s=sum(e['direct_radio_s'] for e in events),
        mean_direct_radio_s_per_case=sum(e['direct_radio_s'] for e in events)/len(cases),
        mean_old_radius_m=mean([e['old_radius_m'] for e in events]),
        mean_nominal_radius_m=mean([e['nominal_radius_m'] for e in events]),
        mean_same_channel_measures_before_clear=mean([e['same_channel_measures_before_clear'] for e in events]),
        tail_cases=sorted(cases,key=lambda c:c['time_per_source_s'],reverse=True)[:3],
        rows=cases,events=events,input_sha256=inputs)


if __name__ == '__main__':
    result = dict(scope='Actual-prefix mechanism only; a single arm gives no same-case baseline savings or regression counterfactual.',
                  splits={name:analyze(name) for name in ('development','development-stress')})
    output = Path(__file__).with_name('development-mechanism.json')
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({s:{k:v for k,v in r.items() if k not in ('rows','events','input_sha256','tail_cases')}
                      for s,r in result['splits'].items()},ensure_ascii=False))
