"""Read-only observation-tree trace checks. No policy, particles, or truth read.

This checks logged arithmetic and decisions against accepted observation
prefixes; it does not prove the assumed posterior probabilities are calibrated.
"""
import math
from collections import Counter


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def point(value):
    require(isinstance(value, (tuple, list)) and len(value) == 2
        and all(isinstance(v, (int,float)) and not isinstance(v,bool) and math.isfinite(v)
                for v in value), 'Nonfinite or nonshared point')
    return tuple(value)


def rounded(value):
    return tuple(round(v,6) for v in value)


def same(a, b):
    return abs(a-b) <= 1e-6*max(1.,abs(a),abs(b))


def audit_observation_prefix(record):
    summary = record['summary']
    params = summary['strategy_parameters']
    logs = params.get('observation_tree_log', [])
    require(params.get('active_probe_algorithm') == 'finite-observation-tree-v1', 'Wrong observation algorithm')
    depth = params['observation_depth']
    require(type(depth) is int and depth in (1,2), 'Invalid configured depth')
    history = summary['action_history']
    wire = [row for row in record['history'] if row['action'] in ('/measure','/clear')]
    require(len(history) == len(wire), 'Wire/phase history length differs')
    # Bind all later prefix statements to actual accepted observations.
    for item, raw in zip(history,wire):
        response = raw['response']
        raw_point = raw['position']
        if isinstance(raw_point,dict):
            raw_point = (raw_point['x'],raw_point['y'])
        require(response.get('accepted') is True and point(item['position']) == point(raw_point)
            and raw['channel'] == item['channel'] and raw['action'] == '/'+item['action'],
            'Phase action differs from wire')
        result_key = 'measure_result' if item['action'] == 'measure' else 'clear_result'
        require(item['result'] == response[result_key]
            and finite(item['virtual_time_s']) and abs(item['virtual_time_s']-response['virtual_time_s']) <= 2e-6,
            'Phase feedback/time differs from wire')
        if item['result'] == 'direction':
            require(item.get('bearing_deg') == response['svd_deg'], 'Phase bearing differs from wire')
    services = params.get('early_service_log', [])
    if services:
        from experiments.audit_q4_scheduling import audit_scheduling_prefix
        audit_scheduling_prefix(record)  # Actual gates, geometry, duration and outcomes.
    times, reasons, used = [], Counter(), set()
    executed = changed = selected_changes = omitted_service = omitted_terminal = 0
    previous_index = -1
    for event in logs:
        n, channel = event['after_actual_action_count'], event['channel']
        require(type(n) is int and 0 <= n <= len(history) and previous_index <= n, 'Invalid/unsorted decision prefix')
        previous_index = n
        require(type(channel) is int and 1 <= channel <= 20, 'Invalid channel')
        prefix = history[:n]
        measures = [h for h in prefix if h['action'] == 'measure' and h['channel'] == channel]
        require(any(h['result'] == 'direction' for h in measures), 'No real positive bearing before decision')
        require(not any(h['result'] == 'near' for h in measures)
            and not any(h['action'] == 'clear' and h['channel'] == channel and h['result'] == 'success'
                        for h in prefix), 'Decision after near or successful clear')
        unique = {point(h['position']) for h in measures}
        require(type(event['prefix_unique']) is int and event['prefix_unique'] == len(unique), 'Unique-prefix count mismatch')
        require(event['version'] == 'finite-observation-tree-v1' and event['depth'] == depth,
                'Decision version/depth mismatch')
        effective = event['effective_depth']
        require(type(effective) is int and 1 <= effective <= depth, 'Invalid effective depth')
        baseline, selected = point(event['baseline']), point(event['selected'])
        seen = {rounded(p) for p in unique}
        require(rounded(baseline) not in seen and rounded(selected) not in seen, 'Repeated same-channel probe')
        require(type(event['fallback']) is bool, 'Invalid fallback flag')
        wall = event['decision_wall_s']
        require(finite(wall) and same(wall,event['cpu_s']), 'Invalid decision wall-time')
        times.append(wall)
        selected_changes += selected != baseline
        current = point(prefix[-1]['position']) if prefix else (0.,0.)
        tuned = next((h['channel'] for h in reversed(prefix) if h['action'] == 'measure'),1)
        scores = event['candidate_scores']
        require(isinstance(scores,list) and len(scores) <= 3, 'Invalid candidate count')
        locations = set()
        for candidate in scores:
            q = point(candidate['position'])
            require(q not in locations and rounded(q) not in seen, 'Duplicate/previously observed candidate')
            locations.add(q)
            require(finite(candidate['cost_s'],positive=True), 'Nonfinite candidate cost')
            branches = candidate['branches']
            require(isinstance(branches,list) and branches, 'Empty branches')
            outcomes, masses, future = set(), 0., 0.
            for branch in branches:
                outcome = tuple(branch['outcome'])
                require(outcome in (('near',),('no_signal',)) or (len(outcome) == 2
                    and outcome[0] == 'bearing' and type(outcome[1]) is int and 0 <= outcome[1] < 36),
                    'Invalid observation bin')
                require(outcome not in outcomes, 'Duplicate observation branch')
                outcomes.add(outcome)
                mass, cost = branch['mass'], branch['continuation_cost_s']
                require(finite(mass,positive=True) and mass <= 1.+1e-9 and finite(cost,positive=True),
                        'Invalid branch probability/cost')
                count, ess = branch['particles'], branch['spatial_ess']
                require(type(count) is int and count >= 1 and finite(ess,positive=True)
                    and 1.-1e-8 <= ess <= count+1e-8, 'Invalid spatial effective sample size')
                continuation = branch['continuation']
                if branch['reason'] == 'shared_second_probe':
                    require(effective == 2 and count >= 2 and ess >= 2.-1e-9
                        and outcome[0] != 'near', 'Singleton/terminal branch claims adaptive action')
                    p = point(continuation)
                    require(rounded(p) not in seen|{rounded(q)}, 'Repeated fixed-error second probe')
                else:
                    require(continuation is None and branch['reason'] in
                        {'near_optical','geometric_optical','depth_leaf','low_spatial_ess_leaf','no_fresh_leaf'},
                        'Nonshared or unknown continuation')
                if outcome[0] == 'near':
                    require(branch['reason'] == 'near_optical' and same(cost,5.), 'Near branch optical fee mismatch')
                masses += mass
                future += mass*cost
            require(abs(masses-1.) <= 1e-7, 'Branch mass does not normalize')
            expected = math.dist(current,q)/5.+5.+float(tuned != channel)+future
            require(same(candidate['cost_s'],expected), 'Root expected-cost arithmetic mismatch')
        if event['fallback']:
            require(selected == baseline, 'Fallback changed the baseline probe')
            require(event['reason'] != 'complete_tree', 'Fallback marked complete')
            reasons[event['reason']] += 1
        else:
            require(scores and event['reason'] == 'complete_tree', 'Missing complete candidate scores')
            require(point(scores[0]['position']) == baseline, 'Complete scoring omitted the baseline first candidate')
            best = min(scores,key=lambda candidate:candidate['cost_s'])
            require(selected == point(best['position']) and same(event['predicted_local_cost_s'],best['cost_s']),
                    'Selected probe is not the shared-cost argmin')
        next_action = history[n] if n < len(history) else None
        if (next_action is not None and next_action['action'] == 'measure'
                and next_action.get('phase') == 'active_localization' and next_action['channel'] == channel):
            require(n not in used and point(next_action['position']) == selected, 'Actual active probe differs from selected')
            used.add(n)
            executed += 1
            changed += selected != baseline
            continue
        matching = [s for s in services if s['channel'] == channel and s['interrupted']
                    and s['after_actual_action_count'] <= n == s['end_actual_action_count']]
        if matching:
            require(len(matching) == 1, 'Ambiguous interrupted service')
            event_service = matching[0]
            start = event_service['after_actual_action_count']
            start_time = history[start-1]['virtual_time_s'] if start else 0.
            now = prefix[-1]['virtual_time_s'] if prefix else 0.
            # Independently reconstruct the actual scheduler's next-action gate.
            next_fee = math.ceil(math.dist(current,selected)/5.*1e6)/1e6 + 5. + (tuned != channel)
            require(now+next_fee > start_time+event_service['budget_s'],
                    'Omitted probe fits the claimed service slice')
            omitted_service += 1
        else:
            require(n == len(history) and (summary.get('completion_reason') in
                {'action_budget','real_deadline','virtual_budget','request_rejected'} or summary.get('error')),
                'Selected probe has no actual action or justified interruption')
            omitted_terminal += 1
    actual_active = {i for i,h in enumerate(history)
                     if h['action'] == 'measure' and h.get('phase') == 'active_localization'}
    require(used == actual_active, 'Actual active probe is missing its decision log')
    return {'passed':True,'decisions':len(logs),'completed_trees':len(logs)-sum(reasons.values()),
        'fallbacks':sum(reasons.values()),'fallback_reason_counts':dict(reasons),
        'executed_decisions':executed,'executed_changed':changed,'selected_changes':selected_changes,
        'unexecuted_service_slice':omitted_service,'unexecuted_terminal':omitted_terminal,
        'decision_wall_s':times,'total_decision_wall_s':sum(times),
        'boundary':'Prefix/log arithmetic and execution audit only; not a proof of posterior likelihood calibration'}


def summarize_observation_audits(audits):
    """Pool audited records; actual change rate excludes unexecuted proposals."""
    audits = list(audits)
    require(all(a.get('passed') is True for a in audits),'Cannot summarize failed audits as valid')
    names = ('decisions','completed_trees','fallbacks','executed_decisions','executed_changed',
             'selected_changes','unexecuted_service_slice','unexecuted_terminal')
    result = {name:sum(a[name] for a in audits) for name in names}
    times = sorted(t for a in audits for t in a['decision_wall_s'])
    reasons = Counter()
    for a in audits:
        reasons.update(a['fallback_reason_counts'])
    result.update(records=len(audits),fallback_reason_counts=dict(reasons),
        fallback_rate=result['fallbacks']/result['decisions'] if result['decisions'] else None,
        actual_probe_change_rate=result['executed_changed']/result['executed_decisions'] if result['executed_decisions'] else None,
        total_decision_wall_s=sum(times),mean_decision_wall_s=sum(times)/len(times) if times else None,
        max_decision_wall_s=max(times) if times else None,
        p95_decision_wall_s=times[max(0,math.ceil(.95*len(times))-1)] if times else None)
    return result


def audit_observation_records(records):
    """Read-only batch convenience; stop on invalid evidence, never drop it."""
    return summarize_observation_audits(audit_observation_prefix(record) for record in records)
