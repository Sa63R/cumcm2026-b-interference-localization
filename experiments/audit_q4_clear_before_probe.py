"""Audit accepted observation prefixes for one speculative clear before probe.

No strategy/planner or hidden-source imports. Radius <=40 is eligibility, not
a radius-20 clearance certificate. Generic physical audit remains required.
"""
import math

from localization import CandidateRegion


def require(ok, message):
    if not ok:
        raise ValueError(message)


def point(p):
    if isinstance(p, dict):
        p = (p['x'], p['y'])
    require(isinstance(p, (tuple, list)) and len(p) == 2 and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in p), 'Invalid position')
    return tuple(p)


def close(actual, expected, message):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and
            math.isfinite(actual) and abs(actual-expected) <= 2e-6, message)


def terminal(summary, end, history):
    return end == len(history) and (summary.get('completion_reason') in
        {'action_budget', 'virtual_budget', 'real_deadline', 'request_rejected', 'protocol_error'}
        or bool(summary.get('error')))


def audit_clear_before_probe_prefix(record):
    summary, raw = record['summary'], record['history']
    params, h = summary['strategy_parameters'], summary['action_history']
    require(params.get('clear_before_probe_config') == 'center_once', 'Unknown insertion configuration')
    wire = []
    accepted_counts, count, nraw, virtual_limit = {0: 0}, 0, 0, 360000.
    for a in raw:
        if a['response'].get('accepted') is not True:
            continue
        if a['action'] == '/enter':
            require(nraw == 0, 'Session enter appeared inside physical history')
            count += 1
            accepted_counts[0] = count
            value = a['response'].get('max_virtual_duration_s')
            if value is not None:
                virtual_limit = min(360000., float(value))
        elif a['action'] in {'/measure', '/clear'}:
            wire.append(a)
            count += 1
            nraw += 1
            accepted_counts[nraw] = count
    require(len(wire) == len(h), 'Accepted action/wire length mismatch')
    current, tuned, elapsed = (0., 0.), 1, 0.
    before = [(current, tuned, elapsed)]
    for item, a in zip(h, wire):
        response = a['response']
        result = response['measure_result' if item['action'] == 'measure' else 'clear_result']
        p = point(item['position'])
        require('/'+item['action'] == a['action'] and item['channel'] == a['channel'] and
                p == point(a['position']) and item['result'] == result, 'Actual action differs from wire')
        close(item['virtual_time_s'], response['virtual_time_s'], 'Wire time mismatch')
        if result == 'direction':
            require(item['bearing_deg'] == response['svd_deg'], 'Wire bearing mismatch')
        if item['action'] == 'measure':
            fee = 5.+int(item['channel'] != tuned)
            tuned = item['channel']
        else:
            require(result in {'success', 'no_target_in_range'}, 'Unknown clear feedback')
            fee = 3.+2.*(result == 'success')
        fee += round(math.dist(current, p)/5.*1e6)/1e6
        close(item['virtual_time_s']-elapsed, fee, 'Actual movement/action fee mismatch')
        current, elapsed = p, item['virtual_time_s']
        before.append((current, tuned, elapsed))
    services = params.get('early_service_log', [])
    if services:
        from experiments.audit_q4_scheduling import audit_scheduling_prefix
        audit_scheduling_prefix(record)
    events = params.get('clear_before_probe_log', [])
    attempts = successes = failures = skipped = measured = interrupted = 0
    wall = 0.
    attempted, consumed, previous_end = set(), set(), 0
    for e in events:
        n, end, c = e['after_actual_action_count'], e['end_actual_action_count'], e['channel']
        require(type(n) is int and type(end) is int and previous_end <= n <= end <= len(h)
                and end-n <= 2, 'Invalid or overlapping insertion interval')
        require(type(c) is int and 1 <= c <= 20 and c not in attempted, 'Repeated accepted attempt for source')
        previous_end = end
        region, near, cleared, observed = CandidateRegion(), False, False, set()
        for a in h[:n]:
            if a['channel'] != c:
                continue
            if a['action'] == 'measure':
                observed.add(tuple(round(x, 6) for x in point(a['position'])))
                if a['result'] == 'direction':
                    region.observe(a['position'], a['bearing_deg'])
                elif a['result'] == 'near':
                    near = True
            elif a['result'] == 'success':
                cleared = True
        require(region.observations and region.vertices and not near and not cleared, 'Source is not observed live/non-near')
        disk = region.enclosing_disk()
        p, center = point(e['position']), tuple(disk.center)
        require(19.9 < disk.radius <= 40. and p == center == point(e['center']), 'Ineligible radius or non-centre planned probe')
        require(tuple(round(x, 6) for x in p) not in observed, 'Original centre probe was already measured')
        require([point(v) for v in e['vertices']] == list(map(tuple, region.vertices)), 'Region differs from actual positive prefix')
        close(e['radius_m'], disk.radius, 'Wrong actual enclosing radius')
        require(type(e['positive_observation_count']) is int and
                e['positive_observation_count'] == len(region.observations), 'Wrong positive observation count')
        require(e['phase'] == 'active_localization' and e['speculative_phase'] == 'speculative_clear_before_probe',
                'Speculative clear falsely labelled certified')
        current, tuned, now = before[n]
        require(point(e['current_position']) == current, 'Stale insertion position')
        b = e['budget']
        limit = b['max_actions']
        require(type(limit) is int and limit >= 2 and b['policy_action_count'] == accepted_counts[n]
                and b['remaining_actions'] == limit-accepted_counts[n], 'Wrong accepted action/exit reserve count')
        if 'spec' in record and 'max_actions' in record['spec'].get('kwargs', {}):
            require(limit == record['spec']['kwargs']['max_actions'], 'Action budget differs from frozen spec')
        movement = math.ceil(math.dist(current, p)/5.*1e6)/1e6
        cost = movement+8.+int(c != tuned)
        close(b['movement_ceiling_s'], movement, 'Wrong conservative incoming movement')
        close(b['worst_failure_cost_s'], cost, 'Failure cost omits real clear/measure/switch')
        close(b['virtual_limit_s'], virtual_limit, 'Virtual budget differs from actual enter')
        containing = [s for s in services if s['channel'] == c and
                      s['after_actual_action_count'] <= n < s['end_actual_action_count']]
        # Zero-action service interruption may end exactly at this event prefix.
        if b['service_deadline'] is not None and not containing:
            containing = [s for s in services if s['channel'] == c and s['interrupted'] and
                          s['after_actual_action_count'] <= n == s['end_actual_action_count']]
        if containing:
            require(len(containing) == 1 and b['service_deadline'] is not None, 'Missing/ambiguous service budget')
            s = containing[0]
            close(b['service_deadline'], before[s['after_actual_action_count']][2]+s['budget_s'], 'Wrong real service deadline')
        else:
            require(b['service_deadline'] is None, 'Fabricated service deadline')
        real = b['remaining_real_s']
        require(real is None or isinstance(real, (int, float)) and math.isfinite(real), 'Invalid real-time budget snapshot')
        reasons = []
        if accepted_counts[n]+3 > limit:reasons.append('action_budget')
        if now+cost > virtual_limit-1e-6:reasons.append('virtual_budget')
        if b['service_deadline'] is not None and now+cost > b['service_deadline']:reasons.append('service_budget')
        if real is not None and real <= 2.:reasons.append('real_deadline')
        require(b['skip_reasons'] == reasons, 'Incorrect insertion budget decision')
        require(type(e['executed']) is bool and isinstance(e['decision_wall_s'], (int, float)) and
                math.isfinite(e['decision_wall_s']) and e['decision_wall_s'] >= 0., 'Invalid decision metadata')
        wall += e['decision_wall_s']
        if reasons:
            require(e['status'] == 'skipped_budget' and not e['executed'] and end == n and
                    e['clear_result'] is None, 'Budget-skipped proposal executed/fabricated response')
            skipped += 1
            resumed = (n < len(h) and h[n]['action'] == 'measure' and h[n]['channel'] == c and
                point(h[n]['position']) == p and h[n].get('phase') == 'active_localization')
            if not resumed:
                original_fee = movement+5.+int(c != tuned)
                service_stop = (len(containing) == 1 and containing[0]['interrupted'] and
                    containing[0]['end_actual_action_count'] == n and
                    now+original_fee > b['service_deadline'])
                require(service_stop or terminal(summary, n, h),
                        'Skipped insertion omitted original measure without service/terminal evidence')
            continue
        if not e['executed']:
            require(end == n and e['clear_result'] is None and e['status'] == 'interrupted'
                    and terminal(summary, end, h) and e.get('interruption_type'),
                    'Unexecuted insertion lacks real terminal interruption')
            interrupted += 1
            continue
        require(end >= n+1 and n not in consumed, 'Missing actual speculative action')
        a = h[n]
        require(a['action'] == 'clear' and a['channel'] == c and point(a['position']) == p
                and a.get('phase') == 'speculative_clear_before_probe' and e['clear_result'] == a['result'],
                'Accepted speculative action mismatch')
        consumed.add(n);attempted.add(c);attempts += 1
        if a['result'] == 'success':
            require(end == n+1 and e['status'] == 'cleared' and
                    all(x['channel'] != c for x in h[end:]), 'Success failed to stop this resolved source')
            successes += 1
        else:
            failures += 1
            if end == n+2:
                a = h[n+1]
                require(e['status'] == 'failed_then_measured' and a['action'] == 'measure' and
                        a['channel'] == c and point(a['position']) == p and
                        a.get('phase') == 'active_localization', 'Failed clear did not resume same actual probe')
                measured += 1
            else:
                require(e['status'] == 'interrupted' and terminal(summary, end, h) and e.get('interruption_type'),
                        'Failure omitted original measure without actual terminal interruption')
                interrupted += 1
    require(consumed == {i for i, a in enumerate(h) if a.get('phase') == 'speculative_clear_before_probe'},
            'Unlogged speculative clear')
    # When final regions are present, they must contain only real direction
    # updates; failed optical checks may not silently clip the true outer C.
    for c, estimate in summary.get('source_estimates', {}).items():
        region = CandidateRegion()
        for a in h:
            if a['channel'] == int(c) and a['action'] == 'measure' and a['result'] == 'direction':
                region.observe(a['position'], a['bearing_deg'])
        if 'vertices' in estimate:
            require([point(v) for v in estimate['vertices']] == list(map(tuple, region.vertices)),
                    'Final C contains non-observation update')
    return dict(passed=True, events=len(events), attempts=attempts, successes=successes, failures=failures,
                skipped_budget=skipped, failed_then_measured=measured, terminal_interruptions=interrupted,
                decision_wall_s=wall, boundary='Actual prefixes and costs; remaining real time is a logged wall-clock snapshot, not a guaranteed future allowance')


def summarize_clear_before_probe_audits(audits):
    keys = ('events', 'attempts', 'successes', 'failures', 'skipped_budget', 'failed_then_measured',
            'terminal_interruptions', 'decision_wall_s')
    require(all(a.get('passed') is True for a in audits), 'Cannot aggregate failed audits')
    return {'records': len(audits), **{k: sum(a[k] for a in audits) for k in keys}}
