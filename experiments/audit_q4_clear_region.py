"""Read-only clear-region certificate/anchor audit, independent of its policy.

Uses accepted observations and actual route witnesses, never scenario truth or
the candidate optimizer. It certifies safety and the logged local proxy only.
"""
import math

from localization import CandidateRegion


def require(ok, message):
    if not ok:
        raise ValueError(message)


def point(value):
    if isinstance(value, dict):
        value = (value['x'], value['y'])
    require(isinstance(value, (tuple, list)) and len(value) == 2 and
            all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                for x in value), 'Invalid finite position')
    return tuple(value)


def close(actual, expected, message):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and
            math.isfinite(actual) and abs(actual-expected) <= 1e-6, message)


def belief(history):
    regions, near, known, cleared = {}, {}, set(), set()
    for a in history:
        c, p = a['channel'], point(a['position'])
        if a['action'] == 'measure':
            if a['result'] in {'near', 'direction'}:
                known.add(c)
            if a['result'] == 'near':
                near[c] = p
            elif a['result'] == 'direction':
                regions.setdefault(c, CandidateRegion()).observe(p, a['bearing_deg'])
        elif a['result'] == 'success':
            known.add(c)
            cleared.add(c)
    return regions, near, known, cleared


def target(c, regions, near):
    if c in near:
        return near[c]
    require(c in regions and regions[c].vertices, 'Anchor source has no observed region')
    return tuple(regions[c].enclosing_disk().center)


def anchor_witness(event, params, history, covers, state):
    n, c = event['after_actual_action_count'], event['channel']
    anchor, source, status = event['anchor'], event['anchor_source'], event['anchor_status']
    if event['config'] == 'incoming':
        require(anchor is None and source is None and status == 'incoming_configuration',
                'Incoming configuration used an anchor')
        return None
    if anchor is None:
        require(source is None and status in {'missing', 'route_prefix_expired'}, 'Invalid absent anchor')
        if status == 'route_prefix_expired':
            routes = [r for r in params.get('chain_route_log', [])
                      if r['after_actual_action_count'] < n]
            require(routes and routes[-1]['selected_kind'] == 'source' and
                    routes[-1]['selected_channel'] == c and
                    all(a['channel'] == c for a in history[routes[-1]['after_actual_action_count']:n]),
                    'Expired route has no actual same-source macro prefix')
        return None
    q = point(anchor)
    require(status == 'valid' and isinstance(source, dict) and source['channel'] == c,
            'Anchor lacks a valid source witness')
    if source['kind'] == 'route_second_task':
        i = source['route_log_index']
        routes = params.get('chain_route_log', [])
        require(type(i) is int and 0 <= i < len(routes), 'Invalid route log index')
        r = routes[i]
        require(source['prefix'] == n == r['after_actual_action_count'] and
                r['selected_kind'] == 'source' and r['selected_channel'] == c,
                'Route anchor is stale or selects another source')
        require(not any(x['after_actual_action_count'] <= n for x in routes[i+1:]),
                'Route anchor is not the latest prefix decision')
        order = r['result']['order']
        require(len(order) >= 2 and order[0][0] == 'source' and
                r['source_channels'][order[0][1]] == c and list(order[1]) == source['task'],
                'Anchor is not the actual route second task')
        kind, j = order[1]
        require(type(j) is int and j >= 0, 'Invalid second-task index')
        regions, near, known, cleared = state
        if kind == 'cover':
            require(j < len(r['remaining_covers']), 'Cover anchor index outside route')
            expected = point(r['remaining_covers'][j])
            visited = list(dict.fromkeys(point(a['position']) for a in history[:n]
                                        if a.get('phase') == 'coverage'))
            require(visited == covers[:len(visited)] and
                    [point(p) for p in r['remaining_covers']] == covers[len(visited):],
                    'Route cover anchor is not an unexecuted fixed station')
        else:
            require(kind == 'source' and j < len(r['source_channels']) and
                    j < len(r['source_positions']), 'Unknown second task')
            next_c = r['source_channels'][j]
            require(next_c != c and next_c in known-cleared, 'Anchor source is not observed live')
            expected = target(next_c, regions, near)
            require(point(r['source_positions'][j]) == expected, 'Stale route source target')
        require(q == expected, 'Anchor differs from route second task position')
    else:
        require(source['kind'] == 'early_service_next_cover', 'Unknown anchor source kind')
        p = source['candidate_prefix']
        require(type(p) is int and 0 <= p <= n, 'Invalid early anchor prefix')
        services = [s for s in params.get('early_service_log', []) if s['channel'] == c and
                    s['after_actual_action_count'] == p and p <= n <= s['end_actual_action_count']]
        require(len(services) == 1, 'Early anchor has no real containing service')
        visited = list(dict.fromkeys(point(a['position']) for a in history[:p]
                                    if a.get('phase') == 'coverage'))
        require(visited == covers[:len(visited)] and len(visited) < len(covers),
                'Early anchor lacks the next fixed coverage station')
        require(q == point(source['next_cover_position']) == covers[len(visited)] and
                all(a['channel'] == c and a.get('phase') != 'coverage' for a in history[p:n]),
                'Early anchor crossed its actual service or changed station')
    return q


def audit_clear_region_prefix(record):
    summary = record['summary']
    params, history = summary['strategy_parameters'], summary['action_history']
    config = params.get('q4_clear_region')
    require(config in {'incoming', 'anchored'}, 'Unknown clear-region configuration')
    wire = [a for a in record['history'] if a['action'] in {'/measure', '/clear'}
            and a['response'].get('accepted') is True]
    require(len(history) == len(wire), 'Accepted wire/phase length mismatch')
    for item, raw in zip(history, wire):
        response = raw['response']
        result = response['measure_result' if item['action'] == 'measure' else 'clear_result']
        require('/'+item['action'] == raw['action'] and item['channel'] == raw['channel'] and
                point(item['position']) == point(raw['position']) and item['result'] == result,
                'Actual action differs from wire')
        close(item['virtual_time_s'], response['virtual_time_s'], 'Wire time mismatch')
        if result == 'direction':
            require(item['bearing_deg'] == response['svd_deg'], 'Wire bearing mismatch')
    services = params.get('early_service_log', [])
    if services:
        from experiments.audit_q4_scheduling import audit_scheduling_prefix
        audit_scheduling_prefix(record)
    covers = [point(p) for p in summary['coverage_points']]
    events = params.get('clear_region_log', [])
    consumed, previous_end = set(), 0
    changed = executed = fallbacks = 0
    saved = wall = 0.
    for e in events:
        n, end, c = e['after_actual_action_count'], e['end_action_count'], e['channel']
        require(type(n) is int and type(end) is int and previous_end <= n <= end <= len(history)
                and end-n <= 1, 'Invalid or overlapping clear event interval')
        require(type(c) is int and 1 <= c <= 20 and e['config'] == config,
                'Invalid channel or changed configuration')
        previous_end = end
        current = point(history[n-1]['position']) if n else (0., 0.)
        original, selected = point(e['original_position']), point(e['selected_position'])
        require(point(e['current_position']) == current, 'Stale current position')
        state = belief(history[:n])
        regions, near, known, cleared = state
        require(c in known-cleared, 'Clear event lacks an observed live source')
        phase = e['phase']
        if phase == 'near_clear':
            require(c in near and original == near[c], 'Original near point differs from accepted near')
            centers, kind, radius, verify = [near[c]], 'near_disk', 14.99998, 14.99999
        else:
            require(phase == 'certified_clear' and c in regions and regions[c].vertices,
                    'Clear lacks observed polygon')
            disk = regions[c].enclosing_disk()
            require(c not in near and disk.radius <= 19.9 and original == tuple(disk.center),
                    'Original certified clear is not the actual ready center')
            centers, kind, radius, verify = list(map(tuple, regions[c].vertices)), 'polygon_vertices', 19.99998, 19.99999
        anchor = anchor_witness(e, params, history, covers, state)
        g, cert = e['geometry'], e['geometry']['certificate']
        require(g['status'] in {'optimized', 'original_best', 'fallback'}, 'Unknown geometry status')
        require(type(cert['passed']) is bool, 'Invalid certificate flag')
        maximum = max(math.dist(selected, p) for p in centers)
        require(maximum <= verify + 1e-8, 'Selected clear lacks conservative safety certificate')
        old_in, new_in = math.dist(current, original), math.dist(current, selected)
        old_obj = old_in + (math.dist(original, anchor) if anchor is not None else 0.)
        new_obj = new_in + (math.dist(selected, anchor) if anchor is not None else 0.)
        require(new_in <= old_in+1e-7 and new_obj <= old_obj+1e-7, 'Incoming or two-segment cost increased')
        if g['status'] == 'fallback':
            require(selected == original and cert['passed'] is False, 'Fallback changed the original clear')
            close(g['proxy_saved_s'], 0., 'Fallback claimed proxy gain')
            fallbacks += 1
        else:
            require(cert['passed'] is True and cert['kind'] == kind, 'Wrong successful certificate kind')
            require(([point(p) for p in cert['vertices']] == centers if kind == 'polygon_vertices'
                     else point(cert['near_point']) == centers[0]), 'Certificate geometry differs from actual prefix')
            close(cert['search_radius_m'], radius, 'Changed search margin')
            close(cert['verification_radius_m'], verify, 'Changed verification margin')
            close(cert['max_distance_m'], maximum, 'Wrong certificate maximum distance')
            for key, value in [('original_incoming_m', old_in), ('selected_incoming_m', new_in),
                               ('original_objective_m', old_obj), ('selected_objective_m', new_obj),
                               ('proxy_saved_s', (old_obj-new_obj)/5.)]:
                close(g[key], value, 'Incorrect '+key)
            require(g['objective'] == ('two_segment' if anchor is not None else 'incoming'), 'Wrong proxy objective')
            require(g['status'] == ('optimized' if selected != original else 'original_best'), 'Status/position mismatch')
            require(type(g['ray_count']) is int and type(g['candidate_count']) is int,
                    'Invalid bounded-search diagnostic types')
            if anchor is None:
                require(g['method'] == 'boundary_candidates' and g['ray_count'] == 0 and
                        1 <= g['candidate_count'] <= 4098 and len(centers) <= 64,
                        'Invalid boundary-enumeration diagnostics')
            else:
                require(g['method'] == 'finite_rays' and 64 <= g['ray_count'] <= 66 and
                        1 <= g['candidate_count'] <= 199 and len(centers) <= 64,
                        'Invalid finite-ray diagnostics')
        for runtime in (g['runtime_s'], e['runtime_s']):
            require(isinstance(runtime, (int, float)) and math.isfinite(runtime) and runtime >= 0., 'Invalid runtime')
        wall += g['runtime_s']
        require(type(e['executed']) is bool and e['executed'] == (end == n+1), 'False executed flag')
        if e['executed']:
            a = history[n]
            require(n not in consumed and a['action'] == 'clear' and a['channel'] == c and
                    a.get('phase') == phase and point(a['position']) == selected == point(e['actual_position'])
                    and a['result'] == e['result'], 'Executed clear differs from selected event')
            before_time = history[n-1]['virtual_time_s'] if n else 0.
            expected_fee = round(new_in/5.*1e6)/1e6+3.+2.*(a['result'] == 'success')
            require(abs(a['virtual_time_s']-before_time-expected_fee) <= 2e-6,
                    'Actual selected-clear cost differs from movement/check/removal')
            consumed.add(n)
            executed += 1
            changed += selected != original
            saved += g['proxy_saved_s']
        else:
            require('actual_position' not in e and 'result' not in e, 'Unexecuted event fabricated a response')
            matches = [s for s in services if s['channel'] == c and s['interrupted'] and
                       s['after_actual_action_count'] <= n == s['end_actual_action_count']]
            if matches:
                require(len(matches) == 1, 'Ambiguous interrupted clear service')
                s = matches[0]
                t0 = history[s['after_actual_action_count']-1]['virtual_time_s'] if s['after_actual_action_count'] else 0.
                now = history[n-1]['virtual_time_s'] if n else 0.
                next_fee = math.ceil(new_in/5.*1e6)/1e6+5.
                require(now+next_fee > t0+s['budget_s'], 'Unexecuted clear fits actual service budget')
            else:
                require(n == len(history) and (summary.get('completion_reason') in
                        {'action_budget', 'real_deadline', 'virtual_budget', 'request_rejected'} or summary.get('error')),
                        'No actual service or terminal reason for omitted clear')
    require(consumed == {i for i, a in enumerate(history)
                         if a['action'] == 'clear' and a.get('phase') in {'certified_clear', 'near_clear'}},
            'Unlogged certified or near clear')
    return dict(passed=True, events=len(events), changed_positions=changed, executed_events=executed,
                fallbacks=fallbacks, proxy_saved_s=saved, decision_wall_s=wall,
                boundary='Actual-prefix safety and anchor; proxy gain counts executed events only, CPU is geometry runtime; no global route optimality claim')


def summarize_clear_region_audits(audits):
    fields = ('events', 'changed_positions', 'executed_events', 'fallbacks', 'proxy_saved_s', 'decision_wall_s')
    require(all(a.get('passed') is True for a in audits), 'Cannot summarize a failed audit')
    return {'records': len(audits), **{key: sum(a[key] for a in audits) for key in fields}}
