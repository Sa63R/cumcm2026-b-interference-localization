"""Bind the independent visibility certificate to accepted observation prefixes.

No strategy or planner imports, and no source-truth input. Optical grid coverage
uses exact edge/strip intersections of the rotated polygon, not point samples.
"""
from fractions import Fraction as F
import math

from localization import CandidateRegion
from experiments.audit_q4_clear_before_probe import require, point, close, terminal
from experiments.audit_q4_joint_visibility import (
    audit_joint_visibility_certificate, convex_polygon, inside, rational)


def points(values):
    return [point(p) for p in values]


def boundary_exclusions(original, output):
    outer = convex_polygon(points(output))
    return [j for j, p in enumerate(points(original)) if not inside(rational(p), outer)]


def wire_prefix(record):
    h = record['summary']['action_history']
    wire = [a for a in record['history'] if a['response'].get('accepted') is True
            and a['action'] in {'/measure', '/clear'}]
    require(len(wire) == len(h), 'Accepted wire/history length mismatch')
    current, tuned, elapsed = (0., 0.), 1, 0.
    before = [(current, tuned, elapsed)]
    for a, w in zip(h, wire):
        require(a['action'] in {'measure', 'clear'} and w['action'] == '/'+a['action'], 'Wire action mismatch')
        p = point(a['position'])
        require(p == point(w['position']) and type(a['channel']) is int and
                1 <= a['channel'] <= 20 and a['channel'] == w['channel'], 'Wire point/channel mismatch')
        key = 'measure_result' if a['action'] == 'measure' else 'clear_result'
        require(a['result'] == w['response'][key], 'Wire feedback mismatch')
        if a['result'] == 'direction':
            require(a['bearing_deg'] == w['response']['svd_deg'], 'Wire bearing mismatch')
        fee = round(math.dist(current, p)/5.*1e6)/1e6
        if a['action'] == 'measure':
            require(a['result'] in {'direction', 'near', 'no_signal'}, 'Invalid measure feedback')
            fee += 5.+int(tuned != a['channel'])
            tuned = a['channel']
        else:
            require(a['result'] in {'success', 'no_target_in_range'}, 'Invalid optical feedback')
            fee += 3.+2.*(a['result'] == 'success')
        close(a['virtual_time_s'], elapsed+fee, 'Accepted action cost mismatch')
        close(a['virtual_time_s'], w['response']['virtual_time_s'], 'Wire time mismatch')
        current, elapsed = p, a['virtual_time_s']
        before.append((current, tuned, elapsed))
    return h, before


def observations(h, end, channel):
    region, positive, negative, near, cleared = CandidateRegion(), [], [], None, False
    for a in h[:end]:
        if a['channel'] != channel:
            continue
        p = point(a['position'])
        if a['action'] == 'measure':
            if a['result'] == 'direction':
                region.observe(p, a['bearing_deg'])
                positive.append(p)
            elif a['result'] == 'near':
                near = p
            else:
                negative.append(p)
        elif a['result'] == 'success':
            cleared = True
    return region, positive, negative, near, cleared


def verify_grid(vertices, bearing, start, grid):
    """Every closed 28m strip intersection is covered by contiguous lattice cells.

    Endpoint/edge intersections use Fractions after the same floating rotation.
    A 1e-7m boundary allowance is absorbed by the 20m optical disk's >0.2m
    slack relative to a 28m square's half-diagonal; it never removes an interior
    cell. This also accepts harmless floating inclusion of an adjacent cell.
    """
    require(vertices and grid and math.isfinite(bearing), 'Empty or invalid optical grid')
    angle = math.radians(bearing)
    co, si = math.cos(angle), math.sin(angle)
    rotated = [(F(x*co+y*si), F(-x*si+y*co)) for x, y in points(vertices)]
    result = points(grid)
    require(len(set(result)) == len(result), 'Repeated optical grid point')
    cells = {}
    for p in result:
        x, y = p[0]*co+p[1]*si, -p[0]*si+p[1]*co
        col, row = round(x/28.-.5), round(y/28.-.5)
        require(abs(x-(col+.5)*28.) < 1e-7 and abs(y-(row+.5)*28.) < 1e-7,
                'Optical point is not on the frozen 28m rotated lattice')
        require((row, col) not in cells, 'Duplicated optical cell')
        cells[row, col] = p
    lo, hi = min(p[1] for p in rotated), max(p[1] for p in rotated)
    allowance = F(1e-7)
    require(math.sqrt(2)*(14.+float(allowance)) < 20., 'Grid allowance exceeds optical radius')
    used_rows = set()
    for row in range(math.floor(lo/28), math.floor(hi/28)+1):
        low, high = F(row*28), F((row+1)*28)
        cloud = {p for p in rotated if low <= p[1] <= high}
        for a, b in zip(rotated, rotated[1:]+rotated[:1]):
            if a[1] == b[1]:
                continue
            for y in (low, high):
                if min(a[1], b[1]) <= y <= max(a[1], b[1]):
                    t = (y-a[1])/(b[1]-a[1])
                    cloud.add((a[0]+t*(b[0]-a[0]), y))
        if not cloud:
            continue
        xmin, xmax = min(p[0] for p in cloud), max(p[0] for p in cloud)
        columns = sorted(col for r, col in cells if r == row)
        # At an exact upper row boundary, the closed row below also covers the
        # degenerate slice. Interior positive-width strips require their own row.
        if not columns and all(abs(p[1]-low) <= allowance for p in cloud):
            columns = sorted(col for r, col in cells if r == row-1)
        require(columns and columns == list(range(columns[0], columns[-1]+1)),
                'Optical grid omits a strip or interior column')
        require(F(columns[0]*28)-allowance <= xmin and
                F((columns[-1]+1)*28)+allowance >= xmax, 'Optical grid fails whole-strip coverage')
        used_rows.add(row)
    # The complete shared nearest-neighbour order is observable and fixed.
    remaining = list(result)
    current = point(start)
    for actual in result:
        expected = min(remaining, key=lambda p: (math.dist(current, p), p[0], p[1]))
        require(actual == expected, 'Grid order differs from original nearest traversal')
        remaining.remove(actual)
        current = actual
    return len(cells)


def stopped(summary, h, before, end, channel, action, position):
    if terminal(summary, end, h):
        return True
    current, tuned, now = before[end]
    cost = math.ceil(math.dist(current, point(position))/5.*1e6)/1e6+5.
    if action == 'measure':
        cost += channel != tuned
    services = summary['strategy_parameters'].get('early_service_log', [])
    return any(s['channel'] == channel and s.get('interrupted') is True and
               s['end_actual_action_count'] == end and s['after_actual_action_count'] <= end and
               now+cost > before[s['after_actual_action_count']][2]+s['budget_s']
               for s in services)


def probe_formula(region, bearing, current, h, n, channel):
    circle = region.enclosing_disk()
    cx, cy = circle.center
    angle = math.radians(bearing)
    co, si = math.cos(angle), math.sin(angle)
    radius = min(180., max(25., circle.radius*.5))
    offsets = [(0., 0.), (-si*radius, co*radius), (si*radius, -co*radius),
               (co*radius, si*radius), (-co*radius, -si*radius)]
    candidates = [(cx+dx, cy+dy) for dx, dy in offsets]
    known = {tuple(round(v, 6) for v in point(a['position'])) for a in h[:n]
             if a['action'] == 'measure' and a['channel'] == channel}
    fresh = [i for i, p in enumerate(candidates) if tuple(round(v, 6) for v in p) not in known]
    selected = 0 if 0 in fresh else min(fresh, key=lambda i: math.dist(current, candidates[i])) if fresh else None
    return candidates, fresh, selected


def canonical_first_grid_point(vertices, bearing, current):
    """Reconstruct the original optical grid using edge/strip intersections."""
    angle = math.radians(bearing)
    co, si = math.cos(angle), math.sin(angle)
    rotated = [(F(x*co+y*si), F(-x*si+y*co)) for x, y in points(vertices)]
    centers = []
    for row in range(math.floor(min(p[1] for p in rotated)/28), math.floor(max(p[1] for p in rotated)/28)+1):
        low, high = F(row*28), F((row+1)*28)
        cloud = {p for p in rotated if low <= p[1] <= high}
        for a, b in zip(rotated, rotated[1:]+rotated[:1]):
            if a[1] == b[1]:
                continue
            for y in (low, high):
                if min(a[1], b[1]) <= y <= max(a[1], b[1]):
                    cloud.add((a[0]+(y-a[1])/(b[1]-a[1])*(b[0]-a[0]), y))
        if cloud:
            for col in range(math.floor(min(p[0] for p in cloud)/28), math.floor(max(p[0] for p in cloud)/28)+1):
                x, y = (col+.5)*28., (row+.5)*28.
                centers.append((x*co-y*si, x*si+y*co))
    require(centers, 'Canonical grid unexpectedly empty')
    return min(centers, key=lambda p: (math.dist(current, p), p[0], p[1]))


def audit_joint_visibility_prefix(record):
    summary = record['summary']
    params = summary['strategy_parameters']
    config = params['joint_visibility_config']
    require(config in {'probe', 'probe_optical'}, 'Unknown joint visibility configuration')
    if 'spec' in record:
        require(config == record['spec'].get('kwargs', {}).get('config', 'probe'), 'Config differs from frozen spec')
    h, before = wire_prefix(record)
    epochs = params['joint_visibility_resolver_log']
    probes, grids = params['joint_visibility_probe_log'], params['joint_visibility_grid_log']
    terminal_clears = params.get('joint_visibility_terminal_clear_log', [])
    states, starts, ends, by_id = {}, {}, {}, {}
    counters = dict(epochs=len(epochs), refined_epochs=0, fallback_epochs=0, probe_decisions=len(probes),
        executed_probes=0, auxiliary_clear_attempts=0, auxiliary_clear_successes=0,
        optical_epochs=len(grids), optical_attempts=0, terminal_clear_events=len(terminal_clears),
        canonical_fallbacks=0, decision_wall_s=0.)
    previous_end = 0
    for index, e in enumerate(epochs):
        n, end, c = e['after_actual_action_count'], e['end_actual_action_count'], e['channel']
        require(e['id'] == index and type(n) is int and type(end) is int and
                previous_end <= n <= end <= len(h) and type(c) is int and 1 <= c <= 20,
                'Invalid resolver prefix/order/channel')
        require(all(a['channel'] == c for a in h[n:end]), 'Resolver interval contains another source')
        previous_end = end
        canonical, positive, negative, near, cleared = observations(h, n, c)
        expected_skip = ('already_cleared' if cleared else 'no_canonical_positive_region'
            if not canonical.observations or not canonical.vertices else 'canonical_ready'
            if near is not None or canonical.enclosing_disk().radius <= 19.9 else
            'no_negative_evidence' if not negative else None)
        aux, evidence = None, e['helper_evidence']
        if expected_skip is not None:
            require(e['skip_reason'] == expected_skip and evidence is None and e['initial_aux_vertices'] is None,
                    'Incorrect resolver skip or evidence created without eligible prefix')
        else:
            require(evidence is not None, 'Eligible resolver lacks one helper certificate')
            require(points(evidence['canonical_vertices']) == list(map(tuple, canonical.vertices)) and
                    points(evidence['positive_positions']) == positive and points(evidence['negative_positions']) == negative,
                    'Helper inputs differ from actual entry prefix')
            checked = audit_joint_visibility_certificate(evidence)
            if checked['fallback']:
                require(e['skip_reason'] == 'helper_fallback' and e['initial_aux_vertices'] is None,
                        'Fallback helper used a reduced auxiliary region')
                counters['fallback_epochs'] += 1
            else:
                excluded = boundary_exclusions(canonical.vertices, evidence['output_vertices'])
                require(evidence['old_vertices_excluded'] == excluded,
                        'Reported boundary reduction differs from exact old-vertex inclusion')
                if not excluded:
                    require(e['skip_reason'] == 'no_boundary_reduction' and e['initial_aux_vertices'] is None,
                            'No-boundary-reduction helper must preserve canonical policy')
                else:
                    require(e['skip_reason'] is None and points(e['initial_aux_vertices']) == points(evidence['output_vertices']),
                            'Initial auxiliary region differs from audited output')
                    aux = canonical.copy()
                    aux.vertices, aux._circle = tuple(points(e['initial_aux_vertices'])), None
                    counters['refined_epochs'] += 1
        updates = []
        state = {n: aux.copy() if aux is not None else None}
        for j in range(n, end):
            a = h[j]
            if aux is not None and a['action'] == 'measure' and a['result'] == 'direction':
                aux.observe(a['position'], a['bearing_deg'])
                updates.append(dict(after_actual_action_count=j+1, position=list(point(a['position'])),
                    result='direction', bearing_deg=a['bearing_deg'], aux_vertices=[list(p) for p in aux.vertices]))
                if not aux.vertices:
                    aux = None
            state[j+1] = aux.copy() if aux is not None else None
        require(e['aux_updates'] == updates, 'Auxiliary update is missing, fabricated, or uses future/nonpositive feedback')
        require(e['status'] in {'cleared', 'unresolved', 'interrupted'}, 'Unfinished resolver event')
        successes = [a for a in h[n:end] if a['action'] == 'clear' and a['result'] == 'success']
        if e['status'] == 'cleared':
            require(cleared or len(successes) == 1 and h[end-1] == successes[0], 'Resolver success lacks last real clear')
        elif e['status'] == 'unresolved':
            require(not successes, 'Unresolved event contains success')
        else:
            require(e.get('interruption_type'), 'Interruption lacks exception metadata')
        require(math.isfinite(e['decision_wall_s']) and e['decision_wall_s'] >= 0., 'Invalid resolver decision CPU')
        counters['decision_wall_s'] += e['decision_wall_s']
        states[index], starts[index], ends[index], by_id[index] = state, n, end, e

    used_probe_actions, used_clears, used_grids, decision_keys = set(), set(), set(), set()
    next_index, previous_probe_end = {}, {}
    probe_budget = record.get('spec', {}).get('kwargs', {}).get('max_active_probes', 6)

    def context(e):
        i, c, n, end = e['resolver_id'], e['channel'], e['after_actual_action_count'], e['end_actual_action_count']
        require(i in by_id and c == by_id[i]['channel'] and type(n) is int and type(end) is int
                and starts[i] <= n <= end <= ends[i], 'Decision is outside its resolver prefix')
        aux = states[i][n]
        require(aux is not None and points(e['aux_vertices']) == list(map(tuple, aux.vertices)),
                'Decision uses stale or fabricated auxiliary region')
        require(math.isfinite(e['decision_wall_s']) and e['decision_wall_s'] >= 0., 'Invalid decision CPU')
        counters['decision_wall_s'] += e['decision_wall_s']
        canonical, _, _, near, cleared = observations(h, n, c)
        require(canonical.observations and near is None and not cleared, 'Decision made for ready-near or cleared source')
        return i, c, n, end, aux, canonical

    for e in probes:
        i, c, n, end, aux, canonical = context(e)
        require((i, n, e['index']) not in decision_keys and type(e['index']) is int and e['index'] >= 0,
                'Duplicate or invalid probe index')
        require(e['index'] == next_index.get(i, 0) and e['index'] < probe_budget
                and n >= previous_probe_end.get(i, starts[i]), 'Probe order or local budget differs')
        next_index[i], previous_probe_end[i] = e['index']+1, end
        decision_keys.add((i, n, e['index']))
        bearing = canonical.observations[0].bearing_deg
        original, _, selected_original = probe_formula(canonical, bearing, before[n][0], h, n, c)
        expected_original = original[selected_original] if selected_original is not None else None
        require((point(e['canonical_point']) if e['canonical_point'] is not None else None) == expected_original,
                'Logged original probe differs from real canonical heuristic')
        circle = aux.enclosing_disk()
        require(point(e['center']) == tuple(circle.center), 'Wrong auxiliary circle center')
        close(e['radius_m'], circle.radius, 'Wrong auxiliary radius')
        actual = h[n:end]
        if e['kind'] == 'clear':
            require(circle.radius <= 19.9 and max(math.dist(circle.center, p) for p in aux.vertices) <= 19.9+1e-9
                    and point(e['point']) == tuple(circle.center) and not e['candidates'] and not e['fresh_indices']
                    and e['selected'] is None and not e['executed_measure'] and not e['r8_clear_before_measure'],
                    'Auxiliary clear lacks full-region certificate')
            if actual:
                require(len(actual) == 1 and actual[0]['action'] == 'clear' and actual[0]['channel'] == c and
                        point(actual[0]['position']) == point(e['point']) and actual[0]['phase'] == 'joint_visibility_clear',
                        'Auxiliary clear action differs')
                success = actual[0]['result'] == 'success'
                require(e['status'] == ('cleared' if success else 'certified_clear_failed'), 'Wrong auxiliary clear outcome')
                counters['auxiliary_clear_attempts'] += 1
                counters['auxiliary_clear_successes'] += success
                used_clears.add(n)
            else:
                require(e['status'] == 'interrupted' and stopped(summary, h, before, end, c, 'clear', e['point']),
                        'Unexecuted auxiliary clear lacks actual service/terminal gate')
            continue
        require(e['kind'] == 'probe' and circle.radius > 19.9, 'Probe used when auxiliary region certifies clear')
        candidates, fresh, selected = probe_formula(aux, bearing, before[n][0], h, n, c)
        require(points(e['candidates']) == candidates and e['fresh_indices'] == fresh and e['selected'] == selected,
                'Auxiliary probe candidate/freshness/choice differs')
        target = candidates[selected] if selected is not None else None
        require((point(e['point']) if e['point'] is not None else None) == target, 'Auxiliary selected point differs')
        if target is None:
            require(not actual and e['status'] == 'no_fresh_probe' and not e['executed_measure']
                    and not e['r8_clear_before_measure'], 'Empty probe set performed an action')
            continue
        require(len(actual) <= 2 and all(a['channel'] == c and point(a['position']) == target for a in actual),
                'Probe physical range differs from selected point')
        measures = [j for j in range(n, end) if h[j]['action'] == 'measure']
        speculative = [j for j in range(n, end) if h[j]['phase'] == 'speculative_clear_before_probe']
        require(bool(measures) == e['executed_measure'] and bool(speculative) == e['r8_clear_before_measure'],
                'Wrong real probe/R8 execution flags')
        require(len(measures) <= 1 and all(h[j]['phase'] == 'active_localization' for j in measures)
                and len(speculative) <= 1 and len(measures)+len(speculative) == len(actual),
                'Probe interval contains unaccounted action')
        if measures:
            require(measures == [end-1] and e['status'] == 'measured', 'Probe was not last real measurement')
            counters['executed_probes'] += 1
            used_probe_actions.update(measures)
        elif actual and actual[-1]['result'] == 'success':
            require(e['status'] == 'cleared_by_r8', 'R8 success was reported as a fictitious probe')
        else:
            require(e['status'] == 'interrupted' and stopped(summary, h, before, end, c, 'measure', target),
                    'Unexecuted probe lacks actual service/terminal gate')

    seen_terminal_epochs = set()
    for e in terminal_clears:
        i, c, n, end, aux, canonical = context(e)
        require(i not in seen_terminal_epochs, 'Repeated terminal auxiliary clear')
        seen_terminal_epochs.add(i)
        bearing = canonical.observations[0].bearing_deg
        require(point(e['canonical_first_point']) == canonical_first_grid_point(
            canonical.vertices, bearing, before[n][0]), 'Terminal clear original canonical point differs')
        circle = aux.enclosing_disk()
        require(point(e['center']) == tuple(circle.center) and circle.radius <= 19.9 and
                max(math.dist(circle.center, p) for p in aux.vertices) <= 19.9+1e-9,
                'Terminal auxiliary clear lacks full-region certificate')
        close(e['radius_m'], circle.radius, 'Wrong terminal auxiliary radius')
        actual = h[n:end]
        if actual:
            require(len(actual) == 1 and n not in used_clears and actual[0]['action'] == 'clear'
                    and actual[0]['channel'] == c and point(actual[0]['position']) == tuple(circle.center)
                    and actual[0]['phase'] == 'joint_visibility_clear', 'Terminal actual clear differs')
            success = actual[0]['result'] == 'success'
            require(e['status'] == ('cleared' if success else 'certified_clear_failed'), 'Wrong terminal clear outcome')
            used_clears.add(n)
            counters['auxiliary_clear_attempts'] += 1
            counters['auxiliary_clear_successes'] += success
        else:
            require(e['status'] == 'interrupted' and stopped(summary, h, before, end, c, 'clear', circle.center),
                    'Unexecuted terminal clear lacks actual service/terminal gate')

    seen_grid_epochs = set()
    for e in grids:
        i, c, n, end, aux, canonical = context(e)
        require(config == 'probe_optical' and i not in seen_grid_epochs, 'Repeated/disabled auxiliary optical grid')
        seen_grid_epochs.add(i)
        bearing = canonical.observations[0].bearing_deg
        require(e['bearing_deg'] == bearing and e['spacing_m'] == 28. and point(e['start']) == before[n][0],
                'Grid rotation/start/spacing differs from actual prefix')
        grid = points(e['grid'])
        verify_grid(aux.vertices, bearing, before[n][0], grid)
        require(point(e['canonical_first_point']) == canonical_first_grid_point(
            canonical.vertices, bearing, before[n][0]), 'Original canonical grid first point differs')
        require(end-n == e['actual_grid_actions'] and end-n <= len(grid), 'Wrong accepted optical count')
        actual = h[n:end]
        require(all(a['action'] == 'clear' and a['channel'] == c and a['phase'] == 'joint_visibility_optical'
                    and point(a['position']) == grid[j] for j, a in enumerate(actual)), 'Optical action does not follow common grid prefix')
        require(all(a['result'] == 'no_target_in_range' for a in actual[:-1]), 'Optical search continued after success')
        if actual and actual[-1]['result'] == 'success':
            require(e['status'] == 'cleared', 'Optical success status differs')
        elif len(actual) == len(grid):
            require(e['status'] == 'exhausted_without_success', 'Full failed grid lost contradiction marker')
            counters['canonical_fallbacks'] += 1
            p = point(e['canonical_first_point'])
            require((end < len(h) and h[end]['action'] == 'clear' and h[end]['channel'] == c
                     and h[end]['phase'] == 'guaranteed_clearance' and point(h[end]['position']) == p)
                    or stopped(summary, h, before, end, c, 'clear', p), 'Exhausted auxiliary grid omitted canonical fallback')
        else:
            require(e['status'] == 'interrupted' and stopped(summary, h, before, end, c, 'clear', grid[len(actual)]),
                    'Partial optical grid lacks actual service/terminal gate')
        counters['optical_attempts'] += len(actual)
        used_grids.update(range(n, end))
    require(used_clears == {j for j, a in enumerate(h) if a['phase'] == 'joint_visibility_clear'},
            'Unlogged auxiliary certified clear')
    require(used_grids == {j for j, a in enumerate(h) if a['phase'] == 'joint_visibility_optical'},
            'Unlogged auxiliary optical action')
    expected_probes = {j for i in states for j in range(starts[i], ends[i])
        if states[i][j] is not None and h[j]['action'] == 'measure' and h[j]['phase'] == 'active_localization'}
    require(used_probe_actions == expected_probes, 'Unlogged auxiliary probe measurement')
    for c, estimate in summary.get('source_estimates', {}).items():
        canonical = observations(h, len(h), int(c))[0]
        if 'vertices' in estimate:
            require(points(estimate['vertices']) == list(map(tuple, canonical.vertices)), 'Canonical final region changed by auxiliary model')
    require(not params['joint_visibility_model_contradictions'], 'Auxiliary model contradiction; cannot qualify')
    return dict(passed=True, **counters, boundary='Actual-prefix and full optical strip coverage; generic physical, coverage/scheduling and inherited R8 audits remain separate requirements')


def summarize_joint_visibility_audits(items):
    require(all(item.get('passed') is True for item in items), 'Cannot aggregate failed auxiliary audits')
    keys = ('epochs', 'refined_epochs', 'fallback_epochs', 'probe_decisions', 'executed_probes',
        'auxiliary_clear_attempts', 'auxiliary_clear_successes', 'optical_epochs', 'optical_attempts',
        'terminal_clear_events', 'canonical_fallbacks', 'decision_wall_s')
    return dict(records=len(items), **{k: sum(item[k] for item in items) for k in keys})
