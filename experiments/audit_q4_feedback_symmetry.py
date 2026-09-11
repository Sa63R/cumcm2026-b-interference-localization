"""Independent observed-prefix reconstruction of a once-locked D7 cover chain.

The new production symmetry helper is never imported. The original certified
station set is retained, and its two angular cycles reconstruct permutations.
"""
import hashlib
import math
from pathlib import Path

from localization import CandidateRegion
from planning.q4_directional_cover import certified_cover_points
from experiments.audit_q4_clear_before_probe import require, point, close, terminal, audit_clear_before_probe_prefix
from experiments.audit_q4_joint_continuation import wire_prefix, audit_joint_continuation_prefix
from experiments.audit_q4_scheduling import audit_scheduling_prefix

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACT = {'experiments/audit_q4_clear_before_probe.py': '68c7ef07dca4cc9f7a60351332911c25054c5e0392220830637bfd89518e62c4',
 'experiments/audit_q4_joint_continuation.py': '5a04f00ee3796312b0ef0862fedf1bfefe1f9d57101e43eacbf11c7708db99e8',
 'experiments/audit_q4_scheduling.py': '836d82d30280d24b612ca96367ca7e9b33b0b1b76f441bbfbcf2d674606ff5be',
 'src/planning/cover_symmetry.py': '9b9e79210e0f7280a0b6db95a979d9e513e9bc48e94717d6692d3432b20be491',
 'src/planning/coverage.py': '4c097ba0d009d0954a96089fb867db5bb4d00020d2cb0b408d8d4c297b427000',
 'src/planning/q4_directional_cover.py': 'e863b3f0fb83e1deba8d1a24c4b929b8fe1caca9927f8b99e514b1deea6da5c1',
 'src/strategies/q4_clear_before_probe.py': '641e8b0e4e6cce7b6445e88117d08ac23bd073487dfb46b87e903330f678ac69',
 'src/strategies/q4_cover_search.py': '2bab5ee2f935967670bb7c1107b99a7bba20208372e15aab328a67b15d651404',
 'src/strategies/q4_feedback_symmetry.py': '48a5b0cf7d82a7891882fe43137e014306fc4ed9ae646b2a3e299c01d5919510',
 'src/strategies/q4_joint_continuation.py': '277480e9c22b0bd983a2971ad04fc97f8ff9e3232ed8198445b9d04dc95e024e',
 'src/strategies/q4_joint_visibility.py': 'eb922aae8052b48c9d54cd4b269a088498a3feeddbae254344b6ed73e31ff0e6',
 'src/strategies/q4_r2_scheduling.py': '48d0f20ed0a073d26f177078f54dd1b40a532038aa74bb0de098c81f84509a44',
 'src/strategies/q4_range_pruning.py': 'c8bd010f186ccd991ffa3cb6727ff1d2cc038c2955bc29f27bb9ab851c37a1d2',
 'src/strategies/q4_range_scheduling.py': 'ad238a4e9c75537223d83ba4b180dc2d316ed71b487497f0f0e5a9bc2d72d76d',
 'src/strategies/q4_state_search.py': '8d105d02f3cb97ef99a1ccef98b63e5510da3c9c5000f77ee6401fbb39b9de59',
 'src/strategies/search.py': '3c30ea448db217b2429d89c69d7db1f8e7fafc14c334043c6c1813ba94c44de5'}
CONFIGS = {'bearing_mean', 'early_centers'}


def verify_source_contract():
    require(bool(SOURCE_CONTRACT) and all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == h
            for p, h in SOURCE_CONTRACT.items()), 'Feedback-symmetry source contract differs')


def original_geometry():
    positions, certificate = certified_cover_points('compact_22')
    result = tuple((p.x, p.y) for p in positions)
    require(certificate.get('passed') is True and len(result) == len(set(result)) == 22
            and result[0] == (0., 0.), 'Original compact_22 certificate/identity differs')
    return result, certificate


def cycle_permutation(base, rotation, reflected):
    """Reconstruct a dihedral map by ring angular labels, not rotated nearest points."""
    outer = 1800./math.cos(math.pi/14.)+5.
    rings = {7: {}, 14: {}}
    identities = {}
    for index, p in enumerate(base[1:], 1):
        distance = math.hypot(*p)
        count = 7 if abs(distance-970.) < 1e-7 else 14
        require(abs(distance-(970. if count == 7 else outer)) < 1e-7, 'Unrecognized original radius')
        angle = math.atan2(p[1], p[0]) % math.tau
        label = round(angle*count/math.tau) % count
        require(label not in rings[count], 'Duplicate angular ring label')
        require(abs(math.remainder(angle-label*math.tau/count, math.tau)) < 1e-10, 'Original point off ring angular grid')
        rings[count][label] = index
        identities[index] = count, label
    require(set(rings[7]) == set(range(7)) and set(rings[14]) == set(range(14)), 'Incomplete original angular cycles')
    result = [0]
    for index in range(1, 22):
        count, label = identities[index]
        shifted = ((-label if reflected else label)+rotation*(count//7)) % count
        result.append(rings[count][shifted])
    require(set(result) == set(range(22)), 'D7 map is not a full permutation')
    return result


def route_length(points):
    return math.fsum(math.dist(a, b) for a, b in zip([(0., 0.)]+list(points), points))


def actual_origin_feedback(history):
    prefix = history[:20]
    require(all(a['action'] == 'measure' and a['phase'] == 'coverage' and
                point(a['position']) == (0., 0.) and a['channel'] == i+1 for i, a in enumerate(prefix)),
            'Initial origin scan is not the actual complete canonical channel sequence')
    known, bearings, centers = [], [], []
    for a in prefix:
        if a['result'] in {'direction', 'near'}:
            known.append(a['channel'])
        if a['result'] == 'direction':
            bearings.append((a['channel'], a['bearing_deg']))
            region = CandidateRegion()
            region.observe((0., 0.), a['bearing_deg'])
            circle = region.enclosing_disk()
            centers.append((a['channel'], tuple(circle.center)))
    return known, bearings, centers


def optional_close(actual, expected, message):
    if expected is None:
        require(actual is None, message)
    else:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and abs(actual-expected) <= 1e-9, message)


def check_locked_chain(history, points, route_log):
    visited, channels_at_station = [], set()
    snapshots = {0: ([], set())}
    known = set()
    for index, action in enumerate(history):
        if action['result'] in {'direction', 'near', 'success'}:
            known.add(action['channel'])
        if action['phase'] == 'coverage':
            p = point(action['position'])
            require(action['action'] == 'measure', 'Coverage credit is not a real measurement')
            if not visited or p != visited[-1]:
                require(len(visited) < len(points) and p == points[len(visited)],
                        'Executed coverage is not the selected locked chain')
                visited.append(p)
                channels_at_station = set()
            require(action['channel'] not in channels_at_station, 'Repeated coverage channel at a locked station')
            channels_at_station.add(action['channel'])
        snapshots[index+1] = (list(visited), set(known))
    for event in route_log:
        prefix = event['after_actual_action_count']
        require(type(prefix) is int and 20 <= prefix <= len(history), 'Planner precedes complete origin scan')
        executed, observed = snapshots[prefix]
        done = len(observed) == 16
        expected = [] if done else points[len(executed):]
        require(event['discovery_count_cap'] == done and [point(p) for p in event['remaining_covers']] == expected,
                'Planner retained an old or rearranged remaining chain')
    return visited


def audit_feedback_symmetry_prefix(record):
    verify_source_contract()
    spec = record['spec']; kwargs = spec.get('kwargs', {}); config = kwargs.get('config', 'bearing_mean')
    require(spec['entrypoint'] == 'strategies.q4_feedback_symmetry:run_q4_feedback_symmetry'
            and config in CONFIGS and set(kwargs) <= {'config', 'max_actions', 'max_active_probes', 'max_expansions'},
            'Unknown feedback symmetry spec')
    for key, default, low, high in [('max_actions',20000,2,1000000), ('max_active_probes',6,0,30), ('max_expansions',200,0,10000)]:
        require(type(kwargs.get(key, default)) is int and low <= kwargs.get(key, default) <= high, 'Invalid frozen execution budget')
    summary = record['summary']; params = summary['strategy_parameters']
    require(params['feedback_symmetry_config'] == config and params['feedback_symmetry_limits'] == dict(
        candidate_count=14, mean_minimum_directions=2, mean_minimum_concentration=.5, early_station_count=3,
        early_improvement_tolerance_m=1e-9, route_length_tolerance_m=1e-7), 'Feedback symmetry limits differ')
    view = dict(record, spec=dict(entrypoint='strategies.q4_joint_continuation:run_q4_joint_continuation',
                                  kwargs={**kwargs, 'config':'after_active_miss_optical'}))
    inherited_r8 = audit_clear_before_probe_prefix(view)
    inherited_r12 = audit_joint_continuation_prefix(view)
    inherited_schedule = audit_scheduling_prefix(record)
    history, before = wire_prefix(record)
    base, certificate = original_geometry()
    report_points = [point(p) for p in summary['coverage_points']]
    require(len(report_points) == 22 and set(report_points) == set(base)
            and summary['coverage_points_total'] == 22, 'Reported route changes the certified station set')
    declared = params['directional_cover_certificate']
    for key in ('kind', 'passed', 'status', 'station_sha256', 'station_count', 'arena_radius', 'reception_radius',
                'range_margin_m', 'orientation_margin_m', 'profile', 'full_leaf_certificate_sha256'):
        require(declared[key] == certificate[key], 'Original coverage certificate identity changed: '+key)
    events = params['feedback_symmetry_log']
    known, bearings, centers = actual_origin_feedback(history)
    if len(history) < 20:
        require(not events and report_points == list(base) and terminal(summary, len(history), history)
                and summary['coverage_points_visited'] == 0, 'Partial origin scan fabricated a selection/coverage credit')
        return dict(passed=True, events=0, changed_routes=0, initial_measurements=len(history),
                    status='origin_interrupted', source_contract=SOURCE_CONTRACT, inherited_r8=inherited_r8,
                    inherited_r12=inherited_r12, inherited_scheduling=inherited_schedule)
    require(len(events) == 1, 'Missing or repeated origin feedback selection')
    require(all(e['after_actual_action_count'] >= 20 for e in params['joint_visibility_resolver_log'])
            and all(e['after_actual_action_count'] >= 20 for e in params['early_service_log']),
            'Source resolver/service preceded the complete origin scan')
    event = events[0]
    require(event['config'] == config and event['origin_scan_start'] == 0 and event['origin_scan_end'] == 20
            and event['after_actual_action_count'] == event['end_actual_action_count'] == 20
            and event['coverage_points_visited'] == 1 and point(event['current_position']) == before[20][0]
            and event['current_channel'] == before[20][1], 'Selection does not follow the single complete origin scan')
    require(event['known_channels'] == known and event['bearings'] == [dict(channel=c, bearing_deg=a) for c,a in bearings],
            'Selection bearings/known channels are not actual first-scan feedback')
    require(event['positive_centers'] == [dict(channel=c, center=list(p)) for c,p in centers],
            'Selection centers are not the canonical first-scan regions')
    require([point(p) for p in event['original_full_points']] == list(base), 'Wrong original certified chain')
    budget = event['budget']
    enters = [a for a in record['history'] if a['action'] == '/enter' and a['response'].get('accepted') is True]
    require(len(enters) <= 1 and budget['policy_actions_before_scan'] == len(enters)
            and budget['policy_actions_after_scan'] == len(enters)+20
            and budget['max_actions'] == kwargs.get('max_actions',20000)
            and len(enters)+20 <= budget['max_actions']-1, 'Origin scan action budget mismatch')
    close(budget['virtual_before_scan'], 0., 'Origin starts after unrecorded paid actions')
    close(budget['virtual_after_scan'], before[20][2], 'Origin scan fee mismatch')
    if enters and 'max_virtual_duration_s' in enters[0]['response']:
        require(budget['max_virtual_duration_s'] == enters[0]['response']['max_virtual_duration_s'], 'Session virtual limit differs from accepted enter')
    for value in (budget['remaining_real_s'], budget['max_virtual_duration_s']):
        require(value is None or type(value) in (float,int) and math.isfinite(value), 'Nonfinite recorded runtime guard')
    maximum = min(360000., budget['max_virtual_duration_s']) if budget['max_virtual_duration_s'] is not None else 360000.
    require(before[20][2] <= maximum-1e-6, 'Complete origin scan exceeds its virtual budget')
    require(type(event['decision_wall_s']) in (float,int) and math.isfinite(event['decision_wall_s']) and event['decision_wall_s'] >= 0,
            'Invalid recorded selection runtime')
    sine = math.fsum(math.sin(math.radians(a)) for _,a in bearings)
    cosine = math.fsum(math.cos(math.radians(a)) for _,a in bearings)
    norm = math.hypot(sine, cosine)
    rho = norm/len(bearings) if bearings else 0.
    mean = math.degrees(math.atan2(sine, cosine)) % 360. if norm > 1e-15 else None
    require(event['bearing_count'] == len(bearings) and event['score_units'] == ('degrees' if config == 'bearing_mean' else 'meters'),
            'Wrong score quantity/units')
    optional_close(event['concentration'], rho, 'Wrong directional concentration')
    optional_close(event['mean_bearing_deg'], mean, 'Wrong circular bearing mean')
    candidates = event['candidates']
    require(len(candidates) == 14, 'Missing/repeated finite candidate')
    orders, scores = [], []
    for identifier, candidate in enumerate(candidates):
        rotation, reflected = identifier % 7, identifier >= 7
        order = cycle_permutation(base, rotation, reflected)
        route = [base[i] for i in order]
        require(type(candidate['id']) is int and type(candidate['rotation_index']) is int
                and all(type(i) is int for i in candidate['permutation'])
                and candidate['id'] == identifier and candidate['rotation_index'] == rotation
                and type(candidate['reflected']) is bool and candidate['reflected'] == reflected
                and candidate['permutation'] == order, 'Candidate is not its original-coordinate D7 permutation')
        require(abs(route_length(route)-route_length(base)) <= 1e-7, 'D7 pure-cover length changed')
        close(candidate['route_length_m'], route_length(route), 'Candidate pure-cover length mismatch')
        if config == 'bearing_mean':
            angle = math.degrees(math.atan2(route[1][1], route[1][0])) % 360.
            score = abs((angle-mean+180.) % 360.-180.) if mean is not None else None
        else:
            score = math.fsum(min(math.dist(center,p) for p in route[1:4]) for _,center in centers)
        optional_close(candidate['score'], score, 'Candidate score does not follow actual feedback')
        orders.append(order); scores.append(score)
    selected, status = 0, 'identity_best'
    if len(known) >= 16: status = 'source_count_cap'
    elif config == 'bearing_mean' and len(bearings) < 2: status = 'insufficient_directions'
    elif config == 'bearing_mean' and rho < .5: status = 'low_concentration'
    elif config == 'early_centers' and not centers: status = 'no_positive_regions'
    else:
        best = min(range(14), key=lambda i:(scores[i],i))
        if best != 0 and (config == 'bearing_mean' or scores[best] < scores[0]-1e-9):
            selected, status = best, 'selected'
    require(type(event['selected_id']) is int and all(type(i) is int for i in event['selected_permutation'])
            and event['selected_id'] == selected and event['selected_permutation'] == orders[selected]
            and event['status'] == status, 'Chosen chain violates score/gate/identity tie rule')
    optional_close(event['original_score'], scores[0], 'Wrong identity score')
    optional_close(event['selected_score'], scores[selected], 'Wrong selected score')
    expected_route = [base[i] for i in orders[selected]]
    require([point(p) for p in event['selected_full_points']] == expected_route == report_points,
            'Selected/report chain differs from chosen original-coordinate permutation')
    visited = check_locked_chain(history, report_points, params['chain_route_log'])
    require(not any(a['phase'] == 'coverage' and point(a['position']) == (0.,0.) for a in history[20:]),
            'Origin was scanned a second time')
    require(summary['coverage_points_visited'] in (len(visited), len(visited)-1)
            and (summary['coverage_points_visited'] == len(visited) or terminal(summary,len(history),history)),
            'Visited coverage count lacks a complete or interrupted actual scan')
    return dict(passed=True, events=1, changed_routes=int(selected != 0), initial_measurements=20,
                selected_id=selected, status=status, source_contract=SOURCE_CONTRACT,
                inherited_r8=inherited_r8, inherited_r12=inherited_r12, inherited_scheduling=inherited_schedule,
                boundary='Same certified coordinate set, actual first-scan feedback and locked chain; finite proxy only, no optimality or inferred coverage. Historical wall-clock guard is not independently recoverable.')
