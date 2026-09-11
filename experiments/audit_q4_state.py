"""Independent offline Q4 wire, observation-prefix and triangular-cover audit.

No simulator, scenario generator, strategy or experiment runner is imported.
Truth is read only from completed evaluation archives to check the physical
ledger; the separate prefix replay uses real observations only. Optical-grid
misses are legal costs, not automatic run failures. Geometry uses ordinary
floating arithmetic with explicit tolerances, not an interval proof.
"""

from collections import defaultdict
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from localization import CandidateRegion

VERSION = 'q4-independent-wire-positive-prefix-triangle-audit-v1'
COMPONENTS = ('movement_s', 'switching_s', 'detection_s', 'optical_s', 'removal_s')
MARGIN = 1e-5


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def coordinates(value):
    result = (float(value['x']), float(value['y'])) if isinstance(value, dict) else tuple(map(float, value))
    require(len(result) == 2 and all(math.isfinite(v) for v in result), 'Invalid coordinates')
    return result


def _near(a, b, tolerance=2e-6):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def _station_key(p):
    return tuple(round(v, 6) for v in p)


def _region_distance(q, vertices):
    require(bool(vertices), 'Empty source region in pair certificate')
    edges = list(zip(vertices, vertices[1:] + vertices[:1]))
    if len(vertices) >= 3 and all((b[0]-a[0])*(q[1]-a[1])-(b[1]-a[1])*(q[0]-a[0]) >= 0 for a, b in edges):
        return 0.
    distances = []
    for a, b in edges:
        dx, dy = b[0]-a[0], b[1]-a[1]
        size = dx*dx+dy*dy
        t = max(0., min(1., ((q[0]-a[0])*dx+(q[1]-a[1])*dy)/size)) if size else 0.
        distances.append(math.hypot(q[0]-a[0]-t*dx, q[1]-a[1]-t*dy))
    return min(distances)


def triangular_certificate_stations():
    """Independent finite enumeration of 990 m triangle witnesses for the arena.

    A triangle meeting the 1800 m disk contributes all three vertices, including
    external stations. Every point of the triangle is <=990 m from each vertex;
    its barycentric coordinates prove some vertex lies in every closed emission
    half-plane. The lattice-coordinate bounds cover the entire source disk.
    """
    spacing, arena = 990.0, 1800.0
    height = spacing * math.sqrt(3) / 2
    bound = math.ceil(2 * arena / (math.sqrt(3) * spacing)) + 2
    retained = set()
    triangles = 0
    def xy(v):
        return (spacing * (v[0] + v[1] / 2), height * v[1])
    for i in range(-bound, bound):
        for j in range(-bound, bound):
            for indices in (((i, j), (i + 1, j), (i, j + 1)),
                            ((i + 1, j + 1), (i, j + 1), (i + 1, j))):
                points = [xy(v) for v in indices]
                edges = list(zip(points, points[1:] + points[:1]))
                signs = [a[0] * b[1] - a[1] * b[0] for a, b in edges]
                contains_origin = all(s >= -1e-8 for s in signs) or all(s <= 1e-8 for s in signs)
                distances = []
                for a, b in edges:
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    t = max(0., min(1., -(a[0] * dx + a[1] * dy) / (dx * dx + dy * dy)))
                    distances.append(math.hypot(a[0] + t * dx, a[1] + t * dy))
                if contains_origin or min(distances) <= arena + 1e-7:
                    retained.update(points)
                    triangles += 1
    return sorted(retained), triangles


def wire_audit(record):
    row, evaluation = record['row'], record['evaluation']
    require(record.get('evaluation_phase') == 'after_policy_termination', 'Truth lacks after-termination marker')
    require(evaluation.get('kind') == 'local_research_only', 'Only completed local research truth is supported')
    truth = evaluation['ground_truth']
    require(truth['problem'] == row['problem'] == 4, 'Not a Q4 record')
    require(truth['case_id'] == row['case_id'] and truth['seed'] == row['seed'], 'Case identity mismatch')
    require(digest(truth) == row['case_sha256'], 'Truth SHA mismatch')
    sources = {s['channel']: s for s in truth['sources']}
    require(10 <= len(sources) <= 16 and len(sources) == len(truth['sources']), 'Q4 requires 10..16 distinct sources')
    require(any(s['orientation_deg'] is None for s in sources.values()) and
            any(s['orientation_deg'] is not None for s in sources.values()), 'Q4 requires both source types')
    for channel, s in sources.items():
        require(type(channel) is int and 1 <= channel <= 20, 'Illegal source channel')
        p = coordinates((s['x'], s['y']))
        require(math.hypot(*p) <= 1800 + 1e-7, 'Source outside arena')
        require(math.isfinite(s['reception_radius_m']) and 1000 <= s['reception_radius_m'] <= 1500,
                'Invalid reception radius')
        angle = s['orientation_deg']
        require(angle is None or (math.isfinite(angle) and 0 <= angle < 360), 'Invalid source orientation')
    history = record['history']
    require(len(history) == row['action_count'] == evaluation['action_count'], 'Action count mismatch')
    require(bool(history) and history[0]['action'] == '/enter', 'Missing enter')
    position, tuned, elapsed = (0., 0.), 1, 0
    parts = dict.fromkeys(COMPONENTS, 0)
    cleared = set()
    measures = failures = 0
    for index, action in enumerate(history):
        response, kind = action['response'], action['action']
        require(action['index'] == index and response.get('accepted') is True, 'Unordered or unaccepted wire action')
        if kind in ('/measure', '/clear'):
            channel, q = action['channel'], coordinates(action['position'])
            require(type(channel) is int and 1 <= channel <= 20, 'Illegal action channel')
            require(all(abs(v) <= 2_000_000 for v in q), 'Action outside coordinate limits')
            movement = round(math.dist(position, q) / 5 * 1_000_000)
            elapsed += movement
            parts['movement_s'] += movement
            s = sources.get(channel) if channel not in cleared else None
            distance = math.dist(q, (s['x'], s['y'])) if s else math.inf
            if kind == '/measure':
                measures += 1
                switch = int(tuned != channel) * 1_000_000
                parts['switching_s'] += switch
                parts['detection_s'] += 5_000_000
                elapsed += switch + 5_000_000
                tuned = channel
                visible = s is not None and distance <= s['reception_radius_m']
                if visible and s['orientation_deg'] is not None:
                    angle = math.radians(s['orientation_deg'])
                    dot = math.cos(angle) * (q[0] - s['x']) + math.sin(angle) * (q[1] - s['y'])
                    visible = dot >= -1e-12 * max(1., distance)
                expected = 'no_signal' if not visible else 'near' if distance <= 5 else 'direction'
                require(response['measure_result'] == expected, 'Feedback inconsistent with Q4 range/half-plane')
                if expected == 'direction':
                    bearing = response['svd_deg']
                    require(math.isfinite(bearing) and 0 <= bearing < 360, 'Invalid bearing')
                    actual = math.degrees(math.atan2(s['y'] - q[1], s['x'] - q[0])) % 360
                    require(abs((bearing - actual + 180) % 360 - 180) <= 1.005 + 1e-8,
                            'Bearing exceeds conservative angular envelope')
            else:
                success = distance <= 20
                require(response['clear_result'] == ('success' if success else 'no_target_in_range'),
                        'Clear feedback inconsistent with 20 m optical range')
                parts['optical_s'] += 3_000_000
                parts['removal_s'] += int(success) * 2_000_000
                elapsed += 3_000_000 + int(success) * 2_000_000
                if success:
                    cleared.add(channel)
                else:
                    failures += 1
            position = q
        else:
            require((kind == '/enter' and index == 0) or (kind == '/exit' and index == len(history) - 1),
                    'Invalid session action/order')
        require(_near(response['virtual_time_s'], elapsed / 1_000_000), 'Per-action microsecond ledger mismatch')
    for key, value in [('source_total', len(sources)), ('cleared_total', len(cleared)),
                       ('measurement_count', measures), ('failed_clear_count', failures)]:
        require(row[key] == evaluation[key] == value, 'Count mismatch: ' + key)
    all_cleared = len(cleared) == len(sources)
    require(row['all_cleared'] == evaluation['all_cleared'] == all_cleared, 'All-cleared flag mismatch')
    require(_near(row['virtual_time_s'], elapsed / 1_000_000) and
            _near(evaluation['virtual_time_s'], elapsed / 1_000_000), 'Final time mismatch')
    require(row['accepted_exit'] == (history[-1]['action'] == '/exit'), 'Accepted-exit flag differs from wire')
    for key, value in parts.items():
        require(_near(row[key], value / 1_000_000) and
                _near(evaluation['time_breakdown_s'][key], value / 1_000_000), 'Time component mismatch: ' + key)
    if row['successful']:
        require(all_cleared and row['completion_certified'] and row['accepted_exit'] and
                history[-1]['action'] == '/exit' and not row['errors'], 'Success lacks complete evidence')
    return {'source_total': len(sources), 'cleared_total': len(cleared), 'measurement_count': measures,
            'failed_clear_count': failures, 'all_cleared': all_cleared, 'virtual_time_s': elapsed / 1_000_000,
            'time_breakdown_s': {k: v / 1_000_000 for k, v in parts.items()}}


def observation_audit(record):
    """This function does not access evaluation truth or hidden source types."""
    row, summary = record['row'], record.get('summary')
    actual = [a for a in record['history'] if a['action'] in ('/measure', '/clear')]
    reports = summary['action_history'] if summary is not None else [None] * len(actual)
    require(len(actual) == len(reports), 'Strategy/physical action-history length mismatch')
    regions, near = {}, defaultdict(list)
    positive, observed, negatives = defaultdict(set), defaultdict(set), defaultdict(set)
    detected, cleared, checks = set(), set(), []
    parameters = (summary or {}).get('strategy_parameters', {})
    require(not parameters.get('inferred_no_signal_constraints'), 'Q4 audit rejects unsupported omni inference events')
    hull_events = defaultdict(list)
    for event in parameters.get('positive_hull_log', []):
        count = event['after_actual_action_count']
        require(type(count) is int and 0 <= count <= len(actual), 'Hull event references nonexistent prefix')
        hull_events[count].append(event)
    hull_checks = []
    skipped_events, skipped_checks = defaultdict(list), []
    for event in parameters.get('skipped_certified_scans', []):
        count = event['after_actual_action_count']
        require(type(count) is int and 0 <= count <= len(actual), 'Skipped scan references nonexistent prefix')
        skipped_events[count].append(event)
    def check_skips(count):
        for event in skipped_events.pop(count, []):
            channel = event['channel']
            require(channel in detected-cleared, 'Skipped scan does not concern a known live source')
            reason = event['reason']
            if reason == 'near':
                require(bool(near[channel]), 'Skipped near source has no real near observation')
                radius = 5.
            else:
                require(reason == 'enclosing_disk' and bool(regions[channel].vertices), 'Invalid skipped-source certificate')
                circle = regions[channel].enclosing_disk()
                radius = max(math.dist(circle.center, p) for p in regions[channel].vertices)
                require(radius <= 19.9+1e-7 and _near(event['radius_m'], circle.radius, 1e-7),
                        'Skipped scan lacks its claimed observed-region clear certificate')
            # A skipped known-channel scan adds NO real negative station.
            skipped_checks.append({'channel': channel, 'after_actual_action_count': count,
                                   'position': list(coordinates(event['position'])), 'radius_upper_m': radius})
    pair_events, required_seconds, pair_checks = defaultdict(list), {}, []
    for event in parameters.get('directional_pair_log', []):
        count = event['after_actual_action_count']
        require(type(count) is int and 0 <= count <= len(actual), 'Pair event references nonexistent prefix')
        pair_events[count].append(event)

    def check_pairs(count):
        entries = pair_events.pop(count, [])
        require(len(entries) <= 1, 'Multiple pair decisions claim the same action prefix')
        expected_second = required_seconds.pop(count, None)
        if expected_second is not None and count < len(actual):
            require(len(entries) == 1 and entries[0]['role'] == 'second', 'Silent first pair probe was not followed by its second')
        for event in entries:
            channel, index = event['channel'], event['index']
            require(channel in detected-cleared and type(index) is int and 0 <= index < 6,
                    'Pair event lacks a live source or exceeds six active probes')
            if event['role'] == 'second':
                require(expected_second is not None, 'Second pair probe lacks a preceding silent first')
                previous, first, second, previous_index = expected_second
                require(channel == previous and index == previous_index+1 and coordinates(event['first']) == first
                        and coordinates(event['position']) == second, 'Second pair metadata differs from its original pair')
                planned = second
            else:
                require(event['role'] == 'first' and expected_second is None, 'Invalid/abandoned pair decision')
                pair = event['pair']
                if pair is None:
                    require(event.get('selected') is None, 'Missing pair has a selected candidate')
                    pair_checks.append({'channel': channel, 'after_actual_action_count': count, 'role': 'first', 'executed': False})
                    continue
                require(index <= 4, 'New pair does not reserve two of the six probes')
                anchor = coordinates(pair['anchor'])
                theta, epsilon = pair['anchor_bearing_deg'], pair['error_deg']
                region = regions[channel]
                require(any(o.position == anchor and o.bearing_deg % 360 == theta and o.error_deg == epsilon
                            for o in region.observations), 'Pair anchor/bearing was not actually observed')
                require(0 <= epsilon < 10, 'Invalid pair angular envelope')
                raw = _region_distance(anchor, region.vertices)
                scale = max(1., *(abs(v) for v in anchor), *(abs(v) for p in region.vertices for v in p))
                expected_lower = max(5., raw-(1e-7+128*math.ulp(scale)))
                lower, step = pair['distance_lower_bound_m'], pair['step_m']
                require(_near(lower, expected_lower, 1e-6) and lower <= max(5., raw)+1e-7,
                        'Pair distance lower bound exceeds actual conservative region distance')
                require(_near(step, .95*lower*math.cos(math.radians(45+epsilon)), 1e-7), 'Pair step formula mismatch')
                first, second = coordinates(pair['first']), coordinates(pair['second'])
                expected = [tuple(anchor[j]+step*(math.cos(math.radians(theta+sign*45)) if j == 0
                           else math.sin(math.radians(theta+sign*45))) for j in (0, 1)) for sign in (-1, 1)]
                require((math.dist(first, expected[0]) <= 1e-7 and math.dist(second, expected[1]) <= 1e-7) or
                        (math.dist(first, expected[1]) <= 1e-7 and math.dist(second, expected[0]) <= 1e-7),
                        'Pair positions do not match the certified two rays')
                require(_station_key(first) != _station_key(second) and
                        all(_station_key(q) not in observed[channel] for q in (first, second)), 'Pair is not two fresh points')
                current = coordinates(actual[count-1]['position']) if count else (0., 0.)
                require(_near(pair['complete_pair_cost_s'], (math.dist(current, first)+math.dist(first, second))/5+10),
                        'Complete two-probe cost mismatch')
                planned = first
            executed = count < len(actual)
            if executed:
                action = actual[count]
                require(action['action'] == '/measure' and action['channel'] == channel and
                        math.dist(coordinates(action['position']), planned) <= 1e-7, 'Planned pair point is not next real measure')
                outcome = action['response']['measure_result']
                if event['role'] == 'second':
                    require(outcome in ('near', 'direction'), 'Both certified pair points returned no signal')
                elif outcome == 'no_signal':
                    required_seconds[count+1] = (channel, first, second, index)
            else:
                require(not row['successful'], 'Successful run leaves a selected pair probe unexecuted')
            pair_checks.append({'channel': channel, 'after_actual_action_count': count,
                                'role': event['role'], 'executed': executed})
        if expected_second is not None and count == len(actual):
            require(not row['successful'], 'Successful run abandons a required second probe')

    def check_hulls(count):
        for event in hull_events.pop(count, []):
            channel = event['channel']
            require(channel in detected - cleared, 'Hull event does not concern an observed live source')
            require({coordinates(p) for p in event['positive_stations']} == positive[channel],
                    'Hull positive stations differ from real prefix')
            for candidate in event['candidates']:
                weights = candidate['weights']
                require(bool(weights), 'Missing convex-combination witness')
                require(all(math.isfinite(w['weight']) and w['weight'] > 0 and
                            coordinates(w['position']) in positive[channel] for w in weights),
                        'Hull witness uses a nonpositive/unknown station')
                require(abs(math.fsum(w['weight'] for w in weights) - 1) <= 1e-12,
                        'Hull weights do not sum to one')
                q = coordinates(candidate['position'])
                reconstructed = tuple(math.fsum(w['weight'] * coordinates(w['position'])[j] for w in weights)
                                      for j in (0, 1))
                require(math.dist(q, reconstructed) <= 1e-7, 'Hull candidate differs from convex witness')
                require(_station_key(q) not in observed[channel], 'Hull candidate repeats an actual measurement')
            selected = event['selected']
            executed = False
            if selected is not None:
                require(type(selected) is int and 0 <= selected < len(event['candidates']), 'Invalid hull selection')
                q = coordinates(event['position'])
                require(q == coordinates(event['candidates'][selected]['position']), 'Hull selected position mismatch')
                if count < len(actual):
                    action = actual[count]
                    require(action['action'] == '/measure' and action['channel'] == channel and
                            math.dist(coordinates(action['position']), q) <= 1e-7, 'Selected hull probe not next actual action')
                    require(action['response']['measure_result'] in ('near', 'direction'), 'Convex-hull probe lost guaranteed reception')
                    executed = True
                else:
                    require(not row['successful'], 'Successful run ends with an unexecuted selected probe')
            else:
                require(event.get('position') is None, 'Unselected hull event has a chosen position')
            hull_checks.append({'channel': channel, 'after_actual_action_count': count,
                                'candidates_verified': len(event['candidates']), 'executed': executed})
    check_hulls(0)
    check_pairs(0)
    check_skips(0)
    for count, (action, report) in enumerate(zip(actual, reports), 1):
        channel, q, response = action['channel'], coordinates(action['position']), action['response']
        kind = action['action'][1:]
        outcome = response['measure_result' if kind == 'measure' else 'clear_result']
        phase = (report or {}).get('phase', '')
        if report is not None:
            require(report['action'] == kind and report['channel'] == channel and coordinates(report['position']) == q
                    and report['result'] == outcome and _near(report['virtual_time_s'], response['virtual_time_s']),
                    'Strategy/physical action-history content mismatch')
        region = regions.setdefault(channel, CandidateRegion())
        if kind == 'measure':
            observed[channel].add(_station_key(q))
            if channel not in cleared:
                if outcome == 'direction':
                    if report is not None:
                        require(report['bearing_deg'] == response['svd_deg'], 'Reported bearing differs from wire')
                    positive[channel].add(q)
                    detected.add(channel)
                    region.observe(q, response['svd_deg'])
                elif outcome == 'near':
                    near[channel].append(q)
                    detected.add(channel)
                else:
                    negatives[channel].add(_station_key(q))
                    # No Q3 no_signal distance half-plane is applied.
                require(channel not in detected or bool(region.vertices), 'Positive source region became empty')
        else:
            radius = max((math.dist(q, p) for p in region.vertices), default=math.inf)
            radius = min(radius, min((math.dist(q, p) + 5 for p in near[channel]), default=math.inf))
            certified = channel not in cleared and radius <= 20 - MARGIN
            claimed = 'certified_clear' in phase or 'near_clear' in phase
            require(not claimed or (certified and outcome == 'success'), 'Claimed safe clear failed or lacks a prefix certificate')
            checks.append({'action_index': action['index'], 'channel': channel, 'phase': phase,
                           'result': outcome, 'prefix_certified': certified,
                           'radius_upper_m': radius if math.isfinite(radius) else None})
            if outcome == 'success':
                cleared.add(channel)
        check_hulls(count)
        check_pairs(count)
        check_skips(count)
    require(not hull_events, 'Unprocessed hull events')
    require(not pair_events and not required_seconds, 'Unprocessed pair decisions')
    require(not skipped_events, 'Unprocessed skipped-scan decisions')
    stations, triangle_count = triangular_certificate_stations()
    station_keys = {_station_key(p) for p in stations}
    cap = len(cleared) == 16
    absence = [{'channel': c, 'certified': cap or station_keys <= negatives[c],
                'method': '16_actual_successful_clears' if cap else '990m_triangle_vertices_actually_negative',
                'missing_station_count': 0 if cap else len(station_keys - negatives[c]),
                'actual_distinct_negative_stations': len(negatives[c])}
               for c in sorted(set(range(1, 21)) - cleared)]
    terminal = 10 <= len(cleared) <= 16 and not (detected - cleared) and all(a['certified'] for a in absence)
    if row['successful'] or (summary or {}).get('completion_certified_under_model'):
        require(terminal, 'Full-clear claim lacks per-channel actual Q4 cover or sixteen actual clears')
    if (summary or {}).get('completion_reason') == 'source_count_upper_bound_reached':
        require(cap, 'Source cap stop requires 16 clears, not detections')
    return {'detected': len(detected), 'cleared': len(cleared), 'terminal_certified': terminal,
            'source_cap_actual_clears': cap, 'triangle_station_count': len(stations),
            'triangle_count': triangle_count, 'absence': absence, 'clear_attempts': checks,
            'successful_clear_without_prefix_certificate': sum(c['result'] == 'success' and not c['prefix_certified'] for c in checks),
            'legal_failed_optical_attempts': sum(c['result'] == 'no_target_in_range' for c in checks),
            'hull_events': hull_checks, 'pair_events': pair_checks,
            'certified_scans_skipped': skipped_checks, 'inferred_coverage_credits': 0}


def audit_record(record):
    """Keep failed runs and all audit errors; a valid optical miss may pass."""
    row = record.get('row', {})
    result = {k: row.get(k) for k in ('seed', 'strategy', 'case_id', 'case_sha256', 'successful',
                                     'virtual_time_s', 'penalized_time_s', 'failed_clear_count')}
    errors = []
    try:
        require(type(row['successful']) is bool, 'Success flag is not boolean')
        require(_near(row['penalized_time_s'], row['virtual_time_s'] if row['successful'] else 360000.),
                'Incomplete/error run must retain the 360000 second penalty')
    except (KeyError, ValueError, TypeError) as exc:
        errors.append('penalty: ' + str(exc))
    for label, function in (('physical', wire_audit), ('observations', observation_audit)):
        try:
            result[label] = function(record)
        except (KeyError, ValueError, TypeError, IndexError, OverflowError, AssertionError) as exc:
            errors.append(label + ': ' + type(exc).__name__ + ': ' + str(exc))
    result.update(audit_passed=not errors, errors=errors)
    return result


def audit_directory(directory):
    """Read one explicitly named complete stage; no artifact discovery scans."""
    began = time.perf_counter()
    directory = Path(directory).resolve()
    load = lambda name: json.loads((directory / name).read_text(encoding='utf-8'))
    manifest, freeze, summary = load('manifest.json'), load('freeze.json'), load('summary.json')
    global_errors = []
    def check(value, message):
        if not value:
            global_errors.append(message)
    check(digest(manifest) == freeze['manifest_sha256'], 'Frozen manifest SHA mismatch')
    seeds, specs, stage = manifest['seeds'], manifest['specs'], manifest['stage']
    bounds = manifest['protocol'][stage]
    check(manifest['protocol']['problem'] == 4 and seeds == list(range(bounds[0], bounds[1] + 1)),
          'Protocol/stage/seed set mismatch')
    check(len(seeds) == len(set(seeds)) and len(specs) >= 2 and 'triangular' in specs, 'Empty/duplicate/unpaired experiment')
    current_source_changes = []
    with zipfile.ZipFile(directory / 'source.zip') as archive:
        expected_sources = manifest['source_sha256']
        check(len(archive.namelist()) == len(set(archive.namelist())) and set(archive.namelist()) == set(expected_sources),
              'Source ZIP inventory mismatch')
        for name, expected in expected_sources.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                check(False, 'Unsafe source member path')
                continue
            try:
                check(hashlib.sha256(archive.read(name)).hexdigest() == expected, 'Source archive hash mismatch: ' + name)
            except KeyError:
                check(False, 'Source archive member missing: ' + name)
            current = ROOT / name
            if not current.is_file() or sha(current) != expected:
                current_source_changes.append(name)
            # Historical strategies may be edited after their completed run.
            # The two imported replay geometry modules must still be identical.
            if name in ('src/geometry/__init__.py', 'src/localization/__init__.py'):
                check(current.is_file() and sha(current) == expected, 'Replay geometry differs from frozen source: ' + name)
        check({'src/geometry/__init__.py', 'src/localization/__init__.py'} <= set(expected_sources),
              'Frozen sources omit replay geometry identity')
    expected = {f'{label}-{seed}.json.gz' for seed in seeds for label in specs}
    actual = {p.name for p in (directory / 'records').iterdir() if p.is_file()}
    check(actual == expected, 'Record inventory mismatch: missing=' + repr(sorted(expected - actual)) +
          ' extra=' + repr(sorted(actual - expected)))
    rows, reports = [], []
    hashes = defaultdict(dict)
    for seed in seeds:
        for label, spec in specs.items():
            path = directory / 'records' / f'{label}-{seed}.json.gz'
            item = {'seed': seed, 'strategy': label, 'input_file': str(path)}
            try:
                item['input_sha256'] = sha(path)
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    record = json.load(stream)
                row = record['row']
                rows.append(row)
                require(row['seed'] == seed and row['strategy'] == label and row['stage'] == stage,
                        'Filename/stage/record identity mismatch')
                require(record['spec'] == spec, 'Record spec differs from frozen manifest')
                hashes[seed][label] = row['case_sha256']
                item.update(audit_record(record))
            except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
                item.update(audit_passed=False, errors=['archive: ' + type(exc).__name__ + ': ' + str(exc)])
            reports.append(item)
    for seed in seeds:
        check(set(hashes[seed]) == set(specs) and len(set(hashes[seed].values())) == 1, 'Case SHA pairing mismatch: ' + str(seed))
    check(sorted(rows, key=lambda r: (r['seed'], r['strategy'])) == summary['rows'], 'Summary rows differ from actual archives')
    return {'kind': VERSION, 'input_directory': str(directory), 'frozen_manifest_sha256': freeze['manifest_sha256'],
            'input_sha256': {n: sha(directory / n) for n in ('manifest.json', 'freeze.json', 'source.zip', 'summary.json')},
            'auditor_sha256': sha(__file__), 'expected_records': len(expected), 'record_count': len(reports),
            'current_source_changes_since_freeze': current_source_changes,
            'identity_scope': 'Archived source ZIP verified; only imported geometry must match current checkout',
            'audits_passed': sum(r['audit_passed'] for r in reports), 'global_errors': global_errors,
            'all_passed': bool(reports) and not global_errors and all(r['audit_passed'] for r in reports),
            'new_scenario_runs': 0, 'http_requests': 0, 'elapsed_s': time.perf_counter() - began, 'records': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_directory(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('all_passed', 'record_count', 'audits_passed', 'global_errors', 'elapsed_s')}))
    return 0 if result['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
