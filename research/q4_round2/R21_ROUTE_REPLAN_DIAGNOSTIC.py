"""Bounded old-prefix route diagnostic; no simulator or strategy imports.

Read only summary/action-history fields of already opened R12 development
archives. No evaluation, ground-truth, posterior, or performance case is used.
The first five nonempty resolver endpoints per case, in fixed 621 case order,
are inspected until twenty prefixes have been obtained. Results describe a
frozen visit-all-stations path, not actual future task time or policy gains.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


WORKSPACE = Path(__file__).resolve().parents[3]
R12 = WORKSPACE / 'q4-r12-joint-continuation'
CORE = WORKSPACE / 'q4-round2'
MAX_PREFIXES, MAX_PER_CASE = 20, 5
MAX_REVERSALS, MAX_CHECKS = 20, 4096
TOLERANCE_M = 1e-8


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def length(start, order, points):
    previous, total = start, 0.
    for i in order:
        total += math.dist(previous, points[i])
        previous = points[i]
    return total


def mst_length(points):
    """Euclidean MST: a lower bound only for this fixed visit-all point task."""
    if len(points) < 2:
        return 0.
    remaining = set(range(1, len(points)))
    best = {i: math.dist(points[0], points[i]) for i in remaining}
    total = 0.
    while remaining:
        chosen = min(remaining, key=lambda i: (best[i], i))
        total += best[chosen]
        remaining.remove(chosen)
        for i in remaining:
            best[i] = min(best[i], math.dist(points[chosen], points[i]))
    return total


def two_opt(start, initial, points):
    """Common open route, deterministic best reversal; preserve incumbent."""
    order, checks, reversals = list(initial), 0, 0
    cost = length(start, order, points)
    while reversals < MAX_REVERSALS and checks < MAX_CHECKS:
        chosen, best = None, cost
        for left in range(len(order)):
            for right in range(left + 1, len(order)):
                if checks >= MAX_CHECKS:
                    break
                checks += 1
                candidate = order[:left] + list(reversed(order[left:right + 1])) + order[right + 1:]
                candidate_cost = length(start, candidate, points)
                if candidate_cost < best - TOLERANCE_M:
                    best, chosen = candidate_cost, candidate
            if checks >= MAX_CHECKS:
                break
        if chosen is None:
            break
        order, cost = chosen, best
        reversals += 1
    assert sorted(order) == sorted(initial)
    return order, cost, dict(checks=checks, accepted_reversals=reversals,
                            budget_exhausted=checks >= MAX_CHECKS or reversals >= MAX_REVERSALS)


def prefix_state(history, prefix, points):
    """Rebuild completed fixed-chain scans from actual radio blocks only.

    Endpoints are after a resolver has returned, so no coverage block is partial.
    Each scan must really measure every channel unknown at its entry. Known
    channel skips do not contribute an invented measurement or coverage credit.
    """
    known, cleared, visited, index = set(), set(), [], 0
    position = (0., 0.)
    while index < prefix:
        action = history[index]
        if action['action'] == 'measure' and action['phase'] == 'coverage':
            point = tuple(action['position'])
            station = len(visited)
            assert station < len(points) and point == points[station], 'Not the recorded fixed coverage chain'
            required = set(range(1, 21)) - known
            measured = set()
            while index < prefix:
                item = history[index]
                if item['action'] != 'measure' or item['phase'] != 'coverage' or tuple(item['position']) != point:
                    break
                assert item['channel'] not in measured, 'Repeated channel within coverage block'
                measured.add(item['channel'])
                if item['result'] in ('direction', 'near'):
                    known.add(item['channel'])
                position = tuple(item['position'])
                index += 1
            assert required <= measured, 'Station lacks actual unknown-channel measurements'
            visited.append(station)
            continue
        if action['action'] == 'measure' and action['result'] in ('direction', 'near'):
            known.add(action['channel'])
        if action['action'] == 'clear' and action['result'] == 'success':
            assert action['channel'] not in cleared
            cleared.add(action['channel'])
            known.add(action['channel'])
        if action.get('position') is not None:
            position = tuple(action['position'])
        index += 1
    assert len(known) <= 16 and cleared <= known
    return position, known, cleared, visited


def run():
    rows, inputs = [], {}
    cases = [('development', seed) for seed in range(621001, 621025)]
    cases += [('development-stress', seed) for seed in range(621031, 621045)]
    for stage, seed in cases:
        if len(rows) >= MAX_PREFIXES:
            break
        path = R12 / f'results/q4_joint_continuation/{stage}/records/compact_joint_continuation-{seed}.json.gz'
        raw = path.read_bytes()
        inputs[path.relative_to(WORKSPACE).as_posix()] = digest_bytes(raw)
        summary = json.loads(gzip.decompress(raw))['summary']
        history = summary['action_history']
        points = [tuple(point) for point in summary['coverage_points']]
        assert len(points) == 22 and len(set(points)) == 22
        epochs = sorted(summary['strategy_parameters']['joint_visibility_resolver_log'],
                        key=lambda e: (e['end_actual_action_count'], e['id']))
        used, seen = 0, set()
        for event in epochs:
            prefix = event['end_actual_action_count']
            if prefix <= event['after_actual_action_count'] or prefix in seen:
                continue
            if len(rows) >= MAX_PREFIXES or used >= MAX_PER_CASE:
                break
            seen.add(prefix)
            used += 1
            position, known, cleared, visited = prefix_state(history, prefix, points)
            physical_remaining = list(range(len(visited), 22))
            remaining = [] if len(known) == 16 else physical_remaining
            old_length = length(position, remaining, points)
            chosen, new_length, work = two_opt(position, remaining, points)
            lower = mst_length([position] + [points[i] for i in remaining])
            assert lower <= new_length + 1e-7 <= old_length + 2e-7
            rows.append(dict(seed=seed, stage=stage, prefix=prefix, resolver_id=event['id'],
                resolver_channel=event['channel'], resolver_status=event['status'],
                current_position=list(position), known_channels=sorted(known), cleared_channels=sorted(cleared),
                completed_station_ids=visited, dormant_or_remaining_station_ids=physical_remaining,
                discovery_count_cap=len(known) == 16, active_remaining_ids=remaining,
                proposed_order=chosen, original_path_m=old_length, two_opt_path_m=new_length,
                achievable_frozen_path_saving_s=(old_length-new_length)/5.,
                first_station_changed=bool(remaining and chosen[0] != remaining[0]),
                fixed_point_mst_m=lower, upper_bound_on_frozen_path_saving_s=(old_length-lower)/5.,
                work=work))
    active = [row for row in rows if row['active_remaining_ids']]
    savings = [row['achievable_frozen_path_saving_s'] for row in rows]
    return dict(scope='OLD R12 development actual resolver endpoints only; no policy/scenario/feedback generation',
        protocol=dict(case_order='621001..621024 then 621031..621044', max_prefixes=MAX_PREFIXES,
            first_nonempty_endpoints_per_case=MAX_PER_CASE, max_two_opt_reversals=MAX_REVERSALS,
            max_two_opt_checks=MAX_CHECKS, improvement_tolerance_m=TOLERANCE_M),
        metric='One frozen current-position-to-all-required-stations OPEN path; no return to origin',
        limitations=['Not a Q4 completion-time estimate or expected discovery score',
            'Overlapping prefix savings MUST NOT be summed into a case saving',
            'Future 16-source early discovery, scans, source services and information are omitted',
            'MST is a lower bound only on this fixed visit-all-points path, not on the actual unknown Q4 task'],
        input_sha256=inputs, script_sha256=digest_bytes(Path(__file__).read_bytes()),
        summary=dict(prefixes=len(rows), cases=len({row['seed'] for row in rows}), active_prefixes=len(active),
            capped_prefixes=sum(row['discovery_count_cap'] for row in rows),
            improved_prefixes=sum(value > 1e-7 for value in savings),
            first_station_changed=sum(row['first_station_changed'] for row in rows),
            mean_frozen_path_saving_s=statistics.mean(savings) if savings else None,
            median_frozen_path_saving_s=statistics.median(savings) if savings else None,
            max_frozen_path_saving_s=max(savings, default=0.),
            max_bound_on_frozen_path_saving_s=max((r['upper_bound_on_frozen_path_saving_s'] for r in rows), default=0.),
            max_checks=max((r['work']['checks'] for r in rows), default=0),
            budget_exhausted=sum(row['work']['budget_exhausted'] for row in rows)), rows=rows)


if __name__ == '__main__':
    destination = Path(__file__).with_suffix('.json')
    if destination.exists():
        raise SystemExit('Preserve existing diagnostic; explicit new version required')
    value = run()
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(value['summary']))
