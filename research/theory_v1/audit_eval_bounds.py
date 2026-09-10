"""Audit completed common Q3 eval.gz records; cache shared physical bounds.

Only post-termination ground truth is consumed. No simulator is constructed.
Default scope is validation seeds 6000..6047; --allow-heldout is an explicit
offline-analysis switch for later frozen evaluations, not an online policy API.
"""
from __future__ import annotations

import argparse
from array import array
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import time

from certify_bounds import improved_bound

VERSION = 'q3-physical-disk-dp-and-information-v2'
GEOMETRIC_MARGIN_M = 1e-6


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def exact_open_graph(first, edges):
    """Exact subset DP of supplied graph, not of continuous disk visitation.

    Flat double array uses 8*n*2**n bytes (8 MiB at n=16); traceback reuses
    labels rather than a second table. Graph edges need not be metric.
    """
    n = len(first)
    if not n:
        return 0., []
    if n > 16 or len(edges) != n or any(len(row) != n for row in edges):
        raise ValueError('Expected square graph, n<=16')
    dp = array('d', [math.inf]) * (n * (1 << n))
    for j in range(n):
        dp[(1 << j)*n + j] = first[j]
    for mask in range(1, 1 << n):
        bits = mask
        while bits:
            bit = bits & -bits
            j = bit.bit_length()-1
            previous = mask ^ bit
            bits ^= bit
            if not previous:
                continue
            prior_bits, best, offset = previous, math.inf, previous*n
            while prior_bits:
                last_bit = prior_bits & -prior_bits
                k = last_bit.bit_length()-1
                prior_bits ^= last_bit
                value = dp[offset+k] + edges[k][j]
                if value < best:
                    best = value
            dp[mask*n+j] = best
    mask = (1 << n)-1
    j = min(range(n), key=lambda k: dp[mask*n+k])
    best, reverse = dp[mask*n+j], []
    while mask:
        reverse.append(j)
        previous = mask ^ (1 << j)
        if previous:
            j = min((k for k in range(n) if previous & (1 << k)),
                    key=lambda k: dp[previous*n+k] + edges[k][j])
        mask = previous
    return best, reverse[::-1]


def physical_bounds(sources):
    ordered = sorted(sources, key=lambda s: s['channel'])
    points = [(s['x'], s['y']) for s in ordered]
    first = [max(0., math.hypot(*p)-20-GEOMETRIC_MARGIN_M) for p in points]
    edges = [[0. if i == j else max(0., math.dist(a, b)-40-GEOMETRIC_MARGIN_M)
              for j, b in enumerate(points)] for i, a in enumerate(points)]
    started = time.perf_counter()
    length, order = exact_open_graph(first, edges)
    elapsed = time.perf_counter()-started
    # Visiting the exact source centres in this order is feasible for the
    # clairvoyant physical problem; no second expensive DP is required.
    centre_length = math.hypot(*points[order[0]]) + sum(
        math.dist(points[a], points[b]) for a, b in zip(order, order[1:]))
    clear = 5*len(points)
    return dict(source_route_lower_m=length,
                source_route_order_channels=[ordered[i]['channel'] for i in order],
                physical_clairvoyant_lower_s=length/5+clear,
                feasible_clairvoyant_upper_s=centre_length/5+clear+1e-6,
                physical_bracket_width_s=(centre_length-length)/5+1e-6,
                dp_seconds=elapsed, dp_array_bytes=8*len(points)*(1 << len(points)))


def audit_record(record):
    """Independently replay legal actions, truth consistency and time ledger."""
    row, evaluation = record['row'], record['evaluation']
    if record.get('evaluation_phase') != 'after_policy_termination':
        raise ValueError('Ground truth must be explicitly post-termination')
    if evaluation.get('kind') != 'local_research_only':
        raise ValueError('Only local research ground truth is supported')
    truth = evaluation['ground_truth']
    if truth['problem'] != 3 or truth['case_id'] != row['case_id'] or truth['seed'] != row['seed']:
        raise ValueError('Scenario identity mismatch')
    if digest(truth) != row['case_sha256']:
        raise ValueError('Ground-truth hash does not match recorded case hash')
    sources = {s['channel']: s for s in truth['sources']}
    if not 10 <= len(sources) <= 16 or len(sources) != len(truth['sources']):
        raise ValueError('Source count / unique-channel rule violated')
    for channel, source in sources.items():
        if not isinstance(channel, int) or not 1 <= channel <= 20 or source['orientation_deg'] is not None:
            raise ValueError('Q3 source rule violated')
        if (not all(math.isfinite(source[k]) for k in ('x', 'y', 'reception_radius_m'))
                or math.hypot(source['x'], source['y']) > 1800+1e-7 or not 1000 <= source['reception_radius_m'] <= 1500):
            raise ValueError('Source geometry/radius invalid')
    history = record['history']
    if len(history) != row['action_count'] or len(history) != evaluation['action_count']:
        raise ValueError('Recorded action count disagrees with history')
    cleared, position, tuned, elapsed_us = set(), (0., 0.), 1, 0
    measures = failures = moving_actions = 0
    components = dict(movement_s=0, switching_s=0, detection_s=0, optical_s=0, removal_s=0)
    for index, action in enumerate(history):
        response, kind = action['response'], action['action']
        if response.get('accepted') is not True or action['index'] != index:
            raise ValueError('Expected ordered accepted-action history')
        if kind in ('/measure', '/clear'):
            channel = action['channel']
            if not isinstance(channel, int) or not 1 <= channel <= 20:
                raise ValueError('Illegal action channel')
            q = (action['position']['x'], action['position']['y'])
            if any(not math.isfinite(v) or abs(v) > 2e6 for v in q):
                raise ValueError('Illegal action position')
            movement = round(math.dist(position, q)/5*1e6)
            moving_actions += 1
            components['movement_s'] += movement
            elapsed_us += movement
            source = sources.get(channel) if channel not in cleared else None
            d = math.dist(q, (source['x'], source['y'])) if source else math.inf
            if kind == '/measure':
                measures += 1
                switch = int(channel != tuned)*1000000
                components['switching_s'] += switch
                components['detection_s'] += 5000000
                elapsed_us += switch+5000000
                tuned = channel
                result = response['measure_result']
                expected = ('no_signal' if source is None or d > source['reception_radius_m']
                            else 'near' if d <= 5 else 'direction')
                if result != expected:
                    raise ValueError('Measurement feedback inconsistent with Q3 truth')
                if result == 'direction':
                    true_bearing = math.degrees(math.atan2(source['y']-q[1], source['x']-q[0])) % 360
                    delta = abs((response['svd_deg']-true_bearing+180) % 360-180)
                    if delta > 1.005+1e-8:
                        raise ValueError('Bearing outside conservative rounding envelope')
            else:
                if response['clear_result'] not in ('success', 'no_target_in_range'):
                    raise ValueError('Unknown clear response')
                success = response['clear_result'] == 'success'
                if success != (d <= 20):
                    raise ValueError('Clear result inconsistent with source geometry')
                components['optical_s'] += 3000000
                components['removal_s'] += int(success)*2000000
                elapsed_us += 3000000+int(success)*2000000
                if success:
                    cleared.add(channel)
                else:
                    failures += 1
            position = q
        elif kind not in ('/enter', '/exit'):
            raise ValueError('Unknown action')
        elif (kind == '/enter' and index != 0) or (kind == '/exit' and index != len(history)-1):
            raise ValueError('Invalid session action ordering')
        if abs(response['virtual_time_s']-elapsed_us/1e6) > 2e-6:
            raise ValueError('Per-action virtual-time ledger mismatch')
    if history and history[0]['action'] != '/enter':
        raise ValueError('Missing enter action')
    for k, count in (('source_total', len(sources)), ('cleared_total', len(cleared)),
                     ('measurement_count', measures), ('failed_clear_count', failures)):
        if row[k] != count or evaluation[k] != count:
            raise ValueError(f'Count mismatch: {k}')
    if row['all_cleared'] != (len(cleared) == len(sources)):
        raise ValueError('False all-cleared flag')
    for k, value in components.items():
        if abs(row[k]-value/1e6) > 2e-6 or abs(evaluation['time_breakdown_s'][k]-value/1e6) > 2e-6:
            raise ValueError(f'Time component mismatch: {k}')
    if abs(row['virtual_time_s']-elapsed_us/1e6) > 2e-6:
        raise ValueError('Final total time mismatch')
    if row['successful'] and not (history and row['all_cleared'] and row['completion_certified'] and
                                  row['accepted_exit'] and history[-1]['action'] == '/exit' and not row['errors']):
        raise ValueError('Successful row lacks full completion evidence')
    return sources, moving_actions


def self_check():
    rng = random.Random(635)
    for n in range(1, 8):
        first = [rng.random()*20 for _ in range(n)]
        edges = [[rng.random()*30 for _ in range(n)] for _ in range(n)]
        cost, order = exact_open_graph(first, edges)
        exact = min(first[p[0]]+sum(edges[a][b] for a, b in zip(p, p[1:])) for p in itertools.permutations(range(n)))
        assert abs(cost-exact) < 1e-9 and sorted(order) == list(range(n))
    return 7


def load_cache(path):
    if not path.exists():
        return {}
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved['version'] != VERSION or digest(saved['entries']) != saved['entries_sha256']:
        raise ValueError('Lower-bound cache version/integrity mismatch')
    return saved['entries']


def analyze(paths, cache_path, output, allow_heldout=False):
    checks = self_check()
    cache, rows = load_cache(cache_path), []
    cache_hits = cache_misses = skipped = 0
    started = time.perf_counter()
    identities = set()
    for path in paths:
        # Avoid opening held-out payloads during current development.
        match = re.search(r'case-(\d+)', path.name)
        if match and not allow_heldout and not 6000 <= int(match.group(1)) <= 6047:
            skipped += 1
            continue
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            record = json.load(stream)
        row = record['row']
        if not allow_heldout and not 6000 <= row['seed'] <= 6047:
            raise ValueError('Only validation 6000..6047 allowed without --allow-heldout')
        identity = (row['case_sha256'], row['strategy'])
        if identity in identities:
            raise ValueError('Duplicate case/strategy record supplied')
        identities.add(identity)
        sources, moving_actions = audit_record(record)
        geometry = [[c, sources[c]['x'], sources[c]['y']] for c in sorted(sources)]
        key = digest(dict(version=VERSION, geometry=geometry))
        if key in cache:
            physical = cache[key]
            cache_hits += 1
        else:
            physical = physical_bounds(sources.values())
            cache[key] = physical
            cache_misses += 1
        bounds = improved_bound(physical['source_route_lower_m'], sources)
        rounding = .5e-6*moving_actions+1e-6
        machine = max(0., bounds['lower_bound_continuous_s']-rounding)
        if row['successful'] and row['virtual_time_s']+1e-8 < machine:
            raise AssertionError('Successful trajectory violates conditional all-clear bound')
        if physical['physical_clairvoyant_lower_s'] > physical['feasible_clairvoyant_upper_s']+1e-8:
            raise AssertionError('Physical interval reversed')
        if physical['physical_bracket_width_s'] > 8*len(sources)-4+1e-4:
            raise AssertionError('Physical interval exceeds analytic additive certificate')
        rows.append(dict(case_id=row['case_id'], case_sha256=row['case_sha256'], seed=row['seed'],
                         strategy=row['strategy'], successful=row['successful'],
                         eligible_for_full_clear_ratio=row['successful'],
                         source_total=row['source_total'], virtual_time_s=row['virtual_time_s'],
                         **{**physical, **bounds}, movement_rounding_allowance_s=rounding,
                         conditional_machine_lower_s=machine,
                         time_over_conditional_lower=row['virtual_time_s']/machine if row['successful'] else None,
                         nonnegative_gap_s=max(0., row['virtual_time_s']-machine) if row['successful'] else None,
                         input_file=str(path), input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         geometry_cache_key=key))
        if cache_misses and len(rows) % 8 == 0:
            atomic_json(cache_path, dict(version=VERSION, entries=cache, entries_sha256=digest(cache)))
    if not rows:
        raise ValueError('No in-scope evaluation records')
    atomic_json(cache_path, dict(version=VERSION, entries=cache, entries_sha256=digest(cache)))
    groups = []
    for strategy in dict.fromkeys(r['strategy'] for r in rows):
        all_rows = [r for r in rows if r['strategy'] == strategy]
        valid = [r for r in all_rows if r['eligible_for_full_clear_ratio']]
        groups.append(dict(strategy=strategy, runs=len(all_rows), successes=len(valid), failures=len(all_rows)-len(valid),
                           mean_success_time_s=statistics.mean(r['virtual_time_s'] for r in valid) if valid else None,
                           mean_success_conditional_lower_s=statistics.mean(r['conditional_machine_lower_s'] for r in valid) if valid else None,
                           mean_physical_bracket_width_s=statistics.mean(r['physical_bracket_width_s'] for r in all_rows),
                           spatial_certification_dominates=sum(r['spatial_certification_lower_m'] > r['source_route_lower_m'] for r in all_rows)))
    result = dict(version=VERSION, simulator_requests_sent=False, truth_access='post-termination records only',
                  execution_platform=platform.platform(), python_version=platform.python_version(),
                  audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  bound_source_sha256=hashlib.sha256(Path(__file__).with_name('certify_bounds.py').read_bytes()).hexdigest(),
                  bound_scope='Conditional on policies guaranteeing all-clear over all legal Q3 scenes; per-case empirical success alone does not prove that guarantee.',
                  route_scope='Exact graph DP lower-bounds the continuous clearance-disk route; it does not solve the continuous route exactly.',
                  upper_scope='Source-centre route is feasible only for the clairvoyant physical task, not for unknown-scene discovery.',
                  rounding='Subtract 0.5 microsecond per physical action plus 1 microsecond numeric margin.',
                  n16='N=16 excludes both empty-channel and trajectory-certification terms.',
                  allow_heldout=allow_heldout, exhaustive_dp_oracles=checks,
                  records=len(rows), scenarios=len({r['case_sha256'] for r in rows}),
                  cache_hits=cache_hits, cache_misses=cache_misses, skipped_out_of_scope=skipped,
                  wall_s=time.perf_counter()-started, groups=groups, rows=rows)
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+', help='Common evaluation gz record(s) or directories')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path(__file__).parent/'results/geometry_cache.json')
    parser.add_argument('--allow-heldout', action='store_true', help='Only after the relevant evaluation has been explicitly frozen')
    args = parser.parse_args()
    paths = []
    for path in args.inputs:
        paths.extend(sorted(path.rglob('case-*.json.gz')) if path.is_dir() else [path])
    result = analyze(paths, args.cache, args.output, args.allow_heldout)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows',)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
