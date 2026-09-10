"""Offline, standard-library certificates for Q3 information lower bounds.

No simulator client is instantiated. Existing experiments are read, never edited.
Run: python research/theory_v1/certify_bounds.py --output research/theory_v1/results
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def empty_action_switch_bound(initial_channel: bool) -> dict:
    """Necessary circumference coverage; entry switch charged only if measured.

    Small outward angular rounding weakens the relaxation safely. Enumeration
    need only include m <= 7: m >= 7 already costs >= 35, exceeding the feasible
    relaxed m=6 candidate. Failed-clear counts come from the coverage inequality.
    This proves a cost bound, NOT feasibility of six disks covering the arena.
    """
    alpha = 2 * math.asin(1000 / 1800) + 1e-12
    beta = 2 * math.asin(20 / 1800) + 1e-12
    circumference = 2 * math.pi - 1e-12
    candidates = []
    for m in range(8):
        f = max(0, math.ceil((circumference - m * alpha) / beta - 1e-12))
        entry = int(m > 0 and not initial_channel)
        candidates.append(dict(measurements=m, failed_clears=f,
                               entry_switch_s=entry, cost_s=5*m + 3*f + entry))
    winner = min(candidates, key=lambda row: row['cost_s'])
    return dict(bound_s=winner['cost_s'], minimizing_relaxation=winner,
                candidates=candidates, circumference_relaxation_only=True)


def spatial_certification_length_bound(arena=1800.0, reception=1000.0) -> float:
    """If an empty channel must be certified, arena lies in path's r-neighborhood.

    A polygonal path of length L has r-neighborhood area <= pi*r*r + 2*r*L.
    Hence covering area pi*R*R requires L >= pi*(R*R-r*r)/(2*r).
    Applies when N < 16 and all allowed scenes must be cleared, not merely to a
    successful run by a policy lacking such a guarantee. Do not add this length
    to another bound for the same movement: take their maximum.
    """
    if arena <= 0 or reception <= 0:
        raise ValueError('Positive radii required')
    return max(0.0, math.pi * (arena**2 - reception**2) / (2*reception) - 1e-8)


def improved_bound(source_route_lower_m: float, occupied_channels) -> dict:
    channels = frozenset(occupied_channels)
    if not 10 <= len(channels) <= 16 or not channels.issubset(range(1, 21)):
        raise ValueError('Expected 10..16 distinct legal channels')
    if not math.isfinite(source_route_lower_m) or source_route_lower_m < 0:
        raise ValueError('Invalid source route bound')
    n = len(channels)
    empty = set(range(1, 21)) - channels
    spatial = spatial_certification_length_bound() if n < 16 else 0.0
    action_switch = sum(empty_action_switch_bound(c == 1)['bound_s']
                        for c in empty) if n < 16 else 0.0
    movement = max(source_route_lower_m, spatial)
    return dict(source_route_lower_m=source_route_lower_m,
                spatial_certification_lower_m=spatial,
                combined_movement_lower_m=movement,
                successful_clear_action_s=5*n,
                empty_action_and_entry_switch_lower_s=action_switch,
                additional_entry_switch_lower_s=(len(empty) - int(1 in empty)) if n < 16 else 0,
                lower_bound_continuous_s=movement/5 + 5*n + action_switch,
                certification_applied=n < 16)


def exact_two_world_policy(left_probability=0.5, coordinates=None) -> dict:
    """Exact belief-subset DP for an explicitly declared two-world problem.

    One known remaining source/channel, x = +/-1500, radius = 1000, start x=0.
    Only existence feedback is used, so bearing error cannot leak source truth.
    B is a bitmask of still possible worlds; each useful nonterminal action
    strictly shrinks B. Non-informative actions can be omitted by triangle
    inequality and positive action cost. This proves optimality on supplied
    actions. The README separately proves continuous optimality for p=1/2.
    """
    if not 0 < left_probability < 1:
        raise ValueError('Strictly positive probabilities required')
    sources = (-1500.0, 1500.0)
    points = tuple(coordinates or (-1500., -1480., -500., 0., 500., 1480., 1500.))
    probs = (left_probability, 1-left_probability)
    policy = {}

    @lru_cache(None)
    def solve(position, mask):
        mass = sum(probs[i] for i in range(2) if mask & (1 << i))
        values = []
        for q in points:
            move = abs(q-position)/5
            for kind, radius in (('measure', 1000.), ('clear', 20.)):
                hit = sum((1 << i) for i in range(2)
                          if mask & (1 << i) and abs(q-sources[i]) <= radius)
                miss = mask ^ hit
                if kind == 'measure':
                    if not hit or not miss:
                        continue
                    p_hit = sum(probs[i] for i in range(2) if hit & (1 << i)) / mass
                    value = move + 5 + p_hit*solve(q, hit) + (1-p_hit)*solve(q, miss)
                else:
                    if not hit:
                        continue
                    p_hit = sum(probs[i] for i in range(2) if hit & (1 << i)) / mass
                    value = move + p_hit*5 + (1-p_hit)*(3 + (solve(q, miss) if miss else 0))
                values.append((value, kind, q))
        value, kind, q = min(values)
        policy[f'{position:g}|{mask}'] = dict(action=kind, position=q, value_s=value)
        return value

    value = solve(0., 3)
    return dict(expected_optimal_s=value, first_action=policy['0|3'],
                left_probability=left_probability,
                clairvoyant_expected_optimal_s=301.,
                actions=list(points), states_evaluated=solve.cache_info().currsize,
                policy=policy)


def analyze_previous_holdout(directory: Path) -> dict:
    """Apply new terms to frozen source-distance bounds; preserve all old files."""
    report_path, runs_path = directory/'lower_bounds.json', directory/'runs.csv'
    old = json.loads(report_path.read_text(encoding='utf-8-sig'))
    expected_hash = old['input_sha256']['runs.csv']
    actual_hash = hashlib.sha256(runs_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError('Frozen runs.csv hash does not match prior bounds')
    for name, expected in old['analysis_source_sha256_start'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original lower-bound source changed: {name}')
    cases = {}
    for c in old['cases']:
        updated = improved_bound(c['lower_route_length_m'], c['lower_graph_order_channels'])
        cases[c['case_id']] = dict(case_id=c['case_id'], **updated,
                                  old_lower_bound_s=c['common_lower_bound_s'],
                                  increase_s=updated['lower_bound_continuous_s']-c['common_lower_bound_s'])
    with runs_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    checks, groups = [], []
    for r in rows:
        b = cases[r['case_id']]
        # Per-action movement rounding can lose at most half a microsecond.
        # action_count also includes enter/exit; this makes subtraction safer.
        rounding_allowance = int(r['action_count'])*0.5e-6 + 1e-6
        machine_lower = b['lower_bound_continuous_s'] - rounding_allowance
        success = r['successful'] == 'True'
        if success and float(r['virtual_time_s']) + 1e-8 < machine_lower:
            raise AssertionError('New bound exceeds successful run')
        checks.append(dict(case_id=r['case_id'], strategy=r['strategy'],
                           successful=success, actual_s=float(r['virtual_time_s']),
                           safe_machine_lower_s=machine_lower,
                           movement_rounding_allowance_s=rounding_allowance))
    for strategy in dict.fromkeys(r['strategy'] for r in checks):
        selected = [r for r in checks if r['strategy'] == strategy]
        groups.append(dict(strategy=strategy, runs=len(selected),
                           successful_runs=sum(r['successful'] for r in selected),
                           mean_time_s=statistics.mean(r['actual_s'] for r in selected),
                           mean_new_machine_bound_s=statistics.mean(r['safe_machine_lower_s'] for r in selected),
                           mean_time_over_mean_bound=statistics.mean(r['actual_s'] for r in selected)/statistics.mean(r['safe_machine_lower_s'] for r in selected)))
    return dict(input_hashes={'runs.csv': actual_hash, 'lower_bounds.json': hashlib.sha256(report_path.read_bytes()).hexdigest()},
                case_count=len(cases), successful_checks=sum(r['successful'] for r in checks),
                mean_bound_increase_s=statistics.mean(c['increase_s'] for c in cases.values()),
                spatial_term_active_cases=sum(c['spatial_certification_lower_m'] > c['source_route_lower_m'] for c in cases.values()),
                cases=list(cases.values()), checks=checks, groups=groups)


def self_check():
    assert empty_action_switch_bound(True)['bound_s'] == 30
    assert empty_action_switch_bound(False)['bound_s'] == 31
    spatial = spatial_certification_length_bound()
    assert math.isclose(spatial, 1120*math.pi, abs_tol=2e-8)
    c = improved_bound(0., range(1, 11))
    assert math.isclose(c['lower_bound_continuous_s'], 224*math.pi + 360, abs_tol=1e-8)
    n16 = improved_bound(0., range(1, 17))
    assert n16['lower_bound_continuous_s'] == 80
    with_initial_empty = improved_bound(4000., range(2, 12))
    assert with_initial_empty['additional_entry_switch_lower_s'] == 9
    toy = exact_two_world_policy()
    assert math.isclose(toy['expected_optimal_s'], 406., abs_tol=1e-9)
    assert toy['first_action']['action'] == 'measure'
    biased = exact_two_world_policy(.995)
    assert biased['first_action']['action'] == 'clear'
    assert math.isclose(biased['expected_optimal_s'], 303.975, abs_tol=1e-9)
    # Independently derived continuous formula, evaluated against exact DP.
    for p in (.005, .02, .1, .3, .5, .8, .97, .995):
        analytic = min(506-200*p, 306+200*p, 896-595*p, 301+595*p)
        assert math.isclose(exact_two_world_policy(p)['expected_optimal_s'], analytic, abs_tol=1e-9)
    # Nested action sets cannot worsen the optimum of the SAME finite model.
    coarse = exact_two_world_policy(coordinates=(-1480., 1480.))
    assert math.isclose(coarse['expected_optimal_s'], 598.5, abs_tol=1e-9)
    assert coarse['expected_optimal_s'] >= toy['expected_optimal_s']
    return dict(checks='passed', two_world_equal_prior=toy,
                two_world_biased_prior=biased,
                coarse_action_expected_optimal_s=coarse['expected_optimal_s'],
                spatial_lower_m=spatial, clustered_ten_sources=c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    parser.add_argument('--holdout', type=Path, default=ROOT/'results/q3_rollout/holdout')
    args = parser.parse_args()
    result = dict(analysis_kind='q3_information_bounds_v1', simulator_requests_sent=False,
                  theory_version='v1', verification=self_check(),
                  empty_initial=empty_action_switch_bound(True),
                  empty_noninitial=empty_action_switch_bound(False),
                  prior_holdout=analyze_previous_holdout(args.holdout))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'certificates.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(checks='passed', certificate=str(args.output/'certificates.json'),
                          mean_bound_increase_s=result['prior_holdout']['mean_bound_increase_s'],
                          groups=result['prior_holdout']['groups']), ensure_ascii=False))


if __name__ == '__main__':
    main()
