"""Read the 38 opened 621 R12 development logs; do not solve any route/case.

Only saved summary chain-route results and real action prefixes are projected.
No scene evaluation, hidden positions, strategy import, or new scenario is used.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


WORKSPACE = Path(__file__).resolve().parents[3]
R12 = WORKSPACE / 'q4-r12-joint-continuation'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stats(events):
    gaps = [event['proxy_gap_s'] for event in events]
    unresolved = [event for event in events if not event['exact']]
    return dict(calls=len(events), exact_calls=len(events)-len(unresolved), nonexact_calls=len(unresolved),
        cases=len({event['seed'] for event in events}),
        cases_with_nonexact=len({event['seed'] for event in unresolved}),
        nonexact_while_covers_remain=sum(event['remaining_covers'] > 0 for event in unresolved),
        nonexact_after_discovery=sum(event['remaining_covers'] == 0 for event in unresolved),
        budget_hit_calls=sum(event['budget_hit'] for event in events),
        nonexact_budget_hit_calls=sum(event['budget_hit'] for event in unresolved),
        mean_logged_proxy_gap_s=statistics.mean(gaps) if gaps else None,
        max_logged_proxy_gap_s=max(gaps, default=0.),
        mean_nonexact_gap_s=statistics.mean(event['proxy_gap_s'] for event in unresolved) if unresolved else None,
        max_expanded=max((event['expanded'] for event in events), default=0),
        max_logged_runtime_s=max((event['logged_runtime_s'] for event in events), default=0.),
        max_dense_dag_label_bound=max((event['dense_dag_label_bound'] for event in events), default=0))


def run():
    events, per_case, inputs = [], [], {}
    cases = [('development', seed) for seed in range(621001, 621025)]
    cases += [('development-stress', seed) for seed in range(621031, 621045)]
    for stage, seed in cases:
        path = R12 / f'results/q4_joint_continuation/{stage}/records/compact_joint_continuation-{seed}.json.gz'
        raw = path.read_bytes()
        inputs[path.relative_to(WORKSPACE).as_posix()] = sha(raw)
        summary = json.loads(gzip.decompress(raw))['summary']
        params, history = summary['strategy_parameters'], summary['action_history']
        cap = params['max_route_expansions']
        total_cap = params['max_total_route_expansions']
        assert cap == 200 and total_cap == 60000
        local, expanded_total = [], 0
        known, cleared, cursor = set(), set(), 0
        for index, event in enumerate(params['chain_route_log']):
            prefix = event['after_actual_action_count']
            assert cursor <= prefix <= len(history)
            while cursor < prefix:
                action = history[cursor]
                if action['action'] == 'measure' and action['result'] in ('near', 'direction'):
                    known.add(action['channel'])
                elif action['action'] == 'clear' and action['result'] == 'success':
                    cleared.add(action['channel'])
                    known.add(action['channel'])
                cursor += 1
            channels = event['source_channels']
            assert len(channels) == len(set(channels)) and set(channels) <= known-cleared
            n, k = len(channels), len(event['remaining_covers'])
            assert 1 <= n <= 16 and 0 <= k <= 22 and len(event['source_positions']) == n
            result = event['result']
            upper, lower = result['cost_s'], result['lower_bound_s']
            assert math.isfinite(upper) and math.isfinite(lower) and 0 <= lower <= upper+1e-7
            expected_budget = max(0, min(cap, total_cap-expanded_total))
            expanded = result['expanded']
            assert type(expanded) is int and 0 <= expanded <= expected_budget
            assert type(result['exact']) is bool
            if not result['exact']:
                assert expanded == expected_budget, 'Nonexact saved search did not reach its reconstructed budget'
            gap = max(0., upper-lower)
            if result['exact']:
                assert gap <= 1e-7
            expanded_total += expanded
            row = dict(seed=seed, stage=stage, route_log_index=index, prefix=prefix,
                schedulable_sources=n, remaining_covers=k, known_count=len(known), cleared_count=len(cleared),
                source_channels=channels, exact=result['exact'], expected_budget=expected_budget,
                expanded=expanded, generated=result['generated'], budget_hit=expanded == expected_budget,
                incumbent_cost_s=upper, lower_bound_s=lower, proxy_gap_s=gap,
                relative_proxy_gap=gap/upper if upper else 0., logged_runtime_s=result['runtime_s'],
                dense_dag_label_bound=(k+1)*(n+1)*(1 << n))
            events.append(row)
            local.append(row)
        per_case.append(dict(seed=seed, stage=stage, **stats(local),
            total_expanded=expanded_total, total_budget_exhausted=expanded_total >= total_cap,
            by_schedulable_sources={str(n): stats([event for event in local if event['schedulable_sources'] == n])
                                   for n in sorted({event['schedulable_sources'] for event in local})},
            wider_than_eight=stats([event for event in local if event['schedulable_sources'] > 8])))
    wide = [event for event in events if event['schedulable_sources'] > 8]
    return dict(scope='All 38 previously opened R12 621 development summary chain logs; saved results only',
        limitations=['n means tasks actually passed to routing, not true scene source total',
            'With covers remaining these are ready sources; with no covers, unresolved non-ready sources may also be included',
            'Saved incumbent-minus-lower-bound is possible improvement of one frozen proxy problem, not achieved savings or whole-case T/N improvement',
            'Overlapping decision gaps must not be summed as case savings',
            'Exact means frozen floating-point frontier closure; not Q4 global optimality'],
        input_sha256=inputs, script_sha256=sha(Path(__file__).read_bytes()),
        summary=dict(**stats(events), development_cases=len(cases),
            maximum_schedulable_sources=max(event['schedulable_sources'] for event in events),
            total_budget_exhausted_cases=sum(row['total_budget_exhausted'] for row in per_case)),
        wider_than_eight=stats(wide),
        by_schedulable_sources={str(n): stats([event for event in events if event['schedulable_sources'] == n])
                               for n in range(1, 17)},
        per_case=per_case, nonexact_events=[event for event in events if not event['exact']],
        wide_events=wide)


if __name__ == '__main__':
    destination = Path(__file__).with_suffix('.json')
    if destination.exists():
        raise SystemExit('Preserve existing diagnostic; explicit new version required')
    value = run()
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value[key] for key in ('summary', 'wider_than_eight')}))
