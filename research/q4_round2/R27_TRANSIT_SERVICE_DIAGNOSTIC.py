"""Fixed OLD 621 prefix diagnostic; no truth, new scene or simulated response.

Only an actually chosen cover edge is considered. A source/early task starting
at the same prefix excludes it, including ambiguous zero-action boundaries.
"""
import gzip
import json
import math
from pathlib import Path
import statistics

from R24_EARLY_SERVICE_DIAGNOSTIC import CandidateRegion, R12, WORKSPACE, scan_blocks, sha


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0)
    t = .95*(len(values)-1); lo = int(t); hi = min(lo+1, len(values)-1)
    return dict(count=len(values), min=values[0], mean=statistics.fmean(values),
                median=statistics.median(values), p95=values[lo]+(t-lo)*(values[hi]-values[lo]), max=values[-1])


def diagnostic(summary, seed, stage):
    params, history = summary['strategy_parameters'], summary['action_history']
    assert params['scheduling_config'] == dict(early_services=4, radius_m=40., detour_m=100., service_budget_s=60.)
    points = [tuple(p) for p in summary['coverage_points']]
    blocks = scan_blocks(history, points)
    chain, early = params['chain_route_log'], params['early_service_log']
    resolvers = params['joint_visibility_resolver_log']
    starts = {(e['after_actual_action_count'], e['selected_channel']) for e in chain if e['selected_kind'] == 'source'}
    source_epochs = [e for e in resolvers if (e['after_actual_action_count'], e['channel']) in starts]
    failed_ends = {}
    for e in source_epochs:
        if e['status'] == 'unresolved':
            failed_ends.setdefault(e['end_actual_action_count'], set()).add(e['channel'])
    prefixes = {}
    for b in blocks[:10]:
        prefixes.setdefault(b['end'], set()).add('first_10_complete_cover_ends')
    for e in chain:
        prefixes.setdefault(e['after_actual_action_count'], set()).add('actual_chain_decision')
    scan_ends = {b['end'] for b in blocks}
    regions, known, cleared, near, blocked = {}, set(), set(), set(), set()
    cursor = visited = 0
    current = (0., 0.)
    counted, found = set(), []
    counts = dict(prefixes=len(prefixes), actual_cover_edges=0, excluded_source_or_early=0,
                  no_actual_cover_next=0, no_cover_or_known16=0, four_cap_exhausted_edges=0,
                  radius_window_snapshot_channels=0, projection_and_detour_pass=0,
                  far_first_approach_snapshot_channels=0)
    for n in sorted(prefixes):
        while cursor < n:
            a = history[cursor]; c = a['channel']; current = tuple(a['position'])
            if a['action'] == 'measure':
                if a['result'] in {'near', 'direction'}:
                    known.add(c)
                if a['result'] == 'near':
                    near.add(c)
                elif a['result'] == 'direction':
                    regions.setdefault(c, CandidateRegion()).observe(current, a['bearing_deg'])
            elif a['result'] == 'success':
                known.add(c); cleared.add(c)
            cursor += 1
            if cursor in scan_ends:
                visited += 1; blocked.clear()
            blocked.update(failed_ends.get(cursor, set()))
        if len(known) >= 16 or visited == len(points):
            counts['no_cover_or_known16'] += 1
            continue
        # Do not pretend an already-selected source or early service is a
        # chosen transit edge. Exclude any ambiguous zero-length start too.
        source_selected = any(e['after_actual_action_count'] == n and e['selected_kind'] == 'source' for e in chain)
        service_active = any(e['after_actual_action_count'] <= n < e['end_actual_action_count']
            or e['after_actual_action_count'] == e['end_actual_action_count'] == n for e in early)
        if source_selected or service_active:
            counts['excluded_source_or_early'] += 1
            continue
        station = points[visited]
        if n >= len(history) or history[n]['action'] != 'measure' or history[n]['phase'] != 'coverage' or tuple(history[n]['position']) != station:
            counts['no_actual_cover_next'] += 1
            continue
        matches = [e for e in chain if e['after_actual_action_count'] == n]
        assert not matches or all(e['selected_kind'] == 'cover' and tuple(e['remaining_covers'][0]) == station for e in matches)
        counts['actual_cover_edges'] += 1
        attempted = {e['channel'] for e in early if e['end_actual_action_count'] <= n}
        if len(attempted) >= 4:
            counts['four_cap_exhausted_edges'] += 1
            continue
        dx, dy = station[0]-current[0], station[1]-current[1]
        edge2 = dx*dx+dy*dy
        if edge2 == 0:
            continue
        length = math.sqrt(edge2)
        for c in sorted(known-cleared-near-blocked-attempted):
            region = regions.get(c)
            if not region or not region.vertices:
                continue
            disk = region.enclosing_disk()
            if not math.isfinite(disk.radius) or not 19.9 < disk.radius <= 40.:
                continue
            counts['radius_window_snapshot_channels'] += 1
            center = tuple(disk.center)
            t = ((center[0]-current[0])*dx+(center[1]-current[1])*dy)/edge2
            approach = math.dist(current, center)
            detour = approach+math.dist(center, station)-length
            if not .1 <= t <= .9 or detour > 100.:
                continue
            counts['projection_and_detour_pass'] += 1
            first = approach/5.+6.
            if first <= 60.:
                continue
            counts['far_first_approach_snapshot_channels'] += 1
            if c in counted:
                continue
            counted.add(c)
            shared = t*length
            found.append(dict(seed=seed, stage=stage, prefix=n, channel=c,
                prefix_kinds=sorted(prefixes[n]), selected_by='chain_cover' if matches else 'direct_no_ready_cover',
                current=list(current), next_station_id=visited, next_station=list(station),
                canonical_center=list(center), radius_m=disk.radius, projection_t=t,
                cover_edge_m=length, approach_m=approach, shared_along_edge_m=shared,
                shared_along_edge_s=shared/5., off_line_m=math.dist(center,(current[0]+t*dx,current[1]+t*dy)),
                detour_m=detour, detour_s=detour/5., first_approach_upper_s=first,
                proposed_approach_plus_service_allowance_s=approach/5.+60.,
                known_count=len(known), cleared_count=len(cleared),
                attempted_channels=sorted(attempted), blocked_channels=sorted(blocked),
                sole_original_barrier='first_approach_distance_div5_plus6_above60'))
    return dict(seed=seed, stage=stage, **counts, opportunities=len(found)), found


def run():
    inputs, cases, opportunities = {}, [], []
    ordered = [('development', s) for s in range(621001,621025)]
    ordered += [('development-stress',s) for s in range(621031,621045)]
    for stage, seed in ordered:
        path=R12/f'results/q4_joint_continuation/{stage}/records/compact_joint_continuation-{seed}.json.gz'
        raw=path.read_bytes(); inputs[path.relative_to(WORKSPACE).as_posix()]=sha(raw)
        # Only the real history and actual diagnostic logs within summary are
        # accessed; evaluation/truth is never used, copied, or returned.
        summary=json.loads(gzip.decompress(raw))['summary']
        detail, found=diagnostic(summary,seed,stage)
        cases.append(detail); opportunities.extend(found)
    keys=['prefixes','actual_cover_edges','excluded_source_or_early','no_actual_cover_next',
          'no_cover_or_known16','four_cap_exhausted_edges','radius_window_snapshot_channels',
          'projection_and_detour_pass','far_first_approach_snapshot_channels']
    aggregate={k:sum(c[k] for c in cases) for k in keys}
    aggregate.update(cases=len(cases),opportunities=len(opportunities),
        cases_with_opportunities=len({e['seed'] for e in opportunities}),
        random_cases_with_opportunities=len({e['seed'] for e in opportunities if e['stage']=='development'}),
        stress_cases_with_opportunities=len({e['seed'] for e in opportunities if e['stage']=='development-stress'}))
    return dict(scope='Fixed 38 OLD 621 R12 development summaries; no strategy execution or counterfactual feedback',
        protocol=dict(radius_lower_exclusive=19.9,radius_upper_inclusive=40.,projection_interval=[.1,.9],
            detour_upper_m=100.,first_approach_upper_s_must_exceed=60.,original_attempt_cap=4,
            prefixes='All actual chain decisions plus first ten complete coverage-block ends per case; deduplicated',
            edge_selection='Next actual action must be coverage at the next original station; exclude any same-prefix source selection/early interior or zero-length start',
            counting='First qualifying case-channel occasion only; not a proposed simultaneous schedule'),
        input_sha256=inputs,script_sha256=sha(Path(__file__).read_bytes()),
        dependencies_sha256={str(p.relative_to(WORKSPACE)):sha(p.read_bytes()) for p in [
            Path(__file__).with_name('R24_EARLY_SERVICE_DIAGNOSTIC.py'),
            R12/'src/localization/__init__.py',R12/'src/geometry/__init__.py',R12/'src/strategies/q4_r2_scheduling.py']},
        summary=aggregate,distributions={k:distribution([e[k] for e in opportunities]) for k in [
            'radius_m','projection_t','detour_m','detour_s','approach_m','first_approach_upper_s',
            'shared_along_edge_m','shared_along_edge_s','off_line_m','proposed_approach_plus_service_allowance_s']},
        per_case=cases,opportunities=opportunities,
        limitations=['Restricted observed prefixes, not exhaustive all-policy opportunity frequency',
          'MEC center is an observable representative, not a known true source position or guaranteed receiver',
          'No counterfactual observation, clearing success, measured saving or improvement is inferred',
          'Shared projection length is part of the original straight edge, not an extra service-cost discount certificate',
          'Allowing approach/5+60 is a direct per-service allowance; ensuing changed decisions can cause unbounded-by-this-allowance relative whole-case regression',
          'Multiple old-path opportunities need not remain available after selecting one; four-service cap must remain enforced'])


if __name__=='__main__':
    destination=Path(__file__).with_suffix('.json')
    if destination.exists():raise SystemExit('Preserve existing R27 diagnostic; use explicit new version')
    result=run()
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'summary':result['summary'],'distributions':result['distributions']}))
