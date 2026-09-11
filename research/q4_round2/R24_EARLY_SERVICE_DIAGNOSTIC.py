"""OLD R12 prefix-only audit of the fixed (40,120]m early-service window.

No counterfactual observation, route execution, source truth, or new case is
used. Geometry is replayed by the unchanged canonical positive-bearing module.
Each channel contributes at most its first geometrically eligible occasion.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


WORKSPACE = Path(__file__).resolve().parents[3]
R12 = WORKSPACE/'q4-r12-joint-continuation'
sys.path.insert(0, str(R12/'src'))
from localization import CandidateRegion


def sha(data):
    return hashlib.sha256(data).hexdigest()


def scan_blocks(history, points):
    result, i, known = [], 0, set()
    while i < len(history):
        action = history[i]
        if action['action'] == 'measure' and action['phase'] == 'coverage':
            start, p, required, measured = i, tuple(action['position']), set(range(1,21))-known, set()
            assert p == points[len(result)]
            while i < len(history):
                action = history[i]
                if action['action'] != 'measure' or action['phase'] != 'coverage' or tuple(action['position']) != p:
                    break
                assert action['channel'] not in measured
                measured.add(action['channel'])
                if action['result'] in ('direction', 'near'):
                    known.add(action['channel'])
                i += 1
            assert required <= measured
            result.append(dict(start=start, end=i, station_id=len(result)))
            continue
        if action['action'] == 'measure' and action['result'] in ('direction', 'near'):
            known.add(action['channel'])
        elif action['action'] == 'clear' and action['result'] == 'success':
            known.add(action['channel'])
        i += 1
    return result


def case_diagnostic(summary, seed, stage):
    params, history = summary['strategy_parameters'], summary['action_history']
    assert params['scheduling_config'] == dict(early_services=4, radius_m=40., detour_m=100., service_budget_s=60.)
    points = [tuple(p) for p in summary['coverage_points']]
    blocks = scan_blocks(history, points)
    early = params['early_service_log']
    chain = params['chain_route_log']
    resolvers = params['joint_visibility_resolver_log']
    source_starts = {(e['after_actual_action_count'], e['selected_channel']) for e in chain if e['selected_kind']=='source'}
    source_epochs = [e for e in resolvers if (e['after_actual_action_count'], e['channel']) in source_starts]
    origins = {}
    def add(prefix, tag): origins.setdefault(prefix, set()).add(tag)
    for block in blocks[:10]: add(block['end'], 'first_10_completed_coverage')
    for event in chain: add(event['after_actual_action_count'], 'chain_macro_start')
    for event in source_epochs: add(event['end_actual_action_count'], 'chain_source_macro_end')
    for event in early:
        add(event['after_actual_action_count'], 'actual_early_start')
        add(event['end_actual_action_count'], 'actual_early_end')
    regions, near, known, cleared, blocked = {}, set(), set(), set(), set()
    p, cursor, visited = (0.,0.), 0, 0
    scan_ends = {b['end']: b for b in blocks}
    failed_source_ends = {}
    for event in source_epochs:
        if event['status']=='unresolved':
            failed_source_ends.setdefault(event['end_actual_action_count'], set()).add(event['channel'])
    used_channels, opportunities = set(), []
    counters = dict(prefixes=0, no_cover_or_count_cap=0, active_or_ambiguous_service=0,
                    radius_window_snapshot_channels=0, geometric_snapshot_opportunities=0)
    for prefix in sorted(origins):
        while cursor < prefix:
            action = history[cursor]
            c = action['channel']
            if action['action']=='measure' and action['result'] in ('direction','near'):
                known.add(c)
                if action['result']=='direction':
                    regions.setdefault(c, CandidateRegion()).observe(action['position'], action['bearing_deg'])
                else: near.add(c)
            elif action['action']=='clear' and action['result']=='success':
                known.add(c); cleared.add(c)
            if action.get('position') is not None: p=tuple(action['position'])
            cursor += 1
            if cursor in scan_ends:
                visited += 1
                blocked.clear()
            blocked.update(failed_source_ends.get(cursor, set()))
        counters['prefixes'] += 1
        # Do not pretend an active early-service interior or an unordered
        # zero-action boundary is an available outer scheduling decision.
        if any(e['after_actual_action_count'] < prefix < e['end_actual_action_count']
               or e['after_actual_action_count']==e['end_actual_action_count']==prefix for e in early):
            counters['active_or_ambiguous_service'] += 1
            continue
        if len(known)==16 or visited==len(points):
            counters['no_cover_or_count_cap'] += 1
            continue
        attempted = {e['channel'] for e in early if e['end_actual_action_count'] <= prefix}
        next_station = points[visited]
        choices, window = [], []
        for c in sorted(known-cleared-blocked-attempted):
            region = regions.get(c)
            if c in near or region is None or not region.vertices:
                continue
            circle = region.enclosing_disk()
            if not math.isfinite(circle.radius) or circle.radius <= 19.9 or circle.radius > 120.:
                continue
            center = tuple(circle.center)
            approach = math.dist(p, center)
            detour = approach+math.dist(center,next_station)-math.dist(p,next_station)
            first = approach/5.+6.
            if circle.radius > 40.:
                counters['radius_window_snapshot_channels'] += 1
            if detour <= 100. and first <= 60.:
                item=(detour,approach,c,circle.radius,center,first)
                choices.append(item)
                if circle.radius > 40.:
                    counters['geometric_snapshot_opportunities'] += 1
                    window.append(item)
        winner = min(choices)[2] if choices and len(attempted)<4 else None
        old_choices = [item for item in choices if item[3]<=40.]
        old_winner = min(old_choices)[2] if old_choices and len(attempted)<4 else None
        for detour,approach,c,radius,center,first in window:
            if c in used_channels:
                continue
            used_channels.add(c)
            opportunities.append(dict(seed=seed,stage=stage,prefix=prefix,channel=c,
                prefix_kinds=sorted(origins[prefix]),position=list(p),next_station_id=visited,
                next_station=list(next_station),canonical_center=list(center),radius_m=radius,
                detour_m=detour,approach_m=approach,first_approach_upper_s=first,
                known_count=len(known),cleared_count=len(cleared),
                already_attempted_channels=sorted(attempted),blocked_channels=sorted(blocked),
                radius_is_only_added_barrier=len(attempted)<4,
                also_blocked_by_four_slice_cap=len(attempted)>=4,
                existing_gate_winner=old_winner,expanded_gate_winner=winner,
                would_win_this_frozen_gate=winner==c))
    return dict(seed=seed,stage=stage,**counters,actual_early_services=len(early),
        actual_early_cleared=sum(e['cleared'] for e in early),
        actual_early_interrupted=sum(e['interrupted'] for e in early),
        actual_early_cost_s=sum(e['actual_cost_s'] for e in early),
        opportunities=len(opportunities),radius_only=sum(e['radius_is_only_added_barrier'] for e in opportunities),
        cap_blocked=sum(e['also_blocked_by_four_slice_cap'] for e in opportunities)), opportunities


def run():
    inputs, cases, opportunities = {}, [], []
    ordered = [('development',s) for s in range(621001,621025)]
    ordered += [('development-stress',s) for s in range(621031,621045)]
    for stage,seed in ordered:
        path=R12/f'results/q4_joint_continuation/{stage}/records/compact_joint_continuation-{seed}.json.gz'
        raw=path.read_bytes(); inputs[path.relative_to(WORKSPACE).as_posix()]=sha(raw)
        summary=json.loads(gzip.decompress(raw))['summary']
        detail, found=case_diagnostic(summary,seed,stage)
        cases.append(detail); opportunities.extend(found)
    radius_only=[e for e in opportunities if e['radius_is_only_added_barrier']]
    return dict(scope='38 OLD 621 R12 development summaries, real feedback only, no new case or hypothetical outcome',
        protocol=dict(radius_window_m=[40,120],left_open=True,max_detour_m=100,
            max_first_approach_upper_s=60,existing_slice_cap=4,
            prefix_selection='First ten completed coverage endpoints plus all recorded chain macro starts/source ends and real early boundaries; deduplicated',
            counting='At most first geometrically eligible occasion per source channel per case; exclude cleared, attempted, blocked and ready'),
        limitations=['Counterfactual opportunity is not an executed service or estimated time saving',
            'Changing an early choice changes later positions, observations and consumption of the four-slice budget',
            'MEC radius and first approach do not guarantee directional reception, one-slice clearing, or useful information',
            'Existing canonical geometry is replayed; this is not independent certification of that geometry module'],
        input_sha256=inputs,source_sha256={name:sha((R12/name).read_bytes()) for name in
            ('src/localization/__init__.py','src/geometry/__init__.py','src/strategies/q4_r2_scheduling.py')},
        script_sha256=sha(Path(__file__).read_bytes()),
        summary=dict(cases=len(cases),prefixes=sum(c['prefixes'] for c in cases),
            unique_case_channel_opportunities=len(opportunities),cases_with_opportunities=len({e['seed'] for e in opportunities}),
            radius_only_opportunities=len(radius_only),cases_with_radius_only=len({e['seed'] for e in radius_only}),
            cap_also_blocked=sum(e['also_blocked_by_four_slice_cap'] for e in opportunities),
            first_occasions_would_win=sum(e['would_win_this_frozen_gate'] for e in opportunities),
            actual_early_services=sum(c['actual_early_services'] for c in cases),
            actual_early_cleared=sum(c['actual_early_cleared'] for c in cases),
            actual_early_interrupted=sum(c['actual_early_interrupted'] for c in cases),
            cases_using_all_four_slices=sum(c['actual_early_services']>=4 for c in cases),
            median_opportunity_radius_m=statistics.median(e['radius_m'] for e in opportunities) if opportunities else None),
        per_case=cases,opportunities=opportunities)


if __name__=='__main__':
    destination=Path(__file__).with_suffix('.json')
    if destination.exists(): raise SystemExit('Preserve existing diagnostic; explicit new version required')
    value=run()
    with destination.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(value['summary']))
