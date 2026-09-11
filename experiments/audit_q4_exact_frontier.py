"""Observed-prefix routing audit; independent backward Bellman verification.

The original frozen A* is replayed only to bind its work/initial incumbent.
The production exact-DP/refinement/strategy modules are never imported.
This does not replace wire/geometry/terminal audits or prove Q4 optimality.
"""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import math
from pathlib import Path

from localization import CandidateRegion
from experiments.audit_q4_clear_before_probe import require, point, terminal, audit_clear_before_probe_prefix
from experiments.audit_q4_joint_continuation import wire_prefix, audit_joint_continuation_prefix

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACT = {
    'src/strategies/q4_joint_visibility.py': 'eb922aae8052b48c9d54cd4b269a088498a3feeddbae254344b6ed73e31ff0e6',
    'src/strategies/q4_clear_before_probe.py': '641e8b0e4e6cce7b6445e88117d08ac23bd073487dfb46b87e903330f678ac69',
    'src/strategies/q4_range_scheduling.py': 'ad238a4e9c75537223d83ba4b180dc2d316ed71b487497f0f0e5a9bc2d72d76d',
    'src/strategies/q4_r2_scheduling.py': '48d0f20ed0a073d26f177078f54dd1b40a532038aa74bb0de098c81f84509a44',
    'src/strategies/q4_range_pruning.py': 'c8bd010f186ccd991ffa3cb6727ff1d2cc038c2955bc29f27bb9ab851c37a1d2',
    'src/strategies/q4_cover_search.py': '2bab5ee2f935967670bb7c1107b99a7bba20208372e15aab328a67b15d651404',
    'src/strategies/q4_state_search.py': '8d105d02f3cb97ef99a1ccef98b63e5510da3c9c5000f77ee6401fbb39b9de59',
    'src/strategies/search.py': '3c30ea448db217b2429d89c69d7db1f8e7fafc14c334043c6c1813ba94c44de5',
    'experiments/audit_q4_clear_before_probe.py': '68c7ef07dca4cc9f7a60351332911c25054c5e0392220830637bfd89518e62c4',
    'src/strategies/q4_exact_frontier.py': 'f261609efacef32336f42a5c45ee507dffdaec45b8fcaff9d6f07f5df82e7790',
    'src/planning/chain_route_exact.py': 'dac4b709d5a318c08fdccf50a1e441c31ac5b087bd33bdc26bc1f4f31dc66468',
    'src/planning/chain_route.py': '0502c7fe525bc64578bda3ecbe5fcfe6f6408a0ad728e3e670223d6d215f29ef',
    'src/strategies/q4_joint_continuation.py': '277480e9c22b0bd983a2971ad04fc97f8ff9e3232ed8198445b9d04dc95e024e',
    'experiments/audit_q4_joint_continuation.py': '5a04f00ee3796312b0ef0862fedf1bfefe1f9d57101e43eacbf11c7708db99e8',
}
LIMITS = dict(max_sources=8, max_covers=22, max_states_per_call=60000, improvement_tolerance_s=1e-9)


def verify_source_contract():
    require(all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == sha
                for p, sha in SOURCE_CONTRACT.items()), 'Exact-frontier source contract differs')


def number(value, expected, message, tolerance=1e-8):
    require(type(value) in (float, int) and math.isfinite(value)
            and abs(value-expected) <= tolerance, message)


def clean_result(result):
    value = dict(result)
    elapsed = value.pop('runtime_s')
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, 'Invalid route CPU time')
    require(type(value['exact']) is bool and all(type(value[k]) is int and value[k] >= 0
            for k in ('expanded','generated','dominance_pruned','bound_pruned')), 'Invalid route work types')
    require(all(type(value[k]) in (int,float) and math.isfinite(value[k]) and value[k] >= 0
                for k in ('cost_s','lower_bound_s')), 'Invalid route cost types')
    value['order'] = [list(x) for x in value['order']]
    return value


def route_cost(order, covers, sources, current, background):
    require(len(order) == len(covers)+len(sources), 'Incomplete route task count')
    covered, served, cost = [], [], 0.
    for action in order:
        require(isinstance(action, (tuple, list)) and len(action) == 2, 'Invalid route task')
        kind, index = action
        require(type(index) is int, 'Invalid route task index')
        if kind == 'cover':
            require(index == len(covered) and index < len(covers), 'Coverage chain reordered/repeated')
            covered.append(index); target, fee = covers[index], background
        else:
            require(kind == 'source' and 0 <= index < len(sources) and index not in served, 'Source task duplicated/missing')
            served.append(index); target, fee = sources[index], 5.
        cost += math.dist(current, target)/5.+fee
        current = target
    require(len(covered) == len(covers) and len(served) == len(sources), 'Route omitted a task')
    return cost


def bellman_cost(covers, sources, start, background):
    """Backward cost-to-go, unlike the production forward label algorithm."""
    covers, sources, start = tuple(covers), tuple(sources), tuple(start)
    n, kmax = len(sources), len(covers)
    require(n <= 8 and kmax <= 22, 'Independent DP fixed budget exceeded')
    full = (1 << n)-1
    @lru_cache(None)
    def remaining(mask, k, endpoint):
        current = sources[endpoint] if endpoint >= 0 else covers[k-1] if k else start
        choices = [math.dist(current, sources[i])/5.+5.+remaining(mask | (1 << i), k, i)
                   for i in range(n) if not mask & (1 << i)]
        if k < kmax:
            choices.append(math.dist(current, covers[k])/5.+background+remaining(mask, k+1, -1))
        return min(choices) if choices else 0.
    value = remaining(0, 0, -1)
    # Count reachable structural states/arcs without using floating labels.
    states = arcs = 0
    for mask in range(full+1):
        for k in range(kmax+1):
            count = mask.bit_count() + int(k > 0 or mask == 0)
            states += count
            arcs += count*(n-mask.bit_count()+int(k < kmax))
    require(states <= 60000 and states == remaining.cache_info().currsize, 'Independent state-count mismatch')
    return value, states, arcs


def audit_refinement(event, covers, sources, current, background):
    old, chosen = event['original_result'], event['selected_result']
    number(old['cost_s'], route_cost(old['order'], covers, sources, current, background), 'Original route cost mismatch')
    require(type(old['exact']) is bool, 'Nonboolean original exact flag')
    called = not old['exact'] and len(sources) <= 8 and len(covers) <= 22
    require(type(event['called']) is bool and event['called'] == called, 'Wrong exact-DP trigger')
    require(event['max_sources'] == 8 and event['max_covers'] == 22 and event['max_states'] == 60000
            and event['improvement_tolerance_s'] == 1e-9, 'DP work limits changed')
    states = arcs = 0
    if called:
        dp = event['dp_result']
        optimal, states, arcs = bellman_cost(covers, sources, current, background)
        number(dp['cost_s'], optimal, 'Independent Bellman optimum differs')
        number(dp['cost_s'], route_cost(dp['order'], covers, sources, current, background), 'DP order/cost differs', 1e-9)
        require(dp['exact'] is True and dp['lower_bound_s'] == dp['cost_s'], 'DP not exact finite-model result')
        require(type(dp['expanded']) is int and dp['expanded'] == states and type(dp['generated']) is int
                and dp['generated'] == arcs and dp['bound_pruned'] == 0, 'DP work accounting differs')
        require(type(dp['dominance_pruned']) is int and 0 <= dp['dominance_pruned'] <= arcs, 'Invalid DP dominance count')
        applied = dp['cost_s'] < old['cost_s']-1e-9
        expected = dp if applied else old
        reason = 'strict_proxy_improvement' if applied else 'no_strict_improvement'
    else:
        require(event['dp_result'] is None, 'Uncalled DP contains a result')
        applied, expected = False, old
        reason = 'incumbent_exact' if old['exact'] else 'fixed_work_limit'
    require(type(event['applied']) is bool and event['applied'] == applied and event['reason'] == reason,
            'Wrong strict-improvement/tie decision')
    require(clean_result(chosen) == clean_result(expected), 'Selected result differs; ties must retain original order')
    number(event['proxy_saved_s'], old['cost_s']-chosen['cost_s'] if applied else 0., 'Wrong improvement ledger', 1e-9)
    require(type(event['first_action_changed']) is bool and event['first_action_changed'] ==
            (applied and list(chosen['order'][0]) != list(old['order'][0])), 'First action change flag differs')
    return states, arcs, int(applied)


def audit_exact_frontier_prefix(record):
    verify_source_contract()
    spec = record['spec']; kwargs = spec.get('kwargs', {})
    require(spec['entrypoint'] == 'strategies.q4_exact_frontier:run_q4_exact_frontier'
            and kwargs.get('config', 'truncated_dag8') == 'truncated_dag8'
            and set(kwargs) <= {'config', 'max_actions', 'max_active_probes', 'max_expansions'}, 'Unknown exact-frontier spec')
    for key, default, low, high in [('max_actions',20000,2,1000000),('max_active_probes',6,0,30),('max_expansions',200,0,10000)]:
        require(type(kwargs.get(key,default)) is int and low <= kwargs.get(key,default) <= high, 'Invalid exact-frontier budget')
    summary, params = record['summary'], record['summary']['strategy_parameters']
    require(params['exact_frontier_config'] == 'truncated_dag8' and params['exact_frontier_limits'] == LIMITS, 'Strategy limits differ')
    require(params['max_route_expansions'] == kwargs.get('max_expansions',200)
            and params['max_total_route_expansions'] == 60000, 'Original A* budget differs')
    # This view changes only the already verified entry/config contract; actual
    # logs, observations and recursive certificates are the original objects.
    view = dict(record, spec=dict(entrypoint='strategies.q4_joint_continuation:run_q4_joint_continuation',
                                kwargs={**kwargs, 'config':'after_active_miss_optical'}))
    inherited_r8 = audit_clear_before_probe_prefix(view)
    inherited_r12 = audit_joint_continuation_prefix(view)
    h, before = wire_prefix(record)
    routes, events = params['chain_route_log'], params['exact_frontier_log']
    require(len(routes) == len(events), 'Missing/repeated refinement log')
    points = [point(p) for p in summary['coverage_points']]
    require(len(points) == len(set(points)) and len(points) <= 22, 'Invalid fixed coverage point chain')
    epochs = params['joint_visibility_resolver_log']; early = params['early_service_log']
    early_starts = {(x['after_actual_action_count'],x['channel']) for x in early}
    known, cleared, near, regions, visited = set(), set(), {}, {}, []
    astar = dp_states = dp_arcs = applied_count = cursor = 0
    selected_resolvers = set(); previous_prefix = -1
    from planning.chain_route import ChainSource, solve_chain_route  # frozen original A* only
    for i, (route, event) in enumerate(zip(routes, events)):
        prefix = route['after_actual_action_count']
        require(type(prefix) is int and previous_prefix <= prefix <= len(h), 'Unsorted/out-of-range route prefix')
        previous_prefix = prefix
        while cursor < prefix:
            a = h[cursor]; c, p = a['channel'], point(a['position'])
            if a['action'] == 'measure':
                if a['result'] in {'direction','near'}: known.add(c)
                if a['result'] == 'near': near[c] = p
                if a['result'] == 'direction': regions.setdefault(c,CandidateRegion()).observe(p,a['bearing_deg'])
                if a.get('phase') == 'coverage' and p not in visited:
                    visited.append(p)
                    require(visited == points[:len(visited)], 'Actual coverage order differs')
            elif a['result'] == 'success': cleared.add(c); known.add(c)
            cursor += 1
        require(len(cleared) < 16, 'Route after 16 actual removals')
        covers = [] if len(known) == 16 else points[len(visited):]
        blocked = set()
        for epoch in epochs:
            end, start, c = epoch['end_actual_action_count'], epoch['after_actual_action_count'], epoch['channel']
            if end <= prefix and epoch['status'] == 'unresolved' and (start,c) not in early_starts:
                if not any(a.get('phase') == 'coverage' for a in h[end:prefix]): blocked.add(c)
        targets = {}; channels = []
        for c in sorted(known-cleared-blocked):
            region = regions.get(c)
            disk = region.enclosing_disk() if region and region.vertices else None
            target = near.get(c, tuple(disk.center) if disk else None)
            if target is not None and (not covers or c in near or disk and disk.radius <= 19.9):
                channels.append(c); targets[c] = target
        sources = [targets[c] for c in channels]
        require(channels and route['source_channels'] == channels and event['source_channels'] == channels, 'Route source set differs from actual known/ready prefix')
        require([point(p) for p in route['remaining_covers']] == covers and
                [point(p) for p in route['source_positions']] == sources and
                route['discovery_count_cap'] is (len(known)==16), 'Route geometry/count cap differs')
        current = before[prefix][0]; background = 6.*max(0,20-len(cleared)-len(channels))
        require(event['after_actual_action_count'] == prefix and event['route_log_index'] == i and
                point(event['current_position']) == current and event['background_scan_s'] == background and
                event['scan_source_s'] == 0., 'Refinement prefix/fee binding differs')
        budget = max(0,min(kwargs.get('max_expansions',200),60000-astar))
        replay = asdict(solve_chain_route(covers,[ChainSource(p,5.) for p in sources],current,
                                        max_expansions=budget,background_scan_s=background,scan_source_s=0.))
        require(clean_result(event['original_result']) == clean_result(replay), 'Original A* result/work does not replay')
        astar += replay['expanded']
        states, arcs, changed = audit_refinement(event,covers,sources,current,background)
        dp_states += states; dp_arcs += arcs; applied_count += changed
        require(all(type(event[k]) is int for k in ('astar_total_expanded','dp_total_states','dp_total_arcs'))
                and event['astar_total_expanded'] == astar <= 60000 and event['dp_total_states'] == dp_states
                and event['dp_total_arcs'] == dp_arcs, 'Cumulative A*/DP budget mismatch')
        require(clean_result(route['result']) == clean_result(event['selected_result']), 'Route log differs from selected refinement')
        kind,index = event['selected_result']['order'][0]
        require(route['selected_kind'] == kind and route['selected_channel'] == (channels[index] if kind=='source' else None), 'Selected macro mismatch')
        if kind == 'source':
            key = prefix,channels[index]
            require(key not in selected_resolvers and sum(e['after_actual_action_count']==prefix and e['channel']==channels[index] for e in epochs)==1,
                    'Selected source lacks unique real resolver')
            selected_resolvers.add(key)
        elif prefix < len(h):
            require(h[prefix].get('phase') == 'coverage' and point(h[prefix]['position']) == covers[0], 'Selected cover differs from next real action')
        else:
            require(terminal(summary,prefix,h), 'Unexecuted cover without terminal reason')
    require(all((e['after_actual_action_count'],e['channel']) in selected_resolvers | early_starts for e in epochs), 'Resolver lacks planning/early-service provenance')
    # Omitting both logs at a cover-selected planning call is also detectable:
    # at a real scan boundary, any live ready source requires a planner call.
    known, cleared, near, regions = set(), set(), set(), {}
    cover_calls = {r['after_actual_action_count'] for r in routes if r['selected_kind']=='cover'}
    for j,a in enumerate(h):
        starts_cover = a.get('phase') == 'coverage' and (j == 0 or h[j-1].get('phase') != 'coverage'
                       or point(h[j-1]['position']) != point(a['position']))
        if starts_cover:
            require(len(known) < 16, 'Coverage continued after 16 real detected sources')
            ready = any(c in near or (regions.get(c) and regions[c].vertices
                        and regions[c].enclosing_disk().radius <= 19.9) for c in known-cleared)
            require(not ready or j in cover_calls, 'Coverage with ready source lacks planner/refinement provenance')
        c = a['channel']
        if a['action']=='measure':
            if a['result'] in {'direction','near'}: known.add(c)
            if a['result']=='near': near.add(c)
            elif a['result']=='direction': regions.setdefault(c,CandidateRegion()).observe(point(a['position']),a['bearing_deg'])
        elif a['result']=='success': known.add(c);cleared.add(c)
    return dict(passed=True, route_calls=len(routes), dp_calls=sum(e['called'] for e in events),
                applied=applied_count, astar_expanded=astar, dp_states=dp_states, dp_arcs=dp_arcs,
                source_contract=SOURCE_CONTRACT, inherited_r8=inherited_r8, inherited_r12=inherited_r12)
