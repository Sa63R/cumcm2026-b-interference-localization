"""Complete finite DAG routing for <=8 sources and <=22 ordered covers.

Only the existing frozen point-task objective is optimized. This is not a
solution of the unknown-source online search problem. No observations/client
or scene data are accepted by this module.
"""
import math
import time

from simulator_client.state import Position
from .chain_route import ChainSource, ChainRouteResult

MAX_SOURCES, MAX_COVERS, MAX_STATES = 8, 22, 60000


def _number(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or positive and value == 0):
        raise ValueError(name+' must be finite and '+('positive' if positive else 'nonnegative'))
    return float(value)


def solve_chain_route_exact(cover_points, sources, start=(0., 0.), *,
                            background_scan_s=0., scan_source_s=6., speed_mps=5.):
    began = time.perf_counter()
    covers, sources = tuple(Position.coerce(p) for p in cover_points), tuple(sources)
    n, kmax = len(sources), len(covers)
    if n > MAX_SOURCES or kmax > MAX_COVERS or any(not isinstance(s, ChainSource) for s in sources):
        raise ValueError('Exact route requires <=8 ChainSource tasks and <=22 covers')
    if (kmax+1)*(n+1)*(1 << n) > MAX_STATES:
        raise ValueError('Exact route state budget exceeded')
    speed = _number(speed_mps, 'speed', True)
    background, dynamic = _number(background_scan_s, 'background'), _number(scan_source_s, 'dynamic scan')
    locations = tuple(s.position for s in sources)+covers+(Position.coerce(start),)
    origin = len(locations)-1
    travel = [[a.distance_to(b)/speed for b in locations] for a in locations]
    if not math.isfinite((n+kmax)*max(max(row) for row in travel)+
                         sum(s.service_s for s in sources)+kmax*(background+dynamic*n)):
        raise ValueError('Exact route cost scale must remain finite')
    full = (1 << n)-1
    labels, parents = {(0, 0, -1): 0.}, {}
    expanded = generated = dominated = 0
    # Both transitions strictly increase (mask, cover_prefix) lexicographically:
    # visit a source sets one previously unset bit, visit a cover increases k.
    # Thus the endpoint label at this state is final when it is processed.
    for mask in range(full+1):
        for k in range(kmax+1):
            for last in range(-1, n):
                state = mask, k, last
                if state not in labels:
                    continue
                expanded += 1
                current = last if last >= 0 else n+k-1 if k else origin
                cost = labels[state]
                moves = [(('source', i), (mask | (1 << i), k, i),
                           travel[current][i]+sources[i].service_s)
                         for i in range(n) if not mask & (1 << i)]
                if k < kmax:
                    moves.append((('cover', k), (mask, k+1, -1),
                                  travel[current][n+k]+(background+dynamic*(n-mask.bit_count()))))
                for action, target, fee in moves:
                    generated += 1
                    new = cost+fee
                    if new < labels.get(target, math.inf):
                        labels[target] = new
                        parents[target] = state, action
                    else:
                        dominated += 1
    cost, end = min((value, state) for state, value in labels.items()
                    if state[0] == full and state[1] == kmax)
    order = []
    while end in parents:
        end, action = parents[end]
        order.append(action)
    return ChainRouteResult(tuple(reversed(order)), cost, cost, expanded, generated, dominated, 0,
                            True, time.perf_counter()-began)


def refine_truncated_route(incumbent, cover_points, sources, start=(0., 0.), *,
                           background_scan_s=0., scan_source_s=6., speed_mps=5.):
    """Preserve original result/first action except for >1e-9 s improvement.

    Original A* work remains counted by the caller. DP work is separately
    reported and never charged against or concealed in the old A* budget.
    """
    if not isinstance(incumbent, ChainRouteResult):
        raise ValueError('Need original ChainRouteResult')
    covers, sources = tuple(cover_points), tuple(sources)
    log = dict(called=False, applied=False, reason=None, dp_result=None, proxy_saved_s=0.,
               first_action_changed=False, max_sources=MAX_SOURCES, max_covers=MAX_COVERS,
               max_states=MAX_STATES, improvement_tolerance_s=1e-9)
    if incumbent.exact:
        log['reason'] = 'incumbent_exact'
        return incumbent, log
    if len(sources) > MAX_SOURCES or len(covers) > MAX_COVERS:
        log['reason'] = 'fixed_work_limit'
        return incumbent, log
    from dataclasses import asdict
    result = solve_chain_route_exact(covers, sources, start, background_scan_s=background_scan_s,
                                    scan_source_s=scan_source_s, speed_mps=speed_mps)
    log.update(called=True, dp_result=asdict(result), reason='no_strict_improvement')
    if result.cost_s < incumbent.cost_s-1e-9:
        log.update(applied=True, reason='strict_proxy_improvement',
            proxy_saved_s=incumbent.cost_s-result.cost_s,
            first_action_changed=result.next_action != incumbent.next_action)
        return result, log
    return incumbent, log
