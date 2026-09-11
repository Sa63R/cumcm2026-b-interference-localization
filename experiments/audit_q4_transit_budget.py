"""Independent actual-prefix transit budget audit; no hidden-source input.

All base source bytes and inherited replay dependencies are pinned. The R12
checker runs with private globals, replacing only its stopping predicate after
an independently checked transit gate. No history or early-service event is
fabricated and no imported module globals are mutated.
"""
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType

from localization import CandidateRegion
from experiments.audit_q4_clear_before_probe import require, point, close, terminal, audit_clear_before_probe_prefix
from experiments.audit_q4_joint_continuation import wire_prefix, audit_joint_continuation_prefix, stopped as original_stopped, probe_formula
from experiments.audit_q4_scheduling import audit_scheduling_prefix
from experiments.audit_q4_range import audit_range_prefix

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'strategies.q4_transit_budget:run_q4_transit_budget'
CONFIG = 'incremental_60'
LABEL = 'compact_transit_budget'
# 49 inherited src files byte-matched to qualified R12 81aa6e1a; replay modules
# are pinned because the limited private-global adaptation depends on them.
SOURCE_CONTRACT = {'experiments/audit_q4_clear_before_probe.py': '68c7ef07dca4cc9f7a60351332911c25054c5e0392220830637bfd89518e62c4',
 'experiments/audit_q4_joint_continuation.py': '5a04f00ee3796312b0ef0862fedf1bfefe1f9d57101e43eacbf11c7708db99e8',
 'experiments/audit_q4_range.py': '8995a087ec085cbef445ba455f68dc81d426c1e51b3406e81184c8888cdd3578',
 'experiments/audit_q4_scheduling.py': '836d82d30280d24b612ca96367ca7e9b33b0b1b76f441bbfbcf2d674606ff5be',
 'src/geometry/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/geometry/__init__.py': 'ab863186eed111790cf712c0cd19c741841087144a8fd9ecd37a3c10d33e46fa',
 'src/localization/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/localization/__init__.py': 'cda7e2f2a45faa60cb7dcef631e54cd384e3f367db96f23f415fc7eb666bb98c',
 'src/localization/omni.py': '35a999a1533a7c557315f719e2dcea91ffd3e1ec695636b94d374d4aaddc1e3b',
 'src/planning/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/planning/__init__.py': '070acfaf1e3aa79d2d4c9c8616f733b512e264729118f720929783a18b227027',
 'src/planning/chain_route.py': '0502c7fe525bc64578bda3ecbe5fcfe6f6408a0ad728e3e670223d6d215f29ef',
 'src/planning/coverage.py': '4c097ba0d009d0954a96089fb867db5bb4d00020d2cb0b408d8d4c297b427000',
 'src/planning/directional_probe_pair.py': '635d46a7a9571ab8f4a9f9f066b0ad5da0d5febcaf297c3520495bff55926f06',
 'src/planning/joint_visibility_region.py': 'c2df945c3a3dc295827fd7afbbb78793b43184cf61e1d1070f47020293953d7a',
 'src/planning/positive_hull_probe.py': 'b03d5906660ee7565f19acaf8a7f86706957818977e526288bd845c40d9ffb5f',
 'src/planning/q4_directional_cover.py': 'e863b3f0fb83e1deba8d1a24c4b929b8fe1caca9927f8b99e514b1deea6da5c1',
 'src/planning/routing.py': 'e9cd4d242de519247a1fd7320add95c0c3d603f21a77899bb88d296d13ae4684',
 'src/practice_control/__init__.py': '45694c74e155e7267239f23656085ae352a59ce7cb5968b0dac6a36abcc9f62d',
 'src/practice_control/__main__.py': '78cb1245f7f355e1e4e7f855314211848687da00c3f4923d57ea8c1441276c46',
 'src/practice_control/branch_worker.py': '6e376f01d02606b251404038fe08c73e732297081d1e322656f24926a38c079b',
 'src/practice_control/bridge.py': 'fc28f4a2f96f335bcac64170b9c0a96cbf30f5e50d81a0bd679daacde4714779',
 'src/practice_control/runner.py': '15c28002cb80b899f7c19b6af2326fd687d0ae7131c05b910bed9ad83f574440',
 'src/practice_control/runtime.py': 'a9e62e85d19fe9081bf5085871715ef809ce9c25c79c58c35358b68b83c43865',
 'src/simulation/__init__.py': '2e01c860eb9cd4ea7fbab3cda9f13eaa4a454bef0ac17541301751dec0925cb6',
 'src/simulation/cases.py': '4d5588d9c11ccda5f251a819b292d9d69f1d5f5f9580be20534aa3bb9b6eec39',
 'src/simulation/engine.py': '3ef36f508f773564baed47569e014309cfb1cbcebb1ba1ad268a4b0f3e125622',
 'src/simulation/q3_branch.py': 'd16b7adf3c317f5c21ca3880916a1736884f396d7b7e964b71b8d8ce89ffe790',
 'src/simulator_client/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/simulator_client/__init__.py': 'aa5f551dfde62358958cc6d56c7ce986f3ce7937883c4edbb64cf8fb2c7aeee9',
 'src/simulator_client/__main__.py': 'd96fddf1ef8a4a824c8c85cba90e8e3b9ffd38b7fc97f21cdcd630875cd146f3',
 'src/simulator_client/client.py': '441230d1f2bb231a7a64beaa119306d89b4535ffaf1f4393c8be9c4d08a81577',
 'src/simulator_client/errors.py': 'c1ad261e6e6a99a691c08fec3f4b331036386542c968b45452881c522dce7723',
 'src/simulator_client/rules.py': '9bd7ca6df718aa42a414a94a08f981b519bfa70dc9ceeccb4f1a194aefcc83af',
 'src/simulator_client/state.py': '7a64a2a871532f76613000994bd850e86148a28f72f415d017fa56553815bb42',
 'src/strategies/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'src/strategies/__init__.py': '7eec10fe0201db03e7fc867e474dadf5a9a19cd1d3985bb2e22cdca88f9873a1',
 'src/strategies/efficient.py': '2349996a10685d62c441f2ebf8ff176d18f37ced660eb260e4daf38cf6b1f37a',
 'src/strategies/q3_belief.py': '35a5222a2c2cd89857f3d8c9d00a12217fe2667a20a2586dc4a1e28295ff465e',
 'src/strategies/q4_clear_before_probe.py': '641e8b0e4e6cce7b6445e88117d08ac23bd073487dfb46b87e903330f678ac69',
 'src/strategies/q4_cover_search.py': '2bab5ee2f935967670bb7c1107b99a7bba20208372e15aab328a67b15d651404',
 'src/strategies/q4_joint_continuation.py': '277480e9c22b0bd983a2971ad04fc97f8ff9e3232ed8198445b9d04dc95e024e',
 'src/strategies/q4_joint_visibility.py': 'eb922aae8052b48c9d54cd4b269a088498a3feeddbae254344b6ed73e31ff0e6',
 'src/strategies/q4_optical_cover.py': '87f8805167a91c364856d3da0f57ea032f4b3ac23b9d9c28c18c0a642136bfa6',
 'src/strategies/q4_r2_scheduling.py': '48d0f20ed0a073d26f177078f54dd1b40a532038aa74bb0de098c81f84509a44',
 'src/strategies/q4_range_pruning.py': 'c8bd010f186ccd991ffa3cb6727ff1d2cc038c2955bc29f27bb9ab851c37a1d2',
 'src/strategies/q4_range_scheduling.py': 'ad238a4e9c75537223d83ba4b180dc2d316ed71b487497f0f0e5a9bc2d72d76d',
 'src/strategies/q4_state_search.py': '8d105d02f3cb97ef99a1ccef98b63e5510da3c9c5000f77ee6401fbb39b9de59',
 'src/strategies/rollout.py': 'f2c7afa93373735a576e364515483e43ab01e8c2ea5e119ddb67f0dfda23838d',
 'src/strategies/search.py': '3c30ea448db217b2429d89c69d7db1f8e7fafc14c334043c6c1813ba94c44de5',
 'src/workflow/__init__.py': '3755f953a7f53ae76381142ab9e08347340462faf3bcb4694b4f76156c734d8c',
 'src/workflow/__main__.py': 'c353609048133fb2d11f9c774263dbb374de42910c816e062bddaa5166120ad6',
 'src/workflow/evidence.py': 'a4c5f37fe120c58be082fd3708020dd844140aceaed01835d8f1c52132ebca80'}
NEW_SOURCE_CONTRACT = {'src/strategies/q4_transit_budget.py': '8bd13c7ee1251ac7e14a542ba5674818427abf54c3d0a9590e2d66dd9acae4ef'}


def verify_source_contract(root=ROOT):
    require(bool(NEW_SOURCE_CONTRACT), 'Missing reviewed transit source contract')
    for name, expected in (SOURCE_CONTRACT | NEW_SOURCE_CONTRACT).items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected,
                'Reviewed transit source/replay changed: '+name)


def integer(value, low=0, high=10**15):
    require(type(value) is int and low <= value <= high, 'Invalid integer counter')
    return value


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid finite number')
    return value


def micros(seconds):
    return round(finite(seconds)*1_000_000)


def movement_upper(a, b):
    return math.ceil(math.dist(a, b)/5.*1_000_000)


def movement_lower(a, b):
    return math.floor(math.dist(a, b)/5.*1_000_000)


class Prefix:
    def __init__(self, record):
        self.record = record
        self.h, self.before = wire_prefix(record)
        self.cache = {}
        self.enter_count = sum(w['action'] == '/enter' and w['response'].get('accepted') is True
                               for w in record['history'])
        require(self.enter_count == 1, 'Audit requires one fresh actual session entry')
        enters = [w['response'] for w in record['history'] if w['action'] == '/enter'
                  and w['response'].get('accepted') is True]
        limit = enters[0].get('max_virtual_duration_s')
        self.virtual_limit = min(360000., limit) if limit is not None else 360000.
        self.max_actions = record['spec'].get('kwargs', {}).get('max_actions', 20000)
        integer(self.max_actions, 2, 1_000_000)

    def snapshot(self, n):
        integer(n, 0, len(self.h))
        if n not in self.cache:
            known, cleared, near, regions = set(), set(), {}, {}
            for a in self.h[:n]:
                c = a['channel']
                if a['action'] == 'measure':
                    if a['result'] in {'direction', 'near'}:
                        known.add(c)
                    if a['result'] == 'direction':
                        regions.setdefault(c, CandidateRegion()).observe(a['position'], a['bearing_deg'])
                    elif a['result'] == 'near':
                        near[c] = point(a['position'])
                elif a['result'] == 'success':
                    cleared.add(c)
                    known.add(c)
            self.cache[n] = known, cleared, near, regions
        return self.cache[n]

    def ready(self, n, c):
        _, _, near, regions = self.snapshot(n)
        r = regions.get(c)
        return c in near or bool(r and r.vertices and r.enclosing_disk().radius <= 19.9)

    def target(self, n, c):
        _, _, near, regions = self.snapshot(n)
        if c in near:
            return near[c]
        r = regions.get(c)
        return tuple(r.enclosing_disk().center) if r and r.vertices else None

    def original_early(self, n, q, blocked, attempted):
        if len(attempted) >= 4:
            return None
        known, cleared, _, regions = self.snapshot(n)
        p = self.before[n][0]
        candidates = []
        for c in sorted(known-cleared-blocked-attempted):
            if self.ready(n, c):
                continue
            r = regions.get(c)
            if not r or not r.vertices:
                continue
            circle = r.enclosing_disk()
            if not math.isfinite(circle.radius) or circle.radius > 40.:
                continue
            center = tuple(circle.center)
            detour = math.dist(p, center)+math.dist(center, q)-math.dist(p, q)
            if detour <= 100. and math.dist(p, center)/5.+6. <= 60.:
                candidates.append((detour, math.dist(p, center), c, circle.radius))
        return min(candidates) if candidates else None


def _range_lower(p, vertices):
    """Independent distance using segment projections, with original guard."""
    if not vertices:
        return 0.
    points = list(map(tuple, vertices))
    sides = list(zip(points, points[1:]+points[:1]))
    cross = [(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]) for a, b in sides]
    area = sum(a[0]*b[1]-a[1]*b[0] for a, b in sides)
    if len(points) >= 3 and abs(area) > 1e-12 and (all(x >= -1e-9 for x in cross) or all(x <= 1e-9 for x in cross)):
        return 0.
    lengths = []
    for a, b in sides:
        dx, dy = b[0]-a[0], b[1]-a[1]
        d2 = dx*dx+dy*dy
        t = min(1., max(0., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/d2)) if d2 else 0.
        lengths.append(math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy))
    scale = max(1., *map(abs, p), *(abs(v) for q in points for v in q))
    return max(0., min(lengths)-(1e-7+128*math.ulp(scale)))


def scan_channels(prefix, n, q):
    known, cleared, _, regions = prefix.snapshot(n)
    channels = [c for c in range(1, 21) if c not in cleared and not prefix.ready(n, c)]
    current = prefix.before[n][1]
    if current in channels:
        channels.remove(current)
        channels.insert(0, current)
    return [c for c in channels if not (c in known-cleared and c in regions
        and regions[c].observations and _range_lower(q, regions[c].vertices) > 1500.+1e-5)]


def check_chain(event, prefix, n, covers, channels, total_expanded, maximum):
    require(event['after_actual_action_count'] == n
            and list(map(point, event['remaining_covers'])) == covers
            and event['source_channels'] == channels, 'Chain inputs differ from actual known/ready prefix')
    targets = [prefix.target(n, c) for c in channels]
    require(list(map(point, event['source_positions'])) == targets, 'Chain source targets changed')
    known, cleared, _, _ = prefix.snapshot(n)
    require(event['discovery_count_cap'] == (len(known) == 16), 'Wrong chain discovery count cap')
    result = event['result']
    order = result['order']
    require(order and len(order) == len(covers)+len(channels), 'Incomplete chain order')
    cover_ids, source_ids, current, cost = [], [], prefix.before[n][0], 0.
    background = 6.*max(0, 20-len(cleared)-len(channels))
    for kind, i in order:
        integer(i)
        if kind == 'cover':
            require(i < len(covers), 'Invalid chain cover index')
            cover_ids.append(i)
            target, service = covers[i], background
        else:
            require(kind == 'source' and i < len(channels), 'Invalid chain source index')
            source_ids.append(i)
            target, service = targets[i], 5.
        cost += math.dist(current, target)/5.+service
        current = target
    require(cover_ids == list(range(len(covers))) and sorted(source_ids) == list(range(len(channels))),
            'Chain duplicates/omits/reorders fixed tasks')
    close(result['cost_s'], cost, 'Chain complete proxy cost differs')
    lower = finite(result['lower_bound_s'])
    require(-1e-7 <= lower <= cost+1e-7, 'Impossible chain proxy lower bound')
    require(type(result['exact']) is bool and (not result['exact'] or abs(lower-cost) <= 2e-6),
            'Incorrect exact chain gap')
    expanded = integer(result['expanded'], 0, max(0, min(maximum, 60000-total_expanded)))
    kind, i = order[0]
    require(event['selected_kind'] == kind and event['selected_channel'] == (channels[i] if kind == 'source' else None),
            'Chain first task differs from frozen order')
    return kind, channels[i] if kind == 'source' else None, total_expanded+expanded


def budget_numbers(prefix, start, n, q, action, target, channel, *, atomic=False):
    """Independent integer accounting, including reserve for real return scan."""
    integer(start, 0, len(prefix.h)); integer(n, start, len(prefix.h))
    require(action in {'measure', 'clear'} and type(channel) is int and 1 <= channel <= 20,
            'Bad proposed physical action')
    current, tuned, now = prefix.before[n]
    target, q = point(target), point(q)
    baseline = movement_lower(prefix.before[start][0], q)
    spent = micros(now)-micros(prefix.before[start][2])
    movement = movement_upper(current, target)
    if atomic:
        require(action == 'measure', 'Atomic R8 reserve must name its original measure')
        action_us, action_count = movement+8_000_000+1_000_000*(channel != tuned), 2
    else:
        action_us = movement+5_000_000+1_000_000*(action == 'measure' and channel != tuned)
        action_count = 1
    return_move = movement_upper(target, q)
    return_first = return_move+6_000_000
    extra = spent+action_us+return_first-baseline
    # Full scan still uses original actual channel rules; twenty radio calls
    # and 6 s each are only resource reserves, never fabricated observations.
    return dict(spent_us=spent, baseline_us=baseline, movement_us=movement,
                action_us=action_us, action_count=action_count,
                return_move_us=return_move, return_first_us=return_first,
                incremental_us=extra,
                reserve_actions=20, reserve_exit_actions=1,
                global_projected_actions=prefix.enter_count+n+action_count+20+1,
                global_projected_us=micros(now)+action_us+return_move+120_000_000,
                virtual_limit_us=micros(prefix.virtual_limit), max_actions=prefix.max_actions)


LIMITS = dict(macros=4, per_source=1, radius_min_exclusive_m=19.9,
    radius_max_m=40., projection_min=.1, projection_max=.9, detour_m=100.,
    old_first_cost_min_exclusive_s=60., incremental_limit_us=60_000_000,
    return_first_measure_upper_us=6_000_000, full_scan_reserve_us=120_000_000,
    full_scan_reserve_actions=20, exit_reserve_actions=1)


def same(actual, expected, message):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), message)
        for k, v in expected.items():
            same(actual[k], v, message+': '+k)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), message)
        for a, e in zip(actual, expected):
            same(a, e, message)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual) and abs(actual-expected) <= 1e-7, message)
    else:
        require(type(actual) is type(expected) and actual == expected, message)


def candidates(prefix, n, q, blocked, early, attempted):
    p = prefix.before[n][0]
    dx, dy = q[0]-p[0], q[1]-p[1]
    norm2 = dx*dx+dy*dy
    if not norm2:
        return []
    known, cleared, near, regions = prefix.snapshot(n)
    choices = []
    for c in sorted(known-cleared-blocked-early-attempted):
        r = regions.get(c)
        if c in near or prefix.ready(n, c) or not r or not r.vertices or not r.observations:
            continue
        disk = r.enclosing_disk()
        if not math.isfinite(disk.radius) or not 19.9 < disk.radius <= 40.:
            continue
        center = tuple(disk.center)
        distance = math.dist(p, center)
        t = ((center[0]-p[0])*dx+(center[1]-p[1])*dy)/norm2
        detour = distance+math.dist(center, q)-math.dist(p, q)
        first = distance/5.+6.
        if .1 <= t <= .9 and detour <= 100. and first > 60.:
            choices.append(dict(channel=c, center=list(center), radius_m=disk.radius,
                projection_fraction=t, detour_m=detour, approach_m=distance,
                original_first_cost_s=first))
    return sorted(choices, key=lambda e: (e['detour_m'], e['approach_m'], e['channel']))


def audit_gates(event, prefix, resolver, stops):
    h, before = prefix.h, prefix.before
    start, end, c = resolver['after_actual_action_count'], resolver['end_actual_action_count'], resolver['channel']
    q = point(event['destination'])
    gates = event['gates']
    action_gates, atomic_gates, rejected = {}, {}, []
    previous = start
    for i, g in enumerate(gates):
        n = integer(g['after_actual_action_count'], previous, end)
        previous = n
        require(g['id'] == i and g['kind'] in {'action', 'atomic_r8'} and g['channel'] == c,
                'Wrong transit gate identity/source')
        atomic = g['kind'] == 'atomic_r8'
        b = budget_numbers(prefix, start, n, q, g['action'], g['point'], c, atomic=atomic)
        expected = dict(current_position=list(before[n][0]), current_channel=before[n][1],
            current_virtual_us=micros(before[n][2]), spent_us=b['spent_us'],
            movement_upper_us=b['movement_us'], action_fee_upper_us=b['action_us']-b['movement_us'],
            action_upper_us=b['action_us'], predicted_actions=b['action_count'],
            return_upper_us=b['return_move_us'], baseline_floor_us=b['baseline_us'],
            incremental_upper_us=b['incremental_us'], incremental_limit_us=60_000_000,
            policy_action_count=prefix.enter_count+n, max_actions=prefix.max_actions,
            full_scan_reserve_actions=20, exit_reserve_actions=1,
            full_scan_reserve_us=120_000_000, return_first_measure_upper_us=6_000_000,
            virtual_limit_us=math.floor(prefix.virtual_limit*1_000_000)-1)
        for k, value in expected.items():
            same(g[k], value, 'Transit budget arithmetic differs: '+k)
        real = g['remaining_real_s']
        require(real is None or type(real) in (int, float) and math.isfinite(real), 'Bad real-time snapshot')
        if atomic:
            region = prefix.snapshot(n)[3].get(c)
            require(region and region.observations and 19.9 < region.enclosing_disk().radius <= 40.
                    and point(g['point']) == tuple(region.enclosing_disk().center)
                    and c not in prefix.snapshot(n)[2], 'Atomic gate lacks original R8 eligibility')
            key = tuple(round(x, 6) for x in point(g['point']))
            require(not any(a['channel'] == c and (a['phase'] == 'speculative_clear_before_probe'
                or a['action'] == 'measure' and tuple(round(x, 6) for x in point(a['position'])) == key)
                for a in h[:n]), 'Atomic R8 gate repeats prior actual attempt/probe')
            require(prefix.enter_count+n+3 <= prefix.max_actions
                    and before[n][2]+b['action_us']/1_000_000 <= prefix.virtual_limit-1e-6
                    and (real is None or real > 2.), 'Atomic gate ran after parent insertion budget refused')
        reasons = []
        if b['incremental_us'] > 60_000_000:
            reasons.append('incremental_budget')
        if b['global_projected_actions'] > prefix.max_actions:
            reasons.append('return_scan_action_reserve')
        if b['global_projected_us'] > expected['virtual_limit_us']:
            reasons.append('return_scan_virtual_reserve')
        same(g['reasons'], reasons, 'Transit rejection reasons differ')
        require(type(g['admitted']) is bool and g['admitted'] == (not reasons), 'Forged gate admission')
        executed = 0
        if g['admitted'] and n < end:
            a = h[n]
            if atomic:
                require(a['action'] == 'clear' and a['phase'] == 'speculative_clear_before_probe'
                        and a['channel'] == c and point(a['position']) == point(g['point']),
                        'Atomic gate did not precede real R8 clear')
                executed = 1
                if a['result'] != 'success' and n+1 < end:
                    a2 = h[n+1]
                    require(a2['action'] == 'measure' and a2['phase'] == 'active_localization'
                            and a2['channel'] == c and point(a2['position']) == point(g['point']),
                            'Atomic R8 reservation did not preserve same-point original measure')
                    executed = 2
                require(n not in atomic_gates, 'Repeated atomic gate for one action')
                atomic_gates[n] = g
            else:
                require(a['action'] == g['action'] and a['channel'] == c
                        and point(a['position']) == point(g['point']), 'Gate does not bind the next real action')
                require(n not in action_gates, 'Repeated real-action gate')
                action_gates[n] = g
                executed = 1
        elif not g['admitted']:
            require(n == end and i == len(gates)-1, 'Rejected gate followed by service action/gate')
            rejected.append(g)
        else:
            require(n == end and terminal(prefix.record['summary'], end, h), 'Admitted unexecuted gate lacks real terminal')
        require(type(g['executed_action_count']) is int and g['executed_action_count'] == executed,
                'Gate executed-action count differs from wire')
    require(set(action_gates) == set(range(start, end)), 'Actual service action lacks exactly one physical gate')
    expected_atomic = {n for n in range(start, end) if h[n]['phase'] == 'speculative_clear_before_probe'}
    require(set(atomic_gates) == expected_atomic, 'Missing/extra R8 atomic pre-reservation')
    # The two nested reservations at one prefix have a fixed causal order.
    for n, g in atomic_gates.items():
        require(g['id'] < action_gates[n]['id'], 'Atomic reserve occurred after clear gate')
    if event['service_status'] == 'slice_expired':
        require(len(rejected) == 1 and event['stopped_gate_id'] == rejected[0]['id']
                and event['stopped_reasons'] == rejected[0]['reasons']
                and resolver['status'] == 'interrupted'
                and resolver.get('interruption_type') == '_TransitSliceExpired', 'Unbound transit interruption')
        g = rejected[0]
        if g['kind'] == 'atomic_r8':
            region = prefix.snapshot(end)[3].get(c)
            require(region and region.observations and 19.9 < region.enclosing_disk().radius <= 40.
                    and point(g['point']) == tuple(region.enclosing_disk().center)
                    and g['action'] == 'measure', 'Rejected atomic reservation lacks actual R8 centre eligibility')
        stops.add((end, c, g['action'], point(g['point'])))
    else:
        require(not rejected and 'stopped_gate_id' not in event and 'stopped_reasons' not in event,
                'Non-slice service claims a rejected budget gate')
    return end-start


def replay_macros(record):
    prefix = Prefix(record)
    summary, h = record['summary'], prefix.h
    params = summary['strategy_parameters']
    require(params['transit_budget_config'] == CONFIG and params['transit_budget_limits'] == LIMITS,
            'Changed transit configuration/limits')
    points = list(map(point, summary['coverage_points']))
    require(len(points) == 22 and len(set(points)) == 22 and points[0] == (0., 0.), 'Wrong fixed cover inventory')
    # Cover coordinate identity is independent of the policy's logged report.
    from planning.q4_directional_cover import certified_cover_points
    original, _ = certified_cover_points('compact_22')
    require(points == [(p.x, p.y) for p in original], 'Original 22-point coordinate/order changed')
    events = params['transit_service_log']
    epochs, early, chains = (params[k] for k in ('joint_visibility_resolver_log', 'early_service_log', 'chain_route_log'))
    ni = ei = si = ci = visited = n = total_expanded = service_actions = services = interrupted = 0
    attempted, early_attempted, blocked, stops = set(), set(), set(), set()
    maximum = record['spec'].get('kwargs', {}).get('max_expansions', 200)
    integer(maximum, 0, 10000)

    def consume_resolver(index, start, channel):
        require(index < len(epochs), 'Missing actual resolver epoch')
        e = epochs[index]
        require(e['id'] == index and e['channel'] == channel and e['after_actual_action_count'] == start,
                'Resolver belongs to a different macro/prefix')
        end = integer(e['end_actual_action_count'], start, len(h))
        require(all(a['channel'] == channel and a['phase'] != 'coverage' for a in h[start:end]),
                'Resolver contains another source/coverage')
        return e, end

    # Each iteration consumes a resolver, an early attempt or a complete scan.
    # Allow zero-action epochs at a shared prefix; their identity/order, rather
    # than comparisons against future intervals, determine macro ownership.
    for _ in range(len(h)+len(epochs)+len(events)+len(early)+5):
        known, cleared, _, _ = prefix.snapshot(n)
        if len(cleared) == 16:
            break
        covers = [] if len(known) == 16 else points[visited:]
        old = prefix.original_early(n, covers[0], blocked, early_attempted) if covers else None
        if old is not None:
            require(si < len(early), 'Missing original-priority early service')
            e = early[si]
            require(e['after_actual_action_count'] == n and e['channel'] == old[2], 'Original early priority/selection changed')
            resolver, end = consume_resolver(ei, n, old[2])
            require(e['end_actual_action_count'] == end, 'Early/parent resolver intervals disagree')
            early_attempted.add(old[2]); si += 1; ei += 1; n = end
            if resolver['status'] == 'interrupted' and not e['interrupted']:
                require(terminal(summary, n, h), 'Unexplained early-service interruption')
                break
            continue
        channels = [c for c in sorted(known-cleared-blocked) if prefix.target(n, c) is not None
                    and (not covers or prefix.ready(n, c))]
        if channels:
            require(ci < len(chains), 'Missing original chain decision')
            kind, source, total_expanded = check_chain(chains[ci], prefix, n, covers, channels, total_expanded, maximum)
            ci += 1
            if kind == 'source':
                resolver, end = consume_resolver(ei, n, source)
                ei += 1; n = end
                if resolver['status'] == 'unresolved':
                    blocked.add(source)
                elif resolver['status'] == 'interrupted':
                    require(terminal(summary, n, h), 'Main source task interrupted without terminal')
                    break
                continue
        elif not covers:
            break
        require(covers and ni < len(events), 'Selected cover omitted its transit hook event')
        event, q = events[ni], covers[0]
        require(event['id'] == ni and event['after_actual_action_count'] == n
                and point(event['origin']) == prefix.before[n][0] and point(event['destination']) == q,
                'Transit hook is not the original selected cover edge')
        require(event['coverage_visited_before'] == visited and event['known_channels'] == sorted(known)
                and event['early_attempted_before'] == sorted(early_attempted)
                and event['transit_attempted_before'] == sorted(attempted), 'Stale transit macro prefix')
        require(event['start_virtual_us'] == micros(prefix.before[n][2])
                and event['baseline_floor_us'] == movement_lower(prefix.before[n][0], q), 'Changed baseline/clock')
        skip = ('discovery_count_cap' if len(known) >= 16 else 'same_position'
                if prefix.before[n][0] == q else 'macro_limit' if len(attempted) >= 4 else None)
        expected_candidates = [] if skip else candidates(prefix, n, q, blocked, early_attempted, attempted)
        if skip is None and not expected_candidates:
            skip = 'no_candidate'
        require(event['old_early_candidate'] is None, 'Transit ran despite original early priority')
        same(event['candidates'], expected_candidates, 'Candidate geometry/eligibility differs')
        expected_selected = expected_candidates[0] if expected_candidates else None
        same(event['selected'], expected_selected, 'Transit selection differs from fixed ranking')
        require(event['skip_reason'] == skip, 'Wrong transit skip reason')
        start = n
        if expected_selected is not None:
            c = expected_selected['channel']
            require(c not in attempted and len(attempted) < 4 and event['resolver_id'] == ei
                    and event['resolver_start_action_count'] == n, 'Transit repeat/cap/resolver binding failed')
            attempted.add(c)
            resolver, n = consume_resolver(ei, start, c)
            ei += 1; services += 1
            require(event['service_end_action_count'] == n
                    and event['service_end_virtual_us'] == micros(prefix.before[n][2])
                    and point(event['service_end_position']) == prefix.before[n][0], 'Wrong service exit prefix')
            require(event['service_status'] in {'cleared', 'unresolved', 'slice_expired', 'interrupted'}, 'Unfinished service')
            if event['service_status'] in {'cleared', 'unresolved'}:
                require(resolver['status'] == event['service_status'], 'Service outcome disagrees with parent resolver')
            elif event['service_status'] == 'interrupted':
                require(resolver['status'] == 'interrupted' and terminal(summary, n, h), 'Unexplained global service interruption')
            service_actions += audit_gates(event, prefix, resolver, stops)
            interrupted += event['service_status'] == 'slice_expired'
        else:
            require(event['resolver_id'] is None and event['resolver_start_action_count'] is None
                    and event['service_status'] == 'not_started' and event['service_end_action_count'] == n
                    and event['gates'] == [], 'Skipped transit fabricated service/budget activity')
        require(event['transit_attempted_after'] == sorted(attempted), 'Attempt ledger changed')
        if event['scan_start_action_count'] is None:
            require(event['service_status'] == 'interrupted' and terminal(summary, n, h)
                    and event['scan_status'] == 'not_started' and event['scan_end_action_count'] is None,
                    'Service omitted forced return without global terminal')
            require(event['first_coverage_action_index'] is None, 'Fabricated first return action')
        else:
            require(event['scan_start_action_count'] == n, 'Another task inserted before forced return scan')
            expected = scan_channels(prefix, n, q)
            scan_end = integer(event['scan_end_action_count'], n, len(h))
            actual = h[n:scan_end]
            require(len(actual) <= len(expected), 'Repeated/excess coverage measurements')
            for a, c in zip(actual, expected):
                require(a['action'] == 'measure' and a['phase'] == 'coverage' and a['channel'] == c
                        and point(a['position']) == q, 'Forced return/complete scan channel sequence differs')
            if event['scan_status'] == 'completed':
                require(len(actual) == len(expected) and actual, 'Incomplete/fictional completed scan')
                visited += 1
                blocked.clear()
            else:
                require(event['scan_status'] == 'interrupted' and terminal(summary, scan_end, h), 'Scan stopped without global terminal')
            if actual:
                require(event['first_coverage_action_index'] == n
                        and event['first_coverage_virtual_us'] == micros(h[n]['virtual_time_s']), 'Fake return first-measure evidence')
                if expected_selected:
                    extra = micros(h[n]['virtual_time_s'])-event['start_virtual_us']-event['baseline_floor_us']
                    require(event['actual_incremental_through_first_measure_us'] == extra and extra <= 60_000_000,
                            'Real service plus returned first measurement exceeds incremental bound')
            else:
                require(event['first_coverage_action_index'] is None, 'Unexecuted return falsely counted')
            n = scan_end
        require(event['end_actual_action_count'] == n and event['coverage_visited_after'] == visited,
                'Premature cover pop/visited count')
        require(finite(event['runtime_s']) >= 0 and finite(event['decision_wall_s']) >= 0, 'Bad CPU runtime log')
        ni += 1
        if event['scan_status'] != 'completed':
            break
    else:
        raise ValueError('Macro replay failed to make bounded progress')
    require(n == len(h) and ei == len(epochs) and ni == len(events)
            and si == len(early) and ci == len(chains), 'Missing/orphan actual macro or recorded decision')
    require(summary['coverage_points_visited'] == visited, 'Final visited count differs from completed scans')
    return prefix, stops, dict(transit_services=services, transit_service_actions=service_actions,
        completed_scans=visited, locally_interrupted_services=interrupted,
        chain_decisions=ci, original_astar_expanded=total_expanded)


def canonical_next_action(prefix, resolver):
    """Unexecuted canonical actions are not covered by the R12 aux checker.

    Reconstruct that frozen resolver's next action, including its common
    rotated-grid suffix. No gate's claimed point is used to choose it.
    """
    start, end, c = resolver['after_actual_action_count'], resolver['end_actual_action_count'], resolver['channel']
    h = prefix.h
    _, cleared, near, regions = prefix.snapshot(end)
    require(c not in cleared, 'Budget-stop source was already cleared')
    if c in near:
        return 'clear', near[c]
    region = regions.get(c)
    require(region and region.observations and region.vertices, 'Canonical stop has no live region')
    disk = region.enclosing_disk()
    tail = [j for j in range(start, end) if h[j]['phase'] == 'guaranteed_clearance']
    if not tail:
        key = tuple(round(x, 6) for x in disk.center)
        attempted = {tuple(round(x, 6) for x in point(a['position'])) for a in h[start:end]
                     if a['phase'] == 'certified_clear'}
        if disk.radius <= 19.9 and key not in attempted:
            return 'clear', tuple(disk.center)
        probes = sum(a['action'] == 'measure' and a['phase'] == 'active_localization' for a in h[start:end])
        maximum = prefix.record['spec'].get('kwargs', {}).get('max_active_probes', 6)
        require(probes <= maximum, 'Canonical resolver exceeded active-probe budget')
        if probes < maximum:
            choices, _, selected = probe_formula(region, region.observations[0].bearing_deg,
                                                  prefix.before[end][0], h, end, c)
            if selected is not None:
                return 'measure', choices[selected]
    # Independent closed-strip intersections after the same float rotation,
    # as in the frozen R12 geometric checker; full list, not a node oracle.
    from fractions import Fraction as F
    angle = math.radians(region.observations[0].bearing_deg)
    co, si = math.cos(angle), math.sin(angle)
    polygon = [(F(x*co+y*si), F(-x*si+y*co)) for x, y in region.vertices]
    grid = []
    for row in range(math.floor(min(p[1] for p in polygon)/28), math.floor(max(p[1] for p in polygon)/28)+1):
        low, high = F(row*28), F((row+1)*28)
        cloud = {p for p in polygon if low <= p[1] <= high}
        for a, b in zip(polygon, polygon[1:]+polygon[:1]):
            if a[1] != b[1]:
                for y in (low, high):
                    if min(a[1], b[1]) <= y <= max(a[1], b[1]):
                        cloud.add((a[0]+(y-a[1])/(b[1]-a[1])*(b[0]-a[0]), y))
        if cloud:
            for col in range(math.floor(min(p[0] for p in cloud)/28), math.floor(max(p[0] for p in cloud)/28)+1):
                x, y = (col+.5)*28., (row+.5)*28.
                grid.append((x*co-y*si, x*si+y*co))
    require(grid, 'Canonical optical suffix is empty')
    current = prefix.before[tail[0]][0] if tail else prefix.before[end][0]
    for j in tail:
        target = min(grid, key=lambda p: (math.dist(current, p), p[0], p[1]))
        require(h[j]['action'] == 'clear' and h[j]['result'] == 'no_target_in_range'
                and math.dist(point(h[j]['position']), target) <= 1e-7,
                'Canonical optical prefix differs from shared remaining-grid route')
        grid.remove(target)
        current = point(h[j]['position'])
    require(grid, 'Canonical optical route fully exhausted before claimed budget stop')
    return 'clear', min(grid, key=lambda p: (math.dist(current, p), p[0], p[1]))


def audit_transit_budget_prefix(record):
    verify_source_contract()
    spec = record['spec']
    require(spec['entrypoint'] == ENTRY and record['row']['strategy'] == LABEL, 'Unreviewed transit entry/label')
    kwargs = spec.get('kwargs', {})
    require(set(kwargs) <= {'config', 'max_expansions', 'max_actions', 'max_active_probes', 'problem'}
            and kwargs.get('config') == CONFIG and kwargs.get('problem', 4) == 4
            and kwargs.get('max_expansions', 200) == 200, 'Unreviewed transit spec')
    prefix, stops, result = replay_macros(record)
    # Only an independently checked rejected gate can extend the parent's
    # stopping semantics. The private function globals do not mutate modules.
    used_stops = set()
    def stopped(summary, h, before, end, channel, action, position):
        if original_stopped(summary, h, before, end, channel, action, position):
            return True
        key = (end, channel, action, point(position))
        if key in stops:
            used_stops.add(key)
            return True
        return False
    replay_globals = dict(audit_joint_continuation_prefix.__globals__, stopped=stopped)
    checker = FunctionType(audit_joint_continuation_prefix.__code__, replay_globals,
                           audit_joint_continuation_prefix.__name__, audit_joint_continuation_prefix.__defaults__,
                           audit_joint_continuation_prefix.__closure__)
    view = dict(record, spec=dict(spec, entrypoint='strategies.q4_joint_continuation:run_q4_joint_continuation',
                                kwargs=dict(kwargs, config='after_active_miss_optical')))
    r12 = checker(view)
    for key in stops-used_stops:
        end, c, action, target = key
        matches = [e for e in record['summary']['strategy_parameters']['transit_service_log']
                   if e['service_status'] == 'slice_expired' and e['service_end_action_count'] == end
                   and e['selected']['channel'] == c]
        require(len(matches) == 1, 'Canonical stop not bound to one real transit resolver')
        epoch = record['summary']['strategy_parameters']['joint_visibility_resolver_log'][matches[0]['resolver_id']]
        expected_action, expected_point = canonical_next_action(prefix, epoch)
        require(action == expected_action and math.dist(target, expected_point) <= 1e-7,
                'Rejected gate was not the actual canonical resolver next action')
    r8 = audit_clear_before_probe_prefix(view)
    range_result = audit_range_prefix(view)
    scheduling_view = dict(view, history=[w for w in view['history']
        if w['action'] not in {'/measure', '/clear'} or w['response'].get('accepted') is True])
    scheduling = audit_scheduling_prefix(scheduling_view)
    for value in (r12, r8, range_result, scheduling):
        require(value.get('passed') is True, 'Inherited transit audit did not pass')
    result.update(passed=True, r12=r12, r8=r8, range=range_result, scheduling=scheduling,
        source_contract_files=len(SOURCE_CONTRACT)+len(NEW_SOURCE_CONTRACT),
        boundary='Actual-prefix macro, integer reserve and first-return bound; not full-session dominance. Generic physical/coverage/terminal audit remains required.')
    return result


def audit_full(record):
    from experiments.audit_q4_cover import audit_record
    result = audit_record(record)
    require(result.get('passed') is True, 'Generic physical/coverage audit failed: '+str(result.get('errors')))
    prefix = audit_transit_budget_prefix(record)
    return dict(passed=True, generic=result, prefix=prefix)
