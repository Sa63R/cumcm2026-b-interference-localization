"""Independent observed-state matrix routing and complete R12-prefix audit.

The small wire/snapshot/range helpers reuse the reviewed R33 audit formulas;
there is no R33 runtime import. No new policy or matrix planner is imported.
Only the inherited R12 entry spec is normalized after source identity and all
new scheduling logic have independently passed. Actual history is unchanged.
"""
import hashlib
import math
from pathlib import Path
from functools import lru_cache

from localization import CandidateRegion
from experiments.audit_q4_clear_before_probe import require, point, close, terminal, audit_clear_before_probe_prefix
from experiments.audit_q4_joint_continuation import wire_prefix, audit_joint_continuation_prefix
from experiments.audit_q4_scheduling import audit_scheduling_prefix
from experiments.audit_q4_range import audit_range_prefix

ROOT=Path(__file__).resolve().parents[1]
ENTRY='strategies.q4_known_source:run_q4_known_source'
LABEL='compact_known_source'
CONFIG='all_known_matrix'
SOURCE_CONTRACT={'experiments/audit_q4_clear_before_probe.py': '68c7ef07dca4cc9f7a60351332911c25054c5e0392220830637bfd89518e62c4',
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
NEW_SOURCE_CONTRACT={
 'src/planning/matrix_chain_route.py': '14f36e6307a6874239f6d564366519d9e46713f3bf6af8f05cf26334ec43fe40',
 'src/strategies/q4_known_source.py': '1591f7888ae3c82e78e9aab797631ae2f1fc05f4d57de03e143806d5ae25d5a8'}

def integer(value, low=0, high=10**15):
    require(type(value) is int and low <= value <= high, 'Invalid integer counter')
    return value


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid finite number')
    return value


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


def verify_source_contract(root=ROOT):
    require(bool(NEW_SOURCE_CONTRACT), 'Known-source implementation not yet reviewed/frozen')
    for name, expected in (SOURCE_CONTRACT | NEW_SOURCE_CONTRACT).items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected,
                'Reviewed source/replay changed: '+name)


def fee_at(prefix, n, q, c):
    """A frozen scan proxy, not a claim of an actual measurement/switch."""
    known, cleared, _, regions = prefix.snapshot(n)
    if c in cleared:
        return 0.
    if prefix.ready(n, c):
        return 0.
    r = regions.get(c)
    if c in known and r and r.observations and _range_lower(q, r.vertices) > 1500.+1e-5:
        return 0.
    return 6.


def frozen_matrix(prefix, n, covers, channels):
    _, cleared, _, _ = prefix.snapshot(n)
    require(len(channels) <= 16 and len(channels) == len(set(channels)), 'Invalid planned source set')
    other = [c for c in range(1, 21) if c not in cleared and c not in channels]
    matrix = [[fee_at(prefix, n, q, c) for c in channels] for q in covers]
    background = [sum(fee_at(prefix, n, q, c) for c in other) for q in covers]
    return other, background, matrix


def matrix_route_cost(covers, sources, start, order, background, matrix):
    """Evaluate a complete common order by actual source identity, never count."""
    require(len(background) == len(covers) and len(matrix) == len(covers)
            and all(len(row) == len(sources) for row in matrix), 'Matrix dimensions differ')
    require(all(finite(v) >= 0 for v in background)
            and all(finite(v) >= 0 for row in matrix for v in row), 'Negative/nonfinite proxy cost')
    require(len(order) == len(covers)+len(sources), 'Incomplete matrix route')
    covered, completed, current, cost = 0, set(), point(start), 0.
    for kind, index in order:
        integer(index)
        if kind == 'cover':
            require(index == covered and index < len(covers), 'Coverage chain reordered/duplicated')
            target = point(covers[index])
            fee = background[index]+sum(matrix[index][i] for i in range(len(sources)) if i not in completed)
            covered += 1
        else:
            require(kind == 'source' and index < len(sources) and index not in completed,
                    'Source task duplicated/invalid')
            target, fee = point(sources[index]), 0.
            completed.add(index)
        cost += math.dist(current, target)/5.+fee
        current = target
    require(covered == len(covers) and len(completed) == len(sources), 'Matrix route omits a task')
    return cost


def root_relaxation(covers, sources, start, background):
    """Independent initial-state lower bound of the frozen point-task proxy."""
    cover_length, current = 0., point(start)
    for p in covers:
        cover_length += math.dist(current, p)
        current = p
    source_length = 0.
    if sources:
        source_length = min(math.dist(start, p) for p in sources)
        connected, remaining = {0}, set(range(1, len(sources)))
        while remaining:
            length, j = min((math.dist(sources[i], sources[j]), j) for i in connected for j in remaining)
            source_length += length
            connected.add(j); remaining.remove(j)
    return max(cover_length, source_length)/5.+sum(background)


def _finite_runtime(value):
    require(finite(value) >= 0, 'Invalid decision runtime')


def small_optimum(covers, sources, start, background, matrix):
    """Independent backward DAG, deliberately restricted to four sources."""
    require(len(sources) <= 4, 'Independent exact check is small-only')
    @lru_cache(None)
    def suffix(mask, k, last):
        if mask == (1 << len(sources))-1 and k == len(covers):
            return 0.
        here = start if last == -1 else sources[last] if last < len(sources) else covers[last-len(sources)]
        choices = [math.dist(here, p)/5.+suffix(mask | (1 << j), k, j)
                   for j, p in enumerate(sources) if not mask & (1 << j)]
        if k < len(covers):
            fee = background[k]+sum(matrix[k][j] for j in range(len(sources)) if not mask & (1 << j))
            choices.append(math.dist(here, covers[k])/5.+fee+suffix(mask, k+1, len(sources)+k))
        return min(choices)
    return suffix(0, 0, -1)


def evidence(prefix, n, c, blocked):
    known, _, near, regions = prefix.snapshot(n)
    r = regions.get(c)
    circle = r.enclosing_disk() if r and r.vertices else None
    target = prefix.target(n, c)
    return dict(channel=c, detected=c in known, blocked=c in blocked,
                target=list(target) if target is not None else None,
                ready=prefix.ready(n, c), radius_m=circle.radius if circle else None,
                near_point=list(near[c]) if c in near else None,
                vertices=[list(v) for v in r.vertices] if r else [],
                positive_observation_count=len(r.observations) if r else 0)


def check_plan(event, prefix, n, covers, channels, blocked, total, index, maximum):
    known, cleared, _, regions = prefix.snapshot(n)
    current, tuned, _ = prefix.before[n]
    targets = [prefix.target(n, c) for c in channels]
    expected = dict(id=index, config=CONFIG, after_actual_action_count=n,
        current_position=list(current), current_channel=tuned,
        known_channels=sorted(known), cleared_channels=sorted(cleared), blocked_channels=sorted(blocked),
        remaining_covers=[list(p) for p in covers], source_channels=channels,
        source_positions=[list(p) for p in targets], source_services_s=[0.]*len(channels),
        source_evidence=[evidence(prefix,n,c,blocked) for c in channels],
        discovery_count_cap=len(known)==16)
    other, background, matrix = frozen_matrix(prefix,n,covers,channels)
    expected.update(background_channels=other, background_scan_s=background, source_scan_s=matrix,
                    background_evidence=[evidence(prefix,n,c,blocked) for c in other])
    cells = []
    for q in covers:
        row=[]
        for c in range(1,21):
            if c in cleared: continue
            r=regions.get(c); lower=None; reason='measurement_proxy'
            if prefix.ready(n,c): reason='ready'
            elif c in known and r and r.observations:
                lower=_range_lower(q,r.vertices)
                if lower>1500.+1e-5: reason='positive_region_beyond_max_reception_radius'
            row.append(dict(channel=c,fee_s=fee_at(prefix,n,q,c),reason=reason,distance_lower_bound_m=lower))
        cells.append(row)
    expected['matrix_evidence']=cells
    for k,v in expected.items(): same(event[k],v,'Plan differs from real prefix: '+k)
    budget=max(0,min(maximum,60000-total))
    require(event['max_expansions']==budget and event['total_expansions_before']==total,'A* budget prefix differs')
    result=event['result']; order=result['order']
    cost=matrix_route_cost(covers,targets,current,order,background,matrix)
    close(result['cost_s'],cost,'Complete common matrix route cost differs')
    lower=finite(result['lower_bound_s'])
    require(-1e-7<=lower<=cost+1e-7 and lower+2e-6>=root_relaxation(covers,targets,current,background),
            'Impossible frozen-model lower bound')
    require(type(result['exact']) is bool and (not result['exact'] or abs(lower-cost)<=2e-6),'Wrong exact-gap claim')
    if len(channels)<=4:
        optimum=small_optimum(covers,targets,current,background,matrix)
        require(lower<=optimum+2e-6 and cost>=optimum-2e-6
                and (not result['exact'] or abs(cost-optimum)<=2e-6),'Independent small DAG disagrees')
    expanded=integer(result['expanded'],0,budget)
    for k in ('generated','dominance_pruned','bound_pruned'): integer(result[k])
    _finite_runtime(result['runtime_s']); _finite_runtime(event['runtime_s'])
    require(event['total_expansions_after']==total+expanded,'A* cumulative counter differs')
    kind,j=order[0]; channel=channels[j] if kind=='source' else None
    require(event['selected_kind']==kind and event['selected_channel']==channel,'Executed task differs from common route')
    return kind,channel,total+expanded


def replay_macros(record):
    prefix=Prefix(record); h=prefix.h; summary=record['summary']; params=summary['strategy_parameters']
    points=list(map(point,summary['coverage_points']))
    chains=params['known_source_plan_log']; services=params['known_source_service_log']
    require(chains==params['chain_route_log'],'Two plan ledgers differ')
    epochs=params['joint_visibility_resolver_log']; early=params['early_service_log']
    maximum=record['spec'].get('kwargs',{}).get('max_expansions',200)
    n=visited=ci=si=ei=mi=total=broad=service_actions=0
    blocked=set(); attempted=set()

    def consume_resolver(index,start,c):
        require(index<len(epochs),'Missing actual resolver')
        e=epochs[index]
        require(e['id']==index and e['channel']==c and e['after_actual_action_count']==start,
                'Resolver identity/prefix differs')
        end=integer(e['end_actual_action_count'],start,len(h))
        require(all(a['channel']==c and a['phase']!='coverage' for a in h[start:end]),
                'Source interval includes coverage/another source')
        status=e['status']
        require(status in {'cleared','unresolved','interrupted'},'Unfinished resolver')
        if status!='interrupted':
            require((c in prefix.snapshot(end)[1])==(status=='cleared'),'Resolver status contradicts actual clearance')
        return e,end

    # Zero-action resolver/early epochs are consumed by identity, never by a
    # comparison with later intervals which can start at the same prefix.
    for _ in range(len(h)+len(chains)+len(epochs)+len(early)+5):
        known,cleared,_,_=prefix.snapshot(n)
        if len(cleared)==16: break
        covers=[] if len(known)==16 else points[visited:]
        old=prefix.original_early(n,covers[0],blocked,attempted) if covers else None
        if old is not None:
            require(si<len(early),'Missing priority early service')
            event=early[si]
            require(event['after_actual_action_count']==n and event['channel']==old[2],'Changed original early priority')
            epoch,end=consume_resolver(ei,n,old[2])
            require(event['end_actual_action_count']==end,'Early/resolver ranges disagree')
            attempted.add(old[2]); si+=1; ei+=1; n=end
            if epoch['status']=='interrupted' and not event['interrupted']:
                require(terminal(summary,n,h),'Unexplained early interruption'); break
            continue
        channels=[c for c in sorted(known-cleared-blocked) if prefix.target(n,c) is not None]
        event=None
        if channels:
            require(ci<len(chains),'Missing all-known matrix decision')
            event=chains[ci]
            kind,c,total=check_plan(event,prefix,n,covers,channels,blocked,total,ci,maximum)
            ci+=1
            if kind=='source':
                require(mi<len(services),'Missing selected-source macro')
                service=services[mi]; region=prefix.snapshot(n)[3].get(c)
                radius=region.enclosing_disk().radius if region and region.vertices else None
                ready=prefix.ready(n,c); start=n
                expected=dict(decision_id=event['id'],selected=dict(channel=c,ready_before=ready,radius_m=radius),
                    remaining_covers_before=[list(p) for p in covers],resolver_start_action_count=n,
                    resolver_id=ei,start_virtual_time_s=prefix.before[n][2],start_position=list(prefix.before[n][0]))
                for k,v in expected.items(): same(service[k],v,'Source macro differs: '+k)
                epoch,n=consume_resolver(ei,start,c); ei+=1; mi+=1
                require(service['service_end_action_count']==n and event['end_actual_action_count']==n,'Wrong service exit')
                same(service['end_position'],list(prefix.before[n][0]),'Wrong actual exit position')
                close(service['actual_cost_s'],prefix.before[n][2]-prefix.before[start][2],'Wrong actual service cost')
                status={'cleared':'resolved','unresolved':'blocked','interrupted':'interrupted'}[epoch['status']]
                require(service['status']==status and event['status']==status
                        and service['cleared']==(c in prefix.snapshot(n)[1]),'Source completion or blockage fabricated')
                service_actions+=n-start
                if not ready and radius is not None and radius>40. and covers: broad+=n-start
                if status=='blocked': blocked.add(c)
                if status=='interrupted':
                    require(terminal(summary,n,h),'Full source resolver interrupted without terminal'); break
                continue
        elif not covers: break
        require(covers,'Cover selection without remaining obligation')
        q=covers[0]; expected=scan_channels(prefix,n,q); start=n
        for c in expected:
            if n==len(h): break
            a=h[n]
            require(a['action']=='measure' and a['phase']=='coverage' and a['channel']==c
                    and point(a['position'])==q,'Selected cover channel/position/order differs')
            n+=1
        complete=n-start==len(expected)
        if complete:
            require(bool(expected),'Fictional zero-action complete cover')
            visited+=1; blocked.clear()
        else: require(terminal(summary,n,h),'Incomplete real coverage scan without terminal')
        if event is not None:
            require(event['end_actual_action_count']==n and event['status']==('cover_completed' if complete else 'interrupted'),
                    'Cover plan exit/status differs')
        if not complete: break
    else: raise ValueError('Macro replay failed bounded progress')
    require(n==len(h) and ci==len(chains) and si==len(early) and ei==len(epochs) and mi==len(services),
            'Missing/orphan actual action, plan or resolver macro')
    require(summary['coverage_points_visited']==visited,'Visited count differs from complete scans')
    return dict(known_source_macros=mi,source_service_actions=service_actions,broad_service_actions=broad,
                completed_scans=visited,chain_decisions=ci,original_astar_expanded=total)


def audit_known_source_prefix(record):
    verify_source_contract()
    spec=record['spec']; kwargs=spec.get('kwargs',{})
    require(spec['entrypoint']==ENTRY and record['row']['strategy']==LABEL,'Unreviewed known-source entry/label')
    require(set(kwargs)<={'config','max_expansions','max_actions','max_active_probes','problem'}
            and kwargs.get('config')==CONFIG and type(kwargs.get('problem',4)) is int and kwargs.get('problem',4)==4
            and type(kwargs.get('max_expansions',200)) is int and kwargs.get('max_expansions',200)==200,'Unreviewed known-source spec')
    params=record['summary']['strategy_parameters']
    require(params['known_source_config']==CONFIG,'Logged configuration differs')
    result=replay_macros(record)
    # Only the verified new scheduler is normalized. No real action, parent
    # epoch, budget, geometry, coverage point or outcome is altered/removed.
    view=dict(record,spec=dict(spec,entrypoint='strategies.q4_joint_continuation:run_q4_joint_continuation',
                              kwargs=dict(kwargs,config='after_active_miss_optical')))
    r12=audit_joint_continuation_prefix(view); r8=audit_clear_before_probe_prefix(view)
    scheduling_view=dict(view,history=[w for w in view['history'] if w['action'] not in {'/measure','/clear'}
                                      or w['response'].get('accepted') is True])
    # These two legacy readers count all supplied physical wire items. The
    # accepted-only view is explicit; full wire/terminal checks above and the
    # generic audit still receive the original rejected request as evidence.
    range_result=audit_range_prefix(scheduling_view)
    scheduling=audit_scheduling_prefix(scheduling_view)
    for item in (r12,r8,range_result,scheduling): require(item.get('passed') is True,'Inherited audit did not pass')
    result.update(passed=True,r12=r12,r8=r8,range=range_result,scheduling=scheduling,
        source_contract_files=len(SOURCE_CONTRACT)+len(NEW_SOURCE_CONTRACT),
        boundary='Frozen prefix matrix and complete physical macro order; source service zero is a proxy omission. '
                 'Only <=4-source optimum is independently recomputed. Generic physics/coverage/terminal audit is still required.')
    return result


def audit_full(record):
    from experiments.audit_q4_cover import audit_record
    generic=audit_record(record)
    require(generic.get('passed') is True,'Generic physical/coverage audit failed: '+str(generic.get('errors')))
    return dict(passed=True,generic=generic,prefix=audit_known_source_prefix(record))
