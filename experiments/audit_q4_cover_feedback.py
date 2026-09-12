"""R39 observed-prefix cover ownership and feedback-value audit.

No hidden-source positions or policy execution are used. Old physical/R12/R8/
range/scheduling checks retain the complete real history. The new route replay
explicitly distinguishes an unexecuted incumbent from an executed cover veto.
"""
import hashlib
import math
from pathlib import Path
from functools import lru_cache
from dataclasses import asdict

from localization import CandidateRegion
from geometry import Circle
from experiments.audit_q4_clear_before_probe import require, point, close, terminal, audit_clear_before_probe_prefix
from experiments.audit_q4_joint_continuation import wire_prefix, audit_joint_continuation_prefix
from experiments.audit_q4_scheduling import audit_scheduling_prefix
from experiments.audit_q4_range import audit_range_prefix
from planning.q4_conditional_belief import build_belief, ModelUnavailable
from planning.chain_route import ChainSource, solve_chain_route
from planning.release_chain_route import solve_release_chain_route

ROOT=Path(__file__).resolve().parents[1]
ENTRY='strategies.q4_cover_feedback:run_q4_cover_feedback'
LABEL='compact_cover_feedback'
CONFIG='centered_one_cover'
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
 'src/planning/q4_conditional_belief.py':'8453de4fc80b94c5428eade327a20c601825808a4facc9515ddecbd6e36ba038',
 'src/planning/matrix_chain_route.py':'14f36e6307a6874239f6d564366519d9e46713f3bf6af8f05cf26334ec43fe40',
 'src/planning/release_chain_route.py':'f0ea8c4438465e9876f7f967fd15de543a751a7e3261579d74160ed27b94593d',
 'src/planning/q4_cover_feedback.py':'2f3a52d59d7752306464fe4497b09fdb9f643e4394d021f960284b6ea5cee0e3',
 'src/strategies/q4_cover_feedback.py':'aa1b7707eb86fdace222ae0b30e416f9871358e5a42a9d4336032c03b1c5a575'}

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



def route_value(covers,positions,start,order,background,matrix,releases):
    """Independent full fixed-chain cost and per-source release feasibility."""
    require(len(releases)==len(positions),'Wrong release vector')
    completed=0
    for kind,index in order:
        if kind=='cover': completed+=1
        else:
            integer(index,0,len(positions)-1)
            require(type(releases[index]) is int and 0<=releases[index]<=len(covers)
                    and completed>=releases[index],'Source precedes its frozen release')
    return matrix_route_cost(covers,positions,start,order,background,matrix)+5.*len(positions)


def small_release_optimum(covers,positions,start,background,matrix,releases):
    """Independent reverse recurrence; no production DP or incumbent helpers."""
    require(len(positions)<=4,'Independent exact validation is small-only')
    @lru_cache(None)
    def f(mask,k,last):
        if mask==(1<<len(positions))-1 and k==len(covers):return 0.
        here=start if last==-1 else positions[last] if last<len(positions) else covers[last-len(positions)]
        values=[]
        for j,p in enumerate(positions):
            if not mask&(1<<j) and k>=releases[j]:
                values.append(math.dist(here,p)/5.+5.+f(mask|(1<<j),k,j))
        if k<len(covers):
            fee=background[k]+sum(matrix[k][j] for j in range(len(positions)) if not mask&(1<<j))
            values.append(math.dist(here,covers[k])/5.+fee+f(mask,k+1,len(positions)+k))
        require(values,'Frozen release model has no completion')
        return min(values)
    return f(0,0,-1)


def validate_result(result,covers,positions,start,background,matrix,releases,budget=200):
    cost=route_value(covers,positions,start,result['order'],background,matrix,releases)
    close(result['cost_s'],cost,'Frozen shared route cost differs')
    lower=finite(result['lower_bound_s']);floor=root_relaxation(covers,positions,start,background)+5.*len(positions)
    require(lower>=floor-2e-6 and 0<=lower<=cost+2e-6,'Invalid route lower bound')
    require(type(result['exact']) is bool and (not result['exact'] or abs(lower-cost)<=2e-6),'Invalid exact-gap claim')
    if len(positions)<=4:
        optimum=small_release_optimum(covers,positions,start,background,matrix,releases)
        require(lower<=optimum+2e-6 and cost>=optimum-2e-6
                and (not result['exact'] or abs(cost-optimum)<=2e-6),'Independent small recurrence disagrees')
    expanded=integer(result['expanded'],0,budget)
    for key in ('generated','dominance_pruned','bound_pruned'):integer(result[key])
    require(finite(result['runtime_s'])>=0,'Invalid solver runtime')
    return expanded


def original_plan(event,prefix,n,covers,channels,total,maximum):
    known,cleared,_,_=prefix.snapshot(n);current=prefix.before[n][0]
    targets=[prefix.target(n,c) for c in channels]
    expected=dict(after_actual_action_count=n,remaining_covers=[list(p) for p in covers],
        source_channels=channels,source_positions=[list(p) for p in targets],discovery_count_cap=len(known)==16)
    for k,v in expected.items():same(event[k],v,'Original R12 plan differs/'+k)
    budget=max(0,min(maximum,60000-total));background=[6.*max(0,20-len(cleared)-len(channels))]*len(covers)
    expanded=validate_result(event['result'],covers,targets,current,background,[[0.]*len(channels) for _ in covers],[0]*len(channels),budget)
    frozen=asdict(solve_chain_route(covers,[ChainSource(p,5.) for p in targets],current,
        max_expansions=budget,background_scan_s=background[0] if covers else 6.*max(0,20-len(cleared)-len(channels)),scan_source_s=0.))
    same(strip_runtime(event['result']),strip_runtime(frozen),'Original incumbent is not the frozen R12 solver result')
    require(finite(event['runtime_s'])>=0,'Invalid original route time')
    first=event['result']['order'][0];kind,index=first
    channel=channels[index] if kind=='source' else None
    require(event['selected_kind']==kind and event['selected_channel']==channel,'Original first task differs')
    return kind,channel,total+expanded



def sampling_quantile(h,c,d):
    return ((h+.5)/4.+((c*(3,5,7,11,13)[d])%23+.5)/23.)%1.


def inverse_mass(weights,u):
    mass=math.fsum(weights)
    if not math.isfinite(mass) or mass<=0:raise ModelUnavailable('empty_sampling_volume')
    threshold=u*mass;cum=0.;positive=[]
    for i,w in enumerate(weights):
        if w<=0:continue
        positive.append(i);cum+=w
        if threshold<cum:return i
    return positive[-1]


def sampled_feedback(belief,q,c,h):
    q=point(q)
    for old in belief.prefix:
        if q==old.position:return dict(channel=c,result=old.result,bearing_deg=old.bearing_deg,origin='locked_actual_feedback',latent=None)
    u=[sampling_quantile(h,c,d) for d in range(5)];j=inverse_mass(belief.spatial_weights,u[0]);node=belief.nodes[j]
    pieces=[];weights=[]
    if node.omni_interval:
        lo,hi=node.omni_interval;pieces.append(('omni',lo,hi,()))
        weights.append(belief.prior_omni*(hi-lo)/500.)
    for seg in node.directional_segments:
        pieces.append(('directional',seg.lower,seg.upper,seg.angles))
        weights.append((1.-belief.prior_omni)*(seg.upper-seg.lower)*math.fsum(b-a for a,b in seg.angles)/(500.*math.tau))
    k=inverse_mass(weights,u[1]);kind,lo,hi,angles=pieces[k];radius=lo+(hi-lo)*u[2];theta=None
    if kind=='directional':
        left=u[3]*math.fsum(b-a for a,b in angles)
        for a,b in angles:
            if left<b-a:theta=a+left;break
            left-=b-a
        if theta is None:theta=math.nextafter(angles[-1][1],angles[-1][0])
    dx=q[0]-node.position[0];dy=q[1]-node.position[1];distance=math.hypot(dx,dy)
    receiving=distance<=radius and (theta is None or math.cos(theta)*dx+math.sin(theta)*dy>=-1e-12*max(1.,distance))
    outcome='no_signal';bearing=None;error=2.*u[4]-1.
    if receiving:
        outcome='near' if distance<=5. else 'direction'
        if outcome=='direction':
            truth_bearing=math.degrees(math.atan2(-dy,-dx))%360.
            bearing=round((truth_bearing+error)%360.,2)%360.
            delta=(bearing-truth_bearing+180.)%360.-180.
            if delta>1.+1e-12:bearing=round(bearing-.01,2)%360.
            elif delta < -1.-1e-12:bearing=round(bearing+.01,2)%360.
    return dict(channel=c,result=outcome,bearing_deg=bearing,origin='conditional_volume',
        latent=dict(node_index=j,position=list(node.position),component_index=k,source_type=kind,
                    radius_m=radius,orientation_rad=theta,bearing_error_deg=error,quantiles=u))


def source_description(item):
    region=item['region'];near=item['near']
    if near is not None:target=list(point(near));radius=None;ready=True
    else:
        if region is None or not region.vertices:raise ModelUnavailable('missing_or_empty_known_region')
        disk=region.copy().enclosing_disk();target=list(disk.center);radius=disk.radius
        if not math.isfinite(radius) or any(not math.isfinite(x) for x in target):raise ModelUnavailable('nonfinite_known_target')
        ready=radius<=19.9
    return dict(channel=item['channel'],target=target,radius_m=radius,ready=ready,
        near_point=list(point(near)) if near is not None else None,
        vertices=[] if region is None else [list(v) for v in region.vertices],
        error_deg=None if region is None else region.error_deg,
        observations=[] if region is None else [asdict(o) for o in region.observations])


def update_hypothesis(item,q,feedback):
    child=dict(item,region=item['region'].copy() if item['region'] is not None else None)
    kind=feedback['result']
    if kind=='direction':
        if child['region'] is None:raise ModelUnavailable('direction_without_known_region')
        child['region'].observe(q,feedback['bearing_deg'])
        if not child['region'].vertices:raise ModelUnavailable('predicted_empty_canonical')
    elif kind=='near':child['near']=point(q)
    elif kind!='no_signal':raise ModelUnavailable('invalid_predicted_feedback')
    source_description(child)
    return child


def strip_runtime(value):
    if isinstance(value,dict):return {k:strip_runtime(v) for k,v in value.items() if k!='runtime_s'}
    if isinstance(value,(tuple,list)):return [strip_runtime(v) for v in value]
    return value


def reference_prediction(current,covers,ready_channels,ready_positions,sources,cleared_count,c,incumbent,allowance):
    """Recompute all model inputs and worlds; pinned solvers plus independent costs.

    Solver runtimes are not deterministic. Their complete route, counters, cost
    and bound are replayed; an additional independent small recurrence validates
    the frozen solver for at most four sources. No live policy is executed.
    """
    log=dict(status='started',fallback_reason=None,recommend_veto=False,information_changed=False,
             input_sources=[],beliefs=[],worlds=[],solve_log=[],expanded=0,max_expansions_per_solve=200)
    def solve(role,inputs,forced=False):
        if log['expanded']+200>allowance:raise ModelUnavailable('prediction_expansion_budget')
        ps=inputs['positions'];qs=inputs['covers'];here=inputs['current'];tasks=[ChainSource(p,5.) for p in ps]
        if forced:
            b=[inputs['background_scan_s']]*len(qs);w=[[0.]*len(ps) for q in qs];rel=[0]*len(ps)
            answer=solve_chain_route(qs,tasks,here,max_expansions=200,background_scan_s=inputs['background_scan_s'],scan_source_s=0.)
        else:
            b=inputs['background_scan_s'];w=inputs['source_scan_s'];rel=inputs['release_indices']
            answer=solve_release_chain_route(qs,tasks,here,max_expansions=200,background_scan_s=b,source_scan_s=w,release_indices=rel)
        result=asdict(answer)
        log['expanded']+=answer.expanded
        log['solve_log'].append(dict(role=role,**inputs,result=result,expanded_charged=answer.expanded))
        return result
    try:
        if not covers or c not in ready_channels or len(ready_channels)!=len(ready_positions):raise ModelUnavailable('invalid_current_ready_task')
        q=point(covers[0]);tail=list(map(point,covers[1:]));ds=[source_description(s) for s in sources]
        log['input_sources']=[dict(**d,prefix=list(s['prefix'])) for s,d in zip(sources,ds)]
        if len({s['channel'] for s in sources})!=len(sources) or len(sources)+cleared_count>16:raise ModelUnavailable('invalid_known_channel_set')
        bg=6.*max(0,20-cleared_count-len(ready_channels))
        forced=solve('forced_q',dict(current=list(q),covers=[list(p) for p in tail],channels=list(ready_channels),
            positions=[list(p) for p in ready_positions],background_scan_s=bg,scan_source_s=0.,service_s=5.),True)
        full=math.dist(current,q)/5.+bg+forced['cost_s']
        log['forced_cover']=dict(cost_s=full,initial_cost_s=full-forced['cost_s'],
            order=[['cover',0]]+[[kind,j+1 if kind=='cover' else j] for kind,j in forced['order']])
        log['D0_s']=max(0.,full-incumbent)
        beliefs={}
        for s,d in zip(sources,ds):
            if d['ready']:continue
            if s['region'].observations and _range_lower(q,s['region'].vertices)>1500.+1e-5:beliefs[s['channel']]='range_skip';continue
            belief=build_belief(s['region'],s['prefix'],node_count=24,prior_omni=.5,max_history=64,work_limit=262144)
            beliefs[s['channel']]=belief;log['beliefs'].append(dict(channel=s['channel'],**asdict(belief)))
        future=[]
        for h in range(4):
            items=[];feedbacks=[];changed=[]
            for s,d in zip(sources,ds):
                if d['ready'] or beliefs.get(s['channel'])=='range_skip':
                    feedback=dict(channel=s['channel'],result=None,bearing_deg=None,origin='ready_skip' if d['ready'] else 'range_skip',latent=None);updated=s
                else:
                    feedback=sampled_feedback(beliefs[s['channel']],q,s['channel'],h)
                    updated=s if feedback['origin']=='locked_actual_feedback' else update_hypothesis(s,q,feedback)
                    after=source_description(updated)
                    if any(after[k]!=d[k] for k in ('vertices','near_point','ready','target')):changed.append(s['channel'])
                items.append(updated);feedbacks.append(feedback)
            future.append(items);log['worlds'].append(dict(index=h,feedbacks=feedbacks,updated_sources=[source_description(s) for s in items],changed_channels=changed))
        log['information_changed']=any(w['changed_channels'] for w in log['worlds'])
        if not log['information_changed']:raise ModelUnavailable('no_information_change')
        def value(role,items,omit):
            kept=[s for s in items if not(omit and s['channel']==c)];dd=[source_description(s) for s in kept]
            background=[6.*max(0,20-cleared_count-int(omit)-len(kept)) for _ in tail]
            matrix=[[0. if d['ready'] or s['region'] is not None and s['region'].observations and _range_lower(p,s['region'].vertices)>1500.+1e-5 else 6. for s,d in zip(kept,dd)] for p in tail]
            inputs=dict(current=list(q),covers=[list(p) for p in tail],channels=[s['channel'] for s in kept],positions=[d['target'] for d in dd],
                ready=[d['ready'] for d in dd],release_indices=[0 if d['ready'] else len(tail) for d in dd],background_scan_s=background,source_scan_s=matrix,service_s=5.)
            return solve(role,inputs)['cost_s']
        vp=value('V0_plus',sources,False);vm=value('V0_minus',sources,True);log['base_marginal_s']=vp-vm;marginals=[]
        for h,items in enumerate(future):
            plus=value(f'world{h}_plus',items,False);minus=value(f'world{h}_minus',items,True)
            marginals.append(plus-minus);log['worlds'][h].update(value_plus_s=plus,value_minus_s=minus)
        log['world_marginals_s']=marginals;log['D_s']=log['D0_s']+math.fsum(marginals)/4.-log['base_marginal_s']
        if not math.isfinite(log['D_s']):raise ModelUnavailable('nonfinite_centered_score')
        log.update(status='scored',recommend_veto=log['D_s'] < -1e-9)
    except ModelUnavailable as error:log.update(status='fallback',fallback_reason=str(error),recommend_veto=False)
    except (ValueError,ArithmeticError) as error:log.update(status='fallback',fallback_reason=type(error).__name__+': '+str(error),recommend_veto=False)
    return log


LIMITS=dict(predictions=32,actual_vetoes=8,vetoes_per_channel=1,worlds=4,
    expansions_per_solve=200,total_prediction_expansions=70400,solves_per_prediction=11,
    belief_nodes=24,belief_prior_omni=.5,belief_history=64,belief_work=262144)


def actual_sources(prefix,n):
    known,cleared,near,regions=prefix.snapshot(n)
    return [dict(channel=c,region=regions.get(c),near=near.get(c),
        prefix=[dict(a) for a in prefix.h[:n] if a['action']=='measure' and a['channel']==c])
        for c in sorted(known-cleared)]


def eligibility(prefix,n,covers,order,c,vetoed,vetoes,calls,expanded):
    if len(order)<2 or order[0][0]!='source' or tuple(order[1])!=('cover',0) or not covers:
        return 'not_source_then_next_cover'
    if c is None or not prefix.ready(n,c):return 'selected_source_not_ready'
    known,cleared,_,_=prefix.snapshot(n)
    if not any(not prefix.ready(n,k) for k in known-cleared):return 'no_known_nonready'
    if c in vetoed:return 'channel_veto_limit'
    if vetoes>=8:return 'session_veto_limit'
    if calls>=32:return 'prediction_call_limit'
    if expanded+200>70400:return 'prediction_expansion_limit'
    return None


def check_prediction(log,prefix,n,covers,channels,c,incumbent,allowance):
    known,cleared,_,_=prefix.snapshot(n)
    expected=reference_prediction(prefix.before[n][0],covers,channels,
        [prefix.target(n,k) for k in channels],actual_sources(prefix,n),len(cleared),c,incumbent,allowance)
    require(isinstance(log,dict) and finite(log.get('runtime_s'))>=0,'Invalid prediction runtime')
    same(strip_runtime(log),strip_runtime(expected),'Observed-prefix feedback prediction differs')
    # Validate logged solver evidence independently, beyond reproducibility of
    # the pinned solver. Never convert an audit failure into model fallback.
    total=0
    for i,step in enumerate(log['solve_log']):
        require(finite(step['result']['runtime_s'])>=0,'Invalid nested solver runtime')
        qs=step['covers'];ps=step['positions']
        if i==0:
            bg=[step['background_scan_s']]*len(qs);matrix=[[0.]*len(ps) for _ in qs];rel=[0]*len(ps)
        else: bg=step['background_scan_s'];matrix=step['source_scan_s'];rel=step['release_indices']
        expanded=validate_result(step['result'],qs,ps,step['current'],bg,matrix,rel)
        require(type(step['expanded_charged']) is int and step['expanded_charged']==expanded,'Prediction expansion charge differs')
        require(total+200<=allowance,'Prediction solve started beyond its conservative expansion allowance')
        total+=expanded
    require(type(log['expanded']) is int and log['expanded']==total,'Prediction cumulative expansion count differs')
    return bool(log['status']=='scored' and log['information_changed'] and log['recommend_veto']),total


def replay_macros(record):
    """Consume all real actions and parent epochs, including zero-action exits."""
    prefix=Prefix(record);h=prefix.h;summary=record['summary'];params=summary['strategy_parameters']
    points=list(map(point,summary['coverage_points']))
    chains=params['chain_route_log'];events=params['cover_feedback_log']
    epochs=params['joint_visibility_resolver_log'];early=params['early_service_log']
    maximum=record['spec'].get('kwargs',{}).get('max_expansions',200)
    require(len(chains)==len(events),'Missing or extra feedback ownership event')
    n=visited=ci=ei=si=total=calls=expanded=vetoes=0
    blocked=set();attempted=set();vetoed=set()

    def resolver(index,start,c):
        require(index<len(epochs),'Missing real resolver epoch')
        e=epochs[index]
        require(e['id']==index and e['channel']==c and e['after_actual_action_count']==start,
                'Resolver identity/prefix differs')
        end=integer(e['end_actual_action_count'],start,len(h))
        require(all(a['channel']==c and a['phase']!='coverage' for a in h[start:end]),
                'Source resolver includes coverage or another channel')
        require(e['status'] in {'cleared','unresolved','interrupted'},'Unfinished real resolver')
        if e['status']!='interrupted':
            require((c in prefix.snapshot(end)[1])==(e['status']=='cleared'),'Resolver clear status contradicts real actions')
        return e,end

    for _ in range(len(h)+len(chains)+len(epochs)+len(early)+5):
        known,cleared,_,_=prefix.snapshot(n)
        if len(cleared)==16:break
        covers=[] if len(known)==16 else points[visited:]
        old=prefix.original_early(n,covers[0],blocked,attempted) if covers else None
        if old is not None:
            require(si<len(early),'Missing original priority early service')
            e=early[si]
            require(e['after_actual_action_count']==n and e['channel']==old[2],'Changed original early priority')
            ep,end=resolver(ei,n,old[2]);require(e['end_actual_action_count']==end,'Early and resolver boundaries differ')
            attempted.add(old[2]);ei+=1;si+=1;n=end
            if ep['status']=='interrupted' and not e['interrupted']:
                require(terminal(summary,n,h),'Unexplained early interruption');break
            continue
        channels=[c for c in sorted(known-cleared-blocked) if prefix.target(n,c) is not None and (not covers or prefix.ready(n,c))]
        event=None;route=None;veto=False;kind='cover';c=None;start=n;before_visited=visited
        if channels:
            require(ci<len(chains),'Missing original ready-source decision')
            route=chains[ci];event=events[ci]
            kind,c,total=original_plan(route,prefix,n,covers,channels,total,maximum)
            why=eligibility(prefix,n,covers,route['result']['order'],c,vetoed,vetoes,calls,expanded)
            original={k:v for k,v in route.items() if k not in {'execution_role','cover_feedback_event_id'}}
            expected=dict(id=ci,route_index=ci,after_actual_action_count=n,channel=c,
                next_cover=list(covers[0]) if covers else None,remaining_covers=[list(p) for p in covers],
                current_position=list(prefix.before[n][0]),current_channel=prefix.before[n][1],
                ready_channels=channels,known_channels=sorted(known),cleared_channels=sorted(cleared),blocked_channels=sorted(blocked),
                incumbent=original,eligibility_reason=why,prediction_calls_before=calls,prediction_expanded_before=expanded,
                actual_vetoes_before=vetoes,vetoed_channels_before=sorted(vetoed),coverage_visited_before=visited)
            for k,v in expected.items():same(event[k],v,'Feedback event differs/'+k)
            if why is None:
                calls+=1
                veto,charge=check_prediction(event['prediction'],prefix,n,covers,channels,c,route['result']['cost_s'],70400-expanded)
                expanded+=charge
            else:require(event['prediction'] is None,'Ineligible original decision ran a prediction')
            require(type(event['veto_selected']) is bool and event['veto_selected']==veto,'Unjustified actual cover veto')
            require(route['execution_role']==('incumbent_not_executed' if veto else 'original_incumbent')
                    and type(route['cover_feedback_event_id']) is int and route['cover_feedback_event_id']==ci,
                    'Original route falsely claims actual ownership')
            actual_kind='cover' if veto else kind
            require(event['executed_kind']==actual_kind,'Actual macro does not match feedback decision')
            ci+=1
            if actual_kind=='source':
                ep,n=resolver(ei,n,c);ei+=1
                if ep['status']=='unresolved':blocked.add(c)
                complete=ep['status']!='interrupted'
            else:complete=None
        elif not covers:break
        else:actual_kind='cover'
        if actual_kind=='cover':
            require(covers,'Coverage chosen with no discovery obligation')
            q=covers[0];expected_channels=scan_channels(prefix,n,q);scan_start=n
            for channel in expected_channels:
                if n==len(h):break
                a=h[n]
                require(a['action']=='measure' and a['phase']=='coverage' and a['channel']==channel and point(a['position'])==q,
                        'Actual cover position/channel/order differs')
                n+=1
            complete=n-scan_start==len(expected_channels)
            if complete:
                require(bool(expected_channels),'Fictional zero-action complete cover')
                visited+=1;blocked.clear()
                if veto:vetoes+=1;vetoed.add(c)
            else:require(terminal(summary,n,h),'Partial actual cover without terminal failure')
        if event is not None:
            expected=dict(end_actual_action_count=n,prediction_calls_after=calls,prediction_expanded_after=expanded,
                actual_vetoes_after=vetoes,vetoed_channels_after=sorted(vetoed),coverage_visited_after=visited,
                executed_cover=bool(actual_kind=='cover' and complete),status='completed' if complete else 'interrupted')
            for k,v in expected.items():same(event[k],v,'Feedback macro exit differs/'+k)
            require(finite(event['runtime_s'])>=0 and finite(event['decision_wall_s'])>=0,'Invalid decision wall time')
        if not complete:
            require(terminal(summary,n,h),'Source macro interrupted without real terminal');break
    else:raise ValueError('Actual macro replay did not make bounded progress')
    require(n==len(h) and ci==len(chains) and ei==len(epochs) and si==len(early),
            'Orphan/missing real action, original plan, feedback event or resolver')
    require(summary['coverage_points_visited']==visited,'Coverage visit total is not actual complete scans')
    return dict(cover_feedback_vetoes=vetoes,prediction_calls=calls,prediction_expanded=expanded,
        original_astar_expanded=total,chain_decisions=ci,completed_scans=visited,resolver_macros=ei)


def verify_source_contract():
    require(len(SOURCE_CONTRACT)==53 and len(NEW_SOURCE_CONTRACT)==5,'Source contract not frozen')
    for name,sha in dict(SOURCE_CONTRACT,**NEW_SOURCE_CONTRACT).items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,'Unreviewed source contract: '+name)
    return 58


def audit_cover_feedback_prefix(record):
    contract=verify_source_contract();spec=record['spec'];kwargs=spec.get('kwargs',{})
    require(spec['entrypoint']==ENTRY and record['row']['strategy']==LABEL,'Unreviewed cover-feedback entry/label')
    require(set(kwargs)<={'config','max_expansions','max_actions','max_active_probes','problem'}
        and kwargs.get('config')==CONFIG and type(kwargs.get('problem',4)) is int and kwargs.get('problem',4)==4
        and type(kwargs.get('max_expansions',200)) is int and kwargs.get('max_expansions',200)==200,'Unreviewed cover-feedback specification')
    params=record['summary']['strategy_parameters']
    require(params['cover_feedback_config']==CONFIG,'Logged cover-feedback configuration differs')
    same(params['cover_feedback_limits'],LIMITS,'Unreviewed cover-feedback limits')
    result=replay_macros(record)
    view=dict(record,spec=dict(spec,entrypoint='strategies.q4_joint_continuation:run_q4_joint_continuation',
        kwargs=dict(kwargs,config='after_active_miss_optical')))
    r12=audit_joint_continuation_prefix(view);r8=audit_clear_before_probe_prefix(view)
    accepted=dict(view,history=[w for w in view['history'] if w['action'] not in {'/measure','/clear'} or w['response'].get('accepted') is True])
    ranged=audit_range_prefix(accepted);scheduling=audit_scheduling_prefix(accepted)
    for child in (r12,r8,ranged,scheduling):require(child.get('passed') is True,'Inherited feedback-prefix audit failed')
    result.update(passed=True,r12=r12,r8=r8,range=ranged,scheduling=scheduling,source_contract_files=contract,
        boundary='Observed-prefix finite-prior four-world proxy, not calibrated expected cost or global Q4 optimum. '
        'Original incumbent ownership is checked against all real macros; hypothetical worlds earn no real scans. '
        'Pinned solver replay is complemented by independent route costs/bounds and <=4-source exact recurrence. '
        'Actual histories are unchanged; only legacy range/scheduling use accepted-wire views, generic keeps all requests.')
    return result


def audit_full(record):
    from experiments.audit_q4_cover import audit_record
    generic=audit_record(record)
    require(generic.get('passed') is True,'Generic physical/coverage audit failed: '+str(generic.get('errors')))
    return dict(passed=True,generic=generic,prefix=audit_cover_feedback_prefix(record))
