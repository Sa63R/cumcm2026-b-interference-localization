"""Bounded, observation-branched Q4 probe ranking; never a truth oracle.

The finite posterior is an explicitly assumed quadrature model, separate from
the conservative geometric region. Every observation branch has one common
continuation action. Optical leaf routes depend on that branch's outer region,
not on a particle's position, and charge the complete route through success.
"""
from dataclasses import dataclass
import math
import time

from geometry import clip_polygon, bearing_halfplanes, disk_halfplanes, distance
from planning.coverage import clearance_grid


@dataclass(frozen=True)
class Hypothesis:
    position: tuple
    radius: float
    orientation: float | None
    weight: float


@dataclass(frozen=True)
class ObservationBranch:
    outcome: tuple  # ('near',), ('no_signal',), or ('bearing', integer bin)
    mass: float
    particles: tuple
    spatial_ess: float


class ModelUnavailable(Exception):
    pass


def key(position):
    return tuple(round(float(v), 6) for v in position)


def unique_prefix(history, channel):
    """A coordinate/channel has fixed error: repeated readings count once."""
    result, seen = [], set()
    for item in history:
        if item.get('action') != 'measure' or item.get('channel') != channel:
            continue
        p = tuple(map(float, item['position']))
        # Exact coordinates identify the physical fixed error. Candidate
        # exclusion is additionally conservative at the baseline's 1e-6 scale.
        if p in seen:
            continue
        seen.add(p)
        result.append({'position': p, 'result': item['result'],
                       'bearing_deg': item.get('bearing_deg')})
    return tuple(result)


def _spatial_nodes(region, count=24):
    vertices = region.vertices
    if not vertices:
        return ()
    if any(not all(math.isfinite(v) for v in p) for p in vertices):
        return ()
    center = tuple(sum(v[i] for v in vertices) / len(vertices) for i in (0, 1))
    triangles, total = [], 0.
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        area = abs((a[0]-center[0])*(b[1]-center[1]) -
                   (a[1]-center[1])*(b[0]-center[0])) / 2
        total += area
        triangles.append((total, a, b))
    if total < 1e-10:
        a, b = max(((a, b) for a in vertices for b in vertices),
                   key=lambda pair: distance(*pair))
        values = [tuple(a[j] + (i+.5)/count*(b[j]-a[j]) for j in (0, 1))
                  for i in range(count)]
    else:
        values = []
        for i in range(count):
            target = (i+.5)/count*total
            _, a, b = next(t for t in triangles if t[0] >= target)
            radial = math.sqrt(((i+1)*.6180339887498949) % 1)
            along = ((i+1)*.7548776662466927) % 1
            values.append(tuple((1-radial)*center[j] +
                                radial*((1-along)*a[j]+along*b[j]) for j in (0, 1)))
    # No deliberately placed center atom: a center probe must not receive an
    # artificial 1/3 'near' probability from a center/axis three-node rule.
    return tuple(dict.fromkeys(p for p in values if region.contains(p)))


def _kind(particle, observer):
    dx = observer[0] - particle.position[0]
    dy = observer[1] - particle.position[1]
    r = math.hypot(dx, dy)
    visible = r <= particle.radius + 1e-9
    if particle.orientation is not None:
        angle = math.radians(particle.orientation)
        visible &= dx*math.cos(angle) + dy*math.sin(angle) >= -1e-9
    if not visible:
        return 'no_signal', None
    if r <= 5.:
        return 'near', None
    return 'bearing', math.degrees(math.atan2(-dy, -dx)) % 360


def make_belief(region, prefix, check=lambda: None):
    """Area quadrature x {1000,1250,1500} x assumed type/orientation prior.

An observation mismatch has likelihood floor .001 to soften discretization,
not to alter the real feasible set. These are not calibrated probabilities.
"""
    nodes = _spatial_nodes(region)
    if not nodes:
        return ()
    weighted = []
    for position in nodes:
        check()
        for radius in (1000., 1250., 1500.):
            for orientation in (None,) + tuple(float(a) for a in range(0, 360, 45)):
                prior = .5 if orientation is None else .5/8
                log_weight = math.log(prior)
                hypothesis = Hypothesis(position, radius, orientation, 0.)
                for item in prefix:
                    kind, bearing = _kind(hypothesis, item['position'])
                    expected = 'bearing' if item['result'] == 'direction' else item['result']
                    likelihood = 1. if kind == expected else .001
                    if expected == 'bearing' and kind == 'bearing':
                        delta = abs((bearing-float(item['bearing_deg'])+180) % 360-180)
                        likelihood *= 1. if delta <= region.error_deg else .001
                    log_weight += math.log(likelihood)
                weighted.append((hypothesis, log_weight))
    maximum = max(v for _, v in weighted)
    normalizer = sum(math.exp(v-maximum) for _, v in weighted)
    return tuple(Hypothesis(h.position, h.radius, h.orientation,
                            math.exp(v-maximum)/normalizer) for h, v in weighted)


def bearing_bin_likelihoods(bearing, *, bin_deg=10., error_deg=1.005):
    """Exact bin integrals for an assumed uniform, bounded angular error."""
    low, high = bearing-error_deg, bearing+error_deg
    count = round(360/bin_deg)
    result = {}
    for unwrapped in range(math.floor(low/bin_deg), math.floor(high/bin_deg)+1):
        overlap = max(0., min(high, (unwrapped+1)*bin_deg)-max(low, unwrapped*bin_deg))
        if overlap:
            result[unwrapped % count] = result.get(unwrapped % count, 0.) + overlap/(2*error_deg)
    return result


def observation_branches(particles, observer, *, bin_deg=10., error_deg=1.005):
    groups = {}
    for h in particles:
        if h.weight <= 0:
            continue
        kind, bearing = _kind(h, observer)
        outcomes = ({('bearing', index): probability for index, probability in
                     bearing_bin_likelihoods(bearing, bin_deg=bin_deg,
                                             error_deg=error_deg).items()}
                    if kind == 'bearing' else {(kind,): 1.})
        for outcome, likelihood in outcomes.items():
            groups.setdefault(outcome, []).append((h, h.weight*likelihood))
    result = []
    for outcome, values in sorted(groups.items()):
        mass = sum(w for _, w in values)
        if mass <= 0:
            continue
        normalized = tuple(Hypothesis(h.position, h.radius, h.orientation, w/mass)
                           for h, w in values)
        spatial = {}
        for h in normalized:
            spatial[h.position] = spatial.get(h.position, 0.) + h.weight
        result.append(ObservationBranch(outcome, mass, normalized,
                                        1/sum(w*w for w in spatial.values())))
    return tuple(result)


class ObservationTree:
    """Finite expected-cost tree with conservative, nonzero optical leaves."""
    def __init__(self, *, depth=1, max_cpu_s=None):
        if type(depth) is not int or depth not in (1, 2):
            raise ValueError('depth must be 1 or 2')
        self.depth = depth
        self.max_cpu_s = (.20 if depth == 1 else .60) if max_cpu_s is None else max_cpu_s
        if not math.isfinite(self.max_cpu_s) or self.max_cpu_s <= 0:
            raise ValueError('positive finite CPU budget required')
        self.bin_deg = 10.
        self.max_grid = 256
        self.max_nodes = 400

    def _check(self):
        if time.perf_counter() > self.deadline:
            raise ModelUnavailable('cpu_budget')
        if self.nodes > self.max_nodes:
            raise ModelUnavailable('node_budget')

    def _candidates(self, region, observed, first_bearing, baseline=None):
        circle = region.enclosing_disk()
        angle = math.radians(first_bearing)
        r = min(180., max(25., circle.radius*.5))
        u = (-math.sin(angle)*r, math.cos(angle)*r)
        center = circle.center
        values = ([] if baseline is None else [baseline]) + [center,
            (center[0]+u[0], center[1]+u[1]), (center[0]-u[0], center[1]-u[1])]
        result = []
        for p in values:
            p = tuple(p)
            if key(p) not in observed and all(key(p) != key(q) for q in result):
                result.append(p)
        return tuple(result[:3])

    def _updated_region(self, region, observer, outcome):
        updated = region.copy()
        if outcome[0] == 'bearing':
            halfwidth = self.bin_deg/2 + region.error_deg
            angle = (outcome[1]+.5)*self.bin_deg
            constraints = bearing_halfplanes(observer, angle, halfwidth)
            # A circumscribed polygon preserves all possible reception points.
            constraints += disk_halfplanes(observer, 1500., 32, outer=True)
        elif outcome[0] == 'near':
            constraints = disk_halfplanes(observer, 5., 32, outer=True)
        else:
            return updated  # Silence changes model weights only.
        vertices = list(updated.vertices)
        for hp in constraints:
            vertices = clip_polygon(vertices, hp)
            if not vertices:
                raise ModelUnavailable('empty_geometric_branch')
        updated.vertices, updated._circle = vertices, None
        return updated

    def _leaf(self, region, observer, first_bearing):
        """Charge one common complete optical route, independent of particles."""
        self._check()
        circle = region.enclosing_disk()
        if circle.radius <= 19.9:
            return distance(observer, circle.center)/5 + 5.
        angle = math.radians(first_bearing)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotated = [(p[0]*cosine+p[1]*sine, -p[0]*sine+p[1]*cosine) for p in region.vertices]
        columns = math.floor(max(p[0] for p in rotated)/28)-math.floor(min(p[0] for p in rotated)/28)+1
        rows = math.floor(max(p[1] for p in rotated)/28)-math.floor(min(p[1] for p in rotated)/28)+1
        if rows*columns > self.max_grid:
            raise ModelUnavailable('optical_grid_budget')
        route = clearance_grid(region.vertices, bearing_deg=first_bearing, start=observer)
        self._check()
        if not route:
            raise ModelUnavailable('empty_optical_route')
        cost, previous = 2., observer  # One successful removal, optical 3/cell.
        for p in route:
            position = (p.x, p.y)
            cost += distance(previous, position)/5 + 3.
            previous = position
        return cost

    def _action_value(self, region, particles, observer, action, observed,
                      first_bearing, depth):
        self._check()
        self.nodes += 1
        branches = observation_branches(particles, action, bin_deg=self.bin_deg,
                                        error_deg=region.error_deg)
        if not branches or abs(sum(b.mass for b in branches)-1.) > 1e-7:
            raise ModelUnavailable('particle_exhaustion')
        total = distance(observer, action)/5 + 5.
        branch_log = []
        for branch in branches:
            self._check()
            child = self._updated_region(region, action, branch.outcome)
            # Near is a real common clearance certificate, not inferred from a
            # lone particle. Angular singleton branches get no adaptive action.
            if branch.outcome[0] == 'near':
                value, selected, reason = 5., None, 'near_optical'
            elif child.enclosing_disk().radius <= 19.9:
                value, selected, reason = self._leaf(child, action, first_bearing), None, 'geometric_optical'
            elif depth <= 1 or branch.spatial_ess < 2.:
                value, selected = self._leaf(child, action, first_bearing), None
                reason = 'depth_leaf' if depth <= 1 else 'low_spatial_ess_leaf'
            else:
                next_observed = observed | {key(action)}
                choices = self._candidates(child, next_observed, first_bearing)
                if not choices:
                    value, selected, reason = self._leaf(child, action, first_bearing), None, 'no_fresh_leaf'
                else:
                    # One shared action minimizes the entire branch expectation.
                    scored = [(self._action_value(child, branch.particles, action, q,
                               next_observed, first_bearing, depth-1)[0], q) for q in choices]
                    value, selected = min(scored, key=lambda item: item[0])
                    reason = 'shared_second_probe'
            total += branch.mass*value
            branch_log.append({'outcome': list(branch.outcome), 'mass': branch.mass,
                'particles': len(branch.particles), 'spatial_ess': branch.spatial_ess,
                'continuation': selected, 'continuation_cost_s': value, 'reason': reason})
        return total, branch_log

    def choose(self, region, current, baseline, prefix, *, first_bearing, remaining_probes=6,
               initial_switch_s=0.):
        started = time.perf_counter()
        self.deadline, self.nodes = started+self.max_cpu_s, 0
        log = {'version': 'finite-observation-tree-v1', 'depth': self.depth,
               'effective_depth': min(self.depth, remaining_probes),
               'baseline': baseline, 'selected': baseline, 'fallback': True,
               'reason': None, 'candidate_scores': [], 'prefix_unique': len(prefix)}
        try:
            if baseline is None or not region.vertices or remaining_probes < 1:
                raise ModelUnavailable('no_active_candidate')
            # Work only on a copy, including the lazy enclosing-disk cache.
            region = region.copy()
            particles = make_belief(region, prefix, self._check)
            if not particles:
                raise ModelUnavailable('particle_exhaustion')
            log['particles'] = len(particles)
            observed = {key(item['position']) for item in prefix}
            candidates = self._candidates(region, observed, first_bearing, baseline)
            if not candidates:
                raise ModelUnavailable('no_fresh_candidate')
            # Full evaluation only. Timeout never favors an early-scored option.
            scored = []
            for q in candidates:
                value, branches = self._action_value(region, particles, current, q,
                    observed, first_bearing, min(self.depth, remaining_probes))
                value += initial_switch_s
                scored.append((value, q))
                log['candidate_scores'].append({'position': q, 'cost_s': value,
                                                'branches': branches})
            self._check()
            value, selected = min(scored, key=lambda item: item[0])
            log.update(selected=selected, fallback=False, reason='complete_tree',
                       predicted_local_cost_s=value)
        except ModelUnavailable as error:
            log['reason'] = str(error)
        finally:
            elapsed = time.perf_counter()-started
            log.update(cpu_s=elapsed, decision_wall_s=elapsed, action_nodes=self.nodes)
        return log['selected'], log
