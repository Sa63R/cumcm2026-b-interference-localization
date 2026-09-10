"""Finite-hypothesis, finite-action single-source measurement trees.

This explicitly declared approximation has a fixed latent source chosen from
weighted polygon quadrature nodes. A fresh measurement location uses a finite
bounded error law; identical reported outcomes are grouped before the next
decision. The terminal policy is a real optical cover, whose expected prefix
cost is evaluated exactly on the finite support. Neither its value nor the
oracle pruning bound is a lower bound for the original continuous Q3 problem.
"""

from dataclasses import dataclass
import math
import time

from planning import clearance_grid
from simulator_client.state import Position


@dataclass(frozen=True)
class WeightedSource:
    position: Position
    weight: float


def polygon_quadrature(region, limit=6):
    """Area-weighted triangle centroids, merged in cyclic order if necessary.

    Exact for affine functions over each polygon triangle before merging;
    nonlinear observation/value expectations remain quadrature approximations.
    Point/segment degeneracies use one point or equal-weight segment midpoints.
    """
    vertices = [Position(*v) for v in region.vertices]
    if not vertices:
        return ()
    if len(vertices) == 1:
        return (WeightedSource(vertices[0], 1.0),)
    if len(vertices) == 2:
        a, b = vertices
        return tuple(WeightedSource(Position(a.x + (i + .5) / limit * (b.x - a.x),
                                             a.y + (i + .5) / limit * (b.y - a.y)),
                                    1.0 / limit) for i in range(limit))
    center = Position(sum(v.x for v in vertices) / len(vertices),
                      sum(v.y for v in vertices) / len(vertices))
    triangles = []
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        area = abs((a.x - center.x) * (b.y - center.y)
                   - (a.y - center.y) * (b.x - center.x)) / 2
        if area > 1e-12:
            triangles.append((area, (center, a, b)))
    if not triangles:
        return (WeightedSource(Position(*region.enclosing_disk().center), 1.0),)
    # Refine even a triangular posterior: increasing the requested support
    # size must add spatial nodes, rather than merely lifting a vertex cap.
    while len(triangles) < limit:
        largest = max(range(len(triangles)), key=lambda i: triangles[i][0])
        area, (a, b, c) = triangles.pop(largest)
        ab = Position((a.x + b.x) / 2, (a.y + b.y) / 2)
        bc = Position((b.x + c.x) / 2, (b.y + c.y) / 2)
        ca = Position((c.x + a.x) / 2, (c.y + a.y) / 2)
        triangles[largest:largest] = [(area / 4, vertices) for vertices in
                                      ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
    nodes = [(area, Position(sum(p.x for p in vertices) / 3,
                             sum(p.y for p in vertices) / 3)) for area, vertices in triangles]
    groups = [[] for _ in range(min(limit, len(nodes)))]
    for i, item in enumerate(nodes):
        groups[i * len(groups) // len(triangles)].append(item)
    total = sum(area for area, _ in nodes)
    result = []
    for group in groups:
        weight = sum(area for area, _ in group)
        result.append(WeightedSource(Position(sum(area * p.x for area, p in group) / weight,
                                              sum(area * p.y for area, p in group) / weight),
                                     weight / total))
    return tuple(result)


def _candidates(region, current, bearing, observed, limit):
    circle = region.enclosing_disk()
    center = Position(*circle.center)
    theta = math.radians(bearing)
    transverse = (-math.sin(theta), math.cos(theta))
    points = [center]
    # Nested candidate sets: 3 adds full-step +/-50; 5 adds halfway +/-50;
    # 7 adds full-step +/-150; 9 adds halfway +/-150. The discrete refinement
    # preserves all prior root actions for otherwise unchanged models.
    for fraction, offset in ((1, -50), (1, 50), (.5, -50), (.5, 50),
                             (1, -150), (1, 150), (.5, -150), (.5, 150)):
        points.append(Position(current.x + fraction * (center.x - current.x) + offset * transverse[0],
                               current.y + fraction * (center.y - current.y) + offset * transverse[1]))
    result = []
    for point in points[:limit]:
        if (round(point.x, 6), round(point.y, 6)) in observed:
            continue
        if any(point.distance_to(Position(*v)) > 1000 for v in region.vertices):
            continue
        if point not in result:
            result.append(point)
    return result


def _bearing(source, observer, error):
    angle = math.degrees(math.atan2(source.y - observer.y, source.x - observer.x)) % 360
    value = round((angle + error) % 360, 2) % 360
    delta = (value - angle + 180) % 360 - 180
    if delta > 1 + 1e-12:
        value = round(value - .01, 2) % 360
    elif delta < -1 - 1e-12:
        value = round(value + .01, 2) % 360
    return value


class ProbeTree:
    """A bounded Bellman search with feasible-policy bounds on a finite model."""

    def __init__(self, *, depth=1, supports=6, candidates=9, inner_candidates=3,
                 noise_nodes=1, max_expansions=500, first_bearing=0.0,
                 root_switch_s=0.0, use_bounds=True, terminal_mode="expected_support"):
        self.depth, self.supports = depth, supports
        self.candidates, self.inner_candidates = candidates, inner_candidates
        self.noise = ((0.0, 1.0),) if noise_nodes == 1 else (
            (-math.sqrt(3 / 5), 5 / 18), (0.0, 4 / 9), (math.sqrt(3 / 5), 5 / 18))
        self.max_expansions, self.first_bearing = max_expansions, first_bearing
        self.root_switch_s, self.use_bounds = root_switch_s, use_bounds
        self.terminal_mode = terminal_mode
        self.expanded = self.pruned = self.cache_hits = self.budget_fallbacks = 0
        self.tail_cache = {}
        self.branch_count = self.singleton_branches = 0

    def _tail(self, region, current, support):
        key = (tuple(region.vertices), current, support)
        if key in self.tail_cache:
            self.cache_hits += 1
            return self.tail_cache[key]
        circle = region.enclosing_disk()
        center = Position(*circle.center)
        if circle.radius <= 19.9:
            value = max(0.0, current.distance_to(center) - (19.9 - circle.radius)) / 5 + 5
        else:
            route = clearance_grid(region.vertices, bearing_deg=self.first_bearing, start=current)
            costs, elapsed, previous = [None] * len(support), 0.0, current
            for point in route:
                elapsed += previous.distance_to(point) / 5 + 5
                previous = point
                for i, source in enumerate(support):
                    if costs[i] is None and point.distance_to(source.position) <= 20:
                        costs[i] = elapsed
                if self.terminal_mode == "expected_support" and all(cost is not None for cost in costs):
                    break
            if any(cost is None for cost in costs):
                raise ValueError("Optical cover did not cover the finite posterior support")
            value = (elapsed if self.terminal_mode == "full_cover_bound" else
                     sum(source.weight * cost for source, cost in zip(support, costs)))
        self.tail_cache[key] = value
        return value

    @staticmethod
    def _oracle(current, support):
        # Every successful clear lies within 20 m of the fixed finite source;
        # delete information costs and allow source knowledge for a valid LB.
        return sum(source.weight * (max(0.0, current.distance_to(source.position) - 20) / 5 + 5)
                   for source in support)

    def _branches(self, region, current, support):
        branches = {}
        for source in support:
            if current.distance_to(source.position) <= 5:
                entries = (("near", 1.0),)
            else:
                entries = ((_bearing(source.position, current, error), chance)
                           for error, chance in self.noise)
            for observation, chance in entries:
                key = ("near", None) if observation == "near" else ("direction", observation)
                branch = branches.setdefault(key, {})
                branch[source.position] = branch.get(source.position, 0.0) + source.weight * chance
        result = []
        for (kind, bearing), weights in branches.items():
            self.branch_count += 1
            self.singleton_branches += len(weights) == 1
            probability = sum(weights.values())
            posterior = tuple(WeightedSource(point, weight / probability)
                              for point, weight in weights.items())
            updated = None if kind == "near" else region.copy().observe(current, bearing)
            if updated is not None and not updated.vertices:
                raise ValueError("Finite outcome contradicts its geometric observation region")
            result.append((probability, updated, posterior))
        return result

    def _action(self, region, current, support, point, depth, observed, cutoff=math.inf):
        movement = current.distance_to(point) / 5 + 5.0
        if self.use_bounds and movement + self._oracle(point, support) >= cutoff - 1e-9:
            self.pruned += 1
            return math.inf
        branches = self._branches(region, point, support)
        # Order branches by probability so feasible-bound pruning triggers early.
        branches.sort(key=lambda branch: branch[0], reverse=True)
        remaining_lb = sum(chance * (5 if updated is None else self._oracle(point, posterior))
                           for chance, updated, posterior in branches)
        value = movement
        new_observed = observed | {(round(point.x, 6), round(point.y, 6))}
        for chance, updated, posterior in branches:
            remaining_lb -= chance * (5 if updated is None else self._oracle(point, posterior))
            if updated is None:
                branch_value = 5.0
            elif depth <= 1:
                branch_value = self._tail(updated, point, posterior)
            else:
                branch_value = self._value(updated, point, posterior, depth - 1, new_observed)
            value += chance * branch_value
            if self.use_bounds and value + remaining_lb >= cutoff - 1e-9:
                self.pruned += 1
                return math.inf
        return value

    def _value(self, region, current, support, depth, observed):
        # A stop action executes the deterministic optical tail. Therefore
        # increasing the allowed depth cannot worsen the exact finite optimum.
        incumbent = self._tail(region, current, support)
        if region.enclosing_disk().radius <= 19.9:
            return incumbent
        for point in _candidates(region, current, self.first_bearing, observed, self.inner_candidates):
            if self.expanded >= self.max_expansions:
                self.budget_fallbacks += 1
                break
            self.expanded += 1
            incumbent = min(incumbent, self._action(region, current, support, point,
                                                      depth, observed, incumbent))
        return incumbent

    def choose(self, region, current, observed):
        began = time.perf_counter()
        support = polygon_quadrature(region, self.supports)
        candidates = _candidates(region, current, self.first_bearing, observed, self.candidates)
        if not candidates:
            return None, {"reason": "no_fresh_guaranteed_candidate"}
        # Root must measure; the live controller owns clear decisions. Internal
        # nodes may stop and use a real optical tail, supplying finite-model UBs.
        best, selected = math.inf, None
        for point in sorted(candidates, key=lambda p: current.distance_to(p) / 5 + self._oracle(p, support)):
            value = self._action(region, current, support, point, self.depth, observed, best)
            if value < best:
                best, selected = value, point
        return selected, {"model": "finite_static_source_discrete_noise_optical_tail",
                          "terminal_mode": self.terminal_mode,
                          "depth": self.depth, "support_nodes": len(support),
                          "noise_nodes": len(self.noise), "candidate_nodes": len(candidates),
                          "finite_model_cost_s": best + self.root_switch_s,
                          "oracle_lower_bound_s": self._oracle(current, support),
                          "expanded": self.expanded, "bound_pruned": self.pruned,
                          "cache_hits": self.cache_hits, "budget_fallbacks": self.budget_fallbacks,
                          "posterior_branches": self.branch_count,
                          "singleton_posterior_branches": self.singleton_branches,
                          "exact_for_declared_tree": self.budget_fallbacks == 0,
                          "runtime_s": time.perf_counter() - began,
                          "support_positions_weights": [[s.position.x, s.position.y, s.weight] for s in support]}
