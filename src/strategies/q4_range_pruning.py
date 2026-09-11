"""Skip only provably unreachable known-channel coverage measurements in Q4.

At every real-positive prefix the true source lies in the outer convex region C.
If dist(p,C)>1500, a measurement at p cannot receive that source for ANY radius
or transmitting orientation allowed by the problem. This implication is one
way; ordinary Q4 no_signal responses still do not shrink C.

Skipped measurements produce neither simulated responses nor observation or
negative-coverage credits. Unknown channels, cover stations, source scheduling,
active probes, optical fallback and actual-clear termination stay inherited.
The current center-plus-180m probe family cannot later choose one of these
faraway points: subsequent regions only shrink. No inferred freshness state is
therefore needed for this standalone configuration.
"""

import math

from simulator_client.state import Position
from .q4_cover_search import Q4CoverSearch


RANGE_MARGIN_M = 1e-5


def region_distance_lower(point, vertices):
    """Conservative floating-point distance to a convex outer polygon.

    Polygon orientation and point/segment degeneracies are handled explicitly.
    The guard is deliberately much larger than arithmetic ulps at arena scale;
    this numerical computation is not claimed to be interval arithmetic.
    """
    p = Position.coerce(point)
    q = (p.x, p.y)
    vertices = [tuple(v) for v in vertices]
    if not vertices:
        return 0.
    if any(len(v) != 2 or not all(math.isfinite(x) for x in v) for v in vertices):
        raise ValueError("Invalid convex-region vertices")
    edges = list(zip(vertices, vertices[1:]+vertices[:1]))
    crosses = [(b[0]-a[0])*(q[1]-a[1])-(b[1]-a[1])*(q[0]-a[0]) for a, b in edges]
    area2 = sum(a[0]*b[1]-a[1]*b[0] for a, b in edges)
    if len(vertices) >= 3 and abs(area2) > 1e-12:
        if all(x >= -1e-9 for x in crosses) or all(x <= 1e-9 for x in crosses):
            return 0.
    values = []
    for a, b in edges:
        dx, dy = b[0]-a[0], b[1]-a[1]
        size = dx*dx+dy*dy
        t = min(1., max(0., ((q[0]-a[0])*dx+(q[1]-a[1])*dy)/size)) if size else 0.
        values.append(math.hypot(q[0]-a[0]-t*dx, q[1]-a[1]-t*dy))
    scale = max(1., abs(q[0]), abs(q[1]), *(abs(x) for v in vertices for x in v))
    return max(0., min(values) - (1e-7+128*math.ulp(scale)))


class Q4RangePruningSearch(Q4CoverSearch):
    def __init__(self, client, max_actions, max_active_probes, *, profile="compact_22",
                 schedule="joint", max_expansions=200, enabled=True):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        super().__init__(client, max_actions, max_active_probes, profile=profile,
                         schedule=schedule, max_expansions=max_expansions)
        self.range_pruning_enabled = enabled
        self.range_skips = []
        self.report.strategy_parameters.update(
            range_pruning_enabled=enabled, range_skipped_scans=self.range_skips,
            range_skip_scope="Known live positive-region distance >1500m+1e-5; no inferred observations or absence credits",
        )

    def _scan(self, point):
        if not self.range_pruning_enabled:
            return super()._scan(point)
        point = Position.coerce(point)
        channels = [c for c in range(1, 21) if c not in self.cleared]
        # Preserve the original state_pruned ready-source rule and its ledger.
        skipped = [c for c in channels if self._ready(c)]
        for c in skipped:
            self.skipped_scans.append({"channel": c, "position": [point.x, point.y],
                "after_actual_action_count": len(self.report.action_history),
                "reason": "near" if c in self.near_points else "enclosing_disk",
                "radius_m": None if c in self.near_points else self.regions[c].enclosing_disk().radius})
        channels = [c for c in channels if c not in skipped]
        current = self.client.state.current_channel
        if current in channels:
            channels.remove(current)
            channels.insert(0, current)
        for channel in channels:
            region = self.regions.get(channel)
            if channel in self.detected-self.cleared and region is not None and region.observations:
                lower = region_distance_lower(point, region.vertices)
                if lower > 1500.+RANGE_MARGIN_M:
                    self.range_skips.append({"channel": channel, "position": [point.x, point.y],
                        "after_actual_action_count": len(self.report.action_history),
                        "reason": "positive_region_beyond_max_reception_radius",
                        "distance_lower_bound_m": lower, "margin_m": RANGE_MARGIN_M,
                        "positive_observation_count": len(region.observations)})
                    continue
            self._perform("measure", point, channel, "coverage")
        self.report.coverage_points_visited += 1
        self.blocked.clear()


def run_q4_range_pruning(client, *, problem=4, max_actions=20000, max_active_probes=6,
                        profile="compact_22", schedule="joint", max_expansions=200, enabled=True):
    if problem != 4:
        raise ValueError("Range pruning is Q4 only")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be integer in [0,30]")
    if type(max_expansions) is not int or not 0 <= max_expansions <= 10000:
        raise ValueError("max_expansions must be integer in [0,10000]")
    return Q4RangePruningSearch(client, max_actions, max_active_probes, profile=profile,
                               schedule=schedule, max_expansions=max_expansions, enabled=enabled).run()
