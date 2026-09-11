"""Q4 optical fallback with certified variable-width rectangular cells.

Only the fallback changes: actual active probes, the 22-point discovery cover,
ready-source scheduling, and near/MEC clearance certificates are inherited.
For a horizontal slab of the observed convex polygon, cover its bounding box
by m equal rectangles. If the slab height is h, their widths are at most
2*sqrt(19.8**2-(h/2)**2); every rectangle is contained in its centre's radius
19.8 disk. Closed consecutive slabs cover the entire polygon, including edges.

The three finite slab counts and the old grid compete on *complete traversal*
cost, not a posterior expected first-hit cost. A lower complete-traversal proxy
does not guarantee earlier actual clearance. No reception assumption or hidden
source coordinate is used, and negative radio feedback does not prune geometry.
"""
import math
import time

from planning.coverage import clearance_grid, nearest_order, _clip_horizontal
from simulator_client.state import Position
from .q4_cover_search import Q4CoverSearch

CELL_RADIUS_M = 19.8


def _traversal_cost(points, start):
    previous = Position.coerce(start)
    length = 0.0
    for p in points:
        length += previous.distance_to(p)
        previous = p
    # Every failed optical attempt costs 3s; the final successful one costs 5s.
    return length / 5.0 + 3.0 * len(points) + (2.0 if points else 0.0)


def _slab_cover(rotated, rows, cosine, sine, start):
    ymin, ymax = min(p[1] for p in rotated), max(p[1] for p in rotated)
    height = (ymax-ymin) / rows
    centers, certificate = [], []
    for row in range(rows):
        low = ymin + row*height
        high = ymax if row == rows-1 else ymin + (row+1)*height
        polygon = _clip_horizontal(rotated, low, True)
        polygon = _clip_horizontal(polygon, high, False)
        if not polygon:
            continue
        x0, x1 = min(p[0] for p in polygon), max(p[0] for p in polygon)
        y0, y1 = min(p[1] for p in polygon), max(p[1] for p in polygon)
        h = y1-y0
        if h >= 2*CELL_RADIUS_M:
            return None, None
        max_width = 2*math.sqrt(CELL_RADIUS_M**2-(h/2)**2)
        columns = max(1, math.ceil((x1-x0)/max_width))
        width = (x1-x0)/columns
        half_diagonal = math.hypot(width/2, h/2)
        if half_diagonal > CELL_RADIUS_M+1e-9:
            return None, None
        for column in range(columns):
            x, y = x0+(column+0.5)*width, (y0+y1)/2
            centers.append(Position(x*cosine-y*sine, x*sine+y*cosine))
        certificate.append({"slab": [low, high], "bounding_box": [x0, x1, y0, y1],
            "polygon": [list(p) for p in polygon], "columns": columns,
            "cell_width_m": width, "cell_half_diagonal_m": half_diagonal})
    return nearest_order(centers, start), certificate


def optical_cover_route(vertices, *, bearing_deg, start):
    """Return a finite full cover and a replayable rectangle certificate.

    At most three counts are tried, starting at ceil(cross-bearing width/28).
    The unmodified legacy route always remains an incumbent. Numerical margins
    are engineering safeguards; the certificate is not interval arithmetic.
    """
    began = time.perf_counter()
    original = tuple(Position.coerce(p) for p in vertices)
    if not math.isfinite(bearing_deg):
        raise ValueError("bearing must be finite")
    start = Position.coerce(start)
    baseline = clearance_grid(original, bearing_deg=bearing_deg, start=start)
    old_cost = _traversal_cost(baseline, start)
    log = {"score_kind": "complete_optical_traversal_not_expected_first_hit",
           "old_count": len(baseline), "old_proxy_s": old_cost,
           "selected": "legacy", "candidates": [], "certificate": None,
           "bearing_deg": bearing_deg, "start": [start.x, start.y],
           "observed_region_vertices": [[p.x, p.y] for p in original]}
    if not original:
        log.update(selected_count=0, selected_proxy_s=0.0, runtime_s=time.perf_counter()-began)
        return baseline, log
    angle = math.radians(bearing_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotated = [(p.x*cosine+p.y*sine, -p.x*sine+p.y*cosine) for p in original]
    cross_width = max(p[1] for p in rotated)-min(p[1] for p in rotated)
    minimum_rows = max(1, math.ceil(cross_width / 28.0))
    best, best_cost = baseline, old_cost
    for rows in range(minimum_rows, minimum_rows+3):
        route, cert = _slab_cover(rotated, rows, cosine, sine, start)
        if route is None:
            continue
        cost = _traversal_cost(route, start)
        log["candidates"].append({"rows": rows, "count": len(route), "proxy_s": cost})
        if cost < best_cost-1e-8:
            best, best_cost = route, cost
            log.update(selected="rectangular_slabs", certificate={
                "radius_m": CELL_RADIUS_M, "rows": rows,
                "rotated_y_range": [min(p[1] for p in rotated), max(p[1] for p in rotated)],
                "slabs": cert})
    log.update(selected_count=len(best), selected_proxy_s=best_cost,
               runtime_s=time.perf_counter()-began)
    return best, log


class Q4OpticalCoverSearch(Q4CoverSearch):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions, optical_mode):
        super().__init__(client, max_actions, max_active_probes, profile="compact_22",
                         schedule="joint", max_expansions=max_expansions)
        self.optical_mode = optical_mode
        self.optical_log = []
        self.report.strategy_parameters.update(q4_optical_mode=optical_mode,
            optical_cover_log=self.optical_log,
            optical_scope="Fallback only; active measurements and macro scheduling inherited unchanged")

    def _resolve(self, channel):
        if self.optical_mode == "off":
            return super()._resolve(channel)
        # The active prefix below is identical to _Search._resolve. It is kept
        # local so the frozen parent modules and other Q4 strategies stay intact.
        if channel in self.cleared:
            return True
        attempted_centers = set()
        probes = self.max_active_probes if self.variant != "baseline" else 0
        for index in range(probes+1):
            if channel in self.near_points:
                return self._clear(self.near_points[channel], channel, "near_clear")
            region = self.regions.get(channel)
            if region is None or not region.vertices:
                return False
            circle = region.enclosing_disk()
            if circle.radius <= 19.9:
                center = Position.coerce(circle.center)
                key = (round(center.x, 6), round(center.y, 6))
                if key not in attempted_centers:
                    attempted_centers.add(key)
                    if self._clear(center, channel, "certified_clear"):
                        return True
            if index == probes:
                break
            point = self._next_probe(channel, index)
            if point is None:
                break
            self._perform("measure", point, channel, "active_localization")
        region = self.regions.get(channel)
        if region is None or not region.vertices:
            return False
        route, log = optical_cover_route(region.vertices,
            bearing_deg=self.first_bearings[channel], start=self.client.state.position)
        self.optical_log.append({"channel": channel,
            "after_actual_action_count": len(self.report.action_history), **log})
        for point in route:
            if self._clear(point, channel, "guaranteed_clearance"):
                return True
        return False


def run_q4_optical_cover(client, *, problem=4, max_actions=20000, max_active_probes=6,
                         max_expansions=200, optical_mode="rectangular"):
    if problem != 4 or optical_mode not in {"off", "rectangular"}:
        raise ValueError("Q4 only; optical_mode must be off or rectangular")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be integer in [0,30]")
    if type(max_expansions) is not int or not 0 <= max_expansions <= 10000:
        raise ValueError("max_expansions must be integer in [0,10000]")
    return Q4OpticalCoverSearch(client, max_actions, max_active_probes,
        max_expansions=max_expansions, optical_mode=optical_mode).run()
