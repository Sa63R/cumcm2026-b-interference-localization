"""Q3 observation-only search with guaranteed discovery and joint routing.

Route scores are heuristics for total time; only the discovery cover, the
feasible-set clear disk, and the public sixteen-source stopping rule carry
certificates. The optical fallback and protocol finalization are inherited.
"""

from dataclasses import asdict, dataclass
import math

from localization.omni import OmniCandidateRegion
from planning import clearance_grid, improve_open_route, nearest_order
from planning.coverage import omni_coverage_points
from planning.routing import exact_open_route
from simulator_client.state import Position

from .search import _Search, _StopSearch


@dataclass(frozen=True)
class EfficientConfig:
    ring_radius: float = 1150.0
    schedule: str = "route"
    detour_limit_m: float = 400.0
    use_negative: bool = True
    dynamic_coverage: bool = True
    edge_clear: bool = True
    speculative_radius: float = 0.0

    @classmethod
    def parse(cls, values):
        if values is not None and not isinstance(values, dict):
            raise ValueError("efficient_config must be a dictionary")
        try:
            result = cls(**(values or {}))
        except TypeError as error:
            raise ValueError(f"Invalid efficient_config: {error}") from error
        omni_coverage_points(result.ring_radius)
        if not isinstance(result.schedule, str) or result.schedule not in {"immediate", "detour", "route"}:
            raise ValueError("schedule must be immediate, detour, or route")
        for name in ("use_negative", "dynamic_coverage", "edge_clear"):
            if not isinstance(getattr(result, name), bool):
                raise ValueError(f"{name} must be boolean")
        if result.schedule == "route" and not result.dynamic_coverage:
            raise ValueError("joint route scheduling requires dynamic_coverage=True")
        for name, maximum in (("detour_limit_m", 10000), ("speculative_radius", 200)):
            value = getattr(result, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= maximum):
                raise ValueError(f"{name} must be finite and between 0 and {maximum}")
        return result


class EfficientSearch(_Search):
    def __init__(self, client, max_actions, max_active_probes, config):
        self.config = EfficientConfig.parse(config)
        super().__init__(client, 3, "adaptive", max_actions, max_active_probes, "center")
        self.variant = self.report.variant = "efficient"
        self.points = nearest_order(omni_coverage_points(self.config.ring_radius))
        self.report.coverage_points = [[p.x, p.y] for p in self.points]
        self.report.coverage_points_total = len(self.points)
        self.report.strategy_parameters = asdict(self.config)
        self.blocked = set()

    def _perform(self, action, position, channel, phase):
        if action == "measure" and self.config.use_negative and channel not in self.regions:
            self.regions[channel] = OmniCandidateRegion()
        response = super()._perform(action, position, channel, phase)
        if len(self.detected | self.cleared) > 16:
            self.report.error = "Observed more than the public maximum of 16 sources"
            raise _StopSearch("source_count_inconsistent")
        if (action == "measure" and self.config.use_negative
                and response["measure_result"] == "no_signal"):
            self.regions[channel].observe_no_signal(position)
        if action == "clear" and response["clear_result"] == "success" and len(self.cleared) == 16:
            # This uses the public upper bound, never the scenario's hidden N.
            self.report.completion_certified_under_model = True
            raise _StopSearch("source_count_upper_bound_reached")
        return response

    def _clear(self, position, channel, phase):
        if phase == "certified_clear" and self.config.edge_clear:
            disk = self.regions[channel].enclosing_disk()
            center = Position.coerce(disk.center)
            safe_radius = max(0.0, 19.9 - disk.radius)
            current = self.client.state.position
            distance = center.distance_to(current)
            if distance > 0:
                fraction = min(1.0, safe_radius / distance)
                position = Position(center.x + fraction * (current.x - center.x),
                                    center.y + fraction * (current.y - center.y))
        return super()._clear(position, channel, phase)

    def _resolve(self, channel):
        if self.config.speculative_radius == 0:
            return super()._resolve(channel)
        if channel in self.cleared:
            return True
        attempted = set()
        for index in range(self.max_active_probes + 1):
            if channel in self.near_points:
                return self._clear(self.near_points[channel], channel, "near_clear")
            region = self.regions.get(channel)
            if region is None or not region.vertices:
                return False
            circle = region.enclosing_disk()
            if circle.radius <= max(19.9, self.config.speculative_radius):
                center = Position.coerce(circle.center)
                key = (round(center.x, 6), round(center.y, 6))
                if key not in attempted:
                    attempted.add(key)
                    phase = "certified_clear" if circle.radius <= 19.9 else "speculative_clear"
                    if self._clear(center, channel, phase):
                        return True
            if index == self.max_active_probes:
                break
            point = self._next_probe(channel, index)
            if point is None:
                break
            self._perform("measure", point, channel, "active_localization")
        region = self.regions.get(channel)
        if region is None or not region.vertices:
            return False
        for point in clearance_grid(region.vertices, bearing_deg=self.first_bearings[channel],
                                    start=self.client.state.position):
            if self._clear(point, channel, "guaranteed_clearance"):
                return True
        return False

    def _target(self, channel):
        if channel in self.near_points:
            return self.near_points[channel]
        region = self.regions.get(channel)
        return Position.coerce(region.enclosing_disk().center) if region and region.vertices else None

    def _scan(self, point):
        channels = [c for c in range(1, 21) if c not in self.cleared]
        current = self.client.state.current_channel
        if current in channels:
            channels.remove(current)
            channels.insert(0, current)
        for channel in channels:
            self._perform("measure", point, channel, "coverage")
        self.report.coverage_points_visited += 1
        self.blocked.clear()

    def _next_task(self, remaining):
        sources = [("source", channel, target)
                   for channel in sorted(self.detected - self.cleared - self.blocked)
                   if (target := self._target(channel)) is not None]
        current = self.client.state.position
        if not remaining:
            if not sources:
                return None
            route = improve_open_route(nearest_order([s[2] for s in sources], start=current),
                                       start=current)
            return next(s for s in sources if s[2] == route[0])
        cover = (exact_open_route(remaining, start=current)[0]
                 if self.config.dynamic_coverage else remaining[0])
        if not sources:
            return ("cover", None, cover)
        if self.config.schedule == "immediate":
            return min(sources, key=lambda s: (current.distance_to(s[2]), s[1]))
        if self.config.schedule == "detour":
            source = min(sources, key=lambda s: (current.distance_to(s[2]) + s[2].distance_to(cover), s[1]))
            extra = current.distance_to(source[2]) + source[2].distance_to(cover) - current.distance_to(cover)
            return source if extra <= self.config.detour_limit_m else ("cover", None, cover)
        # Receding-horizon route: estimates and current position update after
        # every discovery/localization/clear. This is not a global optimum.
        tasks = [("cover", None, point) for point in remaining] + sources
        route = improve_open_route(nearest_order([task[2] for task in tasks], start=current),
                                   start=current)
        return next(task for task in tasks if task[2] == route[0])

    def _execute_plan(self):
        self._scan(self.points[0])
        remaining = list(self.points[1:])
        while True:
            if not remaining:
                self.report.coverage_complete = True
            task = self._next_task(remaining)
            if task is None:
                break
            kind, channel, point = task
            if kind == "cover":
                self._scan(point)
                remaining.remove(point)
            elif not self._resolve(channel):
                # Retry only after a new shared scan adds information.
                self.blocked.add(channel)
