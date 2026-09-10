"""Q3 closed-loop control with an anytime compressed task-order optimizer.

Only legal observation state is used. The finite optimizer freezes region
centres and existing tasks; unknown sources and future geometry are not
predicted. Its bounds are explicitly NOT bounds on the original Q3 problem.
"""

from dataclasses import asdict, dataclass
import math

from planning.state_route import RouteTask, solve_state_route
from planning.disk_cover import disk_cover_radius
from simulator_client.state import Position

from .efficient import EfficientSearch


@dataclass(frozen=True)
class StateSearchConfig:
    ring_radius: float = 1150.0
    max_expansions: int = 3000
    max_total_expansions: int = 60000
    scan_source_s: float = 6.0
    skip_certified_scans: bool = True
    stop_discovery_at_16: bool = True
    opportunistic_measurements: bool = False
    opportunistic_min_radius: float = 60.0
    replace_coverage: bool = False
    coverage_replacement_gain_s: float = 0.0
    active_probe_search: bool = False
    probe_uncertainty_weight: float = 1.0
    probe_model: str = "radius_proxy"
    probe_depth: int = 1
    probe_supports: int = 6
    probe_candidates: int = 9
    probe_inner_candidates: int = 3
    probe_noise_nodes: int = 1
    probe_max_expansions: int = 500
    probe_terminal_mode: str = "expected_support"

    @classmethod
    def parse(cls, values):
        if values is not None and not isinstance(values, dict):
            raise ValueError("state_search_config must be a dictionary")
        try:
            config = cls(**(values or {}))
        except TypeError as error:
            raise ValueError(f"Invalid state_search_config: {error}") from error
        for name in ("max_expansions", "max_total_expansions"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2000000:
                raise ValueError(f"{name} must be an integer in [0, 2000000]")
        if (isinstance(config.scan_source_s, bool)
                or not isinstance(config.scan_source_s, (int, float))
                or not math.isfinite(config.scan_source_s) or not 0 <= config.scan_source_s <= 60):
            raise ValueError("scan_source_s must be finite in [0, 60]")
        for name in ("skip_certified_scans", "stop_discovery_at_16", "opportunistic_measurements", "replace_coverage", "active_probe_search"):
            if not isinstance(getattr(config, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("opportunistic_min_radius", "coverage_replacement_gain_s", "probe_uncertainty_weight"):
            value = getattr(config, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1000):
                raise ValueError(f"{name} must be finite in [0,1000]")
        if config.probe_model not in {"radius_proxy", "optical_tree"}:
            raise ValueError("probe_model must be radius_proxy or optical_tree")
        if config.probe_terminal_mode not in {"expected_support", "full_cover_bound"}:
            raise ValueError("probe_terminal_mode must be expected_support or full_cover_bound")
        for name, values in (("probe_depth", (1, 2)), ("probe_noise_nodes", (1, 3)),
                             ("probe_candidates", (3, 5, 7, 9)), ("probe_inner_candidates", (3, 5, 7, 9))):
            value = getattr(config, name)
            if isinstance(value, bool) or value not in values:
                raise ValueError(f"{name} must be in {values}")
        for name, low, high in (("probe_supports", 1, 24), ("probe_max_expansions", 0, 100000)):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low},{high}]")
        return config


class StateSearch(EfficientSearch):
    def __init__(self, client, max_actions, max_active_probes, config):
        self.state_config = StateSearchConfig.parse(config)
        super().__init__(client, max_actions, max_active_probes,
                         {"ring_radius": self.state_config.ring_radius})
        self.variant = self.report.variant = "state_search"
        self.search_log = []
        self.total_expansions = 0
        self.skipped_measurements = 0
        self.discovery_stations = []
        self.remaining_covers = []
        self.coverage_replacements = []
        self.opportunistic_count = 0
        self.probe_log = []
        self.report.strategy_parameters = {
            **asdict(self.state_config), "model": "frozen_observed_tasks_only",
            "planning_log": self.search_log,
            "search_bound_scope": "finite frozen task model, not original Q3",
            "coverage_replacements": self.coverage_replacements,
            "probe_search_log": self.probe_log,
        }

    def _scan(self, point):
        channels = [c for c in range(1, 21) if c not in self.cleared]
        current = self.client.state.current_channel
        if current in channels:
            channels.remove(current)
            channels.insert(0, current)
        for channel in channels:
            region = self.regions.get(channel)
            certified = (channel in self.near_points or (
                channel in self.detected and region and region.vertices
                and region.enclosing_disk().radius <= 19.9))
            if self.state_config.skip_certified_scans and certified:
                # An already detected source needs no further discovery, and
                # the existing region already certifies a legal clear point.
                self.skipped_measurements += 1
                continue
            self._perform("measure", point, channel, "coverage")
        self.report.coverage_points_visited += 1
        self.discovery_stations.append(point)
        self.report.strategy_parameters["skipped_certified_measurements"] = self.skipped_measurements
        self.blocked.clear()

    def _resolve(self, channel):
        result = super()._resolve(channel)
        if result:
            if self.state_config.replace_coverage:
                self._try_replace_coverage()
            if self.state_config.opportunistic_measurements:
                self._share_observation()
        return result

    def _share_observation(self):
        """Get inexpensive cross bearings from an already reached location.

        Ranking uses a nominal centre and is a heuristic. Actual responses
        alone update the certified region. No predicted bearing is installed
        into live state, and no predicted radius authorizes a clear.
        """
        current = self.client.state.position
        for channel in sorted(self.detected - self.cleared):
            region = self.regions.get(channel)
            if channel in self.near_points or not region or not region.vertices:
                continue
            circle = region.enclosing_disk()
            if circle.radius < self.state_config.opportunistic_min_radius:
                continue
            key = (round(current.x, 6), round(current.y, 6))
            if key in self.observed_positions.get(channel, set()):
                continue
            if any(current.distance_to(Position(*vertex)) > 1000 for vertex in region.vertices):
                continue  # This first implementation requires certain reception.
            center = Position(*circle.center)
            bearing = math.degrees(math.atan2(center.y - current.y, center.x - current.x)) % 360
            hypothetical = region.copy().observe(current, bearing)
            if not hypothetical.vertices:
                continue
            predicted = hypothetical.enclosing_disk().radius
            if predicted > 0.5 * circle.radius or circle.radius - predicted < 30:
                continue
            self._perform("measure", current, channel, "shared_cross_bearing")
            self.opportunistic_count += 1
        self.report.strategy_parameters["shared_cross_bearings"] = self.opportunistic_count

    def _next_probe(self, channel, index):
        if not self.state_config.active_probe_search:
            return super()._next_probe(channel, index)
        region = self.regions[channel]
        if not region.vertices:
            return None
        if self.state_config.probe_model == "optical_tree":
            from planning.probe_tree import ProbeTree
            config = self.state_config
            tree = ProbeTree(depth=config.probe_depth, supports=config.probe_supports,
                             candidates=config.probe_candidates, inner_candidates=config.probe_inner_candidates,
                             noise_nodes=config.probe_noise_nodes, max_expansions=config.probe_max_expansions,
                             first_bearing=self.first_bearings[channel],
                             root_switch_s=float(self.client.state.current_channel != channel),
                             terminal_mode=config.probe_terminal_mode)
            point, log = tree.choose(region, self.client.state.position,
                                     self.observed_positions.get(channel, set()))
            self.probe_log.append({"channel": channel, "index": index, **log})
            return point if point is not None else super()._next_probe(channel, index)
        circle = region.enclosing_disk()
        center = Position(*circle.center)
        current = self.client.state.position
        theta = math.radians(self.first_bearings[channel])
        perpendicular = (-math.sin(theta), math.cos(theta))
        candidates = [center]
        for fraction in (0.5, 1.0):
            anchor = Position(current.x + fraction * (center.x - current.x),
                              current.y + fraction * (center.y - current.y))
            for offset in (-150.0, -50.0, 50.0, 150.0):
                candidates.append(Position(anchor.x + offset * perpendicular[0],
                                           anchor.y + offset * perpendicular[1]))
        vertices = region.vertices
        selected = [vertices[i * len(vertices) // min(4, len(vertices))]
                    for i in range(min(4, len(vertices)))]
        # A finite, explicitly declared empirical prior within the outer
        # feasible region. This is not an official posterior distribution.
        hypotheses = [center] + [Position(0.75 * x + 0.25 * center.x,
                                          0.75 * y + 0.25 * center.y)
                                 for x, y in selected]
        ranked = []
        observed = self.observed_positions.get(channel, set())
        for point in candidates:
            if (round(point.x, 6), round(point.y, 6)) in observed:
                continue
            if any(point.distance_to(Position(*vertex)) > 1000 for vertex in vertices):
                continue
            total = 0.0
            for source in hypotheses:
                d = point.distance_to(source)
                if d <= 5:
                    total += d / 5
                    continue
                bearing = math.degrees(math.atan2(source.y - point.y, source.x - point.x)) % 360
                hypothetical = region.copy().observe(point, bearing)
                if not hypothetical.vertices:
                    total = math.inf
                    break
                posterior_radius = hypothetical.enclosing_disk().radius
                total += (d / 5 + (6.0 if posterior_radius > 19.9 else 0.0)
                          + self.state_config.probe_uncertainty_weight *
                          max(0.0, posterior_radius - 19.9) / 5)
            score = current.distance_to(point) / 5 + total / len(hypotheses)
            ranked.append((score, point.x, point.y, point))
        if not ranked:
            return super()._next_probe(channel, index)
        best = min(ranked)
        self.probe_log.append({"channel": channel, "index": index,
                               "candidates": len(ranked), "hypotheses": len(hypotheses),
                               "score_s": best[0], "position": [best[3].x, best[3].y],
                               "score_kind": "finite_nominal_observation_surrogate"})
        return best[3]

    def _try_replace_coverage(self):
        """Substitute the current location for redundant future cover sites.

        Every still-undetected channel is measured here before any station is
        removed. Thus, for a channel remaining undetected at termination, all
        common discovery stations are genuine no-signal measurements on that
        same channel, never mere trajectory locations.
        """
        if (self.state_config.stop_discovery_at_16
                and len(self.detected | self.cleared) == 16):
            return  # Those cover tasks will not be executed, so cannot save travel.
        remaining = self.remaining_covers
        if not remaining:
            return
        current = self.client.state.position
        unknown = sorted(set(range(1, 21)) - self.detected - self.cleared)
        if not unknown:
            return
        proposed = list(remaining)
        removed = []
        certified_radius = None
        # Greedily delete farthest redundant site; each deletion is independently
        # certified for the full disk, including possible interior holes.
        for station in sorted(remaining, key=current.distance_to, reverse=True):
            alternate = [p for p in proposed if p != station]
            radius = disk_cover_radius(self.discovery_stations + [current] + alternate)
            if radius <= 1000 - 1e-5:
                proposed, certified_radius = alternate, radius
                removed.append(station)
        if not removed:
            return
        # Conservative decision threshold in the *route surrogate*. Geometry
        # certifies completeness; this heuristic alone predicts time savings.
        from planning import nearest_order, improve_open_route
        def distance(points):
            previous, total = current, 0.0
            for point in improve_open_route(nearest_order(points, start=current), start=current):
                total += previous.distance_to(point)
                previous = point
            return total
        predicted_gain = (distance(remaining) - distance(proposed)) / 5 - 6 * len(unknown)
        if predicted_gain < self.state_config.coverage_replacement_gain_s:
            return
        for channel in unknown:
            self._perform("measure", current, channel, "replacement_discovery")
        self.discovery_stations.append(current)
        remaining[:] = proposed
        self.blocked.clear()
        self.coverage_replacements.append({
            "position": [current.x, current.y], "channels_measured": unknown,
            "removed_stations": [[p.x, p.y] for p in removed],
            "certified_cover_radius_m": certified_radius,
            "predicted_gain_s": predicted_gain,
        })

    def _next_task(self, remaining):
        self.remaining_covers = remaining
        sources = [("source", channel, target)
                   for channel in sorted(self.detected - self.cleared - self.blocked)
                   if (target := self._target(channel)) is not None]
        covers = [("cover", None, point) for point in remaining]
        if self.state_config.stop_discovery_at_16 and len(self.detected | self.cleared) == 16:
            covers = []  # Public count cap proves no undiscovered source remains.
        tasks = covers + sources
        if not tasks:
            return None
        budget = max(0, min(self.state_config.max_expansions,
                           self.state_config.max_total_expansions - self.total_expansions))
        # Fixed background-channel scanning is unavoidable in the frozen
        # model, but order-independent; retain it for interpretable costs.
        background = max(0, 20 - len(self.cleared) - len(sources))
        finite_tasks = [RouteTask(task[2], task[0] == "source",
                                  5.0 if task[0] == "source" else 6.0 * background)
                        for task in tasks]
        result = solve_state_route(finite_tasks, self.client.state.position,
                                   scan_source_s=self.state_config.scan_source_s,
                                   max_expansions=budget)
        self.total_expansions += result.expanded
        chosen = tasks[result.order[0]]
        self.search_log.append({
            "task_count": len(tasks), "source_count": len(sources),
            "cover_count": len(covers), "selected_kind": chosen[0],
            "selected_channel": chosen[1], "selected_point": [chosen[2].x, chosen[2].y],
            **{key: value for key, value in asdict(result).items() if key != "order"},
            "model_gap_s": result.cost_s - result.lower_bound_s,
        })
        return chosen


def run_state_search(client, *, problem=3, max_actions=10000, config=None,
                     max_active_probes=6):
    """Public research-harness entry point; no simulator truth is accepted."""
    from .search import run_search
    return run_search(client, problem=problem, variant="state_search",
                      max_actions=max_actions, max_active_probes=max_active_probes,
                      state_search_config=config)
