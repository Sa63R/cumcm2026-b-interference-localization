"""Discovery, active localization, optical fallback, and certified termination.

The client is an observation-only interface. In particular, this module never
reads source truth, the actual source count, reception radii, or source types.
"""

from dataclasses import asdict, dataclass, field
import math
import time

from localization import CandidateRegion, select_next_point
from planning import clearance_grid, coverage_points
from simulator_client.errors import DeadlineExceeded, SimulatorError
from simulator_client.rules import MIN_SOURCES, MAX_SOURCES
from simulator_client.state import Position


@dataclass
class SearchResult:
    problem: int
    variant: str
    active_policy: str = "center"
    completion_reason: str = "not_started"
    coverage_complete: bool = False
    completion_certified_under_model: bool = False
    coverage_points_total: int = 0
    coverage_points_visited: int = 0
    detected_channels: list = field(default_factory=list)
    cleared_channels: list = field(default_factory=list)
    unresolved_channels: list = field(default_factory=list)
    virtual_time_s: float = 0.0
    program_runtime_s: float = 0.0
    accepted_actions: int = 0
    measurement_count: int = 0
    clear_attempt_count: int = 0
    time_breakdown: dict = field(default_factory=dict)
    coverage_points: list = field(default_factory=list)
    action_history: list = field(default_factory=list)
    source_estimates: dict = field(default_factory=dict)
    strategy_parameters: dict = field(default_factory=dict)
    error: str | None = None
    exit_error: str | None = None

    @property
    def cleared_count(self):
        return len(self.cleared_channels)

    def as_dict(self):
        result = asdict(self)
        result["cleared_count"] = self.cleared_count
        result["detected_count"] = len(self.detected_channels)
        result["all_cleared"] = self.completion_certified_under_model
        result["mean_localization_clearance_time_s"] = (
            self.virtual_time_s / self.cleared_count if self.cleared_count else None)
        result["trajectory"] = [[0.0, 0.0]] + [
            item["position"] for item in self.action_history if "position" in item]
        return result


class _StopSearch(Exception):
    def __init__(self, reason):
        self.reason = reason


class _Search:
    def __init__(self, client, problem, variant, max_actions, max_active_probes,
                 active_policy):
        self.client = client
        self.problem = problem
        self.variant = variant
        self.max_actions = max_actions
        self.max_active_probes = max_active_probes
        self.active_policy = active_policy
        self.points = coverage_points(problem, variant=variant)
        self.report = SearchResult(
            problem, variant, active_policy=active_policy,
            coverage_points_total=len(self.points),
            coverage_points=[[p.x, p.y] for p in self.points])
        self.regions = {}
        self.first_bearings = {}
        self.near_points = {}
        self.observed_positions = {}
        self.detected = set()
        self.cleared = {channel for channel, state in client.state.sources.items()
                        if state.status == "cleared"}
        self.actions = 0

    def _check_budget(self, action, position, channel):
        if self.actions >= self.max_actions - 1:  # reserve one explicit /exit
            raise _StopSearch("action_budget")
        remaining = getattr(self.client, "remaining_real_time_s", None)
        if remaining is not None and remaining <= 2.0:
            raise _StopSearch("real_deadline")
        maximum = self.client.state.max_virtual_duration_s
        maximum = min(360000.0, maximum) if maximum is not None else 360000.0
        cost = self.client.state.position.distance_to(position) / 5.0 + 5.0
        if action == "measure":
            cost += channel != self.client.state.current_channel
        if self.client.state.virtual_time_s + cost > maximum - 1e-6:
            raise _StopSearch("virtual_budget")

    def _perform(self, action, position, channel, phase):
        position = Position.coerce(position)
        self._check_budget(action, position, channel)
        response = getattr(self.client, action)(position, channel)
        if response.get("accepted") is not True:
            raise _StopSearch("request_rejected")
        self.actions += 1
        item = {"action": action, "position": [position.x, position.y],
                "channel": channel, "phase": phase,
                "virtual_time_s": self.client.state.virtual_time_s}
        if action == "measure":
            kind = response["measure_result"]
            item["result"] = kind
            self.report.measurement_count += 1
            self.observed_positions.setdefault(channel, set()).add(
                (round(position.x, 6), round(position.y, 6)))
            if kind in {"direction", "near"}:
                self.detected.add(channel)
            if kind == "direction":
                bearing = response["svd_deg"]
                item["bearing_deg"] = bearing
                self.first_bearings.setdefault(channel, bearing)
                self.regions.setdefault(channel, CandidateRegion()).observe(position, bearing)
            elif kind == "near":
                self.near_points[channel] = position
            # no_signal cannot remove candidates with unknown reception radius
            # and, in Q4, unknown transmitting half-plane.
        else:
            item["result"] = response["clear_result"]
            self.report.clear_attempt_count += 1
            if item["result"] == "success":
                self.cleared.add(channel)
        self.report.action_history.append(item)
        return response

    def _clear(self, position, channel, phase):
        return self._perform("clear", position, channel, phase)["clear_result"] == "success"

    def _next_probe(self, channel, index):
        """Use a feasible-set centre, then cross-bearing offsets if needed.

        No distance estimate from truth is used. Candidates are ranked by their
        distance to the current dog position after preferring the set centre.
        A bounded number of probes accelerates easy cases; completeness relies
        on the optical cover, not on a speculative visibility assumption.
        """
        region = self.regions[channel]
        # T03's safe reception region applies only to omnidirectional sources.
        # Q4 retains the same bounded heuristic for either policy label, and
        # relies on the optical cover for completeness after a first detection.
        if self.problem == 3 and self.active_policy == "minimax":
            choice = select_next_point(region, self.client.state.position,
                                       require_guaranteed=True)
            if choice is not None:
                selected = Position.coerce(choice.position)
                key = (round(selected.x, 6), round(selected.y, 6))
                if key not in self.observed_positions.get(channel, set()):
                    return selected
        circle = region.enclosing_disk()
        center = Position.coerce(circle.center)
        angle = math.radians(self.first_bearings[channel])
        perpendicular = (-math.sin(angle), math.cos(angle))
        radius = min(180.0, max(25.0, circle.radius * 0.5))
        offsets = [(0.0, 0.0),
                   (perpendicular[0] * radius, perpendicular[1] * radius),
                   (-perpendicular[0] * radius, -perpendicular[1] * radius),
                   (math.cos(angle) * radius, math.sin(angle) * radius),
                   (-math.cos(angle) * radius, -math.sin(angle) * radius)]
        candidates = [Position(center.x + dx, center.y + dy) for dx, dy in offsets]
        observed = self.observed_positions.get(channel, set())
        fresh = [p for p in candidates if (round(p.x, 6), round(p.y, 6)) not in observed]
        if not fresh:
            return None
        if candidates[0] in fresh:
            return candidates[0]
        return min(fresh, key=self.client.state.position.distance_to)

    def _resolve(self, channel):
        if channel in self.cleared:
            return True
        attempted_centers = set()
        probes = self.max_active_probes if self.variant != "baseline" else 0
        for index in range(probes + 1):
            if channel in self.near_points:
                if self._clear(self.near_points[channel], channel, "near_clear"):
                    return True
                # A failed near clear contradicts the stationary <=5 m model.
                return False
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
        grid = clearance_grid(region.vertices,
                              bearing_deg=self.first_bearings[channel],
                              start=self.client.state.position)
        for point in grid:
            if self._clear(point, channel, "guaranteed_clearance"):
                return True
        return False

    def _resolve_queue(self):
        unresolved = self.detected - self.cleared
        # Greedy source ordering estimates destinations from observations only.
        while unresolved:
            def estimated_distance(channel):
                if channel in self.near_points:
                    target = self.near_points[channel]
                elif self.regions[channel].vertices:
                    target = Position.coerce(self.regions[channel].enclosing_disk().center)
                else:
                    return float("inf")
                return self.client.state.position.distance_to(target)
            channel = min(unresolved, key=lambda c: (estimated_distance(c), c))
            self._resolve(channel)
            unresolved.remove(channel)

    def _execute_plan(self):
        for point in self.points:
            channels = [c for c in range(1, 21) if c not in self.cleared]
            if self.variant != "baseline" and self.client.state.current_channel in channels:
                channels.remove(self.client.state.current_channel)
                channels.insert(0, self.client.state.current_channel)
            for channel in channels:
                self._perform("measure", point, channel, "coverage")
            self.report.coverage_points_visited += 1
            if self.variant == "adaptive":
                self._resolve_queue()
        self.report.coverage_complete = True
        self._resolve_queue()

    def run(self):
        started = time.perf_counter()
        try:
            if self.client.state.session == "new":
                response = self.client.enter()
                if response.get("accepted") is True:
                    self.actions += 1
                else:
                    raise _StopSearch("request_rejected")
            if self.client.state.session != "active":
                raise _StopSearch("session_unavailable")
            self._execute_plan()
            known_channels = self.detected | self.cleared
            if not MIN_SOURCES <= len(known_channels) <= MAX_SOURCES:
                # This uses only the public 10..16 bound, never a hidden count.
                # A broken all-no-signal endpoint must not certify success via
                # the vacuous set relation empty <= empty.
                self.report.completion_reason = "source_count_inconsistent"
                self.report.error = (
                    f"Complete coverage found {len(known_channels)} distinct sources; "
                    f"the model requires {MIN_SOURCES}..{MAX_SOURCES}")
            elif self.detected <= self.cleared:
                self.report.completion_reason = "coverage_exhausted_and_all_detected_cleared"
                self.report.completion_certified_under_model = True
            else:
                self.report.completion_reason = "unresolved_source"
        except _StopSearch as error:
            self.report.completion_reason = error.reason
        except DeadlineExceeded as error:
            self.report.completion_reason = "real_deadline"
            self.report.error = str(error)
        except SimulatorError as error:
            self.report.completion_reason = "protocol_error"
            self.report.error = f"{type(error).__name__}: {error}"
        finally:
            # Unknown action outcomes prohibit a fresh /exit request as well.
            if (self.client.state.session == "active"
                    and getattr(self.client, "pending_request", None) is None):
                try:
                    response = self.client.exit()
                    if response.get("accepted") is True:
                        self.actions += 1
                    else:
                        self.report.exit_error = "Explicit exit request was rejected"
                except SimulatorError as error:
                    self.report.exit_error = f"{type(error).__name__}: {error}"
            self.report.program_runtime_s = time.perf_counter() - started
            self.report.virtual_time_s = self.client.state.virtual_time_s
            self.report.accepted_actions = self.actions
            self.report.detected_channels = sorted(self.detected)
            self.report.cleared_channels = sorted(self.cleared)
            self.report.unresolved_channels = sorted(self.detected - self.cleared)
            self.report.time_breakdown = asdict(self.client.state.time_breakdown)
            for channel, region in self.regions.items():
                if not region.observations:
                    continue  # Stored negative-only priors are not located sources.
                estimate = {"vertices": [list(p) for p in region.vertices],
                            "area_m2": region.area}
                if region.vertices:
                    disk = region.enclosing_disk()
                    estimate.update(center=list(disk.center), radius_m=disk.radius)
                self.report.source_estimates[channel] = estimate
        return self.report


def run_search(client, *, problem=3, variant="adaptive", max_actions=20000,
               max_active_probes=6, active_policy="center", efficient_config=None,
               rollout_config=None, state_search_config=None):
    """Run one bounded session without reading hidden simulator truth.

    ``baseline`` completes all discovery scans before optical localization.
    ``adaptive`` resolves newly detected channels after each shared scan and
    uses up to ``max_active_probes`` extra bearings before the same fallback.
    ``active_policy='minimax'`` uses T03's guaranteed-reception point selector
    for Q3; ``center`` retains the centre/cross-bearing heuristic. Q4 always
    uses that heuristic because directionality invalidates omni safe regions.
    ``deferred`` retains active localization but waits until all scans finish.
    Q4-only ``triangular`` combines deferred localization and a 990 m triangle
    discovery cover. Both avoid repeatedly interrupting the coverage route.
    Q3-only ``efficient`` jointly schedules a shorter guaranteed cover and
    observed sources, and uses omnidirectional negative observations.
    Q3-only ``rollout`` compares macro-tasks by sampled remaining total time,
    with the unchanged efficient policy as tail and fallback.
    No formal GUI test is launched or selected by this function.
    """
    if problem in {"q3", "q4"}:
        problem = int(problem[1:])
    if problem not in (3, 4):
        raise ValueError("problem must be 3 or 4")
    if variant == "improved":
        variant = "adaptive"
    if variant not in {"baseline", "adaptive", "deferred", "triangular", "efficient", "rollout", "state_search"}:
        raise ValueError("unknown strategy variant")
    if active_policy not in {"center", "minimax"}:
        raise ValueError("active_policy must be center or minimax")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 2:
        raise ValueError("max_actions must be an integer >= 2")
    if (isinstance(max_active_probes, bool) or not isinstance(max_active_probes, int)
            or not 0 <= max_active_probes <= 30):
        raise ValueError("max_active_probes must be an integer between 0 and 30")
    if variant == "state_search":
        if (problem != 3 or active_policy != "center" or efficient_config is not None
                or rollout_config is not None):
            raise ValueError("state_search requires Q3, center, and no other variant config")
        from .state_search import StateSearch
        return StateSearch(client, max_actions, max_active_probes, state_search_config).run()
    if state_search_config is not None:
        raise ValueError("state_search_config requires variant=state_search")
    if variant == "rollout":
        if problem != 3 or active_policy != "center" or efficient_config is not None:
            raise ValueError("rollout requires problem=3, active_policy=center, no efficient_config")
        from .rollout import RolloutSearch
        return RolloutSearch(client, max_actions, max_active_probes, rollout_config).run()
    if rollout_config is not None:
        raise ValueError("rollout_config requires variant=rollout")
    if variant == "efficient":
        if problem != 3 or active_policy != "center":
            raise ValueError("efficient requires problem=3 and active_policy=center")
        from .efficient import EfficientSearch
        return EfficientSearch(client, max_actions, max_active_probes,
                               efficient_config).run()
    if efficient_config is not None:
        raise ValueError("efficient_config requires variant=efficient")
    return _Search(client, problem, variant, max_actions, max_active_probes,
                   active_policy).run()
