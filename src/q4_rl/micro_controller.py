"""Q4 public-geometry micro actions; v1 and frozen R8 remain unchanged.

Each learned decision submits exactly one real measure or clear request.
Optical grid misses are billed observations, never convex-region exclusions.
Only actual clear success and the fixed directional cover certify completion.
"""
from dataclasses import dataclass, field
import math
import time

from planning.coverage import clearance_grid
from simulator_client.rules import CHANNELS, MAX_SOURCES, NEAR_RADIUS_M
from simulator_client.state import Position
from strategies.search import _Search, _StopSearch
from strategies.q4_range_pruning import region_distance_lower
from .controller import (Q4RLSearch, GLOBAL_FEATURE_NAMES as V1_GLOBAL_NAMES,
                         CANDIDATE_FEATURE_NAMES as V1_CANDIDATE_NAMES)


FEATURE_SCHEMA_VERSION = "q4-micro-g1-v1"
SAFE_CLEAR_RADIUS_M = 19.9
GEOMETRY_GUARD_M = 1e-5
GRID_SPACING_M = 28.0
GLOBAL_FEATURE_NAMES = V1_GLOBAL_NAMES + (
    "remaining_decisions_fraction", "remaining_requests_fraction",
    "remaining_virtual_fraction",
)
CANDIDATE_FEATURE_NAMES = (
    V1_CANDIDATE_NAMES[:1] + ("is_clear",) + V1_CANDIDATE_NAMES[2:5]
    + ("immediate_cost_upper",) + V1_CANDIDATE_NAMES[6:] + (
        "is_certified_clear", "is_grid_clear",
        "center_relative_current_x", "center_relative_current_y",
        "center_relative_candidate_x", "center_relative_candidate_y",
    ) + tuple(f"outer_support_{i * 45}_deg" for i in range(8)) + (
        "candidate_region_distance_lower", "candidate_region_distance_upper",
        "region_vertex_count", "has_positive_bearing", "region_degenerate",
    ) + tuple(f"{which}_positive_{name}" for which in ("first", "latest")
              for name in ("relative_x", "relative_y", "distance",
                           "view_sin", "view_cos", "error")) + (
        "grid_started", "grid_remaining_fraction", "grid_attempted_fraction",
    )
)
GLOBAL_DIM, CANDIDATE_DIM = len(GLOBAL_FEATURE_NAMES), len(CANDIDATE_FEATURE_NAMES)
SUPPORT_DIRECTIONS = tuple((math.cos(i * math.pi / 4), math.sin(i * math.pi / 4))
                           for i in range(8))


@dataclass(frozen=True)
class MicroCandidate:
    kind: str
    point: Position
    channel: int
    at_cover: bool = False
    role: str = ""


@dataclass
class GridState:
    # Unstarted proposals can be recomputed after a positive-region change.
    # Once one point is executed, this complete cover remains immutable.
    vertices: tuple
    points: tuple
    attempted: set = field(default_factory=set)
    started: bool = False


class Q4MicroSearch(Q4RLSearch):
    def __init__(self, client, policy=None, **kwargs):
        super().__init__(client, policy=policy, **kwargs)
        self._in_fallback = False
        self.grid_states = {}
        self.local_measurements = {c: 0 for c in CHANNELS}
        self.report.learning.update(
            algorithm=FEATURE_SCHEMA_VERSION,
            feature_schema=dict(version=FEATURE_SCHEMA_VERSION,
                global_features=list(GLOBAL_FEATURE_NAMES),
                candidate_features=list(CANDIDATE_FEATURE_NAMES)),
            learned_scope=["initial_position", "scan_position", "scan_channel",
                "localization_position", "continue_or_pause_localization",
                "certified_clear_position_and_order", "optical_grid_order"],
            fixed_scope=["compact_22_discovery_certificate", "geometric_clear_mask",
                "finite_candidate_generator", "actual_clear_confirmation",
                "complete_optical_grid", "budget_fallback"],
            micro_steps=[], grid_ledgers={}, certified_clear_checks=[],
            maximum_candidates=0,
        )
        self.report.strategy_parameters.update(
            q4_rl_feature_schema=FEATURE_SCHEMA_VERSION,
            service_scope="One accepted physical request per micro decision",
            micro_candidate_scope="22 cover stations + current + 5 original probe offsets + 2 safe clear points + 4 grid representatives per live channel",
            micro_candidate_upper_bound=636,
            safe_clear_radius_m=SAFE_CLEAR_RADIUS_M,
            geometry_guard_m=GEOMETRY_GUARD_M,
            clear_certificate_arithmetic="Conservative outer polygon, all vertices checked, engineering floating margin; not interval arithmetic",
            grid_spacing_m=GRID_SPACING_M,
            grid_state_rule="Freeze full queue when first grid action is accepted; no point is lost by candidate subsampling or new positive observations",
            fallback_scope="Frozen R8 resolver for unstarted grids; resume full saved grid for already started grids",
            repetition_rule="Fresh real measure coordinates only, inherited v1 search-space restriction",
            heuristic_probe_limit=self.max_active_probes,
        )

    def _maximum_virtual(self):
        maximum = self.client.state.max_virtual_duration_s
        return 360000. if maximum is None else min(360000., maximum)

    def _upper_cost(self, action, point, channel):
        movement = math.ceil(self.client.state.position.distance_to(point) / 5. * 1e6) / 1e6
        return movement + 5. + (action == "measure" and channel != self.client.state.current_channel)

    def _budget_allows(self, action, point, channel):
        if self.actions >= self.max_actions - 1:
            return False
        if self.action_deadline_epoch is not None and time.time() >= self.action_deadline_epoch:
            return False
        remaining = getattr(self.client, "remaining_real_time_s", None)
        if remaining is not None and remaining <= 2.:
            return False
        return self.client.state.virtual_time_s + self._upper_cost(action, point, channel) <= self._maximum_virtual() - 1e-6

    def _check_budget(self, action, position, channel):
        super()._check_budget(action, position, channel)
        if self.client.state.virtual_time_s + self._upper_cost(action, Position.coerce(position), channel) > self._maximum_virtual() - 1e-6:
            raise _StopSearch("virtual_budget")

    def _perform(self, action, position, channel, phase):
        if self._in_fallback:
            # Keep R8's actual clear-before-probe behavior only in the named tail.
            return super()._perform(action, position, channel, phase)
        try:
            # Deliberately bypass R8 interception: one request, one observation.
            return _Search._perform(self, action, position, channel, phase)
        finally:
            self._consume_actual_history()

    def _can_measure(self, point, channel):
        return (super()._can_measure(point, channel)
                and self._budget_allows("measure", point, channel))

    def _ready(self, channel):
        # Keep measure pruning consistent with this controller's stricter mask.
        # A v1 radius exactly at 19.9m is not enough after the extra guard.
        return bool(self._safe_points(channel))

    def _clear_certificate(self, point, channel):
        if channel not in self.detected or channel in self.cleared:
            return None
        if channel in self.near_points:
            upper = point.distance_to(self.near_points[channel]) + NEAR_RADIUS_M
            proof = "actual_near_observation_plus_5m_disk"
        else:
            region = self.regions.get(channel)
            if region is None or not region.vertices or not region.observations:
                return None
            upper = max(math.hypot(point.x-x, point.y-y) for x, y in region.vertices)
            proof = "all_vertices_of_actual_positive_outer_polygon"
        if not math.isfinite(upper) or upper > SAFE_CLEAR_RADIUS_M - GEOMETRY_GUARD_M:
            return None
        return dict(kind=proof, max_distance_upper_m=upper,
                    operational_radius_m=SAFE_CLEAR_RADIUS_M,
                    guard_m=GEOMETRY_GUARD_M)

    def _safe_points(self, channel):
        if channel in self.near_points:
            proposed = [self.near_points[channel]]
        else:
            region = self.regions.get(channel)
            if region is None or not region.vertices or not region.observations:
                return ()
            disk = region.enclosing_disk()
            center = Position.coerce(disk.center)
            radius = max(math.hypot(center.x-x, center.y-y) for x, y in region.vertices)
            if radius > SAFE_CLEAR_RADIUS_M - GEOMETRY_GUARD_M:
                return ()
            safe_radius = max(0., SAFE_CLEAR_RADIUS_M - 2*GEOMETRY_GUARD_M - radius)
            current = self.client.state.position
            distance = current.distance_to(center)
            fraction = min(1., safe_radius/distance) if distance else 0.
            edge = Position(center.x+fraction*(current.x-center.x),
                            center.y+fraction*(current.y-center.y))
            proposed = [center, edge]
        return tuple(dict.fromkeys(p for p in proposed if self._clear_certificate(p, channel)))

    def _probe_points(self, channel):
        region = self.regions.get(channel)
        if region is None or not region.vertices or not region.observations:
            return ()
        circle = region.enclosing_disk()
        center = Position.coerce(circle.center)
        angle = math.radians(self.first_bearings[channel])
        forward, transverse = (math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))
        radius = min(180., max(25., circle.radius*.5))
        offsets = [(0., 0.)] + [(sign*radius*v[0], sign*radius*v[1])
                                for v in (transverse, forward) for sign in (1., -1.)]
        return tuple(Position(center.x+dx, center.y+dy) for dx, dy in offsets)

    def _grid(self, channel):
        region = self.regions.get(channel)
        if region is None or not region.vertices or not region.observations:
            return None
        state = self.grid_states.get(channel)
        vertices = tuple(tuple(v) for v in region.vertices)
        if state is None or (not state.started and state.vertices != vertices):
            points = tuple(clearance_grid(vertices, bearing_deg=self.first_bearings[channel],
                                          spacing=GRID_SPACING_M, start=self.client.state.position))
            state = self.grid_states[channel] = GridState(vertices, points)
        return state

    def _grid_points(self, channel):
        state = self._grid(channel)
        if state is None:
            return ()
        remaining = [p for p in state.points if p not in state.attempted]
        if not remaining:
            return ()
        return tuple(dict.fromkeys((remaining[0], min(remaining, key=self.client.state.position.distance_to),
                                    remaining[len(remaining)//3], remaining[2*len(remaining)//3])))

    def _candidates(self):
        discovery_done = self._refresh_certificate()
        result, seen = [], set()

        def add(kind, point, channel, role):
            key = kind, point, channel
            if key in seen or channel in self.cleared:
                return
            if kind == "measure":
                if not self._can_measure(point, channel):
                    return
            elif not self._budget_allows("clear", point, channel):
                return
            if kind == "certified_clear" and self._clear_certificate(point, channel) is None:
                return
            seen.add(key)
            result.append(MicroCandidate(kind, point, channel, point in self.cover_ledger, role))

        for point in self.points:
            for channel in CHANNELS:
                if not discovery_done or channel in self.detected:
                    add("measure", point, channel, "cover")
        if self.allow_current_scan:
            for channel in CHANNELS:
                if not discovery_done or channel in self.detected:
                    add("measure", self.client.state.position, channel, "current")
        for channel in sorted(self.detected-self.cleared):
            for point in self._safe_points(channel):
                add("certified_clear", point, channel, "safe_clear")
            if not self._ready(channel):
                for point in self._probe_points(channel):
                    add("measure", point, channel, "localize")
                for point in self._grid_points(channel):
                    add("grid_clear", point, channel, "grid")
        self.report.learning["maximum_candidates"] = max(self.report.learning["maximum_candidates"], len(result))
        return result

    @staticmethod
    def _anchor_features(observation, point):
        if observation is None:
            return [0.] * 6
        ax, ay = observation.position
        dx, dy = point.x-ax, point.y-ay
        distance = math.hypot(dx, dy)
        theta = math.radians(observation.bearing_deg)
        sine = (dy*math.cos(theta)-dx*math.sin(theta))/distance if distance else 0.
        cosine = (dx*math.cos(theta)+dy*math.sin(theta))/distance if distance else 0.
        return [(ax-point.x)/3600., (ay-point.y)/3600., distance/3600.,
                sine, cosine, observation.error_deg/90.]

    def _features(self, candidates):
        # Reuse v1 public summaries; replace its service bit with a clear bit.
        global_features, rows = super()._features(candidates)
        maximum = self._maximum_virtual()
        global_features += [max(0, self.max_decisions-self.report.learning["decisions"])/max(1,self.max_decisions),
                            max(0, self.max_actions-self.actions-1)/max(1,self.max_actions),
                            max(0., maximum-self.client.state.virtual_time_s)/360000.]
        current = self.client.state.position
        geometry = {}
        for channel in {c.channel for c in candidates}:
            region = self.regions.get(channel)
            positive = bool(region and region.vertices and region.observations)
            if positive:
                center = Position.coerce(region.enclosing_disk().center)
                vertices = region.vertices
                support = [max((x-center.x)*u+(y-center.y)*v for x,y in vertices)/1800.
                           for u,v in SUPPORT_DIRECTIONS]
                first, latest = region.observations[0], region.observations[-1]
                degenerate = len(vertices)<3 or region.area <= 1e-8
            else:
                center = self.near_points.get(channel, current)
                vertices, support, first, latest, degenerate = (), [0.]*8, None, None, False
            geometry[channel] = center, vertices, support, first, latest, positive, degenerate
        for candidate, row in zip(candidates, rows):
            point, channel = candidate.point, candidate.channel
            center, vertices, support, first, latest, positive, degenerate = geometry[channel]
            lower = region_distance_lower(point, vertices) if vertices else 0.
            upper = max((math.hypot(point.x-x,point.y-y) for x,y in vertices), default=0.)
            state = self.grid_states.get(channel)
            row[1] = float(candidate.kind != "measure")
            # This is the conservative one-request upper cost; grid miss may cost two seconds less.
            row[5] = self._upper_cost("measure" if candidate.kind=="measure" else "clear", point, channel)/1000.
            row.extend([float(candidate.kind=="certified_clear"), float(candidate.kind=="grid_clear"),
                (center.x-current.x)/3600., (center.y-current.y)/3600.,
                (center.x-point.x)/3600., (center.y-point.y)/3600., *support,
                lower/1800., upper/1800., len(vertices)/256., float(positive), float(degenerate),
                *self._anchor_features(first,point), *self._anchor_features(latest,point),
                float(bool(state and state.started)),
                (len(state.points)-len(state.attempted))/20000. if state else 0.,
                len(state.attempted)/20000. if state else 0.])
        if len(global_features)!=GLOBAL_DIM or any(len(row)!=CANDIDATE_DIM for row in rows):
            raise ValueError("Micro feature schema mismatch")
        if not all(math.isfinite(x) for row in [global_features]+rows for x in row):
            raise ValueError("Non-finite micro public features")
        return global_features, rows

    def _heuristic(self, candidates):
        current = self.client.state.position
        indexed = list(enumerate(candidates))
        rank = lambda pair: (current.distance_to(pair[1].point),
                             pair[1].channel != self.client.state.current_channel, pair[1].channel)
        same_cover = [pair for pair in indexed if pair[1].kind=="measure" and pair[1].at_cover and pair[1].point==current]
        if same_cover:
            return min(same_cover,key=rank)[0]
        safe = [pair for pair in indexed if pair[1].kind=="certified_clear"]
        if safe:
            return min(safe,key=rank)[0]
        local = [pair for pair in indexed if pair[1].role in ("localize","grid")]
        eligible = [pair for pair in local if self.report.learning["discovery_certified"]
                    or self.regions[pair[1].channel].enclosing_disk().radius<=40.]
        if eligible:
            channels = {candidate.channel for _,candidate in eligible}
            channel = min(channels,key=lambda c:(current.distance_to(self._target(c)),c))
            probes = [pair for pair in eligible if pair[1].channel==channel and pair[1].kind=="measure"]
            if probes and self.local_measurements[channel]<self.max_active_probes:
                center = self._target(channel)
                return min(probes,key=lambda pair:(pair[1].point!=center,*rank(pair)))[0]
            grids = [pair for pair in eligible if pair[1].channel==channel and pair[1].kind=="grid_clear"]
            if grids:
                return min(grids,key=rank)[0]
            if probes:
                return min(probes,key=rank)[0]
        covers = [pair for pair in indexed if pair[1].kind=="measure" and pair[1].at_cover]
        return min(covers or indexed,key=rank)[0]

    def _execute_candidate(self, candidate):
        before_index, before_time = len(self.report.action_history), self.client.state.virtual_time_s
        event = dict(decision=self.report.learning["decisions"], kind=candidate.kind,
            position=[candidate.point.x,candidate.point.y], channel=candidate.channel,
            role=candidate.role, before_actual_action_index=before_index,
            end_actual_action_index=before_index, accepted_requests=0, cost_s=0.)
        self.report.learning["micro_steps"].append(event)
        try:
            if candidate.kind=="measure":
                if not self._can_measure(candidate.point,candidate.channel):
                    raise _StopSearch("micro_stale_or_illegal_measure")
                self._perform("measure",candidate.point,candidate.channel,"micro_"+candidate.role)
                if candidate.role=="localize":
                    self.local_measurements[candidate.channel] += 1
            elif candidate.kind=="certified_clear":
                certificate = self._clear_certificate(candidate.point,candidate.channel)
                if certificate is None:
                    raise _StopSearch("micro_clear_certificate_missing")
                event["clear_certificate"] = certificate
                response = self._perform("clear",candidate.point,candidate.channel,"micro_certified_clear")
                self.report.learning["certified_clear_checks"].append(dict(
                    actual_action_index=before_index, channel=candidate.channel,
                    result=response["clear_result"], **certificate))
                if response["clear_result"]!="success":
                    self.report.error = "Actual certified clear contradicted the conservative public geometry"
                    raise _StopSearch("micro_clear_certificate_contradiction")
            elif candidate.kind=="grid_clear":
                state = self._grid(candidate.channel)
                if (candidate.channel in self.cleared or state is None
                        or candidate.point not in state.points or candidate.point in state.attempted):
                    raise _StopSearch("micro_stale_or_illegal_grid")
                response = self._perform("clear",candidate.point,candidate.channel,"micro_grid_clear")
                state.started = True
                state.attempted.add(candidate.point)
                event["grid_result"] = response["clear_result"]
            else:
                raise _StopSearch("micro_unknown_action_kind")
        finally:
            count = len(self.report.action_history)-before_index
            event.update(end_actual_action_index=len(self.report.action_history),
                         accepted_requests=count, cost_s=self.client.state.virtual_time_s-before_time)
            if count>1:
                raise AssertionError("A micro decision submitted multiple accepted physical requests")

    def _resolve(self, channel):
        state = self.grid_states.get(channel)
        if not self._in_fallback or state is None or not state.started:
            return super()._resolve(channel)
        if channel in self.cleared:
            return True
        # A saved full grid remains a cover after positive-region shrinkage.
        # Resume it instead of recreating a queue that forgets optical misses.
        for point in state.points:
            if point in state.attempted:
                continue
            result = self._perform("clear",point,channel,"micro_fallback_grid_resume")
            state.attempted.add(point)
            if result["clear_result"]=="success":
                return True
        return False

    def _finish_with_baseline(self, reason):
        self._in_fallback = True
        try:
            return super()._finish_with_baseline(reason)
        finally:
            self._in_fallback = False

    def run(self):
        report = super().run()
        report.learning["grid_ledgers"] = {str(channel):dict(
            started=state.started, original_vertices=[list(v) for v in state.vertices],
            complete_queue=[[p.x,p.y] for p in state.points],
            attempted_points=[[p.x,p.y] for p in state.points if p in state.attempted],
            remaining=len(state.points)-len(state.attempted), actually_cleared=channel in self.cleared)
            for channel,state in self.grid_states.items() if state.started}
        return report


def run_q4_micro(client, policy=None, *, problem=4, **kwargs):
    if type(problem) is not int or problem!=4:
        raise ValueError("Q4 micro controller supports problem=4 only")
    return Q4MicroSearch(client,policy=policy,**kwargs).run()
