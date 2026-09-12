"""Optimistic release forecasts inside the fixed-chain known-source proxy.

No forecast is an actual observation or a readiness certificate. Every solve
freezes source targets at current real estimates; only the completed-cover
index at which a source may appear in the proxy changes. Actual first-source
execution during discovery still requires the current real ready predicate.
"""
from dataclasses import asdict
import math
import time

from planning.release_chain_route import ChainSource, solve_release_chain_route
from simulator_client.state import Position
from .q4_known_source import Q4KnownSource, _xy
from .search import _StopSearch


CONFIG = "nominal_release"


class Q4ReleaseScheduling(Q4KnownSource):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions,
                 config=CONFIG):
        if config != CONFIG:
            raise ValueError("Unknown release-scheduling configuration")
        super().__init__(client, max_actions, max_active_probes,
                         max_expansions=max_expansions)
        self.report.strategy_parameters.update(
            release_scheduling_config=config, release_scheduling_plan_log=self.route_log,
            state_scope="Fixed cover chain and actual known targets with optimistic, prefix-frozen release indices; only actual ready first-source macros during coverage",
            known_source_proxy={"source_service_s": 5., "scan_upper_s": 6.,
                "range_margin_m": self.report.strategy_parameters["known_source_proxy"]["range_margin_m"],
                "omitted": "Nonready localization, optical search labor and service exit changes; five seconds is a common clear service surrogate, not complete real service cost",
                "matrix_scope": "Original actual-prefix ready/range matrix, unaffected by nominal observations"},
            release_forecast_limits={"minimum_direction_distance_m": 5.,
                "maximum_direction_distance_m": 1500., "ready_radius_m": 19.9,
                "bearing_decimal_places": 2,
                "scope": "Optimistic geometric schedule; fixed current target; no guaranteed reception, radio posterior or real geometry update"})

    def _release_forecast(self, channel, covers, target):
        target = Position.coerce(target)
        region = self.regions.get(channel)
        valid_vertices = bool(region and region.vertices and
            all(all(math.isfinite(x) for x in v) for v in region.vertices))
        try:
            disk = region.enclosing_disk() if valid_vertices else None
        except (ValueError, ArithmeticError):
            disk = None
        ready = channel in self.near_points or bool(disk and disk.radius <= 19.9)
        result = dict(channel=channel, target=_xy(target), ready_before=ready,
            initial_radius_m=disk.radius if disk else None,
            release_index=len(covers), status="planned", steps=[])
        if ready:
            result.update(release_index=0, status="actual_ready")
            return result
        if not covers:
            result.update(release_index=0, status="no_remaining_covers")
            return result
        if (region is None or not region.vertices or not region.observations or disk is None
                or not math.isfinite(disk.radius) or disk.radius <= 19.9
                or any(not all(math.isfinite(x) for x in vertex) for vertex in region.vertices)):
            result["status"] = "fallback_invalid_canonical"
            return result
        predicted = region.copy()
        for index, cover in enumerate(covers):
            q = Position.coerce(cover)
            distance = q.distance_to(target)
            step = dict(cover_index=index, position=_xy(q), nominal_distance_m=distance,
                bearing_deg=None, predicted_radius_m=predicted.enclosing_disk().radius,
                status="planned", stopped=False)
            result["steps"].append(step)
            if not math.isfinite(distance):
                step.update(status="invalid_distance", stopped=True)
                result["status"] = "fallback_invalid_prediction"
                return result
            if distance <= 5.:
                step["status"] = "skip_near_distance"
                continue
            if distance > 1500.:
                step["status"] = "skip_beyond_reception_radius"
                continue
            bearing = round(math.degrees(math.atan2(target.y-q.y, target.x-q.x)) % 360., 2) % 360.
            step["bearing_deg"] = bearing
            try:
                predicted.observe(q, bearing)
            except (ValueError, ArithmeticError):
                step.update(status="invalid_prediction", predicted_radius_m=None, stopped=True)
                result["status"] = "fallback_invalid_prediction"
                return result
            if not predicted.vertices:
                step.update(status="empty_prediction", predicted_radius_m=None, stopped=True)
                result["status"] = "fallback_empty_prediction"
                return result
            try:
                radius = predicted.enclosing_disk().radius
            except (ValueError, ArithmeticError):
                step.update(status="invalid_prediction", predicted_radius_m=None, stopped=True)
                result["status"] = "fallback_invalid_prediction"
                return result
            step["predicted_radius_m"] = radius
            if not math.isfinite(radius) or any(not all(math.isfinite(x) for x in v) for v in predicted.vertices):
                step.update(status="invalid_prediction", stopped=True)
                result["status"] = "fallback_invalid_prediction"
                return result
            if radius <= 19.9:
                step.update(status="predicted_ready", stopped=True)
                result.update(release_index=index+1, status="nominal_ready")
                return result
            step["status"] = "predicted_not_ready"
        result["status"] = "cover_tail"
        return result

    def _execute_plan(self):
        remaining = list(self.points)
        while True:
            if not remaining:
                self.report.coverage_complete = True
            if len(self.cleared) == 16:
                self.report.completion_certified_under_model = True
                raise _StopSearch("source_count_upper_bound_reached")
            known = self.detected | self.cleared
            discovery_done = len(known) == 16
            covers = [] if discovery_done else remaining
            if discovery_done and not self.discovery_stop_log:
                self.discovery_stop_log.append({"after_actual_action_count": len(self.report.action_history),
                    "known_channels": sorted(known), "unresolved_channels": sorted(known-self.cleared),
                    "omitted_cover_stations": len(remaining), "reason": "public_source_count_upper_bound",
                    "discovery_only_not_removal": True})
            if covers:
                candidate = self._early_candidate(covers[0])
                if candidate is not None:
                    self._early_service(candidate)
                    continue
            channels = [c for c in sorted(self.detected-self.cleared-self.blocked)
                        if self._target(c) is not None]
            if not channels:
                if not covers:
                    return
                self._scan(covers[0])
                remaining.pop(0)
                continue
            began = time.perf_counter()
            background, weights, bg_channels, evidence, cells = self._scan_matrix(covers, channels)
            sources = [ChainSource(self._target(c), service_s=5.) for c in channels]
            forecasts = [self._release_forecast(c, covers, source.position)
                         for c, source in zip(channels, sources)]
            releases = [item["release_index"] for item in forecasts]
            before_expanded = self.total_expansions
            budget = max(0, min(self.max_expansions, 60000-before_expanded))
            result = solve_release_chain_route(covers, sources, self.client.state.position,
                max_expansions=budget, background_scan_s=background, source_scan_s=weights,
                release_indices=releases)
            self.total_expansions += result.expanded
            kind, index = result.order[0]
            prefix = len(self.report.action_history)
            event = dict(id=len(self.route_log), config=CONFIG,
                after_actual_action_count=prefix, end_actual_action_count=prefix,
                current_position=_xy(self.client.state.position),
                current_channel=self.client.state.current_channel,
                known_channels=sorted(known), cleared_channels=sorted(self.cleared),
                blocked_channels=sorted(self.blocked),
                remaining_covers=[_xy(p) for p in covers], source_channels=channels,
                source_positions=[_xy(s.position) for s in sources], source_services_s=[5.]*len(channels),
                release_indices=releases, release_forecasts=forecasts,
                source_evidence=[evidence[c] for c in channels],
                background_channels=bg_channels, background_evidence=[evidence[c] for c in bg_channels],
                background_scan_s=background, source_scan_s=weights, matrix_evidence=cells,
                max_expansions=budget, total_expansions_before=before_expanded,
                total_expansions_after=self.total_expansions,
                discovery_count_cap=discovery_done, selected_kind=kind,
                selected_channel=channels[index] if kind == "source" else None,
                result=asdict(result), runtime_s=time.perf_counter()-began, status="planned")
            self.route_log.append(event)
            service = None
            if kind == "source":
                channel = channels[index]
                item = evidence[channel]
                service = dict(decision_id=event["id"],
                    selected=dict(channel=channel, ready_before=item["ready"], radius_m=item["radius_m"]),
                    remaining_covers_before=[_xy(p) for p in covers],
                    resolver_start_action_count=prefix, service_end_action_count=prefix,
                    resolver_id=len(self.joint_resolvers), status="planned",
                    start_virtual_time_s=self.client.state.virtual_time_s,
                    start_position=_xy(self.client.state.position), actual_cost_s=0.)
                self.known_source_service_log.append(service)
            try:
                if kind == "cover":
                    self._scan(covers[0])
                    remaining.pop(0)
                    event["status"] = "cover_completed"
                else:
                    # The proxy may contain future ready tasks, but only the
                    # CURRENT actual certificate can authorize a cover-phase
                    # full service. Known16/no remaining covers reopens all.
                    if covers and not self._ready(channels[index]):
                        raise RuntimeError("Release planner selected a nonready first source during coverage")
                    resolved = self._resolve(channels[index])
                    if not resolved:
                        self.blocked.add(channels[index])
                    service["status"] = event["status"] = "resolved" if resolved else "blocked"
            except Exception as error:
                event.update(status="interrupted", interruption_type=type(error).__name__,
                             interruption_reason=str(error))
                if service is not None:
                    service.update(status="interrupted", interruption_type=type(error).__name__,
                                   interruption_reason=str(error))
                raise
            finally:
                event["end_actual_action_count"] = len(self.report.action_history)
                if service is not None:
                    service.update(service_end_action_count=len(self.report.action_history),
                        actual_cost_s=self.client.state.virtual_time_s-service["start_virtual_time_s"],
                        end_position=_xy(self.client.state.position),
                        cleared=channels[index] in self.cleared)


def run_q4_release_scheduling(client, *, problem=4, config=CONFIG, max_actions=20000,
                              max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config != CONFIG:
        raise ValueError("Q4 only; config must be nominal_release")
    for value, low, high in ((max_actions,2,1_000_000),(max_active_probes,0,30),(max_expansions,0,10000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid integer execution budget")
    return Q4ReleaseScheduling(client,max_actions,max_active_probes,
                               max_expansions=max_expansions,config=config).run()
