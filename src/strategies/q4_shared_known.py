"""One real, stationary cross-channel measurement after a closed clear macro.

A rounded bearing toward the canonical MEC centre is only a nominal screening
hypothesis. It neither predicts guaranteed reception nor updates live geometry.
The unchanged parent receives the actual reply and retains all safety fallbacks.
"""
from dataclasses import asdict
import math
import time

from planning.matrix_chain_route import ChainSource, solve_matrix_chain_route
from simulator_client.state import Position
from .q4_known_source import Q4KnownSource, CONFIG as MATRIX_CONFIG, _xy
from .search import _StopSearch


CONFIG = "nominal_ready_after_clear"
PHASE = "shared_known_observation"


class Q4SharedKnown(Q4KnownSource):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions,
                 config=CONFIG):
        if config != CONFIG:
            raise ValueError("Unknown shared-known configuration")
        super().__init__(client, max_actions, max_active_probes,
                         max_expansions=max_expansions)
        self.shared_seen_clears = set()
        self.shared_attempted_channels = set()
        self._shared_initial_clears = frozenset(self.cleared)
        self.shared_known_observation_log = []
        self.report.strategy_parameters.update(
            shared_known_config=config,
            shared_known_observation_log=self.shared_known_observation_log,
            shared_known_scope="After a completed clear macro; at most one stationary real measurement per trigger and per target channel; canonical nominal copy only",
            shared_known_limits={"nominal_near_exclusion_m": 5.,
                "nominal_max_distance_m": 1500., "ready_radius_m": 19.9,
                "maximum_actual_measurements_under_public_count": 15,
                "direct_measurement_switch_upper_s": 90.,
                "upper_bound_scope": "Only these measurements and channel switches, not total time or regret"})

    def _shared_candidates(self):
        current = self.client.state.position
        key = (round(current.x, 6), round(current.y, 6))
        candidates = []
        for channel in sorted(self.detected - self.cleared):
            region = self.regions.get(channel)
            disk = region.enclosing_disk() if region and region.vertices else None
            item = dict(channel=channel, original_vertices=[list(v) for v in region.vertices] if region else [],
                positive_observation_count=len(region.observations) if region else 0,
                original_center=list(disk.center) if disk else None,
                original_radius_m=disk.radius if disk else None,
                ready=bool(self._ready(channel)), blocked=channel in self.blocked,
                attempted=channel in self.shared_attempted_channels,
                previously_observed=key in self.observed_positions.get(channel, set()),
                nominal_distance_m=None, nominal_bearing_deg=None,
                predicted_vertices=None, predicted_center=None, predicted_radius_m=None,
                ratio=None, eligible=False, reason=None)
            candidates.append(item)
            if item["blocked"]:
                item["reason"] = "blocked"
            elif item["ready"]:
                item["reason"] = "already_ready"
            elif item["attempted"]:
                item["reason"] = "target_already_attempted"
            elif not region or not region.vertices or not region.observations:
                item["reason"] = "no_canonical_positive_region"
            elif item["previously_observed"]:
                item["reason"] = "position_already_observed"
            elif not math.isfinite(disk.radius) or disk.radius <= 19.9:
                item["reason"] = "invalid_or_ready_radius"
            else:
                distance = current.distance_to(Position.coerce(disk.center))
                item["nominal_distance_m"] = distance
                if not math.isfinite(distance) or not 5. < distance <= 1500.:
                    item["reason"] = "nominal_distance_outside_range"
                    continue
                bearing = round(math.degrees(math.atan2(disk.center[1]-current.y,
                                                       disk.center[0]-current.x)) % 360., 2) % 360.
                item["nominal_bearing_deg"] = bearing
                predicted = region.copy().observe(current, bearing)
                item["predicted_vertices"] = [list(v) for v in predicted.vertices]
                if not predicted.vertices:
                    item["reason"] = "empty_nominal_region"
                    continue
                predicted_disk = predicted.enclosing_disk()
                item.update(predicted_center=list(predicted_disk.center),
                    predicted_radius_m=predicted_disk.radius, ratio=predicted_disk.radius/disk.radius)
                if not math.isfinite(predicted_disk.radius) or predicted_disk.radius > 19.9:
                    item["reason"] = "nominal_not_ready"
                else:
                    item.update(eligible=True, reason="nominal_ready")
        return candidates

    def _share_after_clear(self):
        # This is called only at the parent loop boundary, after all finally
        # blocks have closed. In particular it is outside an early-service slice.
        if (self._joint_context is not None or self._probe_resolving_channel is not None
                or self.service_deadline is not None):
            raise RuntimeError("Shared observation requires a closed parent macro")
        history = self.report.action_history
        trigger = next(((i, action) for i, action in enumerate(history)
            if action["action"] == "clear" and action["result"] == "success"
            and action["channel"] not in self.shared_seen_clears
            and action["channel"] not in self._shared_initial_clears), None)
        if trigger is None:
            return False
        began = time.perf_counter()
        index, action = trigger
        channel = action["channel"]
        prefix = len(history)
        parent = next((e for e in reversed(self.joint_resolvers)
            if e["channel"] == channel and e["after_actual_action_count"] <= index
            < e["end_actual_action_count"]), None)
        early = next((i for i, e in enumerate(self.early_service_log)
            if e["channel"] == channel and e["after_actual_action_count"] <= index
            < e["end_actual_action_count"]), None)
        maximum = self.client.state.max_virtual_duration_s
        event = dict(id=len(self.shared_known_observation_log),
            trigger_clear_action_index=index, trigger_after_action_count=index+1,
            trigger_channel=channel, parent_resolver_id=parent["id"] if parent else None,
            parent_end_action_count=parent["end_actual_action_count"] if parent else None,
            early_service_index=early, after_actual_action_count=prefix, end_actual_action_count=prefix,
            current_position=_xy(self.client.state.position), current_channel=self.client.state.current_channel,
            seen_clear_before=sorted(self.shared_seen_clears), seen_clear_after=None,
            attempted_before=sorted(self.shared_attempted_channels), attempted_after=None,
            known_channels=sorted(self.detected | self.cleared), cleared_channels=sorted(self.cleared),
            blocked_channels=sorted(self.blocked), candidates=[], selected_channel=None,
            position=_xy(self.client.state.position), status="planned", actual_result=None,
            actual_bearing_deg=None, actual_cost_s=0., actual_radius_before_m=None,
            actual_radius_after_m=None, actual_ready_after=None,
            start_virtual_time_s=self.client.state.virtual_time_s,
            end_virtual_time_s=self.client.state.virtual_time_s,
            budget=dict(policy_actions_before=self.actions, max_actions=self.max_actions,
                virtual_limit_s=min(360000., maximum) if maximum is not None else 360000.,
                remaining_real_s=getattr(self.client, "remaining_real_time_s", None)),
            decision_wall_s=0., runtime_s=0.)
        self.shared_known_observation_log.append(event)
        self.shared_seen_clears.add(channel)
        selected = None
        try:
            event["candidates"] = self._shared_candidates()
            eligible = [e for e in event["candidates"] if e["eligible"]]
            if not eligible:
                event["status"] = "no_candidate"
                return False
            selected = min(eligible, key=lambda e: (e["ratio"], e["channel"]))
            target = selected["channel"]
            event.update(selected_channel=target, actual_radius_before_m=selected["original_radius_m"])
            self.shared_attempted_channels.add(target)
            event["decision_wall_s"] = time.perf_counter()-began
            response = self._perform("measure", self.client.state.position, target, PHASE)
            event.update(status="measured", actual_result=response["measure_result"],
                         actual_bearing_deg=response.get("svd_deg"))
            return True  # Re-enter ordinary planning with the real updated belief.
        except Exception as error:
            event.update(status="interrupted", interruption_type=type(error).__name__,
                         interruption_reason=str(error))
            raise
        finally:
            event.update(end_actual_action_count=len(history),
                end_virtual_time_s=self.client.state.virtual_time_s,
                actual_cost_s=self.client.state.virtual_time_s-event["start_virtual_time_s"],
                seen_clear_after=sorted(self.shared_seen_clears),
                attempted_after=sorted(self.shared_attempted_channels))
            if selected is not None:
                region = self.regions.get(selected["channel"])
                event["actual_radius_after_m"] = region.enclosing_disk().radius if region and region.vertices else None
                event["actual_ready_after"] = bool(self._ready(selected["channel"]))
            if not event["decision_wall_s"]:
                event["decision_wall_s"] = time.perf_counter()-began
            event["runtime_s"] = time.perf_counter()-began

    def _execute_plan(self):
        remaining = list(self.points)
        while True:
            if not remaining:
                self.report.coverage_complete = True
            if len(self.cleared) == 16:
                self.report.completion_certified_under_model = True
                raise _StopSearch("source_count_upper_bound_reached")
            if self._share_after_clear():
                continue
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
            sources = [ChainSource(self._target(c), service_s=0.) for c in channels]
            before_expanded = self.total_expansions
            budget = max(0, min(self.max_expansions, 60000-before_expanded))
            result = solve_matrix_chain_route(covers, sources, self.client.state.position,
                max_expansions=budget, background_scan_s=background, source_scan_s=weights)
            self.total_expansions += result.expanded
            kind, index = result.order[0]
            prefix = len(self.report.action_history)
            event = dict(id=len(self.route_log), config=MATRIX_CONFIG,
                after_actual_action_count=prefix, end_actual_action_count=prefix,
                current_position=_xy(self.client.state.position),
                current_channel=self.client.state.current_channel,
                known_channels=sorted(known), cleared_channels=sorted(self.cleared),
                blocked_channels=sorted(self.blocked),
                remaining_covers=[_xy(p) for p in covers], source_channels=channels,
                source_positions=[_xy(s.position) for s in sources], source_services_s=[0.]*len(channels),
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


def run_q4_shared_known(client, *, problem=4, config=CONFIG, max_actions=20000,
                        max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config != CONFIG:
        raise ValueError("Q4 only; config must be nominal_ready_after_clear")
    for value, low, high in ((max_actions,2,1_000_000),(max_active_probes,0,30),(max_expansions,0,10000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid integer execution budget")
    return Q4SharedKnown(client,max_actions,max_active_probes,
                         max_expansions=max_expansions,config=config).run()
