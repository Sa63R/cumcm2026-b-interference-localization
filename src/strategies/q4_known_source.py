"""R12 full known-source macros interleaved with the unchanged cover chain.

The matrix is a frozen scanning proxy, not a simulation of future feedback.
Source service is deliberately zero in this proxy: all localization/optical
labor and the changing service exit are omitted, not claimed to be free in
the real task. Every real action still uses the complete inherited resolver.
"""
from dataclasses import asdict
import time

from planning.matrix_chain_route import ChainSource, solve_matrix_chain_route
from .q4_joint_continuation import Q4JointContinuation
from .q4_range_pruning import RANGE_MARGIN_M, region_distance_lower
from .search import _StopSearch


CONFIG = "all_known_matrix"


def _xy(point):
    return [point.x, point.y] if point is not None else None


class Q4KnownSource(Q4JointContinuation):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions,
                 config=CONFIG):
        if config != CONFIG:
            raise ValueError("Unknown known-source configuration")
        super().__init__(client, max_actions, max_active_probes, max_expansions=max_expansions)
        self.known_source_service_log = []
        self.report.strategy_parameters.update(
            known_source_config=config, known_source_plan_log=self.route_log,
            known_source_service_log=self.known_source_service_log,
            state_scope="All actual known live unblocked targets, with fixed cover chain and per-cover/source scan matrix",
            known_source_proxy={"source_service_s": 0., "scan_upper_s": 6.,
                "range_margin_m": RANGE_MARGIN_M,
                "omitted": "Real localization, optical labor and service exit movement; no real-time or global-optimum guarantee",
                "matrix_scope": "Current-prefix ready/range certificates held fixed; planned visits do not create observations or clearance"})

    def _channel_evidence(self, channel):
        region = self.regions.get(channel)
        disk = region.enclosing_disk() if region and region.vertices else None
        return dict(channel=channel, detected=channel in self.detected,
            blocked=channel in self.blocked, target=_xy(self._target(channel)),
            ready=bool(self._ready(channel)), radius_m=disk.radius if disk else None,
            near_point=_xy(self.near_points.get(channel)),
            vertices=[list(v) for v in region.vertices] if region else [],
            positive_observation_count=len(region.observations) if region else 0)

    def _scan_matrix(self, covers, channels):
        live = [c for c in range(1, 21) if c not in self.cleared]
        source_set = set(channels)
        background_channels = [c for c in live if c not in source_set]
        evidence = {c: self._channel_evidence(c) for c in live}
        background, weights, cells = [], [], []
        for point in covers:
            row = []
            fees = {}
            for c in live:
                item = evidence[c]
                lower = None
                if item["ready"]:
                    cost, reason = 0., "ready"
                elif item["detected"] and item["positive_observation_count"]:
                    lower = region_distance_lower(point, item["vertices"])
                    if lower > 1500.+RANGE_MARGIN_M:
                        cost, reason = 0., "positive_region_beyond_max_reception_radius"
                    else:
                        cost, reason = 6., "measurement_proxy"
                else:
                    cost, reason = 6., "measurement_proxy"
                fees[c] = cost
                row.append(dict(channel=c, fee_s=cost, reason=reason,
                                distance_lower_bound_m=lower))
            background.append(sum(fees[c] for c in background_channels))
            weights.append([fees[c] for c in channels])
            cells.append(row)
        return background, weights, background_channels, evidence, cells

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
            sources = [ChainSource(self._target(c), service_s=0.) for c in channels]
            before_expanded = self.total_expansions
            budget = max(0, min(self.max_expansions, 60000-before_expanded))
            result = solve_matrix_chain_route(covers, sources, self.client.state.position,
                max_expansions=budget, background_scan_s=background, source_scan_s=weights)
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


def run_q4_known_source(client, *, problem=4, config=CONFIG, max_actions=20000,
                        max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config != CONFIG:
        raise ValueError("Q4 only; config must be all_known_matrix")
    for value, low, high in ((max_actions,2,1_000_000),(max_active_probes,0,30),(max_expansions,0,10000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid integer execution budget")
    return Q4KnownSource(client,max_actions,max_active_probes,
                         max_expansions=max_expansions,config=config).run()
