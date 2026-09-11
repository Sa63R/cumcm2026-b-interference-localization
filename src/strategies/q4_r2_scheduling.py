"""Scheduling-only Q4 experiment; fixed compact_22 and unchanged active probes.

Sixteen distinct positively observed/cleared channels exhaust the public source
count upper bound. This certifies discovery only, never successful removal.
The optional on-route service uses the original resolver until its scheduling
time slice expires. It does not generate alternative probes or optical grids.
"""
from dataclasses import asdict
import math
import time

from planning.chain_route import ChainSource, solve_chain_route
from simulator_client.state import Position
from .q4_cover_search import Q4CoverSearch
from .search import _StopSearch


CONFIGS = {
    "cap16": {"early_services": 0, "radius_m": 40., "detour_m": 100., "service_budget_s": 60.},
    "onroute": {"early_services": 4, "radius_m": 40., "detour_m": 100., "service_budget_s": 60.},
}


class _ServiceSliceExpired(Exception):
    pass


class Q4R2Scheduling(Q4CoverSearch):
    def __init__(self, client, max_actions, max_active_probes, *, config, max_expansions):
        if config not in CONFIGS:
            raise ValueError("Unknown scheduling configuration")
        super().__init__(client, max_actions, max_active_probes, profile="compact_22",
                         schedule="joint", max_expansions=max_expansions)
        self.scheduling_config = dict(CONFIGS[config])
        self.early_attempted = set()
        self.service_deadline = None
        self.early_service_log, self.discovery_stop_log = [], []
        self.report.strategy_parameters.update(q4_r2_scheduling=config,
            scheduling_config=self.scheduling_config, early_service_log=self.early_service_log,
            discovery_stop_log=self.discovery_stop_log,
            discovery_interruption="16 positively known channels certify discovery, regardless of localization readiness",
            early_service_scope="Unchanged resolver with at most four 60-second scheduling slices; at most one per channel",
            active_probe_algorithm="Unchanged Q4CoverSearch._next_probe and inherited _resolve")

    def _perform(self, action, position, channel, phase):
        if self.service_deadline is not None:
            point = Position.coerce(position)
            movement = math.ceil(self.client.state.position.distance_to(point) / 5 * 1e6) / 1e6
            # Five seconds safely includes optical success or one measurement.
            cost = movement + 5. + (action == "measure" and channel != self.client.state.current_channel)
            if self.client.state.virtual_time_s + cost > self.service_deadline:
                raise _ServiceSliceExpired()
        return super()._perform(action, position, channel, phase)

    def _early_candidate(self, next_cover):
        cfg = self.scheduling_config
        if len(self.early_attempted) >= cfg["early_services"]:
            return None
        current = self.client.state.position
        choices = []
        for channel in sorted(self.detected - self.cleared - self.blocked - self.early_attempted):
            if self._ready(channel):
                continue  # Existing ready-source routing remains in charge.
            region = self.regions.get(channel)
            if region is None or not region.vertices:
                continue
            disk = region.enclosing_disk()
            if not math.isfinite(disk.radius) or disk.radius > cfg["radius_m"]:
                continue
            center = Position.coerce(disk.center)
            detour = current.distance_to(center) + center.distance_to(next_cover) - current.distance_to(next_cover)
            first_cost = current.distance_to(center) / 5. + 6.
            if detour <= cfg["detour_m"] and first_cost <= cfg["service_budget_s"]:
                choices.append((detour, current.distance_to(center), channel, disk.radius))
        return min(choices) if choices else None

    def _early_service(self, candidate):
        detour, _, channel, radius = candidate
        before = self.client.state.virtual_time_s
        event = {"channel": channel, "radius_m": radius, "detour_m": detour,
                 "after_actual_action_count": len(self.report.action_history),
                 "budget_s": self.scheduling_config["service_budget_s"], "interrupted": False}
        self.early_attempted.add(channel)
        self.service_deadline = before + self.scheduling_config["service_budget_s"]
        try:
            self._resolve(channel)
        except _ServiceSliceExpired:
            event["interrupted"] = True
        finally:
            self.service_deadline = None
            event.update(actual_cost_s=self.client.state.virtual_time_s-before,
                         cleared=channel in self.cleared,
                         end_actual_action_count=len(self.report.action_history))
            self.early_service_log.append(event)

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
                        if self._target(c) is not None and (not covers or self._ready(c))]
            if not channels:
                if not covers:
                    return
                self._scan(covers[0])
                remaining.pop(0)
                continue
            began = time.perf_counter()
            sources = [ChainSource(self._target(c), service_s=5.) for c in channels]
            budget = max(0, min(self.max_expansions, 60000-self.total_expansions))
            result = solve_chain_route(covers, sources, self.client.state.position,
                max_expansions=budget, background_scan_s=6.*max(0, 20-len(self.cleared)-len(channels)),
                scan_source_s=0.)
            self.total_expansions += result.expanded
            kind, index = result.order[0]
            self.route_log.append({"after_actual_action_count": len(self.report.action_history),
                "remaining_covers": [[p.x, p.y] for p in covers], "source_channels": channels,
                "source_positions": [[s.position.x, s.position.y] for s in sources],
                "discovery_count_cap": discovery_done, "selected_kind": kind,
                "selected_channel": channels[index] if kind == "source" else None,
                "result": asdict(result), "runtime_s": time.perf_counter()-began})
            if kind == "cover":
                self._scan(covers[0])
                remaining.pop(0)
            elif not self._resolve(channels[index]):
                self.blocked.add(channels[index])


def run_q4_r2_scheduling(client, *, problem=4, config="cap16", max_actions=20000,
                         max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config not in CONFIGS:
        raise ValueError("Q4 only; config must be cap16 or onroute")
    for value, minimum, maximum in ((max_actions, 2, 1_000_000), (max_active_probes, 0, 30),
                                    (max_expansions, 0, 10000)):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("Invalid integer execution budget")
    return Q4R2Scheduling(client, max_actions, max_active_probes,
                          config=config, max_expansions=max_expansions).run()
