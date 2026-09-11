"""Q3 experiment: full nominal source service modes in a frozen route model.

Only the first selected probe is executed. Subsequent feedback, certificates,
and continuation come from the unchanged v1 policy and real observations.
Mean exits and nominal costs are scheduling proxies, never physical evidence.
"""

from dataclasses import asdict
import math
import time

from planning.mode_route import RouteMode, solve_mode_route
from planning.probe_candidates import long_axis
from planning.source_modes import predict_source_mode
from simulator_client.state import Position
from .relocating_state_search import RelocatingStateSearch


class ModeStateSearch(RelocatingStateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, mode):
        super().__init__(client, max_actions, max_active_probes, config, True)
        self.mode_variant = mode
        self.mode_cache = {}
        self.mode_log = []
        self.pending_probe = None
        self.mode_expansions = 0
        self.report.strategy_parameters.update(
            source_service_modes=mode, source_mode_log=self.mode_log,
            source_mode_scope="3 nominal feasible geometry nodes; mean exit; frozen additive route only; no posterior or Q3 bound",
            source_mode_limits={"max_modes": 4, "max_probes": max_active_probes,
                                "supports": 3, "route_expansions": 100},
        )

    def _source_modes(self, channel):
        current = self.client.state.position
        region = self.regions[channel]
        if channel in self.near_points:
            p = self.near_points[channel]
            return [(RouteMode(p, p, 5.), None)], {"kind": "near"}
        disk = region.enclosing_disk()
        if disk is None:
            return [], {"kind": "no_region"}
        if disk.radius <= 19.9:
            center = Position(*disk.center)
            radius = max(0., 19.9-disk.radius)
            d = current.distance_to(center)
            p = current if d <= radius else Position(
                center.x + (current.x-center.x)*radius/d,
                center.y + (current.y-center.y)*radius/d)
            return [(RouteMode(p, p, 5.), None)], {"kind": "certified"}
        # Includes the old first probe; fixed geometry alternatives are cacheable
        # when another source is processed without changing this source's data.
        old = super()._next_probe(channel, 0)
        center, axis, _, low, high = long_axis(region)
        candidates = [old, Position(*disk.center)]
        for q in (.25, .75):
            along = low + q*(high-low)
            candidates.append(Position(center.x+along*axis[0], center.y+along*axis[1]))
        observed = self.observed_positions.get(channel, set())
        geometry_key = (channel, tuple(region.vertices), tuple(sorted(observed)),
                        tuple(region.no_signal_positions))
        seen, modes, logs = set(), [], []
        for p in candidates:
            if p is None:
                continue
            key = (round(p.x, 6), round(p.y, 6))
            if key in seen or key in observed:
                continue
            seen.add(key)
            cache_key = geometry_key + (key,)
            cached = cache_key in self.mode_cache
            if not cached:
                self.mode_cache[cache_key] = predict_source_mode(
                    region, p, self.first_bearings[channel], observed,
                    max_probes=self.max_active_probes,
                    uncertainty_weight=self.state_config.probe_uncertainty_weight)
            prediction = self.mode_cache[cache_key]
            logs.append({"entry": [p.x, p.y], "valid": prediction.valid,
                         "reason": prediction.reason, "cached": cached,
                         "cost_s": prediction.cost_s if math.isfinite(prediction.cost_s) else None,
                         "exit": ([prediction.exit_position.x, prediction.exit_position.y]
                                  if prediction.exit_position else None),
                         "hypotheses": prediction.hypotheses,
                         "measurements_mean": prediction.measurements_mean})
            if prediction.valid:
                modes.append((RouteMode(p, prediction.exit_position, prediction.cost_s), p))
        return modes, {"kind": "nominal", "candidates": logs,
                       "old_probe": [old.x, old.y] if old is not None else None}

    def _next_task(self, remaining):
        old_task = super()._next_task(remaining)
        self.pending_probe = None
        if self.mode_variant == "off" or old_task is None:
            return old_task
        began = time.perf_counter()
        current = self.client.state.position
        if self.mode_variant == "local":
            if old_task[0] != "source":
                return old_task
            modes, details = self._source_modes(old_task[1])
            chosen_data = None
            if modes:
                chosen, probe = min(modes, key=lambda m: current.distance_to(m[0].entry)/5+m[0].service_s)
                chosen_data = {"entry": [chosen.entry.x, chosen.entry.y], "cost_s": chosen.service_s,
                               "exit": [chosen.exit.x, chosen.exit.y], "is_probe": probe is not None}
                if probe is not None:
                    self.pending_probe = old_task[1], probe
            self.mode_log.append({"after_actual_action_count": len(self.report.action_history),
                                  "variant": "local", "channel": old_task[1], **details,
                                  "chosen": chosen_data,
                                  "logged_at_current_channel": self.client.state.current_channel,
                                  "valid_modes": len(modes), "runtime_s": time.perf_counter()-began})
            return old_task
        sources = [("source", c, self._target(c))
                   for c in sorted(self.detected-self.cleared-self.blocked)
                   if self._target(c) is not None]
        covers = [("cover", None, p) for p in remaining]
        if self.state_config.stop_discovery_at_16 and len(self.detected | self.cleared) == 16:
            covers = []
        tasks = covers + sources
        background = max(0, 20-len(self.cleared)-len(sources))
        groups, probes, details = [], [], []
        for kind, channel, p in tasks:
            if kind == "cover":
                modes, info = [(RouteMode(p, p, 6.*background), None)], {"kind": "cover"}
            else:
                modes, info = self._source_modes(channel)
                if not modes:
                    # Whole group reverts to v1's point surrogate. Never mix a
                    # cheap unfinished continuation with valid complete modes.
                    modes = [(RouteMode(p, p, 5.), None)]
                    info["point_fallback"] = True
            groups.append([m for m, _ in modes])
            probes.append([probe for _, probe in modes])
            details.append({"channel": channel, **info})
        budget = max(0, min(100, 60000-self.mode_expansions))
        result = solve_mode_route(groups, current, max_expansions=budget)
        self.mode_expansions += result.expanded
        group, mode = result.order[0]
        selected = tasks[group]
        chosen = groups[group][mode]
        if probes[group][mode] is not None:
            self.pending_probe = selected[1], probes[group][mode]
        self.mode_log.append({"after_actual_action_count": len(self.report.action_history),
                              "variant": "joint", "groups": details,
                              "old_selected": [old_task[0], old_task[1], [old_task[2].x, old_task[2].y]],
                              "selected": [selected[0], selected[1], [selected[2].x, selected[2].y]],
                              "logged_at_current_channel": self.client.state.current_channel,
                              "chosen": {"entry": [chosen.entry.x, chosen.entry.y], "cost_s": chosen.service_s,
                                         "exit": [chosen.exit.x, chosen.exit.y], "is_probe": probes[group][mode] is not None},
                              "selected_mode": mode, "route": asdict(result),
                              "runtime_s": time.perf_counter()-began})
        return selected

    def _next_probe(self, channel, index):
        if index == 0 and self.pending_probe and self.pending_probe[0] == channel:
            _, p = self.pending_probe
            self.pending_probe = None
            region = self.regions[channel]
            key = round(p.x, 6), round(p.y, 6)
            if (key not in self.observed_positions.get(channel, set()) and region.vertices
                    and all(p.distance_to(Position(*v)) <= 1000 for v in region.vertices)):
                return p
        return super()._next_probe(channel, index)

    def _resolve(self, channel):
        try:
            return super()._resolve(channel)
        finally:
            self.pending_probe = None


def run_mode_state_search(client, *, problem=3, max_actions=10000,
                          max_active_probes=6, config=None, mode="joint"):
    if problem != 3 or mode not in {"off", "local", "joint"}:
        raise ValueError("Q3 only; mode must be off, local, or joint")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be integer in [0,30]")
    return ModeStateSearch(client, max_actions, max_active_probes, config, mode).run()
