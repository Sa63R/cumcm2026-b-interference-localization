"""Q4 chain-constrained task scheduling with positive-only localization.

The 31-station triangular discovery certificate is kept. A compact state uses
the executed coverage prefix and a subset of sources, rather than a subset of
every station. Only already-certifiable sources may interrupt discovery.
"""

from dataclasses import asdict
import time

from planning.chain_route import ChainSource, solve_chain_route
from simulator_client.state import Position
from .search import _Search, _StopSearch


class Q4StateSearch(_Search):
    def __init__(self, client, max_actions, max_active_probes, *, mode, max_expansions):
        super().__init__(client, 4, "triangular", max_actions, max_active_probes, "center")
        self.mode = mode
        self.max_expansions = max_expansions
        self.total_expansions = 0
        self.route_log, self.hull_log, self.pair_log = [], [], []
        self.skipped_scans = []
        self.pending_pair = None
        self.blocked = set()
        self.report.strategy_parameters.update(
            q4_state_variant=mode, chain_route_log=self.route_log, positive_hull_log=self.hull_log,
            directional_pair_log=self.pair_log,
            skipped_certified_scans=self.skipped_scans,
            max_route_expansions=max_expansions, max_total_route_expansions=60000,
            state_scope="Fixed remaining coverage chain plus source subset; no unknown sources in predictions",
            discovery_interruption="Only near or enclosing radius <=19.9 m sources",
            negative_observations="Not used to prune source locations or certify coverage without real scans",
        )

    def _target(self, channel):
        if channel in self.near_points:
            return self.near_points[channel]
        region = self.regions.get(channel)
        if region is not None and region.vertices:
            return Position.coerce(region.enclosing_disk().center)
        return None

    def _ready(self, channel):
        region = self.regions.get(channel)
        return channel in self.near_points or bool(
            region and region.vertices and region.enclosing_disk().radius <= 19.9)

    def _scan(self, point):
        channels = [c for c in range(1, 21) if c not in self.cleared]
        if self.mode in {"state_pruned", "state_pruned_rescue"}:
            skipped = [c for c in channels if self._ready(c)]
            for c in skipped:
                self.skipped_scans.append({"channel": c, "position": [point.x, point.y],
                    "after_actual_action_count": len(self.report.action_history),
                    "reason": "near" if c in self.near_points else "enclosing_disk",
                    "radius_m": None if c in self.near_points else self.regions[c].enclosing_disk().radius})
            channels = [c for c in channels if c not in skipped]
        current = self.client.state.current_channel
        if current in channels:
            channels.remove(current)
            channels.insert(0, current)
        for channel in channels:
            self._perform("measure", point, channel, "coverage")
        self.report.coverage_points_visited += 1
        self.blocked.clear()

    def _next_probe(self, channel, index):
        if self.mode in {"state_pair", "state_rescue", "state_pruned_rescue"}:
            from planning.directional_probe_pair import choose_directional_probe_pair
            if self.pending_pair is not None:
                previous_channel, first, second = self.pending_pair
                self.pending_pair = None
                last = self.report.action_history[-1] if self.report.action_history else {}
                if (previous_channel == channel and last.get("channel") == channel
                        and last.get("action") == "measure" and last.get("result") == "no_signal"
                        and last.get("position") == [first.x, first.y]):
                    self.pair_log.append({"channel": channel, "index": index, "role": "second",
                        "after_actual_action_count": len(self.report.action_history),
                        "position": [second.x, second.y], "first": [first.x, first.y]})
                    return second
            last = self.report.action_history[-1] if self.report.action_history else {}
            rescue_needed = (last.get("channel") == channel and last.get("action") == "measure"
                             and last.get("phase") == "active_localization" and last.get("result") == "no_signal")
            if self.max_active_probes-index >= 2 and (self.mode == "state_pair" or rescue_needed):
                pair, log = choose_directional_probe_pair(
                    self.regions[channel], self.client.state.position,
                    self.observed_positions.get(channel, set()))
                self.pair_log.append({"channel": channel, "index": index, "role": "first",
                    "after_actual_action_count": len(self.report.action_history),
                    "pair": asdict(pair) if pair else None, **log})
                if pair is not None:
                    self.pending_pair = channel, pair.first, pair.second
                    return pair.first
        if self.mode == "state_hull":
            from planning.positive_hull_probe import choose_positive_hull_probe
            region = self.regions[channel]
            point, log = choose_positive_hull_probe(
                region, self.client.state.position, self.observed_positions.get(channel, set()))
            self.hull_log.append({"channel": channel, "index": index,
                                  "after_actual_action_count": len(self.report.action_history), **log})
            if point is not None:
                return point
        return super()._next_probe(channel, index)

    def _resolve(self, channel):
        try:
            return super()._resolve(channel)
        finally:
            self.pending_pair = None

    def _execute_plan(self):
        if self.mode == "off":
            return super()._execute_plan()
        remaining = list(self.points)
        while True:
            if not remaining:
                self.report.coverage_complete = True
            if len(self.cleared) == 16:
                self.report.completion_certified_under_model = True
                raise _StopSearch("source_count_upper_bound_reached")
            # With all 16 known AND ready, discovery and further localization
            # scans are unnecessary. Otherwise retain future views of unresolved
            # directional sources instead of forcing expensive optical search.
            known = self.detected | self.cleared
            discovery_done = len(known) == 16 and all(self._ready(c) for c in known-self.cleared)
            covers = [] if discovery_done else remaining
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
            # Each unresolved source's future cover measurements cease after
            # its actual clear. Other channels form an additive background.
            background = 6.*max(0, 20-len(self.cleared)-len(channels))
            result = solve_chain_route(covers, sources, self.client.state.position,
                                      max_expansions=budget, background_scan_s=background,
                                      scan_source_s=0. if self.mode in {"state_pruned", "state_pruned_rescue"} else 6.)
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


def run_q4_state_search(client, *, problem=4, max_actions=20000, max_active_probes=6,
                        mode="state", max_expansions=200):
    if problem != 4 or mode not in {"off", "state", "state_hull", "state_pair", "state_rescue", "state_pruned", "state_pruned_rescue"}:
        raise ValueError("Q4 only; unknown state-search mode")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be integer in [0,30]")
    if type(max_expansions) is not int or not 0 <= max_expansions <= 10000:
        raise ValueError("max_expansions must be integer in [0,10000]")
    return Q4StateSearch(client, max_actions, max_active_probes,
                         mode=mode, max_expansions=max_expansions).run()
