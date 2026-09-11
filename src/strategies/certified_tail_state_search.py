"""Execute an improved complete deterministic clear tail, or retain v1."""

import time

from planning.certified_tail import CertifiedDisk, improve_certified_tail, predict_baseline_tail
from simulator_client.state import Position
from .relocating_state_search import RelocatingStateSearch


class CertifiedTailStateSearch(RelocatingStateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, enabled):
        super().__init__(client, max_actions, max_active_probes, config, True)
        self.certified_tail_enabled = enabled
        self.tail_log = []
        self.tail_visits = []
        self.tail_expected_key = None
        self.tail_record = None
        self.report.strategy_parameters.update({
            "certified_tail_refinement": enabled, "certified_tail_log": self.tail_log,
            "certified_tail_scope": "Completed discovery; 2..4 entirely certified sources; original safe disks only; compare full executable deterministic tails with stepwise microsecond rounding.",
            "certified_tail_limits": {"max_sources": 4, "max_permutations": 24,
                                      "coordinate_sweeps": 32, "line_search_steps": 12,
                                      "minimum_gain_us": 10000},
        })

    def _certified_disks(self, remaining):
        if self.blocked:
            return None
        if remaining and not (self.state_config.stop_discovery_at_16
                              and len(self.detected | self.cleared) == 16):
            return None
        channels = sorted(self.detected - self.cleared)
        if not 2 <= len(channels) <= 4:
            return None
        disks = []
        for channel in channels:
            if channel in self.near_points:
                disks.append(CertifiedDisk(channel, self.near_points[channel], 14.9, True))
                continue
            region = self.regions.get(channel)
            if not region or not region.vertices:
                return None
            circle = region.enclosing_disk()
            if circle.radius > 19.9:
                return None
            disks.append(CertifiedDisk(channel, Position(*circle.center),
                                       max(0.0, 19.9 - circle.radius)))
        return disks

    def _state_key(self, remaining):
        sources = []
        for channel in sorted(self.detected - self.cleared):
            region = self.regions.get(channel)
            sources.append((channel, self.near_points.get(channel),
                            tuple(region.vertices) if region else None,
                            tuple(region.no_signal_positions) if region else None))
        return (self.client.state.position, round(self.client.state.virtual_time_s * 1_000_000),
                self.client.state.current_channel, frozenset(self.detected), frozenset(self.cleared),
                frozenset(self.blocked), tuple(remaining), len(self.report.action_history), tuple(sources))

    def _valid_point(self, channel, position):
        if channel in self.near_points:
            return position.distance_to(self.near_points[channel]) + 5.0 <= 20.0 - 1e-6
        region = self.regions.get(channel)
        return bool(region and region.vertices and all(
            position.distance_to(Position(*v)) <= 20.0 - 1e-6 for v in region.vertices))

    def _cancel_tail(self, reason):
        if self.tail_record is not None:
            self.tail_record["cancelled"] = reason
            self.tail_record["dominance_status"] = "execution_preconditions_invalidated"
        self.tail_visits = []
        self.tail_expected_key = None
        self.tail_record = None

    def _next_task(self, remaining):
        if not self.certified_tail_enabled:
            return super()._next_task(remaining)
        self.remaining_covers = remaining
        if self.tail_visits:
            if self.tail_expected_key == self._state_key(remaining):
                visit = self.tail_visits[0]
                return ("source", visit.channel, visit.position)
            self._cancel_tail("observed_state_changed")
            return super()._next_task(remaining)
        disks = self._certified_disks(remaining)
        before_expanded = self.total_expansions
        selected = super()._next_task(remaining)
        if not disks or selected is None:
            return selected
        real_remaining = getattr(self.client, "remaining_real_time_s", None)
        if real_remaining is not None and real_remaining <= 10.0:
            self.report.strategy_parameters["certified_tail_real_budget_skip"] = (
                self.report.strategy_parameters.get("certified_tail_real_budget_skip", 0) + 1)
            return selected
        if self.actions + len(disks) + 1 > self.max_actions:
            # Do not change the order when a complete tail plus explicit exit
            # is already known to exceed the remaining action budget.
            self.report.strategy_parameters["certified_tail_action_budget_skip"] = (
                self.report.strategy_parameters.get("certified_tail_action_budget_skip", 0) + 1)
            return selected
        began = time.perf_counter()
        current = self.client.state.position
        baseline = predict_baseline_tail(disks, current,
            max_expansions=self.state_config.max_expansions,
            max_total_expansions=self.state_config.max_total_expansions,
            total_expansions=before_expanded,
            scan_source_s=self.state_config.scan_source_s)
        record = {"after_actual_action_count": len(self.report.action_history),
                  "position": [current.x, current.y], "source_count": len(disks),
                  "baseline_cost_us": baseline.cost_us,
                  "baseline_visits": [[v.channel, v.position.x, v.position.y] for v in baseline.visits],
                  "baseline_expanded_after_each": list(baseline.expanded_after_each),
                  "start_time_us": round(self.client.state.virtual_time_s * 1_000_000),
                  "accepted": False, "executed": []}
        self.tail_log.append(record)
        maximum = self.client.state.max_virtual_duration_s
        maximum = min(360000.0, maximum) if maximum is not None else 360000.0
        # The inherited per-action guard compares unrounded travel, whereas
        # actual costs round each segment to microseconds. One millisecond of
        # reserve safely exceeds that rounding difference for at most 4 moves.
        if self.client.state.virtual_time_s + baseline.cost_us / 1e6 > maximum - 0.001:
            record["reason"] = "full_baseline_tail_exceeds_virtual_budget"
            return selected
        if (selected[0] != "source" or selected[1] != baseline.visits[0].channel
                or self.total_expansions != baseline.expanded_after_each[0]):
            record["reason"] = "first_baseline_step_mismatch"
            return selected
        candidate, cost_us = improve_certified_tail(disks, current, baseline.visits)
        record.update(candidate_cost_us=cost_us, gain_us=baseline.cost_us - cost_us,
                      candidate_visits=[[v.channel, v.position.x, v.position.y] for v in candidate],
                      planning_runtime_s=time.perf_counter() - began)
        if baseline.cost_us - cost_us < 10000:
            record["reason"] = "gain_below_10ms"
            return selected
        if not all(self._valid_point(v.channel, v.position) for v in candidate):
            record["reason"] = "candidate_certificate_failed"
            return selected
        self.tail_visits = list(candidate)
        self.tail_expected_key = self._state_key(remaining)
        self.tail_record = record
        record["accepted"] = True
        record["dominance_status"] = "pending_complete_execution"
        return ("source", candidate[0].channel, candidate[0].position)

    def _clear(self, position, channel, phase):
        if not self.certified_tail_enabled or not self.tail_visits:
            return super()._clear(position, channel, phase)
        visit = self.tail_visits[0]
        if (visit.channel != channel or phase not in ("near_clear", "certified_clear")
                or self.tail_expected_key != self._state_key(self.remaining_covers)
                or not self._valid_point(channel, visit.position)):
            self._cancel_tail("clear_precondition_changed")
            return super()._clear(position, channel, phase)
        # EfficientSearch._clear would replace the selected safe point by its
        # incoming-edge projection. Use the inherited actual action machinery
        # directly so the committed complete route is what gets executed.
        before = self.tail_expected_key
        expected_time = before[1] + round(before[0].distance_to(visit.position) / 5 * 1_000_000) + 5_000_000
        expected_after = (visit.position, expected_time, before[2], before[3],
                          before[4] | {channel}, before[5], before[6], before[7] + 1,
                          tuple(item for item in before[8] if item[0] != channel))
        try:
            response = self._perform("clear", visit.position, channel, "certified_tail_clear")
            if response["clear_result"] != "success":
                self._cancel_tail("clear_failed")
            return response["clear_result"] == "success"
        finally:
            # The public sixteen-source stop may raise inside _perform after
            # its successful physical clear. Keep logs truthful in that case.
            if channel in self.cleared:
                self.tail_record["executed"].append([channel, visit.position.x, visit.position.y,
                                                     self.client.state.virtual_time_s])
                self.tail_visits.pop(0)
                if expected_after != self._state_key(self.remaining_covers):
                    self._cancel_tail("actual_clear_state_or_cost_mismatch")
                elif not self.tail_visits:
                    self.tail_record["completed"] = True
                    self.tail_record["actual_cost_us"] = expected_time - self.tail_record["start_time_us"]
                    self.tail_record["dominance_status"] = "complete_execution_matches_compared_route"
                    self.tail_expected_key = expected_after
                else:
                    self.tail_expected_key = expected_after

    def run(self):
        report = super().run()
        for record in self.tail_log:
            if record["accepted"] and not record.get("completed") and not record.get("cancelled"):
                record["interrupted"] = report.completion_reason
                record["dominance_status"] = "incomplete_execution_no_full_tail_claim"
        return report


def run_certified_tail_state_search(client, *, problem=3, max_actions=10000,
                                    max_active_probes=6, config=None, enabled=True):
    if problem != 3:
        raise ValueError("certified deterministic tail is Q3 only")
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be in [0,30]")
    return CertifiedTailStateSearch(client, max_actions, max_active_probes, config, enabled).run()
