"""One additional location for the selected source's first active probe.

All scheduling, geometry updates, service costs, and termination are v1.
The extra choice minimizes the original finite proxy, not actual total time.
"""

import math
from copy import deepcopy

from planning.probe_candidates import geometry_candidates
from planning.radius_probe import choose_radius_probe
from simulator_client.state import Position

from .relocating_state_search import RelocatingStateSearch


class CurrentProbeStateSearch(RelocatingStateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, enabled=True):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        # The inherited v1 relocation remains enabled in both arms.
        super().__init__(client, max_actions, max_active_probes, config, True)
        self.current_probe_enabled = enabled
        self.current_probe_log = []
        self._pending_current_probe = None
        self.report.strategy_parameters.update(
            current_probe_enabled=enabled,
            current_probe_log=self.current_probe_log,
            current_probe_scope="selected known source, first active probe only; original finite score",
        )

    def _current_gate(self, channel, index):
        current = self.client.state.position
        result = {"current_position": [current.x, current.y], "eligible": False}
        if index != 0:
            return {**result, "reason": "not_first_probe"}
        if channel not in self.detected or channel in self.cleared:
            return {**result, "reason": "not_known_active_channel"}
        if channel in self.near_points:
            return {**result, "reason": "near_clear_precedes_probe"}
        region = self.regions.get(channel)
        if not region or len(region.vertices) < 3:
            return {**result, "reason": "empty_or_degenerate_region"}
        vertices = region.vertices
        if not all(math.isfinite(v) for point in [(current.x, current.y), *vertices] for v in point):
            return {**result, "reason": "nonfinite_geometry"}
        area2 = math.fsum(a[0] * b[1] - a[1] * b[0]
                         for a, b in zip(vertices, (*vertices[1:], vertices[0])))
        if not math.isfinite(area2) or area2 == 0.:
            return {**result, "reason": "empty_or_degenerate_region"}
        if region.enclosing_disk().radius <= 19.9:
            return {**result, "reason": "certified_clear_precedes_probe"}
        key = (round(current.x, 6), round(current.y, 6))
        fresh = key not in self.observed_positions.get(channel, set())
        maximum = max(current.distance_to(Position(*v)) for v in vertices)
        guaranteed = maximum <= 1000. - 1e-5
        result.update(fresh_for_channel=fresh, maximum_vertex_distance_m=maximum,
                      guaranteed_reception=guaranteed, reception_margin_m=1e-5)
        return {**result, "eligible": fresh and guaranteed,
                "reason": "eligible" if fresh and guaranteed else
                          "already_measured_here" if not fresh else "reception_not_certified"}

    def _next_probe(self, channel, index):
        if not self.current_probe_enabled:
            return super()._next_probe(channel, index)
        log_count = len(self.probe_log)
        baseline = super()._next_probe(channel, index)
        gate = self._current_gate(channel, index)
        record = {"channel": channel, "index": index,
                  "after_actual_action_count": len(self.report.action_history),
                  "expected_action_ordinal": len(self.report.action_history) + 1,
                  **gate, "changed_point": False, "new_current_selected": False,
                  "baseline_position": [baseline.x, baseline.y] if baseline else None,
                  "selected_position": [baseline.x, baseline.y] if baseline else None,
                  "execution_status": "planned"}
        self.current_probe_log.append(record)
        self._pending_current_probe = record
        if not gate["eligible"]:
            return baseline
        config = self.state_config
        if (not config.active_probe_search or config.probe_model != "radius_proxy"
                or len(self.probe_log) != log_count + 1
                or self.probe_log[-1].get("family") != self.refine_mode
                or self.probe_log[-1].get("old_best_position") is None
                or baseline is None):
            record.update(eligible=False, reason="unchanged_parent_fallback")
            return baseline
        original = self.probe_log[-1]
        record["baseline_probe_log"] = deepcopy(original)
        # Preserve the original nine-point anchor and all fourteen extras.
        extras, _ = geometry_candidates(self.regions[channel], mode=self.refine_mode,
                                       old_best=Position(*original["old_best_position"]))
        current = self.client.state.position
        selected, final = choose_radius_probe(
            self.regions[channel], current, self.first_bearings[channel],
            self.observed_positions.get(channel, set()), config.probe_uncertainty_weight,
            extra_points=[*extras, current])
        # The third pass was paid even if its result is rejected below.
        original["total_geometry_updates"] += final["geometry_updates"]
        original["total_runtime_s"] += final["runtime_s"]
        record.update(baseline_score_s=original["score_s"],
                      baseline_proposed_candidates=original["proposed_candidates"],
                      baseline_candidates=original["candidates"],
                      baseline_geometry_updates=original["geometry_updates"],
                      added_scoring_runtime_s=final["runtime_s"],
                      added_geometry_updates=final["geometry_updates"],
                      extra_point_was_duplicate=(final["proposed_candidates"] ==
                                                 original["proposed_candidates"]))
        if selected is None or not math.isfinite(final["score_s"]):
            record.update(eligible=False, reason="nonfinite_or_empty_extended_score")
            return baseline
        if (selected not in (baseline, current)
                or final["score_s"] > original["score_s"] + 1e-9):
            record.update(eligible=False, reason="extended_score_consistency_fallback")
            return baseline
        record.update(selected_position=[selected.x, selected.y],
                      selected_score_s=final["score_s"],
                      score_improvement_s=original["score_s"] - final["score_s"],
                      changed_point=selected != baseline,
                      changed_on_score_tie=(selected != baseline and
                                            final["score_s"] == original["score_s"]),
                      new_current_selected=selected == current and selected != baseline)
        original.update(final)
        original.update(current_probe_extension=True,
                        finite_family_before_extension="axis_quantile",
                        score_improvement_s=original["old_best_score_s"] - final["score_s"],
                        changed_point=[selected.x, selected.y] != original["old_best_position"])
        return selected

    def _perform(self, action, position, channel, phase):
        if not self.current_probe_enabled or self._pending_current_probe is None:
            return super()._perform(action, position, channel, phase)
        pending = self._pending_current_probe
        before = len(self.report.action_history)
        try:
            return super()._perform(action, position, channel, phase)
        finally:
            # Pure audit metadata; no client state or physical response changes.
            self._pending_current_probe = None
            if len(self.report.action_history) > before:
                actual = self.report.action_history[before]
                matches = (before + 1 == pending["expected_action_ordinal"]
                           and actual["action"] == "measure"
                           and actual["phase"] == "active_localization"
                           and actual["channel"] == pending["channel"]
                           and actual["position"] == pending["selected_position"])
                pending.update(execution_status="accepted" if matches else "different_next_action",
                               actual_action_ordinal=before + 1,
                               actual_position=actual["position"],
                               actual_virtual_time_s=actual["virtual_time_s"])
            else:
                pending["execution_status"] = "not_accepted_or_budget_stopped"


def run_current_probe_state_search(client, *, problem=3, max_actions=10000,
                                   max_active_probes=6, config=None, enabled=True):
    if problem != 3:
        raise ValueError("Current probe candidate supports only Q3")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be integer in [0,30]")
    return CurrentProbeStateSearch(client, max_actions, max_active_probes, config, enabled).run()
