"""Nested finite active-action search using v1's unchanged surrogate score."""

from planning.probe_candidates import geometry_candidates
from planning.radius_probe import choose_radius_probe

from .pruned_state_search import PrunedStateSearch


class RefinedStateSearch(PrunedStateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, mode):
        super().__init__(client, max_actions, max_active_probes, config)
        self.refine_mode = mode
        self.report.strategy_parameters["probe_candidate_family"] = mode

    def _next_probe(self, channel, index):
        config, region = self.state_config, self.regions.get(channel)
        if (not config.active_probe_search or config.probe_model != "radius_proxy"
                or not region or not region.vertices):
            return super()._next_probe(channel, index)
        current = self.client.state.position
        observed = self.observed_positions.get(channel, set())
        old, before = choose_radius_probe(region, current, self.first_bearings[channel],
                                         observed, config.probe_uncertainty_weight)
        if old is None:
            return super()._next_probe(channel, index)
        extra, geometry = geometry_candidates(region, mode=self.refine_mode, old_best=old)
        point, after = choose_radius_probe(region, current, self.first_bearings[channel],
            observed, config.probe_uncertainty_weight, extra_points=extra)
        self.probe_log.append({"channel": channel, "index": index, **after, **geometry,
            "family": self.refine_mode, "old_best_position": [old.x, old.y],
            "old_best_score_s": before["score_s"], "score_improvement_s": before["score_s"] - after["score_s"],
            "changed_point": point != old, "baseline_candidates": before["candidates"],
            "baseline_geometry_updates": before["geometry_updates"],
            "total_geometry_updates": before["geometry_updates"] + after["geometry_updates"],
            "total_runtime_s": before["runtime_s"] + after["runtime_s"]})
        return point


def run_refined_state_search(client, *, problem=3, max_actions=10000,
                             max_active_probes=6, config=None, mode="axis_quantile"):
    if problem != 3:
        raise ValueError("refined state research supports only Q3")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 2:
        raise ValueError("max_actions must be an integer >=2")
    if (isinstance(max_active_probes, bool) or not isinstance(max_active_probes, int)
            or not 0 <= max_active_probes <= 30):
        raise ValueError("max_active_probes must be an integer in [0,30]")
    if mode not in ("axis_quantile", "local_refine"):
        raise ValueError("mode must be axis_quantile or local_refine")
    return RefinedStateSearch(client, max_actions, max_active_probes, config, mode).run()
