"""Thin combination: mean-point task scheduling and certified silence inference."""

from .inferred_state_search import InferredSilenceSearch
from .region_state_search import RegionStateSearch


class InferredMeanPointSearch(InferredSilenceSearch):
    def __init__(self, client, max_actions, max_active_probes, config, enabled):
        super().__init__(client, max_actions, max_active_probes, config, enabled)
        self.region_mode = "mean_point"
        self.region_log = []
        self.report.strategy_parameters.update({
            "region_travel_mode": self.region_mode,
            "region_travel_log": self.region_log,
            "model": "frozen_finite_region_travel_proxy",
            "search_bound_scope": "frozen mean travel matrix only; not original Q3",
            "region_prior": "12 stratified area nodes in conservative bearing polygon; one shared R uniform 1000..1500, conditioned on actual and explicitly inferred reception signs",
            "combination": "mean-point scheduling plus certified known-channel silence; inherited axis probes and physical coverage ledger",
        })

    # The frozen scheduler accesses only shared state and has no super() call.
    # Reusing the method avoids a second implementation of its matrix/search.
    # _scan, _next_probe, real measurements and safety checks remain inherited.
    _next_task = RegionStateSearch._next_task


def run_inferred_mean_point_state_search(client, *, problem=3, max_actions=10000,
                                         max_active_probes=6, config=None, enabled=True):
    if problem != 3:
        raise ValueError("inferred mean-point search supports Q3 only")
    if type(enabled) is not bool:
        raise ValueError("enabled must be boolean")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be in [0,30]")
    return InferredMeanPointSearch(client, max_actions, max_active_probes, config, enabled).run()
