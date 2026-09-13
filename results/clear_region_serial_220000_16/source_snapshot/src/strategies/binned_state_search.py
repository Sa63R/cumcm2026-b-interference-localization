"""Diagnostic Q3 action search with coarse future observations, fine live data."""

from planning.binned_probe_tree import BinnedProbeTree

from .state_search import StateSearch


class BinnedStateSearch(StateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, bin_width_deg):
        super().__init__(client, max_actions, max_active_probes, config)
        self.bin_width_deg = bin_width_deg
        self.report.strategy_parameters["probe_observation_bin_deg"] = bin_width_deg

    def _next_probe(self, channel, index):
        config = self.state_config
        if not config.active_probe_search:
            return super()._next_probe(channel, index)
        region = self.regions[channel]
        tree = BinnedProbeTree(bin_width_deg=self.bin_width_deg,
            depth=config.probe_depth, supports=config.probe_supports,
            candidates=config.probe_candidates, inner_candidates=config.probe_inner_candidates,
            max_expansions=config.probe_max_expansions,
            first_bearing=self.first_bearings[channel],
            root_switch_s=float(self.client.state.current_channel != channel),
            terminal_mode=config.probe_terminal_mode)
        point, log = tree.choose(region, self.client.state.position,
                                 self.observed_positions.get(channel, set()))
        self.probe_log.append({"channel": channel, "index": index, **log})
        return point if point is not None else super()._next_probe(channel, index)


def run_binned_state_search(client, *, problem=3, max_actions=10000,
                            max_active_probes=6, config=None, bin_width_deg=0.5):
    if problem != 3:
        raise ValueError("binned probe research supports only Q3")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 2:
        raise ValueError("max_actions must be an integer >=2")
    if (isinstance(max_active_probes, bool) or not isinstance(max_active_probes, int)
            or not 0 <= max_active_probes <= 30):
        raise ValueError("max_active_probes must be an integer in [0,30]")
    if bin_width_deg not in (.5, 1.0):
        raise ValueError("bin_width_deg must be .5 or 1.0")
    return BinnedStateSearch(client, max_actions, max_active_probes, config, bin_width_deg).run()
