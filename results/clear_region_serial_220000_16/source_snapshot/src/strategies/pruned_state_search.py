"""The original v1 radius policy, with exact score-bound evaluation."""

from planning.radius_probe import choose_radius_probe

from .state_search import StateSearch


class PrunedStateSearch(StateSearch):
    def _next_probe(self, channel, index):
        config = self.state_config
        region = self.regions.get(channel)
        if (not config.active_probe_search or config.probe_model != "radius_proxy"
                or not region or not region.vertices):
            return super()._next_probe(channel, index)
        point, log = choose_radius_probe(region, self.client.state.position,
            self.first_bearings[channel], self.observed_positions.get(channel, set()),
            config.probe_uncertainty_weight)
        self.probe_log.append({"channel": channel, "index": index, **log})
        return point if point is not None else super()._next_probe(channel, index)


def run_pruned_state_search(client, *, problem=3, max_actions=10000,
                            max_active_probes=6, config=None):
    if problem != 3:
        raise ValueError("pruned state research supports only Q3")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 2:
        raise ValueError("max_actions must be an integer >=2")
    if (isinstance(max_active_probes, bool) or not isinstance(max_active_probes, int)
            or not 0 <= max_active_probes <= 30):
        raise ValueError("max_active_probes must be an integer in [0,30]")
    return PrunedStateSearch(client, max_actions, max_active_probes, config).run()
