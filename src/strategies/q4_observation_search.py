"""Experimental finite feedback tree on the unchanged compact-combo skeleton."""
from planning.q4_observation_tree import ObservationTree, unique_prefix
from simulator_client.state import Position
from .q4_range_scheduling import Q4RangeScheduling


class Q4ObservationSearch(Q4RangeScheduling):
    def __init__(self, client, max_actions, max_active_probes, *, depth=1, max_expansions=200):
        self.observation_tree = ObservationTree(depth=depth)
        super().__init__(client, max_actions, max_active_probes,
                         max_expansions=max_expansions, config='onroute')
        self.observation_log = []
        self.report.strategy_parameters.update(
            active_probe_algorithm='finite-observation-tree-v1', observation_depth=depth,
            observation_tree_log=self.observation_log,
            soft_model_scope='Assumed finite prior only; never changes actual C or certifies clearance',
            optical_leaf='Common complete geometric route; conservative proxy, not clairvoyant cost')

    def _next_probe(self, channel, index):
        baseline = super()._next_probe(channel, index)
        if baseline is None:
            return None
        current = self.client.state.position
        selected, log = self.observation_tree.choose(self.regions[channel],
            (current.x, current.y), (baseline.x, baseline.y),
            unique_prefix(self.report.action_history, channel),
            first_bearing=self.first_bearings[channel],
            remaining_probes=self.max_active_probes-index,
            initial_switch_s=float(self.client.state.current_channel != channel))
        log.update(channel=channel, index=index,
                   after_actual_action_count=len(self.report.action_history))
        self.observation_log.append(log)
        return Position.coerce(selected)


def run_q4_observation_search(client, *, problem=4, max_actions=20000,
                              max_active_probes=6, max_expansions=200, depth=1):
    if type(problem) is not int or problem != 4:
        raise ValueError('Q4 only')
    for value, minimum, maximum in ((max_actions, 2, 1_000_000),
        (max_active_probes, 0, 30), (max_expansions, 0, 10000)):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError('Invalid execution budget')
    return Q4ObservationSearch(client, max_actions, max_active_probes,
        depth=depth, max_expansions=max_expansions).run()
