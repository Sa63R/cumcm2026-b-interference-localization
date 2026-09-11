"""Composition of strict known-source range pruning and bounded macro service.

The scheduling constructor cooperatively calls the range constructor and then
the unchanged compact cover. Each component retains its separate prefix log.
"""
from .q4_r2_scheduling import Q4R2Scheduling, CONFIGS
from .q4_range_pruning import Q4RangePruningSearch


class Q4RangeScheduling(Q4R2Scheduling, Q4RangePruningSearch):
    pass


def run_q4_range_scheduling(client, *, problem=4, max_actions=20000,
                            max_active_probes=6, max_expansions=200, config="onroute"):
    if type(problem) is not int or problem != 4 or config not in CONFIGS:
        raise ValueError("Q4 only; unknown schedule")
    for value, minimum, maximum in ((max_actions, 2, 1_000_000),
                                    (max_active_probes, 0, 30), (max_expansions, 0, 10000)):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("Invalid execution budget")
    return Q4RangeScheduling(client, max_actions, max_active_probes,
                             max_expansions=max_expansions, config=config).run()
