from types import SimpleNamespace
import pytest
from simulator_client.state import Position
from strategies.q4_range_scheduling import Q4RangeScheduling, run_q4_range_scheduling
from strategies.q4_r2_scheduling import Q4R2Scheduling
from strategies.q4_range_pruning import Q4RangePruningSearch


def test_cooperative_composition_initializes_both_guarantees():
    client = SimpleNamespace(state=SimpleNamespace(sources={}, position=Position(0, 0)))
    policy = Q4RangeScheduling(client, 20000, 6, max_expansions=200, config="onroute")
    assert policy.range_pruning_enabled
    assert policy.scheduling_config["early_services"] == 4
    assert len(policy.points) == 22
    assert policy.report.strategy_parameters["range_skipped_scans"] is policy.range_skips
    assert policy.report.strategy_parameters["early_service_log"] is policy.early_service_log
    assert type(policy)._scan is Q4RangePruningSearch._scan
    assert type(policy)._execute_plan is Q4R2Scheduling._execute_plan


@pytest.mark.parametrize("kwargs", [{"problem":3}, {"max_active_probes":True}, {"config":"bad"}])
def test_bad_inputs_rejected_without_client(kwargs):
    with pytest.raises(ValueError):
        run_q4_range_scheduling(None, **kwargs)
