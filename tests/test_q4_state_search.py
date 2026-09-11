"""Q4 migration: exact disabled baseline, boundary visibility and completion."""
import pytest

from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from strategies import run_search
from strategies.q4_state_search import run_q4_state_search
from tests.test_strategy import ObservationOnlyClient


def test_disabled_state_is_exact_triangular_action_identity():
    case = random_scenario(4, 114001)
    old = run_search(ObservationOnlyClient(LocalResearchSimulator(case).client()), problem=4, variant="triangular")
    new = run_q4_state_search(ObservationOnlyClient(LocalResearchSimulator(case).client()), mode="off")
    assert old.action_history == new.action_history
    assert old.virtual_time_s == new.virtual_time_s


@pytest.mark.parametrize("mode", ["state", "state_hull", "state_pair", "state_rescue", "state_pruned", "state_pruned_rescue"])
@pytest.mark.parametrize("case", [difficult_scenarios(4)[0], difficult_scenarios(4)[4], random_scenario(4, 114001)])
def test_directional_and_extreme_cases_complete_under_real_observations(mode, case):
    sim = LocalResearchSimulator(case)
    result = run_q4_state_search(ObservationOnlyClient(sim.client()), mode=mode)
    evaluation = sim.evaluation()
    assert result.error is None and result.exit_error is None
    assert evaluation["all_cleared"] and result.completion_certified_under_model
    assert evaluation["virtual_time_s"] == result.virtual_time_s
    assert sum(evaluation["time_breakdown_s"].values()) == pytest.approx(result.virtual_time_s)
    assert sim.observation_history()[-1]["action"] == "/exit"
    assert all(a["phase"] == "guaranteed_clearance" for a in result.action_history
               if a["action"] == "clear" and a["result"] != "success")
    assert result.coverage_complete or len(result.cleared_channels) == 16


@pytest.mark.parametrize("kwargs", [{"problem":3}, {"max_expansions": True}, {"mode":"bad"}, {"max_actions":1}])
def test_invalid_rejected_before_enter(kwargs):
    sim = LocalResearchSimulator(random_scenario(4, 114001))
    with pytest.raises(ValueError):
        run_q4_state_search(ObservationOnlyClient(sim.client()), **kwargs)
    assert not sim.observation_history()
