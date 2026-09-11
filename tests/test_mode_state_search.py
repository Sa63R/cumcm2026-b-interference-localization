"""Integration safety, no hidden-state input, and disabled v1 identity."""
import pytest

from simulation import LocalResearchSimulator, random_scenario
from strategies.mode_state_search import run_mode_state_search
from strategies.relocating_state_search import run_relocating_state_search
from tests.test_inferred_silence import BASE
from tests.test_strategy import ObservationOnlyClient
from experiments.training_stress_reliability import audit_actions


def test_disabled_matches_v1_actions_exactly():
    case = random_scenario(3, 114001)
    old = run_relocating_state_search(ObservationOnlyClient(LocalResearchSimulator(case).client()), config=BASE)
    new = run_mode_state_search(ObservationOnlyClient(LocalResearchSimulator(case).client()), config=BASE, mode="off")
    assert old.action_history == new.action_history
    assert old.virtual_time_s == new.virtual_time_s


@pytest.mark.parametrize("mode", ["local", "joint"])
def test_modes_clear_with_real_feedback_and_physical_ledger(mode):
    sim = LocalResearchSimulator(random_scenario(3, 114001))
    result = run_mode_state_search(ObservationOnlyClient(sim.client()), config=BASE, mode=mode)
    evaluation = sim.evaluation()
    assert result.error is None and result.exit_error is None
    assert result.completion_certified_under_model and evaluation["all_cleared"]
    assert evaluation["failed_clear_count"] == 0
    assert audit_actions(sim.observation_history(), evaluation)["legal_actions_and_cost_ledger"]
    assert result.strategy_parameters["source_mode_log"]


@pytest.mark.parametrize("kwargs", [{"problem": 4}, {"mode": "unknown"}, {"max_active_probes": True}])
def test_reject_invalid_before_enter(kwargs):
    sim = LocalResearchSimulator(random_scenario(3, 114001))
    with pytest.raises(ValueError):
        run_mode_state_search(ObservationOnlyClient(sim.client()), **kwargs)
    assert sim.observation_history() == []
