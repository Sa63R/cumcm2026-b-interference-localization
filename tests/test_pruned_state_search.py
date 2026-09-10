"""Exact v1 action histories are stronger evidence than just similar means."""

import json
from pathlib import Path

import pytest

from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from strategies.state_search import run_state_search
from strategies.pruned_state_search import run_pruned_state_search
from tests.test_strategy import ObservationOnlyClient


BASE = json.loads((Path(__file__).resolve().parents[1] /
                   "experiments/state_search_candidate_v1.json").read_text())["kwargs"]["config"]


@pytest.mark.parametrize("scenario", difficult_scenarios(3) +
                         [random_scenario(3, seed) for seed in (100225, 100226)])
def test_pruning_preserves_every_legal_action_and_virtual_cost(scenario):
    original = LocalResearchSimulator(scenario)
    pruned = LocalResearchSimulator(scenario)
    a = run_state_search(ObservationOnlyClient(original.client()), config=BASE)
    b = run_pruned_state_search(ObservationOnlyClient(pruned.client()), config=BASE)
    assert a.action_history == b.action_history
    assert a.virtual_time_s == b.virtual_time_s
    assert b.completion_certified_under_model
    assert pruned.evaluation()["failed_clear_count"] == 0
    before = a.strategy_parameters["probe_search_log"]
    after = b.strategy_parameters["probe_search_log"]
    assert len(before) == len(after)
    for x, y in zip(before, after):
        assert x["position"] == y["position"]
        assert x["score_s"] == y["score_s"]
        assert y["geometry_updates"] <= y["candidates"] * y["hypotheses"]
