"""Per-channel discovery completion must not be confused with actual removal."""

import json
import math
from pathlib import Path

import pytest

from simulation import LocalResearchSimulator,Scenario,Source,difficult_scenarios,random_scenario
from strategies.discovery_state_search import run_discovery_state_search
from strategies.refined_state_search import run_refined_state_search
from tests.test_strategy import ObservationOnlyClient


BASE=json.loads((Path(__file__).resolve().parents[1]/
                "experiments/state_search_candidate_axis_quantile_v1.json").read_text())["kwargs"]["config"]


@pytest.mark.parametrize("scenario",difficult_scenarios(3)+[random_scenario(3,108001)])
def test_unknown_only_has_actual_channel_coverage_and_actual_clearance(scenario):
    sim=LocalResearchSimulator(scenario)
    report=run_discovery_state_search(ObservationOnlyClient(sim.client()),config=BASE)
    evaluation=sim.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"]==0
    known=set()
    measured={tuple(p):{} for p in report.coverage_points}
    for action in report.action_history:
        c=action["channel"]
        if action["phase"]=="coverage":
            assert c not in known
            readings=measured[tuple(action["position"])]
            assert c not in readings
            readings[c]=action["result"]
        if action["action"]=="measure" and action["result"] in ("direction","near"):
            known.add(c)
    assert len(known)==report.cleared_count
    if len(known)<16:
        for c in set(range(1,21))-known:
            assert all(site.get(c)=="no_signal" for site in measured.values())
    masks=report.strategy_parameters["station_measured_masks"]
    assert masks==[sum(1<<(c-1) for c in readings) for readings in measured.values()]
    assert report.strategy_parameters["probe_candidate_family"]=="axis_quantile"
    assert "active_sharing" not in report.strategy_parameters


@pytest.mark.parametrize("seed",[108001,108002])
def test_disabled_unknown_only_is_identical_to_axis(seed):
    scenario=random_scenario(3,seed)
    original=run_refined_state_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),config=BASE)
    disabled=run_discovery_state_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),config=BASE,unknown_only=False)
    assert original.action_history==disabled.action_history
    assert original.virtual_time_s==disabled.virtual_time_s


def test_detecting_sixteen_cancels_scan_obligations_but_does_not_exit_before_clears():
    sources=tuple(Source(c,math.cos(c),math.sin(c),1000) for c in range(1,17))
    sim=LocalResearchSimulator(Scenario("discovery-sixteen-near",3,0,sources,"zero"))
    report=run_discovery_state_search(ObservationOnlyClient(sim.client()),config=BASE)
    history=report.action_history
    assert sum(a["action"]=="measure" for a in history)==16
    assert sum(a["action"]=="clear" and a["result"]=="success" for a in history)==16
    assert history[15]["action"]=="measure" and history[16]["action"]=="clear"
    assert report.virtual_time_s==175
    assert report.cleared_count==16 and report.completion_certified_under_model
    assert report.strategy_parameters["discovery_scan_ledger"]["stations_removed_without_pending"]


def test_budget_exit_does_not_use_incomplete_masks_as_a_certificate():
    sim=LocalResearchSimulator(random_scenario(3,108001))
    report=run_discovery_state_search(ObservationOnlyClient(sim.client()),config=BASE,max_actions=10)
    assert report.accepted_actions==10
    assert not report.completion_certified_under_model
    assert sim.observation_history()[-1]["action"]=="/exit"


@pytest.mark.parametrize("kwargs",[{"problem":4},{"unknown_only":1},{"max_actions":True},
                                    {"config":{"replace_coverage":True}}])
def test_invalid_configuration_rejected_without_actions(kwargs):
    sim=LocalResearchSimulator(random_scenario(3,108001))
    with pytest.raises(ValueError):
        run_discovery_state_search(ObservationOnlyClient(sim.client()),**kwargs)
    assert not sim.observation_history()
