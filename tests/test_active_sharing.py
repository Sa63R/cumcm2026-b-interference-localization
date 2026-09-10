"""Active sharing preserves axis control and uses only certified receptions."""

import json
import math
from pathlib import Path

import pytest

from localization.omni import OmniCandidateRegion
from planning.shared_observation import shared_observation_value
from simulation import LocalResearchSimulator,Scenario,Source,difficult_scenarios,random_scenario
from simulator_client.state import Position
from strategies.refined_state_search import run_refined_state_search
from strategies.sharing_state_search import run_sharing_state_search
from tests.test_strategy import ObservationOnlyClient


BASE=json.loads((Path(__file__).resolve().parents[1]/
                "experiments/state_search_candidate_axis_quantile_v1.json").read_text())["kwargs"]["config"]


def test_local_value_never_installs_hypothetical_feedback():
    region=OmniCandidateRegion().observe((0,0),0)
    before=(region.vertices,list(region.observations),list(region.no_signal_positions))
    value=shared_observation_value(region,(750,500))
    assert value and value["guaranteed_reception"]
    assert value["travel_saving_proxy_m"]>=0
    assert (region.vertices,region.observations,region.no_signal_positions)==before
    assert shared_observation_value(region,(-1000,0)) is None


@pytest.mark.parametrize("scenario",difficult_scenarios(3)+[random_scenario(3,106001)])
def test_actual_sharing_reception_and_full_time_are_independently_audited(scenario):
    sim=LocalResearchSimulator(scenario)
    report=run_sharing_state_search(ObservationOnlyClient(sim.client()),config=BASE)
    evaluation=sim.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"]==0
    assert report.virtual_time_s==evaluation["virtual_time_s"]
    assert sum(report.time_breakdown.values())==pytest.approx(report.virtual_time_s,abs=1e-4)
    regions,observed,cleared={},set(),set()
    last_primary=None
    shares=0
    for action in report.action_history:
        channel=action["channel"]
        p=Position(*action["position"])
        pair=(tuple(action["position"]),channel)
        if action["phase"]=="active_localization":
            last_primary=action
        if action["phase"]=="axis_active_shared_observation":
            assert channel not in cleared and pair not in observed
            assert channel!=last_primary["channel"] and action["position"]==last_primary["position"]
            assert channel in regions and regions[channel].vertices
            assert all(p.distance_to(Position(*v))<=1000-1e-6+1e-9 for v in regions[channel].vertices)
            assert action["result"] in ("direction","near")
            shares+=1
        if action["action"]=="measure":
            observed.add(pair)
            region=regions.setdefault(channel,OmniCandidateRegion())
            if action["result"]=="direction":
                region.observe(p,action["bearing_deg"])
            elif action["result"]=="no_signal":
                region.observe_no_signal(p)
        else:
            if action["phase"]=="certified_clear":
                assert all(p.distance_to(Position(*v))<=19.9+1e-6 for v in regions[channel].vertices)
            if action["result"]=="success":
                cleared.add(channel)
    stats=report.strategy_parameters["active_sharing"]
    assert shares==stats["shared_measurements"]
    assert stats["actual_measure_cost_s"]==pytest.approx(sum(d["actual_measure_cost_s"] for d in stats["decisions"]))
    for decision in stats["decisions"]:
        assert decision["estimated_net_gain_s"]>=5
        assert decision["actual_measure_cost_s"]==pytest.approx(6.)
        assert decision["conservative_measure_switch_return_s"]==7
        assert decision["reserved_return_switch_s"]==1


@pytest.mark.parametrize("seed",[106001,106002])
def test_disabled_sharing_preserves_axis_actions(seed):
    case=random_scenario(3,seed)
    sims=[LocalResearchSimulator(case),LocalResearchSimulator(case)]
    baseline=run_refined_state_search(ObservationOnlyClient(sims[0].client()),config=BASE)
    disabled=run_sharing_state_search(ObservationOnlyClient(sims[1].client()),config=BASE,sharing_config={"enabled":False})
    assert baseline.action_history==disabled.action_history
    assert baseline.virtual_time_s==disabled.virtual_time_s


def test_sixteenth_clear_and_action_budget_still_exit():
    sources=tuple(Source(c,math.cos(c),math.sin(c),1000) for c in range(1,17))
    sim=LocalResearchSimulator(Scenario("axis-share-near16",3,0,sources,"zero"))
    report=run_sharing_state_search(ObservationOnlyClient(sim.client()),config=BASE)
    assert report.cleared_count==16 and sim.observation_history()[-2]["action"]=="/clear"
    assert report.virtual_time_s==199
    sim=LocalResearchSimulator(random_scenario(3,106001))
    report=run_sharing_state_search(ObservationOnlyClient(sim.client()),config=BASE,max_actions=25)
    assert report.accepted_actions==25 and not report.completion_certified_under_model
    assert sim.observation_history()[-1]["action"]=="/exit"


@pytest.mark.parametrize("kwargs",[{"problem":4},{"sharing_config":{"enabled":1}},
                                    {"sharing_config":{"minimum_net_gain_s":float('nan')}},
                                    {"sharing_config":{"max_shared_per_stop":21}},{"max_actions":True}])
def test_invalid_options_fail_before_action(kwargs):
    with pytest.raises(ValueError):
        run_sharing_state_search(None,**kwargs)
