"""Logical silence constraints remain distinct from actual measurement records."""

import json
import math
from pathlib import Path

import pytest

from localization.omni import OmniCandidateRegion
from planning.silence_certificate import certify_silence,polygon_distance
from simulation import LocalResearchSimulator,difficult_scenarios,random_scenario
from simulator_client.state import Position
from strategies.inferred_state_search import run_inferred_state_search
from strategies.refined_state_search import run_refined_state_search
from tests.test_strategy import ObservationOnlyClient


BASE=json.loads((Path(__file__).resolve().parents[1]/
                "experiments/state_search_candidate_axis_quantile_v1.json").read_text())["kwargs"]["config"]


def test_full_polygon_distance_and_strict_radius_boundary():
    region=OmniCandidateRegion().observe((0,0),0)
    region.vertices=((0.,0.),(100.,0.),(100.,100.),(0.,100.))
    region._circle=None
    assert polygon_distance(region,(50,50))==0
    assert polygon_distance(region,(50,-1500))==1500
    assert certify_silence(region,(50,-1500)) is None
    certificate=certify_silence(region,(50,-1501))
    assert certificate and certificate["method"]=="polygon_edges"
    assert certificate["distance_lower_m"]==pytest.approx(1501)
    assert polygon_distance(region,(200,200))==pytest.approx(math.sqrt(20000))
    region.vertices=((0.,0.),(0.,100.));region._circle=None
    assert polygon_distance(region,(1600,50))==1600
    assert certify_silence(region,(1600,50))
    region.vertices=();region._circle=None
    assert certify_silence(region,(2000,0)) is None
    # No recorded positive-bearing source is not enough to create a known-source fact.
    assert certify_silence(OmniCandidateRegion(),(5000,0)) is None


@pytest.mark.parametrize("scenario",difficult_scenarios(3)+[random_scenario(3,109001)])
def test_inferences_are_legal_separate_and_preserve_actual_completion(scenario):
    sim=LocalResearchSimulator(scenario)
    report=run_inferred_state_search(ObservationOnlyClient(sim.client()),config=BASE)
    evaluation=sim.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"]==0
    assert report.virtual_time_s==evaluation["virtual_time_s"]
    assert evaluation["measurement_count"]==sum(a["action"]=="measure" for a in report.action_history)
    regions,known,actual_readings={},{},{}
    known=set()
    inferences=report.strategy_parameters["inferred_no_signal_constraints"]
    cursor=0
    for index,action in enumerate(report.action_history):
        while cursor<len(inferences) and inferences[cursor]["after_actual_action_count"]==index:
            inferred=inferences[cursor]
            c=inferred["channel"]
            assert c in known and not inferred["physical_measurement"]
            assert certify_silence(regions[c],inferred["position"])
            regions[c].observe_no_signal(inferred["position"])
            cursor+=1
        c=action["channel"]
        p=Position(*action["position"])
        if action["action"]=="measure":
            region=regions.setdefault(c,OmniCandidateRegion())
            actual_readings.setdefault(c,{})[tuple(action["position"])]=action["result"]
            if action["result"]=="direction":
                region.observe(p,action["bearing_deg"]);known.add(c)
            elif action["result"]=="near":
                known.add(c)
            elif action["result"]=="no_signal":
                region.observe_no_signal(p)
        elif action["phase"]=="certified_clear":
            assert all(p.distance_to(Position(*v))<=19.9+1e-6 for v in regions[c].vertices)
    assert cursor==len(inferences)
    if len(known)<16:
        for c in set(range(1,21))-known:
            assert all(actual_readings[c].get(tuple(p))=="no_signal" for p in report.coverage_points)
    for inferred in inferences:
        assert not any(a["phase"]=="coverage" and a["channel"]==inferred["channel"] and a["position"]==inferred["position"]
                       for a in report.action_history)


def test_disabled_inference_is_identical_to_axis():
    scenario=random_scenario(3,109001)
    old=run_refined_state_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),config=BASE)
    disabled=run_inferred_state_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),config=BASE,enabled=False)
    assert old.action_history==disabled.action_history
    assert old.virtual_time_s==disabled.virtual_time_s


@pytest.mark.parametrize("kwargs",[{"problem":4},{"enabled":1},{"max_actions":True}])
def test_invalid_options_do_not_start_a_session(kwargs):
    with pytest.raises(ValueError):
        run_inferred_state_search(None,**kwargs)
