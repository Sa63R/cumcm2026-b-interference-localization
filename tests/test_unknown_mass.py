"""Unknown mass is exact finite-prior algebra, not a geometric certificate."""

import itertools
import json
import math
from pathlib import Path

import pytest

from planning.state_route import RouteTask
from planning.unknown_mass import (build_unknown_belief, existence_posterior,
                                   insertion_recourse, negative_radius_caps,
                                   spatial_nodes)
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from simulator_client.state import Position
from strategies.recourse_state_search import run_recourse_state_search
from tests.test_strategy import ObservationOnlyClient


BASE=json.loads((Path(__file__).resolve().parents[1]/
                "experiments/state_search_candidate_axis_quantile_v1.json").read_text())["kwargs"]["config"]


def test_unobserved_channels_reproduce_uniform_count_prior():
    presence, counts=existence_posterior([1.]*20,0)
    assert presence == pytest.approx([.65]*20)
    assert list(counts.values()) == pytest.approx([1/7]*7)


def test_compressed_count_algebra_matches_labeled_subset_enumeration():
    evidence=[.0,.1,.25,.45,.75,1.]
    presence,counts=existence_posterior(evidence,14)
    weights=[]
    for mask in itertools.product((0,1),repeat=6):
        n=14+sum(mask)
        if n>16:
            continue
        weight=math.prod(e for e,b in zip(evidence,mask) if b)/math.comb(20,n)
        weights.append((mask,n,weight))
    z=sum(w for _,_,w in weights)
    assert presence == pytest.approx([sum(w*m[c] for m,_,w in weights)/z for c in range(6)])
    assert counts == pytest.approx({n:sum(w for _,k,w in weights if k==n)/z for n in range(14,17)})
    assert existence_posterior([0.]*20,0) is None
    assert sum(existence_posterior([0.]*4,16)[0]) == 0


def test_shared_radius_constraints_intersect_instead_of_independent_likelihoods():
    a,b=(0.,0.),(500.,0.)
    ca,la=negative_radius_caps((a,))
    cb,lb=negative_radius_caps((b,))
    cab,lab=negative_radius_caps((a,b))
    assert cab == pytest.approx([min(x,y) for x,y in zip(ca,cb)])
    assert negative_radius_caps((a,a)) == negative_radius_caps((a,))
    assert lab != pytest.approx(sum((x-1000)*(y-1000)/250000 for x,y in zip(ca,cb))/96)
    assert lab <= min(la,lb)
    assert len(spatial_nodes()) == 96


def test_unknown_groups_use_actual_per_channel_negative_history():
    def negative(c,p):
        return {"channel":c,"position":p,"action":"measure","result":"no_signal"}
    history=[negative(c,[0,0]) for c in range(1,21)]
    belief=build_unknown_belief(history,{1,2,3})
    assert len(belief.groups)==1
    changed=build_unknown_belief(history+[negative(4,[1150,0])],{1,2,3})
    assert len(changed.groups)==2
    assert changed.presence[4]<changed.presence[5]
    # Clears stay in known-count evidence rather than becoming unknown again.
    assert set(changed.presence)==set(range(4,21))


def test_certified_full_ring_accounts_for_all_finite_mass():
    history=[{"channel":c,"position":[0,0],"action":"measure","result":"no_signal"}
             for c in range(1,21)]
    belief=build_unknown_belief(history,set())
    route=[RouteTask(Position(1150*math.cos(i*math.pi/3),1150*math.sin(i*math.pi/3)),False)
           for i in range(6)]
    credited=insertion_recourse(route,belief)
    plain=insertion_recourse(route,belief,credit_saved_scans=False)
    assert credited["accounted_unknown_mass"] == pytest.approx(belief.expected_unknown)
    assert credited["missed_unknown_mass"] < 1e-10
    assert credited["correction_s"] <= plain["correction_s"]
    partial=insertion_recourse(route[:1],belief)
    assert partial["missed_unknown_mass"] > 0


@pytest.mark.parametrize("credit",[True,False])
@pytest.mark.parametrize("scenario",difficult_scenarios(3)+[random_scenario(3,102001)])
def test_recourse_observation_only_execution_remains_safe(scenario,credit):
    sim=LocalResearchSimulator(scenario)
    report=run_recourse_state_search(ObservationOnlyClient(sim.client()),config=BASE,
                                    credit_saved_scans=credit)
    evaluation=sim.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"]==0
    for record in report.strategy_parameters["recourse_log"]:
        if record["status"]=="selected":
            assert 0<=record["expected_unknown"]<=16-record["known_count"]+1e-8
            assert all(a["missed_unknown_mass"]<1e-7 for a in record["alternatives"])


@pytest.mark.parametrize("kwargs",[{"problem":4},{"config":[]},{"config":{"scan_source_s":6}},
                                    {"recourse_candidates":True},{"credit_saved_scans":1}])
def test_invalid_recourse_configuration_fails_before_action(kwargs):
    with pytest.raises(ValueError):
        run_recourse_state_search(None,**kwargs)
