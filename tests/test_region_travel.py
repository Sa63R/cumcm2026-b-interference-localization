"""Finite matrix bounds and observation-only safety are separate claims."""

from itertools import permutations,product
import json
from pathlib import Path
import random

import pytest

from localization.omni import OmniCandidateRegion
from planning.region_travel import (RegionMass,area_nodes,point_mass,radius_interval_width,
                                   region_mass,travel_matrix)
from planning.state_route import RouteTask,solve_state_route
from simulation import LocalResearchSimulator,difficult_scenarios,random_scenario
from simulator_client.state import Position
from strategies.region_state_search import run_region_state_search
from tests.test_strategy import ObservationOnlyClient


BASE=json.loads((Path(__file__).resolve().parents[1]/
                "experiments/state_search_candidate_axis_quantile_v1.json").read_text())["kwargs"]["config"]


def explicit_matrix_cost(tasks,order,matrix,initial):
    previous=None
    source_count=sum(t.is_source for t in tasks)
    result=0.
    for i in order:
        result+=initial[i] if previous is None else matrix[previous][i]
        result+=tasks[i].service_s
        if tasks[i].is_source:
            source_count-=1
        else:
            result+=6*source_count
        previous=i
    return result


@pytest.mark.parametrize("seed,n",[(1,3),(2,5),(3,7)])
def test_mst_entry_bound_matches_nonmetric_matrix_exhaustion(seed,n):
    rng=random.Random(seed)
    matrix=[[0.]*n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            matrix[i][j]=matrix[j][i]=float(rng.randint(1,100))
    # Force a triangle inequality violation: the MST proof does not need it.
    matrix[0][1]=matrix[1][0]=100.
    matrix[0][2]=matrix[2][0]=1.
    matrix[1][2]=matrix[2][1]=1.
    initial=[rng.uniform(0,50) for _ in range(n)]
    tasks=[RouteTask(Position(i,0),i%2==0,rng.uniform(0,10)) for i in range(n)]
    optimum=min(explicit_matrix_cost(tasks,p,matrix,initial) for p in permutations(range(n)))
    for budget in (0,1,10,100000):
        result=solve_state_route(tasks,travel_times_s=matrix,initial_times_s=initial,max_expansions=budget)
        assert result.lower_bound_s<=optimum+1e-8<=result.cost_s+1e-8
        assert explicit_matrix_cost(tasks,result.order,matrix,initial)==pytest.approx(result.cost_s)
        if budget==100000:
            assert result.exact and result.cost_s==pytest.approx(optimum)


def test_point_matrix_extension_preserves_original_solver_result():
    tasks=[RouteTask(Position(i*31,i*i*17),i%2==0,float(i)) for i in range(7)]
    start=Position(-200,50)
    matrix,initial,_=travel_matrix([point_mass(t.position) for t in tasks],start,"expected_distance")
    old=solve_state_route(tasks,start,max_expansions=100)
    new=solve_state_route(tasks,start,max_expansions=100,travel_times_s=matrix,initial_times_s=initial)
    assert old.order==new.order
    assert old.cost_s==new.cost_s and old.lower_bound_s==new.lower_bound_s
    assert old.expanded==new.expanded and old.exact==new.exact


@pytest.mark.parametrize("matrix,initial",[([[1.]],[0.]),([[0.,1.],[2.,0.]],[0.,0.]),
                                           ([[0.,-1.],[-1.,0.]],[0.,0.]),([[0.]],[float('nan')]),
                                           ([[0.,1.]],[0.]),([[0.]],None)])
def test_invalid_matrix_fails_explicitly(matrix,initial):
    with pytest.raises(ValueError):
        solve_state_route([RouteTask(Position(0,0),False) for _ in matrix],
                          travel_times_s=matrix,initial_times_s=initial)


def test_radius_interval_and_area_quadrature_do_not_invent_certificates():
    p=Position(1200,0)
    assert radius_interval_width(p,((0,0),(100,0)),((-100,0),(-200,0)))==100
    assert radius_interval_width(p,((0,0),),((-100,0),))==100
    region=OmniCandidateRegion().observe((0,0),0).observe_no_signal((-800,0))
    mass=region_mass(region,Position(*region.enclosing_disk().center))
    assert mass.status=="finite_area_radius_proxy"
    assert 0<len(mass.points)<=12 and sum(mass.weights)==pytest.approx(1)
    assert all(region.contains((p.x,p.y)) for p in mass.points)
    assert all(radius_interval_width(p,((0,0),),((-800,0),))>0 for p in mass.points)
    vertices=region.vertices
    region.vertices=((2000.,0.),(2001.,0.),(2000.,1.))
    fallback=region_mass(region,Position(750,0))
    assert fallback.status=="quadrature_empty_point_fallback" and fallback.mean==Position(750,0)
    assert region.vertices==((2000.,0.),(2001.,0.),(2000.,1.))  # No live geometry mutation.
    assert area_nodes(((1.,1.),(2.,2.)))==()


def test_expected_edges_equal_finite_world_expectation_and_exceed_mean_edges():
    masses=[RegionMass((Position(i*70,-50),Position(i*70+30,90)),(.3,.7),"test") for i in range(3)]
    start=Position(-10,20)
    matrix,initial,variance=travel_matrix(masses,start,"expected_distance")
    means,mean_initial,_=travel_matrix(masses,start,"mean_point")
    assert all(matrix[i][j]>=means[i][j]-1e-9 for i in range(3) for j in range(3))
    assert all(a>=b-1e-9 for a,b in zip(initial,mean_initial))
    assert variance["edge_variance_mean_s2"]>0
    expected=0.
    for choices in product(range(2),repeat=3):
        points=[m.points[k] for m,k in zip(masses,choices)]
        weight=1.
        for m,k in zip(masses,choices):
            weight*=m.weights[k]
        expected+=weight*(start.distance_to(points[0])+points[0].distance_to(points[1])+points[1].distance_to(points[2]))/5
    assert expected==pytest.approx(initial[0]+matrix[0][1]+matrix[1][2])


@pytest.mark.parametrize("mode",["mean_point","expected_distance"])
@pytest.mark.parametrize("scenario",difficult_scenarios(3)+[random_scenario(3,104001)])
def test_changed_scheduler_keeps_legal_axis_measurement_and_clearance(scenario,mode):
    sim=LocalResearchSimulator(scenario)
    report=run_region_state_search(ObservationOnlyClient(sim.client()),config=BASE,mode=mode)
    evaluation=sim.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"]==0
    assert report.strategy_parameters["probe_candidate_family"]=="axis_quantile"
    assert all(p["bound_scope"]=="frozen region travel matrix, not original Q3"
               for p in report.strategy_parameters["planning_log"])


@pytest.mark.parametrize("kwargs",[{"problem":4},{"mode":"variance_penalty"},{"max_actions":True}])
def test_configuration_rejected_before_any_action(kwargs):
    with pytest.raises(ValueError):
        run_region_state_search(None,**kwargs)
