from collections import Counter
import math
from types import SimpleNamespace

import pytest

from planning.coverage import omni_coverage_points
from planning.coverage_relocation import (CoverageOracle, CoverageOracleBudget,
                                           feasible_ray_point, project_to_segment)
from planning.disk_cover import disk_cover_radius
from simulator_client.state import ClientState, Position
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from strategies.geometric_probe_cost import run_probe_cost_search
from strategies.geometric_relocation import GeometricRelocationSearch, run_relocation_search
from strategies.search import _StopSearch
from tests.test_strategy import ObservationOnlyClient


def policy_fixture():
    client = SimpleNamespace(state=ClientState(session='active', position=Position(1700.,300.), current_channel=1))
    policy = GeometricRelocationSearch(client,20000,6,None,120.,True)
    policy.discovery_stations = [policy.points[0]]
    policy.detected = {1}
    policy.near_points = {1:Position(1700.,-300.)}
    return policy


def test_full_disk_ray_geometry_and_positive_route_freedom():
    points = list(omni_coverage_points(1150))
    old = points.pop(1)
    choice = feasible_ray_point(points, old, (1700.,0.))
    assert choice.point == Position(1700.,0.)
    assert choice.coverage_radius_m == pytest.approx(disk_cover_radius(points+[old]),abs=1e-8)
    p,q = Position(1700.,300.),Position(1700.,-300.)
    assert project_to_segment(old,p,q) == choice.point
    assert (p.distance_to(old)+old.distance_to(q)-p.distance_to(q))/5. > 130
    clipped = feasible_ray_point(points,old,(0.,2000.))
    assert 0 < clipped.fraction < 1 and disk_cover_radius(points+[clipped.point]) <= 1000.-1e-5
    for fraction in (.25,.5,.75):
        between = Position(old.x+fraction*(clipped.point.x-old.x),old.y+fraction*(clipped.point.y-old.y))
        assert disk_cover_radius(points+[between]) <= 1000.-1e-5
    bad = [Position(1800.*math.cos(i*math.pi/3),1800.*math.sin(i*math.pi/3)) for i in range(6)]
    assert disk_cover_radius(bad) == pytest.approx(1800.,abs=1e-8) # Interior hole.
    with pytest.raises(ValueError,match='not certified'):
        feasible_ray_point(bad[1:],bad[0],(1600.,0.))


def test_original_ring_analytic_radius_and_deduplicated_oracle_budget():
    points = list(omni_coverage_points(1150))
    expected = max(1150/math.sqrt(3),math.sqrt(1800**2+1150**2-math.sqrt(3)*1800*1150))
    oracle = CoverageOracle(1)
    assert oracle(points) == pytest.approx(expected,abs=1e-8)
    assert oracle(list(reversed(points))+[points[0]]) == oracle(points)
    assert oracle.calls == 1
    with pytest.raises(CoverageOracleBudget):
        oracle(points[:-1])


def test_plan_changes_only_one_future_site_and_needs_real_action_before_another_change():
    policy = policy_fixture()
    remaining = list(policy.points[1:])
    before = remaining.copy()
    policy._next_task(remaining)
    assert len(policy.relocation_log) == 1
    record = policy.relocation_log[0]
    assert record['relocated']
    assert sum(a != b for a,b in zip(before,remaining)) == 1
    assert policy.report.coverage_points_visited == 0
    assert policy.discovery_stations == [policy.points[0]]
    assert not policy.report.action_history
    assert disk_cover_radius(policy.discovery_stations+remaining) <= 1000.-1e-5
    policy._next_task(remaining)
    assert len(policy.relocation_log) == 1


def test_exhausted_geometry_budget_returns_same_existing_task_without_false_scan_evidence():
    policy = policy_fixture()
    remaining = list(policy.points[1:])
    policy.coverage_oracle.maximum = 0
    expected = super(GeometricRelocationSearch,policy)._next_task(remaining)
    assert policy._next_task(remaining) == expected
    assert policy.relocation_log[-1]['oracle_budget_exhausted']
    assert not policy.relocation_log[-1]['relocated']
    assert remaining == list(policy.points[1:])
    assert policy.report.coverage_points_visited == 0


def test_partial_scan_is_never_promoted_to_executed_station():
    simulator = LocalResearchSimulator(random_scenario(3,116001))
    policy = GeometricRelocationSearch(ObservationOnlyClient(simulator.client()),3,6,None,120.,True)
    report = policy.run()
    assert report.completion_reason == 'action_budget'
    assert report.measurement_count == 1
    assert not report.completion_certified_under_model
    assert policy.discovery_stations == []


def test_sixteen_detections_do_not_end_search_but_sixteen_actual_clears_do():
    policy = policy_fixture()
    policy.detected = set(range(1,17))
    policy.cleared = set(range(1,16))
    policy.near_points = {16:Position(1700.,-300.)}
    remaining = list(policy.points[1:])
    assert policy._next_task(remaining) is not None
    assert not policy.report.completion_certified_under_model
    policy.client.clear = lambda point,channel:dict(accepted=True,clear_result='success')
    with pytest.raises(_StopSearch,match='source_count_upper_bound_reached'):
        policy._perform('clear',Position(1700.,-300.),16,'near_clear')
    assert len(policy.cleared) == 16 and policy.report.completion_certified_under_model


def audit_physical_cover(report):
    history = report.action_history
    for record in report.strategy_parameters['relocation_log']:
        previous = history[:record['after_actual_action_count']]
        known = {a['channel'] for a in previous if a['action']=='measure' and a['result'] in ('direction','near')}
        for p in record['executed_discovery_stations']:
            for channel in set(range(1,21))-known:
                assert any(a['action']=='measure' and a['result']=='no_signal' and a['channel']==channel
                           and a['position']==p for a in previous)
        assert record['selected_proxy_s'] <= record['baseline_proxy_s']+1e-6
        assert len(record['evaluated']) <= 4
        assert Counter(record['baseline_order']) == Counter(record['selected_order'])
        if record['relocated']:
            assert sum(a != b for a,b in zip(record['remaining_before'],record['remaining_after'])) == 1
            assert disk_cover_radius(record['executed_discovery_stations']+record['remaining_after']) <= 1000.-1e-5
    success = {a['channel'] for a in history if a['action']=='clear' and a['result']=='success'}
    planned_unvisited = [p for p in report.coverage_points if not any(
        a['action']=='measure' and a['phase']=='coverage' and a['position']==p for a in history)]
    assert not planned_unvisited or len(success) == 16
    if len(success) < 16:
        known = set(report.detected_channels)
        for channel in set(range(1,21))-known:
            actual = [a['position'] for a in history if a['action']=='measure'
                      and a['channel']==channel and a['result']=='no_signal']
            assert disk_cover_radius(actual) <= 1000.-1e-5
    for p in report.coverage_points:
        scans = [a for a in history if a['action']=='measure' and a['phase']=='coverage' and a['position']==p]
        if not scans:
            continue
        finished = max(a['virtual_time_s'] for a in scans)
        cleared = {a['channel'] for a in history if a['action']=='clear' and a['result']=='success'
                   and a['virtual_time_s'] < finished}
        assert cleared | {a['channel'] for a in scans} == set(range(1,21))


@pytest.mark.parametrize('scenario',[difficult_scenarios(3)[0],difficult_scenarios(3)[4],random_scenario(3,116001)])
def test_observation_only_full_clear_and_actual_per_channel_cover(scenario):
    simulator = LocalResearchSimulator(scenario)
    report = run_relocation_search(ObservationOnlyClient(simulator.client()))
    evaluation = simulator.evaluation()
    assert evaluation['all_cleared'] and report.completion_certified_under_model
    assert evaluation['failed_clear_count'] == 0
    assert report.virtual_time_s == evaluation['virtual_time_s']
    assert report.measurement_count == evaluation['measurement_count']
    assert sum(report.time_breakdown.values()) == pytest.approx(report.virtual_time_s,abs=1e-4)
    audit_physical_cover(report)


def test_disabled_is_exact_single_trace():
    scenario = random_scenario(3,116001)
    old = run_probe_cost_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()))
    new = run_relocation_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),enabled=False)
    assert old.action_history == new.action_history and old.virtual_time_s == new.virtual_time_s


@pytest.mark.parametrize('kwargs',[{'problem':4},{'enabled':1},{'max_active_probes':True},{'max_actions':1}])
def test_invalid_options_reject_before_enter(kwargs):
    simulator = LocalResearchSimulator(random_scenario(3,116001))
    with pytest.raises(ValueError):
        run_relocation_search(ObservationOnlyClient(simulator.client()),**kwargs)
    assert simulator.observation_history() == []
