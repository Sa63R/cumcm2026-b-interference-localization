"""Single-site coverage geometry, real discovery evidence and isolated behavior."""

import math

import pytest

from planning.coverage import omni_coverage_points
from planning.coverage_relocation import CoverageOracle, CoverageOracleBudget, feasible_ray_point, project_to_segment
from planning.disk_cover import disk_cover_radius
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from simulator_client.state import Position
from strategies.inferred_state_search import run_inferred_state_search
from strategies.relocating_state_search import run_relocating_state_search
from tests.test_inferred_silence import BASE
from tests.test_strategy import ObservationOnlyClient


def test_one_station_has_hundreds_of_metres_of_feasible_freedom():
    points = list(omni_coverage_points(1150))
    old = points.pop(1)
    choice = feasible_ray_point(points, old, (1700, 0))
    assert choice.point == Position(1700, 0) and choice.fraction == 1
    assert choice.coverage_radius_m == pytest.approx(disk_cover_radius(points+[old]), abs=1e-8)
    p, q = Position(1700, 300), Position(1700, -300)
    assert project_to_segment(old, p, q) == choice.point
    saved = (p.distance_to(old)+old.distance_to(q)-p.distance_to(choice.point)-choice.point.distance_to(q))/5
    assert saved > 130


def test_ray_clips_to_full_disk_cover_and_checks_internal_holes():
    points = list(omni_coverage_points(1150))
    old = points.pop(1)
    choice = feasible_ray_point(points, old, (0, 2000))
    assert 0 <= choice.fraction < 1
    assert disk_cover_radius(points+[choice.point]) <= 1000-1e-5
    for fraction in (.25, .5, .75):
        p = Position(old.x+fraction*(choice.point.x-old.x), old.y+fraction*(choice.point.y-old.y))
        assert disk_cover_radius(points+[p]) <= 1000-1e-5
    bad = [Position(1800*math.cos(i*math.pi/3), 1800*math.sin(i*math.pi/3)) for i in range(6)]
    with pytest.raises(ValueError, match="not certified"):
        feasible_ray_point(bad[1:], bad[0], (1600, 0))


def test_oracle_cache_and_budget_are_deterministic():
    oracle = CoverageOracle(maximum=1)
    points = list(omni_coverage_points(1150))
    assert oracle(points) == oracle(list(reversed(points)))
    assert oracle.calls == 1
    with pytest.raises(CoverageOracleBudget):
        oracle(points[:-1])


@pytest.mark.parametrize("scenario", difficult_scenarios(3)+[random_scenario(3,114001)])
def test_relocated_future_points_do_not_become_fake_discovery_evidence(scenario):
    simulator = LocalResearchSimulator(scenario)
    report = run_relocating_state_search(ObservationOnlyClient(simulator.client()), config=BASE)
    evaluation = simulator.evaluation()
    assert evaluation["all_cleared"] and report.completion_certified_under_model
    assert evaluation["failed_clear_count"] == 0
    assert report.virtual_time_s == evaluation["virtual_time_s"]
    assert report.measurement_count == evaluation["measurement_count"]
    history = report.action_history
    for item in report.strategy_parameters["relocation_log"]:
        previous = history[:item["after_actual_action_count"]]
        known = {a["channel"] for a in previous if a["action"] == "measure" and a["result"] in ("direction", "near")}
        unknown = set(range(1,21))-known
        for p in item["executed_discovery_stations"]:
            for c in unknown:
                assert any(a["action"] == "measure" and a["channel"] == c and a["position"] == p
                           and a["result"] == "no_signal" for a in previous)
        assert item["selected_proxy_s"] <= item["baseline_proxy_s"]+1e-6
        assert len(item["evaluated"]) <= 4
        if item["relocated"]:
            full = item["executed_discovery_stations"]+item["remaining_after"]
            assert disk_cover_radius(full) <= 1000-1e-5
            assert len(item["remaining_before"]) == len(item["remaining_after"])
    known = set(report.detected_channels)
    if len(known) < 16:
        for c in set(range(1,21))-known:
            actual = [a["position"] for a in history if a["action"] == "measure" and a["channel"] == c and a["result"] == "no_signal"]
            assert disk_cover_radius(actual) <= 1000-1e-5
        assert all(any(a["action"] == "measure" and a["phase"] == "coverage" and a["position"] == p for a in history)
                   for p in report.coverage_points)


def test_disabled_relocation_matches_frozen_axis_inferred():
    scenario = random_scenario(3,114001)
    old = run_inferred_state_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),config=BASE)
    new = run_relocating_state_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),config=BASE,enabled=False)
    assert old.action_history == new.action_history and old.virtual_time_s == new.virtual_time_s


@pytest.mark.parametrize("kwargs", [{"problem":4}, {"enabled":1}, {"max_active_probes":True}, {"config":{**BASE,"replace_coverage":True}}])
def test_bad_or_mixed_configuration_is_rejected_before_enter(kwargs):
    sim = LocalResearchSimulator(random_scenario(3,114001))
    with pytest.raises(ValueError):
        run_relocating_state_search(ObservationOnlyClient(sim.client()), **kwargs)
    assert sim.observation_history() == []
