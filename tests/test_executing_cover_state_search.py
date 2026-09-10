"""Immediate execution, real-channel evidence and transferred finite bounds."""

import itertools

import pytest

from planning.disk_cover import disk_cover_radius
from planning.state_route import RouteTask, solve_state_route
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from simulator_client.state import Position
from strategies.executing_cover_state_search import run_executing_cover_state_search
from strategies.inferred_state_search import run_inferred_state_search
from tests.test_inferred_silence import BASE
from tests.test_strategy import ObservationOnlyClient


@pytest.mark.parametrize("scenario", difficult_scenarios(3)+[random_scenario(3, 117001)])
def test_every_changed_site_is_the_next_real_cover_and_no_source_dispatch_changes(scenario):
    simulator = LocalResearchSimulator(scenario)
    report = run_executing_cover_state_search(ObservationOnlyClient(simulator.client()), config=BASE)
    assert simulator.evaluation()["all_cleared"] and simulator.evaluation()["failed_clear_count"] == 0
    assert report.virtual_time_s == simulator.evaluation()["virtual_time_s"]
    history = report.action_history
    for record, plan in zip(report.strategy_parameters["relocation_log"], report.strategy_parameters["planning_log"]):
        assert record["baseline_order"] == record["selected_order"]
        assert record["baseline_selected_kind"] == plan["selected_kind"]
        assert record["baseline_selected_channel"] == plan["selected_channel"]
        previous = history[:record["after_actual_action_count"]]
        known = {a["channel"] for a in previous if a["action"] == "measure" and a["result"] in ("near", "direction")}
        negatives = {(a["channel"], tuple(a["position"])) for a in previous if a["result"] == "no_signal"}
        assert all((c, tuple(p)) in negatives for c in set(range(1,21))-known
                   for p in record["executed_discovery_stations"])
        if record["relocated"]:
            actual = history[record["after_actual_action_count"]]
            assert plan["selected_kind"] == "cover"
            assert actual["phase"] == "coverage" and actual["position"] == record["new_position"]
            assert disk_cover_radius(record["executed_discovery_stations"]+record["remaining_after"]) <= 1000-1e-5
            assert plan["cost_s"] >= plan["lower_bound_s"]-1e-7
            assert not plan["exact"]
        else:
            assert record["baseline_selected_point"] == plan["selected_point"]


def test_disabled_action_history_is_exactly_the_unmodified_axis_inferred():
    scenario = random_scenario(3,117001)
    old = run_inferred_state_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),config=BASE)
    new = run_executing_cover_state_search(ObservationOnlyClient(LocalResearchSimulator(scenario).client()),config=BASE,enabled=False)
    assert new.action_history == old.action_history


def test_old_finite_bound_transfers_when_one_point_moves_for_all_small_graph_orders():
    start = Position(12, 8)
    old = [Position(-20,30),Position(5,-17),Position(31,7),Position(-9,-13),Position(0,0)]
    for site, moved in itertools.product(range(5), (Position(14,19), Position(-50,2))):
        changed = old.copy()
        changed[site] = moved
        tasks = [RouteTask(p, i%2==0, 3+i) for i,p in enumerate(old)]
        original = solve_state_route(tasks, start, scan_source_s=6, max_expansions=10000)
        def cost(order, points):
            current, value, unfinished = start, 0., {0,2,4}
            for i in order:
                value += current.distance_to(points[i])/5 + 3+i
                if i in unfinished:
                    unfinished.remove(i)
                elif i%2:
                    value += 6*len(unfinished)
                current = points[i]
            return value
        exact_new = min(cost(order,changed) for order in itertools.permutations(range(5)))
        assert original.lower_bound_s-2*old[site].distance_to(moved)/5 <= exact_new+1e-8


@pytest.mark.parametrize("kwargs", [{"problem":4}, {"enabled":1}, {"max_actions":True}, {"config":{**BASE,"replace_coverage":True}}])
def test_invalid_config_does_not_enter_simulator(kwargs):
    simulator = LocalResearchSimulator(random_scenario(3,117001))
    with pytest.raises(ValueError):
        run_executing_cover_state_search(ObservationOnlyClient(simulator.client()), **kwargs)
    assert simulator.observation_history() == []
