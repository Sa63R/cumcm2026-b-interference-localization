"""Executable baseline equivalence is the oracle, not optimizer convergence."""

import json
from pathlib import Path

import pytest

from planning.certified_tail import CertifiedDisk, ClearVisit, improve_certified_tail, path_cost_us, predict_baseline_tail
from simulation import LocalResearchSimulator, random_scenario
from simulation.cases import Scenario, Source
from simulator_client.state import Position
from strategies.certified_tail_state_search import CertifiedTailStateSearch, run_certified_tail_state_search
from strategies.relocating_state_search import RelocatingStateSearch, run_relocating_state_search
from tests.test_strategy import ObservationOnlyClient


BASE = json.loads((Path(__file__).resolve().parents[1] /
    "experiments/state_search_candidate_relocating_cover_v1.json").read_text())["kwargs"]["config"]
LOCATIONS = ((0., 0.), (0., 100.), (100., 100.), (100., 0.))


def fixture_run(cls, count=3, *, near=False, total_sources=10, expanded=0,
                stop_budget=False, mutate_after_first=False, later_budget=False,
                client_cls=ObservationOnlyClient):
    case = Scenario("certified-tail-physical-fixture", 3, 0,
        tuple(Source(c, *(LOCATIONS[c-1] if c <= count else (-500., -500.)), 1000.)
              for c in range(1, total_sources+1)), error_mode="zero")
    simulator = LocalResearchSimulator(case)
    policy = cls(client_cls(simulator.client()), 10000, 6, BASE, True)
    capture = {}
    def execute():
        # These explicit physical fixture actions establish all discovery and
        # certificates. The strategy never receives the fixture's source list.
        for point in policy.points:
            policy._scan(point)
        policy.report.coverage_complete = True
        for channel in range(count + 1, total_sources + 1):
            policy._perform("clear", Position(-500, -500), channel, "fixture_setup")
        for channel in range(1, count + 1):
            x, y = LOCATIONS[channel - 1]
            positions = ((x + 4, y),) if near else ((x + 100, y), (x, y + 100))
            for point in positions:
                policy._perform("measure", Position(*point), channel, "fixture_setup")
        policy._perform("measure", Position(200, -100), 20, "fixture_setup")
        policy.total_expansions = expanded
        disks = []
        for channel in sorted(policy.detected - policy.cleared):
            if channel in policy.near_points:
                disks.append(CertifiedDisk(channel, policy.near_points[channel], 14.9, True))
            else:
                circle = policy.regions[channel].enclosing_disk()
                assert circle.radius <= 19.9
                disks.append(CertifiedDisk(channel, Position(*circle.center), 19.9-circle.radius))
        capture["prediction"] = predict_baseline_tail(disks, policy.client.state.position,
            max_expansions=BASE["max_expansions"], max_total_expansions=BASE["max_total_expansions"],
            total_expansions=expanded, scan_source_s=BASE["scan_source_s"])
        capture["start_index"] = len(policy.report.action_history)
        capture["start_time_us"] = round(policy.client.state.virtual_time_s * 1e6)
        capture["channel_before"] = policy.client.state.current_channel
        remaining = []
        if stop_budget:
            policy.max_actions = policy.actions + 2  # one clear, one reserved exit
        index = 0
        while True:
            task = policy._next_task(remaining)
            if task is None:
                break
            assert task[0] == "source"
            if later_budget and index == 1:
                policy.max_actions = policy.actions + 1
            if mutate_after_first and index == 1:
                # A real external no-signal read changes the observed state;
                # cancellation must invalidate the full-tail dominance claim.
                policy._perform("measure", policy.client.state.position, 20, "external_fixture_change")
            assert policy._resolve(task[1])
            index += 1
    policy._execute_plan = execute
    report = policy.run()
    return simulator, policy, report, capture


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("near", [False, True])
@pytest.mark.parametrize("expanded", [0, 59999, 60000])
def test_shadow_baseline_matches_every_dynamic_actual_clear_and_rounding(count, near, expanded):
    sim, policy, report, capture = fixture_run(RelocatingStateSearch, count, near=near, expanded=expanded)
    prediction = capture["prediction"]
    tail = report.action_history[capture["start_index"]:]
    assert all(a["action"] == "clear" and a["result"] == "success" for a in tail)
    assert [(a["channel"], a["position"]) for a in tail] == [
        (v.channel, [v.position.x, v.position.y]) for v in prediction.visits]
    assert round(report.virtual_time_s * 1e6) - capture["start_time_us"] == prediction.cost_us
    assert policy.total_expansions == prediction.expanded_after_each[-1]
    assert policy.client.state.current_channel == capture["channel_before"]
    assert report.completion_certified_under_model and sim.evaluation()["failed_clear_count"] == 0


@pytest.mark.parametrize("near,total_sources", [(False, 10), (True, 10), (True, 16)])
def test_committed_full_tail_physically_executes_and_beats_predicted_baseline(near, total_sources):
    sim, policy, report, capture = fixture_run(CertifiedTailStateSearch, 3, near=near, total_sources=total_sources)
    assert report.completion_certified_under_model and sim.evaluation()["all_cleared"]
    assert sim.evaluation()["failed_clear_count"] == 0
    accepted = [l for l in policy.tail_log if l["accepted"]]
    assert accepted
    assert len(accepted) == 1
    record = accepted[0]
    assert record["completed"] and not record.get("cancelled")
    assert record["dominance_status"] == "complete_execution_matches_compared_route"
    assert record["actual_cost_us"] == record["candidate_cost_us"]
    assert record["baseline_cost_us"] - record["actual_cost_us"] >= 10000
    tail = report.action_history[capture["start_index"]:]
    assert [(a["channel"], *a["position"]) for a in tail] == [tuple(v) for v in record["candidate_visits"]]
    assert all(a["phase"] == "certified_tail_clear" for a in tail)
    assert policy.client.state.current_channel == capture["channel_before"]
    if total_sources == 16:
        assert report.completion_reason == "source_count_upper_bound_reached"


def test_foreseeable_budget_interrupt_does_not_change_original_order():
    sim, policy, report, _ = fixture_run(CertifiedTailStateSearch, 3, near=True, stop_budget=True)
    assert report.completion_reason == "action_budget"
    assert not report.completion_certified_under_model
    assert sim.observation_history()[-1]["action"] == "/exit"
    accepted = [l for l in policy.tail_log if l["accepted"]]
    assert not accepted
    assert report.strategy_parameters["certified_tail_action_budget_skip"] >= 1


def test_low_real_budget_skips_extra_planning_and_keeps_original_tail():
    class LowRealClient(ObservationOnlyClient):
        @property
        def remaining_real_time_s(self):
            return 5.0
    sim, policy, report, capture = fixture_run(CertifiedTailStateSearch, 3, near=True,
                                             client_cls=LowRealClient)
    assert sim.evaluation()["all_cleared"] and not policy.tail_log
    tail = report.action_history[capture["start_index"]:]
    assert [(a["channel"], a["position"]) for a in tail] == [
        (v.channel, [v.position.x, v.position.y]) for v in capture["prediction"].visits]
    assert report.strategy_parameters["certified_tail_real_budget_skip"] >= 1


def test_unexpected_budget_interrupt_withdraws_full_tail_dominance():
    sim, policy, report, _ = fixture_run(CertifiedTailStateSearch, 3, near=True, later_budget=True)
    assert report.completion_reason == "action_budget"
    assert not report.completion_certified_under_model
    assert sim.observation_history()[-1]["action"] == "/exit"
    accepted = [l for l in policy.tail_log if l["accepted"]]
    assert accepted and not accepted[0].get("completed")
    assert accepted[0]["dominance_status"] == "incomplete_execution_no_full_tail_claim"


def test_external_state_change_cancels_route_and_withdraws_dominance_claim():
    sim, policy, report, _ = fixture_run(CertifiedTailStateSearch, 3, near=True, mutate_after_first=True)
    assert report.completion_certified_under_model and sim.evaluation()["all_cleared"]
    cancelled = [l for l in policy.tail_log if l.get("cancelled")]
    assert cancelled and cancelled[0]["dominance_status"] == "execution_preconditions_invalidated"


def test_only_real_discovery_completion_and_unblocked_certified_2to4_qualify():
    sim = LocalResearchSimulator(random_scenario(3, 114005))
    policy = CertifiedTailStateSearch(ObservationOnlyClient(sim.client()), 10000, 6, BASE, True)
    policy.detected = {1, 2}
    policy.near_points = {1: Position(0, 0), 2: Position(0, 100)}
    assert policy._certified_disks([Position(100, 100)]) is None
    assert len(policy._certified_disks([])) == 2
    policy.blocked.add(1)
    assert policy._certified_disks([]) is None
    policy.blocked.clear()
    policy.cleared = set(range(3, 17))
    assert len(policy._certified_disks([Position(100, 100)])) == 2
    no_cap = CertifiedTailStateSearch(ObservationOnlyClient(LocalResearchSimulator(random_scenario(3,114005)).client()),
        10000, 6, {**BASE, "stop_discovery_at_16": False}, True)
    no_cap.detected, no_cap.cleared, no_cap.near_points = policy.detected, policy.cleared, policy.near_points
    assert no_cap._certified_disks([Position(100, 100)]) is None
    assert len(no_cap._certified_disks([])) == 2


def test_optimizer_preserves_incumbent_and_disk_membership_with_coincident_points():
    disks = [CertifiedDisk(i, Position(0, 0), 0) for i in range(1, 5)]
    incumbent = tuple(ClearVisit(i, Position(0, 0)) for i in range(1, 5))
    visits, cost = improve_certified_tail(disks, Position(0, 0), incumbent)
    assert visits == incumbent and cost == 20_000_000
    assert path_cost_us(Position(0.00000249, 0), visits) == 20_000_000
    assert path_cost_us(Position(0.00000251, 0), visits) == 20_000_001


def test_disabled_full_smoke_is_identical_and_enabled_smoke_is_safe():
    case = random_scenario(3, 114005)
    original = run_relocating_state_search(ObservationOnlyClient(LocalResearchSimulator(case).client()), config=BASE)
    disabled = run_certified_tail_state_search(ObservationOnlyClient(LocalResearchSimulator(case).client()), config=BASE, enabled=False)
    assert original.action_history == disabled.action_history
    assert original.virtual_time_s == disabled.virtual_time_s
    sim = LocalResearchSimulator(case)
    enabled = run_certified_tail_state_search(ObservationOnlyClient(sim.client()), config=BASE)
    assert enabled.completion_certified_under_model and sim.evaluation()["all_cleared"]
    assert sim.evaluation()["failed_clear_count"] == 0


@pytest.mark.parametrize("kwargs", [{"problem": 4}, {"enabled": 1}, {"max_actions": 1},
    {"max_active_probes": True}, {"config": {**BASE, "replace_coverage": True}}])
def test_bad_options_do_not_enter(kwargs):
    sim = LocalResearchSimulator(random_scenario(3, 114005))
    with pytest.raises(ValueError):
        run_certified_tail_state_search(ObservationOnlyClient(sim.client()), **kwargs)
    assert sim.observation_history() == []
