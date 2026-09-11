"""Scheduling boundary tests using artificial beliefs and no simulator network."""
from types import SimpleNamespace

import pytest

from simulator_client.state import Position
from strategies.q4_cover_search import Q4CoverSearch
from strategies.q4_r2_scheduling import Q4R2Scheduling, _ServiceSliceExpired, run_q4_r2_scheduling
from strategies.search import _StopSearch


def search(config="cap16"):
    client = SimpleNamespace(state=SimpleNamespace(position=Position(0, 0), current_channel=1,
        virtual_time_s=0., sources={}))
    return Q4R2Scheduling(client, 20000, 6, config=config, max_expansions=0)


def test_sixteen_observed_but_unready_stops_discovery_not_removal(monkeypatch):
    s = search()
    s.detected = set(range(1, 17))
    monkeypatch.setattr(s, "_ready", lambda c: False)
    monkeypatch.setattr(s, "_target", lambda c: Position(0, 0))
    monkeypatch.setattr(s, "_scan", lambda p: pytest.fail("No discovery scan after all 16 known"))
    # An unresolved service failure must never count as successful completion.
    monkeypatch.setattr(s, "_resolve", lambda c: False)
    s._execute_plan()
    assert not s.report.completion_certified_under_model and not s.report.coverage_complete
    assert s.discovery_stop_log[0]["unresolved_channels"] == list(range(1, 17))
    assert not s.cleared


def test_sixteen_known_resolves_all_before_completion(monkeypatch):
    s = search()
    s.detected = set(range(1, 17))
    monkeypatch.setattr(s, "_target", lambda c: Position(0, 0))
    monkeypatch.setattr(s, "_ready", lambda c: False)
    monkeypatch.setattr(s, "_resolve", lambda c: (s.cleared.add(c), True)[1])
    with pytest.raises(_StopSearch):
        s._execute_plan()
    assert s.cleared == s.detected and s.report.completion_certified_under_model


def test_fifteen_known_keeps_remaining_discovery(monkeypatch):
    s = search()
    s.detected = set(range(1, 16))
    s.points = (Position(10, 0),)
    visited = []
    monkeypatch.setattr(s, "_ready", lambda c: False)
    monkeypatch.setattr(s, "_target", lambda c: Position(0, 0))
    monkeypatch.setattr(s, "_scan", lambda p: visited.append(p))
    monkeypatch.setattr(s, "_resolve", lambda c: (s.cleared.add(c), True)[1])
    s._execute_plan()
    assert visited == list(s.points) and s.report.coverage_complete and not s.discovery_stop_log


def test_early_budget_refuses_action_before_client_is_called():
    s = search("onroute")
    s.service_deadline = 60.
    with pytest.raises(_ServiceSliceExpired):
        s._perform("measure", Position(300, 0), 2, "active_localization")
    assert not s.report.action_history


def test_original_resolver_and_probe_method_are_inherited():
    assert Q4R2Scheduling._resolve is Q4CoverSearch._resolve
    assert Q4R2Scheduling._next_probe is Q4CoverSearch._next_probe


def test_cap16_preserves_original_trajectory_on_artificial_ten_source_case():
    from simulation import LocalResearchSimulator
    from simulation.cases import Scenario, Source
    from strategies.q4_cover_search import run_q4_cover_search
    from tests.test_strategy import ObservationOnlyClient
    case = Scenario("q4-scheduling-unit-only", 4, 0,
        tuple(Source(c, 100+8*c, 100+2*c, 1000., 180. if c == 10 else None)
              for c in range(1, 11)), error_mode="zero")
    old = run_q4_cover_search(ObservationOnlyClient(LocalResearchSimulator(case).client()),
                             profile="compact_22", schedule="joint", max_expansions=0)
    new = run_q4_r2_scheduling(ObservationOnlyClient(LocalResearchSimulator(case).client()),
                               config="cap16", max_expansions=0)
    assert new.action_history == old.action_history
    assert new.virtual_time_s == old.virtual_time_s
    assert new.cleared_channels == list(range(1, 11)) and new.completion_certified_under_model


def test_early_services_have_a_fixed_global_attempt_cap():
    s = search("onroute")
    s.early_attempted = {17, 18, 19, 20}
    s.detected.add(1)
    s.regions[1] = SimpleNamespace(vertices=[(100, 0)],
                                  enclosing_disk=lambda: SimpleNamespace(center=(100, 0), radius=30.))
    assert s._early_candidate(Position(200, 0)) is None


def test_early_interruption_restores_full_resolver_and_is_not_retried(monkeypatch):
    s = search("onroute")
    s.detected.add(1)
    s.regions[1] = SimpleNamespace(vertices=[(100, 0)],
                                  enclosing_disk=lambda: SimpleNamespace(center=(100, 0), radius=30.))
    candidate = s._early_candidate(Position(200, 0))
    assert candidate is not None
    def sliced(channel):
        s.client.state.virtual_time_s += 7.
        raise _ServiceSliceExpired()
    monkeypatch.setattr(s, "_resolve", sliced)
    s._early_service(candidate)
    assert s.service_deadline is None and not s.blocked
    assert s.early_service_log[0]["interrupted"] and s.early_service_log[0]["actual_cost_s"] == 7.
    assert s._early_candidate(Position(200, 0)) is None


@pytest.mark.parametrize("radius,center", [(41., (100, 0)), (30., (100, 200))])
def test_large_region_or_detour_does_not_interrupt(radius, center):
    s = search("onroute")
    s.detected.add(1)
    s.regions[1] = SimpleNamespace(vertices=[center], enclosing_disk=lambda: SimpleNamespace(center=center, radius=radius))
    assert s._early_candidate(Position(200, 0)) is None


@pytest.mark.parametrize("kwargs", [{"problem":3}, {"config":"bad"}, {"max_actions":True}, {"max_active_probes":-1}])
def test_bad_config_rejected_before_client_access(kwargs):
    with pytest.raises(ValueError):
        run_q4_r2_scheduling(object(), **kwargs)
