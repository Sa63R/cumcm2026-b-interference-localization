"""Artificial geometry and public-prefix scan tests; no dataset or new cases."""

from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from localization.omni import OmniCandidateRegion
from planning.relative_silence import relative_silence_certificate
from simulation.engine import MemoryClient
from simulator_client.state import Position
from strategies.derived_silence_state_search import (
    DerivedSilenceStateSearch, run_derived_silence_state_search,
)
from strategies.relocating_state_search import RelocatingStateSearch
from strategies.search import _StopSearch


BASE = json.loads((Path(__file__).resolve().parents[1] /
                   "experiments/state_search_candidate_relocating_cover_v1.json").read_text())["kwargs"]["config"]


def region(vertices=((100., 0.), (200., 0.), (200., 50.), (100., 50.))):
    result = OmniCandidateRegion().observe((0., 0.), 0.)
    result.vertices = tuple(vertices)
    result._circle = None
    return result


class ArtificialExchange:
    """A tiny scripted protocol fixture, without hidden source data."""

    def __init__(self, outcomes=None):
        self.outcomes = outcomes or {}
        self.calls = []
        self.position = (0., 0.)
        self.channel = 1
        self.time_us = 0

    def __call__(self, path, fields):
        self.calls.append((path, deepcopy(fields)))
        response = {"accepted": True, "real_timestamp_ms": 1,
                    "virtual_time_s": self.time_us / 1e6}
        if path == "/enter":
            return {**response, "max_virtual_duration_s": 360000.,
                    "max_real_duration_s": 1200., "remaining_real_duration_s": 1200.}
        if path == "/exit":
            return {**response, "exit_reason": "user_exit"}
        if path in ("/measure", "/clear"):
            p = fields["position"]
            target = (p["x"], p["y"])
            self.time_us += round(math.dist(self.position, target) / 5 * 1e6)
            self.position = target
            if path == "/measure":
                self.time_us += 5000000 + 1000000 * (self.channel != fields["channel"])
                self.channel = fields["channel"]
                outcome = self.outcomes.get(fields["channel"], "no_signal")
                response["measure_result"] = outcome
                if outcome == "direction":
                    response["svd_deg"] = 0.
            else:
                self.time_us += 5000000
                response["clear_result"] = "success"
        response["virtual_time_s"] = self.time_us / 1e6
        return response


def policy(*, outcomes=None, max_actions=10000, **kwargs):
    exchange = ArtificialExchange(outcomes)
    client = MemoryClient(exchange, clock=lambda: 0.)
    client.enter()
    search = DerivedSilenceStateSearch(client, max_actions, 6, BASE, **kwargs)
    return search, exchange


def actual(search, channel, outcome, position, **extra):
    search.report.action_history.append({"action": "measure", "channel": channel,
        "position": list(position), "phase": "artificial_fixture", "result": outcome,
        "virtual_time_s": 0., **extra})


def give_relative_history(search, channel=1):
    search.detected.add(channel)
    search.regions[channel] = region()
    actual(search, channel, "direction", (0., 0.), bearing_deg=0.)
    actual(search, channel, "no_signal", (0., 1200.))
    search.regions[channel].observe_no_signal((0., 1200.))


def test_farther_negative_certificate_with_explicit_outward_interval_bound():
    reg = region()
    before = deepcopy(reg.__dict__)
    cert = relative_silence_certificate(reg, (0., 1300.), (0., 1200.))
    assert cert and cert["method"] == "relative_actual_negative"
    assert cert["affine_lower_bound_m2"] > cert["required_affine_margin_m2"]
    assert cert["signed_bisector_distance_lower_m"] > 1000.
    assert reg.__dict__ == before


@pytest.mark.parametrize("vertices,n,q", [
    (((0., 0.), (1., 0.), (1., 1.), (0., 1.)), (2., 0.), (3., 0.)),
    (((-.125, .2), (.33, -.47), (.375, .25)), (2.25, 3.75), (6.1, 7.3)),
    (((1e8, 1e8), (1e8+1, 1e8)), (999., 1.), (998., 1.)),
    (((0., 0.),), (1000., 0.), (1001., 0.)),
    (((0., 0.), (0., 100.)), (1000., 0.), (1001., 0.)),
])
def test_reported_lower_bound_is_below_exact_fraction_vertex_minimum(vertices, n, q):
    cert = relative_silence_certificate(region(vertices), q, n)
    assert cert is not None
    F = Fraction.from_float
    exact = min(sum((F(n[i])-F(q[i]))*(F(v[i])-(F(n[i])+F(q[i]))/2)
                    for i in range(2)) for v in vertices)
    assert F(cert["affine_lower_bound_m2"]) <= exact
    assert exact > 0


@pytest.mark.parametrize("vertices,n,q", [
    (((0., 0.), (1., 0.)), (3., 0.), (2., 0.)),
    (((0., 0.), (4., 0.)), (2., 0.), (3., 0.)),
    (((2.5, 0.),), (2., 0.), (3., 0.)),
    (((2.5-1e-6, 0.),), (2., 0.), (3., 0.)),
    (((0., 0.),), (1000., 0.), (1000., 0.)),
    ((), (1000., 0.), (1001., 0.)),
    (((math.inf, 0.),), (1000., 0.), (1001., 0.)),
])
def test_closer_mixed_boundary_identical_empty_and_nonfinite_fail_closed(vertices, n, q):
    assert relative_silence_certificate(region(vertices), q, n) is None


@pytest.mark.parametrize("margin", [0, -1, True, math.inf, math.nan])
def test_invalid_margin_is_rejected(margin):
    with pytest.raises(ValueError):
        relative_silence_certificate(region(), (0., 1300.), (0., 1200.), margin_m=margin)


def test_negative_only_region_cannot_establish_known_source():
    reg = OmniCandidateRegion().observe_no_signal((0., 1200.))
    assert relative_silence_certificate(reg, (0., 1300.), (0., 1200.)) is None


def test_policy_requires_same_channel_prior_actual_evidence_and_known_active_status():
    search, _ = policy()
    give_relative_history(search)
    cert = search._relative_certificate(1, Position(0., 1300.), search.regions[1])
    assert cert and cert["witness_action_ordinal"] == 2
    assert search._relative_certificate(2, Position(0., 1300.), search.regions[1]) is None
    search.detected.clear()
    assert search._relative_certificate(1, Position(0., 1300.), search.regions[1]) is None
    search.detected.add(1)
    search.cleared.add(1)
    assert search._relative_certificate(1, Position(0., 1300.), search.regions[1]) is None


def test_inferred_negative_point_cannot_become_an_actual_witness():
    search, _ = policy()
    search.detected.add(1)
    search.regions[1] = region().observe_no_signal((0., 1200.))
    actual(search, 1, "direction", (0., 0.), bearing_deg=0.)
    search.inferred.append({"channel": 1, "position": [0., 1200.],
                            "physical_measurement": False, "inference_kind": "inferred_no_signal"})
    assert search._relative_certificate(1, Position(0., 1300.), search.regions[1]) is None
    actual(search, 2, "no_signal", (0., 1200.))
    assert search._relative_certificate(1, Position(0., 1300.), search.regions[1]) is None


def test_successful_clear_in_history_prevents_stale_witness_even_with_stale_set():
    search, _ = policy()
    give_relative_history(search)
    search.report.action_history.append({"action": "clear", "channel": 1,
        "position": [100., 0.], "result": "success", "virtual_time_s": 0.})
    assert search._relative_certificate(1, Position(0., 1300.), search.regions[1]) is None


def test_relative_inference_changes_only_geometry_and_inference_log():
    search, exchange = policy(stop_scan_at_16=False)
    give_relative_history(search)
    before = len(search.report.action_history)
    search._scan(Position(0., 1300.))
    new = search.report.action_history[before:]
    assert {a["channel"] for a in new} == set(range(2, 21))
    assert all(a["action"] == "measure" for a in new)
    assert 1 not in search.client.state.sources
    assert 1 not in search.observed_positions
    assert search.report.measurement_count == 19
    inferred = search.inferred[-1]
    assert inferred["channel"] == 1 and inferred["after_actual_action_count"] == before
    assert inferred["method"] == "relative_actual_negative" and not inferred["physical_measurement"]
    assert inferred["witness_action_ordinal"] == 2
    assert search.client.state.virtual_time_s == pytest.approx(260. + 19*6)
    assert search.discovery_stations == [Position(0., 1300.)]
    assert search.scan_audit[-1]["credited_geometric_station"]
    assert (0., 1300.) in search.regions[1].no_signal_positions
    assert len(exchange.calls) == 20  # one enter and nineteen actual measures


def test_scan_midway_count_cap_keeps_known_uncertified_information_query():
    search, _ = policy(relative_silence=False, outcomes={5: "direction"})
    search.detected = set(range(1, 17)) - {5}
    search.near_points = {c: Position(0., 0.) for c in search.detected if c != 6}
    search.regions[6] = region()
    search._scan(Position(0., 0.))
    assert [a["channel"] for a in search.report.action_history] == [5, 6]
    log = search.scan_audit[-1]
    assert [x["channel"] for x in log["count_skipped"]] == [17, 18, 19, 20]
    assert all(len(x["known_channels"]) == 16 and x["after_actual_action_count"] == 2
               for x in log["count_skipped"])
    assert log["certified_by_count"] and not log["credited_geometric_station"]
    assert search.report.coverage_points_visited == 0 and search.discovery_stations == []


def test_cap_does_not_use_less_than_public_count_or_declared_scenario_count():
    search, _ = policy(relative_silence=False)
    search.detected = set(range(1, 16))
    search.near_points = {c: Position(0., 0.) for c in search.detected}
    search._scan(Position(0., 0.))
    assert [a["channel"] for a in search.report.action_history] == [16, 17, 18, 19, 20]
    assert search.scan_audit[-1]["count_skipped"] == []
    assert search.scan_audit[-1]["credited_geometric_station"]


def test_count_uses_distinct_union_with_cleared_and_all_skipped_never_visit():
    search, exchange = policy(relative_silence=False)
    search.cleared = set(range(1, 9))
    search.detected = set(range(5, 17))  # overlap must not double count
    search.near_points = {c: Position(0., 0.) for c in range(9, 17)}
    before = search.client.state.snapshot()
    search._scan(Position(123., 456.))
    assert search.client.state.snapshot() == before
    assert search.actions == 0 and search.report.action_history == []
    assert len(exchange.calls) == 1 and search.report.measurement_count == 0
    assert search.report.coverage_points_visited == 0 and search.discovery_stations == []
    log = search.scan_audit[-1]
    assert log["completed"] and log["certified_by_count"] and log["physical_action_count"] == 0
    assert not log["credited_geometric_station"]


def test_skipping_current_unknown_channel_does_not_retune_radio():
    search, exchange = policy(relative_silence=False)
    search.detected = set(range(1, 17))
    search.near_points = {c: Position(0., 0.) for c in range(2, 17)}
    search.regions[1] = region()
    search.client.state.current_channel = exchange.channel = 17
    search._scan(Position(0., 100.))
    assert search.scan_audit[-1]["count_skipped"][0]["channel"] == 17
    assert search.client.state.current_channel == 1
    assert search.client.state.time_breakdown.switching_s == 1
    assert search.report.measurement_count == 1


def test_partial_scan_budget_does_not_receive_completion_or_station_credit():
    search, _ = policy(max_actions=2)
    with pytest.raises(_StopSearch, match="action_budget"):
        search._scan(Position(0., 0.))
    assert len(search.report.action_history) == 1
    assert search.scan_audit[-1]["completed"] is False
    assert search.scan_audit[-1]["physical_action_count"] == 1
    assert search.report.coverage_points_visited == 0 and search.discovery_stations == []


@pytest.mark.parametrize("settings", [{"enabled": False},
    {"relative_silence": False, "stop_scan_at_16": False}])
def test_disabled_options_dispatch_directly_to_unchanged_parent(monkeypatch, settings):
    search, _ = policy(**settings)
    calls = []
    monkeypatch.setattr(RelocatingStateSearch, "_scan", lambda self, p: calls.append(p))
    search._scan(Position(2., 3.))
    assert calls == [Position(2., 3.)] and not search.scan_audit
    assert search.relocation_enabled and search.inferred_enabled


def test_original_1500_silence_inference_remains_separate():
    search, _ = policy(relative_silence=False, stop_scan_at_16=True)
    search.detected.add(1)
    search.regions[1] = region()
    actual(search, 1, "direction", (0., 0.), bearing_deg=0.)
    search._scan(Position(0., 2000.))
    assert search.scan_audit[-1]["old_silence_skipped"] == [1]
    assert search.inferred[-1]["method"] in ("enclosing_disk", "polygon_edges")
    assert search.derived_stats["relative_silence_skips"] == 0


@pytest.mark.parametrize("options", [{"enabled": 1}, {"relative_silence": "yes"},
    {"stop_scan_at_16": None}, {"max_actions": True}, {"max_active_probes": True}, {"problem": 4}])
def test_invalid_options_rejected_without_client_access(options):
    with pytest.raises(ValueError):
        run_derived_silence_state_search(None, **options)


def test_artificial_budget_run_retains_explicit_exit():
    exchange = ArtificialExchange()
    client = MemoryClient(exchange, clock=lambda: 0.)
    report = run_derived_silence_state_search(client, config=BASE, max_actions=3)
    assert report.completion_reason == "action_budget"
    assert not report.completion_certified_under_model
    assert exchange.calls[-1][0] == "/exit" and client.state.session == "exited"
    assert report.accepted_actions == 3


def test_three_specs_preserve_all_original_v1_parameters():
    directory = Path(__file__).resolve().parents[1] / "experiments"
    for name, switches in [("relative_only", (True, False)),
                           ("cap_only", (False, True)), ("combined", (True, True))]:
        spec = json.loads((directory / f"state_search_candidate_derived_silence_{name}_v1.json").read_text())
        assert spec["kwargs"]["config"] == BASE
        assert (spec["kwargs"]["relative_silence"], spec["kwargs"]["stop_scan_at_16"]) == switches
        assert spec["kwargs"]["enabled"] is True
