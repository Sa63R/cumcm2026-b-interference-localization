"""Artificial geometry/protocol fixtures only; no generated or stored worlds."""

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from localization.omni import OmniCandidateRegion
from planning.probe_candidates import geometry_candidates
from planning.radius_probe import choose_radius_probe
from simulation.engine import MemoryClient
from simulator_client.state import Position
from strategies.current_probe_state_search import CurrentProbeStateSearch, run_current_probe_state_search
from strategies.relocating_state_search import RelocatingStateSearch
from strategies.search import _StopSearch


ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "experiments/state_search_candidate_relocating_cover_v1.json").read_text())["kwargs"]["config"]


class Exchange:
    def __init__(self, outcome="near"):
        self.outcome = outcome
        self.calls = []
        self.position = (0., 0.)
        self.channel = 1
        self.time_us = 0

    def __call__(self, path, fields):
        self.calls.append((path, deepcopy(fields)))
        if path in ("/measure", "/clear"):
            target = (fields["position"]["x"], fields["position"]["y"])
            self.time_us += round(math.dist(self.position, target) / 5 * 1e6) + 5000000
            self.position = target
            if path == "/measure":
                self.time_us += 1000000 * (self.channel != fields["channel"])
                self.channel = fields["channel"]
        response = {"accepted": True, "real_timestamp_ms": 1, "virtual_time_s": self.time_us / 1e6}
        if path == "/enter":
            response.update(max_virtual_duration_s=360000., max_real_duration_s=1200., remaining_real_duration_s=1200.)
        elif path == "/exit":
            response["exit_reason"] = "user_exit"
        elif path == "/clear":
            response["clear_result"] = "success"
        else:
            response["measure_result"] = self.outcome
            if self.outcome == "direction":
                response["svd_deg"] = 0.
        return response


def setup(*, enabled=True, current=(0., 0.), vertices=((100., 0.), (180., 0.), (180., 40.), (100., 40.))):
    exchange = Exchange()
    client = MemoryClient(exchange, clock=lambda: 0.)
    client.enter()
    client.state.position = Position(*current)
    exchange.position = current
    policy = CurrentProbeStateSearch(client, 10000, 6, BASE, enabled)
    region = OmniCandidateRegion().observe((-300., 0.), 0.)
    region.vertices = tuple(vertices)
    region._circle = None
    policy.regions[3] = region
    policy.first_bearings[3] = 0.
    policy.detected.add(3)
    policy.observed_positions[3] = {(-300., 0.)}
    return policy, exchange


def test_spec_has_exact_original_configuration_and_only_wrapper_switch():
    old = json.loads((ROOT / "experiments/state_search_candidate_relocating_cover_v1.json").read_text())
    new = json.loads((ROOT / "experiments/state_search_candidate_current_probe_v1.json").read_text())
    assert old["kwargs"] == new["kwargs"]
    assert new["entrypoint"] == "strategies.current_probe_state_search:run_current_probe_state_search"


@pytest.mark.parametrize("current", [(0., 0.), (100., 80.), (150., -20.), (200., 80.)])
def test_real_score_is_exact_nested_original_family_without_region_mutation(current):
    policy, exchange = setup(current=current)
    reg = policy.regions[3]
    before = (reg.vertices, deepcopy(reg.observations), deepcopy(policy.observed_positions))
    old9, _ = choose_radius_probe(reg, Position(*current), 0., policy.observed_positions[3], .5)
    extras, _ = geometry_candidates(reg, mode="axis_quantile", old_best=old9)
    old23, oldlog = choose_radius_probe(reg, Position(*current), 0., policy.observed_positions[3], .5, extra_points=extras)
    wanted, wantedlog = choose_radius_probe(reg, Position(*current), 0., policy.observed_positions[3], .5, extra_points=[*extras, Position(*current)])
    actual = policy._next_probe(3, 0)
    audit = policy.current_probe_log[-1]
    assert actual == wanted
    assert audit["baseline_position"] == [old23.x, old23.y]
    assert audit["baseline_score_s"] == oldlog["score_s"]
    assert policy.probe_log[-1]["score_s"] == wantedlog["score_s"]
    assert audit["selected_score_s"] <= audit["baseline_score_s"]
    assert audit["baseline_probe_log"]["position"] == [old23.x, old23.y]
    assert before == (reg.vertices, reg.observations, policy.observed_positions)
    assert len(exchange.calls) == 1  # planning sends no request


def test_disabled_delegates_without_extra_geometry(monkeypatch):
    policy, _ = setup(enabled=False)
    seen = []
    expected = Position(7., 9.)
    monkeypatch.setattr(RelocatingStateSearch, "_next_probe", lambda self, channel, index: seen.append((channel, index)) or expected)
    monkeypatch.setattr(policy, "_current_gate", lambda *args: pytest.fail("disabled gate ran"))
    assert policy._next_probe(3, 0) == expected
    assert seen == [(3, 0)] and policy.current_probe_log == []


def test_actual_finite_geometry_really_selects_new_current_candidate():
    policy, _ = setup(current=(150., -20.))
    assert policy._next_probe(3, 0) == policy.client.state.position
    record = policy.current_probe_log[-1]
    assert record["new_current_selected"] and not record["extra_point_was_duplicate"]
    assert record["score_improvement_s"] > 0 and not record["changed_on_score_tie"]


def test_current_already_in_original_candidate_set_is_not_new_activation():
    policy, _ = setup(current=(140., 20.))
    policy._next_probe(3, 0)
    record = policy.current_probe_log[-1]
    assert record["extra_point_was_duplicate"]
    assert not record["new_current_selected"] and not record["changed_point"]


def test_added_search_cpu_and_geometry_counts_include_all_three_passes():
    policy, _ = setup(current=(150., -20.))
    policy._next_probe(3, 0)
    record = policy.current_probe_log[-1]
    old = record["baseline_probe_log"]
    final = policy.probe_log[-1]
    assert final["total_geometry_updates"] == old["total_geometry_updates"] + record["added_geometry_updates"]
    assert final["total_runtime_s"] == old["total_runtime_s"] + record["added_scoring_runtime_s"]


@pytest.mark.parametrize("kind,reason", [
    ("none", "nonfinite_or_empty_extended_score"),
    ("inf", "nonfinite_or_empty_extended_score"),
    ("unexpected_point", "extended_score_consistency_fallback"),
    ("worse_score", "extended_score_consistency_fallback"),
])
def test_rejected_third_pass_keeps_baseline_but_counts_its_cost(monkeypatch, kind, reason):
    policy, _ = setup(current=(100., 80.))
    def third_pass(*args, **kwargs):
        original = policy.probe_log[-1]
        score = original["score_s"]
        point = policy.client.state.position
        if kind == "none": point = None
        if kind == "inf": score = math.inf
        if kind == "unexpected_point": point = Position(71., 19.)
        if kind == "worse_score": score += 1.
        return point, {"score_s": score, "runtime_s": .125,
                       "geometry_updates": 123,
                       "proposed_candidates": original["proposed_candidates"] + 1}
    monkeypatch.setattr("strategies.current_probe_state_search.choose_radius_probe", third_pass)
    selected = policy._next_probe(3, 0)
    row = policy.current_probe_log[-1]
    old = row["baseline_probe_log"]
    final = policy.probe_log[-1]
    assert row["reason"] == reason and not row["changed_point"]
    assert [selected.x, selected.y] == old["position"] == final["position"]
    assert final["score_s"] == old["score_s"]
    assert final["total_geometry_updates"] == old["total_geometry_updates"] + 123
    assert final["total_runtime_s"] == old["total_runtime_s"] + .125
    json.dumps(policy.report.strategy_parameters, allow_nan=False)


def test_lexically_smaller_current_on_exact_score_tie_is_not_strict_gain(monkeypatch):
    policy, _ = setup(current=(100., 80.))
    def equal_score(*args, **kwargs):
        old = policy.probe_log[-1]
        current = policy.client.state.position
        assert [current.x, current.y] < old["position"]
        final = {k: deepcopy(v) for k, v in old.items()
                 if k in {"score_s", "hypotheses", "candidates", "proposed_candidates",
                          "score_kind", "evaluated_candidates", "pruned_candidates"}}
        final.update(position=[current.x, current.y], runtime_s=.125, geometry_updates=123)
        final["proposed_candidates"] += 1
        return current, final
    monkeypatch.setattr("strategies.current_probe_state_search.choose_radius_probe", equal_score)
    assert policy._next_probe(3, 0) == policy.client.state.position
    row = policy.current_probe_log[-1]
    assert row["changed_on_score_tie"] and row["new_current_selected"]
    assert row["score_improvement_s"] == 0.
    json.dumps(policy.report.strategy_parameters, allow_nan=False)


@pytest.mark.parametrize("kind,expected", [
    ("repeat", "not_first_probe"), ("unknown", "not_known_active_channel"),
    ("cleared", "not_known_active_channel"), ("near", "near_clear_precedes_probe"),
    ("observed", "already_measured_here"), ("far", "reception_not_certified"),
    ("empty", "empty_or_degenerate_region"), ("line", "empty_or_degenerate_region"),
    ("nan", "nonfinite_geometry"), ("small", "certified_clear_precedes_probe"),
])
def test_added_point_gate_fails_closed(kind, expected):
    policy, _ = setup()
    index = 1 if kind == "repeat" else 0
    if kind == "unknown": policy.detected.clear()
    if kind == "cleared": policy.cleared.add(3)
    if kind == "near": policy.near_points[3] = Position(0., 0.)
    if kind == "observed": policy.observed_positions[3].add((0., 0.))
    if kind == "far": policy.client.state.position = Position(-900., 0.)
    if kind == "empty": policy.regions[3].vertices = ()
    if kind == "line": policy.regions[3].vertices = ((0., 0.), (50., 0.), (100., 0.))
    if kind == "nan": policy.regions[3].vertices = ((math.nan, 0.), (50., 1.), (100., 0.))
    if kind == "small": policy.regions[3].vertices = ((0., 0.), (1., 0.), (0., 1.))
    assert policy._current_gate(3, index)["reason"] == expected


def test_freshness_is_channel_local_and_six_decimal_not_only_positive():
    policy, _ = setup()
    policy.observed_positions[4] = {(0., 0.)}
    assert policy._current_gate(3, 0)["eligible"]
    policy.observed_positions[3].add((round(0.0000004, 6), 0.))
    assert not policy._current_gate(3, 0)["eligible"]


def test_reception_uses_all_vertices_not_center_or_selected_hypotheses():
    vertices = ((900., -20.), (900., 20.), (1000.0001, 0.))
    policy, _ = setup(vertices=vertices)
    assert math.dist((0., 0.), policy.regions[3].enclosing_disk().center) < 1000
    assert not policy._current_gate(3, 0)["guaranteed_reception"]


def test_reception_margin_is_conservative_only_for_added_point():
    policy, _ = setup(vertices=((900., -20.), (900., 20.), (1000., 0.)))
    assert policy._current_gate(3, 0)["reason"] == "reception_not_certified"
    assert policy._current_gate(3, 0)["maximum_vertex_distance_m"] == 1000.


def test_parent_fallback_is_not_replaced(monkeypatch):
    policy, _ = setup()
    point = Position(12., 14.)
    monkeypatch.setattr(RelocatingStateSearch, "_next_probe", lambda *args: point)
    assert policy._next_probe(3, 0) == point
    assert policy.current_probe_log[-1]["reason"] == "unchanged_parent_fallback"


def test_real_measure_binds_selected_position_and_charges_channel_switch():
    policy, exchange = setup(current=(150., -20.))
    selected = policy._next_probe(3, 0)
    record = policy.current_probe_log[-1]
    policy._perform("measure", selected, 3, "active_localization")
    assert record["execution_status"] == "accepted"
    assert record["actual_action_ordinal"] == 1
    assert record["actual_position"] == record["selected_position"]
    expected = round(math.dist((150., -20.), (selected.x, selected.y)) / 5 * 1e6) + 6000000
    assert exchange.time_us == expected
    assert policy.client.state.current_channel == 3
    assert len(exchange.calls) == 2


def test_budget_stop_does_not_claim_a_measurement():
    policy, exchange = setup()
    point = policy._next_probe(3, 0)
    policy.actions = policy.max_actions - 1
    with pytest.raises(_StopSearch, match="action_budget"):
        policy._perform("measure", point, 3, "active_localization")
    assert policy.current_probe_log[-1]["execution_status"] == "not_accepted_or_budget_stopped"
    assert not policy.report.action_history and len(exchange.calls) == 1


def test_physical_resolve_keeps_atomic_measure_near_clear_and_no_cross_channel():
    policy, exchange = setup(current=(150., -20.))
    assert policy._resolve(3)
    assert [x[0] for x in exchange.calls] == ["/enter", "/measure", "/clear"]
    assert {x[1]["channel"] for x in exchange.calls[1:]} == {3}
    assert policy.current_probe_log[-1]["execution_status"] == "accepted"
    assert policy.report.action_history[0]["position"] == policy.current_probe_log[-1]["selected_position"]


@pytest.mark.parametrize("kwargs", [{"problem": 4}, {"enabled": 1}, {"max_actions": True},
                                     {"max_actions": 1}, {"max_active_probes": -1},
                                     {"max_active_probes": 31}, {"max_active_probes": True}])
def test_invalid_entrypoint_args_rejected_before_client_use(kwargs):
    with pytest.raises(ValueError):
        run_current_probe_state_search(None, **kwargs)
