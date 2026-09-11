"""Scripted protocol and proof-ledger checks; no simulator or scenario batch."""
import copy
import math

import pytest

from q4_rl.controller import (Q4RLSearch, run_q4_rl, Candidate,
                              GLOBAL_DIM, CANDIDATE_DIM)
from simulator_client.state import ClientState, Position
from strategies.search import _StopSearch


class Replies:
    """An observation-only scripted client; no simulator/ground truth object."""
    def __init__(self):
        self.state = ClientState()
        self.remaining_real_time_s = None
        self.pending_request = None
        self.calls = []
        self.reply = lambda p, c: {"measure_result": "near" if c <= 10 else "no_signal"}
        self.clear_reply = "success"
        self.reject_measure = False
        self.exit_cost = 0.

    def enter(self):
        self.state.session = "active"
        return {"accepted": True}

    def exit(self):
        self.state.session = "exited"
        self.state.virtual_time_s += self.exit_cost
        return {"accepted": True}

    def measure(self, point, channel):
        self.calls.append(("measure", point, channel))
        if self.reject_measure:
            return {"accepted": False}
        result = self.reply(point, channel)
        self._charge(point, 5.+int(channel != self.state.current_channel))
        self.state.current_channel = channel
        return {"accepted": True, **result}

    def clear(self, point, channel):
        self.calls.append(("clear", point, channel))
        self._charge(point, 5. if self.clear_reply == "success" else 3.)
        return {"accepted": True, "clear_result": self.clear_reply}

    def _charge(self, point, fixed):
        self.state.virtual_time_s += round(self.state.position.distance_to(point)/5.*1e6)/1e6+fixed
        self.state.position = point


@pytest.fixture
def make(monkeypatch):
    import planning.q4_directional_cover as cover
    # Tests concern certificate bookkeeping, not this deliberately tiny mock
    # station set's geometry. Production construction verifies compact_22.
    monkeypatch.setattr(cover, "certified_cover_points", lambda profile:
                        ((Position(100, 0), Position(200, 0)),
                         {"passed": True, "test_only": True}))
    def factory(**kwargs):
        client = Replies()
        return Q4RLSearch(client, max_expansions=0, **kwargs), client
    return factory


def test_callback_chooses_initial_position_and_single_channel(make):
    choices = []
    def policy(global_features, rows):
        assert len(global_features) == GLOBAL_DIM
        assert all(len(row) == CANDIDATE_DIM for row in rows)
        # Pick a current-position measurement. No origin scan is forced.
        index = next(i for i, row in enumerate(rows) if row[0] and row[6])
        choices.append(index)
        return index
    search, client = make(policy=policy, max_decisions=1)
    result = search.run()
    assert client.calls[0][0:2] == ("measure", Position(0, 0))
    assert len(choices) == 1
    assert result.learning["decisions"] == 1
    assert result.completion_certified_under_model
    assert result.learning["completion_certificate"]
    assert len(result.learning["transitions"]) == 1
    assert sum(row["cost_s"] for row in result.learning["transitions"]) == pytest.approx(result.virtual_time_s)


def test_cover_credits_require_accepted_measure_at_exact_position(make):
    search, client = make()
    search._perform("measure", Position(100+1e-10, 0), 11, "test")
    assert 11 not in search.cover_ledger[Position(100, 0)]
    search._perform("measure", Position(100, 0), 11, "test")
    assert 11 in search.cover_ledger[Position(100, 0)]
    client.reject_measure = True
    with pytest.raises(_StopSearch):
        search._perform("measure", Position(200, 0), 11, "test")
    assert 11 not in search.cover_ledger[Position(200, 0)]


def test_ten_cleared_sources_do_not_prove_discovery(make):
    search, client = make()
    for channel in range(1, 11):
        search._perform("measure", Position(0, 0), channel, "test")
        search._perform("clear", Position(0, 0), channel, "test")
    assert search._terminal_gate() is False
    assert not search.report.completion_certified_under_model
    for point in search.points:
        for channel in range(11, 21):
            search._perform("measure", point, channel, "test")
    with pytest.raises(_StopSearch):
        search._terminal_gate()
    assert search.report.completion_certified_under_model
    assert search.report.learning["certified_absent_channels"] == list(range(11, 21))


def test_sixteen_positive_channels_are_discovery_not_removal(make):
    search, client = make()
    client.reply = lambda p, c: {"measure_result": "near"}
    for channel in range(1, 17):
        search._perform("measure", Position(0, 0), channel, "test")
    assert search._refresh_certificate()
    assert search._terminal_gate() is False
    for channel in range(1, 17):
        search._perform("clear", Position(0, 0), channel, "test")
    with pytest.raises(_StopSearch):
        search._terminal_gate()
    assert search.report.learning["completion_certificate"].startswith("public_16")


def test_negative_response_never_shrinks_region_or_marks_absent_early(make):
    search, client = make()
    client.reply = lambda p, c: {"measure_result": "direction", "svd_deg": 0.}
    search._perform("measure", Position(-1000, 0), 1, "test")
    vertices = copy.deepcopy(search.regions[1].vertices)
    client.reply = lambda p, c: {"measure_result": "no_signal"}
    search._perform("measure", Position(0, 0), 1, "test")
    search._refresh_certificate()
    assert search.regions[1].vertices == vertices
    assert not search.report.learning["certified_absent_channels"]


def test_failed_clear_and_unresolved_fallback_are_not_success(make):
    search, client = make(max_decisions=1)
    client.clear_reply = "no_target_in_range"
    report = search.run()
    assert not report.completion_certified_under_model
    assert report.completion_reason == "q4_rl_uncertified_or_unresolved"
    assert report.learning["failed_clear_count"] == 10
    assert not report.learning["training_success"]
    assert sum(row["cost_s"] for row in report.learning["transitions"]) == pytest.approx(report.virtual_time_s)


def test_complete_tail_and_exit_cost_are_recorded(make):
    search, client = make(max_decisions=1)
    client.exit_cost = 2.5
    report = search.run()
    transition = report.learning["transitions"][-1]
    assert transition["terminal"] is True
    assert transition["fallback_cost_s"] > 0
    assert report.learning["uncovered_cost_s"] == pytest.approx(2.5)
    assert transition["cost_s"] == pytest.approx(report.virtual_time_s)
    assert report.as_dict()["learning"]["transitions"] == report.learning["transitions"]


def test_features_contain_relations_not_absolute_channel_identity(make):
    search, client = make()
    candidates = [Candidate("measure", Position(100, 0), 11, True),
                  Candidate("measure", Position(100, 0), 12, True)]
    _, rows = search._features(candidates)
    assert rows[0] == rows[1]
    client.state.current_channel = 12
    _, changed = search._features(candidates)
    assert [i for i, (a, b) in enumerate(zip(changed[0], changed[1])) if a != b] == [5, 7]


def _original_uncached_features(search, candidates):
    """Frozen v1 reference for an exact cache-equivalence regression."""
    current = search.client.state.position
    unknown = search._unknown()
    pending = {p: len(unknown-scanned) for p, scanned in search.cover_ledger.items()}
    stations = len(search.points)
    global_features = [current.x/3600., current.y/3600.,
        search.client.state.virtual_time_s/360000., len(search.detected)/20.,
        len(search.cleared)/20., len(unknown)/20.,
        sum(search._ready(c) for c in search.detected-search.cleared)/20.,
        sum(pending.values())/(20.*stations), sum(v > 0 for v in pending.values())/stations,
        sum((current, c) not in search.actual_measurements for c in unknown)/20.]
    rows = []
    for candidate in candidates:
        p, c = candidate.point, candidate.channel
        region = search.regions.get(c)
        radius = region.enclosing_disk().radius if region is not None and region.vertices else 1800.
        area = region.area if region is not None and region.vertices else math.pi*1800.**2
        if c in search.near_points:
            radius, area = 5., math.pi*25.
        distance = current.distance_to(p)
        cost = distance/5.+5.+(candidate.kind == "measure" and c != search.client.state.current_channel)
        rows.append([float(candidate.kind == "measure"), float(candidate.kind == "service"),
            (p.x-current.x)/3600., (p.y-current.y)/3600., distance/3600., cost/1000.,
            float(p == current), float(c == search.client.state.current_channel),
            float(c in search.detected), float(search._ready(c)), radius/1800.,
            area/(math.pi*1800.**2), search.measurement_counts[c]/50.,
            search.negative_counts[c]/50., pending.get(p, 0)/20.,
            sum(c not in scanned for scanned in search.cover_ledger.values())/stations])
    return global_features, rows


def test_cached_features_exactly_equal_v1_before_and_after_real_observations(make, monkeypatch):
    search, client = make()
    stages = [(Position(-1000, 0), 1, {"measure_result": "direction", "svd_deg": 0.}),
              (Position(0, -1000), 1, {"measure_result": "direction", "svd_deg": 90.}),
              (Position(100, 0), 11, {"measure_result": "no_signal"}),
              (Position(200, 0), 2, {"measure_result": "near"})]
    for position, channel, reply in stages:
        candidates = search._candidates()
        expected = _original_uncached_features(search, candidates)
        assert search._features(candidates) == expected
        client.reply = lambda p, c, reply=reply: reply
        search._perform("measure", position, channel, "test")
    search._perform("clear", Position(200, 0), 2, "test")
    candidates = search._candidates()
    expected = _original_uncached_features(search, candidates)
    distance_calls = []
    original_distance = Position.distance_to
    def counted_distance(a, b):
        distance_calls.append((a, b))
        return original_distance(a, b)
    monkeypatch.setattr(Position, "distance_to", counted_distance)
    assert search._features(candidates) == expected
    assert len(distance_calls) == len({candidate.point for candidate in candidates})
    assert len(distance_calls) < len(candidates)


def test_invalid_policy_uses_accounted_fallback(make):
    search, client = make(policy=lambda g, c: -1)
    report = search.run()
    assert report.learning["fallback_reason"] == "invalid_policy_action"
    assert report.completion_certified_under_model
    assert report.learning["decisions"] == 0
    assert report.learning["fallback_cost_s"] == pytest.approx(report.virtual_time_s)


def test_rejected_action_preserves_zero_cost_transition(make):
    search, client = make()
    client.reject_measure = True
    report = search.run()
    assert report.completion_reason == "request_rejected"
    assert len(report.learning["transitions"]) == 1
    assert report.learning["transitions"][0]["cost_s"] == 0.
    assert not report.completion_certified_under_model


def test_r8_clear_interception_never_credits_the_unexecuted_measure(make):
    search, client = make()
    client.reply = lambda p, c: {"measure_result": "direction",
                                "svd_deg": 0. if p.x == -1000 else 90.}
    search._perform("measure", Position(-1000, 0), 1, "test")
    search._perform("measure", Position(0, -1000), 1, "test")
    circle = search.regions[1].enclosing_disk()
    assert 19.9 < circle.radius <= 40.
    center = Position.coerce(circle.center)
    assert search._resolve(1)
    assert search.measurement_counts[1] == 2
    assert (center, 1) not in search.actual_measurements
    assert search.report.action_history[-1]["action"] == "clear"
    assert search.report.action_history[-1]["phase"] == "speculative_clear_before_probe"


def test_factory_rejects_other_problem_before_client_access():
    with pytest.raises(ValueError):
        run_q4_rl(object(), problem=3)
