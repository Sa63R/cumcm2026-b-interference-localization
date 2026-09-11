"""Decision-local cache equivalence and invalidation on scripted public state."""
from unittest.mock import patch

import pytest

from simulator_client.state import Position
from strategies.search import _StopSearch
from tests.test_q4_rl_micro_controller import make, set_rectangle


def prepare(make, enabled):
    search, client = make(decision_cache=enabled)
    set_rectangle(search, half_x=10., half_y=3., channel=1)
    set_rectangle(search, half_x=80., half_y=15., channel=2)
    search.detected.add(3)
    search.near_points[3] = Position(200., 20.)
    client.state.position = Position(100., 0.)
    return search, client


def test_candidates_and_every_feature_exactly_equal_with_fewer_geometry_calls(make):
    import q4_rl.controller as base
    original = base.region_distance_lower
    results, counts = [], []
    for enabled in (False, True):
        search, _ = prepare(make, enabled)
        with patch.object(search, "_compute_safe_points", wraps=search._compute_safe_points) as safe:
            with patch.object(base, "region_distance_lower", wraps=original) as distance:
                candidates = search._candidates()
                features = search._features(candidates)
                results.append((candidates, features))
                counts.append((safe.call_count, distance.call_count))
        assert search._decision_cache is None
    assert results[0] == results[1]
    assert counts[1][0] < counts[0][0]
    assert counts[1][1] < counts[0][1]


def test_public_position_change_cannot_reuse_safe_edge(make):
    search, client = prepare(make, True)
    search._begin_decision_cache()
    first = search._safe_points(1)
    assert search._safe_points(1) is first
    client.state.position = Position(-100., 0.)
    second = search._safe_points(1)
    assert second != first
    assert search._decision_cache is None


@pytest.mark.parametrize("field,value", [("current_channel", 4), ("virtual_time_s", 7.)])
def test_public_channel_and_billing_change_invalidates(make, field, value):
    search, client = prepare(make, True)
    search._begin_decision_cache()
    search._ready(1)
    setattr(client.state, field, value)
    search._ready(1)
    assert search._decision_cache is None


def test_real_observation_and_clear_invalidate_before_and_after_request(make):
    search, client = prepare(make, True)
    search._candidates()
    assert not search._ready(2)
    original_reply = client.reply
    def reply(point, channel):
        assert search._decision_cache is None
        return original_reply(point, channel)
    client.reply = reply
    search._perform("measure", Position(0., 0.), 2, "fixture_near")
    assert search._decision_cache is None
    assert search._ready(2)
    search._begin_decision_cache()
    assert search._ready(2)
    search._perform("clear", Position(0., 0.), 2, "fixture_clear")
    assert search._decision_cache is None
    assert not search._ready(2)


def test_rejected_request_and_feature_exception_discard_cache(make):
    search, client = prepare(make, True)
    search._candidates()
    client.reject_measure = True
    with pytest.raises(_StopSearch):
        search._perform("measure", Position(0., 0.), 2, "fixture_rejected")
    assert search._decision_cache is None
    candidates = search._candidates()
    with patch.object(search, "_build_features", side_effect=ValueError("fixture")):
        with pytest.raises(ValueError):
            search._features(candidates)
    assert search._decision_cache is None


def test_new_decision_refreshes_changed_geometry(make):
    search, _ = prepare(make, True)
    first = search._candidates()
    assert any(c.kind == "certified_clear" and c.channel == 1 for c in first)
    # Synthetic observation update between independent decision constructions.
    set_rectangle(search, half_x=80., half_y=15., channel=1)
    second = search._candidates()
    assert not any(c.kind == "certified_clear" and c.channel == 1 for c in second)


def test_fallback_boundary_cannot_keep_a_candidate_cache(make):
    from q4_rl.controller import Q4RLSearch
    search, _ = prepare(make, True)
    search._candidates()
    assert search._decision_cache is not None
    def check_boundary(reason):
        assert search._decision_cache is None
    with patch.object(Q4RLSearch, "_finish_with_baseline", side_effect=check_boundary):
        search._finish_with_baseline("fixture")
    assert search._decision_cache is None
