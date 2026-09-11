"""Pure geometry and scripted wire feedback, without a scenario or network."""
import copy
import math

import pytest

from planning.cover_symmetry import d7_candidates, route_length, select_feedback_symmetry
from planning.q4_directional_cover import certified_cover_points
from simulator_client.state import Position
from strategies.q4_feedback_symmetry import Q4FeedbackSymmetry, run_q4_feedback_symmetry
from strategies.q4_joint_continuation import Q4JointContinuation
from strategies.search import _StopSearch
from tests.test_q4_clear_before_probe import Replies


def make(config="bearing_mean", *, replies=None, max_actions=20000):
    client = Replies()
    client.measure_replies = list(replies or [("no_signal", None)]*20)
    return Q4FeedbackSymmetry(client, max_actions, 6, max_expansions=0, config=config), client


def capture_parent(monkeypatch):
    captured = {}
    def execute(policy):
        captured.update(points=policy.points, report=copy.deepcopy(policy.report.as_dict()),
                        regions=copy.deepcopy(policy.regions), actions=policy.actions)
        return "captured"
    monkeypatch.setattr(Q4JointContinuation, "_execute_plan", execute)
    return captured


def test_all_d7_permutations_preserve_exact_coordinates_and_rigid_path():
    points, _ = certified_cover_points("compact_22")
    candidates = d7_candidates(points)
    assert len(candidates) == 14 and candidates[0]["permutation"] == list(range(22))
    assert len({tuple(c["permutation"]) for c in candidates}) == 14
    for c in candidates:
        angle = 2*math.pi*c["rotation_index"]/7
        route = [points[i] for i in c["permutation"]]
        assert set(route) == set(points) and route[0] == Position(0, 0)
        assert abs(route_length(route)-route_length(points)) <= 1e-7
        for original, transformed in zip(points, route):
            z = complex(original.x, original.y)
            if c["reflected"]:
                z = z.conjugate()
            z *= complex(math.cos(angle), math.sin(angle))
            assert math.hypot(z.real-transformed.x, z.imag-transformed.y) < 1e-7
            assert any(transformed is p for p in points)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "moved", "origin"])
def test_wrong_public_point_set_rejected(kind):
    points = list(certified_cover_points("compact_22")[0])
    if kind == "missing":
        points.pop()
    elif kind == "duplicate":
        points[2] = points[1]
    elif kind == "moved":
        points[1] = Position(points[1].x+.001, points[1].y)
    else:
        points[0], points[1] = points[1], points[0]
    with pytest.raises(ValueError):
        d7_candidates(points)


@pytest.mark.parametrize("bearings,status", [([], "insufficient_directions"),
    ([20.], "insufficient_directions"), ([0., 180.], "low_concentration")])
def test_bearing_fallbacks_are_identity(bearings, status):
    points = certified_cover_points("compact_22")[0]
    selected, log = select_feedback_symmetry(points, d7_candidates(points),
        config="bearing_mean", bearings=bearings, centers=[], known_count=len(bearings))
    assert selected == 0 and log["status"] == status


def test_circular_mean_wrap_and_equal_first_station_tie():
    points = certified_cover_points("compact_22")[0]
    selected, log = select_feedback_symmetry(points, d7_candidates(points),
        config="bearing_mean", bearings=[359., 1.], centers=[], known_count=2)
    assert min(abs(log["mean_bearing_deg"]), abs(log["mean_bearing_deg"]-360)) < 1e-12
    scores = log["candidates"]
    assert selected == min(range(14), key=lambda i: (scores[i]["score"], i))
    assert selected != 0 and log["concentration"] > .99
    assert sum(c["score"] == log["selected_score"] for c in scores) == 2


@pytest.mark.parametrize("config", ["bearing_mean", "early_centers"])
def test_public_sixteen_cap_keeps_identity(config):
    points = certified_cover_points("compact_22")[0]
    selected, log = select_feedback_symmetry(points, d7_candidates(points),
        config=config, bearings=[359., 1.], centers=[Position(900, 0)], known_count=16)
    assert selected == 0 and log["status"] == "source_count_cap"


def test_centers_scores_are_common_first_three_distance_sum():
    points = certified_cover_points("compact_22")[0]
    centers = [Position(900, 0), Position(1300, 100)]
    selected, log = select_feedback_symmetry(points, d7_candidates(points),
        config="early_centers", bearings=[], centers=centers, known_count=2)
    for c in log["candidates"]:
        expected = sum(min(math.hypot(q.x-points[i].x, q.y-points[i].y)
                           for i in c["permutation"][1:4]) for q in centers)
        assert c["score"] == pytest.approx(expected, abs=1e-10)
    best = min(log["candidates"], key=lambda c: (c["score"], c["id"]))
    assert selected == (best["id"] if best["score"] < log["original_score"]-1e-9 else 0)
    selected, log = select_feedback_symmetry(points, d7_candidates(points),
        config="early_centers", bearings=[], centers=list(points[1:4]), known_count=3)
    assert selected == 0 and log["original_score"] == 0


@pytest.mark.parametrize("config,status", [("bearing_mean", "insufficient_directions"),
    ("early_centers", "no_positive_regions")])
def test_paid_origin_scan_then_parent_gets_only_remaining_twenty_one(monkeypatch, config, status):
    captured = capture_parent(monkeypatch)
    policy, client = make(config)
    policy.actions = 1  # Accepted /enter is counted by inherited run().
    assert policy._execute_plan() == "captured"
    assert len(client.calls) == 20 and [x[2] for x in client.calls] == list(range(1, 21))
    assert all(x[1] == Position(0, 0) for x in client.calls)
    assert client.state.virtual_time_s == 119.
    assert captured["points"] == policy.symmetry_base_points[1:]
    assert len(captured["report"]["coverage_points"]) == 22
    assert captured["report"]["coverage_points_visited"] == 1
    assert captured["actions"] == 21 and len(policy.points) == 22
    event = policy.symmetry_log[0]
    assert event["origin_scan_start"] == 0 and event["origin_scan_end"] == 20
    assert event["after_actual_action_count"] == event["end_actual_action_count"] == 20
    assert event["status"] == status and event["selected_id"] == 0
    assert event["budget"]["policy_actions_before_scan"] == 1
    assert event["budget"]["policy_actions_after_scan"] == 21


@pytest.mark.parametrize("config", ["bearing_mean", "early_centers"])
def test_choice_includes_last_channel_and_does_not_rotate_observed_geometry(monkeypatch, config):
    captured = capture_parent(monkeypatch)
    replies = [("direction", 0.)]+[("no_signal", None)]*18+[("direction", 2.)]
    policy, _ = make(config, replies=replies)
    policy._execute_plan()
    event = policy.symmetry_log[0]
    assert [b["channel"] for b in event["bearings"]] == [1, 20]
    assert [c["channel"] for c in event["positive_centers"]] == [1, 20]
    assert event["selected_id"] != 0
    assert captured["points"] == tuple(policy.symmetry_base_points[i] for i in event["selected_permutation"][1:])
    for channel in (1, 20):
        assert policy.regions[channel].vertices == captured["regions"][channel].vertices
        assert policy.regions[channel].observations[0].position == (0., 0.)
    assert policy.report.coverage_points == event["selected_full_points"]


def test_near_does_not_invent_bearing_or_canonical_region_center(monkeypatch):
    capture_parent(monkeypatch)
    policy, _ = make("early_centers", replies=[("near", None)]+[("no_signal", None)]*19)
    policy._execute_plan()
    event = policy.symmetry_log[0]
    assert event["known_channels"] == [1] and not event["bearings"] and not event["positive_centers"]
    assert event["status"] == "no_positive_regions"


@pytest.mark.parametrize("reason", ["rejected", "action_budget", "virtual_budget", "real_deadline"])
def test_partial_origin_scan_never_selects_or_claims_visit(reason):
    policy, client = make()
    original = copy.deepcopy(policy.report.coverage_points)
    if reason == "action_budget":
        policy.max_actions = 6
    elif reason == "virtual_budget":
        client.state.max_virtual_duration_s = 30.
    else:
        def before_measure():
            if len(client.calls) == 4:
                if reason == "rejected":
                    client.reject_kind = "measure"
                else:
                    client.remaining_real_time_s = 0.
        client.before_measure = before_measure
    with pytest.raises(_StopSearch):
        policy._execute_plan()
    assert 0 < len(policy.report.action_history) < 20
    assert not policy.symmetry_log and policy.report.coverage_points_visited == 0
    assert policy.report.coverage_points == original and policy.points == policy.symmetry_base_points


def test_actual_parent_starts_chosen_next_station_without_repeating_origin():
    policy, client = make(replies=[("direction", 0.), ("direction", 2.)]+[("no_signal", None)]*19,
                          max_actions=22)
    with pytest.raises(_StopSearch) as stopped:
        policy._execute_plan()
    assert stopped.value.reason == "action_budget"
    assert len(client.calls) == 21
    event = policy.symmetry_log[0]
    assert client.calls[20][1] == Position.coerce(event["selected_full_points"][1])
    assert client.calls[20][1] != Position(0, 0)
    assert policy.report.coverage_points_visited == 1 and len(policy.points) == 22


def test_actual_parent_cap_skips_all_remaining_stations_and_clears_real_nears():
    policy, client = make(replies=[("near", None)]*16+[("no_signal", None)]*4)
    client.clear_replies = ["success"]*16
    with pytest.raises(_StopSearch) as stopped:
        policy._execute_plan()
    assert stopped.value.reason == "source_count_upper_bound_reached"
    assert policy.cleared == set(range(1, 17)) and len(client.calls) == 36
    assert all(p == Position(0, 0) for _, p, _ in client.calls)
    assert policy.symmetry_log[0]["status"] == "source_count_cap"
    assert policy.discovery_stop_log[0]["omitted_cover_stations"] == 21
    assert policy.report.coverage_points_visited == 1 and len(policy.points) == 22


def test_physics_resolver_and_scheduling_methods_remain_inherited():
    for method in ("_scan", "_perform", "_resolve", "_next_probe", "_clear", "_early_candidate",
                   "_early_service", "_check_budget", "run"):
        assert getattr(Q4FeedbackSymmetry, method) is getattr(Q4JointContinuation, method)


@pytest.mark.parametrize("kwargs", [{"problem": 3}, {"problem": True}, {"config": "adaptive"},
    {"max_actions": 1}, {"max_active_probes": True}, {"max_expansions": -1}])
def test_invalid_entrypoint_rejected_before_client(kwargs):
    with pytest.raises(ValueError):
        run_q4_feedback_symmetry(object(), **kwargs)
