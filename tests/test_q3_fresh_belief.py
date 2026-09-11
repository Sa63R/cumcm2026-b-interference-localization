"""Fresh integration checks; no official service or validation data is used."""

from collections import Counter
import copy
import math
import time

import pytest

from simulation.cases import Scenario, Source
from simulation.engine import LocalResearchSimulator
from strategies.q3_fresh import FreshQ3
from strategies.q3_fresh_belief import (BeliefSamplingError, BeliefSamplingTimeout,
                                       make_branch, sample_worlds)


def event(channel, position, result, bearing=None):
    row = dict(action="clear" if result in {"success", "no_target_in_range"} else "measure",
               channel=channel, position=position, result=result)
    if bearing is not None:
        row["bearing_deg"] = bearing
    return row


def active_policy():
    # Coordinates belong to this small unit fixture, never a policy input.
    case = Scenario("fresh-belief-unit", 3, 91,
                    (Source(1, 400, 300, 1100),) + tuple(
                        Source(c, -1500, 0, 1000) for c in range(2, 11)), "zero")
    client = LocalResearchSimulator(case).client()
    client.enter()
    return FreshQ3(client, "v1", selective=True)


def assert_history_compatible(world, history):
    sources = {s.channel: s for s in world.sources}
    assert 10 <= len(sources) == len(world.sources) <= 16
    assert all(1000 <= s.reception_radius_m <= 1500 for s in sources.values())
    assert all(math.hypot(s.x, s.y) <= 1800 for s in sources.values())
    removed = set()
    for row in history:
        source = sources.get(row["channel"])
        alive = source is not None and row["channel"] not in removed
        d = math.dist((source.x, source.y), row["position"]) if alive else math.inf
        result = row["result"]
        if result == "success":
            assert alive and d <= 20
            removed.add(row["channel"])
        elif result == "no_target_in_range":
            assert d > 20
        elif result == "no_signal":
            assert not alive or d > source.reception_radius_m
        elif result == "near":
            assert alive and d <= 5
        else:
            assert alive and 5 < d <= source.reception_radius_m
            x, y = row["position"]
            theta = math.degrees(math.atan2(source.y-y, source.x-x)) % 360
            assert abs((theta-row["bearing_deg"]+180) % 360-180) <= 1.005 + 1e-9


def test_wrapper_reuses_existing_posterior_without_changing_history():
    from strategies.q3_belief import sample_worlds as original
    policy = active_policy()
    policy.measure((0, 0), 1, "cover")
    policy.measure((900, 0), 1, "localize")
    policy.measure((1700, 300), 1, "opportunistic")
    history = copy.deepcopy(policy.history)
    states = copy.deepcopy(policy.channels)
    worlds = sample_worlds(policy.history, count=12, seed=461)
    assert worlds == original(history, count=12, seed=461)
    assert policy.history == history
    assert policy.channels[1].status == states[1].status
    assert policy.channels[1].negatives == states[1].negatives
    for world in worlds:
        assert_history_compatible(world, history)


def test_fresh_clear_history_retains_original_source_but_branch_is_silent():
    policy = active_policy()
    policy.measure((0, 0), 1, "cover")
    policy.measure((400, 300), 1, "localize")
    policy.clear((402, 300), 1, "clear")
    history = policy.history + [event(1, (400, 300), "no_signal")]
    for world in sample_worlds(history, count=6, seed=31):
        assert_history_compatible(world, history)
        assert 1 in {s.channel for s in world.sources}
        branch = make_branch(world, policy.client.state, history)
        assert branch.state.sources[1].status == "cleared"
        assert branch.measure((0, 0), 1)["measure_result"] == "no_signal"
        assert branch.clear((402, 300), 1)["clear_result"] == "no_target_in_range"
        assert branch.state.session == "active"


def test_failed_clear_and_strict_negative_radius_survive_sampling():
    history = [event(1, (0, 0), "direction", 36.87),
               event(1, (900, 0), "direction", 149.04),
               event(1, (1700, 300), "no_signal"),
               event(1, (430, 300), "no_target_in_range"),
               event(1, (400, 300), "near")]
    for world in sample_worlds(history, count=24, seed=920):
        assert_history_compatible(world, history)


def test_source_count_posterior_contains_channel_subset_factor():
    # With d=10 observed channels and no optional evidence,
    # P(N | these channels exist) is proportional to C(10,N-10)/C(20,N).
    # It is neither uniform N nor a product of independent Bernoulli draws.
    history = [event(c, (0, 0), "success") for c in range(1, 11)]
    worlds = sample_worlds(history, count=1800, seed=178)
    counts = Counter(len(w.sources) for w in worlds)
    weights = {n: math.comb(10, n-10)/math.comb(20, n) for n in range(10, 17)}
    z = sum(weights.values())
    expected_mean = sum(n*w/z for n, w in weights.items())
    observed_mean = sum(n*c for n, c in counts.items())/len(worlds)
    assert observed_mean == pytest.approx(expected_mean, abs=0.075)
    for n, weight in weights.items():
        p = weight/z
        assert abs(counts[n]-len(worlds)*p) < 5*math.sqrt(len(worlds)*p*(1-p)) + 3
    assert all(set(range(1, 11)) <= {s.channel for s in w.sources} for w in worlds)


def test_sixteen_discovered_channels_exclude_extras_including_cleared_sources():
    history = [event(c, (0, 0), "near") for c in range(1, 17)]
    history += [event(c, (0, 0), "success") for c in range(1, 9)]
    for world in sample_worlds(history, count=5, seed=781):
        assert {s.channel for s in world.sources} == set(range(1, 17))
        assert_history_compatible(world, history)


def test_same_world_shares_fixed_future_error_across_candidate_action_orders():
    policy = active_policy()
    policy.measure((400, 300), 1, "cover")
    world = sample_worlds(policy.history, count=1, seed=143)[0]
    first = make_branch(world, policy.client.state, policy.history)
    second = make_branch(world, policy.client.state, policy.history)
    p, q = (500, 350), (320, 500)
    a = first.measure(p, 1)
    first.measure(q, 1)
    second.measure(q, 1)
    b = second.measure(p, 1)
    c = first.measure(p, 1)
    assert a["measure_result"] == b["measure_result"] == c["measure_result"] == "direction"
    assert a["svd_deg"] == b["svd_deg"] == c["svd_deg"]
    assert first.measure((400, 300), 1)["measure_result"] == "near"
    assert second.measure((400, 300), 1)["measure_result"] == "near"


def test_historical_bearings_and_public_cost_state_are_preserved():
    policy = active_policy()
    policy.measure((0, 0), 1, "cover")
    policy.measure((900, 0), 1, "localize")
    policy.measure((900, 0), 20, "cover")
    before = policy.client.state.snapshot()
    world = sample_worlds(policy.history, count=1, seed=817)[0]
    branch = make_branch(world, policy.client.state, policy.history)
    assert branch.state.snapshot() | {"real_deadline": None} == before | {"real_deadline": None}
    for row in policy.history:
        result = branch.measure(row["position"], row["channel"])
        assert result["measure_result"] == row["result"]
        if row["result"] == "direction":
            assert result["svd_deg"] == row["bearing_deg"]
    assert policy.client.state.snapshot() == before
    assert not hasattr(branch, "evaluation")
    assert not hasattr(branch, "scenario")
    assert not hasattr(branch, "sources")


def test_sampling_timeout_propagates_without_changing_policy_evidence():
    policy = active_policy()
    policy.measure((0, 0), 20, "cover")
    before = copy.deepcopy(policy.history)
    with pytest.raises(BeliefSamplingTimeout):
        sample_worlds(policy.history, count=2, seed=0, deadline=time.perf_counter()-1)
    assert policy.history == before
    assert policy.channels[20].status == "unknown"
    assert 20 in policy.unknown()


def test_unlucky_samples_raise_instead_of_certifying_absence(monkeypatch):
    import strategies.q3_belief as implementation
    policy = active_policy()
    policy.measure((0, 0), 20, "cover")
    monkeypatch.setattr(implementation, "_prior_position", lambda rng: (0.0, 0.0))
    with pytest.raises(BeliefSamplingError, match="exhausted"):
        sample_worlds(policy.history, count=2, seed=0)
    assert policy.channels[20].status == "unknown"
    assert not policy.certified


def test_sampling_deadline_interrupts_work_after_it_begins(monkeypatch):
    import strategies.q3_belief as implementation
    ticks = 0

    def tick():
        nonlocal ticks
        ticks += 1
        return float(ticks)

    monkeypatch.setattr(implementation.time, "perf_counter", tick)
    with pytest.raises(BeliefSamplingTimeout):
        sample_worlds([event(1, (0, 0), "direction", 0)], count=2, seed=0, deadline=12.0)
    assert ticks == 12
