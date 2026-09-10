"""Independent observation consistency and population checks for Q3 proposals."""

from collections import Counter
import math
import time

import pytest

from planning.coverage import omni_coverage_points
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from strategies import run_search
from strategies.q3_belief import BeliefSamplingError, BeliefSamplingTimeout, sample_worlds
from tests.test_strategy import ObservationOnlyClient


def event(channel, position, result, bearing=None):
    action = "clear" if result in ("success", "no_target_in_range") else "measure"
    item = dict(action=action, channel=channel, position=list(position), result=result)
    if bearing is not None:
        item["bearing_deg"] = bearing
    return item


def assert_compatible(world, history):
    assert world.problem == 3
    assert 10 <= len(world.sources) <= 16
    sources = {source.channel: source for source in world.sources}
    assert len(sources) == len(world.sources)
    assert all(source.orientation_deg is None for source in sources.values())
    assert all(math.hypot(source.x, source.y) <= 1800 for source in sources.values())
    assert all(1000 <= source.reception_radius_m <= 1500 for source in sources.values())
    cleared = set()
    for record in history:
        source = sources.get(record["channel"])
        alive = source is not None and record["channel"] not in cleared
        distance = math.dist((source.x, source.y), record["position"]) if alive else math.inf
        if record["action"] == "clear":
            success = alive and distance <= 20
            assert success == (record["result"] == "success")
            if success:
                cleared.add(record["channel"])
        elif record["result"] == "no_signal":
            assert not alive or distance > source.reception_radius_m
        elif record["result"] == "near":
            assert alive and distance <= 5 and distance <= source.reception_radius_m
        else:
            assert alive and 5 < distance <= source.reception_radius_m
            x, y = record["position"]
            bearing = math.degrees(math.atan2(source.y - y, source.x - x)) % 360
            assert abs((bearing - record["bearing_deg"] + 180) % 360 - 180) <= 1.005 + 1e-9


def test_empty_history_retains_declared_prior_and_is_reproducible():
    worlds = sample_worlds([], count=700, seed=128)
    assert worlds == sample_worlds([], count=700, seed=128)
    counts = Counter(len(world.sources) for world in worlds)
    # The uniform N prior must not accidentally become uniform over all
    # possible subsets (which would heavily prefer counts close to ten).
    assert set(counts) == set(range(10, 17))
    assert all(abs(counts[n] - 100) < 35 for n in range(10, 17))
    all_sources = [source for world in worlds for source in world.sources]
    assert sum(source.reception_radius_m for source in all_sources) / len(all_sources) == pytest.approx(1250, abs=8)
    assert sum(source.x**2 + source.y**2 for source in all_sources) / len(all_sources) == pytest.approx(1800**2 / 2, rel=0.025)
    for world in worlds:
        assert_compatible(world, [])
        assert "Approximate" in world.description


def test_directions_near_negative_failed_clear_and_post_clear_history_agree():
    history = [
        event(1, (0, 0), "direction", 36.87),
        event(1, (900, 0), "direction", 149.04),
        event(1, (1700, 300), "no_signal"),
        event(1, (430, 300), "no_target_in_range"),
        event(1, (400, 300), "near"),
        event(1, (402, 300), "success"),
        event(1, (400, 300), "no_signal"),
        event(1, (400, 300), "no_target_in_range"),
    ]
    for world in sample_worlds(history, count=30, seed=11):
        assert_compatible(world, history)
        assert 1 in {source.channel for source in world.sources}


def test_undetected_channel_can_still_exist_after_no_signal():
    history = [event(1, (0, 0), "no_signal")]
    worlds = sample_worlds(history, count=80, seed=17)
    present = [any(source.channel == 1 for source in world.sources) for world in worlds]
    assert any(present)
    assert not all(present)
    for world in worlds:
        assert_compatible(world, history)


def test_negative_evidence_changes_channel_probability_without_deleting_it():
    history = [event(channel, (0, 0), "success") for channel in range(1, 11)]
    history.append(event(11, (0, 0), "no_signal"))
    worlds = sample_worlds(history, count=800, seed=203)
    frequency = Counter(source.channel for world in worlds for source in world.sources)
    assert 0 < frequency[11] < frequency[12] - 80
    assert all(frequency[channel] == 800 for channel in range(1, 11))
    for world in worlds:
        assert_compatible(world, history)


def test_visible_source_radius_is_weighted_by_supported_position_area():
    # For a direction from the arena centre, flat angular-band likelihood and
    # the stated prior imply f(R | direction) proportional to R**2 - 5**2.
    # Uniformly choosing proposal positions without radius-length weighting
    # produces a different mean and is not this conditional research prior.
    worlds = sample_worlds([event(1, (0, 0), "direction", 0)], count=2000, seed=53)
    radii = [next(source.reception_radius_m for source in world.sources if source.channel == 1)
             for world in worlds]
    numerator = (1500**4 - 1000**4) / 4 - 25 * (1500**2 - 1000**2) / 2
    denominator = (1500**3 - 1000**3) / 3 - 25 * (1500 - 1000)
    assert sum(radii) / len(radii) == pytest.approx(numerator / denominator, abs=15)


def test_exhausted_optional_pool_does_not_silently_claim_channel_absent(monkeypatch):
    import strategies.q3_belief as belief
    # Force an unlucky proposal stream, although a source outside the origin's
    # reception disk remains geometrically possible. Safe fallback is required.
    monkeypatch.setattr(belief, "_prior_position", lambda rng: (0.0, 0.0))
    with pytest.raises(BeliefSamplingError, match="exhausted"):
        sample_worlds([event(1, (0, 0), "no_signal")], count=2, seed=0)


def test_repeated_measurements_add_no_independent_evidence():
    history = [event(1, (0, 0), "direction", 359.8), event(2, (0, 0), "no_signal")]
    repeated = history + [dict(history[0]) for _ in range(20)] + [dict(history[1]) for _ in range(20)]
    assert sample_worlds(history, count=12, seed=52) == sample_worlds(repeated, count=12, seed=52)


def test_sixteen_required_sources_leave_no_uncertain_extra_sources():
    history = [event(channel, (0, 0), "near") for channel in range(1, 17)]
    history += [event(channel, (0, 0), "success") for channel in range(1, 17)]
    history += [event(channel, (0, 0), "no_signal") for channel in range(1, 21)]
    for world in sample_worlds(history, count=8, seed=3):
        assert {source.channel for source in world.sources} == set(range(1, 17))
        assert_compatible(world, history)


def test_full_negative_cover_can_certify_absence_without_pool_hits():
    history = [event(channel, (0, 0), "success") for channel in range(1, 11)]
    history += [event(channel, (point.x, point.y), "no_signal")
                for point in omni_coverage_points() for channel in range(11, 21)]
    for world in sample_worlds(history, count=3, seed=5):
        assert {source.channel for source in world.sources} == set(range(1, 11))
        assert_compatible(world, history)


def test_all_silent_cover_conflicts_with_public_minimum_source_count():
    history = [event(channel, (point.x, point.y), "no_signal")
               for point in omni_coverage_points() for channel in range(1, 21)]
    with pytest.raises(BeliefSamplingError, match="10..16"):
        sample_worlds(history, count=2, seed=0)


@pytest.mark.parametrize("history", [
    [event(1, (0, 0), "direction", 0), event(1, (0, 0), "direction", 0.01)],
    [event(1, (0, 0), "near"), event(1, (0, 0), "no_signal")],
    [event(1, (0, 0), "success"), event(1, (0, 0), "direction", 0)],
    [event(1, (0, 0), "success"), event(1, (0, 0), "success")],
    [event(1, (0, 0), "near"), event(1, (100, 0), "near")],
    [event(1, (0, 0), "near"), event(1, (0, 0), "no_target_in_range")],
    [event(1, (2000, 0), "near")],
    [event(channel, (0, 0), "near") for channel in range(1, 18)],
])
def test_contradictory_history_fails_without_inventing_a_world(history):
    with pytest.raises(BeliefSamplingError):
        sample_worlds(history, count=2, seed=0)


@pytest.mark.parametrize("history", [
    None, "history", [None], [{}],
    [event(True, (0, 0), "near")], [event(21, (0, 0), "near")],
    [event(1, (float("nan"), 0), "near")],
    [event(1, (0, 0), "direction")], [event(1, (0, 0), "direction", 360)],
    [dict(action="measure", position=[0], channel=1, result="near")],
    [event(1, (0, 0), "unknown")],
])
def test_invalid_history_is_rejected(history):
    with pytest.raises(BeliefSamplingError):
        sample_worlds(history, count=2, seed=0)


@pytest.mark.parametrize("kwargs", [
    dict(count=0, seed=0), dict(count=True, seed=0), dict(count=1.5, seed=0),
    dict(count=2, seed=True), dict(count=2, seed=0.5),
])
def test_invalid_sampling_options_are_rejected(kwargs):
    with pytest.raises(BeliefSamplingError):
        sample_worlds([], **kwargs)


def test_expired_deadline_raises_specific_cancellation():
    with pytest.raises(BeliefSamplingTimeout, match="deadline"):
        sample_worlds([], count=2, seed=0, deadline=time.perf_counter() - 1)


@pytest.mark.parametrize("history,expire_after", [
    ([event(1, (0, 0), "direction", 0)], 12),
    ([event(1, (0, 0), "no_signal")], 12),
    ([event(1, (point.x, point.y), "no_signal") for point in omni_coverage_points()], 28),
])
def test_deadline_can_cancel_geometry_sampling_or_absence_proof(monkeypatch, history, expire_after):
    import strategies.q3_belief as belief
    ticks = 0

    def ticking_clock():
        nonlocal ticks
        ticks += 1
        return float(ticks)

    monkeypatch.setattr(belief.time, "perf_counter", ticking_clock)
    with pytest.raises(BeliefSamplingTimeout):
        sample_worlds(history, count=2, seed=0, deadline=float(expire_after))
    assert ticks == expire_after


@pytest.mark.parametrize("deadline", [True, "tomorrow", float("inf"), float("nan")])
def test_invalid_deadlines_are_rejected(deadline):
    with pytest.raises(BeliefSamplingError):
        sample_worlds([], count=2, seed=0, deadline=deadline)


@pytest.mark.parametrize("scenario", difficult_scenarios(3) + [random_scenario(3, seed) for seed in range(5)],
                         ids=lambda scenario: scenario.case_id)
def test_real_observation_prefixes_admit_compatible_samples(scenario):
    # Only the local research engine is used. The sampler receives a public
    # action trace, never the scenario or the evaluator's source truth.
    sim = LocalResearchSimulator(scenario)
    report = run_search(ObservationOnlyClient(sim.client()), variant="efficient")
    assert report.completion_certified_under_model
    lengths = {min(20, len(report.action_history)), len(report.action_history) // 2,
               len(report.action_history)}
    for length in sorted(lengths):
        history = report.action_history[:length]
        for world in sample_worlds(history, count=2, seed=1729 + length):
            assert_compatible(world, history)
