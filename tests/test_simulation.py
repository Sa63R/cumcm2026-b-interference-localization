"""Independent physical boundary checks; no fake HTTP fixture is reused."""

import math

import pytest

from simulation import LocalResearchSimulator, Scenario, Source, difficult_scenarios, random_scenario
from simulator_client.errors import DeadlineExceeded, SessionError


def make_case(source=None, *, problem=3, error_mode="zero"):
    first = source or Source(1, 0, 0, 1000)
    fillers = tuple(Source(channel, -1500, 0, 1000) for channel in range(2, 11))
    return Scenario("unit-physical", problem, 17, (first,) + fillers, error_mode)


def new_client(case=None, **kwargs):
    sim = LocalResearchSimulator(case or make_case(), **kwargs)
    client = sim.client()
    client.enter()
    return sim, client


@pytest.mark.parametrize("distance,result", [(1000, "direction"), (1000.00001, "no_signal"),
                                               (5, "near"), (5.00001, "direction"), (0, "near")])
def test_range_and_near_boundaries(distance, result):
    _, client = new_client()
    response = client.measure((distance, 0), 1)
    assert response["measure_result"] == result
    assert ("svd_deg" in response) == (result == "direction")


@pytest.mark.parametrize("position,result", [((0, 100), "direction"), ((0, -100), "direction"),
                                               ((100, 0), "direction"), ((-1e-6, 100), "no_signal"),
                                               ((-1, 0), "no_signal"), ((0, 0), "near")])
def test_directional_half_plane_includes_both_edges(position, result):
    _, client = new_client(make_case(Source(1, 0, 0, 1000, 0), problem=4))
    assert client.measure(position, 1)["measure_result"] == result


def test_near_does_not_override_directional_shadow():
    _, client = new_client(make_case(Source(1, 0, 0, 1000, 0), problem=4))
    assert client.measure((-4, 0), 1)["measure_result"] == "no_signal"
    assert client.measure((4, 0), 1)["measure_result"] == "near"


@pytest.mark.parametrize("distance,result", [(20, "success"), (20.00001, "no_target_in_range")])
def test_clear_boundary_and_shadow_independence(distance, result):
    _, client = new_client(make_case(Source(1, 0, 0, 1000, 0), problem=4))
    assert client.clear((-distance, 0), 1)["clear_result"] == result


def test_removed_source_cannot_be_detected_or_removed_twice():
    _, client = new_client()
    assert client.clear((0, 0), 1)["clear_result"] == "success"
    assert client.measure((100, 0), 1)["measure_result"] == "no_signal"
    assert client.clear((0, 0), 1)["clear_result"] == "no_target_in_range"
    assert client.state.cleared_count == 1


def test_published_199_second_timing_example_and_channel_state():
    sim, client = new_client()
    assert client.measure((300, 400), 1)["virtual_time_s"] == 105
    assert client.measure((300, 400), 2)["virtual_time_s"] == 111
    assert client.clear((300, 0), 3)["virtual_time_s"] == 194
    assert client.state.current_channel == 2
    assert client.measure((300, 0), 2)["virtual_time_s"] == 199
    client.exit()
    evaluation = sim.evaluation()
    assert evaluation["time_breakdown_s"] == {
        "movement_s": 180, "switching_s": 1, "detection_s": 15, "optical_s": 3, "removal_s": 0,
    }
    assert evaluation["virtual_time_s"] == 199
    assert evaluation["mean_time_per_cleared_s"] is None


def test_successful_clear_costs_five_without_switch():
    sim, client = new_client()
    client.measure((0, 0), 2)
    before = client.state.virtual_time_s
    client.clear((0, 0), 1)
    assert client.state.virtual_time_s - before == 5
    assert client.state.current_channel == 2
    client.exit()
    evaluation = sim.evaluation()
    assert evaluation["cleared_fraction"] == .1
    assert evaluation["time_breakdown_s"]["removal_s"] == 2


@pytest.mark.parametrize("error_mode", ["uniform", "positive_extreme", "negative_extreme", "alternating_extreme"])
def test_error_is_location_fixed_bounded_and_centidegree(error_mode):
    _, client = new_client(make_case(error_mode=error_mode))
    for index in range(50):
        angle = index * 7.137
        position = (500 * math.cos(math.radians(angle)), 500 * math.sin(math.radians(angle)))
        first = client.measure(position, 1)["svd_deg"]
        repeated = client.measure(position, 1)["svd_deg"]
        true_bearing = (angle + 180) % 360
        error = (first - true_bearing + 180) % 360 - 180
        assert first == repeated
        assert 0 <= first < 360
        assert abs(error) <= 1 + 1e-9
        assert first * 100 == pytest.approx(round(first * 100))


def test_same_seed_same_observations_without_timestamps():
    sims = [LocalResearchSimulator(random_scenario(4, 19)) for _ in range(2)]
    sequences = []
    for sim in sims:
        client = sim.client()
        client.enter()
        sequence = [client.measure((100, 200), channel) for channel in range(1, 21)]
        sequences.append([{k: v for k, v in item.items() if k != "real_timestamp_ms"} for item in sequence])
        client.exit()
    assert sequences[0] == sequences[1]


def test_truth_is_post_run_and_absent_from_client_responses():
    sim = LocalResearchSimulator(make_case())
    with pytest.raises(SessionError):
        sim.evaluation()
    client = sim.client()
    response = client.enter()
    assert not hasattr(client, "scenario")
    assert not hasattr(client, "sources")
    assert "source_total" not in response
    with pytest.raises(SessionError):
        sim.evaluation()
    client.exit()
    assert sim.evaluation()["source_total"] == 10


def test_microsecond_rounding_and_independent_component_total():
    sim, client = new_client()
    for _ in range(3):
        client.measure((1, 1), 1)
        client.measure((0, 0), 1)
    client.exit()
    evaluation = sim.evaluation()
    expected_movement = 6 * round(math.sqrt(2) / 5, 6)
    assert evaluation["time_breakdown_s"]["movement_s"] == pytest.approx(expected_movement)
    assert sum(evaluation["time_breakdown_s"].values()) == evaluation["virtual_time_s"]


def test_registered_action_may_finish_over_virtual_limit_but_next_is_blocked():
    sim, client = new_client(max_virtual_duration_s=5)
    assert client.measure((10, 0), 1)["virtual_time_s"] == 7
    with pytest.raises(DeadlineExceeded):
        client.measure((10, 0), 1)
    sim.finish_for_evaluation("virtual_budget")
    assert sim.evaluation()["virtual_time_s"] == 7


def test_real_budget_is_enforced():
    now = [0.0]
    sim, client = new_client(max_real_duration_s=10, clock=lambda: now[0])
    now[0] = 11
    with pytest.raises(DeadlineExceeded):
        client.measure((0, 0), 1)
    sim.finish_for_evaluation("real_deadline")
    assert sim.evaluation()["measurement_count"] == 0


def test_generated_scenario_rules_and_difficult_cases():
    for problem in (3, 4):
        cases = [random_scenario(problem, seed) for seed in range(20)] + difficult_scenarios(problem)
        assert len(cases) == 27
        for case in cases:
            assert 10 <= len(case.sources) <= 16
            assert len({source.channel for source in case.sources}) == len(case.sources)
            assert all(math.hypot(s.x, s.y) <= 1800 + 1e-9 for s in case.sources)
            assert all(1000 <= s.reception_radius_m <= 1500 for s in case.sources)
            assert case == (random_scenario(problem, case.seed) if "random" in case.case_id else case)
        if problem == 4:
            outward = cases[20]
            sim = LocalResearchSimulator(outward)
            client = sim.client()
            client.enter()
            assert all(client.measure((0, 0), source.channel)["measure_result"] == "no_signal"
                       for source in outward.sources if source.orientation_deg is not None)


def test_duplicate_channels_and_invalid_counts_are_rejected():
    valid = make_case()
    with pytest.raises(ValueError):
        Scenario("bad", 3, 1, (valid.sources[0],) * 10)
    with pytest.raises(ValueError):
        Scenario("bad", 3, 1, valid.sources[:9])
