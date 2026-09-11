import math
from dataclasses import replace

import pytest

from geometry import distance
from simulation.cases import random_scenario, difficult_scenarios
from simulation.engine import LocalResearchSimulator
from strategies.q3_fresh import (COVER, FreshQ3, covered, nearest_safe_clear,
                                 open_route, run_fresh)


def test_continuous_cover_and_missing_sector():
    assert covered(COVER)
    for k in range(1, 7):
        assert not covered(COVER[:k] + COVER[k + 1:])
    # Two points do not cover a disk merely because sampled endpoints do.
    assert not covered(((1000, 0), (-1000, 0)))


def test_per_channel_evidence_and_sixteen_limit():
    p = FreshQ3(None)
    p.channels[1].negatives.extend(COVER)
    assert 2 in p.unknown()
    for c in range(1, 13):
        p.channels[c].status = "cleared"
    assert len(p.unknown()) == 8
    for c in range(13, 17):
        p.channels[c].status = "cleared"
    assert p.unknown() == []


def test_nearest_clear_and_equilateral_counterexample():
    # Diameter <40 alone is insufficient: circumradius of this triangle >20.
    triangle = [(22 * math.cos(i * 2 * math.pi / 3),
                 22 * math.sin(i * 2 * math.pi / 3)) for i in range(3)]
    assert nearest_safe_clear(triangle, (100, 0)) is None
    q = nearest_safe_clear([(-10, 0), (10, 0)], (100, 0))
    assert q[0] == pytest.approx(10, abs=2e-5)
    assert all(distance(q, v) < 20 for v in [(-10, 0), (10, 0)])


def test_half_step_contracts_extreme_bearings():
    radius, delta = 1500, math.radians(1.005)
    step = radius / (2 * math.cos(delta))
    for d in (0, 0.2, 400, 750, 1499, 1500):
        for error in (-delta, 0, delta):
            assert math.hypot(d * math.cos(error) - step, d * math.sin(error)) <= step + 1e-9
    assert radius / (2 * math.cos(delta)) ** 7 < 12


def test_open_route_does_not_force_return():
    route = open_route([("source", 1, (100, 0)), ("source", 2, (200, 0))], (0, 0))
    assert [x[1] for x in route] == [1, 2]


@pytest.mark.parametrize("version", ["v0", "v1"])
@pytest.mark.parametrize("case", [random_scenario(3, 910001), difficult_scenarios(3)[3]])
def test_complete_and_time_accounting(version, case):
    engine = LocalResearchSimulator(case)
    result = run_fresh(engine.client(), version)
    metrics = engine.evaluation()
    assert result["completion_certified"] and metrics["all_cleared"]
    assert metrics["failed_clear_count"] == 0
    assert sum(metrics["time_breakdown_s"].values()) == pytest.approx(metrics["virtual_time_s"])
    measurements = [(x["channel"], tuple(x["position"])) for x in result["action_history"]
                    if x["action"] == "measure"]
    assert len(set(measurements)) == len(measurements)


def test_identical_error_at_same_point_and_clear_keeps_channel():
    engine = LocalResearchSimulator(random_scenario(3, 910005))
    client = engine.client()
    client.enter()
    source = engine.scenario.sources[0]  # Harness-only setup of this API test.
    p = (source.x + 100, source.y)
    a = client.measure(p, source.channel)
    b = client.measure(p, source.channel)
    assert a["svd_deg"] == b["svd_deg"]
    old_channel = client.state.current_channel
    client.clear((source.x, source.y), source.channel)
    assert client.state.current_channel == old_channel
    client.exit()
