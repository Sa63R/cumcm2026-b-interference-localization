"""Pure geometry/observation tests; no simulator or HTTP is constructed."""

import copy
import json
import math
import random

import pytest

from localization import CandidateRegion
from planning.positive_hull_probe import choose_positive_hull_probe


def bearing(p, source):
    return math.degrees(math.atan2(source[1] - p[1], source[0] - p[0])) % 360


def make_region(source, positives):
    region = CandidateRegion()
    for p in positives:
        region.observe(p, bearing(p, source))
    return region


def verify_witness(candidate, positives):
    weights = candidate['weights']
    assert weights and all(w['weight'] > 0 and tuple(w['position']) in positives for w in weights)
    assert math.fsum(w['weight'] for w in weights) == pytest.approx(1, abs=2e-15)
    reconstructed = [math.fsum(w['weight'] * w['position'][j] for w in weights) for j in (0, 1)]
    assert candidate['position'] == pytest.approx(reconstructed, abs=1e-9)


@pytest.mark.parametrize('radius', [1000.0, 1500.0])
@pytest.mark.parametrize('angle', [0.0, 90.0, 180.0, 270.0, 359.999])
def test_extreme_halfplane_edges_and_fixed_radius_are_preserved(radius, angle):
    source = (100.0, -30.0)
    theta = math.radians(angle)
    n = (math.cos(theta), math.sin(theta))
    positives = [(source[0] + radius * math.cos(theta + delta),
                  source[1] + radius * math.sin(theta + delta))
                 for delta in (-math.pi / 2, -0.5, 0.5, math.pi / 2)]
    region = make_region(source, positives)
    q, log = choose_positive_hull_probe(region, (0, 0), positives)
    assert q is not None and len(log['candidates']) <= 24
    for candidate in log['candidates']:
        verify_witness(candidate, positives)
        p = candidate['position']
        assert math.dist(p, source) <= radius + 1e-8
        assert n[0] * (p[0] - source[0]) + n[1] * (p[1] - source[1]) >= -1e-8


def test_random_fixed_radius_halfdisks_have_no_convex_hull_counterexample():
    rng = random.Random(81031)
    for _ in range(20):
        source = (rng.uniform(-700, 700), rng.uniform(-700, 700))
        theta, radius = rng.uniform(0, 2 * math.pi), rng.uniform(1000, 1500)
        positives = []
        for _ in range(5):
            angle, distance = theta + rng.uniform(-math.pi / 2, math.pi / 2), rng.uniform(20, radius)
            positives.append((source[0] + distance * math.cos(angle), source[1] + distance * math.sin(angle)))
        region = make_region(source, positives)
        _, log = choose_positive_hull_probe(region, positives[-1], positives, 12)
        assert log['candidates']
        for candidate in log['candidates']:
            verify_witness(candidate, positives)
            q = candidate['position']
            assert math.dist(q, source) <= radius + 1e-8
            assert math.cos(theta) * (q[0] - source[0]) + math.sin(theta) * (q[1] - source[1]) >= -1e-8


def test_one_positive_has_no_new_guaranteed_hull_point_even_with_negative_stations():
    region = make_region((1000, 0), [(0, 0)])
    q, log = choose_positive_hull_probe(region, (100, 100), [(0, 0), (0, 100), (0, -100)])
    assert q is None and not log['candidates']


def test_duplicate_positives_do_not_create_new_hull():
    region = make_region((1000, 0), [(0, 0), (0, 0)])
    assert choose_positive_hull_probe(region, (0, 0), [])[0] is None


def test_previously_measured_points_are_excluded_and_never_supply_witnesses():
    positives = [(0, 0), (0, 600)]
    region = make_region((800, 300), positives)
    observed = positives + [(0, 150), (0, 300), (0, 450), (500, -200)]
    q, log = choose_positive_hull_probe(region, (0, 0), observed)
    assert q is None and log['reason'] == 'no_fresh_convex_combination'


@pytest.mark.parametrize('cached', [False, True])
def test_live_region_including_mec_cache_is_unchanged_and_scores_repeat(cached):
    positives = [(0, 0), (0, 500), (300, 600)]
    region = make_region((1100, 250), positives)
    if cached:
        region.enclosing_disk()
    before = copy.deepcopy(region.__dict__)
    old_cache = region._circle
    first = choose_positive_hull_probe(region, (40, -10), positives)
    second = choose_positive_hull_probe(region, (40, -10), positives)
    assert first == second and first[0] is not None
    assert region.__dict__ == before and region._circle is old_cache
    json.dumps(first[1], allow_nan=False)


def test_hull_reception_does_not_incorrectly_require_universal_1000m():
    source, positives = (0, 0), [(1200, -250), (1200, 250)]
    region = make_region(source, positives)
    q, log = choose_positive_hull_probe(region, positives[0], positives)
    assert q is not None and math.hypot(q.x, q.y) > 1000
    assert all(math.dist(source, c['position']) > 1000 for c in log['candidates'])
    assert all(math.dist(source, c['position']) < 1500 for c in log['candidates'])


def test_candidate_estimate_interpolation_is_not_used_as_a_reception_proof():
    # A legal +/-1 degree source estimate can be on the wrong side of the
    # actual emission edge even halfway from a genuine positive station.
    source, observer, estimate = (100, 0), (0, 0), (100, -1)
    n, midpoint = (-0.005, 1), (50, -0.5)
    assert abs(bearing(observer, estimate) - 360) < 1
    assert sum(n[j] * (observer[j] - source[j]) for j in (0, 1)) > 0
    assert sum(n[j] * (midpoint[j] - source[j]) for j in (0, 1)) < 0
    region = make_region(source, [observer])
    assert choose_positive_hull_probe(region, observer, [observer])[0] is None


@pytest.mark.parametrize('limit', [1, 2, 7, 24])
def test_candidate_budget_and_finite_nominal_score_decomposition(limit):
    source = (900, 0)
    positives = [(0, 0), (0, 500), (200, 600), (400, 650)]
    region = make_region(source, positives)
    _, log = choose_positive_hull_probe(region, (0, 0), positives, limit)
    assert 1 <= len(log['candidates']) <= limit and 1 <= log['hypotheses'] <= 5
    for candidate in log['candidates']:
        if candidate['valid_nominal_score']:
            outcomes = candidate['nominal_outcomes']
            assert candidate['score_s'] == pytest.approx(candidate['movement_s'] + 5 +
                math.fsum(o['tail_proxy_s'] for o in outcomes) / len(outcomes))


def test_all_nominally_clearable_priority_and_coordinate_ties_are_explicit():
    positives = [(0, 0), (0, 600), (300, 650)]
    region = make_region((800, 250), positives)
    q, log = choose_positive_hull_probe(region, (0, 0), positives)
    eligible = [c for c in log['candidates'] if c['valid_nominal_score']]
    expected = min(eligible, key=lambda c: (not c['sampled_all_clearable'], c['score_s'], *c['position']))
    assert [q.x, q.y] == expected['position']


@pytest.mark.parametrize('limit', [0, 25, True, 2.5])
def test_invalid_budget_rejected(limit):
    with pytest.raises(ValueError):
        choose_positive_hull_probe(CandidateRegion(), (0, 0), [], limit)


def test_empty_region_fails_closed():
    region = make_region((500, 0), [(0, 0), (0, 100)])
    region.vertices = ()
    assert choose_positive_hull_probe(region, (0, 0), [])[0] is None
