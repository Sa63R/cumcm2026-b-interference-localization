"""Analytic and randomized checks; no simulator sessions or scenario seeds."""

from fractions import Fraction
import math
import random

import pytest

from localization.omni import OmniCandidateRegion
from planning.history_silence import certify_history_silence
from planning.silence_certificate import certify_silence


def _region(vertices, negatives=()):
    region = OmniCandidateRegion().observe((0, 0), 0)
    region.vertices = tuple(vertices)
    region._circle = None
    # Analytic fixtures specify the already constructed conservative polygon.
    region.no_signal_positions = list(negatives)
    return region


def test_historical_radius_cap_can_certify_below_public_cap():
    region = _region(((0., -10.), (20., -10.), (20., 10.), (0., 10.)), [(1100., 0.)])
    assert certify_silence(region, (1200., 0.)) is None
    certificate = certify_history_silence(region, (1200., 0.))
    assert certificate["method"] == "history_negative_bisector"
    assert certificate["witness_no_signal_position"] == [1100., 0.]
    assert certificate["signed_bisector_distance_lower_m"] == 1130
    assert certificate["squared_distance_difference_lower_m2"] == 226000
    assert certificate["margin_m"] == 1e-5


def test_existing_certificate_has_priority_and_identical_fields():
    region = _region(((0., 0.), (20., 0.), (20., 20.), (0., 20.)), [(1100., 0.)])
    assert certify_history_silence(region, (1800., 0.)) == certify_silence(region, (1800., 0.))


@pytest.mark.parametrize("vertices", [((0., 0.),), ((0., -10.), (0., 10.)),
                                      ((0., -10.), (20., -10.), (20., 10.), (0., 10.))])
def test_degenerate_regions_and_reversed_query(vertices):
    region = _region(vertices, [(1100., 0.)])
    assert certify_history_silence(region, (1200., 0.))
    assert certify_history_silence(region, (1000., 0.)) is None
    assert certify_history_silence(region, (1100., 0.)) is None


def test_empty_unknown_and_missing_history_cannot_certify():
    assert certify_history_silence(_region((), [(1100., 0.)]), (1200., 0.)) is None
    unknown = OmniCandidateRegion()
    unknown.no_signal_positions = [(1100., 0.)]
    assert certify_history_silence(unknown, (1200., 0.)) is None
    assert certify_history_silence(_region(((0., 0.),)), (1200., 0.)) is None


def test_all_vertices_not_only_center_or_nearest_vertex():
    # The polygon crosses the n/q perpendicular bisector x=1150.
    region = _region(((900., -20.), (1200., -20.), (1200., 20.), (900., 20.)), [(1100., 0.)])
    assert certify_history_silence(region, (1200., 0.)) is None
    tangent = _region(((1100., 0.), (1150., -20.), (1150., 20.)), [(1100., 0.)])
    assert certify_history_silence(tangent, (1200., 0.)) is None


def test_strict_margin_and_large_coordinate_rounding():
    # Bisector x=10000. The margin here is exactly representable in binary64.
    margin = 2.**-20
    for offset, expected in [(0., False), (margin, False), (2*margin, True)]:
        region = _region(((10000.-offset, 10.),), [(9999., 0.)])
        assert bool(certify_history_silence(region, (10001., 0.), margin)) is expected


@pytest.mark.parametrize("margin", [0, -1, float("nan"), float("inf"), True, "bad"])
def test_invalid_margin(margin):
    with pytest.raises(ValueError):
        certify_history_silence(_region(((0., 0.),)), (1200., 0.), margin)


def test_random_convex_regions_have_sound_complete_polygon_certificates():
    rng = random.Random(428613)  # Geometry test RNG only, never a scenario seed.
    accepted = rejected = 0
    for _ in range(200):
        cx, cy = rng.uniform(-500, 500), rng.uniform(-500, 500)
        radius = rng.uniform(1, 80)
        vertices = tuple((cx+radius*math.cos(a), cy+radius*math.sin(a))
                         for a in (0, math.pi/2, math.pi, 3*math.pi/2))
        angle = rng.uniform(-math.pi, math.pi)
        witness = (cx+1100*math.cos(angle), cy+1100*math.sin(angle))
        step = rng.uniform(-150, 150)
        query = (witness[0]+step*math.cos(angle), witness[1]+step*math.sin(angle))
        region = _region(vertices, [witness])
        certificate = certify_history_silence(region, query)
        if certificate is None:
            rejected += 1
            continue
        accepted += 1
        assert certificate["method"] == "history_negative_bisector"
        reported = Fraction(certificate["signed_bisector_distance_lower_m"])
        norm_upper = Fraction(certificate["query_witness_distance_upper_m"])
        dx, dy = Fraction(query[0])-Fraction(witness[0]), Fraction(query[1])-Fraction(witness[1])
        assert norm_upper*norm_upper >= dx*dx+dy*dy
        for x, y in vertices:
            half = dx*((Fraction(query[0])+Fraction(witness[0]))/2-Fraction(x))
            half += dy*((Fraction(query[1])+Fraction(witness[1]))/2-Fraction(y))
            assert half >= reported*norm_upper > 0
        # Test interior convex combinations as well as exact vertex bounds.
        for _ in range(10):
            weights = [rng.random() for _ in vertices]
            total = sum(weights)
            x = sum(w*v[0] for w, v in zip(weights, vertices))/total
            y = sum(w*v[1] for w, v in zip(weights, vertices))/total
            assert math.dist((x, y), query) > math.dist((x, y), witness)
            # Every physically compatible R remains below the query distance.
            upper = min(1500., math.dist((x, y), witness))
            if upper > 1000:
                reception_radius = (1000+upper)/2
                assert math.dist((x, y), query) > reception_radius
    assert accepted > 50 and rejected > 50
