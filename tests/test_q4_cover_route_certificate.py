import math
from fractions import Fraction

import pytest

from experiments.certify_q4_cover_route import (build_certificate, certify_snapshot,
    euclidean_mst, ideal_edge_classes, pi_interval)


@pytest.fixture(scope="module")
def certificate():
    return build_certificate()


def test_pi_interval_encloses_independent_decimal():
    lo, hi = pi_interval()
    known_lo = Fraction("3.14159265358979323846264338327950288419716939937510")
    known_hi = known_lo + Fraction(1, 10**50)
    assert lo < known_lo < known_hi < hi
    assert hi-lo < Fraction(1, 10**32)


def test_all_edge_classes_have_strict_order_and_complete_multiplicity():
    rows, _, length = ideal_edge_classes()
    assert len(rows) == 20
    assert sum(row["count"] for row in rows) == math.comb(22, 2)
    assert [r["name"] for r in rows[:5]] == ["outer_1", "inner_1", "cross_0", "origin_inner", "cross_1"]
    assert all(a["interval"][1] < b["interval"][0] for a, b in zip(rows, rows[1:]))
    assert float(length[0]) == pytest.approx(17612.4194000923, abs=1e-8)
    assert length[1]-length[0] < Fraction(1, 10**10)


def test_independent_kruskal_square():
    length, edges = euclidean_mst([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert length == 3.
    assert len(edges) == 3


def test_actual_route_attains_tree_and_retains_source_identity(certificate):
    snapshot = certificate["floating_snapshot"]
    assert certificate["passed"]
    assert snapshot["route_indices"][0] == 0
    assert abs(snapshot["route_minus_mst_m"]) < 1e-7
    assert len(snapshot["route_edges"]) == len(snapshot["mst_edges"]) == 21
    assert len(certificate["source_files_sha256"]) == 4
    assert certificate["analytic"]["kruskal_selected_multiplicities"] == {
        "outer_1": 13, "inner_1": 6, "cross_0": 1, "origin_inner": 1}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_start", "wrong_order", "geometry", "nan"])
def test_invalid_snapshot_rejected(certificate, mutation):
    points = [list(p) for p in certificate["floating_snapshot"]["points"]]
    route = list(certificate["floating_snapshot"]["route_indices"])
    if mutation == "missing":
        route.pop()
    elif mutation == "duplicate":
        route[-1] = route[-2]
    elif mutation == "wrong_start":
        route.reverse()
    elif mutation == "wrong_order":
        route[2], route[-1] = route[-1], route[2]
    elif mutation == "geometry":
        points[1][0] += .01
    else:
        points[1][0] = float("nan")
    with pytest.raises(ValueError):
        certify_snapshot(points, route)


def test_reversed_suffix_is_also_optimal_by_symmetry(certificate):
    # Reflect all stations and replay the same indexed path: the ideal class
    # certificate is geometric, not tied to the floating tie-break direction.
    points = [(x, -y) for x, y in certificate["floating_snapshot"]["points"]]
    result = certify_snapshot(points, certificate["floating_snapshot"]["route_indices"])
    assert result["passed"]
