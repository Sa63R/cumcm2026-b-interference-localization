import copy
import math

import pytest

from planning.q4_directional_cover import (certified_cover_points, certify_directional_cover,
                                           concentric_stations, verify_directional_cover_certificate)


def test_one_box_whole_region_certificate_and_replay():
    points = [(-2,-2), (2,-2), (2,2), (-2,2)]
    certificate = certify_directional_cover(points, arena_radius=1., reception_radius=5.)
    assert certificate["passed"] and certificate["visited_cells"] == 1
    assert certificate["covered_leaves"] == 1
    assert verify_directional_cover_certificate(points, certificate)["passed"]


def test_counterexample_is_an_actual_uncovered_direction():
    points = concentric_stations(990.,1958.3,inner_count=8,outer_count=8,inner_phase_deg=22.5)
    result = certify_directional_cover(points)
    assert not result["passed"] and result["status"] == "counterexample"
    witness = result["counterexample"]
    x, normal = witness["source"], witness["normal"]
    assert math.hypot(*x) <= 1800.
    for p in points:
        if math.dist((p.x,p.y),x) <= 1000.:
            assert normal[0]*(p.x-x[0])+normal[1]*(p.y-x[1]) < 0


def test_budget_exhaustion_never_certifies_unprocessed_region():
    points = concentric_stations(950.,1800./math.cos(math.pi/12)+10.,inner_count=12,inner_phase_deg=15.)
    for kwargs in ({"max_cells":1},{"max_depth":0}):
        result = certify_directional_cover(points, **kwargs)
        assert not result["passed"] and result["status"] == "inconclusive"
        with pytest.raises(ValueError):
            verify_directional_cover_certificate(points,result)


def test_compact_profile_full_certificate_and_analytic_mesh_are_consistent():
    route, certificate = certified_cover_points("compact_25")
    assert certificate["passed"] and len(route) == 25
    assert certificate["route_length_m"] == pytest.approx(18014.03099725)
    assert certificate["route_length_m"] < 19000.
    full = certify_directional_cover(route)
    assert verify_directional_cover_certificate(route,full)["passed"]
    mesh = certificate["analytic_certificate"]
    points, triangles = mesh["indexed_stations"], mesh["triangles"]
    # Independent oriented-edge cancellation proves the positive triangle chain
    # has exactly the simple outer-polygon boundary, not holes or duplicate faces.
    edge_balance = {}
    for a,b,c in triangles:
        assert (points[b][0]-points[a][0])*(points[c][1]-points[a][1])-(points[b][1]-points[a][1])*(points[c][0]-points[a][0]) > 0
        for i,j in ((a,b),(b,c),(c,a)):
            assert math.dist(points[i],points[j]) < 1000.-1e-5
            key = min(i,j),max(i,j)
            edge_balance[key] = edge_balance.get(key,0)+(1 if i<j else -1)
    boundary = {key:value for key,value in edge_balance.items() if value}
    expected = {}
    for i in range(12):
        a,b = 13+i,13+(i+1)%12
        expected[min(a,b),max(a,b)] = 1 if a<b else -1
    assert boundary == expected
    assert mesh["outer_polygon_inradius_m"] > 1800.


def test_profile_cache_cannot_be_mutated_by_caller():
    points,a = certified_cover_points()
    a["passed"] = False
    a["analytic_certificate"]["triangles"].clear()
    same,b = certified_cover_points("compact")
    assert points == same and b["passed"] and len(b["analytic_certificate"]["triangles"]) == 36


def test_compact_22_is_separately_certified_and_keeps_default_25():
    route, certificate = certified_cover_points("compact_22")
    assert len(route) == 22 and certificate["passed"]
    assert certificate["route_length_m"] == pytest.approx(17612.419400092313)
    full = certify_directional_cover(route)
    assert verify_directional_cover_certificate(route, full)["passed"]
    assert len(certified_cover_points("compact")[0]) == 25


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "outside", "station", "far", "hull"))
def test_corrupt_leaf_proofs_are_rejected(corruption):
    points = [(-2,-2), (2,-2), (2,2), (-2,2)]
    original = certify_directional_cover(points,arena_radius=1.,reception_radius=5.)
    certificate = copy.deepcopy(original)
    if corruption == "missing": certificate["leaves"]=[]
    elif corruption == "duplicate": certificate["leaves"]*=2
    elif corruption == "outside": certificate["leaves"][0]["kind"]="outside"
    elif corruption == "station": points[0]=(-3,-2)
    elif corruption == "far": certificate["reception_radius"]=1.
    elif corruption == "hull": certificate["leaves"][0]["stations"]=[0,1]
    with pytest.raises(ValueError): verify_directional_cover_certificate(points,certificate)


@pytest.mark.parametrize("kwargs", ({"arena_radius":0},{"reception_radius":math.inf},{"range_margin_m":0},
    {"orientation_margin_m":-1},{"max_depth":25},{"max_depth":True},{"max_cells":0},{"max_cells":1.5},
    {"range_margin_m":1001}))
def test_invalid_certificate_parameters_rejected(kwargs):
    with pytest.raises(ValueError): certify_directional_cover([(0,0)],**kwargs)


def test_invalid_profile_and_points_rejected():
    with pytest.raises(ValueError): certified_cover_points("not-a-certificate")
    with pytest.raises(ValueError): certify_directional_cover([])
    with pytest.raises(ValueError): concentric_stations(950,1900,inner_count=True)
    with pytest.raises(ValueError): concentric_stations(950,1900,inner_phase_deg=math.nan)


def test_leaf_omission_detected_even_when_other_leaves_are_valid():
    points = concentric_stations(950.,1800./math.cos(math.pi/12)+10.,inner_count=12,inner_phase_deg=15.)
    certificate = certify_directional_cover(points)
    certificate["leaves"].pop()
    with pytest.raises(ValueError,match="missing region"):
        verify_directional_cover_certificate(points,certificate)
