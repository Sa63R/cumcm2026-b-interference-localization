"""Single-source geometry constructions only; no simulator or scenario runner."""
import copy
from fractions import Fraction
import math

import pytest

from localization import CandidateRegion
import planning.joint_visibility_region as joint


def contains(vertices, p):
    # Independent exact closed-polygon point check, allowing either orientation.
    q = tuple(Fraction(v) for v in p)
    v = [tuple(Fraction(x) for x in v) for v in vertices]
    values = [(b[0]-a[0])*(q[1]-a[1])-(b[1]-a[1])*(q[0]-a[0])
              for a, b in zip(v, v[1:]+v[:1])]
    return all(x >= 0 for x in values) or all(x <= 0 for x in values)


def angle_in(intervals, theta):
    theta %= 2*math.pi
    return any(a <= theta <= b for a, b in intervals)


def rectangle(a=-2., b=-2., c=2., d=2.):
    return ((a, b), (c, b), (c, d), (a, d))


def test_boundary_is_actually_removed_and_true_directional_source_retained():
    canonical = rectangle(990., -.1, 1015., .1)
    positive = ((0., -100.), (0., 100.))
    negative = ((1005., 0.),)
    frozen = copy.deepcopy((canonical, positive, negative))
    outer, evidence = joint.joint_visibility_outer(canonical, positive, negative)
    assert evidence["status"] == "outer_refined" and evidence["passed"]
    assert evidence["deleted_intersecting_cells"] > 0
    assert contains(outer, (1000., 0.))
    assert not contains(outer, (1014., 0.))
    assert evidence["old_vertices_excluded"] and max(p[0] for p in outer) < 1010.
    assert evidence["new_disk"]["radius_m"] < evidence["old_disk"]["radius_m"]
    assert (canonical, positive, negative) == frozen


def test_deleted_interior_cells_can_be_filled_back_by_safe_convex_hull():
    canonical = rectangle(990., -1., 1015., 1.)
    outer, evidence = joint.joint_visibility_outer(canonical,
        [(0., -100.), (0., 100.)], [(1005., 0.)])
    assert evidence["deleted_intersecting_cells"] > 0
    assert not evidence["old_vertices_excluded"]
    assert all(contains(outer, p) for p in canonical)
    assert evidence["new_disk"]["radius_m"] == pytest.approx(evidence["old_disk"]["radius_m"], abs=1e-10)


def test_uniform_grid_has_all_256_ids_and_shared_closed_edges():
    original = rectangle()
    _, log = joint.joint_visibility_outer(original, [(20., 0.)], [(3000., 0.)])
    assert log["status"] == "unchanged" and len(log["cells"]) == 256
    assert [c["id"] for c in log["cells"]] == list(range(256))
    xs, ys = log["x_edges"], log["y_edges"]
    assert xs[0] < -2 and xs[-1] > 2 and ys[0] < -2 and ys[-1] > 2
    for cell in log["cells"]:
        ix, iy = cell["ix"], cell["iy"]
        assert cell["bbox"] == [xs[ix], ys[iy], xs[ix+1], ys[iy+1]]
        assert not cell["removed"] and cell["reason"] == "omni_branch_retained"


def test_positive_points_surrounding_region_do_not_kill_omni_branch():
    original = rectangle(-1, -1, 1, 1)
    positive = [(500., 0.), (-250., 400.), (-250., -400.)]
    outer, log = joint.joint_visibility_outer(original, positive, [(1400., 0.)])
    assert outer == original and log["deleted_intersecting_cells"] == 0
    assert all(not c["constraints"] for c in log["cells"])


@pytest.mark.parametrize("observer", [(0., 0.), (1., 0.), (1., 1.), (0., -1.), (1.+1e-9, 0.)])
@pytest.mark.parametrize("positive", [True, False])
def test_inside_edge_corner_and_near_zero_vector_constraints_are_full_circle(observer, positive):
    intervals, proof = joint._allowed_orientation(observer, (-1., -1., 1., 1.),
        positive=positive, distance_guard=1e-7)
    assert intervals == ((0., joint.TAU),)
    assert proof["reason"] == "near_or_inside_box"


def test_circular_wrap_and_closed_tangent_singletons_are_retained():
    arc, _ = joint._allowed_orientation((100., 0.), (-1., -1., 1., 1.),
        positive=True, distance_guard=1e-7)
    assert angle_in(arc, 0.) and angle_in(arc, joint.TAU-1e-3)
    assert not angle_in(arc, math.pi)
    assert joint._intersect(((0., math.pi),), ((math.pi, joint.TAU),))
    seam = joint._intersect(joint._normalize(((0., .1),)), joint._normalize(((6., joint.TAU),)))
    assert angle_in(seam, 0.)
    assert joint._intersect(((0., 1.),), ((1.+1e-13, 2.),))  # outward near-gap retention
    assert not joint._intersect(((0., 1.),), ((1.+1e-6, 2.),))


def test_direction_intersection_can_have_multiple_components():
    a = joint._arc_intervals(.2, 5.8)
    b = joint._arc_intervals(3.3, 5.8)
    result = joint._intersect(a, b)
    assert len(result) == 2
    assert angle_in(result, 1.) and angle_in(result, 4.)


def test_each_orientation_union_contains_actual_allowed_angles_throughout_box():
    box = (-2., -3., 4., 1.)
    for observer in ((10., 0.), (-4., 10.), (-6., -10.), (.5, .5)):
        for positive in (True, False):
            intervals, _ = joint._allowed_orientation(observer, box,
                positive=positive, distance_guard=1e-7)
            for location in joint._corners(box)+((0., 0.), (3., -.5)):
                for k in range(72):
                    theta = k*joint.TAU/72
                    dot = ((observer[0]-location[0])*math.cos(theta)
                           +(observer[1]-location[1])*math.sin(theta))
                    if (positive and dot >= 0.) or (not positive and dot <= 0.):
                        assert angle_in(intervals, theta)


@pytest.mark.parametrize("angle", [0., .4, 1.8, 3.1, 4.8, 6.27])
def test_valid_true_directional_positions_survive_positive_bearing_and_real_negatives(angle):
    source = (200., -150.)
    positives = [(source[0]+700*math.cos(angle+t), source[1]+700*math.sin(angle+t))
                 for t in (-1., -.4, .5)]
    negatives = [(source[0]+650*math.cos(angle+t), source[1]+650*math.sin(angle+t))
                 for t in (2., 3., 4.)]
    region = CandidateRegion()
    for i, p in enumerate(positives):
        bearing = math.degrees(math.atan2(source[1]-p[1], source[0]-p[0])) % 360
        region.observe(p, bearing+(-1. if i % 2 else 1.))
    canonical = tuple(region.vertices)
    assert contains(canonical, source)
    outer, evidence = joint.joint_visibility_outer(canonical, positives, negatives)
    assert contains(outer, source) and evidence["passed"]
    assert evidence["status"] != "fallback"
    for cell in evidence["cells"]:
        a, b, c, d = cell["bbox"]
        if a <= source[0] <= c and b <= source[1] <= d:
            assert not cell["removed"]


def test_valid_omni_with_radius_above_1000_and_old_negatives_is_preserved():
    source = (0., 0.)
    positives = [(1200., 0.), (-600., 600.), (-600., -600.)]
    negatives = [(1250., 100.), (-1300., 0.)]
    outer, log = joint.joint_visibility_outer(rectangle(-100., -100., 100., 100.), positives, negatives)
    assert contains(outer, source) and log["passed"]
    for cell in log["cells"]:
        a, b, c, d = cell["bbox"]
        if a <= 0 <= c and b <= 0 <= d:
            assert not cell["forced_negative_indices"] and not cell["removed"]


def test_local_engine_tolerant_positive_halfplane_is_not_excluded():
    # u=(1,0): dot=-5e-10 is accepted for distance1000 with engine tolerance1e-9.
    positive = [(-5e-10, 1000.)]
    negative = [(-500., 0.)]
    outer, log = joint.joint_visibility_outer(rectangle(-.01, -.01, .01, .01), positive, negative)
    assert contains(outer, (0., 0.))
    for cell in log["cells"]:
        if cell["constraints"]:
            assert cell["constraints"][0]["engine_margin_rad"] > 0


def test_radius_equality_is_not_forced_and_no_signal_strictness_is_relaxed():
    box = (0., 0., 0., 0.)
    guard = 1e-7
    rlower = max(1000., joint._distance_min((1000., 0.), box))-guard
    nupper = joint._distance_max((1000., 0.), box)+guard
    assert not nupper < rlower-joint.FORCED_MARGIN_M
    plus, _ = joint._allowed_orientation((1., 0.), box, positive=True, distance_guard=guard)
    minus, _ = joint._allowed_orientation((1., 0.), box, positive=False, distance_guard=guard)
    assert joint._intersect(plus, minus)  # ideal negative strictness is relaxed to closed


def test_exact_clipping_and_outward_conversion_contain_nonbinary_intersections():
    polygon = ((Fraction(0), Fraction(0)), (Fraction(3), Fraction(1)), (Fraction(0), Fraction(2)))
    clipped = joint._clip_box_exact(polygon, (.5, 0., 1., 2.))
    assert any(p[1].denominator == 6 for p in clipped)
    for p in clipped:
        a, b, c, d = map(Fraction, joint._round_box(p))
        assert a <= p[0] <= c and b <= p[1] <= d


@pytest.mark.parametrize("canonical,positives,negatives,reason", [
    ([(0., 0.)], [(1., 0.)], [], "degenerate_canonical_region"),
    ([(0., 0.), (1., 1.)], [(2., 0.)], [], "degenerate_canonical_region"),
    (rectangle(), [], [(1., 0.)], "no_positive_evidence"),
    (rectangle(), [(1., 0.)]*257, [], "constraint_budget"),
])
def test_declared_input_fallbacks_return_original(canonical, positives, negatives, reason):
    outer, log = joint.joint_visibility_outer(canonical, positives, negatives)
    assert outer == tuple(canonical)
    assert log["status"] == "fallback" and log["fallback_reason"] == reason


def test_all_deleted_and_partial_exception_never_return_partial_pruning(monkeypatch):
    original = rectangle(-.01, -.01, .01, .01)
    positive = ((100., 0.), (-50., 86.), (-50., -86.))
    outer, log = joint.joint_visibility_outer(original, positive, [(500., 0.)])
    assert outer == original and log["fallback_reason"] == "all_intersecting_cells_deleted"
    calls = 0
    actual = joint._clip_box_exact
    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 9:
            raise ArithmeticError("synthetic precision failure")
        return actual(*args)
    monkeypatch.setattr(joint, "_clip_box_exact", interrupted)
    outer, log = joint.joint_visibility_outer(original, positive, [(500., 0.)])
    assert outer == original and log["fallback_reason"] == "geometry_exception"
    assert len(log["cells"]) == 8


def test_retained_rounding_boxes_are_all_inside_returned_exact_hull():
    outer, evidence = joint.joint_visibility_outer(rectangle(990., -1., 1015., 1.),
        [(0., -100.), (0., 100.)], [(1005., 0.)])
    for cell in evidence["cells"]:
        if cell["removed"]:
            assert not cell["orientation_intersection"] and cell["forced_negative_indices"]
        for box in cell["retained_intersection_boxes"]:
            assert all(contains(outer, p) for p in joint._corners(box))
