"""Pure geometry and scripted observation tests; no simulator is run."""
import math
from types import SimpleNamespace

import pytest

from simulator_client.state import Position
from strategies.q4_cover_search import Q4CoverSearch
from strategies.q4_optical_cover import (CELL_RADIUS_M, Q4OpticalCoverSearch,
                                        optical_cover_route, run_q4_optical_cover)
from strategies.search import _StopSearch


SHAPES = [((0., 0.),), ((0., 0.), (1500., 0.)),
          ((0., 0.), (1500., -25.), (1500., 25.)),
          ((0., -4.), (1500., -4.), (1500., 4.), (0., 4.)),
          ((0., -40.), (200., -40.), (200., 40.), (0., 40.))]


def transform(p, angle, shift):
    a = math.radians(angle)
    return Position(p[0]*math.cos(a)-p[1]*math.sin(a)+shift[0],
                    p[0]*math.sin(a)+p[1]*math.cos(a)+shift[1])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("angle", [0., 89.9, -173.])
def test_entire_rectangle_certificate_and_independent_interior_samples(shape, angle):
    vertices = tuple(transform(p, angle, (1234., -567.)) for p in shape)
    route, log = optical_cover_route(vertices, bearing_deg=angle, start=(1700., -900.))
    assert route and log["selected_proxy_s"] <= log["old_proxy_s"] + 1e-8
    cert = log["certificate"]
    if cert:
        low, high = cert["rotated_y_range"]
        slabs = cert["slabs"]
        assert slabs[0]["slab"][0] == low and slabs[-1]["slab"][1] == high
        assert all(a["slab"][1] == b["slab"][0] for a, b in zip(slabs, slabs[1:]))
        cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        rotated = [(p.x*cosine+p.y*sine, -p.x*sine+p.y*cosine) for p in vertices]
        expected_centers = []
        for slab in slabs:
            x0, x1, y0, y1 = slab["bounding_box"]
            assert all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in slab["polygon"])
            # Independent convex-intersection extremal certificate: every
            # vertex of polygon intersected with the slab is an original
            # vertex inside it or an original edge crossing one of its lines.
            # Bounding all of them bounds the whole intersection, not samples.
            lower, upper = slab["slab"]
            extremes = [p for p in rotated if lower <= p[1] <= upper]
            for a, b in zip(rotated, rotated[1:]+rotated[:1]):
                for boundary in (lower, upper):
                    if a[1] != b[1] and min(a[1], b[1]) <= boundary <= max(a[1], b[1]):
                        t = (boundary-a[1])/(b[1]-a[1])
                        extremes.append((a[0]+t*(b[0]-a[0]), boundary))
            assert extremes
            assert all(x0-1e-8 <= x <= x1+1e-8 and y0-1e-8 <= y <= y1+1e-8 for x, y in extremes)
            for column in range(slab["columns"]):
                # Independently bound all four corners of every covering cell.
                left = x0 + column*(x1-x0)/slab["columns"]
                right = x0 + (column+1)*(x1-x0)/slab["columns"]
                center = ((left+right)/2, (y0+y1)/2)
                assert max(math.dist(center, p) for p in
                    ((left, y0), (right, y0), (left, y1), (right, y1))) <= CELL_RADIUS_M+1e-8
                expected_centers.append(transform(center, angle, (0., 0.)))
        assert len(expected_centers) == len(route)
        assert all(min(p.distance_to(q) for q in route) < 1e-7 for p in expected_centers)
    # These samples detect rotation/route omissions; the guarantee is the
    # whole-cell certificate above, not this finite sample check.
    for a in vertices:
        for b in vertices:
            for fraction in (0., .1, .35, .5, .9, 1.):
                p = Position(a.x*(1-fraction)+b.x*fraction, a.y*(1-fraction)+b.y*fraction)
                assert min(p.distance_to(q) for q in route) < 20.0


def test_narrow_single_bearing_strip_uses_fewer_optical_points():
    route, log = optical_cover_route(SHAPES[3], bearing_deg=0, start=(0, 0))
    assert log["selected"] == "rectangular_slabs"
    assert len(route) < log["old_count"] * .7
    assert log["selected_proxy_s"] < log["old_proxy_s"]


def test_empty_region_and_nonfinite_bearing():
    route, log = optical_cover_route((), bearing_deg=0, start=(0, 0))
    assert route == () and log["selected_count"] == 0
    with pytest.raises(ValueError): optical_cover_route(SHAPES[0], bearing_deg=float("nan"), start=(0, 0))


class ScriptedRegion:
    def __init__(self):
        self.vertices = SHAPES[3]
        self.radius = 100.

    def enclosing_disk(self):
        return SimpleNamespace(center=(750., 0.), radius=self.radius)


def scripted_policy(cls, end, fail_clear=False):
    # Construct only the inherited local resolver state. There is no client
    # method capable of requesting simulator state or actions in these tests.
    policy = object.__new__(cls)
    policy.client = SimpleNamespace(state=SimpleNamespace(position=Position(0, 0)))
    policy.cleared, policy.near_points = set(), {}
    policy.regions, policy.first_bearings = {1: ScriptedRegion()}, {1: 0.}
    policy.max_active_probes, policy.variant = 6, "triangular"
    policy.report = SimpleNamespace(action_history=[])
    policy.optical_mode, policy.optical_log, policy.pending_pair = "rectangular", [], None
    events = []

    def probe(channel, index):
        return Position(100.+index, 50.)

    def perform(action, p, channel, phase):
        events.append((action, p, channel, phase))
        policy.client.state.position = p
        policy.report.action_history.append({"action": action, "position": [p.x, p.y]})
        if len(events) == 2:
            if end == "near": policy.near_points[channel] = Position(101., 50.)
            if end == "certificate": policy.regions[channel].radius = 1.
        return {"accepted": True}

    def clear(p, channel, phase):
        events.append(("clear", p, channel, phase))
        if end == "stop": raise _StopSearch("action_budget")
        return not fail_clear

    policy._next_probe, policy._perform, policy._clear = probe, perform, clear
    return policy, events


@pytest.mark.parametrize("end", ["near", "certificate", "fallback"])
@pytest.mark.parametrize("fail_clear", [False, True])
def test_active_prefix_and_certified_termination_match_parent(end, fail_clear):
    old, old_events = scripted_policy(Q4CoverSearch, end, fail_clear)
    new, new_events = scripted_policy(Q4OpticalCoverSearch, end, fail_clear)
    assert old._resolve(1) == new._resolve(1)
    if not any(e[3] == "guaranteed_clearance" for e in old_events):
        assert old_events == new_events
        assert not new.optical_log
    else:
        old_prefix = [e for e in old_events if e[3] != "guaranteed_clearance"]
        new_prefix = [e for e in new_events if e[3] != "guaranteed_clearance"]
        assert old_prefix == new_prefix
        assert len(new.optical_log) == 1
        if fail_clear:
            assert len([e for e in new_events if e[3] == "guaranteed_clearance"]) == new.optical_log[0]["selected_count"]


def test_off_mode_is_exact_parent_actions_and_no_new_cover_planning():
    old, old_events = scripted_policy(Q4CoverSearch, "fallback", True)
    new, new_events = scripted_policy(Q4OpticalCoverSearch, "fallback", True)
    new.optical_mode = "off"
    assert old._resolve(1) is False and new._resolve(1) is False
    assert old_events == new_events and not new.optical_log


def test_action_budget_stop_not_swallowed():
    policy, events = scripted_policy(Q4OpticalCoverSearch, "stop")
    with pytest.raises(_StopSearch) as caught: policy._resolve(1)
    assert caught.value.reason == "action_budget"


@pytest.mark.parametrize("kwargs", [{"problem": 3}, {"optical_mode": "unknown"},
    {"max_actions": True}, {"max_actions": 1}, {"max_expansions": -1}, {"max_active_probes": 31}])
def test_invalid_parameters_rejected_without_client_access(kwargs):
    class NoAccess:
        def __getattr__(self, name): raise AssertionError("client accessed before validation")
    with pytest.raises(ValueError): run_q4_optical_cover(NoAccess(), **kwargs)


def test_inheritance_keeps_cover_scan_nextprobe_and_scheduler_unchanged():
    for name in ("_scan", "_next_probe", "_execute_plan", "_perform", "_clear"):
        assert getattr(Q4OpticalCoverSearch, name) is getattr(Q4CoverSearch, name)
