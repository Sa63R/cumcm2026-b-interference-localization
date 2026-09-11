"""Conservative finite certificates for arbitrary-orientation half-disk covers.

For a source x let S_x be the stations at distance <=R. Every closed emission
half-plane through x contains a station iff x belongs to conv(S_x), by convex
separation. We certify whole dyadic boxes, not sample points: stations chosen
for a box reach all four corners, and their convex hull contains the whole box.
Convexity then proves every source in that box receives from at least one of
those stations, for every possible emission direction.

Boxes wholly outside the arena disk are excluded. The remaining boxes form a
complete quadtree partition; unresolved cells never count as covered. Ordinary
floating arithmetic uses conservative distance/orientation margins, not formal
interval arithmetic. Certificates carry independently replayable leaf witnesses.
"""

import hashlib
import json
import math
import time
from functools import lru_cache

from simulator_client.state import Position


def _cross(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])


def _hull(points):
    points = sorted(set(points))
    if len(points) < 2:
        return points
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _corners(box):
    x0, x1, y0, y1 = box
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _children(box):
    x0, x1, y0, y1 = box
    x, y = (x0+x1)/2, (y0+y1)/2
    return ((x0, x, y0, y), (x, x1, y0, y), (x0, x, y, y1), (x, x1, y, y1))


def _outside(box, arena, margin):
    x0, x1, y0, y1 = box
    x, y = max(x0, min(0., x1)), max(y0, min(0., y1))
    return math.hypot(x, y) > arena + margin


def _contains_box(hull, box, margin):
    if len(hull) < 3:
        return False
    corners = _corners(box)
    for a, b in zip(hull, hull[1:]+hull[:1]):
        if min(_cross(a, b, p) for p in corners) < margin*math.dist(a, b):
            return False
    return True


def _support(points, box, radius, range_margin, orientation_margin):
    x0, x1, y0, y1 = box
    eligible = [p for p in points if math.hypot(max(abs(p[0]-x0), abs(p[0]-x1)),
                                               max(abs(p[1]-y0), abs(p[1]-y1))) <= radius-range_margin]
    hull = _hull(eligible)
    return hull if _contains_box(hull, box, orientation_margin) else None


def _separating_witness(points, x, radius, margin):
    # Enlarge the reception disk before seeking a strict separation. This is a
    # diagnostic counterexample with explicit normal, never a coverage proof.
    local = [p for p in points if math.dist(p, x) <= radius+margin]
    if not local:
        return {"source": list(x), "normal": [1., 0.], "local_stations": [], "max_projection_m": None}
    hull = _hull(local)
    if len(hull) >= 3 and all(_cross(a, b, x) >= 0 for a, b in zip(hull, hull[1:]+hull[:1])):
        return None
    best = None
    edges = list(zip(hull, hull[1:]+hull[:1]))
    for a, b in edges:
        dx, dy = b[0]-a[0], b[1]-a[1]
        size = dx*dx+dy*dy
        t = max(0., min(1., ((x[0]-a[0])*dx+(x[1]-a[1])*dy)/size)) if size else 0.
        q = a[0]+t*dx, a[1]+t*dy
        distance = math.dist(x, q)
        if best is None or distance < best[0]:
            best = distance, q
    distance, q = best
    if distance <= margin:
        return None
    normal = (x[0]-q[0])/distance, (x[1]-q[1])/distance
    projection = max(normal[0]*(p[0]-x[0])+normal[1]*(p[1]-x[1]) for p in local)
    if projection >= -margin:
        return None
    return {"source": list(x), "normal": list(normal), "local_stations": [list(p) for p in local],
            "max_projection_m": projection}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name+" must be finite and positive")
    return float(value)


def _points(stations):
    result = [Position.coerce(p) for p in stations]
    if not result:
        raise ValueError("At least one station is required")
    return tuple(sorted(set((float(p.x), float(p.y)) for p in result)))


def _point_hash(points):
    return hashlib.sha256(json.dumps(points, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def certify_directional_cover(stations, *, arena_radius=1800., reception_radius=1000.,
                              range_margin_m=1e-5, orientation_margin_m=1e-7,
                              max_depth=16, max_cells=200000, include_leaves=True):
    """Return a complete conservative certificate, a counterexample, or unknown.

    ``passed`` is true only when every leaf is proved covered or outside the
    arena. Depth/cell exhaustion returns ``inconclusive``. A counterexample is
    checked with a strict separating normal and an enlarged reception radius.
    Runtime is offline construction cost; no source observations are consumed.
    """
    began = time.perf_counter()
    points = _points(stations)
    arena = _positive(arena_radius, "arena_radius")
    radius = _positive(reception_radius, "reception_radius")
    margin = _positive(range_margin_m, "range_margin_m")
    orient = _positive(orientation_margin_m, "orientation_margin_m")
    if margin >= radius:
        raise ValueError("Range margin must be smaller than reception radius")
    if type(max_depth) is not int or not 0 <= max_depth <= 24:
        raise ValueError("max_depth must be an integer in 0..24")
    if type(max_cells) is not int or max_cells < 1:
        raise ValueError("max_cells must be a positive integer")
    indices = {p: i for i, p in enumerate(points)}
    pending = [("", (-arena, arena, -arena, arena))]
    leaves, visited, deepest, outside_count, covered_count = [], 0, 0, 0, 0
    failure, witness, unresolved = None, None, None
    while pending:
        if visited >= max_cells:
            failure = "cell_budget"
            break
        path, box = pending.pop()
        visited += 1
        deepest = max(deepest, len(path))
        if _outside(box, arena, margin):
            outside_count += 1
            leaves.append({"path": path, "kind": "outside"})
            continue
        support = _support(points, box, radius, margin, orient)
        if support is not None:
            covered_count += 1
            leaves.append({"path": path, "kind": "covered", "stations": [indices[p] for p in support]})
            continue
        probes = ((box[0]+box[1])/2, (box[2]+box[3])/2), (max(box[0], min(0., box[1])), max(box[2], min(0., box[3])))
        for probe in probes:
            if math.hypot(*probe) <= arena-margin:
                witness = _separating_witness(points, probe, radius, margin)
                if witness is not None:
                    failure = "strict_separating_counterexample"
                    break
        if witness is not None:
            break
        if len(path) >= max_depth:
            failure = "depth_budget"
            unresolved = {"path": path, "box": list(box)}
            break
        pending.extend((path+str(i), child) for i, child in reversed(list(enumerate(_children(box)))))
    return {"kind": "q4-local-convex-hull-dyadic-certificate-v1", "passed": failure is None,
            "status": "certified" if failure is None else "counterexample" if witness else "inconclusive",
            "reason": failure, "station_count": len(points), "stations": [list(p) for p in points],
            "station_sha256": _point_hash(points), "arena_radius": arena, "reception_radius": radius,
            "range_margin_m": margin, "orientation_margin_m": orient,
            "max_depth": max_depth, "max_cells": max_cells, "visited_cells": visited,
            "deepest_level": deepest, "outside_leaves": outside_count, "covered_leaves": covered_count,
            "leaves": leaves if include_leaves else None, "counterexample": witness,
            "unresolved_cell": unresolved,
            "runtime_s": time.perf_counter()-began,
            "arithmetic_scope": "real convexity theorem with conservative float margins; not interval arithmetic"}


def verify_directional_cover_certificate(stations, certificate):
    """Check every leaf witness and the complete partition without subdivision search."""
    points = _points(stations)
    if not certificate["passed"] or certificate["status"] != "certified" or not certificate.get("leaves"):
        raise ValueError("A full successful leaf certificate is required")
    if certificate["station_sha256"] != _point_hash(points):
        raise ValueError("Station identity mismatch")
    arena = _positive(certificate["arena_radius"], "arena_radius")
    radius = _positive(certificate["reception_radius"], "reception_radius")
    margin = _positive(certificate["range_margin_m"], "range_margin_m")
    orient = _positive(certificate["orientation_margin_m"], "orientation_margin_m")
    tree = {}
    for leaf in certificate["leaves"]:
        path, box, node = leaf["path"], (-arena, arena, -arena, arena), tree
        if not isinstance(path, str) or len(path) > 24 or any(c not in "0123" for c in path):
            raise ValueError("Invalid dyadic path")
        for c in path:
            if "leaf" in node:
                raise ValueError("Overlapping leaf paths")
            node = node.setdefault(c, {})
            box = _children(box)[int(c)]
        if node:
            raise ValueError("Duplicate or overlapping leaf")
        node["leaf"] = True
        if leaf["kind"] == "outside":
            if not _outside(box, arena, margin):
                raise ValueError("Unproved outside cell")
        elif leaf["kind"] == "covered":
            support_ids = leaf["stations"]
            if not support_ids or any(type(i) is not int or not 0 <= i < len(points) for i in support_ids):
                raise ValueError("Invalid support station indices")
            selected = [points[i] for i in support_ids]
            if any(math.dist(p, q) > radius-margin for p in selected for q in _corners(box)):
                raise ValueError("Support station too far from cell")
            if not _contains_box(_hull(selected), box, orient):
                raise ValueError("Support hull does not contain cell")
        else:
            raise ValueError("Unknown leaf type")
    def complete(node):
        return node == {"leaf": True} or (set(node) == set("0123") and all(complete(child) for child in node.values()))
    if not complete(tree):
        raise ValueError("Leaf partition has a missing region")
    return {"passed": True, "station_count": len(points), "leaf_count": len(certificate["leaves"]),
            "station_sha256": certificate["station_sha256"]}


def concentric_stations(inner_radius, outer_radius, *, inner_count=6, outer_count=12,
                        inner_phase_deg=0., outer_phase_deg=0.):
    """Generate a geometric proposal, never an implicit coverage guarantee."""
    inner_radius = _positive(inner_radius, "inner_radius")
    outer_radius = _positive(outer_radius, "outer_radius")
    if any(type(n) is not int or not 3 <= n <= 64 for n in (inner_count, outer_count)):
        raise ValueError("Ring counts must be integers in 3..64")
    if any(isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a)
           for a in (inner_phase_deg, outer_phase_deg)):
        raise ValueError("Ring phases must be finite angles")
    result = [Position(0., 0.)]
    for radius, count, phase in ((inner_radius, inner_count, inner_phase_deg), (outer_radius, outer_count, outer_phase_deg)):
        for i in range(count):
            angle = math.radians(phase) + 2*math.pi*i/count
            result.append(Position(radius*math.cos(angle), radius*math.sin(angle)))
    return tuple(result)


@lru_cache(maxsize=1)
def _compact_25():
    from .coverage import nearest_order, improve_open_route
    count, inner_radius = 12, 950.
    outer_radius = 1800./math.cos(math.pi/count) + 10.
    stations = concentric_stations(inner_radius, outer_radius, inner_phase_deg=15.,
                                  inner_count=count, outer_count=count)
    full = certify_directional_cover(stations)
    verify_directional_cover_certificate(stations, full)
    route = improve_open_route(nearest_order(stations))
    previous, length = Position(0., 0.), 0.
    for station in route:
        length += previous.distance_to(station)
        previous = station
    triangles = []
    for i in range(count):
        j = (i+1) % count
        triangles.extend(((0, 1+i, 1+j), (1+i, 1+count+i, 1+count+j), (1+i, 1+count+j, 1+j)))
    max_edge = max(stations[a].distance_to(stations[b]) for triangle in triangles
                   for a, b in zip(triangle, triangle[1:]+triangle[:1]))
    if max_edge >= 1000.-1e-5 or outer_radius*math.cos(math.pi/count) <= 1800.+1e-5:
        raise ValueError("Compact profile lost its analytic triangle margin")
    summary = {key: value for key, value in full.items() if key != "leaves"}
    summary.update(profile="compact_25", route_length_m=length, route_kind="fixed_origin_open_NN_then_2opt",
                   full_leaf_certificate_sha256=hashlib.sha256(json.dumps(full["leaves"],sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
                   analytic_certificate={"kind": "interleaved_concentric_ring_triangulation",
                       "inner_count": count, "outer_count": count, "inner_radius_m": inner_radius,
                       "outer_radius_m": outer_radius, "inner_phase_deg": 15., "outer_phase_deg": 0.,
                       "outer_polygon_inradius_m": outer_radius*math.cos(math.pi/count),
                       "max_triangle_edge_m": max_edge, "triangle_count": len(triangles),
                       "indexed_stations": [[p.x,p.y] for p in stations], "triangles": [list(t) for t in triangles],
                       "proof": "Center fan plus interleaved annulus triangles partition the outer regular polygon. Every triangle has diameter below1000m; the polygon contains the1800m disk."})
    return tuple(route), summary


def certified_cover_points(profile="compact_25"):
    """Return frozen ordered stations plus a JSON-safe passed certificate summary.

    ``compact`` is an alias for the exact same compact_25 geometry. The first
    call constructs and independently replays its full leaf proof. A caller
    receives a fresh summary so mutation cannot corrupt the cached certificate.
    """
    if profile not in {"compact", "compact_25", "compact_22"}:
        raise ValueError("Unknown certified Q4 cover profile")
    points, summary = _compact_22() if profile == "compact_22" else _compact_25()
    return points, json.loads(json.dumps(summary, allow_nan=False))


@lru_cache(maxsize=1)
def _compact_22():
    from .coverage import nearest_order, improve_open_route
    outer_radius = 1800./math.cos(math.pi/14) + 5.
    stations = concentric_stations(970., outer_radius, inner_count=7, outer_count=14)
    full = certify_directional_cover(stations)
    verify_directional_cover_certificate(stations, full)
    route = improve_open_route(nearest_order(stations))
    previous, length = Position(0., 0.), 0.
    for station in route:
        length += previous.distance_to(station)
        previous = station
    summary = {key: value for key, value in full.items() if key != "leaves"}
    summary.update(profile="compact_22", route_length_m=length, route_kind="fixed_origin_open_NN_then_2opt",
                   proposal={"inner_count":7,"outer_count":14,"inner_radius_m":970.,"outer_radius_m":outer_radius,
                             "inner_phase_deg":0.,"outer_phase_deg":0.},
                   full_leaf_certificate_sha256=hashlib.sha256(json.dumps(full["leaves"],sort_keys=True,separators=(",", ":")).encode()).hexdigest())
    return tuple(route), summary
