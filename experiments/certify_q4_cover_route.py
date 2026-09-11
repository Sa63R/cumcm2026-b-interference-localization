"""Public-station route certificate; no simulator, scenario, or source truth.

The ideal regular rings have an analytic MST attained by a Hamiltonian path.
Rational interval arithmetic verifies the strict ordering of all edge classes.
The implemented floating point snapshot is checked separately, with tolerance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _sub(a, b):
    return a[0] - b[1], a[1] - b[0]


def _mul(a, b):
    values = [x*y for x in a for y in b]
    return min(values), max(values)


def _scale(a, value):
    return _mul(a, (F(value), F(value)))


def _divide(a, b):
    if b[0] <= 0:
        raise ValueError("Positive denominator required")
    return _mul(a, (1/b[1], 1/b[0]))


def _sqrt(a):
    if a[0] < 0:
        raise ValueError("Nonnegative radicand required")
    scale = 10**15
    def floor_scaled(x):
        return math.isqrt(x.numerator*scale*scale // x.denominator)
    return F(floor_scaled(a[0]), scale), F(floor_scaled(a[1])+1, scale)


def _atan_reciprocal(n, terms=24):
    x = F(1, n)
    value = sum((-1)**k*x**(2*k+1)/(2*k+1) for k in range(terms))
    remainder = (-1)**terms*x**(2*terms+1)/(2*terms+1)
    return min(value, value+remainder), max(value, value+remainder)


def pi_interval():
    # Machin's identity, with alternating-series remainder; outward rounding.
    value = _sub(_scale(_atan_reciprocal(5), 16),
                 _scale(_atan_reciprocal(239), 4))
    scale = 10**34
    lo = value[0]*scale
    hi = value[1]*scale
    return F(lo.numerator//lo.denominator, scale), F(-(-hi.numerator//hi.denominator), scale)


def _trig(angle, cosine=False):
    midpoint = sum(angle)/2
    # |sin'|, |cos'| <= 1. Tail is alternating and decreasing here (|x|<=pi).
    terms = 24
    value = sum((-1)**k*midpoint**(2*k+(not cosine))/
                math.factorial(2*k+(not cosine)) for k in range(terms))
    degree = 2*terms+(not cosine)
    error = abs(midpoint)**degree/math.factorial(degree) + (angle[1]-angle[0])/2
    return value-error, value+error


def ideal_edge_classes():
    """All 231 ideal complete-graph edges, grouped into 20 exact classes."""
    pi = pi_interval()
    radius = _add(_divide((F(1800), F(1800)), _trig(_scale(pi, F(1, 14)), True)),
                  (F(5), F(5)))
    rows = []
    def add(name, count, interval):
        rows.append({"name": name, "count": count, "interval": interval})
    add("origin_inner", 7, (F(970), F(970)))
    add("origin_outer", 14, radius)
    for k in range(1, 4):
        add(f"inner_{k}", 7, _scale(_trig(_scale(pi, F(k, 7))), 1940))
    for k in range(1, 8):
        add(f"outer_{k}", 7 if k == 7 else 14,
            _scale(_mul(radius, _trig(_scale(pi, F(k, 14)))), 2))
    for k in range(8):
        if k == 0:
            length = _sub(radius, (F(970), F(970)))
        else:
            squared = _sub(_add(_mul(radius, radius), (F(970**2), F(970**2))),
                           _scale(_mul(radius, _trig(_scale(pi, F(k, 7)), True)), 1940))
            length = _sqrt(squared)
        add(f"cross_{k}", 7 if k in (0, 7) else 14, length)
    rows.sort(key=lambda row: sum(row["interval"]))
    if sum(row["count"] for row in rows) != 231:
        raise AssertionError("Incomplete edge-class partition")
    if any(a["interval"][1] >= b["interval"][0] for a, b in zip(rows, rows[1:])):
        raise AssertionError("Edge classes lack strictly separated interval bounds")
    expected = ["outer_1", "inner_1", "cross_0", "origin_inner"]
    if [row["name"] for row in rows[:4]] != expected:
        raise AssertionError("Kruskal ordering hypothesis failed")
    by_name = {row["name"]: row["interval"] for row in rows}
    length = (F(0), F(0))
    for name, count in zip(expected, (13, 6, 1, 1)):
        length = _add(length, _scale(by_name[name], count))
    return rows, radius, length


def _edge_class(a, b):
    if a == b:
        raise ValueError("Self edge")
    a, b = sorted((a, b))
    if a == 0:
        return "origin_inner" if b <= 7 else "origin_outer"
    if b <= 7:
        d = b-a
        return f"inner_{min(d, 7-d)}"
    if a >= 8:
        d = b-a
        return f"outer_{min(d, 14-d)}"
    d = abs(2*(a-1)-(b-8))
    return f"cross_{min(d, 14-d)}"


def euclidean_mst(points):
    """Independent full-edge Kruskal, with explicit tree edges."""
    points = tuple(tuple(p) for p in points)
    if not points or any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in points):
        raise ValueError("Finite nonempty 2D points required")
    parent = list(range(len(points)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    chosen = []
    for length, a, b in sorted((math.dist(points[a], points[b]), a, b)
                              for a in range(len(points)) for b in range(a+1, len(points))):
        x, y = find(a), find(b)
        if x != y:
            parent[x] = y
            chosen.append({"a": a, "b": b, "length_m": length})
    return math.fsum(edge["length_m"] for edge in chosen), chosen


def _outward_decimal(interval, digits=9):
    scale = 10**digits
    lower, upper = interval[0]*scale, interval[1]*scale
    values = [lower.numerator//lower.denominator, -(-upper.numerator//upper.denominator)]
    return {name: f"{value//scale}.{value%scale:0{digits}d}"
            for name, value in zip(("lower", "upper"), values)}


def certify_snapshot(points, route_indices, *, tolerance_m=1e-7):
    points = tuple(tuple(map(float, p)) for p in points)
    route = list(route_indices)
    if len(points) != 22 or points[0] != (0., 0.) or sorted(route) != list(range(22)) or route[0] != 0:
        raise ValueError("Expected each of 22 canonical stations once, starting at origin")
    if not math.isfinite(tolerance_m) or not 0 < tolerance_m <= 1e-5:
        raise ValueError("Explicit small positive numerical tolerance required")
    classes, radius, ideal_length = ideal_edge_classes()
    intervals = {row["name"]: row["interval"] for row in classes}
    residuals = []
    for a in range(22):
        for b in range(a+1, 22):
            length = math.dist(points[a], points[b])
            if not math.isfinite(length):
                raise ValueError("Nonfinite station geometry")
            lo, hi = intervals[_edge_class(a, b)]
            residuals.append(max(float(lo)-length, length-float(hi), 0.))
    if max(residuals) > tolerance_m:
        raise ValueError("Snapshot no longer matches ideal ring edge classes")
    mst_length, mst_edges = euclidean_mst(points)
    route_edges = [{"a": a, "b": b, "class": _edge_class(a, b),
                    "length_m": math.dist(points[a], points[b])} for a, b in zip(route, route[1:])]
    expected = Counter(outer_1=13, inner_1=6, cross_0=1, origin_inner=1)
    if Counter(e["class"] for e in route_edges) != expected:
        raise ValueError("Current route does not attain the ideal MST edge multiplicities")
    route_length = math.fsum(edge["length_m"] for edge in route_edges)
    if abs(route_length-mst_length) > tolerance_m:
        raise ValueError("Snapshot route does not attain numerical MST")
    if abs(route_length-float(sum(ideal_length)/2)) > tolerance_m:
        raise ValueError("Snapshot differs from ideal formula")
    for edge in mst_edges:
        edge["class"] = _edge_class(edge["a"], edge["b"])
    snapshot = {"points": points, "route_indices": route}
    snapshot_sha = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode()).hexdigest()
    return {"kind": "fixed_22_station_open_route_certificate_v1", "passed": True,
            "scope": "Fixed 22 ideal stations, Euclidean route starting at origin, no required return; no source service or measurements",
            "excluded_claims": ["Q4 global optimum", "optimal station layout", "optimal source-and-cover joint routing",
                                "mandatory visit of all stations after observing 16 sources", "formal interval proof for libm floating coordinates"],
            "analytic": {"pi_bounds": _outward_decimal(pi_interval(), 32),
                "outer_radius_m": _outward_decimal(radius), "mst_length_m": _outward_decimal(ideal_length),
                "mst_time_at_5_m_per_s": _outward_decimal(_scale(ideal_length, F(1, 5))),
                "edge_classes": [{"name": row["name"], "count": row["count"],
                                  "length_m": _outward_decimal(row["interval"])} for row in classes],
                "kruskal_selected_multiplicities": dict(expected),
                "interval_method": "Machin arctan alternating remainders; rational Taylor and Lipschitz argument envelopes; integer-isqrt outward roots"},
            "floating_snapshot": {**snapshot, "sha256": snapshot_sha, "tolerance_m": tolerance_m,
                "max_edge_class_envelope_residual_m": max(residuals),
                "route_length_m": route_length, "mst_length_m": mst_length,
                "route_minus_mst_m": route_length-mst_length, "route_edges": route_edges,
                "mst_edges": mst_edges,
                "statement": "Numerical agreement at stated tolerance; ideal indexed route is analytically optimal"}}


def build_certificate():
    from planning.q4_directional_cover import concentric_stations, certified_cover_points
    radius = 1800./math.cos(math.pi/14)+5.
    stations = concentric_stations(970., radius, inner_count=7, outer_count=14)
    route, cover = certified_cover_points("compact_22")
    points = [(p.x, p.y) for p in stations]
    lookup = {p: i for i, p in enumerate(points)}
    certificate = certify_snapshot(points, [lookup[(p.x, p.y)] for p in route])
    certificate["public_cover_summary"] = {"profile": cover["profile"],
        "station_sha256": cover["station_sha256"], "route_length_m": cover["route_length_m"]}
    paths = ["experiments/certify_q4_cover_route.py", "src/planning/q4_directional_cover.py",
             "src/planning/coverage.py", "src/simulator_client/state.py"]
    certificate["source_files_sha256"] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    return certificate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"research/q4_joint/cover_route_certificate.json")
    args = parser.parse_args(argv)
    certificate = build_certificate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(certificate, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "output": str(args.output),
                      "length_m": certificate["floating_snapshot"]["route_length_m"]}))


if __name__ == "__main__":
    main()
