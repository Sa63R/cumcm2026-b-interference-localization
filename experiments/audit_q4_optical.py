"""Read-only optical certificate audit from accepted observation prefixes.

No strategy, simulator, source truth, database or scenario generator is read.
All geometry uses ordinary floating arithmetic with 1e-7 m checking tolerance;
the certified 19.8 m cells retain 0.2 m clearance slack.
"""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from localization import CandidateRegion
from planning.coverage import clearance_grid


def require(ok, message):
    if not ok:
        raise ValueError(message)


def xy(p):
    p = (p["x"], p["y"]) if isinstance(p, dict) else tuple(p)
    require(len(p) == 2 and all(math.isfinite(v) for v in p), "Invalid optical coordinate")
    return p


def close(a, b, eps=1e-7):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= eps


def certified_route(event, vertices, bearing, start):
    claimed = event["observed_region_vertices"]
    require(len(claimed) == len(vertices) and all(math.dist(xy(a), b) <= 1e-7
            for a, b in zip(claimed, vertices)), "Optical event region differs from actual prefix")
    require(close(event["bearing_deg"], bearing) and math.dist(xy(event["start"]), start) <= 1e-7,
            "Optical event bearing/start differs from actual prefix")
    if event["selected"] == "legacy":
        require(event.get("certificate") is None, "Legacy route has unexpected rectangle certificate")
        route = [xy((p.x, p.y)) for p in clearance_grid(vertices, bearing_deg=bearing, start=start)]
    else:
        require(event["selected"] == "rectangular_slabs", "Unknown optical cover")
        cert = event["certificate"]
        require(close(cert["radius_m"], 19.8, 1e-12), "Optical radius changed")
        angle = math.radians(bearing)
        c, s = math.cos(angle), math.sin(angle)
        polygon = [(x*c+y*s, -x*s+y*c) for x, y in vertices]
        lo, hi = min(p[1] for p in polygon), max(p[1] for p in polygon)
        require(all(close(a, b) for a, b in zip(cert["rotated_y_range"], (lo, hi))), "Wrong slab extent")
        slabs = cert["slabs"]
        require(type(cert["rows"]) is int and cert["rows"] == len(slabs) and len(slabs) > 0,
                "Missing covering slabs")
        previous, centers = lo, []
        for slab in slabs:
            lower, upper = slab["slab"]
            require(close(lower, previous) and upper >= lower and upper <= hi+1e-7, "Slab gap or reversed bounds")
            previous = upper
            # Complete vertices of convex polygon intersected with this slab:
            # original inside vertices and edge intersections with both lines.
            extremes = [p for p in polygon if lower <= p[1] <= upper]
            for a, b in zip(polygon, polygon[1:]+polygon[:1]):
                for bound in (lower, upper):
                    if a[1] != b[1] and min(a[1], b[1]) <= bound <= max(a[1], b[1]):
                        t = (bound-a[1])/(b[1]-a[1])
                        extremes.append((a[0]+t*(b[0]-a[0]), bound))
            require(bool(extremes), "Empty claimed slab intersection")
            x0, x1, y0, y1 = slab["bounding_box"]
            require(all(math.isfinite(v) for v in (x0, x1, y0, y1)) and x1 >= x0 and y1 >= y0,
                    "Invalid slab bounding box")
            require(all(x0-1e-7 <= x <= x1+1e-7 and y0-1e-7 <= y <= y1+1e-7
                        for x, y in extremes), "Slab box does not cover whole convex intersection")
            columns = slab["columns"]
            require(type(columns) is int and 1 <= columns <= 20000, "Invalid optical columns")
            width, height = (x1-x0)/columns, y1-y0
            half = math.hypot(width/2, height/2)
            require(half <= 19.8+1e-7 and close(width, slab["cell_width_m"])
                    and close(half, slab["cell_half_diagonal_m"]), "Cell exceeds certified radius")
            for j in range(columns):
                x, y = x0+(j+.5)*width, (y0+y1)/2
                centers.append((x*c-y*s, x*s+y*c))
        require(close(previous, hi), "Last slab leaves uncovered region")
        # Independent replay of the documented nearest-neighbour ordering.
        route, p = [], start
        while centers:
            index = min(range(len(centers)), key=lambda i: (math.dist(p, centers[i]), *centers[i]))
            p = centers.pop(index)
            route.append(p)
    require(event["selected_count"] == len(route), "Optical route count differs")
    return route


def audit_optical_prefix(record):
    summary = record.get("summary") or {}
    parameters = summary.get("strategy_parameters", {})
    events = parameters.get("optical_cover_log", [])
    active = parameters.get("q4_optical_mode") == "rectangular"
    if not events and not active:
        return {"passed": True, "applicable": False, "events": 0, "optical_actions": 0}
    actual = [a for a in record["history"] if a["action"] in ("/measure", "/clear")]
    reports = summary["action_history"]
    require(len(actual) == len(reports), "Optical wire/report length mismatch")
    indexed = {}
    for event in events:
        n = event["after_actual_action_count"]
        require(type(n) is int and 0 <= n <= len(actual) and n not in indexed, "Invalid optical prefix index")
        indexed[n] = event
    regions, first, cleared, consumed = {}, {}, set(), set()
    start = (0., 0.)
    for n in range(len(actual)+1):
        if n in indexed:
            event = indexed[n]
            ch = event["channel"]
            require(ch in regions and ch not in cleared and bool(regions[ch].vertices), "Optical event lacks live positive region")
            route = certified_route(event, regions[ch].vertices, first[ch], start)
            j = 0
            while n+j < len(actual) and reports[n+j].get("phase") == "guaranteed_clearance":
                a = actual[n+j]
                if a["channel"] != ch:
                    break
                require(j < len(route) and a["action"] == "/clear" and
                        math.dist(xy(a["position"]), route[j]) <= 2e-6, "Optical action is not a certified route prefix")
                consumed.add(n+j)
                j += 1
                if a["response"]["clear_result"] == "success":
                    break
            if n+j < len(actual):
                require(j == len(route) or (j > 0 and actual[n+j-1]["response"]["clear_result"] == "success"),
                        "Optical traversal abandoned before success or full cover")
        if n == len(actual):
            break
        a, report = actual[n], reports[n]
        response, channel = a["response"], a["channel"]
        result = response.get("measure_result") if a["action"] == "/measure" else response.get("clear_result")
        require(response.get("accepted") is True and report["action"] == a["action"][1:]
                and report["channel"] == channel and report["result"] == result
                and math.dist(xy(report["position"]), xy(a["position"])) <= 1e-7,
                "Optical wire/report action mismatch")
        start = xy(a["position"])
        if a["action"] == "/measure" and result == "direction":
            bearing = response["svd_deg"]
            require(close(report["bearing_deg"], bearing), "Reported bearing differs from wire")
            first.setdefault(channel, bearing)
            regions.setdefault(channel, CandidateRegion()).observe(start, bearing)
        if a["action"] == "/clear" and result == "success":
            cleared.add(channel)
    require(all(n in consumed for n, a in enumerate(reports) if a.get("phase") == "guaranteed_clearance"),
            "Uncertified optical action lacks an event")
    return {"passed": True, "applicable": True, "events": len(events), "optical_actions": len(consumed)}
