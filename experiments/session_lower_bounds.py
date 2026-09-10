"""Offline physical time bounds from accepted observations, never source truth.

No simulator requests are made. This is a hindsight analysis, not a search policy.
The oracle benchmark omits discovery/certification costs and need not be
attainable by an online strategy with initially unknown sources.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

from geometry import distance
from localization import CandidateRegion


def shortest_open_path(first, edges):
    """Exact subset DP, fixed origin, free endpoint, no triangle assumption."""
    n = len(first)
    if n == 0:
        return 0.0, []
    if n > 16 or len(edges) != n or any(len(row) != n for row in edges):
        raise ValueError("Expected a square graph with at most 16 targets")
    dp = [[math.inf] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    for j in range(n):
        dp[1 << j][j] = first[j]
    for mask in range(1, 1 << n):
        for j in range(n):
            if not mask & (1 << j):
                continue
            previous = mask ^ (1 << j)
            if not previous:
                continue
            cost, k = min((dp[previous][k] + edges[k][j], k)
                          for k in range(n) if previous & (1 << k))
            dp[mask][j], parent[mask][j] = cost, k
    mask = (1 << n) - 1
    best, j = min((dp[mask][j], j) for j in range(n))
    path = []
    while j >= 0:
        path.append(j)
        k = parent[mask][j]
        mask ^= 1 << j
        j = k
    return best, path[::-1]


def self_check():
    """Independent exhaustive oracle on small, deliberately nonmetric graphs."""
    import random
    rng = random.Random(2634)
    for n in range(1, 8):
        first = [rng.uniform(0, 10) for _ in range(n)]
        edges = [[rng.uniform(0, 20) for _ in range(n)] for _ in range(n)]
        expected = min(first[p[0]] + sum(edges[a][b] for a, b in zip(p, p[1:]))
                       for p in itertools.permutations(range(n)))
        actual, path = shortest_open_path(first, edges)
        assert math.isclose(actual, expected, abs_tol=1e-9)
        assert sorted(path) == list(range(n))
    return 7


def empty_channel_action_bound(problem):
    """Necessary action cost to exclude one extra source on an empty channel.

    Only valid when the actual total is below the public upper limit of 16.
    Q3: cover the arena circumference by detection/failed-clear disks.
    Q4: outside failed-clear disks, at least three nearby detection points
    are necessary almost everywhere to exclude every possible orientation.
    These are relaxations, not claims of attainable coverage constructions.
    """
    arena, reception, optical = 1800.0, 1000.0, 20.0
    if problem == 3:
        alpha = 2 * math.asin(reception / arena)
        beta = 2 * math.asin(optical / arena)
        candidates = [5 * m + 3 * max(0, math.ceil((2 * math.pi - m * alpha) / beta - 1e-10))
                      for m in range(math.ceil(2 * math.pi / alpha) + 1)]
    elif problem == 4:
        candidates = [5 * m + 3 * max(0, math.ceil(
            (3 * arena**2 - m * reception**2) / (3 * optical**2) - 1e-10))
                      for m in range(math.ceil(3 * arena**2 / reception**2) + 1)]
    else:
        raise ValueError("Problem must be 3 or 4")
    return min(candidates)


def analyze(summary_path):
    raw = summary_path.read_bytes()
    report = json.loads(raw.decode("utf-8-sig"))
    if report.get("data_origin") != "simulator_http_session":
        raise ValueError("Expected an HTTP-session summary")
    if report.get("state", {}).get("session") != "exited" or report.get("pending_request"):
        raise ValueError("Session must have exited without pending actions")
    history = report["search"]["action_history"]
    clears = [a for a in history if a["action"] == "clear" and a["result"] == "success"]
    channels = [a["channel"] for a in clears]
    positions = [tuple(a["position"]) for a in clears]
    n = len(clears)
    if not 1 <= n <= 16 or len(set(channels)) != n or n != report["state"]["cleared_count"]:
        raise ValueError("Successful-clear evidence is inconsistent")
    # True source belongs to every recorded disk. Any successful clearance
    # point therefore belongs to each disk expanded by a further 20 metres.
    source_disks = []
    for channel, q in zip(channels, positions):
        disks = [(q, 20.0, "accepted_successful_clear")]
        region = CandidateRegion()
        for action in history:
            if action["channel"] != channel:
                continue
            if action["action"] == "clear" and action["result"] == "success":
                break
            if action["action"] != "measure":
                continue
            if action["result"] == "near":
                disks.append((tuple(action["position"]), 5.0, "accepted_near"))
            elif action["result"] == "direction":
                region.observe(action["position"], action["bearing_deg"])
        if region.observations:
            if not region.vertices:
                raise ValueError("Inconsistent direction region")
            circle = region.enclosing_disk()
            radius = max(distance(circle.center, v) for v in region.vertices) + 1e-6
            disks.append((circle.center, radius, "outer_bearing_region"))
        source_disks.append(disks)

    def lower_graph(disks):
        first = [max(0.0, max(math.hypot(*c) - r - 20.0 - 1e-6
                              for c, r, _ in item)) for item in disks]
        edges = [[max(0.0, max(distance(a, b) - ra - rb - 40.0 - 1e-6
                               for a, ra, _ in left for b, rb, _ in right))
                  if i != j else 0.0 for j, right in enumerate(disks)]
                 for i, left in enumerate(disks)]
        return shortest_open_path(first, edges)

    simple_length, _ = lower_graph([[d[0]] for d in source_disks])
    stronger_length, order = lower_graph(source_disks)
    upper_length, upper_order = shortest_open_path(
        [math.hypot(*p) for p in positions],
        [[distance(a, b) for b in positions] for a in positions])
    lower_time = stronger_length / 5.0 + 5.0 * n
    upper_time = upper_length / 5.0 + 5.0 * n
    if lower_time > upper_time + 1e-7:
        raise AssertionError("Lower bound exceeds feasible hindsight route")
    empty_cost = empty_channel_action_bound(report["problem"]) if n < 16 else 0
    certified_lower = lower_time + (20 - n) * empty_cost
    return {
        "analysis_kind": "offline_physical_oracle_bounds_from_observations",
        "problem": report["problem"], "observed_cleared_sources": n,
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "actual_virtual_time_s": report["state"]["virtual_time_s"],
        "success_only_lower_bound_s": simple_length / 5.0 + 5.0 * n,
        "observation_lower_bound_s": lower_time,
        "feasible_oracle_upper_bound_s": upper_time,
        "lower_bound_average_s": lower_time / n,
        "oracle_upper_average_s": upper_time / n,
        "empty_channel_necessary_action_cost_s": empty_cost,
        "conditional_guaranteed_all_clear_lower_s": certified_lower,
        "conditional_guaranteed_all_clear_average_lower_s": certified_lower / n,
        "all_clear_bound_condition": "Actual source total equals observed cleared count; algorithm must guarantee all-clear for every allowed scene, not just happen to succeed.",
        "lower_route_length_m": stronger_length,
        "feasible_oracle_route_length_m": upper_length,
        "lower_graph_order_channels": [channels[i] for i in order],
        "feasible_oracle_order_channels": [channels[i] for i in upper_order],
        "source_containing_disks": dict(zip(map(str, channels), source_disks)),
        "time_breakdown": report["search"]["time_breakdown"],
        "notes": [
            "Source positions are not read or assumed equal to successful clearance points.",
            "Bounds concern clearing these observed sources; if more sources exist, only the lower bound extends to all-clear.",
            "Upper benchmark reorders known successful points with no detection; it is hindsight, not an executable unknown-scene policy.",
            "Edgewise lower distances may be incompatible at a shared visit, so the lower bound need not be attained.",
            "Guarantee bound adds only actions on actually empty channels; no movement or successful-clear costs are counted twice.",
            "Continuous virtual-cost model; simulator microsecond rounding is immaterial at displayed precision.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    args = parser.parse_args()
    verified_graphs = self_check()
    for path in args.summaries:
        result = analyze(path)
        result["dp_exhaustive_oracle_checks"] = verified_graphs
        target = path.with_name("lower_bounds.json")
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items()
                          if k not in {"source_containing_disks", "notes", "time_breakdown"}},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
