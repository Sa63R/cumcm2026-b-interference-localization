"""Common Q4 lower benchmarks from saved, post-termination evaluation truth.

This module imports no policy, simulator, scenario generator, or HTTP client.
The continuous formula follows experiments/session_lower_bounds.py, but exact
source positions replace observation-containing disks. Thus the two compared
policies share one (usually stronger) hindsight denominator. It is not an online
policy, an attainable optimum, or an approximation guarantee.

For a fixed order of N successful clearances, the first movement is at least
max(|s_i|-20, 0), and a later movement is at least max(|s_i-s_j|-40, 0).
Subset DP minimizes the sum of these edge lower bounds. Its optimum is exact
for this finite graph only: different edges can require incompatible positions
inside the same source's clearance disk.

When N<16, a policy guaranteeing completion for EVERY allowed hidden Q4 scene
must rule out adding a directional source on each of the 20-N empty channels.
Outside f failed-clear disks of radius 20, almost every possible source location
requires >=3 negative measurements within distance 1000 to exclude all emitting
half-planes. With <=2 points only their segment can exclude all orientations;
a finite union of segments has zero area. Hence m*1000**2+3*f*20**2 >=
3*1800**2. Nonnegative integer action counts imply 5*m+3*f >=50 seconds.
This adds only empty-channel actions, so neither travel nor the 5*N successful
clear costs are counted twice. N=16 disables this term (the public count cap).
A successful observed sample alone does not establish the all-scenes premise.

The main field preserves the historical continuous-cost formula. A second
field handles round-to-nearest-microsecond movement: every moving action costs
at least 3 fixed seconds, so K<=T/3 and T>=LB-0.5e-6*K imply
T>=LB/(1+0.5e-6/3). A small numerical guard is additionally subtracted; ordinary
floating-point geometry/DP is not presented as interval-certified arithmetic.
"""

from __future__ import annotations

import argparse
from array import array
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time


VERSION = "q4-common-source-edge-v1"
FAILURE_PENALTY_S = 360000.0
NUMERIC_GUARD_S = 1e-6
ROUNDING_FACTOR = 1.0 + 0.5e-6 / 3.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {name}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {name}")
    return float(value)


def shortest_open_path(first, edges):
    """Exact directed subset DP; fixed origin/free end, no triangle assumption.

    The bitmask and last target are sufficient because edge costs are frozen.
    Compact numeric arrays bound the N=16 workspace below 10 MiB. This is the
    same recurrence as the legacy session analyzer, with no heuristic pruning.
    """
    n = len(first)
    if n > 16 or len(edges) != n or any(len(row) != n for row in edges):
        raise ValueError("Expected a square graph with at most 16 targets")
    first = [finite_nonnegative(v, "first edge") for v in first]
    edges = [[finite_nonnegative(v, "edge") for v in row] for row in edges]
    if not n:
        return 0.0, []
    size = 1 << n
    costs = array("d", [math.inf]) * (size * n)
    parents = array("b", [-1]) * (size * n)
    for j, value in enumerate(first):
        costs[(1 << j) * n + j] = value
    for mask in range(1, size):
        remaining_last = mask
        offset = mask * n
        while remaining_last:
            bit = remaining_last & -remaining_last
            j = bit.bit_length() - 1
            remaining_last ^= bit
            previous = mask ^ bit
            if not previous:
                continue
            previous_offset = previous * n
            candidates, best, best_k = previous, math.inf, -1
            while candidates:
                previous_bit = candidates & -candidates
                k = previous_bit.bit_length() - 1
                candidates ^= previous_bit
                value = costs[previous_offset + k] + edges[k][j]
                if value < best:
                    best, best_k = value, k
            costs[offset + j], parents[offset + j] = best, best_k
    mask = size - 1
    j = min(range(n), key=lambda k: (costs[mask * n + k], k))
    best, path = costs[mask * n + j], []
    while j >= 0:
        path.append(j)
        k = parents[mask * n + j]
        mask ^= 1 << j
        j = k
    return best, list(reversed(path))


def self_check():
    """Independent exhaustive check, including asymmetric/nonmetric graphs."""
    rng = random.Random(42634)
    for n in range(8):
        first = [rng.uniform(0, 10) for _ in range(n)]
        edges = [[rng.uniform(0, 20) for _ in range(n)] for _ in range(n)]
        expected = min((first[p[0]] + sum(edges[a][b] for a, b in zip(p, p[1:]))
                        for p in itertools.permutations(range(n))), default=0.) if n else 0.
        actual, path = shortest_open_path(first, edges)
        if not math.isclose(actual, expected, abs_tol=1e-10) or sorted(path) != list(range(n)):
            raise AssertionError("Subset DP differs from exhaustive permutations")
    return 8


def empty_channel_integer_certificate():
    # For m>=10, measurement cost alone is >=50. Enumerating 0..10 therefore
    # proves the global integer minimum; all arithmetic below is integer.
    values = []
    for m in range(11):
        deficit = max(0, 3 * 1800**2 - m * 1000**2)
        f = (deficit + 3 * 20**2 - 1) // (3 * 20**2)
        values.append({"measurements": m, "failed_clears": f, "cost_s": 5*m + 3*f})
    if min(v["cost_s"] for v in values) != 50:
        raise AssertionError("Q4 empty-channel integer relaxation changed")
    return values


def common_bound(ground_truth):
    """Use only after evaluation termination; input is evaluation.ground_truth."""
    if ground_truth.get("problem") != 4:
        raise ValueError("Expected Q4 evaluation truth")
    sources = ground_truth.get("sources", [])
    if not isinstance(sources, (list, tuple)) or not 10 <= len(sources) <= 16:
        raise ValueError("Expected 10..16 Q4 sources")
    channels = [s.get("channel") for s in sources]
    if any(type(c) is not int or c not in range(1, 21) for c in channels) or len(set(channels)) != len(channels):
        raise ValueError("Channels must be distinct integers in 1..20")
    for source in sources:
        x, y = source.get("x"), source.get("y")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (x, y)):
            raise ValueError("Invalid source position")
        if math.hypot(x, y) > 1800. + 1e-9:
            raise ValueError("Source lies outside the arena")
        radius = finite_nonnegative(source.get("reception_radius_m"), "source radius")
        if not 1000 <= radius <= 1500:
            raise ValueError("Invalid source radius")
        orientation = source.get("orientation_deg")
        if orientation is not None and not 0 <= finite_nonnegative(orientation, "orientation") < 360:
            raise ValueError("Invalid source orientation")
    # Q4 requires a directional source; it does NOT require any omnidirectional
    # source. The local random generator's mixed-type sampling is only a chosen
    # experimental distribution, not the public validity rule.
    if not any(s.get("orientation_deg") is not None for s in sources):
        raise ValueError("Q4 scenes must include at least one directional source")
    sources = sorted(sources, key=lambda s: s["channel"])
    points = [(s["x"], s["y"]) for s in sources]
    first = [max(0., math.hypot(*p) - 20.) for p in points]
    edges = [[max(0., math.dist(p, q) - 40.) for q in points] for p in points]
    length, order = shortest_open_path(first, edges)
    n = len(sources)
    empty_cost = float((20-n) * 50) if n < 16 else 0.
    physical = length / 5. + 5. * n
    common = physical + empty_cost
    return {"version": VERSION, "case_id": ground_truth.get("case_id"),
            "seed": ground_truth.get("seed"), "case_sha256": digest(ground_truth),
            "source_total": n, "lower_route_length_m": length,
            "movement_lower_bound_s": length / 5.,
            "successful_clear_action_lower_s": 5. * n,
            "physical_oracle_lower_bound_s": physical,
            "empty_channel_count": 20-n, "empty_channel_bound_applied": n < 16,
            "empty_channel_action_lower_s": empty_cost,
            "common_lower_bound_s": common,
            "common_lower_bound_rounded_s": max(0., common-NUMERIC_GUARD_S) / ROUNDING_FACTOR,
            "lower_graph_order_channels": [sources[i]["channel"] for i in order]}


def compare_records(records, *, cache=None, expected_strategies=None):
    """Compare every saved row, retaining failed runs at the 360000s penalty.

    The cache is keyed by the entire truth digest, not seed or successful-clear
    coordinates. Cached values only avoid repeated DP; they do not authorize
    reading truth before policy termination. This is not a physical audit:
    completion/observation certificates must also pass the independent auditor.
    """
    if not records:
        raise ValueError("No records")
    if cache is None:
        cache = {}
    seen, cases, rows = set(), {}, []
    evaluations = []
    # Validate all identities before any potentially expensive DP.
    for record in records:
        if record.get("evaluation_phase") != "after_policy_termination":
            raise ValueError("Truth is available only after policy termination")
        row, evaluation = record["row"], record["evaluation"]
        truth = evaluation["ground_truth"]
        sha = digest(truth)
        key = (row["case_id"], row["strategy"])
        if key in seen:
            raise ValueError("Duplicate case/strategy")
        seen.add(key)
        if row.get("problem") != 4 or row.get("case_sha256") != sha:
            raise ValueError("Case truth hash/problem mismatch")
        if truth.get("case_id") != row["case_id"] or truth.get("seed") != row.get("seed"):
            raise ValueError("Case identity mismatch")
        group = cases.setdefault(row["case_id"], {"sha": sha, "strategies": set()})
        if group["sha"] != sha:
            raise ValueError("Different truth across paired arms")
        group["strategies"].add(row["strategy"])
        actual = finite_nonnegative(row["virtual_time_s"], "virtual time")
        if type(row.get("successful")) is not bool:
            raise ValueError("Successful must be a Boolean")
        if row["successful"] and not all(row.get(k) is True for k in
                                         ("all_cleared", "completion_certified", "accepted_exit")):
            raise ValueError("Successful row lacks completion/exit")
        n = len(truth["sources"])
        if evaluation.get("source_total") != n or row.get("source_total") != n:
            raise ValueError("Source count mismatch")
        if (row.get("cleared_total") != evaluation.get("cleared_total") or
                row.get("all_cleared") != evaluation.get("all_cleared")):
            raise ValueError("Evaluation completion mismatch")
        if row["successful"] and row.get("cleared_total") != n:
            raise ValueError("Successful row did not clear every source")
        if not math.isclose(actual, finite_nonnegative(evaluation["virtual_time_s"], "evaluation time"), abs_tol=1e-6, rel_tol=0):
            raise ValueError("Evaluation time mismatch")
        penalty = actual if row["successful"] else FAILURE_PENALTY_S
        if row.get("penalized_time_s") != penalty:
            raise ValueError("Failure penalty must be retained at 360000 seconds")
        evaluations.append((row, truth, sha, actual, penalty))
    expected = set(expected_strategies) if expected_strategies is not None else {r["strategy"] for r, *_ in evaluations}
    if any(c["strategies"] != expected for c in cases.values()):
        raise ValueError("Incomplete paired strategies")
    started, computed, hits = time.perf_counter(), 0, 0
    for row, truth, sha, actual, penalty in evaluations:
        if sha not in cache:
            cache[sha] = common_bound(truth)
            computed += 1
        else:
            hits += 1
        bound = cache[sha]
        if bound.get("version") != VERSION or bound.get("case_sha256") != sha:
            raise ValueError("Stale/mismatched common-bound cache")
        lower = bound["common_lower_bound_s"]
        # No successful sample is discarded or clipped to make an invalid bound fit.
        if row["successful"] and actual < bound["common_lower_bound_rounded_s"] - 1e-6:
            raise ValueError("Claimed complete time is below the conditional lower benchmark")
        rows.append({"case_id": row["case_id"], "seed": row["seed"], "strategy": row["strategy"],
                     "case_sha256": sha, "successful": row["successful"],
                     "virtual_time_s": actual, "penalized_time_s": penalty,
                     "common_lower_bound_s": lower,
                     "common_lower_bound_rounded_s": bound["common_lower_bound_rounded_s"],
                     "time_over_common_lower_bound": actual/lower if row["successful"] else None,
                     "penalized_time_over_common_lower_bound": penalty/lower,
                     "physical_oracle_lower_bound_s": bound["physical_oracle_lower_bound_s"],
                     "empty_channel_action_lower_s": bound["empty_channel_action_lower_s"]})
    summaries = {}
    for strategy in sorted(expected):
        subset = [r for r in rows if r["strategy"] == strategy]
        summaries[strategy] = {"runs": len(subset), "successful": sum(r["successful"] for r in subset),
            "mean_penalized_time_s": statistics.mean(r["penalized_time_s"] for r in subset),
            "mean_common_lower_bound_s": statistics.mean(r["common_lower_bound_s"] for r in subset),
            "mean_of_penalized_ratios": statistics.mean(r["penalized_time_over_common_lower_bound"] for r in subset),
            "ratio_of_mean_penalized_time_to_mean_bound": (
                sum(r["penalized_time_s"] for r in subset)/sum(r["common_lower_bound_s"] for r in subset)),
            "all_complete": all(r["successful"] for r in subset)}
    return {"analysis_kind": VERSION, "case_count": len(cases), "record_count": len(rows),
        "bounds": {sha: cache[sha] for sha in sorted({r["case_sha256"] for r in rows})},
        "rows": sorted(rows, key=lambda r: (r["seed"], r["strategy"])), "summaries": summaries,
        "dp_computed_cases": computed, "dp_cache_hits": hits, "analysis_runtime_s": time.perf_counter()-started,
        "conditions": [
            "Truth is used after termination for evaluation only; no strategy calls this module.",
            "The 50-second empty-channel term assumes completion for every allowed hidden scene, not merely success on the displayed samples.",
            "This exact finite-graph DP is a relaxation of continuous clearance routing, not its exact solution.",
            "The source-truth graph is at least as strong as valid observation-containing-disk graphs in exact arithmetic; an official scene has no truth-based value here.",
            "Continuous historical formula and shared microsecond correction are reported separately; ordinary floating-point arithmetic is not an interval proof.",
            "Failed runs retain the full 360000-second comparison penalty; their actual incomplete times have no all-clear ratio.",
            "Mean per-case ratios and ratio of means are different statistics; neither proves near-optimality under unknown official distributions.",
            "Run the independent physical/certificate auditor separately; this module checks identity, pairing, and scalar outcome consistency only."]}


def analyze_directory(directory, *, cache=None):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8-sig"))
    summary = json.loads((directory/"summary.json").read_text(encoding="utf-8-sig"))
    if manifest.get("protocol", {}).get("problem") != 4:
        raise ValueError("Expected completed Q4 comparison")
    rows, records = summary["rows"], []
    expected = {(seed, label) for seed in manifest["seeds"] for label in manifest["specs"]}
    if len(rows) != len(expected) or {(r["seed"], r["strategy"]) for r in rows} != expected:
        raise ValueError("Incomplete/duplicate summary versus manifest")
    evidence = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ("manifest.json", "summary.json")}
    for row in rows:
        label = row["strategy"]
        if Path(label).name != label or "/" in label or "\\" in label:
            raise ValueError("Unsafe strategy path")
        path = directory/"records"/f"{label}-{row['seed']}.json.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        if record["row"] != row or record["spec"] != manifest["specs"][label]:
            raise ValueError("Record row/spec differs from frozen comparison")
        evidence[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(record)
    result = compare_records(records, cache=cache, expected_strategies=manifest["specs"])
    result["input_sha256"] = evidence
    result["protocol_sha256"] = digest(manifest["protocol"])
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cache = {}
    if args.cache and args.cache.exists():
        saved = json.loads(args.cache.read_text(encoding="utf-8"))
        checksum = saved.pop("payload_sha256", None)
        if saved.get("version") != VERSION or saved.get("implementation_sha256") != source_hash or digest(saved) != checksum:
            raise ValueError("Cache version/source/checksum mismatch")
        cache = saved["entries"]
    checked = self_check()
    result = analyze_directory(args.input, cache=cache)
    result.update(implementation_sha256=source_hash, exhaustive_checks=checked,
                  empty_channel_integer_certificate=empty_channel_integer_certificate())
    write_json(args.output, result)
    if args.cache:
        saved = {"version": VERSION, "implementation_sha256": source_hash, "entries": cache}
        saved["payload_sha256"] = digest(saved)
        write_json(args.cache, saved)
    print(json.dumps({"cases": result["case_count"], "records": result["record_count"],
                      "summaries": result["summaries"], "dp_computed_cases": result["dp_computed_cases"],
                      "analysis_runtime_s": result["analysis_runtime_s"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
