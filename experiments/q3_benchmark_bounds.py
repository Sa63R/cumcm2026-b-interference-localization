"""Post-session oracle lower bounds for completed synthetic Q3 comparisons."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiments.session_lower_bounds import empty_channel_action_bound, self_check, shortest_open_path
from simulation.cases import difficult_scenarios, random_scenario

EPSILON = 1e-6
CRITICAL_CASE_SOURCES = ("src/simulation/cases.py", "src/simulator_client/rules.py")
ANALYSIS_SOURCES = (*CRITICAL_CASE_SOURCES, "experiments/session_lower_bounds.py",
                    "experiments/q3_benchmark_bounds.py")


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {name: file_digest(ROOT / name) for name in ANALYSIS_SOURCES}


def nonnegative(value, name):
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def boolean(value, name):
    if value not in ("True", "False"):
        raise ValueError(f"CSV {name} must be True or False")
    return value == "True"


def read_comparison(directory):
    manifest_path, runs_path = directory / "manifest.json", directory / "runs.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("data_origin") != "synthetic_research" or manifest.get("problem") != 3:
        raise ValueError("Only synthetic Q3 comparison manifests are accepted")
    if manifest.get("run_status") != "completed" or manifest.get("source_changed_during_run") is not False:
        raise ValueError("Comparison must have completed with unchanged source hashes")
    before = manifest.get("source_sha256_start", {})
    after = manifest.get("source_sha256_end", {})
    for name in CRITICAL_CASE_SOURCES:
        actual = file_digest(ROOT / name)
        if not before.get(name) or before.get(name) != after.get(name) or actual != before[name]:
            raise ValueError(f"Scenario source differs from the original comparison: {name}")
    with runs_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != manifest.get("completed_runs") or len(rows) != manifest.get("expected_runs"):
        raise ValueError("runs.csv row count does not match the completed manifest")
    seen = set()
    for row in rows:
        if row.get("problem") != "3" or row.get("case_kind") not in ("random", "hard"):
            raise ValueError("Expected Q3 random/hard rows")
        key = (row["case_id"], row["strategy"])
        if key in seen:
            raise ValueError("Duplicate case/strategy row")
        seen.add(key)
        row["virtual_time_s"] = nonnegative(row["virtual_time_s"], "virtual time")
        row["successful"] = boolean(row["successful"], "successful")
        row["all_cleared"] = boolean(row["all_cleared"], "all_cleared")
        row["completion_certified"] = boolean(row["completion_certified"], "completion_certified")
        row["seed"] = int(row["seed"])
        row["source_total"] = int(row["source_total"])
        row["cleared_total"] = int(row["cleared_total"])
        if row["successful"] and not (row["all_cleared"] and row["completion_certified"]):
            raise ValueError("Successful row lacks full clearance or a completeness certificate")
    expected_strategies = {manifest["baseline"]["name"]} | {item["name"] for item in manifest["configs"]}
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row)
    if any({row["strategy"] for row in case_rows} != expected_strategies for case_rows in by_case.values()):
        raise ValueError("Each case must contain every fixed comparison strategy")
    return manifest, rows, by_case, {"manifest.json": file_digest(manifest_path), "runs.csv": file_digest(runs_path)}


def reconstruct_case(case_id, rows, manifest, hard_cases):
    kinds = {row["case_kind"] for row in rows}
    seeds = {row["seed"] for row in rows}
    if len(kinds) != 1 or len(seeds) != 1:
        raise ValueError(f"Case kind/seed disagrees across strategies: {case_id}")
    seed = next(iter(seeds))
    if kinds == {"random"}:
        start, end = manifest["random_seed_interval_inclusive"]
        if not start <= seed <= end:
            raise ValueError(f"Case seed is outside the comparison interval: {case_id}")
        case = random_scenario(3, seed)
    else:
        if not manifest.get("include_hard") or case_id not in hard_cases:
            raise ValueError(f"Unexpected hard case: {case_id}")
        case = hard_cases[case_id]
    if case.case_id != case_id or case.seed != seed:
        raise ValueError(f"Reconstructed case does not match recorded identity: {case_id}")
    if any(row["source_total"] != len(case.sources) for row in rows):
        raise ValueError(f"Reconstructed source count differs: {case_id}")
    return case


def common_bound(case, empty_action_cost):
    # This function is used only after the comparison ends. True source centres
    # never feed back into the online policy or alter the recorded trajectories.
    sources = sorted(case.sources, key=lambda source: source.channel)
    first = [max(0.0, math.hypot(source.x, source.y) - 20.0 - EPSILON)
             for source in sources]
    edges = [[0.0 if i == j else max(0.0, math.hypot(a.x - b.x, a.y - b.y) - 40.0 - EPSILON)
              for j, b in enumerate(sources)] for i, a in enumerate(sources)]
    length, order = shortest_open_path(first, edges)
    count = len(sources)
    empty_cost = (20 - count) * empty_action_cost if count < 16 else 0.0
    clear_cost = 5.0 * count
    bound = max(0.0, length / 5.0 + clear_cost + empty_cost - EPSILON)
    scenario_json = json.dumps(case.evaluation_config(), sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "case_id": case.case_id, "seed": case.seed, "source_total": count,
        "case_kind": "random" if "-random-" in case.case_id else "hard",
        "lower_route_length_m": length, "movement_lower_bound_s": length / 5.0,
        "successful_clear_action_lower_s": clear_cost,
        "empty_channel_action_lower_s": empty_cost,
        "empty_channel_count": 20 - count,
        "empty_channel_bound_applied": count < 16,
        "common_lower_bound_s": bound,
        "lower_graph_order_channels": [sources[index].channel for index in order],
        "scenario_sha256": hashlib.sha256(scenario_json).hexdigest(),
    }


def summarize(rows):
    groups = []
    for kind in ("random", "hard"):
        for strategy in dict.fromkeys(row["strategy"] for row in rows):
            subset = [row for row in rows if row["case_kind"] == kind and row["strategy"] == strategy]
            if not subset:
                continue
            mean_time = statistics.mean(row["virtual_time_s"] for row in subset)
            mean_bound = statistics.mean(row["common_lower_bound_s"] for row in subset)
            valid = [row for row in subset if row["successful"]]
            groups.append({
                "case_kind": kind, "strategy": strategy, "runs": len(subset),
                "successful_runs": len(valid), "failed_runs": len(subset) - len(valid),
                "mean_virtual_time_s": mean_time, "mean_lower_bound_s": mean_bound,
                "mean_time_over_mean_lower_bound": mean_time / mean_bound if mean_bound else None,
                "mean_nonnegative_gap_s": statistics.mean(row["nonnegative_gap_s"] for row in subset),
                "all_rows_valid_for_bound_comparison": len(valid) == len(subset),
                "successful_mean_time_over_mean_lower_bound": (
                    statistics.mean(row["virtual_time_s"] for row in valid) /
                    statistics.mean(row["common_lower_bound_s"] for row in valid)) if valid else None,
                "successful_mean_nonnegative_gap_s": statistics.mean(row["nonnegative_gap_s"] for row in valid) if valid else None,
                "interpretation": "Descriptive ratio to a common offline lower benchmark on these cases only; not a global approximation ratio. Failed/incomplete rows do not satisfy the full-clear benchmark condition.",
            })
    return groups


def analyze(directory):
    targets = (directory / "lower_bounds.json", directory / "lower_bounds.csv")
    if any(path.exists() for path in targets):
        raise ValueError("lower_bounds.json/csv already exists; existing analyses are not overwritten")
    start_hashes = source_hashes()
    manifest, original_rows, by_case, input_hashes = read_comparison(directory)
    verified_graphs = self_check()
    empty_action_cost = float(empty_channel_action_bound(3))
    if not math.isclose(empty_action_cost, 30.0, abs_tol=1e-12):
        raise AssertionError("Q3 empty-channel action lower bound no longer equals 30 seconds")
    hard_cases = {case.case_id: case for case in difficult_scenarios(3)}
    bounds = {}
    started = time.perf_counter()
    for index, (case_id, case_rows) in enumerate(by_case.items(), 1):
        case = reconstruct_case(case_id, case_rows, manifest, hard_cases)
        bound = common_bound(case, empty_action_cost)
        for row in case_rows:
            if row["successful"] and bound["common_lower_bound_s"] > row["virtual_time_s"] + EPSILON:
                raise AssertionError(f"Common lower bound exceeds a successful run: {case_id} / {row['strategy']}")
        bounds[case_id] = bound
        if index % 20 == 0 or index == len(by_case):
            print(f"[{index}/{len(by_case)} cases] {case_id}: LB={bound['common_lower_bound_s']:.3f}s; "
                  f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    rows = []
    for original in original_rows:
        bound = bounds[original["case_id"]]
        actual, lower = original["virtual_time_s"], bound["common_lower_bound_s"]
        rows.append({
            "problem": 3, "case_id": original["case_id"], "case_kind": original["case_kind"],
            "seed": original["seed"], "strategy": original["strategy"],
            "source_total": bound["source_total"], "cleared_total": original["cleared_total"],
            "successful": original["successful"], "all_cleared": original["all_cleared"],
            "completion_certified": original["completion_certified"],
            "virtual_time_s": actual, "common_lower_bound_s": lower,
            "time_over_lower_bound": actual / lower if lower else None,
            "nonnegative_gap_s": max(actual - lower, 0.0),
            "movement_lower_bound_s": bound["movement_lower_bound_s"],
            "successful_clear_action_lower_s": bound["successful_clear_action_lower_s"],
            "empty_channel_action_lower_s": bound["empty_channel_action_lower_s"],
            "scenario_sha256": bound["scenario_sha256"],
        })
    end_hashes = source_hashes()
    if start_hashes != end_hashes:
        raise ValueError("Analysis source files changed during computation; no results were written")
    if input_hashes != {name: file_digest(directory / name) for name in input_hashes}:
        raise ValueError("Comparison input files changed during computation; no results were written")
    report = {
        "analysis_kind": "offline_synthetic_q3_common_oracle_lower_bounds",
        "data_origin": "synthetic_research", "simulator_requests_sent": False,
        "comparison_stage": manifest.get("stage"), "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hashes, "analysis_source_sha256_start": start_hashes,
        "analysis_source_sha256_end": end_hashes, "case_generator_verified_against_manifest": True,
        "dp_exhaustive_oracle_checks": verified_graphs, "numerical_margin": EPSILON,
        "case_count": len(bounds), "run_count": len(rows),
        "successful_run_bound_checks": sum(row["successful"] for row in rows),
        "formula": "L_lower/5 + 5*N + (20-N)*30 for N<16; empty-channel term is zero for N=16; subtract 1e-6 s",
        "edge_relaxation": "first=max(0,|s|-20-1e-6); edge=max(0,|s_i-s_j|-40-1e-6); exact subset DP with a fixed origin and free endpoint",
        "bound_condition": "Complete clearance plus an online guarantee against any additional permitted source; the empty-channel necessity term is not a bound for merely lucky partial discovery.",
        "notes": [
            "True source centres are reconstructed only after the completed comparison, never supplied to any strategy.",
            "Pairwise minimum distances between clearance disks can be mutually incompatible at a shared visit; this relaxation need not be attained.",
            "This is an offline oracle lower benchmark, not an attainable shortest completion time or an online action plan.",
            "No return to the origin is required. Success costs and empty-channel actions are disjoint; movement is counted once.",
            "Random cases and the shared seven deterministic hard cases are summarized separately.",
            "Mean time divided by mean lower bound and nonnegative gaps are descriptive finite-sample statistics, never global approximation ratios.",
            "Failed or incomplete rows retain their values but cannot support full-clear efficiency comparisons; successful-only summaries are explicitly labelled.",
            "The physical relaxation uses continuous time with a small numerical margin; recorded simulator microsecond rounding is checked against every successful run.",
        ],
        "cases": list(bounds.values()), "groups": summarize(rows),
    }
    # All consistency checks complete before either target is created.
    with targets[0].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with targets[1].open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(targets[0]), "csv": str(targets[1]),
                      "case_count": len(bounds), "run_count": len(rows),
                      "groups": report["groups"]}, ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path, help="Completed run_q3_comparison output directory")
    args = parser.parse_args(argv)
    try:
        return analyze(args.comparison)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
