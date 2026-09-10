"""Paired synthetic experiments with post-session truth and replayable traces."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from strategies import run_search


def source_manifest():
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__)]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def bootstrap_mean_interval(values, seed=2026, samples=2000):
    if not values:
        return None
    rng = random.Random(seed)
    estimates = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return [estimates[int(.025 * samples)], estimates[int(.975 * samples)]]


def summarize(rows):
    groups = []
    for problem in (3, 4):
        labels = sorted({r["strategy"] for r in rows if r["problem"] == problem})
        for kind in ("random", "hard", "all"):
            for label in labels:
                subset = [r for r in rows if r["problem"] == problem and r["strategy"] == label
                          and (kind == "all" or r["case_kind"] == kind)]
                if not subset:
                    continue
                baseline = {r["case_id"]: r for r in rows if r["problem"] == problem
                            and r["strategy"] == "baseline"}
                paired = [baseline[r["case_id"]]["virtual_time_s"] - r["virtual_time_s"] for r in subset
                          if r["case_id"] in baseline]
                total_time = sum(r["virtual_time_s"] for r in subset)
                total_clear = sum(r["cleared_total"] for r in subset)
                group = {
                    "problem": problem, "case_kind": kind, "strategy": label, "runs": len(subset),
                    "all_clear_runs": sum(r["all_cleared"] for r in subset),
                    "certified_runs": sum(r["completion_certified"] for r in subset),
                    "source_total": sum(r["source_total"] for r in subset), "cleared_total": total_clear,
                    "pooled_clear_fraction": total_clear / sum(r["source_total"] for r in subset),
                    "mean_virtual_time_s": statistics.mean(r["virtual_time_s"] for r in subset),
                    "mean_case_time_per_cleared_s": statistics.mean(
                        r["mean_time_per_cleared_s"] for r in subset if r["mean_time_per_cleared_s"] is not None),
                    "pooled_time_per_cleared_s": total_time / total_clear if total_clear else None,
                    "mean_runtime_s": statistics.mean(r["program_runtime_s"] for r in subset),
                    "max_runtime_s": max(r["program_runtime_s"] for r in subset),
                    "mean_measurements": statistics.mean(r["measurement_count"] for r in subset),
                    "mean_failed_clears": statistics.mean(r["failed_clear_count"] for r in subset),
                    "max_actions": max(r["action_count"] for r in subset),
                    "mean_paired_seconds_saved": statistics.mean(paired) if paired else None,
                    "paired_mean_saving_ci95_s": bootstrap_mean_interval(paired) if kind == "random" else None,
                    "mean_time_breakdown_s": {name: statistics.mean(r[name] for r in subset)
                        for name in ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")},
                }
                base_mean = statistics.mean(baseline[r["case_id"]]["virtual_time_s"] for r in subset)
                group["relative_mean_time_reduction_pct"] = 100 * (1 - group["mean_virtual_time_s"] / base_mean)
                groups.append(group)
    return groups


def execute(output, count, start_seed, include_minimax=True):
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    metadata = {
        "data_origin": "synthetic_research", "official_practice": False, "official_formal": False,
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_revision": commit,
        "source_sha256": source_manifest(), "python": sys.version, "platform": platform.platform(),
        "random_cases_per_problem": count, "start_seed": start_seed, "include_minimax": include_minimax,
        "distribution": "uniform area in the radius-1800 disk; uniform R in [1000,1500]; unique sampled channels; uniform orientation; location-fixed hashed errors",
        "error_quantization": "0.01 degree, nudged inward if rounded circular error exceeds 1 degree",
        "runtime_scope": "in-process local engine plus solver, excluding artifact compression; not official HTTP runtime",
        "comparison": "same case and error function across strategies, but different paths query different locations",
    }
    dump_json(output / "manifest.json", metadata)
    rows = []
    for problem in (3, 4):
        cases = [random_scenario(problem, s) for s in range(start_seed, start_seed + count)] + difficult_scenarios(problem)
        policies = [("baseline", "baseline", "center"), ("adaptive_center", "adaptive", "center")]
        if problem == 3 and include_minimax:
            policies.append(("adaptive_minimax", "adaptive", "minimax"))
        if problem == 4:
            policies.extend([("deferred", "deferred", "center"), ("triangular", "triangular", "center")])
        for case in cases:
            for label, variant, active in policies:
                sim = LocalResearchSimulator(case)
                client = sim.client()
                started = time.perf_counter()
                result = run_search(client, problem=problem, variant=variant, active_policy=active)
                elapsed = time.perf_counter() - started
                sim.finish_for_evaluation()
                evaluation = sim.evaluation()
                if result.cleared_count != evaluation["cleared_total"]:
                    raise AssertionError("Client count differs from independent evaluator")
                if abs(result.virtual_time_s - sum(evaluation["time_breakdown_s"].values())) > 1e-6:
                    raise AssertionError("Virtual-time decomposition mismatch")
                if result.completion_certified_under_model and not evaluation["all_cleared"]:
                    raise AssertionError("False completeness certificate")
                trace_name = f"{case.case_id}--{label}.json.gz"
                trace = {"data_origin": "synthetic_research", "strategy": label,
                         "search": result.as_dict(), "evaluation": evaluation}
                with gzip.open(output / "traces" / trace_name, "wt", encoding="utf-8") as stream:
                    json.dump(trace, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                row = {
                    "problem": problem, "case_id": case.case_id,
                    "case_kind": "random" if "-random-" in case.case_id else "hard",
                    "strategy": label, "source_total": evaluation["source_total"],
                    "cleared_total": evaluation["cleared_total"], "all_cleared": evaluation["all_cleared"],
                    "cleared_fraction": evaluation["cleared_fraction"],
                    "completion_certified": result.completion_certified_under_model,
                    "completion_reason": result.completion_reason,
                    "virtual_time_s": evaluation["virtual_time_s"],
                    "mean_time_per_cleared_s": evaluation["mean_time_per_cleared_s"],
                    "program_runtime_s": elapsed, "measurement_count": evaluation["measurement_count"],
                    "failed_clear_count": evaluation["failed_clear_count"], "action_count": evaluation["action_count"],
                    **evaluation["time_breakdown_s"], "trace": "traces/" + trace_name,
                }
                rows.append(row)
                with (output / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Q{problem} {case.case_id} {label}: {row['cleared_total']}/{row['source_total']} "
                      f"{row['virtual_time_s']:.2f}s virtual {elapsed:.3f}s wall", flush=True)
    metadata["source_changed_during_run"] = metadata["source_sha256"] != source_manifest()
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(output / "manifest.json", metadata)
    dump_json(output / "summary.json", {"data_origin": "synthetic_research", "groups": summarize(rows)})
    return 0 if all(r["all_cleared"] and r["completion_certified"] for r in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "study")
    parser.add_argument("--random-cases", type=int, default=40)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--no-minimax", action="store_true")
    args = parser.parse_args()
    if args.random_cases < 1:
        parser.error("--random-cases must be positive")
    return execute(args.output, args.random_cases, args.start_seed, not args.no_minimax)


if __name__ == "__main__":
    raise SystemExit(main())
