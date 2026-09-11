"""Sample source-count timing with the qualified R12 practice-only entrypoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate(rows):
    groups = []
    for n in range(10, 17):
        group = [r for r in rows if r["source_count"] == n]
        if not group:
            groups.append({"source_count": n, "runs": 0})
            continue
        times = [r["actual_time_s"] for r in group]
        bounds = [r["lower_bound_s"] for r in group]
        groups.append({"source_count": n, "runs": len(group),
            "mean_actual_time_s": statistics.mean(times),
            "min_actual_time_s": min(times), "max_actual_time_s": max(times),
            "mean_lower_bound_s": statistics.mean(bounds),
            "sum_time_over_sum_lower_bound": sum(times)/sum(bounds),
            "mean_individual_ratio": statistics.mean(t/b for t,b in zip(times,bounds))})
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-root", required=True, type=Path)
    parser.add_argument("--simulator-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-per-count", default=2, type=int)
    parser.add_argument("--maximum-runs", default=60, type=int)
    args = parser.parse_args()
    if not os.environ.get("CUMCM_ROBOT_ID"):
        raise ValueError("CUMCM_ROBOT_ID is required; do not put identity in source files")
    if not 1 <= args.minimum_per_count <= 10 or not 1 <= args.maximum_runs <= 100:
        raise ValueError("Invalid sampling budget")
    root = args.strategy_root.resolve(strict=True)
    entry = root / "experiments/run_q4_joint_continuation_practice.py"
    entry_hash = hashlib.sha256(entry.read_bytes()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "problem": 4, "mode": "practice", "strategy": "compact_joint_continuation",
        "config": "after_active_miss_optical", "practice_entry_sha256": entry_hash,
        "minimum_per_source_count": args.minimum_per_count,
        "maximum_runs": args.maximum_runs,
        "stopping_rule": "Stop after every source count 10..16 has the required completed runs, or at budget/error/STOP.",
        "selection_rule": "Keep every sampled case; inspect source count only after the official run ends.",
        "lower_bound_kind": "historical_conditional_all_clear_containing_region_bound",
        "primary_ratio": "sum(actual_time_s)/sum(lower_bound_s)"}
    write(output / "protocol.json", protocol)
    rows = []
    reason = "maximum_runs"
    for index in range(1, args.maximum_runs + 1):
        if (output / "STOP").exists():
            reason = "stop_requested"
            break
        if hashlib.sha256(entry.read_bytes()).hexdigest() != entry_hash:
            raise ValueError("Practice entrypoint changed during this batch")
        batch = output / f"case-{index:03d}"
        command = [sys.executable, "-B", str(entry), "--simulator-dir", str(args.simulator_dir.resolve()),
                   "--repeat", "1", "--output", str(batch)]
        with (output / f"case-{index:03d}.log").open("wb") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            reason = "practice_entry_failed"
            write(output / "failure.json", {"run": index, "returncode": result.returncode})
            break
        registered = json.loads((batch / "result-001.json").read_bytes())
        run = batch / "run-001"
        lower = json.loads((run / "lower_bounds.json").read_bytes())
        summary = json.loads((run / "summary.json").read_bytes())
        n = lower["official_source_total"]
        if (registered.get("official_all_clear_verified") is not True
                or registered.get("completed") is not True or n not in range(10,17)):
            raise ValueError("Incomplete or unexpected official practice result; preserve this run")
        row = {"run": index, "source_count": n, "actual_time_s": lower["actual_virtual_time_s"],
               "lower_bound_s": lower["conditional_guaranteed_all_clear_lower_s"],
               "time_over_lower_bound": lower["time_to_conditional_lower_bound_ratio"],
               "all_clear": True, "program_wall_time_s": summary["program_wall_time_s"],
               "summary_sha256": hashlib.sha256((run / "summary.json").read_bytes()).hexdigest(),
               "lower_bounds_sha256": hashlib.sha256((run / "lower_bounds.json").read_bytes()).hexdigest()}
        rows.append(row)
        groups = aggregate(rows)
        write(output / "summary.json", {"protocol": protocol, "rows": rows, "groups": groups,
                                       "status": "running"})
        print(json.dumps({"completed_runs": len(rows), "latest": row,
                          "counts": {g["source_count"]:g["runs"] for g in groups}}), flush=True)
        if all(g["runs"] >= args.minimum_per_count for g in groups):
            reason = "source_counts_covered"
            break
    write(output / "summary.json", {"protocol": protocol, "rows": rows, "groups": aggregate(rows),
                                   "status": reason})
    print(json.dumps({"status": reason, "completed_runs": len(rows), "groups": aggregate(rows)}), flush=True)
    return 0 if reason == "source_counts_covered" else 1


if __name__ == "__main__":
    raise SystemExit(main())
