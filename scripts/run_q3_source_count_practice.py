"""Q3 practice-only source-count sampling using the frozen qualified strategy."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

MAIN = Path(__file__).resolve().parents[1]
COMMIT = "760af8230a8e5bda6050663b19d871a9701c9829"
SPEC_NAME = "experiments/state_search_candidate_derived_silence_combined_v1.json"
SPEC_SHA = "6ac7aabffad58d9b1b733a9bb5b6b5e61026a9e1f5bb232622622f7f8c78f9b9"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def groups(rows):
    result = []
    for n in range(10, 17):
        attempts = [r for r in rows if r.get("source_count") == n]
        complete = [r for r in attempts if r.get("all_clear")]
        group = {"source_count": n, "attempts": len(attempts), "runs": len(complete),
                 "failed_runs": len(attempts) - len(complete)}
        if complete:
            times = [r["actual_time_s"] for r in complete]
            lower = [r["lower_bound_s"] for r in complete]
            group.update(mean_actual_time_s=statistics.mean(times), min_actual_time_s=min(times),
                         max_actual_time_s=max(times), mean_time_per_source_s=statistics.mean(times)/n,
                         min_time_per_source_s=min(times)/n, max_time_per_source_s=max(times)/n,
                         mean_lower_bound_s=statistics.mean(lower),
                         sum_time_over_sum_lower_bound=sum(times)/sum(lower))
        result.append(group)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-root", type=Path, required=True)
    parser.add_argument("--simulator-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-per-count", type=int, default=3)
    parser.add_argument("--maximum-runs", type=int, default=90)
    parser.add_argument("--fixed-runs", type=int, help="Run exactly this many cases regardless of source-count groups")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.fixed_runs is not None and not 1 <= args.fixed_runs <= 200:
        raise ValueError("fixed-runs must be in [1,200]")
    run_limit = args.fixed_runs if args.fixed_runs is not None else args.maximum_runs
    source = args.strategy_root.resolve(strict=True)
    spec_path = source / SPEC_NAME
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != COMMIT
            or sha(spec_path) != SPEC_SHA):
        raise ValueError("Qualified Q3 source/spec identity mismatch")
    spec = json.loads(spec_path.read_bytes())
    sys.path.insert(0, str(MAIN / "src"))
    from practice_control.branch_worker import load_solver
    runner, solver = load_solver(source, spec, spec["kwargs"])
    from practice_control.bridge import PracticeBridge
    from experiments import session_lower_bounds
    tracked = list((source / "src").rglob("*.py")) + [spec_path]
    tracked += list((MAIN / "src/practice_control").glob("*.py"))
    tracked += [MAIN / "experiments/register_practice.py", Path(session_lower_bounds.__file__)]
    frozen = {p: sha(p) for p in tracked}

    def verify():
        if any(sha(p) != digest for p, digest in frozen.items()):
            raise ValueError("Frozen policy/controller changed; no new practice will start")

    verify()
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, "source_commit": COMMIT,
                          "strategy": spec["name"], "simulator_requests_sent": False}))
        return 0
    if args.output is None or args.simulator_dir is None:
        raise ValueError("Need simulator-dir and output")
    if not 1 <= args.minimum_per_count <= 10 or not 1 <= args.maximum_runs <= 200:
        raise ValueError("Invalid sampling budget")
    robot = os.environ.get("CUMCM_ROBOT_ID")
    runner.validate_run(3, "baseline", 1, 20000, robot)
    sim = args.simulator_dir.resolve(strict=True)
    if not (sim / "jammers-simulator-full.exe").is_file():
        raise ValueError("Need original simulator installation directory")
    output = args.output.resolve()
    protocol = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "problem": 3,
        "mode": "practice", "source_commit": COMMIT, "spec": spec, "spec_sha256": sha(spec_path),
        "runner_sha256": sha(Path(__file__)),
        "minimum_per_source_count": args.minimum_per_count if args.fixed_runs is None else None,
        "maximum_runs": run_limit, "fixed_runs": args.fixed_runs,
        "per_source_time_definition": "Full task billed virtual time divided by official terminal source count; includes discovery, localization and clearance.",
        "lower_bound_kind": "historical_conditional_all_clear_containing_region_bound",
        "stopping_rule": ("Exactly fixed_runs cases regardless of source count, or error/STOP; retain all sampled cases."
                          if args.fixed_runs is not None else
                          "Every N=10..16 has minimum completed runs, or maximum/error/STOP; retain all sampled cases."),
        "pooled_per_source_definition": "sum(actual_time_s)/sum(source_count); pooled lower bound uses the same source denominator",
        "count_visibility": "Source count is read only from the matching completed official practice result."}
    rows, failures, status, started_runs = [], [], "running", 0

    def snapshot():
        save(output / "summary.json", {"protocol": protocol, "status": status,
             "attempted_runs": started_runs, "completed_runs": sum(r.get("all_clear", False) for r in rows),
             "failures": failures, "rows": rows, "groups": groups(rows)})

    with runner.controller_lock(sim / ".practice-control/controller.lock"), PracticeBridge(19226) as bridge:
        state = bridge.current_test()
        if (not isinstance(state, dict) or state.get("active") is not False
                or state.get("mode") not in (None, "", "practice") or state.get("case_code") or state.get("phase")):
            raise ValueError("Simulator is not explicitly idle; existing session remains untouched")
        output.mkdir(parents=True, exist_ok=False)
        save(output / "protocol.json", protocol)
        try:
            for index in range(1, run_limit + 1):
                if (output / "STOP").exists():
                    status = "stop_requested"
                    break
                verify()
                started_runs += 1
                run = output / f"run-{index:03d}"
                record = runner.run_once(bridge, problem=3, robot_id=robot, variant="baseline", max_actions=20000,
                    output=run, simulator_dir=sim, solver=solver, method_label=spec["name"],
                    method_metadata={"source_commit": COMMIT, "spec": spec, "spec_sha256": SPEC_SHA})
                verify()
                n = record["source_total"]
                full = record["completed"] is True and record["cleared_count"] == n
                row = {"run": index, "source_count": n, "all_clear": full,
                       "actual_time_s": record["virtual_time_s"], "program_wall_time_s": record["program_wall_time_s"]}
                if not full or type(n) is not int or not 10 <= n <= 16:
                    rows.append(row)
                    status = "incomplete_practice"
                    failures.append({"run": index, "kind": status})
                    break
                bounds = session_lower_bounds.analyze(run / "summary.json")
                lower = bounds["conditional_guaranteed_all_clear_lower_s"]
                if (bounds["problem"] != 3 or bounds["observed_cleared_sources"] != n
                        or bounds["actual_virtual_time_s"] != record["virtual_time_s"] or lower <= 0):
                    raise ValueError("Registered run and lower bound mismatch")
                bounds.update(official_source_total=n, official_all_clear_verified=True,
                              time_to_conditional_lower_bound_ratio=record["virtual_time_s"]/lower)
                save(run / "lower_bounds.json", bounds)
                row.update(lower_bound_s=lower, time_over_lower_bound=record["virtual_time_s"]/lower,
                           time_per_source_s=record["virtual_time_s"]/n,
                           summary_sha256=sha(run / "summary.json"), lower_bounds_sha256=sha(run / "lower_bounds.json"))
                rows.append(row)
                snapshot()
                print(json.dumps({"completed_runs": len(rows), "latest": row,
                      "counts": {g["source_count"]:g["runs"] for g in groups(rows)}}), flush=True)
                if args.fixed_runs is None and all(g["runs"] >= args.minimum_per_count for g in groups(rows)):
                    status = "source_counts_covered"
                    break
            else:
                status = "fixed_count_completed" if args.fixed_runs is not None else "maximum_runs"
        except Exception as exc:
            status = "error"
            failures.append({"run": started_runs, "kind": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            snapshot()
    print(json.dumps({"status": status, "groups": groups(rows)}), flush=True)
    return 0 if status in {"source_counts_covered", "fixed_count_completed", "stop_requested"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
