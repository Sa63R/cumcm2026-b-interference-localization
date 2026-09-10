"""Fixed-config, paired Q3 development/holdout research; never official tests."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from strategies import run_search

BASELINE = "baseline_current"
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def source_manifest():
    files = sorted((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def load_configs(path):
    raw = path.read_bytes()
    configs = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(configs, list) or not configs:
        raise ValueError("--configs must contain a nonempty JSON list")
    names = {BASELINE}
    for entry in configs:
        if not isinstance(entry, dict) or set(entry) != {"name", "efficient_config"}:
            raise ValueError("Each config requires exactly name and efficient_config")
        name = entry["name"]
        if not isinstance(name, str) or not name.strip() or not name.isprintable():
            raise ValueError("Configuration names must be nonempty printable strings")
        if name in names:
            raise ValueError(f"Duplicate or reserved configuration name: {name}")
        names.add(name)
        if not isinstance(entry["efficient_config"], dict):
            raise ValueError("efficient_config must be a JSON object")
    # Reject nonfinite JSON values before creating any output or running a case.
    json.dumps(configs, allow_nan=False)
    return configs, raw


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def bootstrap_mean_interval(values, *, seed=20260910, samples=2000):
    if not values:
        return None
    rng = random.Random(seed)
    count = len(values)
    estimates = [sum(rng.choices(values, k=count)) / count for _ in range(samples)]
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def summarize(rows):
    groups = []
    baselines = {row["case_id"]: row for row in rows if row["strategy"] == BASELINE}
    for kind in ("random", "hard"):
        for label in dict.fromkeys(row["strategy"] for row in rows):
            subset = [row for row in rows if row["case_kind"] == kind and row["strategy"] == label]
            if not subset:
                continue
            pair_rows = [(baselines[row["case_id"]], row) for row in subset]
            raw_savings = [base["virtual_time_s"] - row["virtual_time_s"] for base, row in pair_rows]
            valid_pairs = [(base, row) for base, row in pair_rows if base["successful"] and row["successful"]]
            paired_savings = [base["virtual_time_s"] - row["virtual_time_s"] for base, row in valid_pairs]
            times = [row["virtual_time_s"] for row in subset]
            averages = [row["mean_time_per_cleared_s"] for row in subset
                        if row["mean_time_per_cleared_s"] is not None]
            clear_count = sum(row["cleared_total"] for row in subset)
            base_mean = statistics.mean(base["virtual_time_s"] for base, _ in pair_rows)
            eligible = len(valid_pairs) == len(pair_rows)
            group = {
                "problem": 3, "case_kind": kind, "strategy": label, "runs": len(subset),
                "successful_runs": sum(row["successful"] for row in subset),
                "failed_runs": sum(not row["successful"] for row in subset),
                "incomplete_clear_runs": sum(not row["all_cleared"] for row in subset),
                "uncertified_runs": sum(not row["completion_certified"] for row in subset),
                "error_runs": sum(bool(row["error"]) for row in subset),
                "all_clear_runs": sum(row["all_cleared"] for row in subset),
                "source_total": sum(row["source_total"] for row in subset),
                "cleared_total": clear_count,
                "pooled_clear_fraction": clear_count / sum(row["source_total"] for row in subset),
                "mean_virtual_time_s": statistics.mean(times),
                "median_virtual_time_s": statistics.median(times),
                "p90_virtual_time_s": percentile(times, .90),
                "p95_virtual_time_s": percentile(times, .95),
                "max_virtual_time_s": max(times),
                "mean_case_time_per_cleared_s": statistics.mean(averages) if averages else None,
                "p90_case_time_per_cleared_s": percentile(averages, .90),
                "p95_case_time_per_cleared_s": percentile(averages, .95),
                "pooled_time_per_cleared_s": sum(times) / clear_count if clear_count else None,
                "mean_runtime_s": statistics.mean(row["program_runtime_s"] for row in subset),
                "p95_runtime_s": percentile([row["program_runtime_s"] for row in subset], .95),
                "mean_measurements": statistics.mean(row["measurement_count"] for row in subset),
                "mean_failed_clears": statistics.mean(row["failed_clear_count"] for row in subset),
                "mean_movement_m": statistics.mean(row["movement_m"] for row in subset),
                "max_actions": max(row["action_count"] for row in subset),
                "mean_time_breakdown_s": {name: statistics.mean(row[name] for row in subset)
                                          for name in COMPONENTS},
                "all_pairs_complete_for_time_comparison": eligible,
                "paired_successful_runs": len(valid_pairs),
                "mean_paired_seconds_saved": statistics.mean(paired_savings) if paired_savings else None,
                "paired_mean_saving_ci95_s": bootstrap_mean_interval(paired_savings) if kind == "random" else None,
                "raw_all_run_mean_paired_seconds_saved": statistics.mean(raw_savings),
                "relative_mean_time_reduction_pct": 100 * (1 - statistics.mean(times) / base_mean)
                                                    if eligible and base_mean > 0 else None,
                "paired_time_note": "Positive is faster than baseline_current. CI uses successful pairs only; any failures invalidate an unconditional efficiency comparison.",
            }
            groups.append(group)
    return groups


def run_one(case, label, config, *, max_actions, output, save_traces):
    simulator = LocalResearchSimulator(case)
    client = simulator.client()  # The policy receives observations only.
    result, errors, exception_trace = None, [], None
    interrupted = False
    started = time.perf_counter()
    try:
        arguments = {"problem": 3, "active_policy": "center", "max_actions": max_actions}
        if label == BASELINE:
            arguments["variant"] = "adaptive"
        else:
            arguments.update(variant="efficient", efficient_config=dict(config))
        result = run_search(client, **arguments)
        if result.error:
            errors.append(result.error)
        if result.exit_error:
            errors.append(result.exit_error)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        exception_trace = traceback.format_exc()
    except KeyboardInterrupt:
        interrupted = True
        errors.append("KeyboardInterrupt: research run interrupted")
        exception_trace = traceback.format_exc()
    finally:
        # Strategy exceptions/budgets still get a legal exit when possible.
        # Deadline failures are ended by the research harness, never labelled a
        # successful /exit. Truth is consulted only after all actions terminate.
        if client.state.session == "active" and client.pending_request is None:
            try:
                client.exit()
            except Exception as exc:
                errors.append(f"CleanupExit: {type(exc).__name__}: {exc}")
        simulator.finish_for_evaluation()
    elapsed = time.perf_counter() - started
    evaluation = simulator.evaluation()
    certificate = bool(result and result.completion_certified_under_model)
    if client.state.cleared_count != evaluation["cleared_total"]:
        errors.append("InvariantViolation: client/evaluator clear counts differ")
    if result is not None and result.cleared_count != evaluation["cleared_total"]:
        errors.append("InvariantViolation: strategy/evaluator clear counts differ")
    if not math.isclose(evaluation["virtual_time_s"], sum(evaluation["time_breakdown_s"].values()), abs_tol=1e-6):
        errors.append("InvariantViolation: virtual-time decomposition differs")
    if result is not None and not math.isclose(result.virtual_time_s, evaluation["virtual_time_s"], abs_tol=1e-6):
        errors.append("InvariantViolation: strategy/evaluator virtual times differ")
    if certificate and not evaluation["all_cleared"]:
        errors.append("InvariantViolation: false completeness certificate")
    exited = client.state.session == "exited" and client.pending_request is None
    successful = evaluation["all_cleared"] and certificate and exited and not errors
    trace_path = ""
    # Retain traces for failures even in a fast development run.
    if save_traces or not successful:
        trace_name = f"{case.case_id}--{hashlib.sha256(label.encode('utf-8')).hexdigest()[:12]}.json.gz"
        trace_path = "traces/" + trace_name
        (output / "traces").mkdir(exist_ok=True)
        trace = {
            "data_origin": "synthetic_research", "official_practice": False,
            "strategy": label, "efficient_config": config,
            "legal_observation_history": simulator.observation_history(),
            "search": result.as_dict() if result is not None else None,
            "final_client_state": client.state.snapshot(),
            "evaluation": evaluation, "evaluation_phase": "after_session_termination",
            "accepted_exit": exited,
            "errors": errors, "exception_traceback": exception_trace,
        }
        with gzip.open(output / trace_path, "wt", encoding="utf-8") as stream:
            json.dump(trace, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return {
        "problem": 3, "case_id": case.case_id,
        "case_kind": "random" if "-random-" in case.case_id else "hard",
        "seed": case.seed, "strategy": label,
        "source_total": evaluation["source_total"], "cleared_total": evaluation["cleared_total"],
        "all_cleared": evaluation["all_cleared"], "cleared_fraction": evaluation["cleared_fraction"],
        "completion_certified": certificate, "successful": bool(successful), "accepted_exit": exited,
        "interrupted": interrupted,
        "completion_reason": result.completion_reason if result is not None else "uncaught_exception",
        "error": " | ".join(map(str, errors)),
        "virtual_time_s": evaluation["virtual_time_s"],
        "mean_time_per_cleared_s": evaluation["mean_time_per_cleared_s"],
        "program_runtime_s": elapsed, "measurement_count": evaluation["measurement_count"],
        "failed_clear_count": evaluation["failed_clear_count"], "action_count": evaluation["action_count"],
        "movement_m": evaluation["time_breakdown_s"]["movement_s"] * 5,
        **evaluation["time_breakdown_s"], "trace": trace_path,
    }


def execute(args):
    configs, raw_config = load_configs(args.configs)
    if "efficient_config" not in inspect.signature(run_search).parameters:
        raise ValueError("run_search does not yet expose efficient_config; wait for the new strategy interface")
    if args.random_cases < 1 or args.start_seed < 0 or args.max_actions < 2:
        raise ValueError("Require random-cases >= 1, start-seed >= 0, max-actions >= 2")
    if args.stage == "holdout" and not args.save_traces:
        raise ValueError("Holdout runs require --save-traces")
    end_seed = args.start_seed + args.random_cases - 1
    development = None
    if args.development_manifest:
        development = json.loads(args.development_manifest.read_text(encoding="utf-8"))
        previous = development["random_seed_interval_inclusive"]
        if args.stage == "holdout" and max(args.start_seed, previous[0]) <= min(end_seed, previous[1]):
            raise ValueError("Holdout seeds overlap the supplied development seed interval")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
    before = source_manifest()
    metadata = {
        "data_origin": "synthetic_research", "official_practice": False, "official_formal": False,
        "stage": args.stage, "problem": 3, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision, "python": sys.version, "platform": platform.platform(),
        "random_cases": args.random_cases, "random_seed_interval_inclusive": [args.start_seed, end_seed],
        "include_hard": args.include_hard, "hard_cases_are_independent_holdout": False,
        "hard_case_note": "The seven shared deterministic stress cases are reported separately and are not an independent holdout sample.",
        "development_manifest": str(args.development_manifest) if args.development_manifest else None,
        "development_seed_interval_inclusive": development["random_seed_interval_inclusive"] if development else None,
        "holdout_seed_disjointness_checked": args.stage == "holdout" and development is not None,
        "holdout_note": "Holdout is user-declared; no automatic best-config selection occurs. Without a development manifest, seed independence is not verified.",
        "configs": configs, "configs_sha256": hashlib.sha256(raw_config).hexdigest(),
        "configuration_selection": "All supplied configurations fixed before this run; no automatic tuning or post-hoc selection",
        "baseline": {"name": BASELINE, "variant": "adaptive", "active_policy": "center"},
        "save_traces": args.save_traces, "failure_traces_always_saved": True,
        "max_actions": args.max_actions, "source_sha256_start": before,
        "source_sha256_end": None, "source_changed_during_run": None,
        "runtime_scope": "In-process research engine and solver including exit; excludes evaluation and artifact compression; not official HTTP runtime",
        "comparison": "Identical source scenario and location-fixed error function for every policy; different paths can query different locations",
        "distribution": "Uniform area in radius-1800 disk, uniform reception radius [1000,1500], 10..16 sources, unique sampled channels, bounded quantized direction error",
        "bootstrap": {"seed": 20260910, "samples": 2000, "unit": "paired random case", "interval": "percentile 95%"},
        "run_status": "running",
    }
    (args.output / "configs.json").write_bytes(raw_config)
    dump_json(args.output / "manifest.json", metadata)
    rows = []
    cases = [random_scenario(3, seed) for seed in range(args.start_seed, end_seed + 1)]
    if args.include_hard:
        cases.extend(difficult_scenarios(3))
    policies = [(BASELINE, None)] + [(entry["name"], entry["efficient_config"]) for entry in configs]
    total_runs = len(cases) * len(policies)
    interrupted = False
    try:
        with (args.output / "runs.csv").open("x", encoding="utf-8-sig", newline="") as stream:
            writer = None
            for case in cases:
                for label, config in policies:
                    row = run_one(case, label, config, max_actions=args.max_actions,
                                  output=args.output, save_traces=args.save_traces)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
                    if row["interrupted"]:
                        raise KeyboardInterrupt
                    if len(rows) % 20 == 0 or len(rows) == total_runs or not row["successful"]:
                        print(f"[{len(rows)}/{total_runs}] {case.case_id} {label}: "
                              f"{row['cleared_total']}/{row['source_total']} "
                              f"{row['virtual_time_s']:.2f}s virtual {row['program_runtime_s']:.3f}s wall "
                              f"success={row['successful']}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; completed rows and available traces are retained.", file=sys.stderr, flush=True)
    finally:
        after = source_manifest()
        metadata.update(source_sha256_end=after, source_changed_during_run=before != after,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        expected_runs=total_runs, completed_runs=len(rows),
                        run_status="interrupted" if interrupted else "completed" if len(rows) == total_runs else "failed")
        dump_json(args.output / "manifest.json", metadata)
        dump_json(args.output / "summary.json", {
            "data_origin": "synthetic_research", "stage": args.stage,
            "source_consistent": before == after, "expected_runs": total_runs,
            "completed_runs": len(rows), "failed_runs": sum(not row["successful"] for row in rows),
            "groups": summarize(rows),
        })
    if interrupted:
        return 130
    if before != after:
        print("Source files changed during the run; results are not a fixed-version comparison.", file=sys.stderr)
        return 2
    return 0 if len(rows) == total_runs and all(row["successful"] for row in rows) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are never overwritten")
    parser.add_argument("--configs", type=Path, required=True, help="JSON list of name/efficient_config entries")
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--random-cases", type=int, default=12)
    hard = parser.add_mutually_exclusive_group()
    hard.add_argument("--include-hard", dest="include_hard", action="store_true")
    hard.add_argument("--no-hard", dest="include_hard", action="store_false")
    parser.set_defaults(include_hard=True)
    parser.add_argument("--save-traces", action="store_true", help="Save gzipped legal actions and post-exit evaluation for every case")
    parser.add_argument("--stage", choices=("development", "holdout"), default="development")
    parser.add_argument("--development-manifest", type=Path, help="Earlier development manifest, to check holdout seed disjointness")
    parser.add_argument("--max-actions", type=int, default=20000)
    args = parser.parse_args(argv)
    try:
        return execute(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
