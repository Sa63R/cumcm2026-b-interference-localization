"""Reproducible, paired Q3 local research on new development/holdout seeds.

Every worker runs the baseline followed by the selected candidates on the same
immutable scenario. Only scenarios are parallelized. Raw traces are retained
even for failures; no official simulator, network, or old sealed seeds are used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
from queue import Empty
import random
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
BASE_SPEC = ROOT / "experiments/state_search_candidate_relocating_cover_v1.json"
SEED_INTERVALS = {"development": (220000, 230000), "holdout": (230000, 240000),
                  "stress": (240000, 250000)}
STRESS_FAMILIES = ("minimum_radius", "boundary", "cluster", "positive_error",
                   "negative_error", "alternating_error", "narrow_strip")
LIMITS = {"max_actions": 10000, "max_active_probes": 6,
          "real_seconds_per_method": 1200, "virtual_seconds_per_method": 360000}
BOOTSTRAP_SEED = 20260912
BOOTSTRAP_SAMPLES = 5000
WORKER_SETTINGS = None
WORKER_QUEUE = None


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha256(canonical(value).encode("utf-8"))


def dump_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_files():
    return sorted({*(ROOT / "src").rglob("*.py"), Path(__file__).resolve(), BASE_SPEC})


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()) for p in source_files()}


def source_changes(expected):
    actual = source_hashes()
    return [name for name in sorted(expected.keys() | actual.keys())
            if expected.get(name) != actual.get(name)]


def assert_sources(expected):
    changed = source_changes(expected)
    if changed:
        raise RuntimeError("Source changed during evaluation: " + ", ".join(changed))


def snapshot_sources(output):
    records = {}
    for source in source_files():
        relative = source.relative_to(ROOT)
        contents = source.read_bytes()
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        records[relative.as_posix()] = sha256(contents)
        if sha256(target.read_bytes()) != records[relative.as_posix()]:
            raise RuntimeError(f"Snapshot verification failed: {relative}")
    assert_sources(records)
    return records


def load_specs(modes):
    sys.path.insert(0, str(ROOT / "src"))
    from strategies.state_search import StateSearchConfig

    original = json.loads(BASE_SPEC.read_text(encoding="utf-8-sig"))
    baseline = deepcopy(original)
    # Materialize function/config defaults so the manifest describes the exact
    # effective call, while preserving the unmodified reference JSON separately.
    baseline["kwargs"] = {"problem": 3, "max_actions": LIMITS["max_actions"],
                          "max_active_probes": LIMITS["max_active_probes"],
                          **baseline["kwargs"]}
    baseline["kwargs"]["config"] = asdict(StateSearchConfig.parse(baseline["kwargs"]["config"]))
    specs = {"baseline": baseline}
    for mode in modes:
        candidate = deepcopy(baseline)
        candidate["name"] = f"clear_region_{mode}"
        candidate["entrypoint"] = "strategies.clear_region_state_search:run_clear_region_state_search"
        candidate["kwargs"]["clear_mode"] = "nearest" if mode in {"nearest", "both"} else "disabled"
        candidate["kwargs"]["history_silence"] = mode in {"history", "both"}
        specs[mode] = candidate
    return original, specs


def make_case(stage, seed):
    from simulation import random_scenario

    case = random_scenario(3, seed)
    if stage != "stress":
        return case
    family = STRESS_FAMILIES[(seed - SEED_INTERVALS["stress"][0]) % len(STRESS_FAMILIES)]
    rng = random.Random(seed + 991731)
    phase = rng.uniform(0, 2 * math.pi)
    n = len(case.sources)
    sources = tuple(replace(s, reception_radius_m=1000.0) for s in case.sources)
    error_mode = "uniform"
    if family == "boundary":
        sources = tuple(replace(s, x=1799.9 * math.cos(phase + 2 * math.pi * i / n),
                                y=1799.9 * math.sin(phase + 2 * math.pi * i / n))
                        for i, s in enumerate(sources))
    elif family == "cluster":
        cx, cy = 1000 * math.cos(phase), 1000 * math.sin(phase)
        sources = tuple(replace(s, x=cx + rng.uniform(-10, 10), y=cy + rng.uniform(-10, 10))
                        for s in sources)
    elif family == "narrow_strip":
        sources = tuple(replace(s, x=(850 + 45 * i) * math.cos(phase) - (-1)**i * .1 * math.sin(phase),
                                y=(850 + 45 * i) * math.sin(phase) + (-1)**i * .1 * math.cos(phase))
                        for i, s in enumerate(sources))
    elif family != "minimum_radius":
        error_mode = {"positive_error": "positive_extreme", "negative_error": "negative_extreme",
                      "alternating_error": "alternating_extreme"}[family]
    return replace(case, case_id=f"q3-clear-region-stress-{family}-{seed}", sources=sources,
                   error_mode=error_mode, description=f"New-seed synthetic stress family: {family}")


def log_summary(parameters):
    """Compact diagnostics; the unabridged strategy parameters remain in traces."""
    summary = {"scalars": {}, "logs": {}}
    for name, value in parameters.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            summary["scalars"][name] = value
        elif isinstance(value, list):
            numeric, true_counts = {}, {}
            for item in value:
                if not isinstance(item, dict):
                    continue
                for key, number in item.items():
                    if isinstance(number, bool):
                        true_counts[key] = true_counts.get(key, 0) + int(number)
                    elif isinstance(number, (int, float)) and math.isfinite(number):
                        numeric[key] = numeric.get(key, 0) + number
            summary["logs"][name] = {"entries": len(value), "numeric_sums": numeric,
                                      "true_counts": true_counts}
    return summary


def run_method(case, label, spec, settings):
    from simulation import LocalResearchSimulator

    simulator = LocalResearchSimulator(case, max_real_duration_s=LIMITS["real_seconds_per_method"],
                                       max_virtual_duration_s=LIMITS["virtual_seconds_per_method"])
    client = simulator.client()
    report, errors, exception_trace = None, [], None
    started = time.perf_counter()
    try:
        module, function = spec["entrypoint"].split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        report = callback(client, **deepcopy(spec["kwargs"]))
        errors.extend(str(error) for error in (report.error, report.exit_error) if error)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        exception_trace = traceback.format_exc()
    finally:
        if client.state.session == "active" and client.pending_request is None:
            try:
                client.exit()
            except Exception as error:
                errors.append(f"CleanupExit: {type(error).__name__}: {error}")
        simulator.finish_for_evaluation()
    elapsed = time.perf_counter() - started
    # Ground truth is first consulted after policy termination.
    evaluation = simulator.evaluation()
    certificate = bool(report and report.completion_certified_under_model)
    exited = (client.state.session == "exited" and client.pending_request is None
              and evaluation["simulator_stop_reason"] == "exited")
    if client.state.cleared_count != evaluation["cleared_total"]:
        errors.append("Client/evaluator clear-count mismatch")
    if not math.isclose(sum(evaluation["time_breakdown_s"].values()), evaluation["virtual_time_s"], abs_tol=1e-6):
        errors.append("Virtual-time ledger mismatch")
    if report and (report.cleared_count != evaluation["cleared_total"] or not math.isclose(
            report.virtual_time_s, evaluation["virtual_time_s"], abs_tol=1e-6)):
        errors.append("Strategy/evaluator count or time mismatch")
    if certificate and not evaluation["all_cleared"]:
        errors.append("False completeness certificate")
    success = bool(evaluation["all_cleared"] and certificate and exited and not errors)
    total, cleared, virtual = (evaluation["source_total"], evaluation["cleared_total"],
                               evaluation["virtual_time_s"])
    trace_relative = f"traces/{case.case_id}--{label}.json.gz"
    parameters = report.strategy_parameters if report else {}
    row = {"case_id": case.case_id, "seed": case.seed, "stage": settings["stage"],
           "case_sha256": digest(case.evaluation_config()), "method": label,
           "strategy": spec["name"], "spec_sha256": digest(spec),
           "success": success, "all_cleared": evaluation["all_cleared"],
           "completion_certified": certificate, "accepted_exit": exited,
           "source_total": total, "cleared_total": cleared,
           "cleared_fraction": evaluation["cleared_fraction"], "virtual_time_s": virtual,
           "penalized_time_s": virtual if success else LIMITS["virtual_seconds_per_method"],
           "time_per_target_s": virtual / total,
           "time_per_cleared_s": virtual / cleared if cleared else None,
           "time_breakdown_s": evaluation["time_breakdown_s"],
           "failed_clear_count": evaluation["failed_clear_count"],
           "measurement_count": evaluation["measurement_count"],
           "action_count": evaluation["action_count"], "wall_s": elapsed,
           "strategy_log_summary": log_summary(parameters), "errors": errors,
           "trace": trace_relative}
    trace = {"row": row, "spec": spec, "report": report.as_dict() if report else None,
             "evaluation": evaluation, "observation_history": simulator.observation_history(),
             "final_client_state": client.state.snapshot(), "exception_traceback": exception_trace,
             "evaluation_phase": "after_policy_termination", "data_origin": "synthetic_research"}
    with gzip.open(Path(settings["output"]) / trace_relative, "wt", encoding="utf-8") as stream:
        json.dump(trace, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return row


def initialize_worker(settings, queue):
    global WORKER_SETTINGS, WORKER_QUEUE
    WORKER_SETTINGS, WORKER_QUEUE = settings, queue
    sys.path.insert(0, str(ROOT / "src"))


def run_pair(seed):
    settings = WORKER_SETTINGS
    assert_sources(settings["source_sha256"])
    case = make_case(settings["stage"], seed)
    case_hash = digest(case.evaluation_config())
    for label, spec in settings["specs"].items():
        row = run_method(case, label, spec, settings)
        if row["case_sha256"] != case_hash:
            raise RuntimeError(f"Scenario mutated in case {seed}, method {label}")
        WORKER_QUEUE.put(row)
    assert_sources(settings["source_sha256"])
    return {"seed": seed, "case_sha256": case_hash}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def bootstrap_ci(values):
    if not values:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = [sum(rng.choices(values, k=n)) / n for _ in range(BOOTSTRAP_SAMPLES)]
    return [percentile(means, .025), percentile(means, .975)]


def method_summary(rows):
    times = [r["virtual_time_s"] for r in rows]
    penalized = [r["penalized_time_s"] for r in rows]
    targets, cleared = sum(r["source_total"] for r in rows), sum(r["cleared_total"] for r in rows)
    mean = lambda values: statistics.mean(values) if values else None
    per_clear = [r["time_per_cleared_s"] for r in rows if r["time_per_cleared_s"] is not None]
    return {"runs": len(rows), "successes": sum(r["success"] for r in rows),
            "failures": sum(not r["success"] for r in rows), "source_total": targets,
            "cleared_total": cleared, "raw_mean_time_s": mean(times),
            "raw_p95_time_s": percentile(times, .95), "raw_max_time_s": max(times, default=None),
            "penalized_mean_time_s": mean(penalized), "penalized_p95_time_s": percentile(penalized, .95),
            "pooled_time_per_target_s": sum(times) / targets if targets else None,
            "mean_case_time_per_target_s": mean([r["time_per_target_s"] for r in rows]),
            "pooled_time_per_cleared_s": sum(times) / cleared if cleared else None,
            "mean_case_time_per_cleared_s": mean(per_clear),
            "cases_with_zero_cleared": sum(r["cleared_total"] == 0 for r in rows),
            "penalized_pooled_time_per_target_s": sum(penalized) / targets if targets else None,
            "mean_wall_s": mean([r["wall_s"] for r in rows]),
            "p95_wall_s": percentile([r["wall_s"] for r in rows], .95),
            "mean_measurement_count": mean([r["measurement_count"] for r in rows]),
            "failed_clear_count": sum(r["failed_clear_count"] for r in rows),
            "mean_time_breakdown_s": {key: mean([r["time_breakdown_s"][key] for r in rows])
                                      for key in rows[0]["time_breakdown_s"]} if rows else {}}


def paired_summary(base_rows, candidate_rows):
    baseline = {r["seed"]: r for r in base_rows}
    candidate = {r["seed"]: r for r in candidate_rows}
    common = sorted(baseline.keys() & candidate.keys())
    pairs = [(baseline[s], candidate[s]) for s in common]
    if any(b["case_sha256"] != c["case_sha256"] for b, c in pairs):
        raise RuntimeError("Paired scenarios differ")
    savings = [b["penalized_time_s"] - c["penalized_time_s"] for b, c in pairs]
    raw_savings = [b["virtual_time_s"] - c["virtual_time_s"] for b, c in pairs]
    bm = statistics.mean(b["penalized_time_s"] for b, _ in pairs) if pairs else None
    cm = statistics.mean(c["penalized_time_s"] for _, c in pairs) if pairs else None
    worst = min(range(len(savings)), key=savings.__getitem__) if savings else None
    both_successful = all(b["success"] and c["success"] for b, c in pairs)
    return {"paired_cases": len(pairs), "unpaired_baseline_seeds": sorted(baseline.keys() - candidate.keys()),
            "unpaired_candidate_seeds": sorted(candidate.keys() - baseline.keys()),
            "all_pairs_successful": bool(pairs) and both_successful,
            "all_candidate_clears_successful": all(c["failed_clear_count"] == 0 for _, c in pairs),
            "comparison_metric": "penalized_time_s; failed method = 360000 s",
            "baseline_mean_s": bm, "candidate_mean_s": cm,
            "mean_saving_s": statistics.mean(savings) if savings else None,
            "mean_reduction_fraction": 1 - cm / bm if bm else None,
            "mean_saving_ci95_s": bootstrap_ci(savings),
            "wins": sum(s > 1e-6 for s in savings), "losses": sum(s < -1e-6 for s in savings),
            "ties": sum(abs(s) <= 1e-6 for s in savings),
            "max_regression_s": max(0.0, -min(savings)) if savings else None,
            "worst_delta_s": -min(savings) if savings else None,
            "worst_case_seed": common[worst] if worst is not None else None,
            "raw_mean_saving_s_descriptive_only": statistics.mean(raw_savings) if raw_savings else None,
            "raw_mean_saving_s_all_successful": statistics.mean(raw_savings) if pairs and both_successful else None,
            "ci_scope": "Fixed 5000 paired bootstrap resamples of case-level penalized savings; exploratory when used to select among candidates."}


def execute(args):
    low, high = SEED_INTERVALS[args.stage]
    if args.count <= 0 or args.workers <= 0 or not low <= args.seed < args.seed + args.count <= high:
        raise ValueError(f"Require count/workers > 0 and {args.stage} seeds in [{low}, {high})")
    modes = list(dict.fromkeys(args.candidate_mode))
    if args.stage == "holdout" and len(modes) != 1:
        raise ValueError("Select one candidate using development data before running holdout")
    if args.output.exists():
        raise FileExistsError(f"Output already exists; use a new directory: {args.output}")
    original, specs = load_specs(modes)
    hashes_before = source_hashes()
    freeze = None
    if args.freeze_manifest:
        freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
        if freeze["source_sha256"] != hashes_before:
            raise ValueError("Source does not match the supplied freeze manifest")
        if any(freeze["specs"].get(label) != spec for label, spec in specs.items()):
            raise ValueError("Selected exact specs do not match the supplied freeze manifest")
    # Fail before generating scenarios if an entrypoint is unavailable.
    for spec in specs.values():
        module, function = spec["entrypoint"].split(":", 1)
        if not callable(getattr(importlib.import_module(module), function)):
            raise ValueError(f"Not callable: {spec['entrypoint']}")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "traces").mkdir()
    sources = snapshot_sources(args.output)
    if sources != hashes_before:
        raise RuntimeError("Source changed while loading specs/snapshotting")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"],
                                    cwd=ROOT, text=True)
    manifest = {"schema_version": 1, "data_origin": "synthetic_research", "problem": 3,
                "official_practice": False, "official_formal": False,
                "created_utc": datetime.now(timezone.utc).isoformat(), "stage": args.stage,
                "seed_start": args.seed, "seed_stop_exclusive": args.seed + args.count,
                "count": args.count, "workers": args.workers, "method_order": list(specs),
                "baseline_original_spec": original, "specs": specs,
                "spec_sha256": {label: digest(spec) for label, spec in specs.items()},
                "base_commit": commit, "git_status_src_experiments": dirty,
                "source_sha256": sources, "source_snapshot": "source_snapshot",
                "python": sys.version, "python_executable": sys.executable,
                "platform": platform.platform(), "machine": platform.machine(),
                "limits": LIMITS, "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "freeze_manifest": str(args.freeze_manifest.resolve()) if args.freeze_manifest else None,
                "freeze_manifest_sha256": sha256(args.freeze_manifest.read_bytes()) if args.freeze_manifest else None,
                "freeze_source_and_selected_specs_verified": freeze is not None,
                "random_distribution": "random_scenario(3, seed): assumed local research distribution, not the official generator law",
                "stress_families": list(STRESS_FAMILIES) if args.stage == "stress" else None,
                "source_checks": "Before and after every complete case; before and after the complete experiment. Any mismatch invalidates the experiment.",
                "success_rule": "all_cleared AND completion certificate AND accepted exit AND no report, exit, exception, or invariant errors",
                "failure_rule": "Retain all attempted methods and failures; assign unsuccessful methods 360000 seconds in comparisons. Missing cases invalidate completeness.",
                "selection_rule": "Choose using development only; freeze code/config before holdout. Do not tune on viewed holdout outcomes.",
                "wall_time_caveat": "Methods are sequential within a worker, but workers run concurrently. Wall times include CPU contention and fixed-order warmup effects; do not infer serial performance from these values."}
    dump_json(args.output / "manifest.json", manifest)
    started = time.perf_counter()
    rows, seen, completed, errors = [], set(), [], []
    settings = {"stage": args.stage, "output": str(args.output), "source_sha256": sources, "specs": specs}
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    expected_runs = args.count * len(specs)

    def write_progress():
        dump_json(args.output / "progress.json", {
            "updated_utc": datetime.now(timezone.utc).isoformat(), "completed_cases": len(completed),
            "expected_cases": args.count, "completed_runs": len(rows), "expected_runs": expected_runs,
            "successful_runs": sum(r["success"] for r in rows),
            "failed_runs": sum(not r["success"] for r in rows), "harness_errors": errors,
            "elapsed_wall_s": time.perf_counter() - started})

    def drain(stream):
        while True:
            try:
                row = queue.get_nowait()
            except Empty:
                break
            key = (row["seed"], row["method"])
            if key in seen:
                raise RuntimeError(f"Duplicate method record: {key}")
            seen.add(key)
            rows.append(row)
            stream.write(canonical(row) + "\n")
            stream.flush()
            print(f"[{len(rows)}/{expected_runs}] seed={row['seed']} method={row['method']} "
                  f"success={row['success']} clear={row['cleared_total']}/{row['source_total']} "
                  f"virtual={row['virtual_time_s']:.3f}s wall={row['wall_s']:.3f}s", flush=True)

    write_progress()
    try:
        with (args.output / "runs.jsonl").open("w", encoding="utf-8") as stream:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                     initializer=initialize_worker, initargs=(settings, queue)) as executor:
                pending = {executor.submit(run_pair, seed): seed
                           for seed in range(args.seed, args.seed + args.count)}
                last_progress = time.monotonic()
                while pending:
                    finished, _ = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
                    drain(stream)
                    for future in finished:
                        seed = pending.pop(future)
                        try:
                            completed.append(future.result())
                        except Exception as error:
                            errors.append({"seed": seed, "error": f"{type(error).__name__}: {error}",
                                           "traceback": "".join(traceback.format_exception(error))})
                    if finished or time.monotonic() - last_progress >= 5:
                        write_progress()
                        last_progress = time.monotonic()
            # Worker shutdown joins the queue feeder, so all emitted rows are now visible.
            drain(stream)
    finally:
        queue.close()
        queue.join_thread()
    changed = source_changes(sources)
    if changed:
        errors.append({"error": "Source changed during evaluation", "paths": changed})
    by_method = {label: sorted((r for r in rows if r["method"] == label), key=lambda r: r["seed"])
                 for label in specs}
    complete = len(rows) == expected_runs and len(completed) == args.count and not errors
    summary = {"complete": complete, "expected_cases": args.count, "verified_completed_cases": len(completed),
               "expected_runs": expected_runs, "recorded_runs": len(rows),
               "all_runs_successful": complete and all(r["success"] for r in rows),
               "source_unchanged": not changed, "source_changes": changed, "harness_errors": errors,
               "elapsed_wall_s": time.perf_counter() - started,
               "methods": {label: method_summary(subset) for label, subset in by_method.items()},
               "comparisons": {label: paired_summary(by_method["baseline"], by_method[label])
                               for label in modes},
               "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_SAMPLES,
                             "interval": "percentile 95%; resample paired case savings"},
               "raw_time_caveat": "Early failed runs may have low raw times. Use penalized paired comparisons; efficiency claims require all expected cases to succeed.",
               "wall_time_caveat": manifest["wall_time_caveat"],
               "generalization_caveat": "Synthetic local cases only. Multiple-candidate development CIs are exploratory; stress CIs are descriptive, not distribution-wide probability claims."}
    dump_json(args.output / "source_verification.json", {
        "verified_utc": datetime.now(timezone.utc).isoformat(), "unchanged": not changed,
        "expected_source_sha256": sources, "actual_source_sha256": source_hashes(),
        "changed_paths": changed, "completed_case_checks": sorted(completed, key=lambda c: c["seed"])})
    dump_json(args.output / "summary.json", summary)
    write_progress()
    print(json.dumps({"complete": complete, "all_runs_successful": summary["all_runs_successful"],
                      "comparisons": summary["comparisons"]}, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["all_runs_successful"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage", choices=tuple(SEED_INTERVALS), default="development")
    parser.add_argument("--candidate-mode", nargs="+", choices=("nearest", "history", "both", "disabled"),
                        default=["nearest"], help="One or more candidates in execution order; holdout accepts one")
    parser.add_argument("--freeze-manifest", type=Path,
                        help="Optionally require source hashes and selected exact specs to match a previous manifest")
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
