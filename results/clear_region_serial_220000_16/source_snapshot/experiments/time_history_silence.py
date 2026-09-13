"""Serial wall-time check of frozen baseline/history on reused development cases.

This is an overhead measurement, not another strategy selection or benefit
evaluation. Exactly one method executes at a time, in this Python process.
The parent task must confirm other evaluation/audit CPU jobs have ended before
invoking this script with --cpu-idle-confirmed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import evaluate_clear_region as evaluation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEVELOPMENT = ROOT / "results/clear_region_dev_220000_32/manifest.json"
DEFAULT_OUTPUT = ROOT / "results/clear_region_serial_220000_16"
SEEDS = tuple(range(220000, 220016))
METHODS = ("baseline", "history")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def distribution(values):
    return {"count": len(values), "mean_s": statistics.mean(values),
            "median_s": statistics.median(values),
            "p95_s": evaluation.percentile(values, .95),
            "minimum_s": min(values), "maximum_s": max(values)}


def inputs(manifest_path):
    manifest = read_json(manifest_path)
    summary_path = manifest_path.parent / "summary.json"
    runs_path = manifest_path.parent / "runs.jsonl"
    summary = read_json(summary_path)
    if (manifest["stage"] != "development" or manifest["seed_start"] != 220000
            or manifest["seed_stop_exclusive"] != 220032
            or not summary["complete"] or not summary["all_runs_successful"]
            or not summary["source_unchanged"]):
        raise RuntimeError("Expected the completed, unchanged 32-case development run")
    specs = {label: deepcopy(manifest["specs"][label]) for label in METHODS}
    for label, spec in specs.items():
        if evaluation.digest(spec) != manifest["spec_sha256"][label]:
            raise RuntimeError(f"Development spec digest mismatch: {label}")
    if (specs["history"]["kwargs"]["clear_mode"] != "disabled"
            or specs["history"]["kwargs"]["history_silence"] is not True):
        raise RuntimeError("Only frozen history silence with clear_mode disabled is allowed")
    if manifest["limits"] != evaluation.LIMITS:
        raise RuntimeError("Runtime/evaluation limits differ from frozen development limits")
    evaluation.assert_sources(manifest["source_sha256"])
    expected = {}
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["seed"], row["method"])
        if row["seed"] not in SEEDS or row["method"] not in METHODS:
            continue
        if key in expected or not row["success"]:
            raise RuntimeError(f"Duplicate or unsuccessful development reference: {key}")
        if row["spec_sha256"] != manifest["spec_sha256"][row["method"]]:
            raise RuntimeError(f"Reference row uses a different spec: {key}")
        expected[key] = row
    if set(expected) != {(seed, label) for seed in SEEDS for label in METHODS}:
        raise RuntimeError("Missing frozen development reference rows")
    provenance = {"development_manifest": str(manifest_path),
                  "development_manifest_sha256": evaluation.sha256(manifest_path.read_bytes()),
                  "development_summary_sha256": evaluation.sha256(summary_path.read_bytes()),
                  "development_runs_sha256": evaluation.sha256(runs_path.read_bytes())}
    return manifest, specs, expected, provenance


def execute(args):
    if not args.cpu_idle_confirmed:
        raise RuntimeError("Parent must first confirm other evaluation/audit CPU jobs ended; "
                           "then pass --cpu-idle-confirmed")
    sys.path.insert(0, str(ROOT / "src"))
    manifest_path = args.development_manifest.resolve()
    frozen, specs, expected, provenance = inputs(manifest_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    (output / "warmup/traces").mkdir(parents=True)
    sources = evaluation.snapshot_sources(output)
    if sources != frozen["source_sha256"]:
        raise RuntimeError("Snapshot differs from the frozen development source")
    script_path = Path(__file__).resolve()
    script_contents = script_path.read_bytes()
    script_hash = evaluation.sha256(script_contents)
    snapshot_script = output / "source_snapshot/experiments/time_history_silence.py"
    snapshot_script.write_bytes(script_contents)
    schedule = [{"seed": seed, "method_order": list(METHODS if seed % 2 == 0 else reversed(METHODS))}
                for seed in SEEDS]
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "serial_computer_wall_overhead_only_not_candidate_selection",
        "data_origin": "synthetic_research_reused_development_cases",
        "official_practice": False, "official_formal": False,
        "cpu_idle_confirmed_by_parent": True, "concurrent_methods": 1,
        "execution": "one process, serial calls, no executor or child workers",
        "warmup": {"seed": 220000, "method_order": list(METHODS),
                   "repetitions_per_method": 1, "included_in_statistics": False},
        "seeds": list(SEEDS), "schedule": schedule,
        "specs": specs, "spec_sha256": {label: evaluation.digest(spec) for label, spec in specs.items()},
        "source_sha256": sources, "timing_script_sha256": script_hash,
        "limits": evaluation.LIMITS, "python": sys.version,
        "python_executable": sys.executable, "platform": platform.platform(),
        "machine": platform.machine(), **provenance,
        "wall_s_definition": "evaluate_clear_region.run_method perf_counter around policy execution and cleanup; excludes scenario construction, trace serialization and source hashing",
        "timed_call_wall_s_definition": "outer perf_counter around run_method, including trace serialization; excludes scenario construction and source hashing",
        "virtual_reproducibility_rule": "Every measured and warmup run must exactly equal its existing development seed/method virtual_time_s, case hash and spec hash; no float tolerance",
        "summary_percentile": "p95 via linear interpolation at (n-1)*0.95",
        "caveat": "Small reused development subset; wall results describe this machine/run after warmup, not official service latency or a new virtual-time benefit test",
    }
    evaluation.dump_json(output / "manifest.json", run_manifest)
    rows, warmup_rows, errors = [], [], []
    started = time.perf_counter()

    def verify_sources():
        evaluation.assert_sources(sources)
        if evaluation.sha256(script_path.read_bytes()) != script_hash:
            raise RuntimeError("Timing script changed during serial measurement")
        if evaluation.sha256(manifest_path.read_bytes()) != provenance["development_manifest_sha256"]:
            raise RuntimeError("Development manifest changed during serial measurement")

    def measure(seed, label, warmup, order_index, stream):
        verify_sources()
        case = evaluation.make_case("development", seed)
        settings = {"stage": "development", "output": str(output / "warmup" if warmup else output)}
        began = time.perf_counter()
        row = evaluation.run_method(case, label, specs[label], settings)
        row["timed_call_wall_s"] = time.perf_counter()-began
        row.update(warmup=warmup, included_in_statistics=not warmup,
                   within_case_order=order_index, serial_sequence=len(rows)+len(warmup_rows),
                   timing_source_sha256=script_hash)
        reference = expected[(seed, label)]
        row["development_virtual_time_s"] = reference["virtual_time_s"]
        row["exact_virtual_match"] = row["virtual_time_s"] == reference["virtual_time_s"]
        row["reference_case_hash_match"] = row["case_sha256"] == reference["case_sha256"]
        row["reference_spec_hash_match"] = row["spec_sha256"] == reference["spec_sha256"]
        row["development_trace"] = reference["trace"]
        (warmup_rows if warmup else rows).append(row)
        stream.write(evaluation.canonical(row)+"\n")
        stream.flush()
        print(f"{'warmup' if warmup else 'timed'} seed={seed} method={label} "
              f"wall={row['wall_s']:.6f}s virtual={row['virtual_time_s']:.6f}s "
              f"exact={row['exact_virtual_match']}", flush=True)
        if (not row["success"] or not row["exact_virtual_match"]
                or not row["reference_case_hash_match"] or not row["reference_spec_hash_match"]):
            raise RuntimeError(f"Frozen development reproduction failed for {(seed, label)}")
        verify_sources()
        evaluation.dump_json(output / "progress.json", {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "completed_warmups": len(warmup_rows), "expected_warmups": 2,
            "completed_measured_runs": len(rows), "expected_measured_runs": 32,
            "elapsed_wall_s": time.perf_counter()-started})

    try:
        with (output / "warmups.jsonl").open("w", encoding="utf-8") as stream:
            for index, label in enumerate(METHODS):
                measure(220000, label, True, index, stream)
        with (output / "runs.jsonl").open("w", encoding="utf-8") as stream:
            for item in schedule:
                for index, label in enumerate(item["method_order"]):
                    measure(item["seed"], label, False, index, stream)
        verify_sources()
    except Exception as error:
        errors.append({"error": f"{type(error).__name__}: {error}",
                       "traceback": traceback.format_exc()})
    changed = evaluation.source_changes(sources)
    complete = len(rows) == 32 and len(warmup_rows) == 2 and not errors and not changed
    methods = {}
    for label in METHODS:
        subset = [row for row in rows if row["method"] == label]
        if subset:
            methods[label] = {"wall_s": distribution([row["wall_s"] for row in subset]),
                              "timed_call_wall_s": distribution([row["timed_call_wall_s"] for row in subset])}
    serial_summary = {
        "complete": complete, "recorded_measured_runs": len(rows), "warmup_runs": len(warmup_rows),
        "all_virtual_times_exactly_match_development": all(row["exact_virtual_match"] for row in rows+warmup_rows),
        "all_runs_successful": all(row["success"] for row in rows+warmup_rows),
        "source_unchanged": not changed, "source_changes": changed,
        "errors": errors, "methods": methods, "elapsed_wall_s": time.perf_counter()-started,
        "warmup_included_in_statistics": False, "purpose": run_manifest["purpose"],
        "timing_script_sha256": script_hash, **provenance,
    }
    if complete:
        baseline_mean = methods["baseline"]["wall_s"]["mean_s"]
        history_mean = methods["history"]["wall_s"]["mean_s"]
        paired = {label: {r["seed"]: r for r in rows if r["method"] == label} for label in METHODS}
        serial_summary["history_wall_overhead"] = {
            "difference_of_means_s": history_mean-baseline_mean,
            "ratio_of_means": history_mean/baseline_mean,
            "percent_of_baseline_mean": 100*(history_mean/baseline_mean-1),
            "paired_difference_distribution": distribution([
                paired["history"][seed]["wall_s"]-paired["baseline"][seed]["wall_s"] for seed in SEEDS])}
    evaluation.dump_json(output / "source_verification.json", {
        "expected_source_sha256": sources, "actual_source_sha256": evaluation.source_hashes(),
        "changed_paths": changed, "timing_script_sha256": script_hash,
        "actual_timing_script_sha256": evaluation.sha256(script_path.read_bytes())})
    evaluation.dump_json(output / "summary.json", serial_summary)
    print(json.dumps(serial_summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Must be a new directory; never overwrites existing evidence")
    parser.add_argument("--cpu-idle-confirmed", action="store_true",
                        help="Set only after the parent confirms other evaluation/audit CPU jobs ended")
    return execute(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
