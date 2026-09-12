"""Reproducible source-count Q3 batch using the repository's unchanged evaluator."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.research_v1_eval import run_case
from simulation import random_scenario

SPEC = "experiments/state_search_candidate_derived_silence_combined_v1.json"
PROTOCOL = {"limits": {"max_actions": 10000, "real_seconds_per_case": 1200,
                       "virtual_seconds_per_case": 360000}}
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def identity():
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")]
    names += [SPEC, "experiments/research_v1_eval.py", "experiments/run_q3_comparison.py",
              "experiments/run_q3_random_source_batch.py"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}
    snapshot = ROOT / "BATCH_SOURCE_SNAPSHOT.json"
    if snapshot.exists():
        source = read_json(snapshot)
        if hashes != source["source_sha256"]:
            raise ValueError("Frozen source snapshot changed")
        commit = source["commit"]
    else:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"commit": commit, "source_sha256": hashes, "aggregate_sha256": digest(hashes)}


def execute(seed, output, spec):
    case = random_scenario(3, seed)
    record = run_case(case, spec, PROTOCOL)
    payload = canonical(record).encode("utf-8")
    path = Path(output) / "cases" / f"case-{seed}.json.gz"
    temp = path.with_suffix(".tmp")
    with temp.open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=1, mtime=0) as zipped:
            zipped.write(payload)
    temp.replace(path)
    return record["row"]


def percentile(values, fraction):
    values = sorted(values)
    t = (len(values) - 1) * fraction
    i = int(t)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (t - i)


def summary(rows, manifest, bootstrap=False):
    if not rows:
        return {"completed_cases": 0, "complete": False}
    total = sum(r["virtual_time_s"] for r in rows)
    sources = sum(r["source_total"] for r in rows)
    result = {
        "complete": len(rows) == len(manifest["cases"]),
        "expected_cases": len(manifest["cases"]), "completed_cases": len(rows),
        "total_sources": sources, "cleared_sources": sum(r["cleared_total"] for r in rows),
        "successful_cases": sum(r["successful"] for r in rows),
        "all_clear_cases": sum(r["all_cleared"] for r in rows),
        "failed_clear_attempts": sum(r["failed_clear_count"] for r in rows),
        "failed_seeds": [r["seed"] for r in rows if not r["successful"]],
        "total_virtual_time_s": total,
        "pooled_time_per_source_s": total / sources,
        "episode_equal_weight_time_per_source_s": statistics.mean(r["time_per_source_s"] for r in rows),
        "mean_episode_virtual_time_s": total / len(rows),
        "p95_episode_virtual_time_s": percentile([r["virtual_time_s"] for r in rows], .95),
        "mean_case_program_runtime_s": statistics.mean(r["program_runtime_s"] for r in rows),
        "total_measurements": sum(r["measurement_count"] for r in rows),
        "time_components_total_s": {k: sum(r[k] for r in rows) for k in COMPONENTS},
        "source_count_strata": {},
        "scope": "Local synthetic research distribution; not official simulator results. Failures retained. Raw time does not qualify a failed batch.",
        "metric": "sum(full task billed virtual time)/sum(source count), matching the historical 226.55 metric",
    }
    for n in range(10, 17):
        group = [r for r in rows if r["source_total"] == n]
        if group:
            result["source_count_strata"][str(n)] = {
                "cases": len(group), "successful": sum(r["successful"] for r in group),
                "mean_episode_time_s": statistics.mean(r["virtual_time_s"] for r in group),
                "pooled_time_per_source_s": sum(r["virtual_time_s"] for r in group) / (n * len(group)),
            }
    if bootstrap:
        rng = random.Random(20260913)
        pairs = [(r["virtual_time_s"], r["source_total"]) for r in rows]
        estimates = []
        for _ in range(4000):
            sample = rng.choices(pairs, k=len(pairs))
            estimates.append(sum(p[0] for p in sample) / sum(p[1] for p in sample))
        result["pooled_bootstrap_ci95_s"] = [percentile(estimates, .025), percentile(estimates, .975)]
        result["bootstrap"] = {"unit": "whole scene", "resamples": 4000, "seed": 20260913}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-sources", type=int, default=10000)
    parser.add_argument("--seed-start", type=int, default=2026091300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Run only this many initial scenes; preserve them on resume")
    args = parser.parse_args()
    if args.workers < 1 or args.target_sources < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Counts must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(exist_ok=True)
    spec = read_json(ROOT / SPEC)
    source = identity()
    cases, count, seed = [], 0, args.seed_start
    while count < args.target_sources:
        case = random_scenario(3, seed)
        cases.append({"seed": seed, "source_total": len(case.sources),
                      "case_sha256": digest(case.evaluation_config())})
        count += len(case.sources)
        seed += 1
    manifest = {"source": source, "spec": spec, "protocol": PROTOCOL,
                "seed_start": args.seed_start, "target_sources": args.target_sources,
                "actual_sources": count, "cases": cases,
                "sampling": "Consecutive new seeds of unchanged random_scenario(3, seed); stop after first complete scene reaching source target, before observing outcomes."}
    mp = output / "manifest.json"
    if mp.exists() and read_json(mp) != manifest:
        raise ValueError("Resume manifest mismatch")
    if not mp.exists():
        write_json(mp, manifest)
    rows, pending = [], []
    for item in cases[:args.limit]:
        path = output / "cases" / f"case-{item['seed']}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                row = json.load(stream)["row"]
            if row["case_sha256"] != item["case_sha256"] or row["source_total"] != item["source_total"]:
                raise ValueError("Saved case identity mismatch")
            rows.append(row)
        else:
            pending.append(item["seed"])
    started = time.perf_counter()
    initial = len(rows)
    print(canonical({"phase": "start", "expected_cases": len(cases), "sources": count,
                     "pending_cases": len(pending), "resumed_cases": initial, "workers": args.workers}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, s, str(output), spec): s for s in pending}
        for future in as_completed(futures):
            rows.append(future.result())
            if (len(rows) - initial) % 24 == 0 or len(rows) == initial + len(pending):
                update = summary(rows, manifest)
                update["current_invocation_wall_s"] = time.perf_counter() - started
                write_json(output / "progress.json", update)
                print(canonical({k: update[k] for k in ("completed_cases", "total_sources", "successful_cases", "pooled_time_per_source_s", "current_invocation_wall_s")}), flush=True)
    elapsed = time.perf_counter() - started
    rows.sort(key=lambda r: r["seed"])
    write_json(output / "rows.json", rows)
    with (output / "rows.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = summary(rows, manifest, bootstrap=len(rows) == len(cases))
    result["execution"] = {"workers": args.workers, "new_cases": len(pending), "resumed_cases": initial,
                           "invocation_wall_s": elapsed, "platform": platform.platform(),
                           "hostname": platform.node(), "python": platform.python_version(),
                           "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "summary.json", result)
    print(canonical({"phase": "finish", **result}), flush=True)


if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    main()
