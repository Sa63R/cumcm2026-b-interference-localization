"""Fixed same-world Q3 comparison; two unchanged policies, one physical engine."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q3_random_source_batch import execute, read_json, write_json, digest, summary, PROTOCOL
from simulation import random_scenario

STATE_SPEC = read_json(ROOT / "experiments/state_search_candidate_derived_silence_combined_v1.json")
V3_SPEC = {"name": "v3_origin20", "entrypoint": "experiments.q3_v3_local_adapter:run_v3_origin20", "kwargs": {}}
SPECS = {"state_search": STATE_SPEC, "v3_origin20": V3_SPEC}


def run_pair(seed, directory):
    rows = {}
    order = list(SPECS) if seed % 2 == 0 else list(reversed(SPECS))
    for method in order:
        folder = Path(directory) / method
        path = folder / "cases" / f"case-{seed}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                row = json.load(stream)["row"]
        else:
            row = execute(seed, str(folder), SPECS[method])
        rows[method] = row
    if rows["state_search"]["case_sha256"] != rows["v3_origin20"]["case_sha256"]:
        raise ValueError("Unpaired physical worlds")
    return rows


def compare(pairs, expected, bootstrap=False):
    import numpy as np
    base = [p["state_search"] for p in pairs]
    other = [p["v3_origin20"] for p in pairs]
    n = np.array([r["source_total"] for r in base], dtype=float)
    a = np.array([r["virtual_time_s"] for r in base])
    b = np.array([r["virtual_time_s"] for r in other])
    d = b - a
    result = {"complete": len(pairs) == expected, "paired_cases": len(pairs),
              "sources_per_method": int(n.sum()),
              "all_runs_successful": all(r["successful"] for r in base + other),
              "state_pooled_seconds_per_source": float(a.sum() / n.sum()),
              "v3_pooled_seconds_per_source": float(b.sum() / n.sum()),
              "v3_minus_state_pooled_seconds_per_source": float(d.sum() / n.sum()),
              "state_reduction_relative_to_v3": float(1 - a.sum() / b.sum()),
              "state_faster_cases": int((d > 1e-6).sum()), "v3_faster_cases": int((d < -1e-6).sum()),
              "tied_cases": int((np.abs(d) <= 1e-6).sum()),
              "mean_paired_v3_minus_state_seconds_per_case": float(d.mean()),
              "paired_delta_quantiles_seconds_per_case": dict(zip(["p05", "p50", "p95"], map(float, np.quantile(d, [.05, .5, .95])))),
              "state_worst_regression_case": {"seed": base[int(np.argmin(d))]["seed"], "extra_seconds": float(-d.min())},
              "state_largest_gain_case": {"seed": base[int(np.argmax(d))]["seed"], "saved_seconds": float(d.max())},
              "comparison_scope": "All fixed random cases retained; raw time is qualified only if all runs successful. Synthetic local distribution, not official simulator."}
    if bootstrap:
        rng = np.random.default_rng(20260913)
        savings, reductions, am, bm = [], [], [], []
        for _ in range(40):
            idx = rng.integers(0, len(n), size=(100, len(n)))
            aa, bb, nn = a[idx].sum(axis=1), b[idx].sum(axis=1), n[idx].sum(axis=1)
            savings.extend((bb - aa) / nn)
            reductions.extend(1 - aa / bb)
            am.extend(aa / nn); bm.extend(bb / nn)
        for name, values in (("paired_pooled_saving_ci95_s", savings), ("relative_reduction_ci95", reductions),
                             ("state_pooled_ci95_s", am), ("v3_pooled_ci95_s", bm)):
            result[name] = list(map(float, np.quantile(values, [.025, .975])))
        result["bootstrap"] = {"resamples": 4000, "unit": "paired whole scenario", "seed": 20260913}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=5000)
    parser.add_argument("--seed-start", type=int, default=2026092300)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if min(args.cases, args.workers) < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Counts must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = read_json(ROOT / "PAIRED_SOURCE_SNAPSHOT.json")
    for name, sha in frozen["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Frozen source changed: " + name)
    cases = []
    for seed in range(args.seed_start, args.seed_start + args.cases):
        case = random_scenario(3, seed)
        cases.append({"seed": seed, "source_total": len(case.sources), "case_sha256": digest(case.evaluation_config())})
    manifest = {"source": frozen, "cases": cases, "specs": SPECS, "protocol": PROTOCOL,
                "sampling": "Fixed 5000 new consecutive seeds; common coordinate-keyed error field and unchanged full-action billing. No result-based stopping.",
                "metric": "sum(full task virtual time) / sum(source count)",
                "order": "Within each pair, method order alternates by seed parity"}
    mp = output / "manifest.json"
    if mp.exists() and read_json(mp) != manifest:
        raise ValueError("Resume identity mismatch")
    write_json(mp, manifest)
    for method, spec in SPECS.items():
        folder = output / method
        (folder / "cases").mkdir(parents=True, exist_ok=True)
        write_json(folder / "manifest.json", {**manifest, "spec": spec})
    requested = cases[:args.limit]
    pairs, pending = [], []
    for item in requested:
        if all((output / method / "cases" / f"case-{item['seed']}.json.gz").exists() for method in SPECS):
            pair = run_pair(item["seed"], str(output))
            if any(r["case_sha256"] != item["case_sha256"] or r["strategy"] != SPECS[k]["name"] for k, r in pair.items()):
                raise ValueError("Stored record identity mismatch")
            pairs.append(pair)
        else:
            pending.append(item["seed"])
    print(json.dumps({"phase": "warmup", "pending_pairs": len(pending), "workers": args.workers,
                      "total_sources_per_method": sum(c["source_total"] for c in cases)}), flush=True)
    from experiments.q3_v3_local_adapter import warmup
    warmup()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_pair, seed, str(output)): seed for seed in pending}
        for future in as_completed(futures):
            pairs.append(future.result())
            if len(pairs) % 100 == 0 or len(pairs) == len(requested):
                progress = compare(pairs, args.cases)
                progress["current_invocation_wall_s"] = time.perf_counter() - started
                write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    elapsed = time.perf_counter() - started
    pairs.sort(key=lambda p: p["state_search"]["seed"])
    for method in SPECS:
        rows = [p[method] for p in pairs]
        folder = output / method
        write_json(folder / "rows.json", rows)
        with (folder / "rows.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        write_json(folder / "summary.json", summary(rows, manifest, bootstrap=False))
    result = compare(pairs, args.cases, bootstrap=len(pairs) == args.cases)
    result["execution"] = {"wall_seconds_excluding_warmup": elapsed, "new_pairs": len(pending),
                           "workers": args.workers, "hostname": platform.node(),
                           "python": platform.python_version(), "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "comparison.json", result)
    with (output / "paired.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "source_count", "state_seconds", "v3_seconds", "v3_minus_state_seconds", "state_successful", "v3_successful"])
        for p in pairs:
            a, b = p["state_search"], p["v3_origin20"]
            writer.writerow([a["seed"], a["source_total"], a["virtual_time_s"], b["virtual_time_s"], b["virtual_time_s"] - a["virtual_time_s"], a["successful"], b["successful"]])
    print(json.dumps({"phase": "complete", **result}), flush=True)


if __name__ == "__main__":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    main()
