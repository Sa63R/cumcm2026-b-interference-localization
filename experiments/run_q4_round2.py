"""Fresh paired Q4 trials. Policies see observations; bounds use truth after exit."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q4_state_study import run_one, source_hashes, digest, write_json, percentile
from experiments.run_study import bootstrap_mean_interval
from experiments.q4_comparison_bounds import common_bound

BASE = "compact_baseline"
BASE_SPEC = {"entrypoint": "strategies.q4_cover_search:run_q4_cover_search",
             "kwargs": {"profile": "compact_22", "schedule": "joint", "max_expansions": 200}}


def hashes():
    result = source_hashes()
    for name in ("run_q4_round2.py", "q4_comparison_bounds.py", "audit_q4_cover.py", "audit_q4_state.py"):
        p = ROOT / "experiments" / name
        result[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def one(seed, stage, label, spec, expected):
    if hashes() != expected:
        raise ValueError("Source changed after freeze")
    record = run_one(seed, stage, label, spec, source_hashes())
    bound = common_bound(record["evaluation"]["ground_truth"])
    lower = bound["common_lower_bound_s"]
    record["common_lower_bound"] = bound
    record["row"].update(common_lower_bound_s=lower,
        time_over_lower_bound=record["row"]["virtual_time_s"] / lower,
        penalized_time_over_lower_bound=record["row"]["penalized_time_s"] / lower)
    return record


def report_rows(rows):
    groups = {label: [r for r in rows if r["strategy"] == label] for label in sorted({r["strategy"] for r in rows})}
    base = {r["seed"]: r for r in groups[BASE]}
    if len(base) != len(groups[BASE]):
        raise ValueError("Duplicate baseline seed")
    summaries, pairs = {}, {}
    for label, group in groups.items():
        if len(group) != len(base) or {r["seed"] for r in group} != base.keys():
            raise ValueError("Incomplete or duplicated pairing")
        if any(r["case_sha256"] != base[r["seed"]]["case_sha256"] or
               r["common_lower_bound_s"] != base[r["seed"]]["common_lower_bound_s"] for r in group):
            raise ValueError("Unequal paired scenario/bound")
        times = [r["penalized_time_s"] for r in group]
        mean, lower = statistics.mean(times), statistics.mean(r["common_lower_bound_s"] for r in group)
        summaries[label] = {"runs": len(group), "successful": sum(r["successful"] for r in group),
            "mean_time_s": mean, "mean_lower_bound_s": lower, "mean_time_over_mean_lower_bound": mean/lower,
            "mean_individual_ratio": statistics.mean(r["penalized_time_over_lower_bound"] for r in group),
            "p95_time_s": percentile(times, .95), "max_time_s": max(times),
            "mean_runtime_s": statistics.mean(r["program_runtime_s"] for r in group),
            "mean_components_s": {k: statistics.mean(r[k] for r in group) for k in
                ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")}}
        if label == BASE:
            continue
        savings = [base[r["seed"]]["penalized_time_s"]-r["penalized_time_s"] for r in group]
        ci = bootstrap_mean_interval(savings, seed=610941, samples=10000)
        base_mean = statistics.mean(r["penalized_time_s"] for r in base.values())
        pairs[label] = {"pairs": len(group), "all_complete": all(r["successful"] and base[r["seed"]]["successful"] for r in group),
            "mean_saved_s": statistics.mean(savings), "saving_ci95_s": ci,
            "mean_reduction_fraction": 1-mean/base_mean,
            "wins": sum(s>1e-6 for s in savings), "losses": sum(s < -1e-6 for s in savings),
            "ties": sum(abs(s)<=1e-6 for s in savings), "worst_regression_s": max(0., -min(savings)),
            "paired_ratio_p95": percentile([r["penalized_time_s"]/base[r["seed"]]["penalized_time_s"] for r in group], .95)}
    return {"summaries": summaries, "paired_vs_compact_baseline": pairs, "rows": rows}


def audit(directory):
    from experiments.audit_q4_cover import audit_record
    manifest = json.loads((directory/"manifest.json").read_bytes())
    freeze = json.loads((directory/"freeze.json").read_bytes())
    errors, audits, rows, keys = [], [], [], set()
    if digest(manifest) != freeze["manifest_sha256"]:
        errors.append("Manifest changed")
    with zipfile.ZipFile(directory/"source.zip") as archive:
        for p, h in manifest["source_sha256"].items():
            if hashlib.sha256(archive.read(p)).hexdigest() != h or hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != h:
                errors.append("Source mismatch: " + p)
    for path in sorted((directory/"records").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        row = record["row"]
        key = (row["seed"], row["strategy"])
        if key in keys or record["spec"] != manifest["specs"][row["strategy"]]:
            errors.append("Duplicate key or spec mismatch")
        keys.add(key)
        rows.append(row)
        audits.append(audit_record(record))
    if keys != {(seed, label) for seed in manifest["seeds"] for label in manifest["specs"]}:
        errors.append("Incomplete matrix")
    try:
        if report_rows(sorted(rows, key=lambda r: (r["seed"],r["strategy"]))) != json.loads((directory/"summary.json").read_bytes()):
            errors.append("Summary differs from records")
    except ValueError as exc:
        errors.append(str(exc))
    result = {"records": len(audits), "passed_records": sum(a["passed"] for a in audits),
              "all_passed": not errors and all(a["passed"] for a in audits), "errors": errors, "audits": audits}
    destination = directory/"independent_audit.json"
    if destination.exists():
        raise ValueError("Preserve existing audit")
    write_json(destination, result)
    print(json.dumps({k: v for k,v in result.items() if k != "audits"}), flush=True)
    return int(not result["all_passed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path)
    parser.add_argument("--stage", choices=("pilot", "confirmation", "stress"), default="pilot")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        return audit(args.output.resolve())
    specs = json.loads(args.specs.read_bytes())
    if specs.get(BASE) != BASE_SPEC or any(not s.startswith("compact_") for s in specs):
        raise ValueError("Need exact compact baseline and independently certified compact labels")
    if not 1 <= args.workers <= 4 or args.count is None or not 1 <= args.count <= 128 or args.start is None:
        raise ValueError("Invalid worker/seed range")
    frozen = hashes()
    seeds = list(range(args.start, args.start+args.count))
    if args.stage != "pilot":
        selection = json.loads(args.selection.read_bytes())
        if (selection["source_sha256"] != frozen or selection["specs"] != specs
                or selection["reserved_seeds"][args.stage] != seeds):
            raise ValueError("Independent validation source/spec/reserved seed mismatch")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"stage": args.stage, "seeds": seeds, "specs": specs, "source_sha256": frozen,
        "baseline_commit": "1e58ce9fe3f72c8043d8ac35cce8a90e102d81af", "scope": "Local synthetic paired validation only",
        "failure_penalty_s": 360000, "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest() if args.selection else None}
    write_json(directory/"manifest.json", manifest)
    write_json(directory/"freeze.json", {"manifest_sha256": digest(manifest),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    with zipfile.ZipFile(directory/"source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in frozen:
            archive.write(ROOT/p, p)
    (directory/"records").mkdir()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(one, seed, args.stage, label, spec, frozen) for seed in seeds for label,spec in specs.items()]
        for future in as_completed(jobs):
            record = future.result()
            row = record["row"]
            with gzip.open(directory/"records"/f"{row['strategy']}-{row['seed']}.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(record, stream, allow_nan=False)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("seed", "strategy", "successful", "virtual_time_s", "common_lower_bound_s", "time_over_lower_bound", "errors")}), flush=True)
    if hashes() != frozen:
        raise ValueError("Source changed during experiment")
    result = report_rows(sorted(rows, key=lambda r:(r["seed"], r["strategy"])))
    write_json(directory/"summary.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
