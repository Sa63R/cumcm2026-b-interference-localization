"""Frozen local Q4 cover comparisons; every reported time includes T/LB."""
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
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from experiments.run_q4_state_study import (
    SPECS as OLD_SPECS, digest, percentile, run_one, source_hashes, summarize, write_json)
from experiments.run_study import bootstrap_mean_interval

PROTOCOL = {
    "problem": 4,
    "pilot": [394001, 394016], "confirmation": [394101, 394164],
    "stress": [394201, 394228],
    "baseline": "state_pruned", "scope": "Synthetic assumed scenarios; not official scores",
    "selection": "Pilot-only selection, then frozen source/spec before independent confirmation",
    "acceptance": "All cleared and independent certificate audit; >=5% mean saving against v1, paired CI lower >0; report every loss",
    "failure_penalty_s": 360000,
    "bound": "Post-termination common oracle movement/5 +5N +50(20-N) if N<16; not an attainable optimum",
}


def frozen_hashes():
    hashes = source_hashes()
    for name in ("experiments/run_q4_cover_study.py", "experiments/q4_comparison_bounds.py"):
        hashes[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return hashes


def one(seed, stage, label, spec, expected):
    if frozen_hashes() != expected:
        raise ValueError("Frozen source changed")
    record = run_one(seed, stage, label, spec, source_hashes())
    from experiments.q4_comparison_bounds import common_bound
    bound = common_bound(record["evaluation"]["ground_truth"])
    lower = bound["common_lower_bound_s"]
    record["common_lower_bound"] = bound
    record["row"].update(common_lower_bound_s=lower,
        time_over_lower_bound=record["row"]["virtual_time_s"]/lower,
        penalized_time_over_lower_bound=record["row"]["penalized_time_s"]/lower)
    return record


def report_rows(rows):
    report = summarize(rows)
    for label, group in report["summaries"].items():
        subset = [r for r in rows if r["strategy"] == label]
        group.update(mean_lower_bound_s=statistics.mean(r["common_lower_bound_s"] for r in subset),
            mean_time_over_mean_lower_bound=group["mean_time_s"]/statistics.mean(r["common_lower_bound_s"] for r in subset),
            mean_individual_time_over_lower_bound=statistics.mean(r["penalized_time_over_lower_bound"] for r in subset))
    base = {r["case_id"]: r for r in rows if r["strategy"] == "state_pruned"}
    pairs = {}
    for label in report["summaries"]:
        if label in {"triangular", "state_pruned"}:
            continue
        subset = [r for r in rows if r["strategy"] == label]
        if len(subset) != len(base):
            raise ValueError("Incomplete pairing")
        if any(r["case_sha256"]!=base[r["case_id"]]["case_sha256"] or
               r["common_lower_bound_s"]!=base[r["case_id"]]["common_lower_bound_s"] for r in subset):
            raise ValueError("Pairing identity/bound mismatch")
        saved = [base[r["case_id"]]["penalized_time_s"]-r["penalized_time_s"] for r in subset]
        pairs[label] = {"pairs":len(subset), "all_complete":all(r["successful"] and base[r["case_id"]]["successful"] for r in subset),
            "mean_saved_s":statistics.mean(saved),
            "mean_time_reduction_fraction":1-report["summaries"][label]["mean_time_s"]/report["summaries"]["state_pruned"]["mean_time_s"],
            "saving_ci95_s":bootstrap_mean_interval(saved,seed=43941,samples=10000),
            "wins":sum(x>1e-6 for x in saved), "losses":sum(x < -1e-6 for x in saved),
            "worst_regression_s":max(0.,-min(saved)),
            "paired_time_ratio_p95":percentile([r["penalized_time_s"]/base[r["case_id"]]["penalized_time_s"] for r in subset],.95)}
    report["paired_vs_state_pruned"] = pairs
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(k for k in PROTOCOL if k in {"pilot","confirmation","stress"}), default="pilot")
    parser.add_argument("--profile", default="compact")
    parser.add_argument("--schedules", nargs="+", choices=("joint","deferred"), default=["joint","deferred"])
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or len(set(args.schedules))!=len(args.schedules):
        raise ValueError("Invalid worker/schedule list")
    hashes = frozen_hashes()
    specs = {k: OLD_SPECS[k] for k in ("triangular","state_pruned")}
    specs.update({"compact_"+s:{"entrypoint":"strategies.q4_cover_search:run_q4_cover_search",
        "kwargs":{"profile":args.profile,"schedule":s,"max_expansions":200}} for s in args.schedules})
    if args.stage != "pilot":
        if args.selection is None:
            raise ValueError("A frozen selection is required")
        selected = json.loads(args.selection.read_text(encoding="utf-8"))
        if selected["source_sha256"]!=hashes or selected["specs"]!=specs:
            raise ValueError("Frozen selection mismatch")
    start, stop = PROTOCOL[args.stage]
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest={"protocol":PROTOCOL,"stage":args.stage,"seeds":list(range(start,stop+1)),"specs":specs,
              "source_sha256":hashes,"workers":args.workers,"python":sys.version}
    write_json(output/"manifest.json",manifest)
    write_json(output/"freeze.json",{"manifest_sha256":digest(manifest),
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()})
    with zipfile.ZipFile(output/"source.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in hashes: archive.write(ROOT/path,path)
    (output/"records").mkdir()
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,seed,args.stage,label,spec,hashes)
                 for seed in range(start,stop+1) for label,spec in specs.items()]
        for future in as_completed(futures):
            record=future.result(); row=record["row"]
            with gzip.open(output/"records"/f"{row['strategy']}-{row['seed']}.json.gz","wt",encoding="utf-8") as stream:
                json.dump(record,stream,allow_nan=False)
            rows.append(row)
            print(json.dumps({k:row[k] for k in ("seed","strategy","successful","virtual_time_s","common_lower_bound_s","time_over_lower_bound","errors")}),flush=True)
    if frozen_hashes()!=hashes: raise ValueError("Source changed during experiment")
    report=report_rows(sorted(rows,key=lambda r:(r["seed"],r["strategy"])))
    write_json(output/"summary.json",report)
    print(json.dumps({"groups":report["summaries"],"vs_v1":report["paired_vs_state_pruned"]}),flush=True)


if __name__=="__main__": main()
