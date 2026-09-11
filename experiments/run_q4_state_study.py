"""Local paired Q4 evaluation; frozen identities, full records, no HTTP."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import gzip
import hashlib
import importlib
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
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_study import bootstrap_mean_interval
from simulation import LocalResearchSimulator, random_scenario

PROTOCOL = {
    "problem": 4, "scope": "Local assumed distribution; not official scores",
    "pilot": [294101, 294116], "confirmation": [294117, 294180], "stress": [294201, 294228],
    "limits": {"virtual_seconds": 360000, "real_seconds": 1200, "actions": 20000},
    "pilot_gate": "All complete and audited; mean saving positive and paired 95% CI lower >0",
    "confirmation_gate": "Frozen selected candidate; all complete and audited; mean saving >=5%, paired 95% CI lower >0, p95 ratio <=1.05",
    "failure_policy": "Keep all records; incomplete/error/invalid certificate receives 360000s. Legal optical search failures count their full physical cost, not automatic run failure.",
    "bootstrap": {"seed": 42941, "samples": 10000},
    "stress_families": ["minimum_radius", "boundary_outward", "cluster", "positive_error", "negative_error", "alternating_error", "narrow_strip"],
}
SPECS = {
    "triangular": {"entrypoint": "strategies:run_search", "kwargs": {"variant": "triangular"}},
    "state": {"entrypoint": "strategies.q4_state_search:run_q4_state_search", "kwargs": {"mode": "state", "max_expansions": 200}},
    "state_hull": {"entrypoint": "strategies.q4_state_search:run_q4_state_search", "kwargs": {"mode": "state_hull", "max_expansions": 200}},
    "state_pair": {"entrypoint": "strategies.q4_state_search:run_q4_state_search", "kwargs": {"mode": "state_pair", "max_expansions": 200}},
    "state_rescue": {"entrypoint": "strategies.q4_state_search:run_q4_state_search", "kwargs": {"mode": "state_rescue", "max_expansions": 200}},
    "state_pruned": {"entrypoint": "strategies.q4_state_search:run_q4_state_search", "kwargs": {"mode": "state_pruned", "max_expansions": 200}},
    "state_pruned_rescue": {"entrypoint": "strategies.q4_state_search:run_q4_state_search", "kwargs": {"mode": "state_pruned_rescue", "max_expansions": 200}},
}


class ObservationOnlyClient:
    __slots__ = ("_client",)

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        if name not in {"state", "remaining_real_time_s", "pending_request", "enter", "measure", "clear", "exit"}:
            raise AssertionError(f"Policy requested non-observation attribute {name}")
        return getattr(self._client, name)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [Path(__file__), ROOT / "experiments/run_study.py"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def make_case(seed, stage):
    case = random_scenario(4, seed)
    if stage != "stress":
        return case
    family = PROTOCOL["stress_families"][(seed-PROTOCOL["stress"][0]) % 7]
    rng = random.Random(seed+839)
    phase = rng.uniform(0, 2*math.pi)
    sources = []
    for i, source in enumerate(case.sources):
        theta = phase+2*math.pi*i/len(case.sources)
        changes = {"reception_radius_m": 1000.}
        if family == "boundary_outward":
            changes.update(x=1799.9*math.cos(theta), y=1799.9*math.sin(theta),
                           orientation_deg=math.degrees(theta) % 360 if source.orientation_deg is not None else None)
        elif family == "cluster":
            changes.update(x=1100*math.cos(phase)+rng.uniform(-.25, .25),
                           y=1100*math.sin(phase)+rng.uniform(-.25, .25))
        elif family == "narrow_strip":
            along, across = 900+30*i, (-1)**i*.2
            changes.update(x=along*math.cos(phase)-across*math.sin(phase),
                           y=along*math.sin(phase)+across*math.cos(phase))
        sources.append(replace(source, **changes))
    error = {"positive_error": "positive_extreme", "negative_error": "negative_extreme",
             "alternating_error": "alternating_extreme", "cluster": "alternating_extreme",
             "narrow_strip": "alternating_extreme"}.get(family, "uniform")
    return replace(case, case_id=f"q4-state-stress-{family}-{seed}", sources=tuple(sources),
                   error_mode=error, description="Predeclared Q4 state-search stress")


def run_one(seed, stage, label, spec, expected_hashes):
    if source_hashes() != expected_hashes:
        raise ValueError("Source changed after experiment freeze")
    case = make_case(seed, stage)
    sim = LocalResearchSimulator(case)
    client = ObservationOnlyClient(sim.client())
    result, errors, diagnostic = None, [], None
    began = time.perf_counter()
    try:
        module, name = spec["entrypoint"].split(":", 1)
        result = getattr(importlib.import_module(module), name)(
            client, problem=4, max_actions=20000, max_active_probes=6, **spec["kwargs"])
        errors.extend(str(e) for e in (result.error, result.exit_error) if e)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        diagnostic = traceback.format_exc()
    finally:
        if client.state.session == "active" and client.pending_request is None:
            try:
                client.exit()
            except Exception as exc:
                errors.append(f"CleanupExit: {exc}")
        sim.finish_for_evaluation()
    elapsed = time.perf_counter()-began
    evaluation, history = sim.evaluation(), sim.observation_history()  # Truth after exit only.
    certified = bool(result and result.completion_certified_under_model)
    exited = client.state.session == "exited" and client.pending_request is None
    if result and (result.cleared_count != evaluation["cleared_total"] or
                   abs(result.virtual_time_s-evaluation["virtual_time_s"]) > 1e-6):
        errors.append("Strategy/evaluation mismatch")
    if abs(sum(evaluation["time_breakdown_s"].values())-evaluation["virtual_time_s"]) > 1e-6:
        errors.append("Physical cost sum mismatch")
    if certified and not evaluation["all_cleared"]:
        errors.append("False complete certificate")
    success = bool(certified and exited and evaluation["all_cleared"] and not errors)
    row = {"case_id": case.case_id, "seed": seed, "stage": stage, "problem": 4, "strategy": label,
           "case_sha256": digest(evaluation["ground_truth"]), "successful": success,
           "all_cleared": evaluation["all_cleared"], "completion_certified": certified, "accepted_exit": exited,
           "source_total": evaluation["source_total"], "cleared_total": evaluation["cleared_total"],
           "virtual_time_s": evaluation["virtual_time_s"], "penalized_time_s": evaluation["virtual_time_s"] if success else 360000,
           "program_runtime_s": elapsed, "measurement_count": evaluation["measurement_count"],
           "failed_clear_count": evaluation["failed_clear_count"], "action_count": evaluation["action_count"],
           **evaluation["time_breakdown_s"], "errors": errors}
    return {"row": row, "summary": result.as_dict() if result else None,
            "evaluation": evaluation, "history": history, "spec": spec,
            "evaluation_phase": "after_policy_termination", "exception_traceback": diagnostic}


def percentile(values, q):
    values = sorted(values)
    at = q*(len(values)-1)
    low = math.floor(at)
    return values[low]*(1-(at-low))+values[math.ceil(at)]*(at-low)


def summarize(rows):
    grouped = {label: [r for r in rows if r["strategy"] == label] for label in sorted({r["strategy"] for r in rows})}
    summaries = {label: {"runs": len(group), "successful": sum(r["successful"] for r in group),
        "mean_time_s": statistics.mean(r["penalized_time_s"] for r in group),
        "p95_time_s": percentile([r["penalized_time_s"] for r in group], .95),
        "mean_runtime_s": statistics.mean(r["program_runtime_s"] for r in group),
        "mean_measurements": statistics.mean(r["measurement_count"] for r in group),
        "mean_failed_optical_attempts": statistics.mean(r["failed_clear_count"] for r in group),
        "mean_components_s": {k: statistics.mean(r[k] for r in group) for k in
                              ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")}}
        for label, group in grouped.items()}
    base = {r["case_id"]: r for r in grouped["triangular"]}
    comparisons = {}
    for label, group in grouped.items():
        if label == "triangular":
            continue
        if len(group) != len(base) or {r["case_id"] for r in group} != base.keys():
            raise ValueError("Incomplete or duplicate pairing")
        if any(r["case_sha256"] != base[r["case_id"]]["case_sha256"] for r in group):
            raise ValueError("Unequal case truth")
        savings = [base[r["case_id"]]["penalized_time_s"]-r["penalized_time_s"] for r in group]
        ci = bootstrap_mean_interval(savings, **PROTOCOL["bootstrap"])
        reduction = 1-summaries[label]["mean_time_s"]/summaries["triangular"]["mean_time_s"]
        p95ratio = summaries[label]["p95_time_s"]/summaries["triangular"]["p95_time_s"]
        complete = all(r["successful"] for r in group+grouped["triangular"])
        comparisons[label] = {"pairs": len(group), "all_complete": complete,
            "mean_saved_s": statistics.mean(savings), "saving_ci95_s": ci, "mean_reduction_fraction": reduction,
            "p95_ratio": p95ratio, "worst_regression_s": max(0., -min(savings)),
            "wins": sum(s>1e-6 for s in savings), "losses": sum(s < -1e-6 for s in savings),
            "ties": sum(abs(s)<=1e-6 for s in savings), "pilot_gate_met_before_audit": complete and ci[0]>0,
            "confirmation_gate_met_before_audit": complete and ci[0]>0 and reduction>=.05 and p95ratio<=1.05}
    return {"summaries": summaries, "paired_vs_triangular": comparisons, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "confirmation", "stress"), default="pilot")
    parser.add_argument("--variants", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers not in range(1,5) or "triangular" not in args.variants or len(args.variants)!=len(set(args.variants)):
        raise ValueError("Need triangular, unique variants, 1..4 workers")
    hashes = source_hashes()
    if args.stage != "pilot":
        if args.selection is None:
            raise ValueError("Freeze a single selected candidate before confirmation/stress")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if selection["source_sha256"] != hashes or args.variants != ["triangular", selection["selected"]]:
            raise ValueError("Frozen selection/source mismatch")
    start, stop = PROTOCOL[args.stage]
    seeds = list(range(start, stop+1))
    specs = {v: SPECS[v] for v in args.variants}
    manifest = {"protocol": PROTOCOL, "stage": args.stage, "seeds": seeds, "specs": specs,
                "source_sha256": hashes, "workers": args.workers, "python": sys.version,
                "platform": platform.platform(), "runtime_caveat": "Concurrent local workers; timing diagnostic"}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", manifest)
    write_json(output / "freeze.json", {"git_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                                        "manifest_sha256": digest(manifest)})
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in hashes:
            archive.write(ROOT/path, path)
    (output / "records").mkdir()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(run_one, seed, args.stage, label, spec, hashes)
                for seed in seeds for label, spec in specs.items()]
        for future in as_completed(jobs):
            record = future.result()
            row = record["row"]
            with gzip.open(output / "records" / f"{row['strategy']}-{row['seed']}.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(record, stream, allow_nan=False)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("seed","strategy","successful","virtual_time_s","failed_clear_count","program_runtime_s","errors")}),flush=True)
    if source_hashes()!=hashes:
        raise ValueError("Source changed during run; no qualification")
    report = summarize(sorted(rows,key=lambda r:(r["seed"],r["strategy"])))
    write_json(output / "summary.json", report)
    print(json.dumps(report["paired_vs_triangular"]),flush=True)


if __name__ == "__main__":
    main()
