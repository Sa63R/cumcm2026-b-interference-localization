"""Paired local Q3 development of source service modes. No HTTP or formal tests."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.research_v1_eval import digest, write_json, summarize, paired_comparison
from experiments.training_stress_reliability import ObservationOnlyClient, audit_actions
from simulation import LocalResearchSimulator, random_scenario

PROTOCOL = {
    "scope": "Fresh state-method development; not globally unseen across historical RL training",
    "pilot_seeds": [190101, 190116], "confirmation_seeds": [190117, 190180],
    "limits": {"virtual_seconds_per_case": 360000, "real_seconds_per_case": 1200, "max_actions": 10000},
    "acceptance": {"bootstrap_seed": 39251, "bootstrap_resamples": 10000},
    "pilot_gate": "Candidate: all 16 successful, zero failed clears, mean savings positive, paired 95% CI lower > 0; select highest mean savings among qualifiers",
    "confirmation_gate": "Freeze selected variant before 64 new cases; all successful/zero failed clears, paired 95% CI lower > 0, >=1% mean savings, p95 ratio <=1.05; then stress audit",
    "failure_policy": "Retain failures and raw records; unsuccessful/failed-clear/audit-error run costs 360000 s",
    "reference": "v1 relocating cover state policy remains unchanged; no old final or future validation seeds opened",
}


def source_hashes():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [Path(__file__), ROOT / "experiments/research_v1_eval.py",
              ROOT / "experiments/run_q3_comparison.py", ROOT / "experiments/training_stress_reliability.py"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_one(seed, label, spec, expected_hashes):
    if source_hashes() != expected_hashes:
        raise ValueError("Source changed after experiment freeze")
    case = random_scenario(3, seed)
    sim = LocalResearchSimulator(case, max_real_duration_s=1200, max_virtual_duration_s=360000)
    client = ObservationOnlyClient(sim.client())
    started, result, errors, diagnostic = time.perf_counter(), None, [], None
    try:
        module, name = spec["entrypoint"].split(":", 1)
        callback = getattr(importlib.import_module(module), name)
        result = callback(client, problem=3, max_actions=10000, **spec["kwargs"])
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
    elapsed = time.perf_counter()-started
    evaluation = sim.evaluation()  # Hidden world accessed only after termination.
    history = sim.observation_history()
    audit = audit_actions(history, evaluation)
    errors.extend(audit["errors"])
    certified = bool(result and result.completion_certified_under_model)
    exited = client.state.session == "exited" and client.pending_request is None
    if result and (result.cleared_count != evaluation["cleared_total"] or
                   abs(result.virtual_time_s-evaluation["virtual_time_s"]) > 1e-6):
        errors.append("Strategy/evaluation mismatch")
    if certified and not evaluation["all_cleared"]:
        errors.append("False completeness certificate")
    success = bool(evaluation["all_cleared"] and certified and exited and not errors
                   and evaluation["failed_clear_count"] == 0)
    row = {"case_id": case.case_id, "seed": seed, "strategy": label,
           "case_sha256": digest(evaluation["ground_truth"]), "successful": success,
           "all_cleared": evaluation["all_cleared"], "completion_certified": certified,
           "accepted_exit": exited, "source_total": evaluation["source_total"],
           "cleared_total": evaluation["cleared_total"], "cleared_fraction": evaluation["cleared_fraction"],
           "virtual_time_s": evaluation["virtual_time_s"],
           "penalized_time_s": evaluation["virtual_time_s"] if success else 360000,
           "time_per_source_s": evaluation["virtual_time_s"]/evaluation["source_total"],
           "program_runtime_s": elapsed, "measurement_count": evaluation["measurement_count"],
           "failed_clear_count": evaluation["failed_clear_count"], "action_count": evaluation["action_count"],
           **evaluation["time_breakdown_s"], "errors": errors}
    return {"row": row, "summary": result.as_dict() if result else None,
            "evaluation": evaluation, "history": history, "spec": spec,
            "exception_traceback": diagnostic, "audit": audit,
            "evaluation_phase": "after_policy_termination"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--variants", nargs="+", choices=("v1", "local", "joint"), default=["v1", "local", "joint"])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers not in range(1, 5) or "v1" not in args.variants or len(set(args.variants)) != len(args.variants):
        raise ValueError("Need v1, unique variants, and 1..4 workers")
    start, stop = PROTOCOL[args.stage+"_seeds"]
    seeds = list(range(start, stop+1))
    definitions = json.loads((ROOT / "experiments/state_modes_v2_specs.json").read_text())
    specs = {v: definitions[v] for v in args.variants}
    hashes = source_hashes()
    manifest = {"protocol": PROTOCOL, "stage": args.stage, "seeds": seeds, "specs": specs,
                "source_sha256": hashes, "workers": args.workers,
                "python": sys.version, "platform": platform.platform(),
                "runtime_caveat": "Concurrent local workers; elapsed CPU figures diagnostic, not controlled hardware comparison"}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mp = output / "manifest.json"
    if mp.exists():
        if json.loads(mp.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume manifest mismatch; keep old evidence and use a new output directory")
    else:
        write_json(mp, manifest)
        write_json(output / "freeze.json", {"git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "manifest_sha256": digest(manifest)})
        with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in hashes:
                archive.write(ROOT / path, path)
    (output / "records").mkdir(exist_ok=True)
    jobs, rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for seed in seeds:
            for label, spec in specs.items():
                path = output / "records" / f"{label}-{seed}.json.gz"
                if path.exists():
                    with gzip.open(path, "rt", encoding="utf-8") as stream:
                        record = json.load(stream)
                    if record["row"]["seed"] != seed or record["spec"] != spec:
                        raise ValueError("Resume record identity mismatch")
                    rows.append(record["row"])
                else:
                    jobs.append(executor.submit(run_one, seed, label, spec, hashes))
        for future in as_completed(jobs):
            record = future.result()
            row = record["row"]
            path = output / "records" / f"{row['strategy']}-{row['seed']}.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(record, stream, allow_nan=False)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("seed", "strategy", "successful", "virtual_time_s", "program_runtime_s", "errors")}), flush=True)
    if source_hashes() != hashes:
        raise ValueError("Source mutated during experiment; no qualification")
    rows.sort(key=lambda r: (r["seed"], r["strategy"]))
    groups = {label: [r for r in rows if r["strategy"] == label] for label in specs}
    comparisons = {label: paired_comparison(groups["v1"], values, PROTOCOL)
                   for label, values in groups.items() if label != "v1"}
    for label, comparison in comparisons.items():
        comparison["all_pairs_zero_failed_clear"] = all(
            r["successful"] and r["failed_clear_count"] == 0 for r in groups["v1"]+groups[label])
        comparison["pilot_gate_met"] = comparison["all_pairs_zero_failed_clear"] and comparison["saving_ci95_s"][0] > 0
        comparison["confirmation_gate_met"] = (comparison["pilot_gate_met"]
            and comparison["mean_reduction_fraction"] >= .01 and comparison["p95_time_ratio"] <= 1.05)
    report = {"stage": args.stage, "summaries": {k: summarize(v, len(seeds)) for k, v in groups.items()},
              "paired_vs_v1": comparisons, "rows": rows}
    write_json(output / "summary.json", report)
    print(json.dumps({"paired_vs_v1": comparisons}), flush=True)


if __name__ == "__main__":
    main()
