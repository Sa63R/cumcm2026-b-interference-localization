"""Predeclared Q3 evaluation; observation-only policies and held-out test gate."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.run_q3_comparison import bootstrap_mean_interval, percentile
from simulation import LocalResearchSimulator, random_scenario

PROTOCOL = ROOT / "research/v1_protocol.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                         + "\n", encoding="utf-8")
    temporary.replace(path)


def code_hashes():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [Path(__file__), ROOT / "experiments/run_q3_comparison.py"]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def identity(spec, protocol):
    checkpoints = {}
    for key, value in spec.get("kwargs", {}).items():
        if key in {"checkpoint", "weights"} and value:
            p = Path(value)
            checkpoints[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"source_sha256": code_hashes(), "spec_sha256": digest(spec),
            "checkpoint_sha256": checkpoints, "protocol_sha256": digest(protocol)}


def make_case(split, seed, protocol):
    case = random_scenario(3, seed)
    if split != "final_stress":
        return case
    start = protocol["partitions"][split]["seed_start"]
    family = protocol["final_stress_families"][(seed - start) % 7]
    rng = random.Random(seed + 991731)
    phase = rng.uniform(0, 2 * math.pi)
    n = len(case.sources)
    sources = case.sources
    mode = "uniform"
    if family == "minimum_radius":
        sources = tuple(replace(s, reception_radius_m=1000.0) for s in sources)
    elif family == "boundary":
        sources = tuple(replace(s, x=1799.9 * math.cos(phase + 2 * math.pi * i / n),
                                y=1799.9 * math.sin(phase + 2 * math.pi * i / n),
                                reception_radius_m=1000.0) for i, s in enumerate(sources))
    elif family == "cluster":
        cx, cy = 1000 * math.cos(phase), 1000 * math.sin(phase)
        sources = tuple(replace(s, x=cx + rng.uniform(-10, 10),
                                y=cy + rng.uniform(-10, 10), reception_radius_m=1000.0)
                        for s in sources)
    elif family == "narrow_strip":
        sources = tuple(replace(s, x=(850 + 45 * i) * math.cos(phase) - (-1)**i * .1 * math.sin(phase),
                                y=(850 + 45 * i) * math.sin(phase) + (-1)**i * .1 * math.cos(phase),
                                reception_radius_m=1000.0) for i, s in enumerate(sources))
    else:
        mode = {"positive_error": "positive_extreme", "negative_error": "negative_extreme",
                "alternating_error": "alternating_extreme"}[family]
        sources = tuple(replace(s, reception_radius_m=1000.0) for s in sources)
    return replace(case, case_id=f"q3-v1-stress-{family}-{seed}", sources=sources,
                   error_mode=mode, description=f"Predeclared unseen stress family: {family}")


def run_case(case, spec, protocol, real_budget=None):
    limits = protocol["limits"]
    sim = LocalResearchSimulator(case, max_real_duration_s=real_budget or limits["real_seconds_per_case"],
                                 max_virtual_duration_s=limits["virtual_seconds_per_case"])
    client = sim.client()
    result, errors, diagnostic = None, [], None
    started = time.perf_counter()
    try:
        module, function = spec["entrypoint"].split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        kwargs = {"problem": 3, "max_actions": limits["max_actions"], **spec.get("kwargs", {})}
        result = callback(client, **kwargs)
        errors.extend(str(v) for v in (result.error, result.exit_error) if v)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        diagnostic = traceback.format_exc()
    finally:
        if client.state.session == "active" and client.pending_request is None:
            try:
                client.exit()
            except Exception as exc:
                errors.append(f"CleanupExit: {type(exc).__name__}: {exc}")
        sim.finish_for_evaluation()
    elapsed = time.perf_counter() - started
    evaluation = sim.evaluation()  # Truth is first read after policy termination.
    summary = result.as_dict() if result is not None else None
    certified = bool(result and result.completion_certified_under_model)
    exited = client.state.session == "exited" and client.pending_request is None
    if client.state.cleared_count != evaluation["cleared_total"]:
        errors.append("Client/evaluator cleared-count mismatch")
    if not math.isclose(sum(evaluation["time_breakdown_s"].values()), evaluation["virtual_time_s"], abs_tol=1e-6):
        errors.append("Time ledger mismatch")
    if result and (result.cleared_count != evaluation["cleared_total"] or
                   not math.isclose(result.virtual_time_s, evaluation["virtual_time_s"], abs_tol=1e-6)):
        errors.append("Strategy/evaluator count or time mismatch")
    if certified and not evaluation["all_cleared"]:
        errors.append("False completeness certificate")
    success = bool(evaluation["all_cleared"] and certified and exited and not errors)
    row = {"case_id": case.case_id, "seed": case.seed, "strategy": spec["name"],
           "case_sha256": digest(case.evaluation_config()), "successful": success,
           "all_cleared": evaluation["all_cleared"], "completion_certified": certified,
           "accepted_exit": exited, "source_total": evaluation["source_total"],
           "cleared_total": evaluation["cleared_total"], "cleared_fraction": evaluation["cleared_fraction"],
           "virtual_time_s": evaluation["virtual_time_s"],
           "penalized_time_s": evaluation["virtual_time_s"] if success else limits["virtual_seconds_per_case"],
           "time_per_source_s": evaluation["virtual_time_s"] / evaluation["source_total"],
           "program_runtime_s": elapsed, "measurement_count": evaluation["measurement_count"],
           "failed_clear_count": evaluation["failed_clear_count"], "action_count": evaluation["action_count"],
           **evaluation["time_breakdown_s"], "errors": errors}
    return {"row": row, "summary": summary, "evaluation": evaluation,
            "history": sim.observation_history(), "exception_traceback": diagnostic,
            "evaluation_phase": "after_policy_termination", "spec": spec}


def summarize(rows, expected):
    values = [r["virtual_time_s"] for r in rows]
    penalized = [r["penalized_time_s"] for r in rows]
    return {"runs": len(rows), "expected_runs": expected, "complete": len(rows) == expected,
            "successful_runs": sum(r["successful"] for r in rows),
            "failed_clear_count": sum(r["failed_clear_count"] for r in rows),
            "raw_mean_total_time_s": statistics.mean(values) if rows else None,
            "penalized_mean_total_time_s": statistics.mean(penalized) if rows else None,
            "raw_p95_total_time_s": percentile(values, .95),
            "raw_max_total_time_s": max(values) if rows else None,
            "mean_time_per_source_s": statistics.mean(r["time_per_source_s"] for r in rows) if rows else None,
            "mean_program_runtime_s": statistics.mean(r["program_runtime_s"] for r in rows) if rows else None,
            "raw_time_caveat": "Failed early termination can reduce raw time. Qualification requires every case successful."}


def paired_comparison(base, candidate, protocol):
    if len({r["case_id"] for r in base}) != len(base) or len({r["case_id"] for r in candidate}) != len(candidate):
        raise ValueError("Duplicate case ids")
    b = {r["case_id"]: r for r in base}
    c = {r["case_id"]: r for r in candidate}
    if b.keys() != c.keys() or not b:
        raise ValueError("Paired comparison requires identical, nonempty case sets")
    pairs = [(b[k], c[k]) for k in sorted(b)]
    if any(x["case_sha256"] != y["case_sha256"] for x, y in pairs):
        raise ValueError("Same id has different scenario truth")
    savings = [x["penalized_time_s"] - y["penalized_time_s"] for x, y in pairs]
    settings = protocol["acceptance"]
    ci = bootstrap_mean_interval(savings, seed=settings["bootstrap_seed"], samples=settings["bootstrap_resamples"])
    bm = statistics.mean(x["penalized_time_s"] for x, _ in pairs)
    cm = statistics.mean(y["penalized_time_s"] for _, y in pairs)
    ratio = percentile([y["penalized_time_s"] for _, y in pairs], .95) / percentile([x["penalized_time_s"] for x, _ in pairs], .95)
    valid = all(x["successful"] and y["successful"] and y["failed_clear_count"] == 0 for x, y in pairs)
    return {"pairs": len(pairs), "all_pairs_successful_no_candidate_failed_clear": valid,
            "mean_seconds_saved": statistics.mean(savings), "saving_ci95_s": ci,
            "mean_reduction_fraction": 1-cm/bm, "p95_time_ratio": ratio,
            "worst_paired_regression_s": max(-s for s in savings),
            "wins": sum(s > 1e-6 for s in savings), "losses": sum(s < -1e-6 for s in savings),
            "ties": sum(abs(s) <= 1e-6 for s in savings),
            "performance_target_met_on_supplied_cases": valid and 1-cm/bm >= .05 and ci[0] > 0 and ratio <= 1.05,
            "scope": "This comparison alone does not establish completeness of the full final test suite."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "validation_extended", "final_random", "final_stress"])
    parser.add_argument("--freeze-record", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    spec, protocol = read_json(args.spec), read_json(PROTOCOL)
    ident = identity(spec, protocol)
    if args.command == "freeze":
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments", "research/v1_protocol.json"], cwd=ROOT, text=True)
        if dirty.strip():
            raise ValueError("Commit code and protocol before freezing")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        write_json(args.output, {"frozen_at": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
                                 "strategy": spec["name"], "identity": ident})
        return
    if args.split.startswith("final_"):
        if args.freeze_record is None or read_json(args.freeze_record)["identity"] != ident:
            raise ValueError("Final tests require an unchanged code/config/checkpoint freeze record")
    partition = protocol["partitions"][args.split]
    seeds = list(range(partition["seed_start"], partition["seed_stop_exclusive"]))
    if args.offset < 0 or args.offset >= len(seeds) or (args.count is not None and args.count <= 0):
        raise ValueError("Invalid shard range")
    seeds = seeds[args.offset:None if args.count is None else args.offset+args.count]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"identity": ident, "strategy": spec["name"], "split": args.split, "seeds": seeds}
    mp = args.output / "manifest.json"
    if mp.exists() and read_json(mp) != manifest:
        raise ValueError("Resume identity mismatch; use a distinct output directory")
    write_json(mp, manifest)
    rows = []
    deadline = datetime.fromisoformat(protocol["hard_deadline"]).timestamp()
    for seed in seeds:
        path = args.output / f"case-{seed}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                record = json.load(stream)
        else:
            remaining = deadline-time.time()-30
            if remaining <= 2:
                print("HARD_DEADLINE_SAVE_AND_STOP", flush=True)
                break
            record = run_case(make_case(args.split, seed, protocol), spec, protocol,
                              min(protocol["limits"]["real_seconds_per_case"], remaining))
            temporary = path.with_suffix(".tmp")
            with gzip.open(temporary, "wt", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            temporary.replace(path)
        rows.append(record["row"])
        print(canonical({k: record["row"][k] for k in ["seed", "strategy", "successful", "virtual_time_s", "program_runtime_s"]}), flush=True)
        write_json(args.output / "rows.json", rows)
        write_json(args.output / "summary.json", summarize(rows, len(seeds)))


if __name__ == "__main__":
    main()
