"""Isolated, frozen-source, paired Q3 local evaluation. No HTTP or database.

Every case/policy runs in a fresh process, preventing cross-worktree imports.
Validation DB use is a separate, strictly post-freeze operation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import importlib
import json
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
PROTOCOL = ROOT / "research/round2/protocol.json"
CRITICAL = ("simulation/cases.py", "simulation/engine.py", "simulator_client/rules.py",
            "simulator_client/client.py", "simulator_client/state.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def hashes(repo):
    return {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((repo / "src").rglob("*.py"))}


def prepare(args):
    protocol = read(PROTOCOL)
    trial = protocol["created_experiments"][args.trial]
    if args.stage == "confirmation":
        pilot = read(ROOT / "results/round2" / args.trial / "pilot/summary.json")
        if not pilot["comparisons"][args.trial]["pilot_expansion_gate"]:
            raise ValueError("Pilot gate did not pass; confirmation is not authorized by the protocol")
    if args.stage in ("final", "stress"):
        confirmed = read(ROOT / "results/round2" / args.trial / "confirmation/summary.json")
        if not confirmed["comparisons"][args.trial]["pareto_confirmation_gate"]:
            raise ValueError("Independent confirmation did not pass; do not open final cases")
        start, stop = protocol["reserved_final_seeds" if args.stage == "final" else "reserved_stress_seeds"]
    else:
        start, stop = trial[args.stage+"_seeds"]
    recipes = {"baseline": {"repository": str(ROOT.parent / "q3-state-search"),
                             "spec": protocol["baseline_spec"]},
               args.trial: {"repository": str(ROOT.parent / trial["directory"]), "spec": args.spec}}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    policies, critical = {}, None
    for label, recipe in recipes.items():
        repo = Path(recipe["repository"]).resolve()
        spec_path = repo / recipe["spec"]
        if subprocess.check_output(["git", "status", "--porcelain", "--", "src", recipe["spec"]], cwd=repo, text=True).strip():
            raise ValueError(f"Commit policy source/config before freeze: {label}")
        spec = read(spec_path)
        source_hashes = hashes(repo)
        common = {p: source_hashes["src/"+p] for p in CRITICAL}
        if critical is None:
            critical = common
        elif critical != common:
            raise ValueError("Physical engine/rules differ across policies")
        policies[label] = {"source_root": str(repo), "spec": spec, "spec_sha256": digest(spec),
                           "source_sha256": source_hashes,
                           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
        evaluation_helper = "experiments/training_stress_reliability.py"
        policies[label]["evaluation_helper_sha256"] = {evaluation_helper: hashlib.sha256((repo/evaluation_helper).read_bytes()).hexdigest()}
        with zipfile.ZipFile(output / f"{label}-source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in source_hashes:
                archive.write(repo / path, path)
            archive.write(spec_path, recipe["spec"])
            archive.write(repo/evaluation_helper, evaluation_helper)
        policies[label]["source_archive_sha256"] = hashlib.sha256((output / f"{label}-source.zip").read_bytes()).hexdigest()
    previous_stage = "pilot" if args.stage == "confirmation" else "confirmation" if args.stage in ("final", "stress") else None
    if previous_stage:
        previous = read(ROOT / "results/round2" / args.trial / previous_stage / "manifest.json")
        for label, policy in policies.items():
            earlier = previous["policies"][label]
            if (policy["source_sha256"] != earlier["source_sha256"]
                    or policy["spec_sha256"] != earlier["spec_sha256"]
                    or policy["evaluation_helper_sha256"] != earlier.get("evaluation_helper_sha256")):
                raise ValueError("Candidate changed after its previous evaluation; register a new experiment")
    manifest = {"trial": args.trial, "stage": args.stage, "seeds": list(range(start, stop+1)),
                "policies": policies, "protocol_sha256": digest(protocol),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "limits": {"real_s": 1200, "virtual_s": 360000, "max_actions": 10000},
                "python": sys.version, "platform": platform.platform(),
                "source_scope": "Local Q3 simulations only; actual world inaccessible to policies",
                "runtime_scope": "Concurrent independent processes; diagnostic elapsed wall time"}
    write(output / "manifest.json", manifest)
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"frozen": str(output), "cases_per_policy": len(manifest["seeds"]),
                      "policies": {k: p["commit"] for k,p in policies.items()}}))


def make_case(seed, stage):
    from simulation import Scenario, Source, random_scenario
    if stage != "stress":
        return random_scenario(3, seed)
    # Public-law stress families; a new seed block, independent of old hard
    # cases and any practice data. This function is frozen with the runner.
    offset = seed-211001
    if not 0 <= offset < 28:
        raise ValueError("Unexpected reserved stress identity")
    families = ("minimum_radius", "boundary", "cluster", "positive_error",
                "negative_error", "alternating_error", "narrow_strip")
    family, replicate = families[offset//4], offset % 4
    count = (10, 12, 14, 16)[replicate]
    import math
    rng = random.Random(seed)
    phase = rng.uniform(0, 2*math.pi)
    sources = []
    for index, channel in enumerate(rng.sample(range(1, 21), count)):
        theta, radial = rng.uniform(0, 2*math.pi), 1800*math.sqrt(rng.random())
        x, y = radial*math.cos(theta), radial*math.sin(theta)
        if family == "boundary":
            angle = phase+2*math.pi*index/count
            x, y = 1800*math.cos(angle), 1800*math.sin(angle)
        elif family == "cluster":
            radius = (750., 1100., 1450., 1650.)[replicate]
            x, y = radius*math.cos(phase)+.25*math.cos(theta), radius*math.sin(phase)+.25*math.sin(theta)
        elif family == "narrow_strip":
            along, across = 1000+24*index, (-1)**index*.2
            x, y = along*math.cos(phase)-across*math.sin(phase), along*math.sin(phase)+across*math.cos(phase)
        sources.append(Source(channel, x, y, 1000.))
    mode = {"positive_error": "positive_extreme", "negative_error": "negative_extreme",
            "alternating_error": "alternating_extreme", "cluster": "alternating_extreme",
            "narrow_strip": "alternating_extreme"}.get(family, "uniform")
    return Scenario(f"q3-round2-stress-{family}-{seed}", 3, seed, tuple(sources), mode,
                    "Frozen independent final stress geometry; not an official distribution")


def worker(args):
    manifest = read(args.output / "manifest.json")
    policy = manifest["policies"][args.policy]
    repo = Path(policy["source_root"])
    if args.seed not in manifest["seeds"] or hashes(repo) != policy["source_sha256"]:
        raise ValueError("Frozen worker identity mismatch")
    for path, value in policy.get("evaluation_helper_sha256", {}).items():
        if hashlib.sha256((repo/path).read_bytes()).hexdigest() != value:
            raise ValueError("Evaluation interface changed after freeze")
    sys.path[:0] = [str(repo / "src"), str(repo)]
    from simulation import LocalResearchSimulator, random_scenario
    import simulation.engine
    from experiments.training_stress_reliability import ObservationOnlyClient, audit_actions
    if Path(simulation.engine.__file__).resolve() != (repo / "src/simulation/engine.py").resolve():
        raise ValueError("Cross-worktree module import")
    limits = manifest["limits"]
    simulator = LocalResearchSimulator(make_case(args.seed, manifest["stage"]),
        max_real_duration_s=limits["real_s"], max_virtual_duration_s=limits["virtual_s"])
    client = ObservationOnlyClient(simulator.client())
    report, errors, diagnostic, started = None, [], None, time.perf_counter()
    cpu_started = time.process_time()
    try:
        module, function = policy["spec"]["entrypoint"].split(":", 1)
        callback = getattr(importlib.import_module(module), function)
        report = callback(client, problem=3, max_actions=limits["max_actions"], **policy["spec"].get("kwargs", {}))
        errors.extend(str(e) for e in (report.error, report.exit_error) if e)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        diagnostic = traceback.format_exc()
    finally:
        if client.state.session == "active" and client.pending_request is None:
            try:
                client.exit()
            except Exception as exc:
                errors.append(f"CleanupExit: {exc}")
        simulator.finish_for_evaluation()
    elapsed = time.perf_counter()-started
    cpu_elapsed = time.process_time()-cpu_started
    evaluation = simulator.evaluation()  # First access to truth is post-termination.
    history = simulator.observation_history()
    audit = audit_actions(history, evaluation)
    errors.extend(audit["errors"])
    certified = bool(report and report.completion_certified_under_model)
    exited = client.state.session == "exited" and client.pending_request is None
    if report and (report.cleared_count != evaluation["cleared_total"] or
                   abs(report.virtual_time_s-evaluation["virtual_time_s"]) > 1e-6):
        errors.append("Report/evaluation mismatch")
    if certified and not evaluation["all_cleared"]:
        errors.append("False completeness certificate")
    success = bool(evaluation["all_cleared"] and certified and exited and not errors and evaluation["failed_clear_count"] == 0)
    truth = evaluation["ground_truth"]
    row = {"case_id": truth["case_id"], "seed": args.seed, "case_sha256": digest(truth),
           "strategy": args.policy, "successful": success, "all_cleared": evaluation["all_cleared"],
           "completion_certified": certified, "accepted_exit": exited,
           "source_total": evaluation["source_total"], "cleared_total": evaluation["cleared_total"],
           "cleared_fraction": evaluation["cleared_fraction"], "virtual_time_s": evaluation["virtual_time_s"],
           "penalized_time_s": evaluation["virtual_time_s"] if success else limits["virtual_s"],
           "time_per_source_s": evaluation["virtual_time_s"]/evaluation["source_total"],
           "program_runtime_s": elapsed, "program_cpu_s": cpu_elapsed,
           "failed_clear_count": evaluation["failed_clear_count"],
           "measurement_count": evaluation["measurement_count"], "action_count": evaluation["action_count"],
           **evaluation["time_breakdown_s"], "errors": errors}
    record = {"row": row, "summary": report.as_dict() if report else None, "history": history,
              "evaluation": evaluation, "evaluation_phase": "after_policy_termination",
              "spec": policy["spec"], "audit": audit, "exception_traceback": diagnostic,
              "frozen_manifest_sha256": digest(manifest)}
    path = args.output / "records" / f"{args.policy}-{args.seed}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(record, stream, allow_nan=False)
    print(json.dumps({k: row[k] for k in ("strategy", "seed", "successful", "virtual_time_s", "program_runtime_s", "errors")}))


def percentile(values, p):
    values = sorted(values)
    x = (len(values)-1)*p
    low = int(x)
    return values[low]+(x-low)*(values[min(low+1, len(values)-1)]-values[low])


def summarize(output):
    manifest = read(output / "manifest.json")
    rows, file_hashes = [], {}
    for policy in manifest["policies"]:
        for seed in manifest["seeds"]:
            path = output / "records" / f"{policy}-{seed}.json.gz"
            record = json.load(gzip.open(path, "rt", encoding="utf-8"))
            if (record.get("frozen_manifest_sha256") != digest(manifest)
                    or record["spec"] != manifest["policies"][policy]["spec"]
                    or record["row"]["seed"] != seed or record["row"]["strategy"] != policy):
                raise ValueError("Record provenance mismatch")
            rows.append(record["row"])
            file_hashes[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    by_policy = {label: [r for r in rows if r["strategy"] == label] for label in manifest["policies"]}
    averages = {}
    for label, group in by_policy.items():
        values = [r["penalized_time_s"] for r in group]
        averages[label] = {"runs": len(group), "successful": sum(r["successful"] for r in group),
            "failed_clears": sum(r["failed_clear_count"] for r in group),
            "mean_s": statistics.mean(values), "p95_s": percentile(values, .95), "max_s": max(values),
            "mean_runtime_s": statistics.mean(r["program_runtime_s"] for r in group),
            "p95_runtime_s": percentile([r["program_runtime_s"] for r in group], .95),
            "max_runtime_s": max(r["program_runtime_s"] for r in group),
            "mean_cpu_s": statistics.mean(r["program_cpu_s"] for r in group) if all("program_cpu_s" in r for r in group) else None,
            "mean_measurements": statistics.mean(r["measurement_count"] for r in group),
            "mean_movement_s": statistics.mean(r["movement_s"] for r in group)}
    comparisons = {}
    base = {r["seed"]: r for r in by_policy["baseline"]}
    for label, group in by_policy.items():
        if label == "baseline":
            continue
        if any(base[r["seed"]]["case_sha256"] != r["case_sha256"] for r in group):
            raise ValueError("Pair case hash mismatch")
        saves = [base[r["seed"]]["penalized_time_s"]-r["penalized_time_s"] for r in group]
        rng = random.Random(52173)
        boots = [statistics.mean(rng.choices(saves, k=len(saves))) for _ in range(10000)]
        ci = [percentile(boots, .025), percentile(boots, .975)]
        safe = all(r["successful"] and r["failed_clear_count"] == 0 for r in by_policy["baseline"]+group)
        comparisons[label] = {"pairs": len(group), "safe": safe, "mean_saved_s": statistics.mean(saves),
            "ci95_saved_s": ci, "mean_reduction_fraction": statistics.mean(saves)/averages["baseline"]["mean_s"],
            "wins": sum(s>1e-6 for s in saves), "losses": sum(s < -1e-6 for s in saves),
            "ties": sum(abs(s)<=1e-6 for s in saves), "worst_paired_regression_s": max(-s for s in saves),
            "p90_paired_regression_s": percentile([-s for s in saves], .9),
            "pilot_expansion_gate": safe and statistics.mean(saves)>0 and (ci[0]>0 or
                (statistics.mean(saves)>=15 and percentile([-s for s in saves], .9)<=120)),
            "pareto_confirmation_gate": safe and ci[0]>0 and averages[label]["p95_s"] <= averages["baseline"]["p95_s"]
                 and averages[label]["max_s"] <= averages["baseline"]["max_s"],
            "gate_scope": "Only supplied cases; no promotion until final independent and certificate audits"}
    result = {"stage": manifest["stage"], "averages": averages, "comparisons": comparisons,
              "records_sha256": file_hashes, "rows": rows}
    write(output / "summary.json", result)
    print(json.dumps({"averages": averages, "comparisons": comparisons}), flush=True)


def run(args):
    output = args.output.resolve()
    manifest = read(output / "manifest.json")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest["runner_sha256"]:
        raise ValueError("Runner changed after freeze")
    for label, policy in manifest["policies"].items():
        if hashes(Path(policy["source_root"])) != policy["source_sha256"]:
            raise ValueError("Policy changed after freeze")
        if hashlib.sha256((output / f"{label}-source.zip").read_bytes()).hexdigest() != policy["source_archive_sha256"]:
            raise ValueError("Frozen source archive integrity mismatch")
    def launch(policy, seed):
        path = output / "records" / f"{policy}-{seed}.json.gz"
        if path.exists():
            record = json.load(gzip.open(path, "rt", encoding="utf-8"))
            if record.get("frozen_manifest_sha256") != digest(manifest):
                raise ValueError("Resume provenance mismatch")
            return f"resumed {policy} {seed}"
        cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
               "--policy", policy, "--seed", str(seed)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=manifest["limits"]["real_s"]+60)
        if result.returncode:
            write(output / f"worker-error-{policy}-{seed}.json", {"returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr})
            raise RuntimeError(f"Worker failed {policy}/{seed}: {result.stderr[-1000:]}")
        return result.stdout.strip()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(launch, policy, seed) for seed in manifest["seeds"] for policy in manifest["policies"]]
        for future in as_completed(pending):
            print(future.result(), flush=True)
    summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--trial", required=True)
    prepare_parser.add_argument("--stage", choices=("pilot", "coverage_dev", "confirmation", "final", "stress"), default="pilot")
    prepare_parser.add_argument("--spec", required=True)
    for command in (prepare_parser, sub.add_parser("run"), sub.add_parser("worker"), sub.add_parser("summarize")):
        command.add_argument("--output", type=Path, required=True)
    run_parser = sub.choices["run"]
    run_parser.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    worker_parser = sub.choices["worker"]
    worker_parser.add_argument("--policy", required=True)
    worker_parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "worker": worker,
     "summarize": lambda a: summarize(a.output.resolve())}[args.command](args)


if __name__ == "__main__":
    main()
