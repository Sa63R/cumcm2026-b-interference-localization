"""Frozen named Q3 recipes, common physics, and CPU-only paired evaluation.

The unchanged round2 worker supplies physical execution and case generation.
This wrapper adds model identities, source checks after execution, registered
stages and RL comparisons. No HTTP, SQLite, training, or official interface.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/round3/protocol.json"
LEGACY_PATH = Path(__file__).with_name("round2_runner.py")
LEGACY_SHA = "4119190603ffb4a7d28d7e3bc1e286bdda88c3d6667c59f5659119152d759535"
_module_spec = importlib.util.spec_from_file_location("_round3_frozen_round2", LEGACY_PATH)
legacy = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(legacy)
read, write, digest, hashes = legacy.read, legacy.write, legacy.digest, legacy.hashes
HELPERS = ("experiments/training_stress_reliability.py", "experiments/research_v1_eval.py",
           "experiments/run_q3_comparison.py")
COMMON = tuple("src/" + p for p in legacy.CRITICAL) + (
    "src/simulation/__init__.py", "src/geometry/__init__.py", "src/localization/omni.py")
LIMITS = {"max_actions": 10000, "virtual_s": 360000, "real_s": 1200}
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
THREAD_ENV["CUDA_VISIBLE_DEVICES"] = "-1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path, base):
    path = Path(path)
    return (path if path.is_absolute() else base / path).resolve()


def runtime_identity():
    packages = {}
    for name in ("torch", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"executable": str(Path(sys.executable).resolve()), "python": sys.version,
            "packages": packages}


def inspect_recipe(label, recipe):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", label):
        raise ValueError("Unsafe recipe label")
    repo = resolve(recipe["repository"], ROOT.parent)
    spec_path = resolve(recipe["spec"], repo)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != recipe["commit"]:
        raise ValueError(f"Recipe commit mismatch: {label}")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", *HELPERS],
                               cwd=repo, text=True).strip():
        raise ValueError(f"Commit policy sources/helpers before freeze: {label}")
    spec = read(spec_path)
    if not isinstance(spec.get("kwargs", {}), dict) or ":" not in spec.get("entrypoint", ""):
        raise ValueError("Invalid callback specification")
    kwargs = spec.get("kwargs", {})
    if "problem" in kwargs or "max_actions" in kwargs:
        raise ValueError("Problem and action limits belong to the common runner")
    if kwargs.get("device", "cpu") != "cpu" or kwargs.get("num_threads", 1) != 1:
        raise ValueError("Only one-thread CPU inference is allowed")
    model_args = {k for k in ("checkpoint", "weights") if kwargs.get(k)}
    declared = recipe.get("weights", {})
    if set(declared) - model_args:
        raise ValueError("A declared weight has no callback argument")
    weights = {}
    for key in sorted(model_args):
        if key in declared:
            entry = declared[key]
            path = resolve(entry["path"], ROOT)
            expected = entry["sha256"]
        else:
            path = Path(kwargs[key])
            if not path.is_absolute():
                raise ValueError("Relative model arguments require an explicit weights path/hash")
            expected = sha(path)
        if sha(path) != expected:
            raise ValueError(f"Checkpoint hash mismatch: {label}/{key}")
        weights[key] = {"input_path": str(path), "sha256": expected, "bytes": path.stat().st_size}
    return {"source_root": str(repo), "commit": commit,
            "original_spec": spec, "original_spec_sha256": digest(spec),
            "spec_input_path": str(spec_path), "spec_file_sha256": sha(spec_path),
            "source_sha256": hashes(repo),
            "evaluation_helper_sha256": {p: sha(repo / p) for p in HELPERS}, "weights": weights}


def policy_identity(policy):
    """Stage-invariant identity: archived model absolute paths may change."""
    return {k: policy[k] for k in ("commit", "original_spec_sha256", "source_sha256",
                                  "evaluation_helper_sha256")} | {
        "weights": {k: v["sha256"] for k, v in policy["weights"].items()}}


def validate_references(policies, protocol):
    if not {"baseline", "rl"} <= set(policies):
        raise ValueError("Recipes must include the derived baseline and frozen rl reference")
    for label, key in (("baseline", "primary_comparator"), ("original", "original_reference")):
        if label not in policies:
            continue
        ref = protocol[key]
        if (policies[label]["commit"] != ref["commit"] or
                policies[label]["original_spec_sha256"] != digest(read(ROOT.parent / ref["directory"] / ref["spec"]))):
            raise ValueError(f"The {label} recipe does not match its protected reference")
    ref, rl = protocol["rl_reference_selection"], policies["rl"]
    if (rl["commit"] != ref["commit"] or
            rl["original_spec_sha256"] != digest(read(ROOT / ref["spec"])) or
            rl["weights"].get("checkpoint", {}).get("sha256") != ref["checkpoint_sha256"]):
        raise ValueError("RL reference source/spec/checkpoint differs from protocol")


def stage_definition(protocol, trial_name, stage, recipes):
    if stage == "smoke":
        return [200114], None
    trial = protocol["created_experiments"][trial_name]
    if not (trial.get("recipes_sha256") == digest(recipes) or trial.get("recipes") == recipes):
        raise ValueError("Recipes are not registered for this experiment")
    start, stop = trial[stage + "_seeds"]
    if (type(start) is not int or type(stop) is not int or start > stop or start <= 200114 <= stop):
        raise ValueError("Invalid registered new-stage seed interval")
    return list(range(start, stop + 1)), trial


def verify_prior_audit(output, manifest, summary):
    """Require an independent, complete and still-bound preceding-stage audit."""
    audit_path = output / "audit.json"
    audit = read(audit_path)
    if (audit.get("all_audits_passed") is not True or
            audit.get("all_runner_archives_available_and_verified") is not True):
        raise ValueError("Preceding-stage independent audit did not pass")
    expected = {f"records/{p}-{s}.json.gz" for p in manifest["policies"] for s in manifest["seeds"]}
    if set(summary["records_sha256"]) != expected:
        raise ValueError("Preceding-stage summary has incomplete record coverage")
    actual = {p.relative_to(output).as_posix() for p in (output / "records").glob("*.json.gz")}
    if actual != expected:
        raise ValueError("Preceding-stage record files differ from the declared set")
    evidence = [b for b in audit["batches"] if Path(b["path"]).resolve() == output.resolve()]
    if (len(evidence) != 1 or evidence[0]["manifest_sha256"] != sha(output / "manifest.json")
            or evidence[0]["summary_sha256"] != sha(output / "summary.json")):
        raise ValueError("Preceding-stage audit does not bind manifest and summary")
    audit_rows = {Path(r["input_path"]).resolve(): r for r in audit["rows"]}
    paths = {(output / p).resolve() for p in expected}
    if len(audit_rows) != len(audit["rows"]) or set(audit_rows) != paths or audit["records"] != len(paths):
        raise ValueError("Preceding-stage audit has incomplete/duplicate/extra records")
    for relative, expected_hash in summary["records_sha256"].items():
        path = output / relative
        row = audit_rows[path.resolve()]
        if (sha(path) != expected_hash or row["input_sha256"] != expected_hash
                or row["audit_passed"] is not True):
            raise ValueError("Preceding-stage audited record changed or did not pass")
    return {"path": str(audit_path.resolve()), "sha256": sha(audit_path)}


def prepare(args):
    if sha(LEGACY_PATH) != LEGACY_SHA:
        raise ValueError("The reused round2 runner has changed")
    protocol, recipes = read(PROTOCOL), read(args.recipes)
    runtime = runtime_identity()
    if runtime["packages"] != {"torch": "2.9.1+cpu", "numpy": "2.2.6"}:
        raise ValueError("Use the frozen CPU Python environment: torch2.9.1+cpu/numpy2.2.6")
    if {k: protocol["limits"][k] for k in LIMITS} != LIMITS:
        raise ValueError("Unexpected common evaluation limits")
    seeds, trial = stage_definition(protocol, args.trial, args.stage, recipes)
    policies = {label: inspect_recipe(label, recipe) for label, recipe in recipes.items()}
    validate_references(policies, protocol)
    common = None
    for policy in policies.values():
        identity = {p: policy["source_sha256"][p] for p in COMMON} | policy["evaluation_helper_sha256"]
        if common is not None and identity != common:
            raise ValueError("Common physical, geometry, scenario, or evaluation helper mismatch")
        common = identity
    previous = previous_audit = None
    if args.stage in ("confirmation", "final", "stress"):
        earlier_stage = "pilot" if args.stage == "confirmation" else "confirmation"
        previous_dir = resolve(trial.get(earlier_stage + "_output", f"results/round3/{args.trial}/{earlier_stage}"), ROOT)
        previous = read(previous_dir / "manifest.json")
        previous_result = read(previous_dir / "summary.json")
        previous_audit = verify_prior_audit(previous_dir, previous, previous_result)
        candidate = trial.get("candidate_label", args.trial)
        gate = "pilot_expansion_gate" if args.stage == "confirmation" else "pareto_confirmation_gate"
        if not previous_result["comparisons"][candidate][gate]:
            raise ValueError("The prerequisite whole-episode stage gate did not pass")
        if {k: policy_identity(v) for k, v in policies.items()} != {
                k: policy_identity(v) for k, v in previous["policies"].items()}:
            raise ValueError("Source/spec/model changed after the preceding stage")
        if previous["runner_sha256"] != sha(__file__) or previous["limits"] != LIMITS:
            raise ValueError("Runner/limits changed after preceding stage")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for label, policy in policies.items():
        spec = deepcopy(policy["original_spec"])
        for key, weight in policy["weights"].items():
            suffix = Path(weight["input_path"]).suffix or ".bin"
            relative = f"weights/{label}-{key}{suffix}"
            dest = output / relative
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(Path(weight["input_path"]).read_bytes())
            weight["archive_path"] = relative
            if sha(dest) != weight["sha256"]:
                raise ValueError("Checkpoint changed while copying")
            spec.setdefault("kwargs", {})[key] = str(dest.resolve())
        policy["spec"], policy["spec_sha256"] = spec, digest(spec)
        with zipfile.ZipFile(output / f"{label}-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            repo = Path(policy["source_root"])
            for path in policy["source_sha256"]:
                archive.write(repo / path, path)
            for path in policy["evaluation_helper_sha256"]:
                archive.write(repo / path, path)
            archive.writestr("frozen/input_spec.json", json.dumps(policy["original_spec"], ensure_ascii=False))
            archive.writestr("frozen/effective_spec.json", json.dumps(spec, ensure_ascii=False))
        policy["source_archive_sha256"] = sha(output / f"{label}-source.zip")
    (output / "round2_runner.py").write_bytes(LEGACY_PATH.read_bytes())
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    write(output / "protocol.json", protocol)
    manifest = {"trial": args.trial, "stage": args.stage, "seeds": seeds, "policies": policies,
                "protocol_sha256": digest(protocol), "protocol_file_sha256": sha(output / "protocol.json"),
                "recipes_sha256": digest(recipes), "recipes": recipes,
                "runner_sha256": sha(__file__), "runner_dependencies": {"round2_runner.py": LEGACY_SHA},
                "limits": LIMITS, "python": sys.version, "platform": platform.platform(),
                "python_executable_sha256": sha(sys.executable), "max_workers": 3,
                "runtime_identity": runtime,
                "thread_environment": THREAD_ENV, "main_comparator": "baseline", "rl_comparator": "rl",
                "source_scope": "Local Q3 only, fresh observation-only clients; truth only after policy termination",
                "case_generation": "Unmodified frozen round2 make_case on one platform; each paired world SHA must agree",
                "runtime_scope": "Separate CPU process per case and policy; startup and concurrent wall time are not virtual cost"}
    if previous:
        manifest["previous_manifest_sha256"] = digest(previous)
        manifest["previous_independent_audit"] = previous_audit
    write(output / "manifest.json", manifest)
    verify_frozen(output)
    print(json.dumps({"frozen": str(output), "stage": args.stage, "cases": len(seeds), "policies": list(policies)}))


def verify_policy(output, label, policy):
    repo = Path(policy["source_root"])
    if hashes(repo) != policy["source_sha256"]:
        raise ValueError(f"Policy source changed: {label}")
    for path, expected in policy["evaluation_helper_sha256"].items():
        if sha(repo / path) != expected:
            raise ValueError(f"Evaluation helper changed: {label}/{path}")
    if sha(policy["spec_input_path"]) != policy["spec_file_sha256"]:
        raise ValueError(f"Input spec changed: {label}")
    if digest(policy["spec"]) != policy["spec_sha256"] or digest(policy["original_spec"]) != policy["original_spec_sha256"]:
        raise ValueError("Frozen specification digest mismatch")
    if sha(output / f"{label}-source.zip") != policy["source_archive_sha256"]:
        raise ValueError(f"Source archive changed: {label}")
    for key, weight in policy["weights"].items():
        archive = (output / weight["archive_path"]).resolve()
        if (sha(weight["input_path"]) != weight["sha256"] or sha(archive) != weight["sha256"]
                or archive.stat().st_size != weight["bytes"]
                or Path(policy["spec"]["kwargs"][key]).resolve() != archive):
            raise ValueError(f"Frozen checkpoint changed: {label}/{key}")


def verify_frozen(output):
    manifest = read(output / "manifest.json")
    if sha(__file__) != manifest["runner_sha256"] or sha(output / "runner.py") != manifest["runner_sha256"]:
        raise ValueError("Runner changed after freeze")
    if sha(LEGACY_PATH) != LEGACY_SHA or sha(output / "round2_runner.py") != LEGACY_SHA:
        raise ValueError("Reused round2 dependency changed after freeze")
    if (sha(output / "protocol.json") != manifest["protocol_file_sha256"] or
            digest(read(output / "protocol.json")) != manifest["protocol_sha256"]):
        raise ValueError("Archived protocol changed")
    if (manifest["limits"] != LIMITS or sha(sys.executable) != manifest["python_executable_sha256"]
            or runtime_identity() != manifest["runtime_identity"]):
        raise ValueError("Common limits/Python executable changed")
    for label, policy in manifest["policies"].items():
        verify_policy(output, label, policy)
    return manifest


def verify_record(record, manifest, label, seed):
    if (record.get("frozen_manifest_sha256") != digest(manifest) or
            record["spec"] != manifest["policies"][label]["spec"] or
            record["row"]["strategy"] != label or record["row"]["seed"] != seed or
            record.get("round3_integrity", {}).get("before_and_after_verified") is not True):
        raise ValueError("Record provenance/integrity mismatch")
    row = record["row"]
    expected = row["virtual_time_s"] if row["successful"] else manifest["limits"]["virtual_s"]
    if row["penalized_time_s"] != expected:
        raise ValueError("Failure penalty was changed or a failed episode was discounted")


def worker(args):
    output = args.output.resolve()
    os.environ.update(THREAD_ENV)
    manifest = verify_frozen(output)
    if args.policy not in manifest["policies"] or args.seed not in manifest["seeds"]:
        raise ValueError("Worker is outside frozen recipe/seed set")
    dest = output / "records" / f"{args.policy}-{args.seed}.json.gz"
    if dest.exists():
        raise FileExistsError(dest)
    legacy.worker(args)
    verify_frozen(output)
    record = json.load(gzip.open(dest, "rt", encoding="utf-8"))
    record["round3_integrity"] = {"before_and_after_verified": True,
        "weights_sha256": {k: v["sha256"] for k, v in manifest["policies"][args.policy]["weights"].items()},
        "runner_dependencies": manifest["runner_dependencies"], "thread_environment": THREAD_ENV}
    temporary = dest.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False)
    temporary.replace(dest)
    verify_record(record, manifest, args.policy, args.seed)


def compare_rows(reference, candidate):
    left = {r["seed"]: r for r in reference}
    if not left or len(left) != len(reference) or set(left) != {r["seed"] for r in candidate} or len(candidate) != len(reference):
        raise ValueError("Paired episode identities are incomplete/duplicated")
    if any(left[r["seed"]]["case_sha256"] != r["case_sha256"] for r in candidate):
        raise ValueError("Pair case hash mismatch")
    saves = [left[r["seed"]]["penalized_time_s"] - r["penalized_time_s"] for r in candidate]
    rng = random.Random(52173)
    boots = [statistics.mean(rng.choices(saves, k=len(saves))) for _ in range(10000)]
    return {"pairs": len(saves), "safe": all(r["successful"] and r["failed_clear_count"] == 0 for r in reference + candidate),
        "mean_saved_s": statistics.mean(saves),
        "ci95_saved_s": [legacy.percentile(boots, .025), legacy.percentile(boots, .975)],
        "wins": sum(s > 1e-6 for s in saves), "losses": sum(s < -1e-6 for s in saves),
        "ties": sum(abs(s) <= 1e-6 for s in saves),
        "worst_paired_regression_s": max(-s for s in saves),
        "scope": "Reference minus candidate complete penalized episode cost; same world, no failure exclusion"}


def summarize(output):
    output = output.resolve()
    manifest = verify_frozen(output)
    expected = {f"{p}-{s}.json.gz" for p in manifest["policies"] for s in manifest["seeds"]}
    if {p.name for p in (output / "records").glob("*.json.gz")} != expected:
        raise ValueError("Cannot summarize an incomplete or extra-record batch")
    by_policy = {p: [] for p in manifest["policies"]}
    case_hashes = {}
    for label in manifest["policies"]:
        for seed in manifest["seeds"]:
            r = json.load(gzip.open(output / "records" / f"{label}-{seed}.json.gz", "rt", encoding="utf-8"))
            verify_record(r, manifest, label, seed)
            if seed in case_hashes and case_hashes[seed] != r["row"]["case_sha256"]:
                raise ValueError("Pair case hash mismatch")
            case_hashes[seed] = r["row"]["case_sha256"]
            by_policy[label].append(r["row"])
    legacy.summarize(output)
    result = read(output / "summary.json")
    result["comparisons_vs_rl"] = {p: compare_rows(by_policy["rl"], rows)
                                   for p, rows in by_policy.items() if p != "rl"}
    result["case_sha256"] = case_hashes
    result["source_and_weights_verified_after_batch"] = True
    result["old_lower_bound_status"] = "Requires independent completed-record posthoc audit; no new denominator here"
    if manifest["stage"] == "smoke":
        for comparison in result["comparisons"].values():
            comparison.update(pilot_expansion_gate=False, pareto_confirmation_gate=False,
                              gate_scope="Old 200114 implementation smoke only; never performance promotion")
    verify_frozen(output)
    write(output / "summary.json", result)
    return result


def run(args):
    output = args.output.resolve()
    manifest = verify_frozen(output)
    if not 1 <= args.workers <= manifest["max_workers"]:
        raise ValueError("At most three independent CPU workers")
    failures = []

    def launch(policy, seed):
        verify_frozen(output)
        dest = output / "records" / f"{policy}-{seed}.json.gz"
        if dest.exists():
            verify_record(json.load(gzip.open(dest, "rt", encoding="utf-8")), manifest, policy, seed)
            return f"resumed {policy} {seed}"
        cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
               "--policy", policy, "--seed", str(seed)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **THREAD_ENV},
                                    timeout=manifest["limits"]["real_s"] + 60)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Worker process timeout: {policy}/{seed}") from error
        if result.returncode:
            write(output / f"worker-error-{policy}-{seed}.json", {"returncode": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr})
            raise RuntimeError(f"Worker failed: {policy}/{seed}: {result.stderr[-1000:]}")
        return result.stdout.strip()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(launch, p, s): (p, s) for s in manifest["seeds"] for p in manifest["policies"]}
        for future in as_completed(pending):
            try:
                print(future.result(), flush=True)
            except Exception as error:
                p, s = pending[future]
                failures.append({"policy": p, "seed": s, "error": f"{type(error).__name__}: {error}"})
    if failures:
        write(output / "execution_errors.json", {"complete": False, "failures": failures,
              "scope": "Infrastructure errors retained; no completed summary or silent episode deletion"})
        raise RuntimeError("Batch has incomplete/unverified worker executions; inspect execution_errors.json")
    summarize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--trial", required=True)
    prep.add_argument("--stage", choices=("smoke", "pilot", "confirmation", "final", "stress"), required=True)
    prep.add_argument("--recipes", type=Path, required=True)
    for command in (prep, sub.add_parser("run"), sub.add_parser("worker"), sub.add_parser("summarize")):
        command.add_argument("--output", type=Path, required=True)
    sub.choices["run"].add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    sub.choices["worker"].add_argument("--policy", required=True)
    sub.choices["worker"].add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "worker": worker,
     "summarize": lambda a: summarize(a.output)}[args.command](args)


if __name__ == "__main__":
    main()
