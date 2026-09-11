"""Independent, frozen TRAINING stress cases; never reads any held-out split.

Prepare once before policies run. Each frozen policy source archive is loaded
in its own child process and working directory, preventing module-cache mixing.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gzip
import hashlib
import importlib
import io
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
FAMILIES = ("minimum_radius", "boundary", "cluster", "positive_error",
            "negative_error", "alternating_error", "narrow_strip")
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
CRITICAL = ("simulation/cases.py", "simulation/engine.py", "simulator_client/rules.py",
            "simulator_client/state.py", "simulator_client/client.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate_cases():
    # This generator deliberately does not import a common/final stress entry.
    # Counts, rotated geometries, exclusive channels and seeds differ from the
    # old seven hard cases. All cases use only public physical parameter bounds.
    result = []
    for family_index, family in enumerate(FAMILIES):
        for repetition, count in enumerate((10, 12, 14, 16)):
            seed = 113001 + 4 * family_index + repetition
            rng = random.Random(seed)
            phase = rng.uniform(0, 2 * math.pi)
            channels = rng.sample(range(1, 21), count)
            sources = []
            for index, channel in enumerate(channels):
                angle = rng.uniform(0, 2 * math.pi)
                radius = 1800 * math.sqrt(rng.random())
                x, y = radius * math.cos(angle), radius * math.sin(angle)
                if family == "boundary":
                    angle = phase + 2 * math.pi * index / count
                    x, y = 1800 * math.cos(angle), 1800 * math.sin(angle)
                elif family == "cluster":
                    radial = (750., 1100., 1450., 1650.)[repetition]
                    x = radial * math.cos(phase) + .25 * math.cos(angle)
                    y = radial * math.sin(phase) + .25 * math.sin(angle)
                elif family == "narrow_strip":
                    along, across = 1000 + 24 * index, (-1) ** index * .2
                    x = along * math.cos(phase) - across * math.sin(phase)
                    y = along * math.sin(phase) + across * math.cos(phase)
                sources.append({"channel": channel, "x": x, "y": y,
                                "reception_radius_m": 1000., "orientation_deg": None})
            mode = {"positive_error": "positive_extreme", "negative_error": "negative_extreme",
                    "alternating_error": "alternating_extreme", "cluster": "alternating_extreme",
                    "narrow_strip": "alternating_extreme"}.get(family, "uniform")
            case = {"case_id": f"q3-training-stress-{family}-{seed}", "problem": 3,
                    "seed": seed, "sources": sources, "error_mode": mode,
                    "description": f"Independent training reliability family {family}, replicate {repetition+1}; not a final or validation case."}
            result.append({"family": family, "replicate": repetition + 1,
                           "case_sha256": digest(case), "scenario": case})
    return result


def physical_signature(case):
    # Ignore names/descriptions and error-field seed when checking duplicated
    # geometry against already-exposed hard cases. Equal => investigate reuse.
    return digest({k: case[k] for k in ("problem", "sources", "error_mode")})


def freeze(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = generate_cases()
    sys.path.insert(0, str(ROOT / "src"))
    from simulation import Scenario, Source, difficult_scenarios
    old = {physical_signature(case.evaluation_config()): case.case_id for case in difficult_scenarios(3)}
    for item in cases:
        raw = item["scenario"]
        Scenario(**{**raw, "sources": tuple(Source(**s) for s in raw["sources"])})
        if physical_signature(raw) in old:
            raise ValueError("Equivalent exposed hard geometry exists: reuse rather than rerun")
    dataset = {"scope": "training_reliability_only", "seed_interval": [113001, 113028],
               "family_order": FAMILIES, "cases": cases,
               "old_hard_geometry_duplicates": [], "policy_adaptation_forbidden": True}
    (output / "cases.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    definitions = json.loads(args.policies.read_text(encoding="utf-8"))
    frozen = []
    common = None
    for definition in definitions:
        repository = Path(definition["repository"]).resolve()
        commit = subprocess.check_output(["git", "rev-parse", definition["commit"]], cwd=repository, text=True).strip()
        archive = subprocess.check_output(["git", "archive", "--format=zip", commit, "src"], cwd=repository)
        archive_path = output / f"{definition['name']}-source.zip"
        archive_path.write_bytes(archive)
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            source_hashes = {name: hashlib.sha256(zipped.read(name)).hexdigest()
                             for name in zipped.namelist() if name.endswith(".py")}
        critical = {p: source_hashes["src/" + p] for p in CRITICAL}
        if common is None:
            common = critical
        elif common != critical:
            raise ValueError("Cross-policy physical/client implementation mismatch")
        spec = json.loads((repository / definition["spec"]).read_text(encoding="utf-8"))
        frozen.append({"name": definition["name"], "commit": commit, "spec": spec,
                       "archive": archive_path.name, "archive_sha256": hashlib.sha256(archive).hexdigest(),
                       "source_sha256": source_hashes})
    manifest = {"scope": "training_reliability_only", "cases_sha256": digest(dataset),
                "case_count": len(cases), "policies": frozen, "shared_physical_source_sha256": common,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "platform": platform.platform(), "python": sys.version,
                "limits": {"virtual_seconds": 360000, "real_seconds": 1200, "max_actions": 20000},
                "runtime_comparison": "Concurrent load: runtime is diagnostic, not a hardware-speed comparison."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "cases_sha256": manifest["cases_sha256"],
                      "policies": [p["name"] for p in frozen], "physical_source_equal": True}))


class ObservationOnlyClient:
    __slots__ = ("_client",)

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        if name not in {"state", "remaining_real_time_s", "pending_request", "enter", "measure", "clear", "exit"}:
            raise AssertionError(f"Policy requested non-observation attribute {name}")
        return getattr(self._client, name)


def audit_actions(history, evaluation):
    errors = []
    totals = {key: 0 for key in COMPONENTS}
    current, channel, microseconds = (0., 0.), 1, 0
    physical_measures = successful_clears = failed_clears = 0
    entered = exited = False
    for index, action in enumerate(history):
        name, response = action["action"], action["response"]
        position = (action["position"]["x"], action["position"]["y"])
        c = action["channel"]
        if not response.get("accepted") or action["index"] != index:
            errors.append(f"Bad action acceptance/index {index}")
        if name == "/enter":
            if entered or index != 0:
                errors.append("Noninitial/repeated enter")
            entered = True
        elif name == "/exit":
            if not entered or exited or index != len(history) - 1:
                errors.append("Invalid exit order")
            exited = True
        elif name in ("/measure", "/clear"):
            if not entered or exited:
                errors.append(f"Action outside active session {index}")
            if type(c) is not int or not 1 <= c <= 20:
                errors.append(f"Illegal channel {index}")
            if not all(math.isfinite(v) and abs(v) <= 2_000_000 for v in position):
                errors.append(f"Illegal coordinate {index}")
            costs = {"movement_s": round(math.dist(position, current) / 5 * 1_000_000)}
            current = position
            if name == "/measure":
                physical_measures += 1
                costs.update(switching_s=int(channel != c) * 1_000_000, detection_s=5_000_000)
                channel = c
                if response.get("measure_result") == "direction":
                    bearing = response.get("svd_deg")
                    if not isinstance(bearing, (float, int)) or not math.isfinite(bearing) or not 0 <= bearing < 360:
                        errors.append(f"Invalid bearing {index}")
                elif response.get("measure_result") not in ("near", "no_signal"):
                    errors.append(f"Invalid measure feedback {index}")
            else:
                success = response.get("clear_result") == "success"
                if response.get("clear_result") not in ("success", "no_target_in_range"):
                    errors.append(f"Invalid clear feedback {index}")
                successful_clears += success
                failed_clears += not success
                costs.update(optical_s=3_000_000, removal_s=2_000_000 * success)
            for key, value in costs.items():
                totals[key] += value
            microseconds += sum(costs.values())
        else:
            errors.append(f"Unknown action {index}: {name}")
        if abs(response["virtual_time_s"] - microseconds / 1_000_000) > 1e-6:
            errors.append(f"Per-action cost mismatch {index}")
    for key in COMPONENTS:
        if abs(totals[key] / 1_000_000 - evaluation["time_breakdown_s"][key]) > 1e-6:
            errors.append(f"Component cost mismatch: {key}")
    for expected, observed, label in ((physical_measures, evaluation["measurement_count"], "measurements"),
                                    (successful_clears, evaluation["cleared_total"], "clears"),
                                    (failed_clears, evaluation["failed_clear_count"], "failed clears")):
        if expected != observed:
            errors.append(f"Evaluator count mismatch: {label}")
    return {"errors": errors, "legal_actions_and_cost_ledger": not errors,
            "accepted_exit": entered and exited, "physical_measures": physical_measures,
            "reconstructed_time_s": microseconds / 1_000_000}


def worker(args):
    # First and only policy imports happen after setting the isolated src path.
    source = Path.cwd() / "src"
    sys.path.insert(0, str(source))
    from simulation import LocalResearchSimulator, Scenario, Source
    import simulation.engine
    if source.resolve() not in Path(simulation.engine.__file__).resolve().parents:
        raise AssertionError("Policy worker imported another branch's simulator")
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    dataset = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    assert digest(dataset) == manifest["cases_sha256"]
    frozen = next(p for p in manifest["policies"] if p["name"] == args.policy)
    for name, expected in frozen["source_sha256"].items():
        assert hashlib.sha256((Path.cwd() / name).read_bytes()).hexdigest() == expected
    policy_output = output / args.policy
    policy_output.mkdir(exist_ok=False)
    module, function = frozen["spec"]["entrypoint"].split(":", 1)
    callback = getattr(importlib.import_module(module), function)
    rows = []
    with (policy_output / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for item in dataset["cases"]:
            raw = item["scenario"]
            assert 113001 <= raw["seed"] <= 113028 and raw["case_id"].startswith("q3-training-stress-")
            assert digest(raw) == item["case_sha256"]
            case = Scenario(**{**raw, "sources": tuple(Source(**s) for s in raw["sources"])})
            limits = manifest["limits"]
            sim = LocalResearchSimulator(case, max_virtual_duration_s=limits["virtual_seconds"],
                                         max_real_duration_s=limits["real_seconds"])
            client = sim.client()
            result, errors, exception = None, [], None
            started = time.perf_counter()
            try:
                result = callback(ObservationOnlyClient(client), problem=3, max_actions=limits["max_actions"],
                                  **frozen["spec"].get("kwargs", {}))
                errors.extend(str(value) for value in (result.error, result.exit_error) if value)
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                exception = traceback.format_exc()
            finally:
                if client.state.session == "active" and client.pending_request is None:
                    try:
                        client.exit()
                    except Exception as error:
                        errors.append(f"CleanupExit: {type(error).__name__}: {error}")
                sim.finish_for_evaluation()
            wall = time.perf_counter() - started
            evaluation = sim.evaluation()
            history = sim.observation_history()
            ledger = audit_actions(history, evaluation)
            errors.extend(ledger["errors"])
            report = result.as_dict() if result else None
            certified = bool(result and result.completion_certified_under_model)
            if result and (result.cleared_count != evaluation["cleared_total"] or
                           abs(result.virtual_time_s - evaluation["virtual_time_s"]) > 1e-6):
                errors.append("Strategy/evaluator count or time mismatch")
            if result and abs(sum(result.time_breakdown.values()) - evaluation["virtual_time_s"]) > 1e-4:
                errors.append("Strategy aggregate cost mismatch")
            if certified and not evaluation["all_cleared"]:
                errors.append("False completeness certificate")
            success = bool(evaluation["all_cleared"] and certified and ledger["accepted_exit"] and not errors)
            row = {"case_id": case.case_id, "case_sha256": item["case_sha256"], "family": item["family"],
                   "seed": case.seed, "policy": args.policy, "success": success,
                   "all_cleared": evaluation["all_cleared"], "completion_certified": certified,
                   "accepted_exit": ledger["accepted_exit"], "time_s": evaluation["virtual_time_s"],
                   "penalized_time_s": evaluation["virtual_time_s"] if success else limits["virtual_seconds"],
                   "wall_s": wall, "cleared": evaluation["cleared_total"], "sources": evaluation["source_total"],
                   "failed_clears": evaluation["failed_clear_count"], "measurements": evaluation["measurement_count"],
                   "actions": len(history), "breakdown": evaluation["time_breakdown_s"], "errors": errors}
            with gzip.open(policy_output / f"{case.case_id}.json.gz", "wt", encoding="utf-8") as trace:
                json.dump({"row": row, "report": report, "history": history, "ledger_audit": ledger,
                           "evaluation_after_termination": evaluation, "exception_traceback": exception}, trace)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            rows.append(row)
            print(f"{args.policy} {case.case_id} {row['time_s']:.3f}s success={success} failed_clear={row['failed_clears']} wall={wall:.2f}s", flush=True)
    return 0 if all(r["success"] and r["failed_clears"] == 0 for r in rows) else 1


def execute(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != manifest["runner_sha256"]:
        raise ValueError("Runner changed after scene/source freeze")
    def run_policy(frozen):
        archive = (output / frozen["archive"]).read_bytes()
        assert hashlib.sha256(archive).hexdigest() == frozen["archive_sha256"]
        work = (ROOT / "tmp" / output.name / frozen["name"]).resolve()
        if (ROOT / "tmp").resolve() not in work.parents:
            raise ValueError("Temporary work path escaped this worktree")
        work.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            for name in zipped.namelist():
                destination = (work / name).resolve()
                if work != destination and work not in destination.parents:
                    raise ValueError("Archive member escaped isolated work directory")
            zipped.extractall(work)
        with (output / f"{frozen['name']}-console.txt").open("w", encoding="utf-8") as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--output", str(output),
                                      "--policy", frozen["name"]], cwd=work, stdout=log, stderr=subprocess.STDOUT)
        print(f"{frozen['name']} finished code={process.returncode}", flush=True)
        return {"policy": frozen["name"], "exit_code": process.returncode}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        completions = list(pool.map(run_policy, manifest["policies"]))
    (output / "worker_completion.json").write_text(json.dumps(completions, indent=2) + "\n", encoding="utf-8")
    return int(any(row["exit_code"] != 0 for row in completions))


def summarize(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    policies = [p["name"] for p in manifest["policies"]]
    indexed, summary = {}, {}
    def stats(rows):
        times = sorted(r["time_s"] for r in rows)
        idx = .95 * (len(times) - 1)
        lo = int(idx)
        return {"n": len(rows), "successes": sum(r["success"] for r in rows),
                "failed_clears": sum(r["failed_clears"] for r in rows),
                "errors": sum(len(r["errors"]) for r in rows), "mean_s": statistics.mean(times),
                "penalized_mean_s": statistics.mean(r["penalized_time_s"] for r in rows),
                "p95_s": times[lo] + (times[min(lo+1, len(times)-1)] - times[lo]) * (idx-lo),
                "maximum_s": times[-1], "maximum_case": max(rows, key=lambda r: r["time_s"])["case_id"],
                "mean_breakdown": {key: statistics.mean(r["breakdown"][key] for r in rows) for key in COMPONENTS}}
    for policy in policies:
        rows = [json.loads(line) for line in (output / policy / "rows.jsonl").read_text().splitlines()]
        indexed[policy] = {r["case_id"]: r for r in rows}
        summary[policy] = {"overall": stats(rows), "families": {f: stats([r for r in rows if r["family"] == f]) for f in FAMILIES}}
    baseline, pairs = indexed[policies[0]], {}
    for policy in policies[1:]:
        assert baseline.keys() == indexed[policy].keys()
        rows = []
        for case_id, old in baseline.items():
            new = indexed[policy][case_id]
            assert old["case_sha256"] == new["case_sha256"]
            rows.append({"case_id": case_id, "family": old["family"], "case_sha256": old["case_sha256"],
                         "saved_s": old["time_s"] - new["time_s"],
                         "ratio": new["time_s"] / old["time_s"], "both_successful": old["success"] and new["success"]})
        pairs[policy] = {"mean_saved_s": statistics.mean(r["saved_s"] for r in rows),
                         "wins": sum(r["saved_s"] > 1e-6 for r in rows),
                         "losses": sum(r["saved_s"] < -1e-6 for r in rows),
                         "ties": sum(abs(r["saved_s"]) <= 1e-6 for r in rows),
                         "worst_regression": min(rows, key=lambda r: r["saved_s"]), "cases": rows}
    result = {"scope": "training_reliability_only_not_final_test", "case_count": manifest["case_count"],
              "platform": manifest["platform"], "policies": summary, "paired_vs_rollout": pairs,
              "runtime_caveat": manifest["runtime_comparison"]}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({p: summary[p]["overall"] for p in policies}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--policies", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--jobs", type=int, choices=(1, 2, 3, 4), default=3)
    child = sub.add_parser("worker")
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--policy", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return {"prepare": freeze, "run": execute, "worker": worker, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
