"""Serial paired Q3 runtime measurements with exact trajectory equivalence.

Every episode runs in an isolated interpreter. Imports, case generation, output
serialization and post-session physical bounds are outside measured time.
Only the frozen derived-silence policy is accepted; no HTTP or training is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


SPEC_PATH = "experiments/state_search_candidate_derived_silence_combined_v1.json"
BASELINE_COMMIT = "760af8230a8e5bda6050663b19d871a9701c9829"
FAMILIES = ["minimum_radius", "boundary", "cluster", "positive_error",
            "negative_error", "alternating_error", "narrow_strip"]
HARNESS_FILES = ["experiments/research_v1_eval.py", "experiments/run_q3_comparison.py",
                 "experiments/q3_benchmark_bounds.py", "experiments/session_lower_bounds.py"]
SUMMARY_RUNTIME_KEYS = {"program_runtime_s", "runtime_s", "total_runtime_s"}
PROTOCOL = {"limits": {"max_actions": 10000, "real_seconds_per_case": 1200,
                       "virtual_seconds_per_case": 360000},
            "partitions": {"final_stress": {"seed_start": 972100}},
            "final_stress_families": FAMILIES}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def without_keys(value, ignored):
    if isinstance(value, dict):
        return {k: without_keys(v, ignored) for k, v in value.items() if k not in ignored}
    if isinstance(value, list):
        return [without_keys(v, ignored) for v in value]
    return value


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    offset = (len(values) - 1) * fraction
    lower = int(offset)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (offset - lower)


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        if ".." in part:
            first, last = map(int, part.split(".."))
            if last < first:
                raise ValueError("Seed ranges must be ascending and inclusive")
            seeds.extend(range(first, last + 1))
        else:
            seeds.append(int(part))
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Seeds must be nonempty and distinct")
    return seeds


def source_identity(root):
    files = sorted(p.relative_to(root).as_posix() for p in (root / "src").rglob("*.py"))
    files += HARNESS_FILES + [SPEC_PATH]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    if (root / "SOURCE_SNAPSHOT.json").exists():
        frozen = read_json(root / "SOURCE_SNAPSHOT.json")
        for name, expected in frozen["source_sha256"].items():
            path = (root / name).resolve()
            if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen source snapshot differs from SOURCE_SNAPSHOT.json: " + name)
        if not set(hashes).issubset(frozen["source_sha256"]):
            raise ValueError("Snapshot identity omits benchmark-critical files")
        head = frozen["commit"]
    else:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {"git_head": head, "source_sha256": hashes, "aggregate_sha256": digest(hashes),
            "spec_sha256": digest(read_json(root / SPEC_PATH))}


def load_average():
    try:
        return list(os.getloadavg())
    except (AttributeError, OSError):
        return None


def physical_bound(case, shortest_open_path):
    """Previously used oracle physical relaxation; no discovery/certification cost."""
    sources = sorted(case.sources, key=lambda source: source.channel)
    points = [(source.x, source.y) for source in sources]
    first = [max(0.0, math.hypot(*point) - 20.0 - 1e-6) for point in points]
    edges = [[0.0 if i == j else max(0.0, math.dist(a, b) - 40.0 - 1e-6)
              for j, b in enumerate(points)] for i, a in enumerate(points)]
    length, order = shortest_open_path(first, edges)
    return {"case_id": case.case_id, "source_total": len(sources),
            "physical_lower_bound_s": length / 5.0 + 5.0 * len(sources),
            "movement_lower_bound_s": length / 5.0,
            "clear_action_lower_bound_s": 5.0 * len(sources),
            "lower_route_length_m": length,
            "lower_graph_order_channels": [sources[i].channel for i in order],
            "bound_scope": "Oracle physical bound: exact subset-DP on relaxed distance to 20m clearance disks, movement at 5m/s plus 5s per source; excludes discovery, localization, channel switching and empty-channel certification. Not generally attainable online."}


class RunCaseClock:
    """Instrument only run_case's module, without modifying global time APIs."""
    def __init__(self):
        self.samples = []

    def perf_counter(self):
        cpu, wall = time.process_time(), time.perf_counter()
        self.samples.append((cpu, wall))
        return wall


def worker(args):
    source = args.source.resolve()
    os.chdir(source)
    sys.path[:0] = [str(source / "src"), str(source)]
    evaluator = importlib.import_module("experiments.research_v1_eval")
    from simulation import Scenario, Source
    spec = read_json(source / SPEC_PATH)
    module_name, callback = spec["entrypoint"].split(":")
    module = importlib.import_module(module_name)  # Import startup is excluded.
    getattr(module, callback)
    if source not in Path(module.__file__).resolve().parents:
        raise RuntimeError("Worker loaded policy from the wrong source tree")
    raw = read_json(args.cases)[args.case_index]["scenario"]
    case = Scenario(**{**raw, "sources": tuple(Source(**item) for item in raw["sources"])})
    clock = RunCaseClock()
    started_at = datetime.now(timezone.utc).isoformat()
    initial_load = load_average()
    previous_clock = evaluator.time
    evaluator.time = clock
    try:
        record = evaluator.run_case(case, spec, PROTOCOL)
    finally:
        evaluator.time = previous_clock
    if len(clock.samples) != 2:
        raise RuntimeError("run_case timing boundaries changed; refuse ambiguous measurements")
    cpu_seconds = clock.samples[1][0] - clock.samples[0][0]
    summary = record["summary"]
    actions = summary.get("action_history", []) if summary else []
    metrics = {**record["row"], "program_cpu_s": cpu_seconds,
               "action_history_sha256": digest(actions),
               "action_step_sha256": [digest(action) for action in actions],
               "summary_without_runtime_sha256": digest(without_keys(summary, SUMMARY_RUNTIME_KEYS)),
               "evaluation_without_runtime_sha256": digest(without_keys(record["evaluation"], {"wall_time_s"})),
               "observations_without_timestamp_sha256": digest(without_keys(record["history"], {"real_timestamp_ms"})),
               "row_without_runtime_sha256": digest(without_keys(record["row"], {"program_runtime_s"})),
               "started_at_utc": started_at, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
               "initial_load_average": initial_load, "final_load_average": load_average()}
    record["runtime_equivalence_metrics"] = metrics
    with args.record.open("xb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as zipped:
            zipped.write(canonical(record).encode("utf-8"))
    print(canonical(metrics), flush=True)


COMPARISONS = ("case_sha256", "action_history_sha256", "action_step_sha256",
               "summary_without_runtime_sha256", "evaluation_without_runtime_sha256",
               "observations_without_timestamp_sha256", "row_without_runtime_sha256",
               "successful", "all_cleared", "completion_certified", "accepted_exit",
               "source_total", "cleared_total", "measurement_count", "failed_clear_count",
               "action_count", "virtual_time_s")


def compare_pair(baseline, candidate):
    differences = [name for name in COMPARISONS if baseline[name] != candidate[name]]
    steps_b, steps_c = baseline["action_step_sha256"], candidate["action_step_sha256"]
    first_different = next((i for i, (b, c) in enumerate(zip(steps_b, steps_c)) if b != c), None)
    if first_different is None and len(steps_b) != len(steps_c):
        first_different = min(len(steps_b), len(steps_c))
    return {"identical": not differences, "different_fields": differences,
            "first_different_action_index": first_different}


def summarize(rows, pairs):
    groups = {}
    for side in ("baseline", "candidate"):
        selected = [r for r in rows if r["side"] == side]
        groups[side] = {"runs": len(selected), "success_rate": statistics.mean(r["successful"] for r in selected),
                        "failed_clear_count": sum(r["failed_clear_count"] for r in selected),
                        "mean_virtual_time_s": statistics.mean(r["virtual_time_s"] for r in selected),
                        "p95_virtual_time_s": percentile([r["virtual_time_s"] for r in selected], .95),
                        "max_virtual_time_s": max(r["virtual_time_s"] for r in selected),
                        "mean_time_over_physical_lower_bound": statistics.mean(r["time_over_physical_lower_bound"] for r in selected)}
        for metric in ("program_runtime_s", "program_cpu_s"):
            values = [r[metric] for r in selected]
            groups[side].update({"mean_" + metric: statistics.mean(values),
                                 "p95_" + metric: percentile(values, .95),
                                 "max_" + metric: max(values)})
    reductions = {}
    for aggregate in ("mean", "p95"):
        for metric in ("program_runtime_s", "program_cpu_s"):
            name = aggregate + "_" + metric
            reductions[name + "_reduction_fraction"] = 1 - groups["candidate"][name] / groups["baseline"][name]
    return {"groups": groups, "runtime_reductions": reductions,
            "all_pairs_exactly_identical": all(p["identical"] for p in pairs),
            "different_pairs": [p for p in pairs if not p["identical"]],
            "all_runs_successful": all(r["successful"] for r in rows),
            "scope": "Synthetic research cases, not official tests. Runtime reduction is hardware/load dependent. Identical observed trajectories are finite-case evidence, not a proof for every possible input."}


def main(args):
    baseline, candidate, output = args.baseline.resolve(), args.candidate.resolve(), args.output.resolve()
    if args.repeats < 1:
        raise ValueError("At least one repeat is required; two are needed for AB/BA balancing")
    if args.stress_count < 0 or args.stress_count > 14:
        raise ValueError("Stress count must be in 0..14")
    baseline_identity, candidate_identity = source_identity(baseline), source_identity(candidate)
    if baseline_identity["git_head"] != BASELINE_COMMIT:
        raise ValueError("Baseline HEAD must equal frozen derived-silence commit " + BASELINE_COMMIT)
    if (baseline / "SOURCE_SNAPSHOT.json").exists():
        if read_json(baseline / "SOURCE_SNAPSHOT.json").get("source_dirty") is not False:
            raise ValueError("Frozen baseline snapshot must have source_dirty=false")
    else:
        subprocess.run(["git", "diff", "--exit-code", BASELINE_COMMIT, "--", "src", *HARNESS_FILES, SPEC_PATH],
                       cwd=baseline, check=True, stdout=subprocess.DEVNULL)
    if baseline_identity["spec_sha256"] != candidate_identity["spec_sha256"]:
        raise ValueError("Policy specs differ; runtime-only comparison requires identical settings")
    common = HARNESS_FILES + [name for name in baseline_identity["source_sha256"]
                             if name.startswith(("src/simulation/", "src/simulator_client/"))]
    if any(baseline_identity["source_sha256"][name] != candidate_identity["source_sha256"].get(name) for name in common):
        raise ValueError("Simulator/client/evaluation code differs between sources")
    output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(baseline / "src"), str(baseline)]
    from experiments.research_v1_eval import make_case
    from experiments.session_lower_bounds import shortest_open_path
    seed_text = args.seeds or ("6000" if args.smoke else "6000..6007,972000..972031")
    cases = []
    for seed in parse_seeds(seed_text):
        case = make_case("validation", seed, PROTOCOL)
        cases.append({"split": "development" if 6000 <= seed <= 6007 else "independent_random",
                      "scenario": case.evaluation_config()})
    stress_count = 0 if args.smoke else args.stress_count
    for seed in range(972100, 972100 + stress_count):
        case = make_case("final_stress", seed, PROTOCOL)
        cases.append({"split": "independent_stress", "scenario": case.evaluation_config()})
    write_json(output / "cases.json", cases)
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                "baseline": baseline_identity, "candidate": candidate_identity,
                "cases_sha256": digest(cases), "protocol": PROTOCOL, "spec": read_json(baseline / SPEC_PATH),
                "repeats": args.repeats, "case_count": len(cases), "python": sys.version,
                "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "timing_scope": "Original run_case perf_counter boundaries; CPU measured at those same boundaries. Strategy modules preimported. Interpreter startup, scenario generation, lower bounds and output serialization excluded. Includes local client and policy cleanup; no network latency.",
                "schedule": "serial paired subprocesses; AB/BA reversed each case and each repeat",
                "balanced_ab_ba": args.repeats % 2 == 0,
                "initial_load_average": load_average(),
                "normalization": {"summary": sorted(SUMMARY_RUNTIME_KEYS), "evaluation": ["wall_time_s"],
                                  "observation_history": ["real_timestamp_ms"]}}
    write_json(output / "manifest.json", manifest)
    rows, pairs = [], []
    environment = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    started = time.perf_counter()
    with (output / "runs.jsonl").open("x", encoding="utf-8") as stream:
        for repeat in range(args.repeats):
            for index, item in enumerate(cases):
                order = ("baseline", "candidate") if (repeat + index) % 2 == 0 else ("candidate", "baseline")
                current = {}
                for side in order:
                    record_name = f"r{repeat:02d}-c{index:03d}-{side}.json.gz"
                    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--source",
                               str(baseline if side == "baseline" else candidate), "--cases", str(output / "cases.json"),
                               "--case-index", str(index), "--record", str(output / record_name)]
                    proc = subprocess.run(command, text=True, encoding="utf-8", capture_output=True, env=environment,
                                          timeout=PROTOCOL["limits"]["real_seconds_per_case"] + 60)
                    if proc.returncode:
                        write_json(output / "worker_failure.json", {"repeat": repeat, "case_index": index,
                                   "side": side, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
                        raise RuntimeError("Worker failed; see worker_failure.json")
                    row = json.loads(proc.stdout)
                    row.update(side=side, repeat=repeat, split=item["split"], record_file=record_name)
                    rows.append(row)
                    current[side] = row
                    stream.write(canonical(row) + "\n")
                    stream.flush()
                pair = {"repeat": repeat, "case_id": current["baseline"]["case_id"],
                        **compare_pair(current["baseline"], current["candidate"])}
                pairs.append(pair)
                print(canonical({**pair, "baseline_wall_s": current["baseline"]["program_runtime_s"],
                                 "candidate_wall_s": current["candidate"]["program_runtime_s"]}), flush=True)
    # The evaluator first uses source truth for bounds after every policy has ended.
    bounds = {}
    for item in cases:
        from simulation import Scenario, Source
        raw = item["scenario"]
        case = Scenario(**{**raw, "sources": tuple(Source(**s) for s in raw["sources"])})
        bounds[case.case_id] = physical_bound(case, shortest_open_path)
    write_json(output / "lower_bounds.json", bounds)
    for row in rows:
        row["physical_lower_bound_s"] = bounds[row["case_id"]]["physical_lower_bound_s"]
        row["time_over_physical_lower_bound"] = row["virtual_time_s"] / row["physical_lower_bound_s"]
    write_json(output / "runs_with_bounds.json", rows)
    unchanged = source_identity(baseline) == baseline_identity and source_identity(candidate) == candidate_identity
    result = {**summarize(rows, pairs), "case_count": len(cases), "repeats": args.repeats,
              "paired_runs": len(pairs), "source_unchanged": unchanged,
              "balanced_ab_ba": args.repeats % 2 == 0,
              "final_load_average": load_average(),
              "elapsed_including_startup_and_bounds_s": time.perf_counter() - started,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "pairs": pairs}
    result["valid_runtime_only_result"] = unchanged and result["all_pairs_exactly_identical"] and result["all_runs_successful"]
    write_json(output / "summary.json", result)
    print(canonical({k: v for k, v in result.items() if k != "pairs"}), flush=True)
    return 0 if result["valid_runtime_only_result"] else 2


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--baseline", type=Path)
    result.add_argument("--candidate", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--seeds", help="Comma-separated seeds or inclusive ranges, e.g. 6000..6007,972000")
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument("--stress-count", type=int, default=14)
    result.add_argument("--smoke", action="store_true", help="One development case, no stress, still AB/BA")
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--cases", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--case-index", type=int, help=argparse.SUPPRESS)
    result.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.worker:
        worker(arguments)
    elif None in (arguments.baseline, arguments.candidate, arguments.output):
        parser().error("--baseline, --candidate and --output are required")
    else:
        sys.exit(main(arguments))
