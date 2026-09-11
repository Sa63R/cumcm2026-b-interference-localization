"""Paired Q3 round-two offline experiments; never uses official simulator APIs.

No parallel workers are started here.  Development and untouched evaluation
suites have disjoint default seed ranges, and summaries never pool their
different distribution groups.  A failed/incomplete run cannot improve the
reported completion-time mean by spending less time before failure.
"""

import argparse
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.q3_confirmation_bound import (
    CERTIFICATE_PATH, confirmation_length_lower_bound, guarantee_time_lower_bound,
)
from experiments.run_q3_fresh import StressEngine, lower_bound, suite as previous_suite
from simulation.cases import Scenario, Source, random_scenario
from strategies.q3_fresh_stepper import StepperQ3


CONFIGS = {
    "B": dict(movable_tail=False, rollout=False, attempts=False),
    "C": dict(movable_tail=True, rollout=False, attempts=False),
    "R": dict(movable_tail=False, rollout=True, attempts=False),
    "CR": dict(movable_tail=True, rollout=True, attempts=False),
    "RA": dict(movable_tail=False, rollout=True, attempts=True),
    "CRA": dict(movable_tail=True, rollout=True, attempts=True),
}
DEFAULT_START = {"development": 960000, "nominal": 980000, "pressure": 970000}
DEFAULT_COUNT = {"development": 12, "nominal": 64, "pressure": 32}
PRESSURE_MODES = (
    ("cluster", 14), ("cluster", 16),
    ("hiddenfar", 14), ("hiddenfar", 16),
    ("collinear", 14), ("collinear", 16),
    ("boundary_min", 10), ("boundary_min", 16),
    ("minimum_radius", 10), ("minimum_radius", 16),
    ("correlated", 12), ("correlated_boundary", 14),
)


def pressure_suite(start=970000, count=32):
    """Deterministic stress cases; adjacent 14/16 layouts share their first 14 sources."""
    answer = []
    for index in range(count):
        family, n = PRESSURE_MODES[index % len(PRESSURE_MODES)]
        seed = start + index
        # Paired source counts change the count, not the pre-existing geometry.
        rng = random.Random(start + (index // 2) * 2 + 710003)
        channels = rng.sample(range(1, 21), 16)
        rotation = rng.uniform(0, 2 * math.pi)
        cos, sin = math.cos(rotation), math.sin(rotation)
        sources = []
        for i in range(16):
            if family in {"cluster", "hiddenfar"}:
                x, y = 850 + rng.uniform(-20, 20), 300 + rng.uniform(-20, 20)
                if family == "hiddenfar" and i == 13:
                    x, y = -1720, rng.uniform(-100, 100)
                radius = 1000.0
            elif family == "collinear":
                x, y = 900 + i * 31, rng.uniform(-0.25, 0.25)
                radius = 1000.0
            elif family in {"boundary_min", "correlated_boundary"}:
                theta = 2 * math.pi * (i + 0.37) / 16
                edge = rng.uniform(1780, 1800)
                x, y = edge * math.cos(theta), edge * math.sin(theta)
                radius = 1000.0
            else:
                theta = rng.uniform(0, 2 * math.pi)
                r = 1800 * math.sqrt(rng.random())
                x, y = r * math.cos(theta), r * math.sin(theta)
                radius = 1000.0 if family == "minimum_radius" else rng.uniform(1000, 1500)
            sources.append(Source(channels[i], x * cos - y * sin, x * sin + y * cos, radius))
        correlated = family.startswith("correlated")
        case = Scenario(f"q3-r2-{family}-{n}-{seed}", 3, start + (index // 2) * 2, tuple(sources[:n]),
                        "uniform" if correlated else "alternating_extreme",
                        "smooth_spatial_error" if correlated else f"Round-two research stress: {family}")
        answer.append(case)
    return answer


def build_suite(name, start=None, count=None):
    start = DEFAULT_START[name] if start is None else start
    count = DEFAULT_COUNT[name] if count is None else count
    if count < 0:
        raise ValueError("Case count must be nonnegative")
    if name == "nominal":
        cases = [random_scenario(3, seed) for seed in range(start, start + count)]
        groups = {case.case_id: "nominal" for case in cases}
    elif name == "pressure":
        cases = pressure_suite(start, count)
        groups = {case.case_id: "pressure" for case in cases}
    else:
        cases = previous_suite(910100, 12, True)
        groups = {case.case_id: "development_original_nominal" if i < 12
                  else "development_original_pressure" for i, case in enumerate(cases)}
        added = pressure_suite(start, count)
        cases += added
        groups.update({case.case_id: "development_added_pressure" for case in added})
    return cases, {"suite_groups": groups}


def all_lower_bounds(case, uncertainty=None):
    bounds = lower_bound(case, uncertainty)
    n = len(case.sources)
    length = max(0.0, 5 * (bounds["physical_lower_bound_s"] - 5 * n))
    bounds.update(source_visit_length_lower_bound_m=length,
                  confirmation_length_lower_bound_m=confirmation_length_lower_bound(n),
                  guarantee_lower_bound_s=guarantee_time_lower_bound(n, length))
    return bounds


def _fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _make_planner(config):
    if not config["rollout"]:
        return None
    from strategies.q3_fresh_rollout import FreshRollout
    return FreshRollout(allow_attempts=config["attempts"], enabled=True)


def run_case(case, names, *, anchors=(), group="unspecified", original_case_group=None,
             uncertainty=None, on_result=None):
    rows, traces = [], {}
    # Evaluator-only calculations: neither the bounds nor ground truth are
    # passed to the controller or its planner.
    bounds = all_lower_bounds(case)
    robust = all_lower_bounds(case, uncertainty) if uncertainty is not None else None
    for name in names:
        config = CONFIGS[name]
        engine = StressEngine(case, anchors)
        controller = StepperQ3(engine.client(), movable_tail=config["movable_tail"])
        planner, result, error = None, {}, None
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        try:
            planner = _make_planner(config)
            result = controller.run(planner=planner)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            engine.finish_for_evaluation("policy_error")
            # Preserve the charged trajectory even if no success result exists.
            result = controller.result()
        wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
        metrics = engine.evaluation()
        truth = metrics.pop("ground_truth")
        certified = bool(result.get("completion_certified", False))
        exited = metrics["simulator_stop_reason"] == "exited"
        # Engine uses its actual session enum; accepted /exit is independently
        # recognized from the legal action history as well.
        observations = engine.observation_history()
        exited = exited or bool(observations and observations[-1]["action"] == "/exit")
        success = bool(error is None and metrics["all_cleared"] and certified and exited)
        row = dict(case_id=case.case_id, strategy=name, suite_group=group,
                   original_case_group=original_case_group or case.case_id,
                   scenario_sha256=_fingerprint(truth), error=error,
                   completion_certified=certified, correct_exit=success,
                   incorrect_exit=bool(exited and not (metrics["all_cleared"] and certified)),
                   success=success, policy_wall_s=wall, policy_cpu_s=cpu,
                   planner_stats=getattr(planner, "stats", {}) if planner else {},
                   confirmation_tail_s=result.get("confirmation_tail_s") if success else None,
                   time_per_source_s=metrics["virtual_time_s"] / len(case.sources),
                   **{k: v for k, v in metrics.items() if k != "case_id"}, **bounds)
        for label in ("physical", "certified", "guarantee"):
            row[f"time_over_{label}_lower_bound"] = row["virtual_time_s"] / bounds[f"{label}_lower_bound_s"]
        if robust:
            for label in ("physical", "certified", "guarantee"):
                value = robust[f"{label}_lower_bound_s"]
                row[f"observation_robust_{label}_lower_bound_s"] = value
                row[f"time_over_observation_robust_{label}_lower_bound"] = row["virtual_time_s"] / value
        if success and row["virtual_time_s"] + 1e-5 < bounds["guarantee_lower_bound_s"]:
            row.update(error="AssertionError: successful execution is below its guarantee lower bound", success=False)
        trace = dict(search=result, observations=observations, evaluation=metrics,
                     scenario=truth, row=row, planner_stats=row["planner_stats"])
        rows.append(row)
        traces[name] = trace
        if on_result is not None:
            on_result(row, trace)
    return rows, traces


def _mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


@lru_cache(maxsize=1024)
def _bootstrap_interval(delta):
    rng = random.Random(20260912)
    boot = sorted(statistics.mean(rng.choices(delta, k=len(delta))) for _ in range(1000))
    return [boot[24], boot[974]]


def _paired(rows, baseline):
    pairable = [r for r in rows if r["case_id"] in baseline]
    completed = [r for r in pairable if r["success"] and baseline[r["case_id"]]["success"]]
    by_original = {}
    for row in completed:
        by_original.setdefault(row["original_case_group"], []).append(
            row["virtual_time_s"] - baseline[row["case_id"]]["virtual_time_s"])
    delta = [_mean(values) for values in by_original.values()]
    ci = None
    if delta:
        ci = _bootstrap_interval(tuple(delta))
    return dict(paired_cases=len(pairable), completed_pairs=len(completed),
                excluded_failed_pairs=len(pairable)-len(completed), original_groups=len(delta),
                mean_group_delta_s=_mean(delta), paired_group_bootstrap_95_delta_s=ci,
                wins=sum(d < -1e-6 for d in delta), losses=sum(d > 1e-6 for d in delta),
                worst_group_regression_s=max(delta) if delta else None,
                best_group_improvement_s=min(delta) if delta else None,
                mean_completed_delta_s=_mean(r["virtual_time_s"]-baseline[r["case_id"]]["virtual_time_s"]
                                             for r in completed))


def summarize_group(rows):
    answer = {}
    baseline = {r["case_id"]: r for r in rows if r["strategy"] == "B"}
    for name in sorted({row["strategy"] for row in rows}):
        subset = [r for r in rows if r["strategy"] == name]
        successful = [r for r in subset if r["success"]]
        times = sorted(r["virtual_time_s"] for r in successful)
        failed = len(successful) != len(subset)
        summary = dict(cases=len(subset), success=len(successful), failures=len(subset)-len(successful),
            incorrect_exits=sum(r["incorrect_exit"] for r in subset),
            incomplete_clearances=sum(not r["all_cleared"] for r in subset),
            missing_certificates=sum(not r["completion_certified"] for r in subset),
            failure_case_ids=[r["case_id"] for r in subset if not r["success"]],
            mean_s=None if failed else _mean(times),
            mean_successful_completion_s=_mean(times),
            mean_observed_including_partial_s=_mean(r["virtual_time_s"] for r in subset),
            p95_s=times[math.ceil(.95*len(times))-1] if times and not failed else None,
            worst_s=max(times) if times and not failed else None,
            worst_case_id=max(successful, key=lambda r: r["virtual_time_s"])["case_id"] if successful else None,
            mean_time_per_source_s=_mean(r["time_per_source_s"] for r in successful),
            mean_measurements=_mean(r["measurement_count"] for r in successful),
            failed_clears=sum(r["failed_clear_count"] for r in subset),
            mean_policy_wall_s=_mean(r["policy_wall_s"] for r in subset),
            max_policy_wall_s=max(r["policy_wall_s"] for r in subset),
            total_policy_cpu_s=sum(r["policy_cpu_s"] for r in subset),
            mean_tail_s=_mean(r["confirmation_tail_s"] for r in successful),
            mean_components={key: _mean(r["time_breakdown_s"][key] for r in successful)
                             for key in subset[0]["time_breakdown_s"]},
            by_source_count={})
        for label in ("physical", "certified", "guarantee"):
            value = _mean(r[f"{label}_lower_bound_s"] for r in subset)
            summary[f"mean_{label}_lb_s"] = value
            summary[f"ratio_{label}"] = summary["mean_s"] / value if not failed else None
        # These are descriptive strata, not extra independent runs.
        for n in sorted({r["source_total"] for r in subset}):
            stratum = [r for r in subset if r["source_total"] == n]
            ok = all(r["success"] for r in stratum)
            summary["by_source_count"][str(n)] = dict(cases=len(stratum),
                success=sum(r["success"] for r in stratum),
                failures=sum(not r["success"] for r in stratum),
                mean_s=_mean(r["virtual_time_s"] for r in stratum) if ok else None,
                mean_time_per_source_s=_mean(r["time_per_source_s"] for r in stratum) if ok else None,
                mean_guarantee_lb_s=_mean(r["guarantee_lower_bound_s"] for r in stratum))
        if baseline:
            summary["paired_vs_B"] = _paired(subset, baseline)
        answer[name] = summary
    return answer


def summarize(rows):
    return {group: summarize_group([r for r in rows if r["suite_group"] == group])
            for group in sorted({r["suite_group"] for r in rows})}


def source_hashes():
    names = ["src/strategies/q3_fresh.py", "src/strategies/q3_fresh_stepper.py",
             "src/strategies/q3_movable_tail.py", "src/strategies/q3_fresh_rollout.py",
             "src/strategies/q3_fresh_belief.py", "src/simulation/engine.py",
             "src/simulation/cases.py", "src/geometry/__init__.py", "src/localization/__init__.py",
             "experiments/run_q3_fresh_round2.py", "experiments/run_q3_fresh.py",
             "experiments/session_lower_bounds.py", "experiments/q3_confirmation_bound.py"]
    paths = [ROOT / name for name in names] + [CERTIFICATE_PATH]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def write_snapshot(out, rows, manifest):
    payload = dict(manifest=manifest, summary=summarize(rows), rows=rows)
    raw = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode()
    temporary = out / "progress.tmp"
    temporary.write_bytes(raw)
    temporary.replace(out / "results.json")
    url = os.environ.get("Q3_PROGRESS_PUT_URL")
    if url:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=raw, method="PUT"), timeout=20) as response:
                response.read()
        except Exception as exc:
            # Never echo the signed URL or credentials into logs.
            print(f"progress_sync_error={type(exc).__name__}", flush=True)


def _load_cases(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = [Scenario(**dict(item, sources=tuple(Source(**s) for s in item["sources"])))
             for item in data["cases"]]
    if any(case.problem != 3 for case in cases):
        raise ValueError("Only Q3 cases are accepted")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique for paired comparison")
    return cases, data.get("metadata", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=DEFAULT_START, default="development")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--configs", choices=CONFIGS, nargs="+", default=["B", "C", "R", "CR"])
    args = parser.parse_args()
    if len(set(args.configs)) != len(args.configs):
        parser.error("Each configuration may appear only once")
    cases, metadata = (_load_cases(args.cases) if args.cases else
                       build_suite(args.suite, args.start, args.count))
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    manifest = dict(kind="offline_practice_reconstruction" if args.cases else "offline_synthetic_research",
        suite="external" if args.cases else args.suite, cases=len(cases), configs=args.configs,
        status="running", completed_cases=0, completed_runs=0, source_sha256=hashes,
        validation_labels_used_for_fitting=False, max_concurrent_processes=1,
        case_ids=[c.case_id for c in cases],
        case_definition_sha256=_fingerprint([c.evaluation_config() for c in cases]),
        uncertainty_bounds_available=bool(metadata.get("uncertainty_radii")),
        total_cpu_s=0.0, total_wall_s=0.0)
    rows = []
    write_snapshot(args.out, rows, manifest)
    for index, case in enumerate(cases):
        def save_run(row, trace):
            rows.append(row)
            with gzip.open(args.out / f"trace-{index:03d}-{row['strategy']}.json.gz", "wt", encoding="utf-8") as file:
                json.dump(trace, file, ensure_ascii=False, allow_nan=False)
            manifest.update(completed_runs=len(rows), total_wall_s=time.perf_counter()-wall_start,
                            total_cpu_s=time.process_time()-cpu_start)
            write_snapshot(args.out, rows, manifest)
            print(json.dumps({key: row[key] for key in ("case_id", "strategy", "success", "virtual_time_s",
                "physical_lower_bound_s", "guarantee_lower_bound_s", "time_over_physical_lower_bound",
                "time_over_guarantee_lower_bound", "policy_cpu_s", "error")}), flush=True)
        groups = metadata.get("suite_groups", {})
        original_groups = metadata.get("original_case_groups", metadata.get("case_groups", {}))
        # Both supported reconstruction conventions preserve the raw-case
        # identity while changing the radius reconstruction suffix.
        inferred_original = case.case_id.split("@")[0].split("-radius-")[0]
        original = original_groups.get(case.case_id, inferred_original)
        run_case(case, args.configs, anchors=metadata.get("anchors", {}).get(case.case_id, ()),
                 group=groups.get(case.case_id, "reconstruction"), original_case_group=original,
                 uncertainty=metadata.get("uncertainty_radii", {}).get(case.case_id), on_result=save_run)
        manifest["completed_cases"] = index + 1
    manifest.update(status="completed", source_unchanged=hashes == source_hashes(),
                    total_wall_s=time.perf_counter()-wall_start, total_cpu_s=time.process_time()-cpu_start)
    write_snapshot(args.out, rows, manifest)


if __name__ == "__main__":
    main()
