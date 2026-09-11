"""Offline paired V0/V1 experiments; never invokes official simulator APIs."""

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.session_lower_bounds import shortest_open_path, empty_channel_action_bound
from simulation.cases import Scenario, Source, random_scenario, difficult_scenarios
from simulation.engine import LocalResearchSimulator
from strategies.q3_fresh import run_fresh


CONFIGS = {"v0": ("v0", {}), "v1": ("v1", {}),
           "v1_selective": ("v1", {"selective": True}),
           "v1_fixed_cover": ("v1", {"dynamic": False}),
           "v1_center_clear": ("v1", {"nearest": False})}


class StressEngine(LocalResearchSimulator):
    def __init__(self, case, anchors=()):
        super().__init__(case)
        self.anchors = {(a["channel"], tuple(a["position"])): a["response"] for a in anchors}

    def _execute(self, path, payload):
        response = super()._execute(path, payload)
        if path == "/measure" and payload["channel"] not in self._cleared:
            q = payload["position"]
            old = self.anchors.get((payload["channel"], (q["x"], q["y"])))
            if old is not None:
                response.pop("svd_deg", None)
                response["measure_result"] = old["measure_result"]
                if "svd_deg" in old:
                    response["svd_deg"] = old["svd_deg"]
                self._history[-1]["response"] = response.copy()
        return response

    def _error(self, position, channel):
        if self.scenario.description == "smooth_spatial_error":
            return math.sin(position.x / 250 + position.y / 400 + channel)
        return super()._error(position, channel)


def suite(start, count, hard):
    cases = [random_scenario(3, i) for i in range(start, start + count)]
    if hard:
        cases += difficult_scenarios(3)
        base = random_scenario(3, 920000)
        for radius in (1000, 1500):
            for mode in ("positive_extreme", "negative_extreme"):
                sources = tuple(Source(i + 1,
                    1800 * math.cos((i + 0.5) * 2 * math.pi / 16),
                    1800 * math.sin((i + 0.5) * 2 * math.pi / 16), radius)
                    for i in range(16))
                cases.append(Scenario(f"edge16-{radius}-{mode}", 3, 920000,
                                      sources, mode, "Rotated boundary; 16 sources"))
        cases.append(replace(base, case_id="smooth-spatial", description="smooth_spatial_error"))
    return cases


def lower_bound(case, uncertainty=None):
    """Same open-Hamiltonian edge relaxation used by session_lower_bounds.

    Synthetic cases use true centers after completion; reconstructed cases also
    report an observation-robust relaxation with source-label radius deducted.
    """
    radii = [0.0 if uncertainty is None else uncertainty[str(s.channel)] for s in case.sources]
    first = [max(0.0, math.hypot(s.x, s.y) - 20 - r - 1e-6)
             for s, r in zip(case.sources, radii)]
    edges = [[0.0 if i == j else max(0.0,
              math.hypot(a.x - b.x, a.y - b.y) - 40 - radii[i] - radii[j] - 1e-6)
              for j, b in enumerate(case.sources)] for i, a in enumerate(case.sources)]
    length, _ = shortest_open_path(first, edges)
    physical = length / 5 + 5 * len(case.sources)
    certificate = physical + ((20 - len(case.sources)) * empty_channel_action_bound(3)
                              if len(case.sources) < 16 else 0)
    return dict(physical_lower_bound_s=physical, certified_lower_bound_s=certificate)


def run_case(case, names, anchors=()):
    rows = []
    traces = {}
    for name in names:
        version, config = CONFIGS[name]
        engine = StressEngine(case, anchors)
        started = time.perf_counter()
        result, error = {}, None
        try:
            result = run_fresh(engine.client(), version, **config)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            engine.finish_for_evaluation("policy_error")
        metrics = engine.evaluation()
        elapsed = time.perf_counter() - started
        truth = metrics.pop("ground_truth")
        rows.append(dict(case_id=case.case_id, strategy=name, error=error,
                         completion_certified=result.get("completion_certified", False),
                         confirmation_tail_s=result.get("confirmation_tail_s"),
                         policy_wall_s=elapsed, **{k: v for k, v in metrics.items() if k != "case_id"}))
        traces[name] = dict(search=result, evaluation=metrics, scenario=truth)
    bound = lower_bound(case)
    for row in rows:
        row.update(bound)
        row["time_over_physical_lower_bound"] = row["virtual_time_s"] / bound["physical_lower_bound_s"]
        row["time_over_certified_lower_bound"] = row["virtual_time_s"] / bound["certified_lower_bound_s"]
        if row["all_cleared"] and row["virtual_time_s"] + 1e-5 < bound["physical_lower_bound_s"]:
            raise AssertionError("Invalid lower bound")
    return rows, traces


def summarize(rows):
    answer = {}
    for name in sorted({r["strategy"] for r in rows}):
        subset = [r for r in rows if r["strategy"] == name]
        times = sorted(r["virtual_time_s"] for r in subset)
        physical = statistics.mean(r["physical_lower_bound_s"] for r in subset)
        certified = statistics.mean(r["certified_lower_bound_s"] for r in subset)
        successful = [r for r in subset if r["all_cleared"] and r["completion_certified"] and r["error"] is None]
        answer[name] = dict(cases=len(subset), success=len(successful),
            mean_s=statistics.mean(times), p95_s=times[math.ceil(0.95 * len(times)) - 1],
            worst_s=max(times), mean_physical_lb_s=physical, mean_certified_lb_s=certified,
            ratio_physical=statistics.mean(times) / physical,
            ratio_certified=statistics.mean(times) / certified,
            mean_case_ratio=statistics.mean(r["time_over_physical_lower_bound"] for r in subset),
            mean_measurements=statistics.mean(r["measurement_count"] for r in subset),
            failed_clears=sum(r["failed_clear_count"] for r in subset),
            mean_policy_wall_s=statistics.mean(r["policy_wall_s"] for r in subset),
            max_policy_wall_s=max(r["policy_wall_s"] for r in subset),
            mean_tail_s=statistics.mean(r["confirmation_tail_s"] or 0 for r in subset),
            mean_components={k: statistics.mean(r["time_breakdown_s"][k] for r in subset)
                             for k in subset[0]["time_breakdown_s"]})
    base = {r["case_id"]: r for r in rows if r["strategy"] == "v0"}
    for name, stats in answer.items():
        paired = [r for r in rows if r["strategy"] == name and r["case_id"] in base]
        if paired:
            delta = [r["virtual_time_s"] - base[r["case_id"]]["virtual_time_s"] for r in paired]
            stats["paired_vs_v0"] = dict(mean_delta_s=statistics.mean(delta),
                wins=sum(d < -1e-6 for d in delta), losses=sum(d > 1e-6 for d in delta),
                worst_regression_s=max(delta), best_improvement_s=min(delta))
    return answer


def write_snapshot(out, rows, manifest):
    payload = dict(manifest=manifest, summary=summarize(rows) if rows else {}, rows=rows)
    raw = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode()
    temp = out / "progress.tmp"
    temp.write_bytes(raw)
    temp.replace(out / "results.json")
    url = os.environ.get("Q3_PROGRESS_PUT_URL")
    if url:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=raw, method="PUT"), timeout=30) as response:
                response.read()
        except Exception as exc:
            print(f"progress_sync_error={type(exc).__name__}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=910100)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=["v0", "v1"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = suite(args.start, args.count, args.hard)
    metadata = {}
    if args.cases:
        data = json.loads(args.cases.read_text())
        metadata = data.get("metadata", {})
        cases = [Scenario(**dict(c, sources=tuple(Source(**s) for s in c["sources"]))) for c in data["cases"]]
    files = ["src/strategies/q3_fresh.py", "experiments/run_q3_fresh.py", "src/simulation/engine.py",
             "src/geometry/__init__.py", "src/localization/__init__.py"]
    hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    public_metadata = {k: v for k, v in metadata.items() if k != "anchors"}
    manifest = dict(kind="offline_synthetic_research" if not args.cases else "offline_practice_reconstruction",
                    cases=len(cases), configs=args.configs, status="running", source_sha256=hashes,
                    validation_labels_used_for_fitting=False, metadata=public_metadata)
    rows = []
    for i, case in enumerate(cases):
        batch, traces = run_case(case, args.configs, metadata.get("anchors", {}).get(case.case_id, ()))
        if case.case_id in metadata.get("uncertainty_radii", {}):
            robust = lower_bound(case, metadata["uncertainty_radii"][case.case_id])
            for row in batch:
                row["observation_robust_physical_lower_bound_s"] = robust["physical_lower_bound_s"]
                row["time_over_observation_robust_lower_bound"] = row["virtual_time_s"] / robust["physical_lower_bound_s"]
        rows += batch
        import gzip
        with gzip.open(args.out / f"trace-{i:03d}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False)
        manifest["completed_cases"] = i + 1
        write_snapshot(args.out, rows, manifest)
        print(json.dumps(dict(case=i + 1, total=len(cases), case_id=case.case_id,
                             runs=[{k: r[k] for k in ("strategy", "all_cleared", "virtual_time_s",
                                   "time_over_physical_lower_bound", "error")} for r in batch])), flush=True)
    manifest["status"] = "completed"
    manifest["source_unchanged"] = hashes == {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    write_snapshot(args.out, rows, manifest)


if __name__ == "__main__":
    main()
