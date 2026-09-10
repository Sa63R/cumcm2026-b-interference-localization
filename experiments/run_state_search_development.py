"""Small paired development runs; no official simulator or final-test seeds."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation import LocalResearchSimulator, difficult_scenarios, random_scenario
from strategies import run_search
from strategies.state_search import run_state_search


def execute(args):
    if not (2000 <= args.seed and args.seed + args.count <= 5100 or
            6000 <= args.seed and args.seed + args.count <= 6048 or
            100000 <= args.seed and args.seed + args.count <= 200000):
        raise ValueError("Only declared development/validation seed intervals are allowed")
    configs = json.loads(args.configs.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (ROOT / "src").rglob("*.py")}
    manifest = {"origin": "synthetic_research", "stage": "development",
                "seed_start": args.seed, "count": args.count, "hard": args.hard,
                "configs": configs, "source_sha256": sources,
                "model_bounds_are_not_q3_bounds": True}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    cases = [random_scenario(3, seed) for seed in range(args.seed, args.seed + args.count)]
    if args.hard:
        cases.extend(difficult_scenarios(3))
    rows = []
    with (args.output / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            for label, config in [("efficient", None), *configs.items()]:
                sim = LocalResearchSimulator(case)
                started = time.perf_counter()
                report = (run_search(sim.client(), variant="efficient") if config is None else
                          run_state_search(sim.client(), config=config))
                elapsed = time.perf_counter() - started
                evaluation = sim.evaluation()
                log = report.strategy_parameters.get("planning_log", [])
                row = {"case_id": case.case_id, "seed": case.seed, "policy": label,
                       "time_s": report.virtual_time_s, "wall_s": elapsed,
                       "success": bool(evaluation["all_cleared"] and
                                       report.completion_certified_under_model and
                                       not report.error and not report.exit_error),
                       "failed_clears": evaluation["failed_clear_count"],
                       "measurements": evaluation["measurement_count"],
                       "source_total": evaluation["source_total"],
                       "breakdown": evaluation["time_breakdown_s"],
                       "planning_s": sum(p["runtime_s"] for p in log),
                       "expanded": sum(p["expanded"] for p in log),
                       "exact_plans": sum(p["exact"] for p in log), "plans": len(log)}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
                if args.traces or not row["success"]:
                    with gzip.open(args.output / f"{case.case_id}--{label}.json.gz", "wt", encoding="utf-8") as trace:
                        json.dump({"report": report.as_dict(), "evaluation_after_exit": evaluation}, trace)
                print(f"{case.case_id} {label} {row['time_s']:.2f}s {elapsed:.3f}s wall success={row['success']}", flush=True)
    summary = {}
    for label in ["efficient", *configs]:
        subset = [row for row in rows if row["policy"] == label and "-random-" in row["case_id"]]
        summary[label] = {"mean_s": statistics.mean(row["time_s"] for row in subset),
                          "mean_wall_s": statistics.mean(row["wall_s"] for row in subset),
                          "successes": sum(row["success"] for row in subset), "n": len(subset)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--traces", action="store_true")
    raise SystemExit(execute(parser.parse_args()))
