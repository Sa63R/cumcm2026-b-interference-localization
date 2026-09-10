"""Aggregate both fixed candidate families, including every training case."""

import gzip
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.research_v1_eval import bootstrap_mean_interval, percentile


def main():
    root = ROOT / "results/state_search"
    summary = {"stage": "training_development_not_final", "experiments": {}}
    for folder, expected in (("refinement_pilot", 16), ("refinement_training", 64)):
        rows = [json.loads(line) for line in (root / folder / "runs.jsonl").read_text().splitlines()]
        groups = {key: sorted([r for r in rows if r["policy"] == key], key=lambda r: r["seed"])
                  for key in sorted({r["policy"] for r in rows})}
        base = groups["pruned_v1"]
        if len(base) != expected or len(rows) != 4 * expected:
            raise ValueError("All planned cases must finish before aggregation")
        result = {}
        for name, group in groups.items():
            if [r["seed"] for r in group] != [r["seed"] for r in base]:
                raise ValueError("Incomplete or unpaired group")
            saving = [b["time_s"] - r["time_s"] for r, b in zip(group, base)]
            probes = []
            for row in group:
                with gzip.open(root / folder / f'{row["case_id"]}--{name}.json.gz', "rt", encoding="utf-8") as f:
                    probes.extend(json.load(f)["report"]["strategy_parameters"].get("probe_search_log", []))
            result[name] = {"n": len(group), "successes": sum(r["success"] for r in group),
                "failed_clears": sum(r["failed_clears"] for r in group),
                "mean_s": statistics.mean(r["time_s"] for r in group),
                "p95_s": percentile([r["time_s"] for r in group], .95),
                "saving_vs_v1_s": statistics.mean(saving),
                "saving_ci95_s": bootstrap_mean_interval(saving, seed=20260911, samples=5000),
                "wins_vs_v1": sum(s > 1e-6 for s in saving),
                "losses_vs_v1": sum(s < -1e-6 for s in saving),
                "worst_regression_s": max(-s for s in saving),
                "mean_wall_s": statistics.mean(r["wall_s"] for r in group),
                "mean_measurements": statistics.mean(r["measurements"] for r in group),
                "mean_components_s": {key: statistics.mean(r["breakdown"][key] for r in group)
                                      for key in group[0]["breakdown"]}}
            if probes:
                result[name]["probe_diagnostics"] = {"decisions": len(probes),
                    "mean_legal_candidates": statistics.mean(p["candidates"] for p in probes),
                    "geometry_updates": sum(p.get("total_geometry_updates", p["geometry_updates"]) for p in probes),
                    "pruned_candidates": sum(p["pruned_candidates"] for p in probes),
                    "changed_from_old_best": sum(p.get("changed_point", False) for p in probes),
                    "mean_local_score_improvement_s": statistics.mean(p.get("score_improvement_s", 0) for p in probes)}
        summary["experiments"][folder] = result
    output = root / "refinement_summary"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["experiments"]["refinement_training"], indent=2))


if __name__ == "__main__":
    main()
