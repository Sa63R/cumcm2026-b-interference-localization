"""Summarize frozen layout experiments and exact v1 pruning evidence."""

import gzip
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.research_v1_eval import bootstrap_mean_interval, percentile


def read_trace(root, seed, method):
    with gzip.open(root / f"q3-random-{seed}--{method}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)["report"]


def main():
    root = ROOT / "results/state_search"
    summary = {"stage": "training_development_not_final", "experiments": {}}
    for folder in ("layout_pilot", "layout_ablation", "layout_training", "radius_pruning_training"):
        rows = [json.loads(line) for line in (root / folder / "runs.jsonl").read_text().splitlines()]
        methods = sorted({r["policy"] for r in rows})
        groups = {key: sorted([r for r in rows if r["policy"] == key], key=lambda r: r["seed"])
                  for key in methods}
        base = groups["radius_v1"]
        results = {}
        for name, group in groups.items():
            if [r["seed"] for r in group] != [r["seed"] for r in base]:
                raise ValueError("Unpaired seeds")
            delta = [r["time_s"] - b["time_s"] for r, b in zip(group, base)]
            results[name] = {"n": len(group), "successes": sum(r["success"] for r in group),
                "failed_clears": sum(r["failed_clears"] for r in group),
                "mean_s": statistics.mean(r["time_s"] for r in group),
                "p95_s": percentile([r["time_s"] for r in group], .95),
                "delta_vs_v1_s": statistics.mean(delta),
                "delta_ci95_s": bootstrap_mean_interval(delta, seed=20260911, samples=5000),
                "wins_vs_v1": sum(d < -1e-6 for d in delta),
                "worst_regression_s": max(delta),
                "mean_wall_s": statistics.mean(r["wall_s"] for r in group),
                "mean_components_s": {key: statistics.mean(r["breakdown"][key] for r in group)
                                      for key in group[0]["breakdown"]}}
        summary["experiments"][folder] = results
    exact = []
    for seed in range(100225, 100241):
        before = read_trace(root / "radius_pruning_training", seed, "radius_v1")
        after = read_trace(root / "radius_pruning_training", seed, "pruned_v1")
        same = (before["action_history"] == after["action_history"]
                and before["virtual_time_s"] == after["virtual_time_s"])
        if not same:
            raise AssertionError(f"Pruning altered action history: {seed}")
        exact.append({"seed": seed, "exactly_equal": same, "actions": len(after["action_history"])})
    summary["pruning_exact_history_audit"] = exact
    output = root / "layout_summary"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["experiments"]["layout_training"], indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    methods = ["radius_v1", "fixed1123", "phase1123"]
    for ax, method in zip(axes, methods):
        report = read_trace(root / "layout_ablation", 100157, method)
        points = [[0, 0]] + [item["position"] for item in report["action_history"]]
        ax.plot([p[0] for p in points], [p[1] for p in points], color="#64748b", lw=1)
        covers = {tuple(item["position"]) for item in report["action_history"] if item["phase"] == "coverage"}
        ax.scatter([p[0] for p in covers], [p[1] for p in covers], marker="s", s=22, color="#2563eb", label="Discovery scans")
        clears = [item["position"] for item in report["action_history"] if item["action"] == "clear"]
        ax.scatter([p[0] for p in clears], [p[1] for p in clears], marker="x", s=23, color="#c2410c", label="Legal clears")
        ax.add_patch(plt.Circle((0, 0), 1800, fill=False, color="#94a3b8", lw=.8))
        ax.set_aspect("equal")
        ax.set_xlim(-1900, 1900)
        ax.set_ylim(-1900, 1900)
        ax.set_title(f"{method}: {report['virtual_time_s']:.1f} s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Observed-history failure case 100157: finite route gain did not predict later discoveries")
    fig.savefig(output / "layout_regression_case.png", dpi=180)
    fig.savefig(output / "layout_regression_case.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
