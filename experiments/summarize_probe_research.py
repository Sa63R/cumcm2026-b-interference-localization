"""Aggregate the fixed 128-case tree comparison and export scientific figures."""

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.research_v1_eval import bootstrap_mean_interval, percentile


def main():
    root = ROOT / "results/state_search"
    rows = [json.loads(line) for part in ("tree_training_a", "tree_training_b")
            for line in (root / part / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 640:
        raise ValueError(f"Require all 640 declared runs before aggregation, got {len(rows)}")
    groups = {key: sorted([r for r in rows if r["policy"] == key], key=lambda r: r["seed"])
              for key in sorted({r["policy"] for r in rows})}
    for group in groups.values():
        if [r["seed"] for r in group] != list(range(100001, 100129)):
            raise ValueError("Missing or duplicate training seeds")
    base = groups["radius_v1"]
    summary = {"stage": "training_development_not_final", "seeds": [100001, 100128],
               "parallel_timing_caveat": "Two Windows processes ran shards concurrently; wall times are not isolated throughput measurements",
               "candidate_reference": "radius_v1 from 8aeb3f0, no replacement implied", "policies": {}}
    for name, group in groups.items():
        savings = [b["time_s"] - r["time_s"] for b, r in zip(base, group)]
        summary["policies"][name] = {
            "n": len(group), "successes": sum(r["success"] for r in group),
            "failed_clears": sum(r["failed_clears"] for r in group),
            "mean_s": statistics.mean(r["time_s"] for r in group),
            "p95_s": percentile([r["time_s"] for r in group], .95),
            "mean_wall_s": statistics.mean(r["wall_s"] for r in group),
            "mean_measurements": statistics.mean(r["measurements"] for r in group),
            "mean_components_s": {key: statistics.mean(r["breakdown"][key] for r in group)
                                   for key in group[0]["breakdown"]},
            "paired_mean_saving_vs_v1_s": statistics.mean(savings),
            "paired_saving_ci95_s": bootstrap_mean_interval(savings, seed=20260911, samples=5000),
            "worst_regression_vs_v1_s": max(-s for s in savings),
            "wins_vs_v1": sum(s > 1e-6 for s in savings),
            "losses_vs_v1": sum(s < -1e-6 for s in savings)}
    output = root / "tree_training_summary"
    # Regeneration only replaces these derived summary/figure artifacts.
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    bench = json.loads((root / "tree_search_benchmark/summary.json").read_text())
    axes[0].bar(["Full tree", "Bound pruning"],
                [bench["depth2_boundsFalse"]["mean_runtime_s"], bench["depth2_boundsTrue"]["mean_runtime_s"]],
                color=["#94a3b8", "#0f766e"])
    axes[0].set_title("A. Same finite-tree optimum")
    axes[0].set_ylabel("Mean two-step search time (s)")
    axes[0].text(.5, .92, "16 replayed observation states\nAll optimum values matched", ha="center", va="top", transform=axes[0].transAxes)
    post = json.loads((root / "posterior_diagnostic/summary.json").read_text())
    names = ["discrete_noise1", "discrete_noise3", "bin0.5", "bin1.0"]
    for i, (supports, color) in enumerate(((6, "#94a3b8"), (24, "#0f766e"))):
        axes[1].bar([j + (i - .5) * .36 for j in range(4)],
                    [post[f"{name}_support{supports}"]["mean_singleton_probability"] for name in names],
                    width=.34, color=color, label=f"{supports} source nodes")
    axes[1].set_xticks(range(4), ["1 error", "3 errors", "0.5 deg bin", "1 deg bin"], rotation=20)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Probability of a singleton posterior")
    axes[1].set_title("B. Discretization can invent certainty")
    axes[1].legend(loc="lower left", fontsize=8)
    labels = {"radius_v1": "v1 radius proxy", "tree_fast": "1-step tree",
              "tree_deep": "2-step tree", "tree_dense_noise": "24 nodes + 3 errors"}
    for i, (name, label) in enumerate(labels.items()):
        result = summary["policies"][name]
        axes[2].scatter(result["mean_wall_s"], result["mean_s"], s=65, color=["#0f766e", "#64748b", "#2563eb", "#c2410c"][i])
        offset = (-4, 7) if name == "tree_dense_noise" else (4, 7)
        axes[2].annotate(label, (result["mean_wall_s"], result["mean_s"]),
                         xytext=offset, textcoords="offset points", fontsize=8,
                         ha="right" if name == "tree_dense_noise" else "left")
    axes[2].set_xscale("log")
    from matplotlib.ticker import NullLocator
    axes[2].set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    axes[2].xaxis.set_minor_locator(NullLocator())
    axes[2].set_xlim(.7, 10)
    axes[2].set_xlabel("Parallel development wall time (s)")
    axes[2].set_ylabel("Mean virtual completion time (s)")
    axes[2].set_title("C. 128 training cases, all cases retained")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "finite_tree_tradeoff.png", dpi=180)
    fig.savefig(output / "finite_tree_tradeoff.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
