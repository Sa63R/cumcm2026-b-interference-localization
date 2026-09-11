"""Render the frozen independent Q4 clear-before-probe comparison."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research/q4_clear_before_probe"
LABELS = ("compact_baseline", "compact_combo", "compact_clear_before_probe")
COLORS = ("#94A3B8", "#475569", "#0F766E")


def main():
    qualification = json.loads((RESEARCH / "qualification.json").read_bytes())
    if not qualification["passed"] or qualification["selected"] != LABELS[-1]:
        raise ValueError("This figure requires the frozen qualifying comparison")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), layout="constrained")
    for column, (stage, title) in enumerate((("confirmation", "Random: 64 paired cases"),
                                            ("stress", "Stress: 42 paired cases"))):
        summaries = qualification["summaries"][stage]
        comparison = qualification["comparisons_vs_incumbent"][LABELS[-1]][stage]
        means = [summaries[label]["mean_time_s"] for label in LABELS]
        ratios = [summaries[label]["mean_time_over_mean_lower_bound"] for label in LABELS]
        ax = axes[0, column]
        ax.bar(range(3), means, color=COLORS, width=.65)
        for x, (time, ratio) in enumerate(zip(means, ratios)):
            ax.text(x, time + 85, f"{time:,.2f} s\nT / LB = {ratio:.3f}", ha="center", fontsize=9)
        bound = summaries[LABELS[-1]]["mean_lower_bound_s"]
        ax.axhline(bound, linestyle="--", color="#B45309", linewidth=1.3,
                   label=f"Historical mean LB: {bound:,.2f} s")
        ax.set_xticks(range(3), ("Original 22 stations", "Previous combo", "Clear before probe"))
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylim(0, max(means)*1.18)
        ax.set_ylabel("Mean virtual task time (s)")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax = axes[1, column]
        report = json.loads((ROOT / "results/q4_clear_before_probe" / stage / "summary.json").read_bytes())
        rows = report["rows"]
        pairs = {}
        for row in rows:
            pairs.setdefault(row["seed"], {})[row["strategy"]] = row
        seeds = sorted(pairs)
        saved = [pairs[seed][LABELS[1]]["penalized_time_s"] - pairs[seed][LABELS[2]]["penalized_time_s"] for seed in seeds]
        ax.scatter(range(1, len(saved)+1), saved, s=23,
                   color=["#B91C1C" if value < 0 else COLORS[-1] for value in saved])
        ax.axhline(0, color="#94A3B8", linewidth=1)
        ax.axhline(comparison["mean_saved_s"], color=COLORS[-1], linestyle="--", linewidth=1)
        low, high = comparison["saving_ci95_s"]
        ax.set_title(f"Mean saved {comparison['mean_saved_s']:.2f} s "
                     f"({100*comparison['mean_reduction_fraction']:.3f}%)\n"
                     f"Paired 95% CI: [{low:.2f}, {high:.2f}] s", fontsize=11)
        ax.set_xlabel("Case order (all cases retained)")
        ax.set_ylabel("Previous time - new time (s)")
    fig.suptitle("Q4: same-point clear before a planned measurement", fontsize=16, fontweight="bold")
    fig.supxlabel("Local generated mixed-source scenarios; all cases cleared. "
                  "Historical oracle LB remains a relaxation, not an achievable-time claim.", fontsize=9)
    destination = RESEARCH / "figures"
    destination.mkdir(exist_ok=True)
    fig.savefig(destination / "independent-comparison.png", dpi=180)
    fig.savefig(destination / "independent-comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
