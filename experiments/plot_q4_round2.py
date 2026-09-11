"""Plot completed local synthetic Q4 round-2 summaries, without running policies."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
METHODS = (("compact_baseline", "Baseline"), ("compact_range", "Range"), ("compact_combo", "Combo"))
COMPONENTS = (("movement_s", "Movement", "#446d98"), ("detection_s", "Detection", "#2a9a94"),
              ("switching_s", "Channel switch", "#d6ac51"), ("optical_s", "Optical checks", "#bd795f"),
              ("removal_s", "Removal", "#8d839e"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/q4_round2")
    parser.add_argument("--output", type=Path, default=ROOT / "research/q4_round2/comparison.png")
    args = parser.parse_args(argv)
    inputs = {}
    for stage in ("confirmation", "stress"):
        path = args.input / stage / "summary.json"
        raw = path.read_bytes()
        inputs[stage] = (json.loads(raw)["summaries"], hashlib.sha256(raw).hexdigest())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "text.color": "#27364b", "axes.labelcolor": "#4b5b6d"})
    figure, axes = plt.subplots(1, 2, figsize=(12., 5.3), sharex=True)
    figure.subplots_adjust(left=.09, right=.985, top=.72, bottom=.24, wspace=.27)
    for axis, (stage, title) in zip(axes, (("confirmation", "Confirmation"), ("stress", "Stress"))):
        summaries = inputs[stage][0]
        baseline = summaries[METHODS[0][0]]
        n, lower = baseline["runs"], baseline["mean_lower_bound_s"]
        for y, (method, _) in zip((2, 1, 0), METHODS):
            row = summaries[method]
            if row["successful"] != n or row["runs"] != n:
                raise ValueError("This stacked figure requires fully completed matched groups; failures need an explicit penalty component")
            if not math.isclose(row["mean_lower_bound_s"], lower, abs_tol=1e-8):
                raise ValueError("Paired mean lower bound differs")
            mean = row["mean_time_s"]
            parts = row["mean_components_s"]
            if not math.isclose(sum(parts.values()), mean, abs_tol=1e-6):
                raise ValueError("Stack components do not sum to the mean time")
            ratio = mean / lower
            if not math.isclose(ratio, row["mean_time_over_mean_lower_bound"], abs_tol=1e-10):
                raise ValueError("Ratio of means mismatch")
            left = 0.
            for key, _, color in COMPONENTS:
                axis.barh(y, parts[key], left=left, height=.50, color=color, edgecolor="white", linewidth=.35)
                left += parts[key]
            axis.text(mean+130., y, f"{mean:,.0f} s\n{ratio:.3f} x", va="center", ha="left", fontsize=10)
        axis.set_title(f"{title}: {n} paired cases\nMean LB = {lower:,.2f} s", fontsize=12, pad=15, linespacing=1.65)
        axis.set(yticks=(2, 1, 0), yticklabels=[name for _, name in METHODS], ylim=(-.6, 2.6),
                 xlim=(0., 10800.), xticks=(0, 2000, 4000, 6000, 8000, 10000), xlabel="Mean virtual time (s)")
        axis.grid(axis="x", color="#dfe5ec", linewidth=.65)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", length=0, labelsize=10)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle("Q4 Round 2: local synthetic evaluation", fontsize=16, fontweight="bold", y=.97)
    figure.text(.5, .885, "All methods completed every case. Bar labels show mean time and mean T / mean LB.",
                ha="center", fontsize=10.5, color="#54647a")
    figure.legend(handles=[Patch(facecolor=color, label=name) for _, name, color in COMPONENTS],
                  loc="lower center", bbox_to_anchor=(.5, .085), ncol=5, frameon=False,
                  handlelength=1.4, columnspacing=1.8, fontsize=10)
    figure.text(.5, .025, "LB: historical conditional all-clear lower bound. Ratios are ratios of means, not means of individual ratios.",
                ha="center", fontsize=9.1, color="#677489")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, facecolor="white", metadata={
        "Description": "Local synthetic paired results only; ratios of means; full optical-failure cost retained",
        "InputsSHA256": json.dumps({stage: item[1] for stage, item in inputs.items()}, sort_keys=True),
        "GeneratorSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    plt.close(figure)
    print(json.dumps({"output": str(args.output), "input_sha256": {stage: item[1] for stage, item in inputs.items()}}, indent=2))


if __name__ == "__main__":
    main()
