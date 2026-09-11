"""Plot the frozen Q4 coverage geometry; not an official performance figure.

Run from this worktree with Python and matplotlib installed:
    python experiments/plot_q4_cover.py
No simulator, scenario, heldout trajectory or fitted model is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from planning.coverage import coverage_points
from planning.q4_directional_cover import certified_cover_points


def route_length(points):
    previous, length = (0., 0.), 0.
    for point in points:
        position = (point.x, point.y)
        length += math.dist(previous, position)
        previous = position
    return length


def draw_panel(axis, points, title, color):
    xy = [(0., 0.)] + [(point.x / 1000, point.y / 1000) for point in points]
    xy = [point for i, point in enumerate(xy) if i == 0 or point != xy[i-1]]
    axis.add_patch(Circle((0, 0), 1.8, facecolor="#eef2f6", edgecolor="#758399",
                          linewidth=1.25, linestyle=(0, (4, 3)), zorder=0))
    axis.plot(*zip(*xy), color=color, linewidth=1.65, alpha=.91, zorder=2)
    for first, second in zip(xy, xy[1:]):
        center = tuple(.5 * (a+b) for a, b in zip(first, second))
        before = tuple(c-.07*(b-a) for c, a, b in zip(center, first, second))
        after = tuple(c+.07*(b-a) for c, a, b in zip(center, first, second))
        axis.annotate("", xy=after, xytext=before, zorder=3,
                      arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1,
                                  "mutation_scale": 8, "shrinkA": 0, "shrinkB": 0})
    axis.scatter([p.x / 1000 for p in points], [p.y / 1000 for p in points],
                 s=31, color=color, edgecolor="white", linewidth=.65, zorder=4)
    axis.scatter([xy[-1][0]], [xy[-1][1]], marker="s", s=72,
                 color="#d69329", edgecolor="white", linewidth=1.0, zorder=5)
    axis.scatter([0], [0], marker="*", s=170, color="#183044", edgecolor="white", linewidth=.8, zorder=6)
    axis.set(title=title, xlabel="x (km)", xlim=(-3.1, 3.1), ylim=(-3.1, 3.1),
             xticks=range(-3, 4), yticks=range(-3, 4))
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#dce2e9", linewidth=.6, alpha=.7)
    axis.set_axisbelow(True)
    axis.title.set_fontsize(12)
    axis.title.set_linespacing(1.55)
    for spine in axis.spines.values():
        spine.set_color("#bdc6d2")
    axis.tick_params(colors="#536276", labelsize=10)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=ROOT / "research/q4_cover_search/selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "research/q4_cover_search/coverage-route-comparison.png")
    args = parser.parse_args(argv)
    selection = json.loads(args.selection.read_bytes())
    if (selection.get("git_commit") != "eeb69fec0d0e0db4597f21fbe4abd594079b9f34"
            or selection.get("profile") != "compact_22"):
        raise ValueError("Expected the frozen compact_22 research selection")
    for relative, expected in selection["source_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen source changed: " + relative)
    old = coverage_points(4, variant="triangular")
    new, _ = certified_cover_points("compact_22")
    lengths = [route_length(old), route_length(new)]
    if len(old) != 31 or len(new) != 22:
        raise ValueError("Frozen station counts changed")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#26354b", "axes.labelcolor": "#445166"})
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 6.8), sharex=True, sharey=True)
    figure.subplots_adjust(left=.075, right=.975, bottom=.215, top=.80, wspace=.13)
    draw_panel(axes[0], old, f"(a) Original triangular cover  |  31 stations\n{lengths[0]/1000:.3f} km", "#3b6294")
    draw_panel(axes[1], new, f"(b) Frozen compact cover  |  22 stations\n{lengths[1]/1000:.3f} km", "#177e79")
    axes[0].set_ylabel("y (km)")
    figure.suptitle("Q4 coverage routes at the same spatial scale", fontsize=16, fontweight="bold", y=.965)
    figure.text(.5, .915, "Both start at the origin and finish at a free endpoint; source-region radius = 1.8 km",
                ha="center", fontsize=10.5, color="#56647a")
    handles = [Patch(facecolor="#eef2f6", edgecolor="#758399", linestyle="--", label="Source region"),
               Line2D([], [], marker="o", markersize=5, color="#3b6294", linestyle="none", label="Station"),
               Line2D([], [], marker="*", markersize=11, color="#183044", linestyle="none", label="Start (origin)"),
               Line2D([], [], marker="s", markersize=6, color="#d69329", linestyle="none", label="Route endpoint")]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .08), ncol=4,
                  frameon=False, fontsize=10, handlelength=1.4, columnspacing=2.1)
    figure.text(.5, .045, "Coverage geometry only. Lengths exclude sensing time, localization and clearance detours.",
                ha="center", color="#626d7e", fontsize=9.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, facecolor="white", metadata={
        "Description": "Frozen Q4 coverage geometry, not official performance results",
        "SourceCommit": selection["git_commit"],
        "GeneratorSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    plt.close(figure)
    print(json.dumps({"output": str(args.output), "source_commit": selection["git_commit"],
                      "old_stations": 31, "new_stations": 22, "old_route_m": lengths[0],
                      "new_route_m": lengths[1], "geometry_only": True}, indent=2))


if __name__ == "__main__":
    main()
