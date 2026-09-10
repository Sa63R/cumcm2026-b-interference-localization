"""Post-experiment scientific figure; fine grid is visualization, not policy."""

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.benchmark_probe_tree import snapshots
from planning.probe_candidates import geometry_candidates, long_axis
from planning.radius_probe import choose_radius_probe
from simulator_client.state import Position


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = ROOT / "results/state_search"
    # Select the first archived pilot state with a nontrivial (>1 s) score
    # change, only after all experiments. This is explanatory, not selection.
    selected = None
    for path in sorted((root / "refinement_pilot").glob("*--pruned_v1.json.gz")):
        for region, current, observed, bearing, channel in snapshots(path):
            old, before = choose_radius_probe(region, current, bearing, observed)
            extra, _ = geometry_candidates(region, mode="axis_quantile", old_best=old)
            best, after = choose_radius_probe(region, current, bearing, observed, extra_points=extra)
            if before["score_s"] - after["score_s"] > 1:
                selected = (path.name, channel, region, current, observed, bearing, old, best, before, after, extra)
                break
        if selected:
            break
    if selected is None:
        raise ValueError("No declared diagnostic state found")
    path, channel, region, current, observed, bearing, old, best, before, after, extra = selected
    center, axis, perp, low, high = long_axis(region)
    theta = math.radians(bearing)
    lateral = (-math.sin(theta), math.cos(theta))
    legacy = [center]
    for fraction in (.5, 1.0):
        anchor = Position(current.x + fraction * (center.x - current.x),
                          current.y + fraction * (center.y - current.y))
        for offset in (-150., -50., 50., 150.):
            legacy.append(Position(anchor.x + offset * lateral[0], anchor.y + offset * lateral[1]))
    vertices = region.vertices
    support = [vertices[i * len(vertices) // min(4, len(vertices))] for i in range(min(4, len(vertices)))]
    hypotheses = [center] + [Position(.75*x+.25*center.x, .75*y+.25*center.y) for x,y in support]

    def score(point):
        if any(point.distance_to(Position(*v)) > 1000 for v in vertices):
            return math.nan
        total = 0.
        for source in hypotheses:
            d = point.distance_to(source)
            if d <= 5:
                total += d/5
                continue
            angle = math.degrees(math.atan2(source.y-point.y, source.x-point.x)) % 360
            copy = region.copy().observe(point, angle)
            if not copy.vertices:
                return math.nan
            radius = copy.enclosing_disk().radius
            total += d/5 + (6. if radius > 19.9 else 0.) + .5*max(0.,radius-19.9)/5
        return current.distance_to(point)/5 + total/len(hypotheses)

    assert abs(score(old) - before["score_s"]) < 1e-9
    assert abs(score(best) - after["score_s"]) < 1e-9

    def project(p):
        return ((p.x-center.x)*axis[0]+(p.y-center.y)*axis[1],
                (p.x-center.x)*perp[0]+(p.y-center.y)*perp[1])

    cloud = [project(p) for p in legacy + extra]
    xs = np.linspace(min(p[0] for p in cloud)-25, max(p[0] for p in cloud)+25, 45)
    ys = np.linspace(min(p[1] for p in cloud)-25, max(p[1] for p in cloud)+25, 35)
    costs = np.array([[score(Position(center.x+x*axis[0]+y*perp[0],
                                      center.y+x*axis[1]+y*perp[1]))-before["score_s"]
                       for x in xs] for y in ys])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), constrained_layout=True)
    mesh = axes[0].pcolormesh(xs, ys, np.ma.masked_invalid(costs), cmap="viridis", shading="auto")
    fig.colorbar(mesh, ax=axes[0], label="Surrogate score minus old best (s)")
    for points, marker, color, label in ((legacy, "o", "white", "Original candidates"),
                                         (extra, "s", "#f97316", "Added axis candidates")):
        legal = [project(p) for p in points if math.isfinite(score(p))]
        axes[0].scatter([p[0] for p in legal], [p[1] for p in legal], marker=marker,
                         s=30, facecolors="none", edgecolors=color, linewidths=1.2, label=label)
    axes[0].scatter(*project(old), marker="x", s=90, color="white", label="Old best")
    axes[0].scatter(*project(best), marker="*", s=140, color="#ef4444", label="Expanded best")
    axes[0].set_xlabel("Along region long axis (m)")
    axes[0].set_ylabel("Across region long axis (m)")
    axes[0].set_title("A. Fixed finite surrogate, richer legal actions")
    legend = axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2, fontsize=7)
    # White markers need a dark outline in the legend's white background.
    legend.legend_handles[0].set_edgecolor("#334155")
    legend.legend_handles[2].set_edgecolor("#334155")
    legend.legend_handles[2].set_facecolor("#334155")
    rows = [json.loads(line) for line in (root / "refinement_training/runs.jsonl").read_text().splitlines()]
    base = {r["seed"]:r["time_s"] for r in rows if r["policy"]=="pruned_v1"}
    values = {mode:[base[r["seed"]]-r["time_s"] for r in rows if r["policy"]==mode]
              for mode in ("axis_quantile", "local_refine")}
    bins = np.linspace(min(min(v) for v in values.values())-5, max(max(v) for v in values.values())+5, 25)
    for mode, color in (("axis_quantile", "#0f766e"), ("local_refine", "#2563eb")):
        axes[1].hist(values[mode], bins=bins, alpha=.5, color=color, label=mode)
    axes[1].axvline(0, color="#334155", lw=1)
    axes[1].set_xlabel("Paired virtual time saved vs v1 (s)")
    axes[1].set_ylabel("Training cases")
    axes[1].set_title("B. All 64 new training cases, including regressions")
    axes[1].legend(fontsize=8)
    axes[1].spines[["top", "right"]].set_visible(False)
    output = root / "refinement_summary"
    fig.savefig(output / "refinement_mechanism.png", dpi=180)
    fig.savefig(output / "refinement_mechanism.pdf")
    plt.close(fig)
    (output / "figure_state.json").write_text(json.dumps({"trace":path,"channel":channel,
        "old_score_s":before["score_s"],"expanded_score_s":after["score_s"],
        "old_point":[old.x,old.y],"expanded_point":[best.x,best.y],
        "region_vertices":vertices,"grid_role":"visualization_only_after_experiments"},indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
