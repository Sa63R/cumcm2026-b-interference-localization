"""Measure finite-source posterior collapse under discrete versus binned data."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.benchmark_probe_tree import snapshots
from planning.binned_probe_tree import BinnedProbeTree
from planning.probe_tree import ProbeTree, _candidates, polygon_quadrature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted((ROOT / "results/state_search/tree_pilot").glob("*--radius_v1.json.gz")):
        for region, current, observed, bearing, channel in snapshots(path):
            points = _candidates(region, current, bearing, observed, 9)
            for support_count in (6, 24):
                support = polygon_quadrature(region, support_count)
                models = {
                    "discrete_noise1": ProbeTree(noise_nodes=1),
                    "discrete_noise3": ProbeTree(noise_nodes=3),
                    "bin0.5": BinnedProbeTree(bin_width_deg=.5),
                    "bin1.0": BinnedProbeTree(bin_width_deg=1.0)}
                for name, model in models.items():
                    for point in points:
                        branches = model._branches(region, point, support)
                        probability = sum(chance for chance, _, _ in branches)
                        if abs(probability - 1) > 1e-9:
                            raise AssertionError("Observation probabilities do not sum to one")
                        rows.append({"trace": path.name, "channel": channel,
                            "support_count": len(support), "model": name,
                            "position": [point.x, point.y], "branches": len(branches),
                            "singleton_probability": sum(chance for chance, _, posterior in branches if len(posterior) == 1),
                            "expected_ess": sum(chance / sum(s.weight**2 for s in posterior)
                                                for chance, _, posterior in branches)})
    summary = {}
    for support_count in (6, 24):
        for name in models:
            group = [r for r in rows if r["support_count"] == support_count and r["model"] == name]
            summary[f"{name}_support{support_count}"] = {
                "actions": len(group),
                "mean_singleton_probability": sum(r["singleton_probability"] for r in group) / len(group),
                "mean_expected_ess": sum(r["expected_ess"] for r in group) / len(group),
                "mean_branches": sum(r["branches"] for r in group) / len(group)}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
