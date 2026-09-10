"""Compare exact v1 scores and work on 16 archived legal observation states."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.benchmark_probe_tree import snapshots
from planning.radius_probe import choose_radius_probe
from strategies.state_search import StateSearch


class CountedRegion:
    def __init__(self, region, counter):
        self.region, self.counter = region, counter

    @property
    def vertices(self):
        return self.region.vertices

    def enclosing_disk(self):
        return self.region.enclosing_disk()

    def copy(self):
        return CountedRegion(self.region.copy(), self.counter)

    def observe(self, *args):
        self.counter[0] += 1
        self.region.observe(*args)
        return self


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/state_search/tree_pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input.glob("*--radius_v1.json.gz")):
        for region, current, observed, bearing, channel in snapshots(path):
            scores, points = [], []
            for method in ("full", "pruned"):
                times, counter = [], [0]
                for repeat in range(3):
                    counted = CountedRegion(region, counter)
                    began = time.perf_counter()
                    if method == "full":
                        search = object.__new__(StateSearch)
                        search.state_config = SimpleNamespace(active_probe_search=True,
                            probe_model="radius_proxy", probe_uncertainty_weight=.5)
                        search.regions = {channel: counted}
                        search.client = SimpleNamespace(state=SimpleNamespace(position=current))
                        search.first_bearings = {channel: bearing}
                        search.observed_positions = {channel: observed}
                        search.probe_log = []
                        point = search._next_probe(channel, 0)
                        log = search.probe_log[-1]
                    else:
                        point, log = choose_radius_probe(counted, current, bearing, observed, .5)
                    times.append(time.perf_counter() - began)
                scores.append(log["score_s"])
                points.append(point)
                rows.append({"trace": path.name, "channel": channel, "method": method,
                    "runtime_s": statistics.mean(times), "geometry_updates": counter[0] / 3,
                    "position": [point.x, point.y], "score_s": log["score_s"]})
            if scores[0] != scores[1] or points[0] != points[1]:
                raise AssertionError("Pruning changed v1's exact floating score or action")
    summary = {}
    for method in ("full", "pruned"):
        selected = [r for r in rows if r["method"] == method]
        summary[method] = {"n": len(selected),
            "mean_runtime_s": statistics.mean(r["runtime_s"] for r in selected),
            "total_geometry_updates": sum(r["geometry_updates"] for r in selected),
            "all_scores_and_actions_exactly_equal": True}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
