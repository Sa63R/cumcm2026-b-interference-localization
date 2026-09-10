"""Compare bounded and unpruned finite trees on replayed legal belief states."""

import argparse
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from localization.omni import OmniCandidateRegion
from planning.probe_tree import ProbeTree
from simulator_client.state import Position


def snapshots(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        report = json.load(stream)["report"]
    regions, observed, bearings = {}, {}, {}
    current, count = Position(0, 0), 0
    for item in report["action_history"]:
        channel, point = item["channel"], Position(*item["position"])
        if (item["phase"] == "active_localization" and channel in regions
                and regions[channel].vertices and regions[channel].enclosing_disk().radius > 50):
            yield regions[channel].copy(), current, set(observed[channel]), bearings[channel], channel
            count += 1
            if count == 2:
                return
        current = point
        if item["action"] != "measure":
            continue
        region = regions.setdefault(channel, OmniCandidateRegion())
        observed.setdefault(channel, set()).add((round(point.x, 6), round(point.y, 6)))
        if item["result"] == "direction":
            region.observe(point, item["bearing_deg"])
            bearings.setdefault(channel, item["bearing_deg"])
        elif item["result"] == "no_signal":
            region.observe_no_signal(point)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/state_search/tree_pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input.glob("*--radius_v1.json.gz")):
        for region, current, observed, bearing, channel in snapshots(path):
            for depth in (1, 2):
                costs = []
                for bounds in (False, True):
                    tree = ProbeTree(depth=depth, candidates=9, inner_candidates=3,
                                     supports=6, max_expansions=100000,
                                     first_bearing=bearing, use_bounds=bounds)
                    point, log = tree.choose(region, current, observed)
                    costs.append(log["finite_model_cost_s"])
                    rows.append({"trace": path.name, "channel": channel, "depth": depth,
                                 "use_bounds": bounds, "selected_point": [point.x, point.y], **log})
                if abs(costs[0] - costs[1]) > 1e-7:
                    raise AssertionError(f"Pruning changed exact tree value: {costs}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = {}
    for depth in (1, 2):
        for bounds in (False, True):
            group = [r for r in rows if r["depth"] == depth and r["use_bounds"] == bounds]
            summary[f"depth{depth}_bounds{bounds}"] = {
                "n": len(group), "mean_runtime_s": sum(r["runtime_s"] for r in group) / len(group),
                "expanded": sum(r["expanded"] for r in group),
                "bound_pruned": sum(r["bound_pruned"] for r in group),
                "branches": sum(r["posterior_branches"] for r in group),
                "singleton_branches": sum(r["singleton_posterior_branches"] for r in group),
                "all_exact": all(r["exact_for_declared_tree"] for r in group)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
