"""Read completed training histories, never scenario truth or new simulations."""

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

import numpy as np


def timeline(report):
    history = report["action_history"]
    discovered, seen, covers, clears, movement = [], set(), [], [], defaultdict(float)
    previous = (0., 0.)
    for index, action in enumerate(history):
        position = action["position"]
        movement[action["phase"]] += math.dist(previous, position)/5.
        previous = position
        if action["action"] == "measure" and action["result"] in ("near", "direction") and action["channel"] not in seen:
            seen.add(action["channel"])
            discovered.append({"channel": action["channel"], "time_s": action["virtual_time_s"],
                               "action_index": index, "phase": action["phase"]})
        if action["phase"] == "coverage" and (index == 0 or history[index-1]["phase"] != "coverage"
                                               or history[index-1]["position"] != position):
            covers.append({"position": position, "time_s": action["virtual_time_s"], "action_index": index})
        if action["action"] == "clear" and action["result"] == "success":
            clears.append({"channel": action["channel"], "time_s": action["virtual_time_s"], "position": position})
    return {"discovery": discovered, "coverage_begins": covers, "clears": clears,
            "movement_by_action_phase_s": dict(movement)}


def audit(directory):
    summary = json.loads((directory/"paired_summary.json").read_text())
    categories, cases, changes = defaultdict(list), [], []
    n = 40000
    index = np.arange(n)
    radius = 1800*np.sqrt((index+.5)/n)
    angle = index*math.pi*(3-math.sqrt(5))
    points = np.stack((radius*np.cos(angle), radius*np.sin(angle)), axis=1)
    for case in summary["cases"]:
        seed = case["seed"]
        reports = [json.load(gzip.open(directory/f"q3-random-{seed}--{policy}.json.gz", "rt"))["report"]
                   for policy in ("axis_inferred", "relocating_cover")]
        old, new = reports
        a, b = old["action_history"], new["action_history"]
        first = next(i for i, (x,y) in enumerate(zip(a,b)) if x != y)
        category = "cover_location" if a[first]["phase"] == b[first]["phase"] == "coverage" else "source_order_or_source_cover"
        ta, tb = map(timeline, reports)
        same_order = [c["channel"] for c in ta["clears"]] == [c["channel"] for c in tb["clears"]]
        categories[category].append(case["saved_s"])
        categories["same_clear_order" if same_order else "different_clear_order"].append(case["saved_s"])
        logs = new["strategy_parameters"]["relocation_log"]
        latest = max((r for r in logs if r["after_actual_action_count"] <= first), key=lambda r:r["after_actual_action_count"])
        if case["saved_s"] < 0:
            cases.append({"seed": seed, "saved_s": case["saved_s"], "first_difference_index": first,
                          "old_action": a[first], "new_action": b[first], "same_clear_order": same_order,
                          "category": category, "latest_plan_proxy_gain_s": latest["baseline_proxy_s"]-latest["selected_proxy_s"],
                          "old_timeline": ta, "new_timeline": tb,
                          "fee_delta_s": {k:new["time_breakdown"][k]-old["time_breakdown"][k] for k in old["time_breakdown"]}})
        for record in logs:
            if not record["relocated"]:
                continue
            old_site, new_site = np.array(record["old_position"]), np.array(record["new_position"])
            displacement = new_site-old_site
            length2 = float(np.dot(displacement, displacement))
            outside = np.ones(n, dtype=bool)
            for station in record["executed_discovery_stations"]:
                outside &= np.sum((points-np.array(station))**2, axis=1) > 1e6
            promised = outside & (np.sum((points-old_site)**2, axis=1) <= 1e6)
            lost = promised & (np.sum((points-new_site)**2, axis=1) > 1e6)
            relative = old_site-points[promised]
            if len(relative):
                qb = 2*np.dot(relative, displacement)
                qc = np.sum(relative**2, axis=1)-1e6
                alpha = min(1., float(np.min((-qb+np.sqrt(qb*qb-4*length2*qc))/(2*length2))))
            else:
                alpha = 1.
            changes.append({"seed":seed, "action_count":record["after_actual_action_count"],
                            "executed_stations":len(record["executed_discovery_stations"]),
                            "shift_m":math.sqrt(length2), "grid_prefix_preserving_shift_m":alpha*math.sqrt(length2),
                            "sample_old_guaranteed_region":int(np.sum(promised)), "sample_guarantee_lost":int(np.sum(lost)),
                            "proxy_gain_s":record["baseline_proxy_s"]-record["selected_proxy_s"]})
    result = {"origin":"read_only_completed_training_histories", "scenario_truth_read":False,
              "categories":{k:{"n":len(v), "wins":sum(x>0 for x in v), "losses":sum(x<0 for x in v),
                               "mean_saved_s":statistics.mean(v), "total_saved_s":sum(v)} for k,v in categories.items()},
              "negative_cases":sorted(cases,key=lambda c:c["saved_s"]),
              "prefix_diagnostic":{"sample_points":n, "sampling":"equal-area golden-angle disk points",
                                   "formal_coverage_certificate":False,
                                   "changes":len(changes), "erode_sampled_min_radius_promise":sum(c["sample_guarantee_lost"]>0 for c in changes),
                                   "original_total_shift_m":sum(c["shift_m"] for c in changes),
                                   "sample_prefix_constrained_total_shift_m":sum(c["grid_prefix_preserving_shift_m"] for c in changes),
                                   "details":changes},
              "scope":"First-divergence and discovery timelines are descriptive, not isolated causal attribution. Sampling is only a diagnostic against implementing a restrictive prefix condition; never used to certify actual strategy coverage."}
    (directory/"tail_audit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"categories":result["categories"], "negative_cases":len(cases),
                      "prefix_diagnostic":{k:v for k,v in result["prefix_diagnostic"].items() if k!="details"}},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    audit(parser.parse_args().directory)
