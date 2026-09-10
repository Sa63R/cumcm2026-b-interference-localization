"""Full-run costs and observed-channel coverage audit for dynamic future sites."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from planning.disk_cover import disk_cover_radius
from simulator_client.state import Position
from summarize_region_travel import paired_summary, percentile
from summarize_inferred_region import audit


def station_execution_audit(report):
    """Track stable future-site identities; a planned move is not a visit."""
    log = report["strategy_parameters"]["relocation_log"]
    planning = report["strategy_parameters"]["planning_log"]
    assert len(log) == len(planning)
    if not log:
        return {"sites": [], "tasks_without_actual_action": 0}
    sites = [{"id": i, "position": p, "planned_changes": 0,
              "visited": False, "source_tasks_before_visit": 0}
             for i, p in enumerate(log[0]["remaining_before"])]
    remaining = list(sites)
    history = report["action_history"]
    no_action = 0
    for index, (record, plan) in enumerate(zip(log, planning)):
        assert [s["position"] for s in remaining] == record["remaining_before"]
        if record["relocated"]:
            changed = [i for i, (old, new) in enumerate(zip(record["remaining_before"],
                                                          record["remaining_after"])) if old != new]
            assert len(changed) == 1
            site = remaining[changed[0]]
            site["position"] = record["new_position"]
            site["planned_changes"] += 1
        begin = record["after_actual_action_count"]
        end = log[index+1]["after_actual_action_count"] if index+1 < len(log) else len(history)
        actual = history[begin:end]
        no_action += not actual
        if plan["selected_kind"] == "cover":
            site = next(s for s in remaining if s["position"] == plan["selected_point"])
            scans = [a for a in actual if a["phase"] == "coverage"]
            assert scans and all(a["position"] == site["position"] for a in scans)
            site["visited"] = True
            remaining.remove(site)
        else:
            assert not any(a["phase"] == "coverage" for a in actual)
            for site in remaining:
                site["source_tasks_before_visit"] += 1
    # A planned future site may be canceled only by the observed public cap.
    if remaining:
        assert len(report["detected_channels"]) == len(report["cleared_channels"]) == 16
    assert report["coverage_points_visited"] == 1+sum(s["visited"] for s in sites)
    return {"sites": sites, "tasks_without_actual_action": no_action,
            "canceled_by_observed_16_cap": len(remaining),
            "plan_count": len(log), "actual_future_visits": sum(s["visited"] for s in sites)}


def summarize(directory):
    rows = [json.loads(line) for line in (directory/"runs.jsonl").read_text().splitlines()]
    policies = ("axis_inferred", "relocating_cover")
    indexed = {p: {r["seed"]: r for r in rows if r["policy"] == p} for p in policies}
    assert indexed[policies[0]].keys() == indexed[policies[1]].keys()
    overall, cases, all_changes, calls, plans = {}, [], [], [], []
    for policy, subset in indexed.items():
        overall[policy] = {"n": len(subset), "mean_s": statistics.mean(r["time_s"] for r in subset.values()),
            "mean_wall_s": statistics.mean(r["wall_s"] for r in subset.values()),
            "p95_s": percentile([r["time_s"] for r in subset.values()], .95),
            "successes": sum(r["success"] for r in subset.values()),
            "failed_clears": sum(r["failed_clears"] for r in subset.values()),
            "time_breakdown": {k: statistics.mean(r["breakdown"][k] for r in subset.values())
                               for k in next(iter(subset.values()))["breakdown"]}}
    for seed in indexed[policies[0]]:
        for policy in policies:
            with gzip.open(directory/f"q3-random-{seed}--{policy}.json.gz", "rt", encoding="utf-8") as stream:
                report = json.load(stream)["report"]
            audit(report)
            if policy == policies[0]:
                continue
            history = report["action_history"]
            log = report["strategy_parameters"]["relocation_log"]
            for record in log:
                earlier = history[:record["after_actual_action_count"]]
                known = {a["channel"] for a in earlier if a["action"] == "measure" and a["result"] in ("direction", "near")}
                negatives = {(a["channel"], tuple(a["position"])) for a in earlier
                             if a["action"] == "measure" and a["result"] == "no_signal"}
                for channel in set(range(1,21))-known:
                    assert all((channel,tuple(p)) in negatives for p in record["executed_discovery_stations"])
                assert record["selected_proxy_s"] <= record["baseline_proxy_s"]+1e-6
                if record["relocated"]:
                    assert disk_cover_radius(record["executed_discovery_stations"]+record["remaining_after"]) <= 1000-1e-5
                    assert len(record["remaining_before"]) == len(record["remaining_after"])
                    all_changes.append({"seed": seed, "distance_m": Position(*record["old_position"]).distance_to(Position(*record["new_position"])),
                                        "proxy_gain_s": record["baseline_proxy_s"]-record["selected_proxy_s"],
                                        "coverage_radius_m": record["certified_cover_radius_m"]})
            calls.append(sum(r["oracle_calls"] for r in log))
            plans.extend(log)
            execution_audit = station_execution_audit(report)
            cases.append({"seed": seed, "saved_s": indexed[policies[0]][seed]["time_s"]-indexed[policies[1]][seed]["time_s"],
                          "relocations": sum(r["relocated"] for r in log),
                          "oracle_calls": calls[-1], "inferred_silences": len(report["strategy_parameters"]["inferred_no_signal_constraints"]),
                          "station_execution_audit": execution_audit,
                          "all_unknown_channel_coverage_and_inference_audits_passed": True})
    sites = [s for case in cases for s in case["station_execution_audit"]["sites"]]
    summary = {"origin": "synthetic_research_training_only", "platform": "Windows", "overall": overall,
               "pair": paired_summary(indexed[policies[0]], indexed[policies[1]]),
               "relocation_count": len(all_changes), "cases_with_relocation": sum(c["relocations"] > 0 for c in cases),
               "mean_relocation_distance_m": statistics.mean(c["distance_m"] for c in all_changes) if all_changes else 0.,
               "max_relocation_distance_m": max((c["distance_m"] for c in all_changes), default=0.),
               "mean_oracle_calls_per_case": statistics.mean(calls), "max_oracle_calls_per_case": max(calls),
               "oracle_budget_exhaustions": sum(p["oracle_budget_exhausted"] for p in plans),
               "mean_evaluated_variants_per_plan": statistics.mean(len(p["evaluated"]) for p in plans),
               "station_execution_summary": {
                   "initial_future_sites": len(sites),
                   "distinct_replanned_sites": sum(s["planned_changes"] > 0 for s in sites),
                   "sites_replanned_multiple_times": sum(s["planned_changes"] > 1 for s in sites),
                   "max_planned_changes_to_one_site": max(s["planned_changes"] for s in sites),
                   "actual_future_visits": sum(s["visited"] for s in sites),
                   "replanned_then_actually_visited": sum(s["visited"] and s["planned_changes"] > 0 for s in sites),
                   "canceled_by_observed_16_cap": sum(c["station_execution_audit"]["canceled_by_observed_16_cap"] for c in cases),
                   "tasks_without_actual_action": sum(c["station_execution_audit"]["tasks_without_actual_action"] for c in cases),
                   "max_source_tasks_before_future_visit_or_cap": max(s["source_tasks_before_visit"] for s in sites)},
               "all_full_history_geometry_audits_passed": True, "cases": cases, "relocations": all_changes,
               "proxy_caveat": "Successive plan gains overlap and are not additive realized savings or bounds on Q3."}
    (directory/"paired_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("cases", "relocations")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
