"""Frozen training comparison of axis, future relocation, immediate relocation."""

import argparse
import gzip
import json
from pathlib import Path
import statistics

from summarize_coverage_relocation import audit, paired_summary, percentile, station_execution_audit


def summarize(directory):
    rows = [json.loads(line) for line in (directory/"runs.jsonl").read_text().splitlines()]
    policies = ("axis_inferred", "relocating_cover", "executing_cover")
    indexed = {p:{r["seed"]:r for r in rows if r["policy"]==p} for p in policies}
    assert all(indexed[policies[0]].keys()==indexed[p].keys() for p in policies)
    diagnostics, audits = {}, []
    for policy, subset in indexed.items():
        oracle, changes, stations = [], [], []
        for seed in sorted(subset):
            report = json.load(gzip.open(directory/f"q3-random-{seed}--{policy}.json.gz", "rt"))["report"]
            audit(report)
            if policy == "axis_inferred":
                continue
            execution = station_execution_audit(report)
            audits.append({"policy":policy,"seed":seed,**execution})
            log = report["strategy_parameters"]["relocation_log"]
            oracle.append(sum(r["oracle_calls"] for r in log))
            changes.append(sum(r["relocated"] for r in log))
            stations.extend(execution["sites"])
            if policy == "executing_cover":
                for record, plan in zip(log,report["strategy_parameters"]["planning_log"]):
                    assert record["baseline_order"] == record["selected_order"]
                    assert record["baseline_selected_kind"] == plan["selected_kind"]
                    assert record["baseline_selected_channel"] == plan["selected_channel"]
                    if record["relocated"]:
                        actual=report["action_history"][record["after_actual_action_count"]]
                        assert actual["phase"]=="coverage" and actual["position"]==record["new_position"]
                assert all(s["planned_changes"]<=1 for s in execution["sites"])
                assert all(s["visited"] for s in execution["sites"] if s["planned_changes"])
        if oracle:
            diagnostics[policy]={"mean_oracle_calls":statistics.mean(oracle),"max_oracle_calls":max(oracle),
                                 "planned_changes":sum(changes),"max_changes_per_site":max(s["planned_changes"] for s in stations),
                                 "replanned_sites":sum(s["planned_changes"]>0 for s in stations),
                                 "replanned_sites_actually_visited":sum(s["planned_changes"]>0 and s["visited"] for s in stations)}
    result={"origin":"synthetic_research_training_only","platform":"Windows",
            "overall":{p:{"n":len(rs),"mean_s":statistics.mean(r["time_s"] for r in rs.values()),
                          "p95_s":percentile([r["time_s"] for r in rs.values()],.95),
                          "mean_wall_s":statistics.mean(r["wall_s"] for r in rs.values()),
                          "successes":sum(r["success"] for r in rs.values()),
                          "failed_clears":sum(r["failed_clears"] for r in rs.values()),
                          "breakdown":{k:statistics.mean(r["breakdown"][k] for r in rs.values()) for k in next(iter(rs.values()))["breakdown"]}}
                       for p,rs in indexed.items()},
            "pairs":{f"{a}--{b}":paired_summary(indexed[a],indexed[b]) for a,b in
                      (("axis_inferred","relocating_cover"),("axis_inferred","executing_cover"),("relocating_cover","executing_cover"))},
            "all_full_history_and_immediate_execution_audits_passed":True,
            "relocation_diagnostics":diagnostics,"station_audits":audits}
    (directory/"paired_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="station_audits"},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    summarize(parser.parse_args().directory)
