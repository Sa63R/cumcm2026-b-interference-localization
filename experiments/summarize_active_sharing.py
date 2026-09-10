"""Training-only paired sharing outcome and added/avoided measurement counts."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics

from summarize_region_travel import paired_summary,percentile


def summarize(directory):
    rows=[json.loads(line) for line in (directory/"runs.jsonl").read_text().splitlines()]
    policies=("axis_v1","axis_active_sharing")
    indexed={p:{r["seed"]:r for r in rows if r["policy"]==p} for p in policies}
    overall={}
    decisions=[]
    stats=[]
    cases=[]
    for policy,subset in indexed.items():
        phases=Counter()
        for seed in subset:
            with gzip.open(directory/f"q3-random-{seed}--{policy}.json.gz","rt",encoding="utf-8") as stream:
                report=json.load(stream)["report"]
            phases.update(a["phase"] for a in report["action_history"] if a["action"]=="measure")
            if policy=="axis_active_sharing":
                sharing=report["strategy_parameters"]["active_sharing"]
                stats.append(sharing)
                decisions.extend(sharing["decisions"])
                cases.append({"seed":seed,"saved_s":indexed["axis_v1"][seed]["time_s"]-subset[seed]["time_s"],
                              "shared_measurements":sharing["shared_measurements"],
                              "actual_extra_measure_cost_s":sharing["actual_measure_cost_s"],
                              "sum_local_proxy_net_s":sum(d["estimated_net_gain_s"] for d in sharing["decisions"])})
        overall[policy]={"n":len(subset),"mean_s":statistics.mean(r["time_s"] for r in subset.values()),
            "mean_wall_s":statistics.mean(r["wall_s"] for r in subset.values()),
            "p95_s":percentile([r["time_s"] for r in subset.values()],.95),
            "successes":sum(r["success"] for r in subset.values()),"failed_clears":sum(r["failed_clears"] for r in subset.values()),
            "mean_measurements_by_phase":{k:v/len(subset) for k,v in phases.items()},
            "time_breakdown":{k:statistics.mean(r["breakdown"][k] for r in subset.values())
                              for k in next(iter(subset.values()))["breakdown"]}}
    summary={"origin":"synthetic_research_training_only","platform":"Windows","overall":overall,
             "pair":paired_summary(indexed[policies[0]],indexed[policies[1]]),
             "sharing_diagnostics":{"stops":sum(s["stops_considered"] for s in stats),
                 "candidates_scored":sum(s["candidates_scored"] for s in stats),"selected":len(decisions),
                 "mean_extra_measure_cost_s":statistics.mean(s["actual_measure_cost_s"] for s in stats),
                 "mean_local_proxy_net_s":statistics.mean(d["estimated_net_gain_s"] for d in decisions) if decisions else 0.,
                 "mean_scoring_wall_s":statistics.mean(s["planning_s"] for s in stats),
                 "all_receptions_positive":all(d["response"] in ("direction","near") for d in decisions)},
             "cases":cases}
    (directory/"paired_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="cases"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    summarize(parser.parse_args().directory)
