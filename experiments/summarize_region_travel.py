"""Paired training summary for the scheduling-only region matrix ablation."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random
import statistics


def percentile(values,q):
    values=sorted(values)
    index=(len(values)-1)*q
    lo=int(index)
    return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(index-lo)


def paired_summary(first,second):
    seeds=sorted(first.keys()&second.keys())
    delta=[first[s]["time_s"]-second[s]["time_s"] for s in seeds]
    rng=random.Random(911104001)
    boot=sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(10000))
    return {"n":len(delta),"mean_saved_s":statistics.mean(delta),
            "relative_mean_saved":sum(delta)/sum(first[s]["time_s"] for s in seeds),
            "paired_bootstrap_95ci_saved_s":[boot[249],boot[9749]],
            "wins":sum(d>1e-6 for d in delta),"losses":sum(d< -1e-6 for d in delta),
            "ties":sum(abs(d)<=1e-6 for d in delta),"max_regression_s":max(-d for d in delta),
            "p95_ratio":percentile([second[s]["time_s"]/first[s]["time_s"] for s in seeds],.95)}


def summarize(directory):
    rows=[json.loads(line) for line in (directory/"runs.jsonl").read_text().splitlines()]
    policies=("axis_v1","mean_point","expected_distance")
    indexed={p:{r["seed"]:r for r in rows if r["policy"]==p} for p in policies}
    overall={}
    for policy,subset in indexed.items():
        overall[policy]={"n":len(subset),"mean_s":statistics.mean(r["time_s"] for r in subset.values()),
            "mean_wall_s":statistics.mean(r["wall_s"] for r in subset.values()),
            "p95_s":percentile([r["time_s"] for r in subset.values()],.95),
            "successes":sum(r["success"] for r in subset.values()),
            "failed_clears":sum(r["failed_clears"] for r in subset.values()),
            "time_breakdown":{k:statistics.mean(r["breakdown"][k] for r in subset.values())
                              for k in next(iter(subset.values()))["breakdown"]}}
    diagnosis={}
    for policy in policies[1:]:
        statuses=Counter()
        sources=[]
        plans=[]
        for seed in indexed[policy]:
            with gzip.open(directory/f"q3-random-{seed}--{policy}.json.gz","rt",encoding="utf-8") as stream:
                report=json.load(stream)["report"]
            for log in report["strategy_parameters"]["region_travel_log"]:
                sources.extend(log["sources"])
                statuses.update(s["status"] for s in log["sources"])
            plans.extend(report["strategy_parameters"]["planning_log"])
        uncertain=[s for s in sources if s["status"]=="finite_area_radius_proxy"]
        diagnosis[policy]={"source_task_observations":len(sources),"statuses":dict(statuses),
            "mean_uncertain_nodes":statistics.mean(s["nodes"] for s in uncertain),
            "mean_uncertain_mean_shift_from_mec_m":statistics.mean(s["mean_shift_from_mec_m"] for s in uncertain),
            "mean_uncertain_variance_m2":statistics.mean(s["variance_m2"] for s in uncertain),
            "plans":len(plans),"exact_plans":sum(p["exact"] for p in plans),
            "mean_model_gap_s":statistics.mean(p["model_gap_s"] for p in plans),
            "mean_planning_s_per_case":sum(p["runtime_s"] for p in plans)/len(indexed[policy])}
    summary={"origin":"synthetic_research_training_only","platform":"Windows",
             "overall":overall,"pairs":{f"{a}--{b}":paired_summary(indexed[a],indexed[b])
                        for a,b in (("axis_v1","mean_point"),("axis_v1","expected_distance"),("mean_point","expected_distance"))},
             "region_diagnostics":diagnosis}
    (directory/"paired_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    summarize(parser.parse_args().directory)
