"""Paired runtime effects of logically redundant known-channel measurements."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics

from summarize_region_travel import paired_summary,percentile


def summarize(directory):
    rows=[json.loads(line) for line in (directory/"runs.jsonl").read_text().splitlines()]
    policies=("axis_v1","inferred_silence")
    indexed={p:{r["seed"]:r for r in rows if r["policy"]==p} for p in policies}
    overall={}
    cases=[]
    methods=Counter()
    for policy,subset in indexed.items():
        overall[policy]={"n":len(subset),"mean_s":statistics.mean(r["time_s"] for r in subset.values()),
            "mean_wall_s":statistics.mean(r["wall_s"] for r in subset.values()),
            "p95_s":percentile([r["time_s"] for r in subset.values()],.95),
            "successes":sum(r["success"] for r in subset.values()),"failed_clears":sum(r["failed_clears"] for r in subset.values()),
            "time_breakdown":{k:statistics.mean(r["breakdown"][k] for r in subset.values())
                              for k in next(iter(subset.values()))["breakdown"]}}
    for seed in indexed["axis_v1"]:
        reports={}
        for policy in policies:
            with gzip.open(directory/f"q3-random-{seed}--{policy}.json.gz","rt",encoding="utf-8") as stream:
                reports[policy]=json.load(stream)["report"]
        inferred=reports["inferred_silence"]["strategy_parameters"]["inferred_no_signal_constraints"]
        methods.update(v["method"] for v in inferred)
        deleted={(tuple(v["position"]),v["channel"]) for v in inferred}
        def normalize(history,remove):
            return [{k:v for k,v in a.items() if k!="virtual_time_s"} for a in history
                    if not (remove and a["phase"]=="coverage" and (tuple(a["position"]),a["channel"]) in deleted)]
        old,new=(reports[p]["action_history"] for p in policies)
        cases.append({"seed":seed,"inferences":len(inferred),
                      "saved_s":indexed["axis_v1"][seed]["time_s"]-indexed["inferred_silence"][seed]["time_s"],
                      "filtered_actual_history_identical":normalize(old,True)==normalize(new,False),
                      "movement_change_s":indexed["inferred_silence"][seed]["breakdown"]["movement_s"]-indexed["axis_v1"][seed]["breakdown"]["movement_s"]})
    summary={"origin":"synthetic_research_training_only","platform":"Windows","overall":overall,
             "pair":paired_summary(indexed[policies[0]],indexed[policies[1]]),
             "inferences":sum(c["inferences"] for c in cases),"inference_methods":dict(methods),
             "filtered_actual_history_identical_cases":sum(c["filtered_actual_history_identical"] for c in cases),
             "max_absolute_movement_change_s":max(abs(c["movement_change_s"]) for c in cases),"cases":cases}
    (directory/"paired_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="cases"},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    summarize(parser.parse_args().directory)
