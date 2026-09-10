"""Training-only cost decomposition for the mandatory-discovery ablation."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics

from summarize_region_travel import paired_summary,percentile


def summarize(directory):
    rows=[json.loads(line) for line in (directory/"runs.jsonl").read_text().splitlines()]
    policies=("axis_v1","unknown_only")
    indexed={p:{r["seed"]:r for r in rows if r["policy"]==p} for p in policies}
    overall={}
    for policy,subset in indexed.items():
        phases,known_results=Counter(),Counter()
        for seed in subset:
            with gzip.open(directory/f"q3-random-{seed}--{policy}.json.gz","rt",encoding="utf-8") as stream:
                report=json.load(stream)["report"]
            known=set()
            for action in report["action_history"]:
                if action["action"]!="measure":
                    continue
                phases[action["phase"]]+=1
                c=action["channel"]
                if action["phase"]=="coverage" and c in known:
                    known_results[action["result"]]+=1
                if action["result"] in ("direction","near"):
                    known.add(c)
        overall[policy]={"n":len(subset),"mean_s":statistics.mean(r["time_s"] for r in subset.values()),
            "mean_wall_s":statistics.mean(r["wall_s"] for r in subset.values()),
            "p95_s":percentile([r["time_s"] for r in subset.values()],.95),
            "successes":sum(r["success"] for r in subset.values()),"failed_clears":sum(r["failed_clears"] for r in subset.values()),
            "mean_measurements_by_phase":{k:v/len(subset) for k,v in phases.items()},
            "mean_known_coverage_readings_by_result":{k:v/len(subset) for k,v in known_results.items()},
            "time_breakdown":{k:statistics.mean(r["breakdown"][k] for r in subset.values())
                              for k in next(iter(subset.values()))["breakdown"]}}
    summary={"origin":"synthetic_research_training_only","platform":"Windows","overall":overall,
             "pair":paired_summary(indexed[policies[0]],indexed[policies[1]])}
    (directory/"paired_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    summarize(parser.parse_args().directory)
