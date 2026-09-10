"""Summarize one explicitly named training run, including surrogate mismatch."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random
import statistics


def summarize(directory):
    rows=[json.loads(line) for line in (directory/"runs.jsonl").read_text().splitlines()]
    indexed={policy:{r["seed"]:r for r in rows if r["policy"]==policy}
             for policy in ("axis_v1","recourse","recourse_no_scan_credit")}
    paired=[]
    decisions=[]
    statuses=Counter()
    equal_histories=0
    for seed,baseline in indexed["axis_v1"].items():
        traces={}
        for policy in ("recourse","recourse_no_scan_credit"):
            with gzip.open(directory/f"q3-random-{seed}--{policy}.json.gz","rt",encoding="utf-8") as stream:
                traces[policy]=json.load(stream)["report"]
        equal_histories+=traces["recourse"]["action_history"]==traces["recourse_no_scan_credit"]["action_history"]
        logs=traces["recourse"]["strategy_parameters"]["recourse_log"]
        statuses.update(log["status"] for log in logs)
        selected=[log for log in logs if log["status"]=="selected"]
        changes=[]
        for log in selected:
            alternatives=log["alternatives"]
            old=alternatives[0]
            best=min(alternatives,key=lambda a:a["score_s"])
            # This proxy prediction is within a frozen information state;
            # case savings below are a separate empirical outcome.
            item={"seed":seed,"changed":log["changed_first_task"],
                  "predicted_surrogate_gain_s":old["score_s"]-best["score_s"],
                  "known_route_cost_increase_s":best["score_s"]-best["correction_s"]-old["score_s"]+old["correction_s"],
                  "expected_unknown":log["expected_unknown"],
                  "from_kind":log["baseline_first_kind"],"to_kind":log["selected_kind"],
                  "runtime_s":log["runtime_s"]}
            decisions.append(item)
            if item["changed"]:
                changes.append(item)
        candidate=indexed["recourse"][seed]
        paired.append({"seed":seed,"baseline_s":baseline["time_s"],"candidate_s":candidate["time_s"],
                       "saved_s":baseline["time_s"]-candidate["time_s"],
                       "wall_increase_s":candidate["wall_s"]-baseline["wall_s"],
                       "changed_first_tasks":len(changes),
                       "sum_predicted_surrogate_gains_s":sum(d["predicted_surrogate_gain_s"] for d in changes)})
    saved=[r["saved_s"] for r in paired]
    rng=random.Random(911102001)
    draws=sorted(statistics.mean(rng.choices(saved,k=len(saved))) for _ in range(10000))
    changed=[d for d in decisions if d["changed"]]
    result={"n":len(saved),"origin":"synthetic_research_training_only",
            "mean_baseline_s":statistics.mean(r["baseline_s"] for r in paired),
            "mean_candidate_s":statistics.mean(r["candidate_s"] for r in paired),
            "mean_saved_s":statistics.mean(saved),"paired_bootstrap_mean_saved_95ci_s":[draws[249],draws[9749]],
            "wins":sum(v>1e-6 for v in saved),"losses":sum(v< -1e-6 for v in saved),
            "ties":sum(abs(v)<=1e-6 for v in saved),"max_regression_s":max(-v for v in saved),
            "mean_wall_increase_s":statistics.mean(r["wall_increase_s"] for r in paired),
            "identical_credit_ablation_histories":equal_histories,
            "statuses":dict(statuses),"selected_decisions":len(decisions),"changed_decisions":len(changed),
            "changed_kind_pairs":dict(Counter(d["from_kind"]+"->"+d["to_kind"] for d in changed)),
            "mean_changed_predicted_gain_s":statistics.mean(d["predicted_surrogate_gain_s"] for d in changed),
            "mean_changed_known_route_increase_s":statistics.mean(d["known_route_cost_increase_s"] for d in changed),
            "mean_recourse_planning_s":sum(d["runtime_s"] for d in decisions)/len(saved),
            "time_breakdown":{policy:{key:statistics.mean(r["breakdown"][key] for r in values.values())
                                      for key in next(iter(values.values()))["breakdown"]}
                              for policy,values in indexed.items()},
            "stop_decision":"No meaningful pilot gain; do not spend confirmation set on this additive recourse candidate."}
    (directory/"paired_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    (directory/"paired_cases.json").write_text(json.dumps(paired,indent=2)+"\n",encoding="utf-8")
    (directory/"surrogate_decisions.json").write_text(json.dumps(decisions,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    summarize(parser.parse_args().directory)
