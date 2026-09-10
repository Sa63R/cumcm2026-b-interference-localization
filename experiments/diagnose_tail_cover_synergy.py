"""Hindsight replay certificates for coordinating scans at already visited clears.

Only completed training action histories are read. Removing only coverage
actions AFTER the final clear cannot invalidate earlier clear certificates.
Inserting unknown-channel scans at one/two earlier actual clear locations is
charged at an upper bound, including subsequent retuning. This diagnoses a
specific finite batch opportunity; its hindsight choice is not an online rule.
"""

import gzip
from itertools import combinations
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src")]
from planning.disk_cover import disk_cover_radius
from simulator_client.state import Position


def main():
    root=ROOT/"results/state_search/refinement_training"
    rows=[]
    for path in sorted(root.glob("*--axis_quantile.json.gz")):
        seed=int(path.name.split('--')[0].split('-')[-1])
        if not 100273<=seed<=100336:
            raise ValueError("Training scope only")
        with gzip.open(path,"rt",encoding="utf-8") as f:history=json.load(f)["report"]["action_history"]
        last=max(i for i,item in enumerate(history) if item["action"]=="clear")
        tail_s=history[-1]["virtual_time_s"]-history[last]["virtual_time_s"]
        if tail_s<=0:
            rows.append({"seed":seed,"tail_s":tail_s,"single_net_s":0,"pair_net_s":0})
            continue
        covered=[]; available=[]; seen=set()
        for item in history[:last+1]:
            point=Position(*item["position"])
            if item["action"]=="measure" and item["result"] in ("direction","near"):
                seen.add(item["channel"])
            if item["phase"]=="coverage" and point not in covered:
                covered.append(point)
            if item["action"]=="clear":
                available.append((point,6*(20-len(seen))+1,item["channel"]))
        best={1:None,2:None}
        for size in (1,2):
            for selected in combinations(available,size):
                extra_cost=sum(item[1] for item in selected)
                if extra_cost>=tail_s or best[size] and extra_cost>=best[size]["extra_cost_upper_s"]:
                    continue
                radius=disk_cover_radius(covered+[item[0] for item in selected])
                if radius<=1000-1e-5:
                    best[size]={"extra_cost_upper_s":extra_cost,"certified_radius_m":radius,
                        "scan_after_clear_channels":[item[2] for item in selected],
                        "scan_points":[[item[0].x,item[0].y] for item in selected]}
        rows.append({"seed":seed,"tail_s":tail_s,
            "single_net_s":tail_s-best[1]["extra_cost_upper_s"] if best[1] else 0,
            "pair_net_s":tail_s-best[2]["extra_cost_upper_s"] if best[2] else 0,
            "single":best[1],"pair":best[2]})
    summary={"n":len(rows),"positive_tails":sum(r["tail_s"]>0 for r in rows),
        "mean_tail_s":statistics.mean(r["tail_s"] for r in rows),
        "beneficial_single_cases":sum(r["single_net_s"]>0 for r in rows),
        "beneficial_pair_cases":sum(r["pair_net_s"]>0 for r in rows),
        "pair_only_cases":sum(r["pair_net_s"]>0 and r["single_net_s"]==0 for r in rows),
        "mean_best_certified_replay_saving_s":statistics.mean(max(r["single_net_s"],r["pair_net_s"]) for r in rows),
        "maximum_certified_replay_saving_s":max(max(r["single_net_s"],r["pair_net_s"]) for r in rows)}
    output=ROOT/"results/state_search/tail_synergy_diagnostic"
    output.mkdir(parents=True,exist_ok=False)
    (output/"rows.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
