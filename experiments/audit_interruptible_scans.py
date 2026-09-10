"""Read legal histories only; audit atomic scanning and return-cost scales."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from localization.omni import OmniCandidateRegion
from planning.coverage import clearance_grid
from simulator_client.state import Position


def closed_grid_upper(region,q,bearing):
    grid=clearance_grid(region.vertices,bearing_deg=bearing,start=q)
    previous=q
    travel=0.
    for point in grid:
        travel+=previous.distance_to(point)
        previous=point
    travel+=previous.distance_to(q)
    # All optical attempts cost 3, successful removal costs another 2.
    # Stopping at the first success and returning is no longer than traversing
    # the complete remaining route then returning, by triangle inequality.
    return {"optical_grid_count":len(grid),"constructive_clear_return_upper_s":travel/5+3*len(grid)+2}


def analyze(path):
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        history=json.load(stream)["report"]["action_history"]
    cleared_labels={a["channel"] for a in history if a["action"]=="clear" and a["result"]=="success"}
    # The set is observable after exit. It is used only as an optimistic
    # accounting ceiling, never as an online channel-absence decision.
    groups=[]
    for i,a in enumerate(history):
        if a["phase"]=="coverage":
            if not groups or groups[-1][-1]!=i-1 or history[groups[-1][0]]["position"]!=a["position"]:
                groups.append([])
            groups[-1].append(i)
    by_start={indices[0]:indices for indices in groups}
    active_group=None
    known=set()
    regions={}
    previous=Position(0,0)
    previous_time=0.
    records=[]
    for index,a in enumerate(history):
        q=Position(*a["position"])
        if index in by_start:
            indices=by_start[index]
            active_group={"start_index":index,"position":a["position"],"measurements":len(indices),
                          "first_new_channel":None,"new_channels":[],"indices":indices,
                          "scan_service_s":history[indices[-1]]["virtual_time_s"]-previous_time-previous.distance_to(q)/5}
            records.append(active_group)
        channel=a["channel"]
        was_known=channel in known
        if a["action"]=="measure":
            region=regions.setdefault(channel,OmniCandidateRegion())
            if a["result"]=="direction":
                region.observe(q,a["bearing_deg"])
                known.add(channel)
            elif a["result"]=="near":
                known.add(channel)
            elif a["result"]=="no_signal":
                region.observe_no_signal(q)
            if a["phase"]=="coverage" and not was_known and channel in known:
                active_group["new_channels"].append(channel)
                if active_group["first_new_channel"] is None:
                    pending=[history[k]["channel"] for k in active_group["indices"] if k>index]
                    active_group.update({"first_new_channel":channel,"first_new_index":index,
                        "remaining_measurements_after_first_new":len(pending),
                        "remaining_service_after_first_new_s":history[active_group["indices"][-1]]["virtual_time_s"]-a["virtual_time_s"],
                        "pending_channels":pending,"already_known_pending":sorted(set(pending)&known),
                        "already_known_pending_scan_cost_ceiling_s":6*len(set(pending)&known),
                        "expost_existing_pending_scan_cost_ceiling_s":6*len(set(pending)&cleared_labels),
                        "clearing_first_new_deletes_current_pending":channel in pending})
                    if a["result"]=="near":
                        active_group.update({"successful_clear_return_leg_upper_s":5/5,
                                             "optical_grid_count":1,"constructive_clear_return_upper_s":5.})
                    else:
                        distance=max(q.distance_to(Position(*v)) for v in region.vertices)
                        active_group["successful_clear_return_leg_upper_s"]=(distance+20)/5
                        active_group.update(closed_grid_upper(region,q,a["bearing_deg"]))
        previous,previous_time=q,a["virtual_time_s"]
    for record in records:
        del record["indices"]
    return {"trace":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"groups":records}


def summarize(directory,output):
    paths=sorted(directory.glob("*--axis_v1.json.gz"))
    if not paths:
        raise ValueError("No explicit axis_v1 training histories")
    output.mkdir(parents=True,exist_ok=False)
    records=[analyze(path) for path in paths]
    groups=[g for record in records for g in record["groups"]]
    discoveries=[g for g in groups if g["first_new_channel"] is not None]
    result={"origin":"post_run_observation_history_audit","cases":len(paths),"scan_groups":len(groups),
        "groups_with_new_source":len(discoveries),
        "mean_scan_service_s_per_case":sum(g["scan_service_s"] for g in groups)/len(paths),
        "mean_scan_group_service_s":statistics.mean(g["scan_service_s"] for g in groups),
        "mean_scan_group_measurements":statistics.mean(g["measurements"] for g in groups),
        "mean_remaining_service_after_first_new_s":statistics.mean(g["remaining_service_after_first_new_s"] for g in discoveries),
        "mean_remaining_reads_after_first_new":statistics.mean(g["remaining_measurements_after_first_new"] for g in discoveries),
        "sum_remaining_service_after_first_new_per_case_s":sum(g["remaining_service_after_first_new_s"] for g in discoveries)/len(paths),
        "groups_where_newly_found_channel_still_pending":sum(g["clearing_first_new_deletes_current_pending"] for g in discoveries),
        "mean_already_known_pending_scan_cost_ceiling_s":statistics.mean(g["already_known_pending_scan_cost_ceiling_s"] for g in discoveries),
        "mean_expost_existing_pending_scan_cost_ceiling_s":statistics.mean(g["expost_existing_pending_scan_cost_ceiling_s"] for g in discoveries),
        "mean_successful_clear_return_leg_upper_s":statistics.mean(g["successful_clear_return_leg_upper_s"] for g in discoveries),
        "mean_constructive_clear_return_upper_s":statistics.mean(g["constructive_clear_return_upper_s"] for g in discoveries),
        "mean_optical_grid_count":statistics.mean(g["optical_grid_count"] for g in discoveries),
        "max_optical_grid_count":max(g["optical_grid_count"] for g in discoveries),
        "scope":"Ceilings/constructive bounds are conditional at a recorded observation; not predicted policy gains or original-task lower bounds."}
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    (output/"groups.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    summarize(args.directory,args.output)
