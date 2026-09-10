"""Audit skipped-command geometry on explicit, already observed training traces."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from localization.omni import OmniCandidateRegion
from planning.silence_certificate import certify_silence
from simulator_client.state import Position


def update(region,action):
    if action["result"]=="direction":
        region.observe(action["position"],action["bearing_deg"])
    elif action["result"]=="no_signal":
        region.observe_no_signal(action["position"])


def analyze(path):
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        history=json.load(stream)["report"]["action_history"]
    actual,inferred,omitted={},{},{}
    known=set()
    flags=[]
    mismatches=omitted_differences=future_positive_differences=0
    meaningful_omission_differences=0
    maximum_radius_difference=maximum_area_difference=0.
    kinds=Counter()
    current=retained_current=1
    old_service=new_service=0.
    affected=set()
    for index,a in enumerate(history):
        if a["action"]!="measure":
            continue
        c=a["channel"]
        for regions in (actual,inferred,omitted):
            regions.setdefault(c,OmniCandidateRegion())
        certificate=None
        if a["phase"]=="coverage" and c in known:
            kinds[a["result"]]+=1
            certificate=certify_silence(actual[c],a["position"])
        old_service+=5+int(current!=c)
        current=c
        if certificate:
            assert a["result"]=="no_signal"
            inferred[c].observe_no_signal(a["position"])
            affected.add(c)
            flags.append({"index":index,"channel":c,"position":a["position"],**certificate,
                          "inference_kind":"logical_no_signal_not_physical_measurement"})
        else:
            update(inferred[c],a)
            update(omitted[c],a)
            new_service+=5+int(retained_current!=c)
            retained_current=c
        update(actual[c],a)
        assert actual[c].no_signal_positions==inferred[c].no_signal_positions
        mismatches+=actual[c].vertices!=inferred[c].vertices
        if c in affected and omitted[c].vertices!=actual[c].vertices:
            omitted_differences+=1
            future_positive_differences+=a["result"]=="direction"
            radius_difference=abs(omitted[c].enclosing_disk().radius-actual[c].enclosing_disk().radius)
            area_difference=abs(omitted[c].area-actual[c].area)
            maximum_radius_difference=max(maximum_radius_difference,radius_difference)
            maximum_area_difference=max(maximum_area_difference,area_difference)
            meaningful_omission_differences+=radius_difference>1e-5 or area_difference>1e-4
        if a["result"] in ("direction","near"):
            known.add(c)
    return {"trace":path.name,"trace_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "known_coverage_feedbacks":dict(kinds),"inferred_events":flags,
            "inferred_geometry_mismatches":mismatches,
            "omitted_geometry_different_updates":omitted_differences,
            "omitted_geometry_different_future_positives":future_positive_differences,
            "omitted_geometry_meaningful_differences":meaningful_omission_differences,
            "max_omitted_radius_difference_m":maximum_radius_difference,
            "max_omitted_area_difference_m2":maximum_area_difference,
            "frozen_trace_detect_switch_saving_s":old_service-new_service}


def run(directory,output):
    files=sorted(directory.glob("*--axis_v1.json.gz"))
    if not files:
        raise ValueError("Explicit axis_v1 training traces required")
    output.mkdir(parents=True,exist_ok=True)
    rows=[analyze(path) for path in files]
    events=[e for row in rows for e in row["inferred_events"]]
    summary={"origin":"post_run_legal_observation_history_audit","cases":len(rows),
        "known_coverage_feedbacks":dict(sum((Counter(r["known_coverage_feedbacks"]) for r in rows),Counter())),
        "certified_events":len(events),"methods":dict(Counter(e["method"] for e in events)),
        "cases_with_certificate":sum(bool(r["inferred_events"]) for r in rows),
        "mean_detection_saving_ceiling_s":5*len(events)/len(rows),
        "mean_frozen_trace_detect_switch_saving_s":statistics.mean(r["frozen_trace_detect_switch_saving_s"] for r in rows),
        "inferred_geometry_mismatches":sum(r["inferred_geometry_mismatches"] for r in rows),
        "omitted_geometry_different_updates":sum(r["omitted_geometry_different_updates"] for r in rows),
        "omitted_geometry_different_future_positives":sum(r["omitted_geometry_different_future_positives"] for r in rows),
        "omitted_geometry_meaningful_differences":sum(r["omitted_geometry_meaningful_differences"] for r in rows),
        "max_omitted_radius_difference_m":max(r["max_omitted_radius_difference_m"] for r in rows),
        "max_omitted_area_difference_m2":max(r["max_omitted_area_difference_m2"] for r in rows),
        "scope":"Frozen action-position replay only; action ordering, noise, and future control may change after physical deletion."}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    (output/"events.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.directory,args.output)
