"""Audit only previously opened public histories; no model/simulator/DB/network.

Run with the project src on PYTHONPATH. Evaluator ground_truth is never indexed.
Input hashes, exact phase ledgers and per-probe public geometry are retained.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from localization.omni import OmniCandidateRegion
from research_rl.controller import DeepRLSearch
from simulator_client.state import Position

DEST = Path(__file__).resolve().parent
BATCH = ROOT / "results/paired_rl_state_20260911"
POLICIES = ("rl_trial1", "state_certified_tail", "state_v1")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_replay(actions):
    state = SimpleNamespace(sources={}, position=Position(0, 0), current_channel=1)
    controller = DeepRLSearch(SimpleNamespace(state=state), lambda f, c, t: t)
    records = []
    for index, item in enumerate(actions):
        channel, action = item["channel"], item["action"]
        point = Position(*item["position"])
        before = None
        if item.get("phase") == "rl_active_localization":
            region = controller.regions[channel]
            choices = controller._source_candidates(channel)
            matches = [c for c in choices if c.point.distance_to(point) <= 1e-6]
            assert matches, f"Public candidate reconstruction mismatch at action {index}"
            disk = region.enclosing_disk()
            maximum = max(math.hypot(x-point.x, y-point.y) for x, y in region.vertices)
            center=Position(*disk.center)
            angle=math.radians(controller.first_bearings[channel])
            perp=(-math.sin(angle),math.cos(angle))
            anchor=Position((state.position.x+center.x)/2,(state.position.y+center.y)/2)
            new_points=[Position(anchor.x+offset*perp[0],anchor.y+offset*perp[1])
                        for offset in (-150.,-50.,50.,150.)]
            seen=set(controller.observed_positions.get(channel,()))|{(round(c.point.x,6),round(c.point.y,6)) for c in choices}
            new_legal=[p for p in new_points if (round(p.x,6),round(p.y,6)) not in seen
                       and max(math.hypot(x-p.x,y-p.y) for x,y in region.vertices)<=1000-1e-7]
            safe_base=sum(max(math.hypot(x-c.point.x,y-c.point.y) for x,y in region.vertices)<=1000-1e-7
                          for c in choices if c.kind=='probe')
            before = dict(action_index=index, channel=channel, option=matches[0].option,
                          position=item["position"], result=item["result"],
                          radius_before=disk.radius, area_before=region.area,
                          maximum_vertex_distance_m=maximum,
                          guaranteed_reception=maximum <= 1000-1e-7,
                          center_distance_m=math.dist(disk.center, item["position"]),
                          move_s=state.position.distance_to(point)/5,
                          previous_actual_measurement_count=len(controller.observed_positions.get(channel, ())),
                          rl_probes_before=controller.probe_counts.get(channel, 0),
                          existing_guaranteed_probe_choices=safe_base,
                          proposed_midcross_novel_legal_count=len(new_legal),
                          same_current_position=state.position.distance_to(point) <= 1e-6)
        if action == "measure":
            region = controller.regions.setdefault(channel, OmniCandidateRegion())
            controller.observed_positions.setdefault(channel, set()).add((round(point.x,6),round(point.y,6)))
            if item["result"] == "direction":
                controller.detected.add(channel)
                controller.first_bearings.setdefault(channel, item["bearing_deg"])
                region.observe(point, item["bearing_deg"])
            elif item["result"] == "near":
                controller.detected.add(channel)
                controller.near_points[channel] = point
            else:
                region.observe_no_signal(point)
            state.current_channel = channel
            if before is not None:
                controller.probe_counts[channel] = controller.probe_counts.get(channel,0)+1
                before.update(area_after=region.area, radius_after=region.enclosing_disk().radius)
                before["area_ratio"] = before["area_after"]/before["area_before"] if before["area_before"] else None
                records.append(before)
        elif item.get("result") == "success":
            controller.cleared.add(channel)
        state.position = point
    return records


def main():
    bounds = {(r["policy"],int(r["seed"])):float(r["primary_lower_bound_s"])
              for r in csv.DictReader((BATCH/"per_case.csv").open(encoding="utf-8-sig"))}
    result = dict(scope="Previously opened 2100001..2100048 development trajectories only",
                  provenance={"comparison_sha256":sha(BATCH/"comparison.json"),
                              "lower_bound_table_sha256":sha(BATCH/"per_case.csv")},
                  restrictions={"training":False,"policy_execution":False,"new_simulator_actions":False,
                                "database_access":False,"ground_truth_access":False,
                                "geometry_source":"Only preceding actual public action/feedback entries",
                                "lower_bound":"Existing post-hoc L/5+5N, not an online-achievable optimum"},
                  methods={}, paired=[])
    all_probes=[]
    for policy in POLICIES:
        cases=[]
        for path in sorted((BATCH/policy).glob("case-*.json.gz")):
            data=json.loads(gzip.decompress(path.read_bytes()))
            row=data["row"]; actions=data["summary"]["action_history"]
            counts=Counter(); outcomes=Counter(); phase_cost=defaultdict(Counter)
            seen=set(); known=set(); cleared=set(); repeated=0; after_clear=0
            position=(0,0);channel_now=1; previous_t=0.; active_by_source=Counter()
            for item in actions:
                phase=item.get("phase","unknown"); point=item["position"]
                movement=round(math.dist(position,point)/5*1e6)/1e6
                phase_cost[phase]["movement_s"]+=movement
                costs=movement
                if item["action"]=="measure":
                    counts[phase]+=1
                    outcomes["|".join((phase,"known" if item["channel"] in known else "unknown",item["result"]))]+=1
                    switching=int(channel_now!=item["channel"])
                    costs+=5+switching
                    phase_cost[phase]["detection_s"]+=5
                    phase_cost[phase]["switching_s"]+=switching
                    channel_now=item["channel"]
                    key=(channel_now,tuple(round(float(x),6) for x in point))
                    repeated+=key in seen;seen.add(key);after_clear+=channel_now in cleared
                    if item["result"] in ("direction","near"):known.add(channel_now)
                    if phase!="coverage":active_by_source[channel_now]+=1
                else:
                    costs+=3+2*(item["result"]=="success")
                    phase_cost[phase]["optical_s"]+=3
                    phase_cost[phase]["removal_s"]+=2*(item["result"]=="success")
                    if item["result"]=="success":cleared.add(item["channel"])
                assert math.isclose(previous_t+costs,item["virtual_time_s"],abs_tol=2e-6)
                previous_t=item["virtual_time_s"];position=point
            assert sum(counts.values())==row["measurement_count"]
            total=Counter()
            for ledger in phase_cost.values():total.update(ledger)
            for k,v in total.items():assert math.isclose(v,row[k],abs_tol=2e-6)
            probes=geometry_replay(actions) if policy=="rl_trial1" else []
            for p in probes:p["seed"]=row["seed"]
            all_probes.extend(probes)
            learning=data["summary"].get("learning",{})
            cases.append(dict(seed=row["seed"],input_sha256=sha(path),time_s=row["virtual_time_s"],
                lower_bound_s=bounds[(policy,row["seed"])],time_over_lower=row["virtual_time_s"]/bounds[(policy,row["seed"])],
                measurement_count=row["measurement_count"],phase_measurements=dict(counts),outcomes=dict(outcomes),
                phase_costs={k:dict(v) for k,v in phase_cost.items()},
                repeated_same_channel_position=repeated,measurements_after_clear=after_clear,
                discovered_sources=len(known),sources_with_no_noncoverage_measurement=len(known-set(active_by_source)),
                noncoverage_measurements_per_source=dict(Counter(active_by_source.values())),
                fallback_counts=learning.get("fallback_counts",{}),fallback_actions=learning.get("fallback_actions",0),
                fallback_time_s=learning.get("fallback_virtual_time_s",0),
                raw_interrupted_scans_counter=learning.get("interrupted_scans",0),
                success=row["successful"],failed_clear_count=row["failed_clear_count"]))
        phase=Counter();outcomes=Counter();fallback=Counter();costs=defaultdict(Counter)
        for c in cases:
            phase.update(c["phase_measurements"]);outcomes.update(c["outcomes"]);fallback.update(c["fallback_counts"])
            for k,v in c["phase_costs"].items():costs[k].update(v)
        result["methods"][policy]=dict(cases=cases,phase_totals=dict(phase),phase_means={k:v/48 for k,v in phase.items()},
            outcome_totals=dict(outcomes),phase_mean_costs={k:{c:v/48 for c,v in val.items()} for k,val in costs.items()},
            total_measurements=sum(phase.values()),mean_measurements=sum(phase.values())/48,
            fallback_counts=dict(fallback),cases_with_fallback=sum(bool(c["fallback_counts"]) for c in cases),
            mean_fallback_time_s=statistics.mean(c["fallback_time_s"] for c in cases),
            sources_with_no_noncoverage_measurement=sum(c["sources_with_no_noncoverage_measurement"] for c in cases),
            repeated_same_channel_position=sum(c["repeated_same_channel_position"] for c in cases),
            measurements_after_clear=sum(c["measurements_after_clear"] for c in cases),
            mean_time_s=statistics.mean(c["time_s"] for c in cases),
            mean_lower_bound_s=statistics.mean(c["lower_bound_s"] for c in cases),
            total_time_over_total_lower=sum(c["time_s"] for c in cases)/sum(c["lower_bound_s"] for c in cases))
    by={p:{c["seed"]:c for c in result["methods"][p]["cases"]} for p in POLICIES}
    for seed in sorted(by["rl_trial1"]):
        a,b=by["rl_trial1"][seed],by["state_certified_tail"][seed]
        result["paired"].append(dict(seed=seed,rl_minus_state_s=a["time_s"]-b["time_s"],
            rl_minus_state_measures=a["measurement_count"]-b["measurement_count"],
            rl_minus_state_coverage=a["phase_measurements"].get("coverage",0)-b["phase_measurements"].get("coverage",0),
            rl_minus_state_noncoverage=(a["measurement_count"]-a["phase_measurements"].get("coverage",0))-(b["measurement_count"]-b["phase_measurements"].get("coverage",0))))
    no_signal=[p for p in all_probes if p["result"]=="no_signal"]
    result["public_geometry"] = dict(rl_probe_count=len(all_probes),all_positions_match_legal_candidates=True,
        no_signal_count=len(no_signal),no_signal_guaranteed_reception=sum(p["guaranteed_reception"] for p in no_signal),
        no_signal_at_current_position=sum(p["same_current_position"] for p in no_signal),
        no_signal_with_existing_guaranteed_alternative=sum(p["existing_guaranteed_probe_choices"]>0 for p in no_signal),
        midcross_novel_legal_count=sum(p["proposed_midcross_novel_legal_count"] for p in all_probes),
        midcross_prefixes_with_novel_legal_choice=sum(p["proposed_midcross_novel_legal_count"]>0 for p in all_probes),
        no_signal_area_unchanged=sum(abs(p["area_after"]-p["area_before"])<=1e-6 for p in no_signal),
        no_signal_radius_unchanged=sum(abs(p["radius_after"]-p["radius_before"])<=1e-6 for p in no_signal),
        no_signal_mean_area_ratio=statistics.mean(p["area_ratio"] for p in no_signal),
        no_signal_mean_incoming_move_s=statistics.mean(p["move_s"] for p in no_signal),
        center_choice_count=sum(p["center_distance_m"]<=1e-6 for p in all_probes),
        probe_guaranteed_reception_count=sum(p["guaranteed_reception"] for p in all_probes),
        near_observations=sum(p["result"]=="near" for p in all_probes),
        caution="Negative feedback can shrink a feasible region; incoming travel may serve later tasks and is not wasted-cost proof.")
    (DEST/"public_probe_geometry.json").write_text(json.dumps(all_probes,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    result["provenance"]["analysis_sha256"]=sha(Path(__file__))
    (DEST/"diagnosis.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"public_geometry":result["public_geometry"],"phase_means":{p:result["methods"][p]["phase_means"] for p in POLICIES}},indent=2))


if __name__=="__main__":main()
