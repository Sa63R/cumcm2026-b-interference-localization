"""Post-termination accounting of existing training traces; no policy changes.

Only recorded legal positions/results are read. Exact clear-point reordering
is an optimistic fixed-endpoint reference, not an executable unknown-state
policy, a source-truth oracle, or a lower bound for all original Q3 policies.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "research/theory_v1")]
from audit_eval_bounds import exact_open_graph
from localization.omni import OmniCandidateRegion
from simulator_client.state import Position


def length(points, start=(0., 0.)):
    total = 0.
    for point in points:
        total += math.dist(start, point)
        start = point
    return total


def optimum(points, start=(0., 0.)):
    first = [math.dist(start, point) for point in points]
    edges = [[math.dist(a, b) for b in points] for a in points]
    return exact_open_graph(first, edges)


def analyze(path, exact=True):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        report = json.load(stream)["report"]
    history = report["action_history"]
    clears = [(i, item) for i, item in enumerate(history) if item["action"] == "clear" and item["result"] == "success"]
    clear_points = [item["position"] for _, item in clears]
    clear_channels = [item["channel"] for _, item in clears]
    all_positions = [item["position"] for item in history]
    movement_m = length(all_positions)
    events, previous_cover = [], None
    for item in history:
        point = item["position"]
        if item["phase"] == "coverage":
            if point != previous_cover:
                events.append(point)
                previous_cover = point
        else:
            previous_cover = None
        if item["action"] == "clear":
            events.append(point)
    event_m, clearance_m = length(events), length(clear_points)
    # The event route drops active probe positions; the clearance route also
    # drops cover visits. Both are subsequences, so triangle inequality gives
    # nonnegative differences, including any scans after the final clear.
    local_extra = movement_m - event_m
    cover_extra = event_m - clearance_m
    assert local_extra >= -1e-6 and cover_extra >= -1e-6
    assert abs(movement_m - (clearance_m + local_extra + cover_extra)) < 1e-5
    direct_by_phase, previous = {}, (0., 0.)
    detections, beginnings, active_counts = {}, {}, {}
    coverage_groups, last_cover = 0, None
    publicly_closed_index = None
    for index, item in enumerate(history):
        phase, point, channel = item["phase"], item["position"], item["channel"]
        direct_by_phase[phase] = direct_by_phase.get(phase, 0.) + math.dist(previous, point)/5
        previous = point
        if phase == "coverage":
            if point != last_cover:
                coverage_groups += 1
                last_cover = point
        else:
            last_cover = None
        if item["action"] == "measure" and item["result"] in ("direction", "near"):
            detections.setdefault(channel, index)
        if phase == "active_localization":
            beginnings.setdefault(channel, index)
            active_counts[channel] = active_counts.get(channel, 0) + 1
        if item["action"] == "clear":
            beginnings.setdefault(channel, index)
        # Seven complete coverage scans require the full final scan group,
        # not merely the first measurement at its station.
        end_group = index + 1 == len(history) or history[index+1]["phase"] != "coverage" or history[index+1]["position"] != point
        if publicly_closed_index is None and (len(detections) == 16 or (coverage_groups == 7 and end_group)):
            publicly_closed_index = index
    # One best reversal whose sources had all been detected before its first
    # resolution began. This checks only discovery availability, not that
    # future clearance coordinates had already been certified at that time.
    known_gain, unrestricted_gain = 0., 0.
    for i in range(len(clears)-1):
        before = (0., 0.) if not i else clear_points[i-1]
        for j in range(i+1, len(clears)):
            old = math.dist(before, clear_points[i])
            new = math.dist(before, clear_points[j])
            if j+1 < len(clears):
                old += math.dist(clear_points[j], clear_points[j+1])
                new += math.dist(clear_points[i], clear_points[j+1])
            saving = max(0., old-new)/5
            unrestricted_gain = max(unrestricted_gain, saving)
            if all(detections[c] <= beginnings[clear_channels[i]] for c in clear_channels[i:j+1]):
                known_gain = max(known_gain, saving)
    result = {"trace": path.name, "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "virtual_time_s": report["virtual_time_s"], "source_count_from_successful_clears":len(clears),
        "movement_s": movement_m/5, "clear_order_polyline_s": clearance_m/5,
        "cover_excursion_s":cover_extra/5, "local_probe_detour_s":local_extra/5,
        "movement_by_destination_phase_s":direct_by_phase,
        "measurements_by_phase": {phase:sum(item["action"]=="measure" and item["phase"]==phase for item in history)
                                   for phase in {item["phase"] for item in history}},
        "active_probes_per_source":[active_counts.get(c,0) for c in clear_channels],
        "best_known_at_start_clear_reversal_s":known_gain,
        "best_unrestricted_clear_reversal_s":unrestricted_gain,
        "planning_calls":len(report["strategy_parameters"]["planning_log"])}
    plans = report["strategy_parameters"]["planning_log"]
    result["planning_exact_count"] = sum(p["exact"] for p in plans)
    result["planning_max_model_gap_s"] = max(p["model_gap_s"] for p in plans)
    result["planning_mean_model_gap_s"] = statistics.mean(p["model_gap_s"] for p in plans)
    result["after_last_clear_scan_s"] = history[-1]["virtual_time_s"] - clears[-1][1]["virtual_time_s"]
    # Reconstruct the same legal geometric regions to classify discovery
    # work and the source-position approximation used by frozen task routing.
    regions, seen, removed, started_sources = {}, set(), set(), set()
    measured_positions = {}
    final_clear = {item["channel"]:item["position"] for _,item in clears}
    scan_kinds = {key:0 for key in ("absent_channel", "existing_not_yet_detected", "already_detected")}
    known_scan_unchanged = known_scan_certified = 0
    resolve_target_errors = []
    interruption_events = 0
    joint_probe_opportunities = []
    for item in history:
        channel, point = item["channel"], Position(*item["position"])
        region = regions.setdefault(channel, OmniCandidateRegion())
        if ((item["phase"]=="active_localization" or item["action"]=="clear")
                and channel not in started_sources and region.vertices):
            started_sources.add(channel)
            resolve_target_errors.append(math.dist(region.enclosing_disk().center, final_clear[channel]))
        if item["action"]=="clear":
            removed.add(channel)
            continue
        measured_positions.setdefault(channel,set()).add((round(point.x,6),round(point.y,6)))
        known_coverage = item["phase"]=="coverage" and channel in seen
        if item["phase"]=="coverage":
            label = ("absent_channel" if channel not in final_clear else
                     "already_detected" if channel in seen else "existing_not_yet_detected")
            scan_kinds[label]+=1
        old_vertices = tuple(region.vertices) if known_coverage else ()
        old_radius = region.enclosing_disk().radius if known_coverage and region.vertices else math.inf
        if item["result"]=="direction":
            region.observe(point,item["bearing_deg"])
            seen.add(channel)
        elif item["result"]=="no_signal":
            region.observe_no_signal(point)
        elif item["result"]=="near":
            seen.add(channel)
        if known_coverage:
            known_scan_unchanged += tuple(region.vertices)==old_vertices
            known_scan_certified += (old_radius>19.9 and region.vertices
                                     and region.enclosing_disk().radius<=19.9)
        if item["phase"]=="active_localization" and region.vertices:
            own_distance=point.distance_to(Position(*region.enclosing_disk().center))
            for other in seen-removed-{channel}:
                other_region=regions[other]
                if other_region.vertices:
                    disk=other_region.enclosing_disk()
                    distance=point.distance_to(Position(*disk.center))
                    if disk.radius<=19.9 and distance<min(own_distance,200):
                        interruption_events+=1
                        break
            for other in seen-removed-{channel}:
                other_region=regions[other]
                if (not other_region.vertices or
                    (round(point.x,6),round(point.y,6)) in measured_positions.get(other,set())):
                    continue
                disk=other_region.enclosing_disk()
                if disk.radius<60 or any(point.distance_to(Position(*v))>1000 for v in other_region.vertices):
                    continue
                angle=math.degrees(math.atan2(disk.center[1]-point.y,disk.center[0]-point.x))%360
                hypothetical=other_region.copy().observe(point,angle)
                if not hypothetical.vertices:
                    continue
                predicted=hypothetical.enclosing_disk().radius
                if predicted<=.5*disk.radius and disk.radius-predicted>=30:
                    joint_probe_opportunities.append({"active_channel":channel,"other_channel":other,
                        "before_radius_m":disk.radius,"nominal_after_radius_m":predicted,
                        "position":[point.x,point.y]})
    result.update({"coverage_scan_classes":scan_kinds,
        "known_coverage_scans_with_unchanged_vertices":known_scan_unchanged,
        "known_coverage_scans_obtaining_clear_certificate":known_scan_certified,
        "resolve_target_error_m":resolve_target_errors,
        "probe_events_with_closer_certified_other_source_within_200m":interruption_events,
        "joint_probe_opportunities":joint_probe_opportunities})
    if exact:
        began=time.perf_counter()
        best, order = optimum(clear_points)
        result.update({"fixed_clear_points_optimum_s":best/5,
            "clear_order_reordering_gap_s":(clearance_m-best)/5,
            "reference_order_channels":[clear_channels[i] for i in order],
            "dp_runtime_s":time.perf_counter()-began})
        assert result["clear_order_reordering_gap_s"] >= -1e-6
        for label, index in (("last_actual_detection",max(detections.values())),
                             ("publicly_closed_discovery",publicly_closed_index)):
            if index is None:
                result[label+"_suffix_gap_s"] = None
                continue
            tail = [item["position"] for i,item in clears if i>index]
            start = history[index]["position"]
            best_tail,_ = optimum(tail,start)
            result[label+"_suffix_gap_s"] = (length(tail,start)-best_tail)/5
            result[label+"_remaining_clears"] = len(tail)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"results/state_search/refinement_training")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for policy in ("pruned_v1","axis_quantile"):
        paths=sorted(args.input.glob(f"*--{policy}.json.gz"))
        if len(paths)!=64 or any(not 100273<=int(p.name.split('--')[0].split('-')[-1])<=100336 for p in paths):
            raise ValueError("Only the fixed 64 training traces are allowed")
        for index,path in enumerate(paths):
            rows.append({"policy":policy,**analyze(path,exact=policy=="axis_quantile")})
            if (index+1)%16==0:
                print(f"{policy}: {index+1}/64",flush=True)
    summary={}
    fields=("virtual_time_s","movement_s","clear_order_polyline_s","cover_excursion_s","local_probe_detour_s",
            "best_known_at_start_clear_reversal_s","best_unrestricted_clear_reversal_s","planning_max_model_gap_s",
            "planning_mean_model_gap_s","after_last_clear_scan_s","fixed_clear_points_optimum_s",
            "clear_order_reordering_gap_s","last_actual_detection_suffix_gap_s","publicly_closed_discovery_suffix_gap_s")
    for policy in ("pruned_v1","axis_quantile"):
        group=[r for r in rows if r["policy"]==policy]
        entry={key:statistics.mean(r[key] for r in group if r.get(key) is not None)
               for key in fields if any(r.get(key) is not None for r in group)}
        entry["exact_plan_fraction"]=sum(r["planning_exact_count"] for r in group)/sum(r["planning_calls"] for r in group)
        entry["mean_active_probes_per_source"]=statistics.mean(n for r in group for n in r["active_probes_per_source"])
        entry["sources_with_two_or_more_probes"]=sum(n>=2 for r in group for n in r["active_probes_per_source"])
        entry["total_sources"]=sum(len(r["active_probes_per_source"]) for r in group)
        entry["mean_measurements_by_phase"]={phase:statistics.mean(r["measurements_by_phase"].get(phase,0) for r in group)
            for phase in sorted(set().union(*(r["measurements_by_phase"] for r in group)))}
        entry["mean_coverage_scan_classes"]={key:statistics.mean(r["coverage_scan_classes"][key] for r in group)
                                            for key in group[0]["coverage_scan_classes"]}
        entry["mean_known_scans_with_unchanged_vertices"]=statistics.mean(r["known_coverage_scans_with_unchanged_vertices"] for r in group)
        entry["mean_known_scans_obtaining_clear_certificate"]=statistics.mean(r["known_coverage_scans_obtaining_clear_certificate"] for r in group)
        errors=[v for r in group for v in r["resolve_target_error_m"]]
        entry["mean_resolve_target_error_m"]=statistics.mean(errors)
        entry["resolve_targets_shifted_over_100m"]=sum(v>100 for v in errors)
        entry["probe_events_with_closer_certified_other_source_within_200m"]=sum(r["probe_events_with_closer_certified_other_source_within_200m"] for r in group)
        entry["mean_joint_probe_opportunities"]=statistics.mean(len(r["joint_probe_opportunities"]) for r in group)
        entry["cases_with_joint_probe_opportunities"]=sum(bool(r["joint_probe_opportunities"]) for r in group)
        summary[policy]=entry
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"rows.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
