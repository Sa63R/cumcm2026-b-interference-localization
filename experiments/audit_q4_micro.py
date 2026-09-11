"""Post-hoc observation-prefix audit of saved Q4 micro records; no simulator."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from localization import CandidateRegion
from planning.coverage import clearance_grid


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def close(left,right,message,tolerance=2e-5):
    require(math.isfinite(left) and math.isfinite(right) and abs(left-right)<=tolerance,message)


def audit_micro_record(record):
    """Only summary, public accepted history and aggregate billed time are read."""
    summary = record["summary"]
    learning = summary["learning"]
    require(learning["algorithm"]=="q4-micro-g1-v1","Unexpected micro schema")
    physical = [a for a in record["history"] if a["action"] in ("/measure","/clear")]
    steps, history = learning["micro_steps"],summary["action_history"]
    require(len(physical)==len(history),"Public/strategy action history differs")
    require(len(steps)==learning["decisions"],"Decision count mismatch")
    require(learning["maximum_candidates"]<=636,"Candidate cap exceeded")
    require(len(learning["feature_schema"]["global_features"])==13 and
            len(learning["feature_schema"]["candidate_features"])==50,"Feature dimensions changed")
    by_index = {}
    for i,event in enumerate(steps):
        require(event["before_actual_action_index"]==i and event["end_actual_action_index"]==i+1,
                "Micro decisions are not a consecutive single-request prefix")
        require(event["accepted_requests"]==1,"Micro decision did not execute one request")
        by_index[i] = event
    regions, near, first_bearings, cleared = {},{},{},set()
    started_grids, grid_attempts = {},{}
    grid_ledgers = learning["grid_ledgers"]
    previous_time, micro_cost, safe_count, grid_misses = 0.,0.,0,0
    for i,(wire,action) in enumerate(zip(physical,history)):
        response, channel = wire["response"],wire["channel"]
        point = (wire["position"]["x"],wire["position"]["y"])
        require(response["accepted"] is True,"Unaccepted request in physical history")
        require(wire["action"]=="/"+action["action"] and list(point)==action["position"]
                and channel==action["channel"],"Requested/executed action mismatch")
        current_time = float(response["virtual_time_s"])
        close(action["virtual_time_s"],current_time,"Strategy/public time mismatch")
        event = by_index.get(i)
        if event:
            kind = event["kind"]
            require(list(point)==event["position"] and channel==event["channel"],"Micro target differs from request")
            require((kind=="measure")==(wire["action"]=="/measure"),"Micro action type mismatch")
            close(event["cost_s"],current_time-previous_time,"Micro billed delta mismatch")
            micro_cost += event["cost_s"]
            if kind=="certified_clear":
                require(channel not in cleared,"Repeated certified clear of cleared source")
                if channel in near:
                    maximum = math.dist(point,near[channel])+5.
                    proof = "actual_near_observation_plus_5m_disk"
                else:
                    require(channel in regions and regions[channel].vertices,"Clear without positive location region")
                    maximum = max(math.dist(point,v) for v in regions[channel].vertices)
                    proof = "all_vertices_of_actual_positive_outer_polygon"
                certificate = event["clear_certificate"]
                require(certificate["kind"]==proof,"Certificate provenance differs from public history")
                close(certificate["max_distance_upper_m"],maximum,"Clear distance certificate mismatch",1e-7)
                require(maximum<=19.9-1e-5,"Clear is not certified by the complete outer geometry")
                require(response["clear_result"]=="success","Certified clear contradicted by real response")
                safe_count += 1
        is_grid = (event and event["kind"]=="grid_clear") or action["phase"]=="micro_fallback_grid_resume"
        if is_grid:
            require(str(channel) in grid_ledgers,"Missing complete grid ledger")
            ledger = grid_ledgers[str(channel)]
            queue = tuple(tuple(p) for p in ledger["complete_queue"])
            if channel not in started_grids:
                require(channel in regions and regions[channel].vertices,"Grid without positive location region")
                vertices = tuple(tuple(v) for v in regions[channel].vertices)
                require(vertices==tuple(tuple(v) for v in ledger["original_vertices"]),"First executed grid snapshot differs from observed region")
                regenerated = clearance_grid(vertices,bearing_deg=first_bearings[channel],spacing=28.,start=point)
                require(set(queue)=={(p.x,p.y) for p in regenerated},"Full coverage grid omitted or invented a cell")
                require(len(queue)==len(set(queue)),"Duplicate point in full grid queue")
                started_grids[channel] = queue
                grid_attempts[channel] = []
            require(point in started_grids[channel],"Executed grid point absent from frozen queue")
            require(point not in grid_attempts[channel],"Repeated grid attempt")
            grid_attempts[channel].append(point)
            grid_misses += response["clear_result"]=="no_target_in_range"
        if wire["action"]=="/measure":
            result = response["measure_result"]
            if result=="direction":
                first_bearings.setdefault(channel,response["svd_deg"])
                regions.setdefault(channel,CandidateRegion()).observe(point,response["svd_deg"])
            elif result=="near":
                near[channel] = point
        elif response["clear_result"]=="success":
            cleared.add(channel)
        previous_time = current_time
    require(set(grid_ledgers)=={str(c) for c in started_grids},"Unexecuted grid credited as started")
    for channel,queue in started_grids.items():
        ledger = grid_ledgers[str(channel)]
        require(set(grid_attempts[channel])=={tuple(p) for p in ledger["attempted_points"]},"Lost or invented grid progress")
        require(ledger["remaining"]==len(queue)-len(grid_attempts[channel]),"Grid remainder mismatch")
        require(ledger["actually_cleared"]==(channel in cleared),"Grid completion uses unobserved truth")
    for channel,region in regions.items():
        estimate = summary["source_estimates"][str(channel)]
        require(tuple(tuple(v) for v in estimate["vertices"])==tuple(tuple(v) for v in region.vertices),
                "Final convex region differs from positive-only replay")
    close(micro_cost,learning["decision_cost_s"],"Decision-cost sum mismatch")
    close(micro_cost+learning["fallback_cost_s"]+learning["uncovered_cost_s"],
          record["row"]["virtual_time_s"],"Complete episode bill mismatch")
    require(learning["fallback_actions"]==len(physical)-len(steps),"Fallback request count mismatch")
    return dict(passed=True,decisions=len(steps),actual_micro_requests=len(steps),
                certified_clears=safe_count,started_grids=len(started_grids),grid_misses=grid_misses,
                fallback_actual_requests=learning["fallback_actions"],
                decision_cost_s=micro_cost,fallback_cost_s=learning["fallback_cost_s"],
                time_over_lower_bound=record["row"]["time_over_lower_bound"],
                public_prefix_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    evidence = json.loads((args.archive/"evidence.json").read_bytes())
    manifest_raw = (args.archive/"manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    require(sha(manifest_raw)==evidence["manifest_sha256"],"Manifest evidence hash mismatch")
    require(sha((args.archive/"summary.json").read_bytes())==evidence["summary_sha256"],"Summary evidence hash mismatch")
    with zipfile.ZipFile(args.archive/"source.zip") as archive:
        require(set(archive.namelist())==set(manifest["source_sha256"]),"Frozen source file set mismatch")
        for name,expected in manifest["source_sha256"].items():
            require(sha(archive.read(name))==expected,"Frozen source bytes mismatch: "+name)
    results = {}
    for name,expected in evidence["record_sha256"].items():
        path = args.archive/"records"/name
        require(sha(path.read_bytes())==expected,"Raw record hash mismatch")
        record = json.loads(gzip.decompress(path.read_bytes()))
        if record["row"]["strategy"].startswith("micro_"):
            results[name] = audit_micro_record(record)
    output = dict(version="q4-micro-public-prefix-audit-v1",passed=all(r["passed"] for r in results.values()),
        records=results,source_archive_verified=True,raw_record_hashes_verified=True,
        audit_source_sha256=sha(Path(__file__).read_bytes()),
        limitations="Scripted tests plus four development cases; geometric arithmetic uses engineering margins, not interval arithmetic. No global performance or official simulator claim.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    require(not args.output.exists(),"Refusing to overwrite an existing audit")
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(passed=output["passed"],records=len(results),certified_clears=sum(r["certified_clears"] for r in results.values())),allow_nan=False))


if __name__=="__main__":
    main()
