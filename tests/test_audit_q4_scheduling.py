import copy
import math
import pytest
from localization import CandidateRegion
from experiments.audit_q4_scheduling import audit_scheduling_prefix


def example():
    actions = [("measure",1,(-900.,0.),"direction",0.,"active_localization"),
               ("measure",1,(100.,-1000.),"direction",90.,"active_localization"),
               ("measure",20,(0.,-50.),"no_signal",None,"coverage"),
               ("measure",1,(100.,0.),"near",None,"active_localization"),
               ("clear",1,(100.,0.),"success",None,"near_clear")]
    h, wire, elapsed, current, tuned = [], [], 0., (0.,0.), 1
    region = CandidateRegion()
    for kind,c,p,result,bearing,phase in actions:
        elapsed += round(math.dist(current,p)/5*1e6)/1e6 + 5 + (kind=="measure" and c!=tuned)
        current=p
        if kind=="measure":tuned=c
        item={"action":kind,"channel":c,"position":list(p),"result":result,"phase":phase,"virtual_time_s":elapsed}
        response={"accepted":True,"virtual_time_s":elapsed,"measure_result" if kind=="measure" else "clear_result":result}
        if bearing is not None:
            item["bearing_deg"]=response["svd_deg"]=bearing
            region.observe(p,bearing)
        h.append(item);wire.append({"action":"/"+kind,"channel":c,"position":list(p),"response":response})
    disk=region.enclosing_disk();p=(0.,-50.);n=(200.,0.)
    event={"channel":1,"after_actual_action_count":3,"end_actual_action_count":5,
           "radius_m":disk.radius,"detour_m":math.dist(p,disk.center)+math.dist(disk.center,n)-math.dist(p,n),
           "actual_cost_s":h[4]["virtual_time_s"]-h[2]["virtual_time_s"],"budget_s":60.,"interrupted":False,"cleared":True}
    return {"row":{"successful":False},"history":wire,"summary":{"coverage_points":[list(p),list(n)],
        "action_history":h,"strategy_parameters":{"q4_r2_scheduling":"onroute","early_service_log":[event],"discovery_stop_log":[]}}}


def test_positive_prefix_and_unchanged_input():
    r=example();before=copy.deepcopy(r)
    assert audit_scheduling_prefix(r)["early_services"]==1
    assert r==before


@pytest.mark.parametrize("change",["cost","repeat","cap","radius","interrupted"])
def test_corrupt_service_evidence_rejected(change):
    r=example();params=r["summary"]["strategy_parameters"];e=params["early_service_log"][0]
    if change=="cost":e["actual_cost_s"]+=1
    elif change=="repeat":params["early_service_log"].append(copy.deepcopy(e))
    elif change=="cap":params["q4_r2_scheduling"]="cap16"
    elif change=="radius":e["radius_m"]=40.
    else:e["interrupted"]=True
    with pytest.raises(ValueError):audit_scheduling_prefix(r)


def test_fifteen_fabricated_known_channels_cannot_certify_discovery():
    r=example();p=r["summary"]["strategy_parameters"]
    p["discovery_stop_log"]=[{"after_actual_action_count":5,"known_channels":list(range(1,17)),
        "unresolved_channels":list(range(2,17)),"omitted_cover_stations":1,"discovery_only_not_removal":True}]
    with pytest.raises(ValueError,match="16 real known"):audit_scheduling_prefix(r)


def test_real_sixteen_positive_prefix_certifies_discovery_only():
    r=example();h=r["summary"]["action_history"];elapsed=h[-1]["virtual_time_s"]
    for c in range(2,17):
        elapsed+=6.
        h.append({"action":"measure","channel":c,"position":[100.,0.],"result":"near",
                  "phase":"active_localization","virtual_time_s":elapsed})
        r["history"].append({"action":"/measure","channel":c,"position":[100.,0.],
            "response":{"accepted":True,"measure_result":"near","virtual_time_s":elapsed}})
    r["summary"]["strategy_parameters"]["discovery_stop_log"]=[{
        "after_actual_action_count":len(h),"known_channels":list(range(1,17)),
        "unresolved_channels":list(range(2,17)),"omitted_cover_stations":1,"discovery_only_not_removal":True}]
    assert audit_scheduling_prefix(r)["discovery_stops"]==1
    r["row"]["successful"]=True
    with pytest.raises(ValueError,match="all-clear"):audit_scheduling_prefix(r)
