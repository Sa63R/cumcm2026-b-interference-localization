"""Scripted public-receipt/geometry tests; no simulator or dataset access."""
from copy import deepcopy
from dataclasses import replace

import pytest

from planning.q4_directional_cover import certified_cover_points
from simulator_client.state import Position
from q4_rl.shared_cover import (SharedCoverLedger, PlannedAction, ServicePoint,
    public_plan_cost, replay_shared_cover)

ORIGIN = Position(0,0)
STATION = Position(970,0)
Q1,Q2 = Position(770,200),Position(1170,-200)


def receipt(ledger, point, channel, result="no_signal", **kwargs):
    return ledger.observe(len(ledger.history),"measure",point,channel,result,accepted=True,**kwargs)


def fixture(max_queries=8):
    fixed = certified_cover_points("compact_22")[0]
    ledger = SharedCoverLedger(fixed,max_certificate_queries=max_queries,query_wall_s=2.)
    for c in range(1,11):
        receipt(ledger,Q1 if c==1 else Q2 if c==2 else ORIGIN,c,"near")
    points = tuple(p for p in ledger.service_points(ORIGIN) if p.position in (Q1,Q2))
    route = tuple(p for p in fixed if p!=STATION)+(STATION,)
    baseline = (PlannedAction("clear",Q1,1),PlannedAction("clear",Q2,2))+tuple(
        PlannedAction("measure",p,c,"coverage") for p in route for c in range(11,21))
    return ledger,points,baseline


@pytest.fixture(scope="module")
def proposed():
    ledger,points,baseline = fixture()
    plans = ledger.proposals(points,baseline,ORIGIN,10,target_station=STATION)
    assert len(plans)==1 and len(plans[0].service_points)==2
    return ledger,plans[0],baseline


def committed(proposed):
    ledger,plan,_ = deepcopy(proposed)
    for q in (Q1,Q2):
        for c in range(11,21):
            receipt(ledger,q,c)
    assert ledger.cancel_station(plan)
    return ledger


def test_public_cost_counts_full_route_detection_switch_and_clear():
    result = public_plan_cost((PlannedAction("measure",ORIGIN,2),
        PlannedAction("clear",Position(5,0),1),
        PlannedAction("measure",Position(5,0),2)),ORIGIN,1)
    assert result==dict(movement_s=1.,detection_s=10.,switching_s=1.,clear_upper_s=5.,
                       total_upper_s=17.,action_count=3)


def test_proposals_are_prospective_only(proposed):
    ledger,plan,_ = deepcopy(proposed)
    assert plan.estimated_saved_s>0
    assert plan.after_cost["detection_s"]-plan.before_cost["detection_s"]==50.
    assert plan.before_cost["movement_s"]>plan.after_cost["movement_s"]
    assert ledger.cancelled_stations==set() and ledger.events==[]
    assert all((970.,0.) in ledger.pending[c] for c in range(11,21))
    assert all(not ledger.negatives[c] for c in range(11,21))
    assert not ledger.cancel_station(plan)


def test_single_points_do_not_gain_joint_certificate(proposed):
    ledger,_,_ = proposed
    assert all(a["diagnostic"]["status"]!="exact_certified" for a in ledger.attempts if len(a["service_points"])==1)
    assert any(a["diagnostic"]["status"]=="exact_certified" for a in ledger.attempts if len(a["service_points"])==2)


def test_missing_one_channel_preserves_all_station_obligations(proposed):
    ledger,plan,_ = deepcopy(proposed)
    for q in (Q1,Q2):
        for c in range(11,20): receipt(ledger,q,c)
    snapshot = deepcopy(ledger.pending)
    assert not ledger.cancel_station(plan)
    assert ledger.pending==snapshot and ledger.events==[] and not ledger.transactions
    receipt(ledger,Q1,20)
    assert not ledger.cancel_station(plan)
    receipt(ledger,Q2,20)
    assert ledger.cancel_station(plan)
    assert not ledger.cancel_station(plan)


def test_rejected_or_wrong_coordinate_is_not_actual_credit(proposed):
    ledger,plan,_ = deepcopy(proposed)
    assert not ledger.observe(999,"measure",Q1,20,"no_signal",accepted=False)
    for q in (Q1,Q2):
        for c in range(11,21):
            receipt(ledger,Position(q.x+.001,q.y) if c==20 else q,c)
    assert not ledger.cancel_station(plan)


def test_actual_positive_channel_leaves_unknown_packet(proposed):
    ledger,plan,_ = deepcopy(proposed)
    for q in (Q1,Q2):
        for c in range(11,21):
            if q==Q2 and c==12: continue
            receipt(ledger,q,c,"near" if c==12 else "no_signal")
    assert ledger.cancel_station(plan)
    assert len(ledger.events)==9 and ledger.transactions[0]["channels"]==[11]+list(range(13,21))
    assert replay_shared_cover(ledger.history,ledger.artifact())["cancelled_station_count"]==1


def test_whole_station_and_pair_counts_separate_and_replay(proposed):
    ledger = committed(proposed)
    result = replay_shared_cover(ledger.history,ledger.artifact())
    assert result["passed"] and result["deleted_obligations"]==10
    assert result["cancelled_station_count"]==1 and not result["complete"]
    assert result["certified_absent"]==[]
    assert all((970.,0.) not in ledger.pending[c] for c in range(11,21))
    assert all(len(ledger.pending[c])==21 for c in range(11,21))


@pytest.mark.parametrize("tamper",["inflated_count","missing_channel","future_scan","unmapped_pairs","duplicate_group","future_prefix"])
def test_replay_rejects_false_station_claims(proposed,tamper):
    ledger = committed(proposed)
    artifact = deepcopy(ledger.artifact())
    group = artifact["station_transactions"][0]
    if tamper=="inflated_count": artifact["cancelled_station_count"]=10
    if tamper=="missing_channel": group["channels"].pop()
    if tamper=="future_scan": group["service_points"][0][0]+=.001
    if tamper=="unmapped_pairs": artifact["station_transactions"]=[]
    if tamper=="duplicate_group": artifact["station_transactions"].append(deepcopy(group))
    if tamper=="future_prefix": group["after_action_index"]+=1
    with pytest.raises(ValueError): replay_shared_cover(ledger.history,artifact)


def test_visiting_cancelled_station_invalidates_physical_skip_count(proposed):
    ledger = committed(proposed)
    receipt(ledger,STATION,1,"near")
    assert ledger.artifact()["cancelled_stations_not_actually_visited"]==[]
    assert replay_shared_cover(ledger.history,ledger.artifact())["passed"]


def test_cost_negative_two_point_plan_filtered_before_certificate():
    ledger,points,_ = fixture()
    baseline = (PlannedAction("clear",Q1,1),)+tuple(PlannedAction("measure",STATION,c,"coverage") for c in range(11,21))
    baseline += (PlannedAction("clear",Q2,2),)+tuple(PlannedAction("measure",Position(*p),c,"coverage")
        for p in sorted(ledger.fixed) if p!=(970.,0.) for c in range(11,21))
    assert ledger.proposals(points,baseline,ORIGIN,10,target_station=STATION)==()
    # The one-point substitutions can save one switch but fail geometry.
    # The costly two-point proposal must not reach exact certification.
    assert all(len(attempt["service_points"])==1 for attempt in ledger.attempts)
    assert ledger.certificate_queries==1+len(ledger.attempts)


def test_truncated_proxy_plan_rejected_before_certificate():
    ledger,points,baseline = fixture()
    with pytest.raises(ValueError,match="omits"):
        ledger.proposals(points,baseline[:-1],ORIGIN,10,target_station=STATION)
    assert ledger.certificate_queries==1


def test_service_point_must_match_public_actual_geometry():
    ledger,points,baseline = fixture()
    with pytest.raises(ValueError,match="history"):
        ledger.proposals((ServicePoint(Position(900,50),1,"actual_near"),),baseline,ORIGIN,10)
    assert len(ledger.service_points(ORIGIN))<=4
    with pytest.raises(ValueError): ledger.proposals(points*3,baseline,ORIGIN,10)


def test_kept_old_station_service_prevents_whole_station_proposal():
    ledger,points,baseline = fixture()
    baseline += (PlannedAction("measure",STATION,1,"service"),)
    assert ledger.proposals(points,baseline,ORIGIN,10,target_station=STATION)==()


def test_certificate_episode_budget_counts_initial_and_repeated_calls():
    ledger,points,baseline = fixture(max_queries=1)
    for _ in range(3):
        assert ledger.proposals(points,baseline,ORIGIN,10,target_station=STATION)==()
    assert ledger.certificate_queries==1
    with pytest.raises(ValueError): SharedCoverLedger(ledger.fixed,max_certificate_queries=9)


def test_direction_geometry_and_accepted_order_only():
    ledger,_,_ = fixture()
    with pytest.raises(ValueError):
        ledger.observe(0,"measure",ORIGIN,11,"near",accepted=True)
    with pytest.raises(ValueError): receipt(ledger,ORIGIN,11,"direction")
    assert 11 not in ledger.known
    receipt(ledger,ORIGIN,11,"direction",bearing_deg=30.)
    assert ledger.regions[11].observations[0].bearing_deg==30.
    assert len(ledger.service_points(ORIGIN))<=4


def test_damaged_plan_and_noninner_target_rejected(proposed):
    ledger,plan,baseline = deepcopy(proposed)
    with pytest.raises(ValueError): ledger.cancel_station(replace(plan,proof_root="bad"))
    with pytest.raises(ValueError): ledger.cancel_station(replace(plan,station=ORIGIN))
    assert ledger.proposals(plan.service_points,baseline,ORIGIN,10,target_station=ORIGIN)==()
