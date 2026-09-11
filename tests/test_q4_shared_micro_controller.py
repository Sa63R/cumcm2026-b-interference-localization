"""Actual-response contracts; scripted clients only, no sampled scenarios."""
from copy import deepcopy

import pytest

from tests.test_q4_rl_micro_controller import ScriptedClient
from q4_rl.micro_controller import run_q4_micro,MicroCandidate
from q4_rl.shared_micro_controller import (Q4SharedMicroSearch,run_q4_shared_micro,
    FEATURE_SCHEMA_VERSION,GLOBAL_DIM,CANDIDATE_DIM)
from q4_rl.shared_cover import PlannedAction,replay_shared_cover,position_key
from simulator_client.state import Position
from strategies.search import _StopSearch

ORIGIN=Position(0,0)
Q1,Q2=Position(770,200),Position(1170,-200)
STATION=Position(970,0)


def prepared():
    client=ScriptedClient()
    search=Q4SharedMicroSearch(client,max_shared_reviews=0,record_transitions=False)
    for c in range(1,11):
        search._perform("measure",Q1 if c==1 else Q2 if c==2 else ORIGIN,c,"scripted_positive")
    return search,client


@pytest.fixture(scope="module")
def packet():
    search,client=prepared()
    points=tuple(p for p in search.shared.service_points(ORIGIN) if p.position in (Q1,Q2))
    route=tuple(p for p in search.points if p!=STATION)+(STATION,)
    baseline=(PlannedAction("clear",Q1,1),PlannedAction("clear",Q2,2))+tuple(
        PlannedAction("measure",p,c,"coverage") for p in route for c in range(11,21))
    plans=search.shared.proposals(points,baseline,ORIGIN,client.state.current_channel,target_station=STATION)
    assert len(plans)==1
    search.shared_plan=plans[0]
    return search,client


def execute_packet(packet):
    search,client=deepcopy(packet)
    steps=[]
    while search.shared_plan is not None:
        assert len(steps)<50
        candidates=search._candidates()
        selected=candidates[search._heuristic(candidates)]
        before=len(search.report.action_history)
        search._execute_candidate(selected)
        assert len(search.report.action_history)-before==1
        steps.append(selected)
    return search,client,steps


def test_disabled_dispatch_is_original_actions_and_cost():
    original_client,disabled_client=ScriptedClient(),ScriptedClient()
    original=run_q4_micro(original_client,max_decisions=32,record_transitions=False)
    disabled=run_q4_shared_micro(disabled_client,shared_enabled=False,max_decisions=32,record_transitions=False)
    assert original.action_history==disabled.action_history
    assert original.virtual_time_s==disabled.virtual_time_s
    assert original.completion_reason==disabled.completion_reason
    assert original_client.calls==disabled_client.calls
    assert original.learning["feature_schema"]==disabled.learning["feature_schema"]


def test_enabled_without_selected_packet_preserves_physical_behavior():
    original=run_q4_micro(ScriptedClient(),max_decisions=32,record_transitions=False)
    enabled=run_q4_shared_micro(ScriptedClient(),max_decisions=32,record_transitions=False)
    assert original.action_history==enabled.action_history
    assert original.virtual_time_s==enabled.virtual_time_s
    assert enabled.learning["shared_completion_replay"]["complete"]
    assert enabled.learning["shared_service_actual_measurements"]==0


def test_service_scan_candidates_finite_and_checkpoint_schema_explicit():
    search,_=prepared()
    candidates=search._candidates()
    added=[c for c in candidates if c.role=="shared_service"]
    assert added and len({c.point for c in added})<=4
    assert len(added)<=4*len(search._unknown())
    assert all(c.kind=="measure" and c.channel in search._unknown() for c in added)
    features,rows=search._features(candidates)
    assert FEATURE_SCHEMA_VERSION!="q4-micro-g1-v1"
    assert (GLOBAL_DIM,CANDIDATE_DIM)==(17,54)
    assert len(features)==17 and all(len(row)==54 for row in rows)
    with pytest.raises(ValueError,match="incompatible"):
        Q4SharedMicroSearch(ScriptedClient(),policy=lambda g,c:0)


def test_no_certified_packet_rule_ignores_added_service_scans():
    search,_=prepared()
    candidates=search._candidates()
    assert candidates[search._heuristic(candidates)].role!="shared_service"
    assert search.shared.events==[] and search.shared.cancelled_stations==set()


def test_packet_executes_one_request_per_decision_and_no_false_fixed_credit(packet):
    search,_,steps=execute_packet(packet)
    assert len([c for c in steps if c.kind=="measure"])==20
    assert search.shared_scan_count==20
    assert search.cover_ledger[STATION]==set()
    assert not any(Position.coerce(a["position"])==STATION for a in search.report.action_history)
    assert len(search.shared.events)==10 and len(search.shared.cancelled_stations)==1
    assert all(position_key(STATION) not in search.shared.pending[c] for c in range(11,21))
    assert all((STATION,c) not in search.actual_measurements for c in range(11,21))
    assert replay_shared_cover(search.report.action_history,search.shared.artifact())["passed"]


def test_cancelled_unknown_candidates_masked_but_known_service_not_fabricated(packet):
    search,_,_=execute_packet(packet)
    assert all(not search._can_measure(STATION,c) for c in range(11,21))
    assert all(c.channel not in range(11,21) or c.point!=STATION for c in search._candidates())
    assert search._pending_unknown(STATION)==set()
    assert not search._terminal_gate()
    assert not search.report.completion_certified_under_model


def test_final_completion_proves_actual_support_after_station_removed(packet):
    search,_,_=execute_packet(packet)
    for c in sorted(search.detected-search.cleared):
        search._perform("clear",search.near_points[c],c,"scripted_safe_clear")
    for point in search.points:
        for c in sorted(search._pending_unknown(point)):
            search._perform("measure",point,c,"scripted_actual_residual_cover")
    assert search.cover_ledger[STATION]==set()
    with pytest.raises(_StopSearch,match="q4_shared_micro_certified_complete"):
        search._terminal_gate()
    result=search.report.learning["shared_completion_replay"]
    assert result["complete"] and result["actual_cleared"]==list(range(1,11))
    assert result["certified_absent"]==list(range(11,21))


def test_rejected_shared_measure_never_reaches_either_ledger(packet):
    search,client=deepcopy(packet)
    client.reject_measure=True
    history=deepcopy(search.report.action_history)
    artifact=deepcopy(search.shared.artifact())
    with pytest.raises(_StopSearch,match="request_rejected"):
        search._execute_candidate(MicroCandidate("measure",Q1,11,False,"shared_service"))
    assert search.report.action_history==history
    assert search.shared.artifact()==artifact
    assert search.shared_scan_count==0


def test_no_credit_before_complete_channel_packet(packet):
    search,_=deepcopy(packet)
    for point in (Q1,Q2):
        for c in range(11,20):
            search._execute_candidate(MicroCandidate("measure",point,c,False,"shared_service"))
    assert search.shared_plan is not None
    assert all(position_key(STATION) in search.shared.pending[c] for c in range(11,21))
    assert search.shared.events==[] and search.cover_ledger[STATION]==set()


def test_public_proxy_covers_every_retained_unknown_obligation():
    search,_=prepared()
    base=super(Q4SharedMicroSearch,search)._candidates()
    points=search._public_service_candidates(base)
    schedule=search._public_schedule(points,base)
    actual={(a.position,a.channel) for a in schedule if a.kind=="measure"}
    assert all((Position(*p),c) in actual for c in search._unknown() for p in search.shared.pending[c])
    assert len({p.source_channel for p in points})==len(points)<=4
