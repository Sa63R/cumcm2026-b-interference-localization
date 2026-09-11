"""Scripted observations and geometry contracts; no simulator batches."""
import copy
import math

import pytest

from localization import CandidateRegion, BearingObservation
from q4_rl.micro_controller import (
    Q4MicroSearch, MicroCandidate, run_q4_micro, FEATURE_SCHEMA_VERSION,
    GLOBAL_DIM, CANDIDATE_DIM, GLOBAL_FEATURE_NAMES, CANDIDATE_FEATURE_NAMES,
    SAFE_CLEAR_RADIUS_M, GEOMETRY_GUARD_M,
)
from simulator_client.state import ClientState, Position
from strategies.search import _StopSearch


class ScriptedClient:
    def __init__(self):
        self.state = ClientState()
        self.remaining_real_time_s = None
        self.pending_request = None
        self.calls = []
        self.reply = lambda p,c: {"measure_result":"near" if c<=10 else "no_signal"}
        self.clear_reply = lambda p,c: "success"
        self.reject_measure = False
        self.exit_cost = 0.

    def enter(self):
        self.state.session = "active"
        return {"accepted":True}

    def exit(self):
        self.state.session = "exited"
        self.state.virtual_time_s += self.exit_cost
        return {"accepted":True}

    def _charge(self, point, fixed):
        self.state.virtual_time_s += math.ceil(self.state.position.distance_to(point)/5.*1e6)/1e6+fixed
        self.state.position = point

    def measure(self, point, channel):
        self.calls.append(("measure",point,channel))
        if self.reject_measure:
            return {"accepted":False}
        response = self.reply(point,channel)
        self._charge(point,5.+int(channel!=self.state.current_channel))
        self.state.current_channel = channel
        return {"accepted":True,**response}

    def clear(self, point, channel):
        self.calls.append(("clear",point,channel))
        result = self.clear_reply(point,channel)
        self._charge(point,5. if result=="success" else 3.)
        return {"accepted":True,"clear_result":result}


@pytest.fixture
def make(monkeypatch):
    import planning.q4_directional_cover as cover
    monkeypatch.setattr(cover,"certified_cover_points",lambda profile:
                        ((Position(100,0),Position(200,0)),{"passed":True,"test_only":True}))
    def factory(**kwargs):
        client = ScriptedClient()
        return Q4MicroSearch(client,max_expansions=0,**kwargs),client
    return factory


def set_rectangle(search, *, half_x=40., half_y=2., channel=1):
    # Analytic outer geometry fixture, not a scenario or hidden simulator state.
    region = CandidateRegion()
    region.vertices = ((-half_x,-half_y),(half_x,-half_y),(half_x,half_y),(-half_x,half_y))
    region.observations = [BearingObservation((-1000.,0.),0.)]
    region._circle = None
    search.regions[channel] = region
    search.first_bearings[channel] = 0.
    search.detected.add(channel)
    return region


def test_one_real_measure_bypasses_r8_interception(make):
    search,client = make()
    client.reply = lambda p,c: {"measure_result":"direction","svd_deg":0. if p.x==-1000 else 90.}
    search._perform("measure",Position(-1000,0),1,"test")
    search._perform("measure",Position(0,-1000),1,"test")
    disk = search.regions[1].enclosing_disk()
    assert 19.9<disk.radius<=40.
    center = Position.coerce(disk.center)
    search._probe_resolving_channel = 1  # Would activate R8's interception.
    client.reply = lambda p,c: {"measure_result":"no_signal"}
    before = len(client.calls)
    search._perform("measure",center,1,"active_localization")
    assert client.calls[before:]==[("measure",center,1)]
    assert 1 not in search.cleared
    assert (center,1) in search.actual_measurements
    assert not search.clear_before_probe_log


def test_certified_candidates_check_every_vertex_and_margin(make):
    search,client = make()
    region = set_rectangle(search,half_x=10.,half_y=3.)
    client.state.position = Position(100,0)
    candidates = [c for c in search._candidates() if c.kind=="certified_clear"]
    assert len(candidates)==2
    for candidate in candidates:
        assert max(math.dist((candidate.point.x,candidate.point.y),v) for v in region.vertices)<=SAFE_CLEAR_RADIUS_M-GEOMETRY_GUARD_M
        assert search._clear_certificate(candidate.point,1)["kind"].startswith("all_vertices")
    # The actor cannot submit an unsafe location even through a forged candidate.
    before = len(client.calls)
    with pytest.raises(_StopSearch,match="micro_clear_certificate_missing"):
        search._execute_candidate(MicroCandidate("certified_clear",Position(100,0),1))
    assert len(client.calls)==before
    assert search.report.learning["micro_steps"][-1]["accepted_requests"]==0


def test_ready_radius_without_margin_does_not_create_clear_certificate(make):
    search,client = make()
    region = set_rectangle(search,half_x=19.9,half_y=0.)
    assert not search._safe_points(1)
    assert not search._ready(1)
    assert search._clear_certificate(Position(0,0),1) is None
    assert not [c for c in search._candidates() if c.kind=="certified_clear"]
    assert [c for c in search._candidates() if c.kind=="grid_clear"]


def test_near_clear_is_actual_success_and_miss_is_a_contradiction(make):
    search,client = make()
    search._perform("measure",Position(0,0),1,"test")
    candidate = MicroCandidate("certified_clear",Position(0,0),1)
    client.clear_reply = lambda p,c:"no_target_in_range"
    with pytest.raises(_StopSearch,match="micro_clear_certificate_contradiction"):
        search._execute_candidate(candidate)
    assert 1 not in search.cleared
    assert not search.report.completion_certified_under_model
    assert search.report.error
    event = search.report.learning["micro_steps"][-1]
    assert event["accepted_requests"]==1 and event["cost_s"]==3.
    assert search.report.action_history[-1]["result"]=="no_target_in_range"


def test_grid_cover_retained_across_miss_positive_shrink_and_fallback(make):
    search,client = make()
    region = set_rectangle(search,half_x=80.,half_y=15.)
    state = search._grid(1)
    original_points, original_vertices = state.points,state.vertices
    # Dense deterministic geometry witnesses exercise cover retention, not learning.
    for x in range(-80,81,8):
        for y in range(-15,16,3):
            assert min(math.dist((x,y),(p.x,p.y)) for p in original_points)<20.
    assert len(search._grid_points(1))<=4<len(original_points)
    candidate = MicroCandidate("grid_clear",original_points[0],1,role="grid")
    client.clear_reply = lambda p,c:"no_target_in_range"
    before_vertices = copy.deepcopy(region.vertices)
    search._execute_candidate(candidate)
    assert region.vertices==before_vertices
    assert state.started and state.attempted=={original_points[0]}
    assert original_points[0] not in search._grid_points(1)
    # Only a new actual positive measurement shrinks the convex location region.
    client.reply = lambda p,c:{"measure_result":"direction","svd_deg":0.}
    search._perform("measure",Position(-1000,0),1,"test")
    assert region.vertices!=before_vertices
    assert search._grid(1) is state and state.points==original_points and state.vertices==original_vertices
    client.clear_reply = lambda p,c:"success" if p==original_points[-1] else "no_target_in_range"
    search._in_fallback = True
    try:
        assert search._resolve(1)
    finally:
        search._in_fallback = False
    actual_grid_points = [p for action,p,c in client.calls if action=="clear"]
    assert actual_grid_points==list(original_points)
    assert state.attempted==set(original_points)
    assert 1 in search.cleared


def test_unstarted_grid_recomputes_only_after_geometry_changes(make):
    search,client = make()
    region = set_rectangle(search,half_x=80.,half_y=15.)
    original = search._grid(1)
    client.state.position = Position(50,50)
    assert search._grid(1) is original
    region.observe((-1000,0),0.)
    replacement = search._grid(1)
    assert replacement is not original and not replacement.started and not replacement.attempted


def test_negative_measure_has_no_geometry_or_planned_cover_credit(make):
    search,client = make()
    region = set_rectangle(search)
    before = copy.deepcopy(region.vertices)
    client.reply = lambda p,c:{"measure_result":"no_signal"}
    search._candidates()
    assert not any(search.cover_ledger.values())
    search._perform("measure",Position(100+1e-10,0),1,"test")
    assert region.vertices==before and 1 not in search.cover_ledger[Position(100,0)]
    search._perform("measure",Position(100,0),1,"test")
    assert 1 in search.cover_ledger[Position(100,0)]


def test_g1_distinguishes_equal_radius_area_rotated_shapes_and_uses_budget(make):
    first,client = make(max_decisions=128)
    second,_ = make(max_decisions=128)
    set_rectangle(first,half_x=40.,half_y=2.)
    set_rectangle(second,half_x=2.,half_y=40.)
    candidate = [MicroCandidate("measure",Position(0,0),1)]
    g1,rows1 = first._features(candidate)
    g2,rows2 = second._features(candidate)
    assert rows1[0][:16]==rows2[0][:16]
    indexes = [CANDIDATE_FEATURE_NAMES.index(f"outer_support_{i*45}_deg") for i in range(8)]
    assert [rows1[0][i] for i in indexes]!=[rows2[0][i] for i in indexes]
    assert len(g1)==GLOBAL_DIM and len(rows1[0])==CANDIDATE_DIM
    assert CANDIDATE_DIM!=16 and GLOBAL_DIM==13
    first.report.learning["decisions"] = 64
    first.actions = 11
    client.state.virtual_time_s = 180000.
    changed,_ = first._features(candidate)
    assert changed[GLOBAL_FEATURE_NAMES.index("remaining_decisions_fraction")]==.5
    assert changed[-1]==.5
    assert changed[-2]==(first.max_actions-12)/first.max_actions
    assert first.report.learning["feature_schema"]["version"]==FEATURE_SCHEMA_VERSION


def test_g1_anchor_features_are_only_relative_actual_bearings(make):
    search,client = make()
    region = set_rectangle(search)
    candidate = [MicroCandidate("measure",Position(0,100),1)]
    _,rows = search._features(candidate)
    row = dict(zip(CANDIDATE_FEATURE_NAMES,rows[0]))
    assert row["first_positive_relative_x"]==-1000/3600
    assert row["first_positive_relative_y"]==-100/3600
    assert row["first_positive_view_sin"]==pytest.approx(100/math.hypot(1000,100))
    assert row["first_positive_view_cos"]==pytest.approx(1000/math.hypot(1000,100))
    region.observations.append(BearingObservation((0,-1000),90.))
    _,changed = search._features(candidate)
    latest = dict(zip(CANDIDATE_FEATURE_NAMES,changed[0]))
    assert latest["latest_positive_relative_y"]==-1100/3600
    assert latest["latest_positive_view_sin"]==pytest.approx(0.,abs=1e-15)


def test_action_budget_and_ceiling_movement_mask(make):
    search,client = make()
    point = Position(1e-7,0)
    client.state.max_virtual_duration_s = 5.0000015
    assert not search._can_measure(point,1)
    with pytest.raises(_StopSearch,match="virtual_budget"):
        search._perform("measure",point,1,"test")
    assert not client.calls
    client.state.max_virtual_duration_s = None
    search.actions = search.max_actions-1
    assert not search._candidates()


@pytest.mark.parametrize("limit",[0,1,128])
def test_complete_accounted_tail_and_one_request_per_micro_step(make,limit):
    observed = []
    def policy(global_features,rows):
        assert len(global_features)==GLOBAL_DIM and all(len(row)==CANDIDATE_DIM for row in rows)
        assert all(type(x) is float for row in [global_features]+rows for x in row)
        observed.append((global_features,rows))
        return next((i for i,row in enumerate(rows) if row[0] and row[6]),0)
    search,client = make(max_decisions=limit,policy=policy)
    client.exit_cost = 2.5
    report = search.run()
    assert report.completion_certified_under_model and not report.error
    assert len(observed)==report.learning["decisions"]
    assert all(row["accepted_requests"]==1 for row in report.learning["micro_steps"])
    assert all(row["end_actual_action_index"]-row["before_actual_action_index"]==1 for row in report.learning["micro_steps"])
    if limit:
        assert sum(row["cost_s"] for row in report.learning["transitions"])==pytest.approx(report.virtual_time_s)
    assert report.learning["decision_cost_s"]+report.learning["fallback_cost_s"]+report.learning["uncovered_cost_s"]==pytest.approx(report.virtual_time_s)
    assert report.learning["uncovered_cost_s"]==pytest.approx(2.5)


def test_rejected_request_gets_no_ledger_or_phantom_action(make):
    search,client = make(max_decisions=1)
    client.reject_measure = True
    report = search.run()
    assert report.completion_reason=="request_rejected"
    assert report.learning["micro_steps"][0]["accepted_requests"]==0
    assert report.learning["transitions"][0]["cost_s"]==0.
    assert not search.actual_measurements


def test_ten_actual_clears_still_need_unknown_channel_discovery(make):
    search,client = make()
    for channel in range(1,11):
        search._perform("measure",Position(0,0),channel,"test")
        search._execute_candidate(MicroCandidate("certified_clear",Position(0,0),channel))
    assert search._terminal_gate() is False
    for point in search.points:
        for channel in range(11,21):
            search._perform("measure",point,channel,"test")
    with pytest.raises(_StopSearch,match="q4_rl_certified_complete"):
        search._terminal_gate()


def test_factory_rejects_other_problem_without_accessing_client():
    with pytest.raises(ValueError):
        run_q4_micro(object(),problem=3)
