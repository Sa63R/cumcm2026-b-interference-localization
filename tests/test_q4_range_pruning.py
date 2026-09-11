"""Constructed geometry and scan bookkeeping only; no new scenario batches."""

import copy
import math
from types import SimpleNamespace

import pytest

from localization import CandidateRegion
from simulator_client.state import Position
from strategies.q4_range_pruning import Q4RangePruningSearch, region_distance_lower, run_q4_range_pruning


@pytest.mark.parametrize("polygon,point,expected", [
    ([(-1,-1),(1,-1),(1,1),(-1,1)],(4,5),5.),
    ([(-1,1),(1,1),(1,-1),(-1,-1)],(4,5),5.),
    ([(-1,-1),(1,-1),(1,1),(-1,1)],(0,0),0.),
    ([(0,0)],(3,4),5.), ([(0,0),(2,0)],(1,4),4.),
    ([(0,0),(1,0),(2,0)],(3,4),math.sqrt(17)),
    ([],(2000,0),0.),
])
def test_convex_distance_is_conservative_including_degeneracies(polygon, point, expected):
    actual = region_distance_lower(point, polygon)
    assert 0 <= actual <= expected
    assert abs(actual-expected) < 1e-6


def test_threshold_guard_does_not_skip_a_boundary_reachable_source():
    for x in (1499.999, 1500., 1500.00001):
        assert region_distance_lower((x,0),[(0,0)]) <= 1500.00001
    assert region_distance_lower((1500.001,0),[(0,0)]) > 1500.00001


def search(enabled=True):
    # Unit-level scan fixture: no client methods exist, so accidental HTTP or
    # real simulator calls cannot occur. Only _perform is explicitly recorded.
    s = Q4RangePruningSearch.__new__(Q4RangePruningSearch)
    s.mode = 'state_pruned'
    s.range_pruning_enabled, s.range_skips, s.skipped_scans = enabled, [], []
    region = CandidateRegion(); region.observe((1400.,0.),0.)
    s.regions, s.detected, s.cleared, s.near_points = {7:region}, {7}, set(), {}
    s.observed_positions = {7:{(1400.,0.)}}
    s.blocked = {7}
    s.client = SimpleNamespace(state=SimpleNamespace(current_channel=7))
    s.report = SimpleNamespace(action_history=[],coverage_points_visited=0)
    def perform(action, point, channel, phase):
        s.client.state.current_channel = channel
        s.report.action_history.append({'action':action,'channel':channel,
            'position':[point.x,point.y],'phase':phase,'result':'no_signal'})
        s.observed_positions.setdefault(channel,set()).add((point.x,point.y))
    s._perform = perform
    return s


def test_only_known_far_channel_deleted_and_no_observation_or_region_written():
    s = search(); region = s.regions[7]
    vertices, observations = copy.deepcopy(region.vertices), list(region.observations)
    observed = copy.deepcopy(s.observed_positions[7])
    s._scan(Position(-1800,0))
    assert [r['channel'] for r in s.report.action_history] == [c for c in range(1,21) if c!=7]
    assert len(s.range_skips) == 1 and s.range_skips[0]['channel'] == 7
    assert s.range_skips[0]['after_actual_action_count'] == 0
    assert s.range_skips[0]['distance_lower_bound_m'] > 1500.00001
    assert region.vertices == vertices and region.observations == observations
    assert s.observed_positions[7] == observed
    assert s.report.coverage_points_visited == 1 and not s.blocked


def test_unknown_channel_never_uses_a_stray_region_as_evidence():
    s = search(); s.detected.clear()
    s._scan(Position(-1800,0))
    assert len(s.report.action_history)==20 and not s.range_skips


def test_ready_skip_uses_original_certificate_log_and_cleared_channel_is_ignored():
    s = search(); s.near_points[8] = Position(1000,0); s.detected.add(8); s.cleared.add(9)
    s._scan(Position(-1800,0))
    assert [v['channel'] for v in s.skipped_scans] == [8]
    assert s.skipped_scans[0]['reason'] == 'near'
    assert [v['channel'] for v in s.range_skips] == [7]
    assert all(r['channel'] not in (7,8,9) for r in s.report.action_history)


def test_disabled_exactly_delegates_legacy_scan_and_no_skip_case_is_identical():
    from strategies.q4_cover_search import Q4CoverSearch
    a, b = search(enabled=False), search(enabled=True)
    # A point near C has no eligible deletion; both orders and real prefixes match.
    a._scan(Position(1000,0)); b._scan(Position(1000,0))
    assert a.report.action_history == b.report.action_history
    assert not a.range_skips and not b.range_skips
    c, d = search(enabled=False), search(enabled=False)
    c._scan(Position(-1800,0)); Q4CoverSearch._scan(d,Position(-1800,0))
    assert c.report.action_history == d.report.action_history


@pytest.mark.parametrize('kwargs',[{'problem':3},{'max_actions':True},{'max_active_probes':31},{'max_expansions':-1}])
def test_invalid_options_rejected_without_client_access(kwargs):
    with pytest.raises(ValueError): run_q4_range_pruning(None,**kwargs)
