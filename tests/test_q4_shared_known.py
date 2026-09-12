"""Constructed replies and actual inherited controller calls; no scenes/network."""
import ast
import copy
import inspect
import math
import textwrap

import pytest

from geometry import Circle
from localization import BearingObservation, CandidateRegion
from simulator_client.state import Position
from strategies.q4_known_source import Q4KnownSource
from strategies.q4_shared_known import Q4SharedKnown, PHASE, run_q4_shared_known
from strategies.search import _StopSearch
from tests.test_q4_known_source import ScriptedReplies


def make(monkeypatch, *, points=(), max_actions=20000):
    monkeypatch.setattr('planning.q4_directional_cover.certified_cover_points', lambda profile:
        (tuple(Position(*p) for p in points), {'passed': True, 'test_only': True}))
    client = ScriptedReplies()
    return Q4SharedKnown(client, max_actions, 6, max_expansions=0), client


def prime_target(policy, client, channel=1):
    for point, bearing in [((-1000., 0.), 0.), ((0., -1000.), 90.)]:
        client.measure_replies = [('direction', bearing)]
        policy._perform('measure', Position(*point), channel, 'constructed_observation')
    assert 19.9 < policy.regions[channel].enclosing_disk().radius < 40.


def prime_clearable(policy, client, channel=2, position=(0., 100.)):
    client.measure_replies = [('near', None)]
    policy._perform('measure', Position(*position), channel, 'constructed_observation')


def trigger(policy, client, channel=2, position=(0., 100.)):
    prime_clearable(policy, client, channel, position)
    client.clear_replies = ['success']
    assert policy._resolve(channel)
    assert policy._joint_context is policy._probe_resolving_channel is None


def test_real_macro_share_then_normal_clear_no_fabricated_prediction(monkeypatch):
    policy, client = make(monkeypatch)
    prime_target(policy, client)
    prime_clearable(policy, client, position=(0., 20.))
    original = copy.deepcopy(policy.regions[1].__dict__)
    before_count = policy.report.measurement_count
    client.measure_replies = [('near', None)]
    client.clear_replies = ['success', 'success']
    def inspect_before_measure():
        assert policy._joint_context is policy._probe_resolving_channel is None
        assert policy.service_deadline is None
        assert policy.known_source_service_log[-1]['status'] == 'resolved'
        assert policy.known_source_service_log[-1]['service_end_action_count'] == 4
        assert policy.joint_resolvers[0]['status'] == 'cleared'
        assert not policy._ready(1) and policy.cleared == {2}
        assert policy.regions[1].__dict__ == original
    client.before_measure = inspect_before_measure
    policy._execute_plan()
    events = policy.shared_known_observation_log
    assert [(e['trigger_channel'], e['selected_channel'], e['status']) for e in events] == [
        (2, 1, 'measured'), (1, None, 'no_candidate')]
    event = events[0]
    assert event['after_actual_action_count'] == 4 and event['end_actual_action_count'] == 5
    assert event['parent_resolver_id'] == 0 and event['parent_end_action_count'] == 4
    assert event['actual_cost_s'] == 6. and event['actual_result'] == 'near'
    assert event['candidates'][0]['predicted_radius_m'] <= 19.9
    assert event['actual_radius_after_m'] > 19.9  # Near feedback does not clip C.
    assert event['actual_ready_after'] and policy.cleared == {1, 2}
    assert [a['phase'] for a in policy.report.action_history[3:]] == ['near_clear', PHASE, 'near_clear']
    assert policy.report.measurement_count == before_count + 1
    assert policy.report.coverage_points_visited == 0 and not policy.range_skips
    assert len(policy.joint_radio[1]) == 3 and not policy.continuation_log
    assert all(e['config'] == 'all_known_matrix' for e in policy.route_log)


@pytest.mark.parametrize('response', [('no_signal', None), ('direction', 269.5), ('near', None)])
def test_actual_feedback_not_nominal_controls_live_belief(monkeypatch, response):
    policy, client = make(monkeypatch); prime_target(policy, client); trigger(policy, client)
    old = copy.deepcopy(policy.regions[1].__dict__)
    old_seen = copy.deepcopy(policy.observed_positions)
    position = client.state.position
    choices = policy._shared_candidates()
    assert choices[0]['eligible'] and choices[0]['nominal_bearing_deg'] == 270.18
    assert policy.regions[1].__dict__ == old and policy.observed_positions == old_seen
    client.measure_replies = [response]
    assert policy._share_after_clear()
    assert client.state.position == position and policy.cleared == {2}
    assert (round(position.x, 6), round(position.y, 6)) in policy.observed_positions[1]
    if response[0] == 'direction':
        expected = CandidateRegion()
        for obs in old['observations']:
            expected.observe(obs.position, obs.bearing_deg)
        expected.observe(position, 269.5)
        assert policy.regions[1].vertices == expected.vertices
        assert policy.regions[1].vertices != tuple(map(tuple, choices[0]['predicted_vertices']))
    else:
        assert policy.regions[1].__dict__ == old
    assert policy._ready(1) == (response[0] == 'near' or policy.regions[1].enclosing_disk().radius <= 19.9)
    assert len(policy.report.action_history) == 5  # No immediate forced clear.


def test_once_per_trigger_and_target_even_real_miss(monkeypatch):
    policy, client = make(monkeypatch); prime_target(policy, client); trigger(policy, client)
    client.measure_replies = [('no_signal', None)]
    assert policy._share_after_clear()
    calls = len(client.calls)
    assert not policy._share_after_clear() and len(client.calls) == calls
    trigger(policy, client, 3, (0., 101.))
    assert not policy._share_after_clear()
    event = policy.shared_known_observation_log[-1]
    assert event['candidates'][0]['reason'] == 'target_already_attempted'
    assert event['attempted_before'] == event['attempted_after'] == [1]
    assert event['seen_clear_after'] == [2, 3]


def test_no_candidate_consumes_trigger_and_cannot_retry_after_belief_changes(monkeypatch):
    policy, client = make(monkeypatch); trigger(policy, client)
    assert not policy._share_after_clear()
    prime_target(policy, client)
    assert not policy._share_after_clear() and len(policy.shared_known_observation_log) == 1


@pytest.mark.parametrize('gate', ['request', 'actions', 'virtual', 'real'])
def test_real_global_gates_reject_without_extra_action_or_finally_measure(monkeypatch, gate):
    policy, client = make(monkeypatch); prime_target(policy, client); trigger(policy, client)
    old = copy.deepcopy(policy.regions[1].__dict__)
    before = len(policy.report.action_history)
    client.measure_replies = [('direction', 270.)]
    if gate == 'request': client.reject_kind = 'measure'
    elif gate == 'actions': policy.max_actions = policy.actions + 1
    elif gate == 'virtual': client.state.max_virtual_duration_s = client.state.virtual_time_s + 5.
    else: client.remaining_real_time_s = 2.
    calls = len(client.calls)
    with pytest.raises(_StopSearch): policy._share_after_clear()
    event = policy.shared_known_observation_log[-1]
    assert event['status'] == 'interrupted'
    assert event['after_actual_action_count'] == event['end_actual_action_count'] == before
    assert event['actual_result'] is None and event['actual_cost_s'] == 0.
    assert len(client.calls) == calls + (gate == 'request')
    assert policy.regions[1].__dict__ == old and policy.cleared == {2}
    assert policy.shared_seen_clears == {2} and policy.shared_attempted_channels == {1}


def test_no_extra_safety_reserve_a_single_measure_at_global_boundary(monkeypatch):
    policy, client = make(monkeypatch); prime_target(policy, client); trigger(policy, client)
    policy.max_actions = policy.actions + 2  # One real measure and the inherited exit reserve.
    client.measure_replies = [('no_signal', None)]
    assert policy._share_after_clear()
    assert policy.actions == policy.max_actions - 1


def test_same_channel_cost_five_and_switch_cost_six(monkeypatch):
    policy, client = make(monkeypatch); prime_target(policy, client)
    prime_clearable(policy, client)
    client.measure_replies = [('no_signal', None)]
    policy._perform('measure', Position(1., 100.), 1, 'constructed_observation')
    client.clear_replies = ['success']; assert policy._resolve(2)
    assert client.state.current_channel == 1  # Clear does not retune the radio.
    client.measure_replies = [('no_signal', None)]
    assert policy._share_after_clear()
    assert policy.shared_known_observation_log[-1]['actual_cost_s'] == 5.


@pytest.mark.parametrize('condition,reason', [
    ('blocked', 'blocked'), ('near', 'already_ready'), ('attempted', 'target_already_attempted'),
    ('observed', 'position_already_observed'), ('empty', 'no_canonical_positive_region')])
def test_ineligible_candidate_is_logged_and_not_measured(monkeypatch, condition, reason):
    policy, client = make(monkeypatch); prime_target(policy, client); trigger(policy, client)
    if condition == 'blocked': policy.blocked.add(1)
    elif condition == 'near': policy.near_points[1] = Position(0., 100.)
    elif condition == 'attempted': policy.shared_attempted_channels.add(1)
    elif condition == 'observed': policy.observed_positions[1].add((0., 100.))
    else: policy.regions[1].vertices = ()
    before = len(client.calls)
    assert not policy._share_after_clear()
    assert policy.shared_known_observation_log[-1]['candidates'][0]['reason'] == reason
    assert len(client.calls) == before


def test_freshness_uses_six_decimal_position_key(monkeypatch):
    policy, client = make(monkeypatch); prime_target(policy, client)
    client.measure_replies = [('no_signal', None)]
    policy._perform('measure', Position(0.0000004, 100.), 1, 'constructed_observation')
    trigger(policy, client)
    assert not policy._share_after_clear()
    assert policy.shared_known_observation_log[-1]['candidates'][0]['previously_observed']


@pytest.mark.parametrize('distance,screened', [(5., True), (5.000001, False), (20., False),
                                                (1500., False), (1500.000001, True)])
def test_nominal_physical_thresholds_are_five_and_1500_not_twenty(monkeypatch, distance, screened):
    policy, client = make(monkeypatch)
    # Public constructed polygon isolates distance screening; no simulated source.
    region = CandidateRegion()
    region.vertices = ((distance-30., -1.), (distance+30., -1.),
                       (distance+30., 1.), (distance-30., 1.))
    region.observations = [BearingObservation((-1000., 0.), 0.)]
    region._circle = Circle((distance, 0.), math.hypot(30., 1.))
    policy.regions[1] = region; policy.detected.add(1)
    item = policy._shared_candidates()[0]
    assert (item['reason'] == 'nominal_distance_outside_range') == screened
    assert (item['nominal_bearing_deg'] is None) == screened


def test_ratio_ranking_and_channel_tie_are_stable(monkeypatch):
    policy, client = make(monkeypatch)
    prime_target(policy, client, 3); prime_target(policy, client, 1)
    trigger(policy, client, 2)
    items = policy._shared_candidates()
    assert [e['channel'] for e in items] == [1, 3]
    assert items[0]['ratio'] == items[1]['ratio']
    client.measure_replies = [('no_signal', None)]
    assert policy._share_after_clear()
    assert policy.shared_known_observation_log[-1]['selected_channel'] == 1


def test_parent_early_finally_closes_before_share(monkeypatch):
    policy, client = make(monkeypatch, points=((100., 0.),))
    prime_target(policy, client, 1); prime_target(policy, client, 2)
    client.measure_replies = [('no_signal', None)]
    policy._perform('measure', Position(0., 100.), 20, 'constructed_observation')
    assert policy._early_candidate(policy.points[0]) is not None
    client.clear_replies = ['success', 'success']
    policy._execute_plan()
    event = policy.shared_known_observation_log[0]
    assert event['early_service_index'] == 0
    assert policy.early_service_log[0]['end_actual_action_count'] == event['after_actual_action_count']
    assert event['parent_end_action_count'] == event['after_actual_action_count']
    assert policy.service_deadline is None and policy._joint_context is None


def test_known16_ends_before_final_trigger_and_never_scans(monkeypatch):
    policy, client = make(monkeypatch, points=((100., 0.),))
    for channel in range(1, 17): prime_clearable(policy, client, channel, (0., 0.))
    client.clear_replies = ['success'] * 16
    with pytest.raises(_StopSearch, match='source_count_upper_bound'): policy._execute_plan()
    assert len(policy.shared_known_observation_log) == 15
    assert all(e['status'] == 'no_candidate' for e in policy.shared_known_observation_log)
    assert not policy.shared_attempted_channels and policy.report.coverage_points_visited == 0



def test_six_distinct_shares_have_no_invented_four_batch_cap(monkeypatch):
    policy, client = make(monkeypatch)
    for channel in range(1, 7): prime_target(policy, client, channel)
    for trigger_channel in range(11, 17):
        trigger(policy, client, trigger_channel)
        client.measure_replies = [('no_signal', None)]
        assert policy._share_after_clear()
    events = policy.shared_known_observation_log
    assert [e['selected_channel'] for e in events] == list(range(1, 7))
    assert len({e['trigger_channel'] for e in events}) == 6
    assert sum(e['actual_cost_s'] for e in events) == 36.
    assert len([a for a in policy.report.action_history if a['phase'] == PHASE]) == 6
    assert policy.report.coverage_points_visited == 0


def test_rejected_parent_clear_never_creates_a_share_trigger(monkeypatch):
    policy, client = make(monkeypatch); prime_target(policy, client); prime_clearable(policy, client)
    client.reject_kind = 'clear'
    with pytest.raises(_StopSearch): policy._execute_plan()
    assert not policy.shared_known_observation_log and not policy.cleared
    assert policy.joint_resolvers[-1]['status'] == 'interrupted'
    assert not any(a['phase'] == PHASE for a in policy.report.action_history)


def test_actual_non_nominal_direction_can_empty_region_without_fake_readiness(monkeypatch):
    policy, client = make(monkeypatch); prime_target(policy, client); trigger(policy, client)
    client.measure_replies = [('direction', 90.)]  # Adversarial constructed reply.
    assert policy._share_after_clear()
    e = policy.shared_known_observation_log[-1]
    assert e['candidates'][0]['predicted_radius_m'] <= 19.9
    assert e['actual_radius_after_m'] is None and not e['actual_ready_after']
    assert not policy.regions[1].vertices and policy.cleared == {2}


def test_parent_action_and_safety_implementations_unchanged():
    for name in ('_perform', '_resolve', '_clear', '_next_probe', '_scan', '_ready', '_target',
                 '_early_candidate', '_early_service', '_check_budget', '_insertion_budget', '_scan_matrix', 'run'):
        assert getattr(Q4SharedKnown, name) is getattr(Q4KnownSource, name)
    child = ast.parse(textwrap.dedent(inspect.getsource(Q4SharedKnown._execute_plan)))
    parent = ast.parse(textwrap.dedent(inspect.getsource(Q4KnownSource._execute_plan)))
    loop = child.body[0].body[1]
    assert isinstance(loop.body[2], ast.If) and ast.unparse(loop.body[2].test) == 'self._share_after_clear()'
    del loop.body[2]
    for node in ast.walk(child):
        if isinstance(node, ast.Name) and node.id == 'MATRIX_CONFIG': node.id = 'CONFIG'
    assert ast.dump(child) == ast.dump(parent)


@pytest.mark.parametrize('kwargs', [{'problem': 3}, {'problem': True}, {'config': 'all_known_matrix'},
    {'max_actions': True}, {'max_active_probes': 31}, {'max_expansions': -1}])
def test_invalid_entrypoint_before_client(kwargs):
    with pytest.raises(ValueError): run_q4_shared_known(object(), **kwargs)
