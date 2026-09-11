import copy
import math

import pytest

from localization import CandidateRegion
from experiments.audit_q4_clear_region import audit_clear_region_prefix, summarize_clear_region_audits


def record(actions, config='incoming'):
    h, wire, now, position, tuned = [], [], 0., (0., 0.), 1
    for action, channel, p, result, phase, bearing in actions:
        move = round(math.dist(position, p)/5.*1e6)/1e6
        now += move + (5+(channel != tuned) if action == 'measure' else 3+2*(result == 'success'))
        position = p
        if action == 'measure':
            tuned = channel
        a = dict(action=action, channel=channel, position=list(p), result=result,
                 phase=phase, virtual_time_s=now)
        response = dict(accepted=True, virtual_time_s=now)
        response['measure_result' if action == 'measure' else 'clear_result'] = result
        if bearing is not None:
            a['bearing_deg'] = response['svd_deg'] = bearing
        h.append(a)
        wire.append(dict(action='/'+action, channel=channel, position=list(p), response=response))
    return dict(row={'successful': False}, history=wire, summary=dict(action_history=h,
        coverage_points=[[200., 100.]], strategy_parameters=dict(q4_clear_region=config,
        clear_region_log=[], chain_route_log=[], early_service_log=[])))


def attach(r, *, original=(100., 0.), selected=(90., 0.), n=2,
           phase='near_clear', anchor=None, source=None, status=None, vertices=None, executed=True):
    h = r['summary']['action_history'];params = r['summary']['strategy_parameters']
    config = params['q4_clear_region'];current = tuple(h[n-1]['position'])
    old_in, new_in = math.dist(current, original), math.dist(current, selected)
    old_obj = old_in+(math.dist(original, anchor) if anchor is not None else 0)
    new_obj = new_in+(math.dist(selected, anchor) if anchor is not None else 0)
    if vertices is None:
        cert = dict(kind='near_disk', near_point=list(original), search_radius_m=14.99998,
                    verification_radius_m=14.99999, max_distance_m=math.dist(original, selected), passed=True)
    else:
        cert = dict(kind='polygon_vertices', vertices=[list(p) for p in vertices], search_radius_m=19.99998,
                    verification_radius_m=19.99999, max_distance_m=max(math.dist(p, selected) for p in vertices), passed=True)
    geometry = dict(certificate=cert, original_incoming_m=old_in, selected_incoming_m=new_in,
        original_objective_m=old_obj, selected_objective_m=new_obj, proxy_saved_s=(old_obj-new_obj)/5.,
        objective='two_segment' if anchor else 'incoming', method='finite_rays' if anchor else 'boundary_candidates',
        status='optimized' if original != selected else 'original_best', ray_count=65 if anchor else 0,
        candidate_count=3, runtime_s=.001)
    event = dict(after_actual_action_count=n, end_action_count=n+int(executed), channel=1, phase=phase,
        config=config, original_position=list(original), current_position=list(current),
        selected_position=list(selected), anchor=list(anchor) if anchor else None, anchor_source=source,
        anchor_status=status or ('incoming_configuration' if config == 'incoming' else 'missing'),
        geometry=geometry, executed=executed, runtime_s=1.25)
    if executed:
        event.update(actual_position=list(selected), result=h[n]['result'])
    params['clear_region_log'].append(event)
    return event


def simple(config='incoming'):
    r = record([('measure',1,(100.,0.),'near','active_localization',None),
                ('measure',2,(0.,0.),'no_signal','active_localization',None),
                ('clear',1,(90.,0.),'success','near_clear',None)], config)
    attach(r)
    return r


def test_observed_near_safe_incoming_and_pure_planner_time():
    r = simple();before = copy.deepcopy(r)
    a = audit_clear_region_prefix(r)
    assert a['events'] == a['executed_events'] == a['changed_positions'] == 1
    assert a['proxy_saved_s'] == 2.
    assert a['decision_wall_s'] == .001  # Excludes physical request waiting in event runtime.
    assert r == before
    assert summarize_clear_region_audits([a, a])['executed_events'] == 2


@pytest.mark.parametrize('bad', ['unsafe', 'old', 'current', 'certificate', 'fee', 'executed',
                                'unlogged', 'wire', 'runtime', 'method'])
def test_inconsistent_evidence_rejected(bad):
    r = simple();p = r['summary']['strategy_parameters'];e = p['clear_region_log'][0]
    if bad == 'unsafe':e['selected_position'] = [70.,0.]
    elif bad == 'old':e['original_position'] = [101.,0.]
    elif bad == 'current':e['current_position'] = [1.,0.]
    elif bad == 'certificate':e['geometry']['certificate']['near_point'] = [99.,0.]
    elif bad == 'fee':e['geometry']['proxy_saved_s'] += 1.
    elif bad == 'executed':e['executed'] = False
    elif bad == 'unlogged':p['clear_region_log'] = []
    elif bad == 'wire':r['history'][0]['response']['measure_result'] = 'no_signal'
    elif bad == 'runtime':e['geometry']['runtime_s'] = float('nan')
    else:e['geometry']['method'] = 'finite_rays'
    with pytest.raises(ValueError):audit_clear_region_prefix(r)


def test_route_cover_and_source_anchors_are_actual_second_task():
    for kind in ('cover','source'):
        actions = [('measure',1,(100.,0.),'near','active_localization',None)]
        if kind == 'source':actions.append(('measure',2,(200.,100.),'near','active_localization',None))
        actions += [('measure',3,(0.,0.),'no_signal','active_localization',None),
                    ('clear',1,(90.,0.),'success','near_clear',None)]
        r = record(actions,'anchored');n = len(actions)-1;p = r['summary']['strategy_parameters']
        task = [kind, 0 if kind == 'cover' else 1]
        p['chain_route_log'] = [dict(after_actual_action_count=n, selected_kind='source', selected_channel=1,
            source_channels=[1,2] if kind == 'source' else [1], source_positions=[[100.,0.],[200.,100.]],
            remaining_covers=[[200.,100.]], result={'order':[['source',0],task]})]
        source = dict(kind='route_second_task', route_log_index=0, prefix=n, channel=1, task=task)
        attach(r,n=n,anchor=(200.,100.),source=source,status='valid')
        assert audit_clear_region_prefix(r)['executed_events'] == 1
        bad = copy.deepcopy(r);bad['summary']['strategy_parameters']['clear_region_log'][0]['anchor'][0] += 1
        with pytest.raises(ValueError, match='Anchor differs'):audit_clear_region_prefix(bad)
        bad = copy.deepcopy(r);bad['summary']['strategy_parameters']['chain_route_log'][0]['after_actual_action_count'] -= 1
        with pytest.raises(ValueError, match='stale'):audit_clear_region_prefix(bad)


def test_expired_route_cannot_supply_two_segment_objective():
    r = record([('measure',1,(100.,0.),'near','active_localization',None),
                ('measure',2,(0.,0.),'no_signal','active_localization',None),
                ('measure',1,(50.,0.),'no_signal','active_localization',None),
                ('clear',1,(90.,0.),'success','near_clear',None)], 'anchored')
    p = r['summary']['strategy_parameters']
    p['chain_route_log'] = [dict(after_actual_action_count=2, selected_kind='source', selected_channel=1)]
    attach(r,n=3,status='route_prefix_expired')
    assert audit_clear_region_prefix(r)['events'] == 1
    p['clear_region_log'][0]['anchor'] = [200.,100.]
    with pytest.raises(ValueError):audit_clear_region_prefix(r)


def test_polygon_reconstructed_from_two_actual_bearings():
    region = CandidateRegion()
    for p,b in [((-400.,0.),0.),((100.,-500.),90.)]:region.observe(p,b)
    disk = region.enclosing_disk();assert disk.radius < 19.9
    original = tuple(disk.center)
    r = record([('measure',1,(-400.,0.),'direction','active_localization',0.),
                ('measure',1,(100.,-500.),'direction','active_localization',90.),
                ('clear',1,original,'success','certified_clear',None)])
    attach(r,original=original,selected=original,phase='certified_clear',vertices=region.vertices)
    assert audit_clear_region_prefix(r)['changed_positions'] == 0
    r['summary']['strategy_parameters']['clear_region_log'][0]['geometry']['certificate']['vertices'][0][0] += .1
    with pytest.raises(ValueError, match='geometry differs'):audit_clear_region_prefix(r)


def service_record():
    r = record([('measure',1,(-900.,0.),'direction','active_localization',0.),
                ('measure',1,(100.,-1000.),'direction','active_localization',90.),
                ('measure',20,(-150.,-50.),'no_signal','coverage',None),
                ('measure',1,(100.,0.),'near','active_localization',None)], 'anchored')
    s = r['summary'];p = s['strategy_parameters'];s['coverage_points'] = [[-150.,-50.],[200.,0.]]
    region = CandidateRegion();region.observe((-900.,0.),0.);region.observe((100.,-1000.),90.)
    disk = region.enclosing_disk();start=(-150.,-50.);next_cover=(200.,0.)
    p.update(q4_r2_scheduling='onroute',discovery_stop_log=[],early_service_log=[dict(channel=1,
        after_actual_action_count=3,end_actual_action_count=4,radius_m=disk.radius,
        detour_m=math.dist(start,disk.center)+math.dist(disk.center,next_cover)-math.dist(start,next_cover),
        budget_s=60.,actual_cost_s=s['action_history'][3]['virtual_time_s']-s['action_history'][2]['virtual_time_s'],
        interrupted=True,cleared=False)])
    source=dict(kind='early_service_next_cover',candidate_prefix=3,channel=1,next_cover_position=[200.,0.])
    attach(r,n=4,original=(100.,0.),selected=(100.,0.),anchor=(200.,0.),source=source,status='valid',executed=False)
    return r


def test_real_early_prefix_and_60s_gate_justify_unexecuted_clear():
    r = service_record();a = audit_clear_region_prefix(r)
    assert a['events'] == 1 and a['executed_events'] == a['proxy_saved_s'] == 0
    bad = copy.deepcopy(r);bad['summary']['strategy_parameters']['clear_region_log'][0]['anchor_source']['candidate_prefix'] = 2
    with pytest.raises(ValueError):audit_clear_region_prefix(bad)
    # Same valid service geometry, but force a shorter elapsed near action:
    # the next clear now fits the slice, so "interrupted" is not sufficient evidence.
    bad = copy.deepcopy(r);bad['summary']['action_history'][-1]['virtual_time_s'] -= 10
    bad['history'][-1]['response']['virtual_time_s'] -= 10
    bad['summary']['strategy_parameters']['early_service_log'][0]['actual_cost_s'] -= 10
    with pytest.raises(ValueError, match='fits actual service'):audit_clear_region_prefix(bad)


def test_terminal_omission_needs_reason_and_does_not_count_proposed_gain():
    r = simple();e = r['summary']['strategy_parameters']['clear_region_log'][0]
    r['summary']['action_history'].pop();r['history'].pop()
    e.update(executed=False,end_action_count=2);e.pop('actual_position');e.pop('result')
    with pytest.raises(ValueError, match='No actual service'):audit_clear_region_prefix(r)
    r['summary']['completion_reason'] = 'action_budget'
    audited = audit_clear_region_prefix(r)
    assert audited['proxy_saved_s'] == audited['changed_positions'] == 0


def test_fallback_preserves_safe_original():
    r = simple();e = r['summary']['strategy_parameters']['clear_region_log'][0]
    e['selected_position'] = e['actual_position'] = [100.,0.]
    r['summary']['action_history'][-1]['position'] = [100.,0.]
    r['history'][-1]['position'] = [100.,0.]
    r['summary']['action_history'][-1]['virtual_time_s'] += 2.
    r['history'][-1]['response']['virtual_time_s'] += 2.
    e['geometry'] = dict(status='fallback',certificate={'passed':False},proxy_saved_s=0.,runtime_s=.002)
    assert audit_clear_region_prefix(r)['fallbacks'] == 1


def test_no_hidden_fields_or_optimizer_reentry(monkeypatch):
    class ObservationsOnly(dict):
        def __getitem__(self, key):
            assert key in {'summary','row','history'}
            return super().__getitem__(key)
    import planning.certified_clear_region as planner
    monkeypatch.setattr(planner,'choose_certified_clear_point',lambda *a,**k:pytest.fail('Auditor called optimizer'))
    assert audit_clear_region_prefix(ObservationsOnly(simple()))['passed']


@pytest.mark.parametrize('config', ['incoming', 'anchored'])
def test_actual_inherited_controller_event_is_compatible(monkeypatch, config):
    # Actual policy/controller with scripted measured replies; no Scenario or
    # simulator execution. Audit remains forbidden to call the optimizer.
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('clear_region_scripted_fixture',
             Path(__file__).with_name('test_q4_clear_region.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    policy, client = fixture.make(monkeypatch, config)
    center = fixture.prime_region(policy, client)
    if config == 'anchored':
        policy.route_log.append(dict(after_actual_action_count=2, selected_kind='source', selected_channel=1,
            source_channels=[1], source_positions=[[center.x, center.y]],
            remaining_covers=[[p.x, p.y] for p in policy.points],
            result={'order': [['source', 0], ['cover', 0]]}))
    assert policy._resolve(1)
    summary = policy.report.as_dict();wire = []
    for a in summary['action_history']:
        response = {'accepted': True, 'virtual_time_s': a['virtual_time_s'],
                    'measure_result' if a['action'] == 'measure' else 'clear_result': a['result']}
        if a['result'] == 'direction':response['svd_deg'] = a['bearing_deg']
        wire.append(dict(action='/'+a['action'],channel=a['channel'],position=a['position'],response=response))
    import planning.certified_clear_region as planner
    monkeypatch.setattr(planner,'choose_certified_clear_point',lambda *a,**k:pytest.fail('Audit reentered optimizer'))
    assert audit_clear_region_prefix({'summary':summary,'history':wire,'row':{'successful':False}})['executed_events'] == 1
