"""Pure single-source geometry and frozen-tail tests; no case generation."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math

import pytest

from localization import CandidateRegion
from planning.chain_route import ChainSource, solve_chain_route
from planning.q4_conditional_belief import (build_belief, ConditionalBelief, SpatialNode,
    RadiusSegment, RadioObservation)
import planning.q4_cover_feedback as model
from simulator_client.state import Position


def source(channel=2):
    region = CandidateRegion()
    prefix = []
    for p, b in [((-1000.,0.),0.),((0.,-1000.),90.)]:
        region.observe(p,b)
        prefix.append(dict(action='measure',channel=channel,position=list(p),result='direction',bearing_deg=b))
    return dict(channel=channel,region=region,near=None,prefix=prefix)


def arguments(q=(100.,100.), tail=((300.,0.),)):
    c = dict(channel=1,region=None,near=Position(50.,0.),prefix=[])
    other = source()
    covers = [Position.coerce(q),*[Position.coerce(p) for p in tail]]
    result = solve_chain_route(covers,[ChainSource(c['near'],5.)],(0.,0.),
        max_expansions=200,background_scan_s=114.,scan_source_s=0.)
    return dict(current=Position(0.,0.),covers=covers,ready_channels=[1],ready_positions=[c['near']],
        sources=[c,other],cleared_count=0,selected_channel=1,incumbent_cost_s=result.cost_s)


def simple_belief(kind='omni'):
    node = SpatialNode((0.,0.),(1000.,1500.) if kind=='omni' else None,
        () if kind=='omni' else (RadiusSegment(1000.,1500.,((0.,math.pi/2.),)),),1.)
    return ConditionalBelief((node,),(1.,),(),1. if kind=='omni' else 0.,1.005,1.,1,262144)


@pytest.mark.parametrize('channel',[1,2,20])
def test_four_strata_and_channel_phases(channel):
    for dimension in range(5):
        values = [model.quantile(h,channel,dimension) for h in range(4)]
        assert all(0 <= u < 1 for u in values)
        sorted_values=sorted(values)
        assert all(b-a==pytest.approx(.25) for a,b in zip(sorted_values,sorted_values[1:]))
    assert model.quantile(0,channel,0) != model.quantile(0,channel,1)


@pytest.mark.parametrize('kind',['omni','directional'])
def test_samples_stay_in_radius_and_orientation_volume(kind):
    belief = simple_belief(kind)
    for h in range(4):
        event = model.sample_feedback(belief,(10.,10.),2,h)
        latent = event['latent']
        assert 1000 <= latent['radius_m'] <= 1500
        assert latent['source_type']==kind
        assert event['result']=='direction'
        true_bearing=225.
        delta=(event['bearing_deg']-true_bearing+180)%360-180
        assert abs(delta)<=1.+1e-12
        if kind=='directional':
            assert 0 <= latent['orientation_rad'] <= math.pi/2


def test_joint_radius_orientation_components_not_independent_marginals():
    segments=(RadiusSegment(1000.,1100.,((0.,.1),)),RadiusSegment(1200.,1500.,((3.,4.),)))
    node=SpatialNode((0.,0.),None,segments,1.)
    belief=replace(simple_belief('directional'),nodes=(node,))
    for channel in range(1,21):
        for world in range(4):
            latent=model.sample_feedback(belief,(20.,20.),channel,world)['latent']
            segment=segments[latent['component_index']]
            assert segment.lower<=latent['radius_m']<=segment.upper
            assert any(a<=latent['orientation_rad']<=b for a,b in segment.angles)


@pytest.mark.parametrize('result,bearing',[('no_signal',None),('near',None),('direction',123.45)])
def test_fixed_point_uses_actual_reply(result,bearing):
    belief=replace(simple_belief(),prefix=(RadioObservation((10.,10.),result,bearing),))
    for h in range(4):
        assert model.sample_feedback(belief,(10.,10.),2,h)==dict(channel=2,result=result,
            bearing_deg=bearing,origin='locked_actual_feedback',latent=None)


def test_feedback_only_updates_private_region_never_uses_latent():
    item=source(); before=deepcopy(item['region'].__dict__)
    feedback=dict(result='direction',bearing_deg=225.,latent={'position':[1e6,-1e6]})
    updated=model.apply_feedback(item,(100.,100.),feedback)
    other=model.apply_feedback(item,(100.,100.),dict(feedback,latent={'position':[0.,0.]}))
    assert item['region'].__dict__==before
    assert updated['region'] is not item['region']
    assert model.describe_source(updated)==model.describe_source(other)
    assert updated['region'].vertices != item['region'].vertices
    assert model.apply_feedback(item,(100.,100.),{'result':'no_signal'})['region'].__dict__==before
    assert model.describe_source(model.apply_feedback(item,(5.,6.),{'result':'near'}))['target']==[5.,6.]


def test_actual_belief_full_evaluation_is_json_and_preserves_inputs():
    args=arguments(); before=[deepcopy(s['region'].__dict__) if s['region'] else None for s in args['sources']]
    event=model.evaluate_cover_feedback(**args)
    assert event['status']=='scored', event.get('fallback_reason')
    assert len(event['solve_log'])==11 and event['expanded']<=2200
    assert event['information_changed']
    assert [r['role'] for r in event['solve_log']]==[
        'forced_q','V0_plus','V0_minus',*[f'world{h}_{side}' for h in range(4) for side in ('plus','minus')]]
    assert event['D_s']==pytest.approx(event['D0_s']+sum(event['world_marginals_s'])/4-event['base_marginal_s'])
    json.dumps(event,allow_nan=False)
    for s,old in zip(args['sources'],before):
        assert (s['region'].__dict__ if s['region'] else None)==old
    for call in event['solve_log'][1:]:
        assert call['channels']==([1,2] if call['role'].endswith('plus') else [2])
        assert call['background_scan_s']==[108.]
        if call['role'].endswith('plus'):
            assert call['source_scan_s'][0][0]==0.
            assert call['release_indices'][0]==0
    # Every task coordinate is from the disclosed feedback-updated geometry.
    for h,w in enumerate(event['worlds']):
        inputs={d['channel']:d['target'] for d in w['updated_sources']}
        for call in event['solve_log'][3+2*h:5+2*h]:
            assert call['positions']==[inputs[ch] for ch in call['channels']]


def test_locked_existing_measure_cannot_trigger_numerical_information():
    event=model.evaluate_cover_feedback(**arguments(q=(-1000.,0.)))
    assert event['status']=='fallback' and event['fallback_reason']=='no_information_change'
    assert not event['recommend_veto'] and len(event['solve_log'])==1
    assert all(w['changed_channels']==[] for w in event['worlds'])


def test_uninformative_silence_keeps_original(monkeypatch):
    monkeypatch.setattr(model,'sample_feedback',lambda belief,q,c,h:dict(channel=c,result='no_signal',
        bearing_deg=None,origin='conditional_volume',latent=None))
    event=model.evaluate_cover_feedback(**arguments())
    assert event['fallback_reason']=='no_information_change' and not event['recommend_veto']


def test_expansion_cap_and_model_exhaustion_do_not_throw_away_incumbent(monkeypatch):
    event=model.evaluate_cover_feedback(**arguments(),expansion_allowance=199)
    assert event['expanded']==0 and event['fallback_reason']=='prediction_expansion_budget'
    def unavailable(*a,**kw): raise model.ModelUnavailable('constructed_empty_volume')
    monkeypatch.setattr(model,'build_belief',unavailable)
    event=model.evaluate_cover_feedback(**arguments())
    assert event['fallback_reason']=='constructed_empty_volume' and not event['recommend_veto']
    assert len(event['solve_log'])==1


def test_centering_and_strict_veto_rule_with_controlled_feasible_cost_container(monkeypatch):
    original=model.solve_release_chain_route
    def scored(costs):
        costs=iter(costs)
        def fake(**kwargs):
            result=original(**kwargs)
            return replace(result,cost_s=next(costs))
        monkeypatch.setattr(model,'solve_release_chain_route',fake)
        return model.evaluate_cover_feedback(**arguments())
    # Costs are a controlled score-unit test, not geometric optimality evidence.
    same=scored([100.,50.]+[100.,50.]*4)
    assert same['D_s']==pytest.approx(same['D0_s']) and not same['recommend_veto']
    better=scored([10000.,0.]+[100.,50.]*4)
    assert better['D_s']<0 and better['recommend_veto']


def test_no_remaining_tail_still_complete_service_models():
    event=model.evaluate_cover_feedback(**arguments(tail=()))
    assert event['status']=='scored'
    assert all(call['release_indices']==[0]*len(call['channels']) for call in event['solve_log'][1:])
