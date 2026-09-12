"""R39 accounting, conditional-world evidence and actual-prefix tampering."""
import copy
from dataclasses import asdict
from itertools import permutations
import math

import pytest

from experiments import audit_q4_cover_feedback as audit
from planning.chain_route import ChainSource
from planning.release_chain_route import solve_release_chain_route


def model_problem():
    return dict(covers=[(10.,0.),(20.,0.)],positions=[(5.,3.),(12.,2.),(22.,-2.)],
        start=(0.,0.),background=[12.,6.],matrix=[[6.,0.,6.],[0.,0.,6.]],releases=[1,0,2])


def result(problem,budget=200):
    return asdict(solve_release_chain_route(problem['covers'],[ChainSource(p,service_s=5.) for p in problem['positions']],
        problem['start'],background_scan_s=problem['background'],source_scan_s=problem['matrix'],
        release_indices=problem['releases'],max_expansions=budget))


def test_frozen_release_cost_and_independent_small_recurrence_match_exhaustive_orders():
    p=model_problem();costs=[]
    tasks=[('cover',j) for j in range(2)]+[('source',j) for j in range(3)]
    for order in permutations(tasks):
        try:costs.append(audit.route_value(p['covers'],p['positions'],p['start'],order,p['background'],p['matrix'],p['releases']))
        except ValueError:pass
    expected=min(costs)
    assert audit.small_release_optimum(**p)==pytest.approx(expected)
    actual=result(p)
    assert actual['cost_s']==pytest.approx(expected)
    assert audit.validate_result(actual,**p)==actual['expanded']


@pytest.mark.parametrize('change',['early_source','missing_cover','missing_source','cost','lower','service','expanded'])
def test_incomplete_or_false_solver_evidence_rejected(change):
    p=model_problem();r=result(p)
    if change=='early_source':r['order']=[('source',0)]+[x for x in r['order'] if x!=('source',0)]
    elif change=='missing_cover':r['order']=[x for x in r['order'] if x!=('cover',1)]
    elif change=='missing_source':r['order']=[x for x in r['order'] if x!=('source',2)]
    elif change=='cost':r['cost_s']-=1.
    elif change=='lower':r['lower_bound_s']=r['cost_s']+1.
    elif change=='service':r['cost_s']-=15.
    elif change=='expanded':r['expanded']=201
    with pytest.raises((ValueError,IndexError)):
        audit.validate_result(r,**p)


def test_no_source_tail_preserves_full_background_and_cover_chain():
    p=dict(covers=[(10.,0.),(20.,0.)],positions=[],start=(0.,0.),background=[12.,6.],matrix=[[],[]],releases=[])
    r=result(p)
    assert r['cost_s']==22.
    assert audit.validate_result(r,**p)==r['expanded']


def prediction_arguments(q=(100.,100.),tail=((300.,0.),)):
    from localization import CandidateRegion
    from planning.chain_route import solve_chain_route
    r=CandidateRegion();history=[]
    for p,b in [((-1000.,0.),0.),((0.,-1000.),90.)]:
        r.observe(p,b);history.append(dict(action='measure',channel=2,position=list(p),result='direction',bearing_deg=b))
    sources=[dict(channel=1,region=None,near=(50.,0.),prefix=[]),dict(channel=2,region=r,near=None,prefix=history)]
    covers=[q,*tail];inc=solve_chain_route(covers,[ChainSource((50.,0.),5.)],(0.,0.),background_scan_s=114.,scan_source_s=0.)
    return dict(current=(0.,0.),covers=covers,ready_channels=[1],ready_positions=[(50.,0.)],sources=sources,
        cleared_count=0,c=1,incumbent=inc.cost_s,allowance=70400)


def production_prediction(args):
    from planning.q4_cover_feedback import evaluate_cover_feedback
    return evaluate_cover_feedback(**{('selected_channel' if k=='c' else 'incumbent_cost_s' if k=='incumbent'
        else 'expansion_allowance' if k=='allowance' else k):v for k,v in args.items()})


@pytest.mark.parametrize('case',['full','no_tail','locked','budget'])
def test_independent_shared_world_and_complete_tail_replay(case):
    args=prediction_arguments(q=(-1000.,0.) if case=='locked' else (100.,100.),tail=() if case=='no_tail' else ((300.,0.),))
    if case=='budget':args['allowance']=199
    before=copy.deepcopy(args['sources'][1]['region'].__dict__)
    expected=audit.reference_prediction(**args);actual=production_prediction(args)
    audit.same(audit.strip_runtime(actual),audit.strip_runtime(expected),'Prediction reproduction')
    assert args['sources'][1]['region'].__dict__==before
    if case in {'full','no_tail'}:
        assert actual['status']=='scored' and len(actual['solve_log'])==11
        assert actual['D_s']==pytest.approx(actual['D0_s']+math.fsum(actual['world_marginals_s'])/4.-actual['base_marginal_s'])
        for step in actual['solve_log'][1:]:
            assert step['service_s']==5.
            if step['role'].endswith('plus'):
                assert all(row[0]==0 for row in step['source_scan_s'])
    else:assert actual['status']=='fallback' and not actual['recommend_veto']
    if case=='locked':
        assert len(actual['solve_log'])==1 and not actual['information_changed']
        assert all(not w['changed_channels'] for w in actual['worlds'])


def test_silent_geometry_and_latent_coordinates_do_not_leak_into_targets():
    args=prediction_arguments();s=args['sources'][1];before=copy.deepcopy(s['region'].__dict__)
    silent=audit.update_hypothesis(s,(100.,100.),dict(result='no_signal',latent=dict(position=[1e9,1e9])))
    assert audit.source_description(silent)==audit.source_description(s)
    x=audit.update_hypothesis(s,(100.,100.),dict(result='direction',bearing_deg=225.,latent=dict(position=[1e9,1e9])))
    y=audit.update_hypothesis(s,(100.,100.),dict(result='direction',bearing_deg=225.,latent=dict(position=[0.,0.])))
    assert audit.source_description(x)==audit.source_description(y)
    assert s['region'].__dict__==before
    assert audit.source_description(audit.update_hypothesis(s,(5.,6.),dict(result='near')))['target']==[5.,6.]


@pytest.mark.parametrize('kind',['omni','directional'])
def test_joint_radius_orientation_sampler_and_near_reception_boundary(kind):
    from planning.q4_conditional_belief import ConditionalBelief,SpatialNode,RadiusSegment
    from planning.q4_cover_feedback import sample_feedback
    node=SpatialNode((0.,0.),(1000.,1500.) if kind=='omni' else None,
        () if kind=='omni' else (RadiusSegment(1000.,1100.,((0.,.1),)),RadiusSegment(1200.,1500.,((3.,4.),))),1.)
    b=ConditionalBelief((node,),(1.,),(),1. if kind=='omni' else 0.,1.005,1.,1,262144)
    for channel in range(1,21):
        for h in range(4):
            for q in [(5.,0.),(5.0001,0.),(-10.,0.),(1600.,0.)]:
                r=audit.sampled_feedback(b,q,channel,h)
                assert r==sample_feedback(b,q,channel,h)
                latent=r['latent']
                if kind=='directional':
                    seg=node.directional_segments[latent['component_index']]
                    assert seg.lower<=latent['radius_m']<=seg.upper
                    assert any(lo<=latent['orientation_rad']<=hi for lo,hi in seg.angles)
                if q==(1600.,0.):assert r['result']=='no_signal'
                if r['result']=='near':assert math.dist(q,(0.,0.))<=5.
                if r['result']=='direction':
                    true=math.degrees(math.atan2(-q[1],-q[0]))%360.
                    assert abs((r['bearing_deg']-true+180.)%360.-180.)<=1.+1e-12


def actual_record(monkeypatch,*,veto=False,partial=None,reject=False):
    """Whole controller with scripted observations, no scenario or hidden truth.

    Two test stations are not a whole-domain coverage certificate. Prefix
    inherited safety is real; generic whole-domain validation is tested apart.
    Forced-veto score stubbing isolates ownership from the independently tested
    real conditional model; it is not performance or model evidence.
    """
    from strategies.q4_cover_feedback import Q4CoverFeedback
    from strategies.search import _StopSearch
    from simulator_client.state import Position
    from tests.test_q4_clear_before_probe import Replies
    from tests.test_audit_q4_clear_before_probe import wrap
    monkeypatch.setattr('planning.q4_directional_cover.certified_cover_points',lambda profile:
        ((Position(0.,0.),Position(100.,0.)),{'passed':True,'test_only':True}))
    client=Replies();max_actions=20000 if partial is None else 22+partial
    policy=Q4CoverFeedback(client,max_actions,6,max_expansions=200)
    policy.actions=1  # Accepted enter, also present in the wire wrapper.
    client.measure_replies=[('direction',0.),('near',None)]+[('no_signal',None)]*18
    # The final origin channel 20 stays tuned even after source 2 is cleared.
    client.measure_replies += [('no_signal',None),('near',None)]+[('no_signal',None)]*17
    client.clear_replies=['success']*2
    if veto:
        def prediction(*a):return dict(status='scored',information_changed=True,recommend_veto=True,expanded=7)
        monkeypatch.setattr(policy,'_prediction',prediction)
        def verify(log,*a):
            assert log==prediction()
            return True,7
        monkeypatch.setattr(audit,'check_prediction',verify)
    if reject:
        def reject_at_second_scan():
            if len(policy.report.action_history)>=20:client.reject_kind='measure'
        client.before_measure=reject_at_second_scan
    reason=None
    try:policy._execute_plan()
    except _StopSearch as error:reason=str(error)
    record=wrap(policy,entered=True,reason=reason)
    record['row']['strategy']=audit.LABEL
    record['spec']=dict(entrypoint=audit.ENTRY,kwargs=dict(config=audit.CONFIG,max_actions=max_actions,max_active_probes=6,max_expansions=200))
    if reject:
        _,p,c=client.calls[-1]
        record['history'].append(dict(action='/measure',position=[p.x,p.y],channel=c,response={'accepted':False}))
    return record


@pytest.mark.parametrize('veto',[False,True])
def test_whole_real_coverage_and_resolver_prefix_replays_all_inherited_checks(monkeypatch,veto):
    record=actual_record(monkeypatch,veto=veto);before=copy.deepcopy(record)
    result=audit.audit_cover_feedback_prefix(record)
    assert result['passed'] and result['cover_feedback_vetoes']==int(veto)
    assert result['completed_scans']==2 and result['resolver_macros']==2
    assert all(result[k]['passed'] for k in ('r12','r8','range','scheduling'))
    assert record==before


@pytest.mark.parametrize('partial',[0,1,5])
def test_actual_budget_partial_scan_has_zero_complete_veto(monkeypatch,partial):
    record=actual_record(monkeypatch,veto=True,partial=partial)
    result=audit.audit_cover_feedback_prefix(record)
    assert result['passed'] and result['cover_feedback_vetoes']==0 and result['completed_scans']==1
    assert len(record['summary']['action_history'])==20+partial
    event=record['summary']['strategy_parameters']['cover_feedback_log'][0]
    assert event['status']=='interrupted' and not event['executed_cover']


def test_rejected_zero_action_veto_keeps_original_wire_and_inherited_safety(monkeypatch):
    record=actual_record(monkeypatch,veto=True,reject=True)
    original=copy.deepcopy(record)
    result=audit.audit_cover_feedback_prefix(record)
    assert result['cover_feedback_vetoes']==0 and result['completed_scans']==1 and record==original
    assert record['history'][-1]['response']['accepted'] is False


@pytest.mark.parametrize('change',['missing_event','orphan_event','role','route_id','route_cost','incumbent',
    'counter','expanded','known','position','channel','veto_channel','visited','end','actual_kind','measurement','early_limit'])
def test_whole_veto_prefix_tampering_cannot_hide_real_macro(monkeypatch,change):
    record=actual_record(monkeypatch,veto=True);p=record['summary']['strategy_parameters']
    e=p['cover_feedback_log'][0];r=p['chain_route_log'][0]
    if change=='missing_event':p['cover_feedback_log'].pop(0)
    elif change=='orphan_event':p['cover_feedback_log'].append(copy.deepcopy(e))
    elif change=='role':r['execution_role']='original_incumbent'
    elif change=='route_id':r['cover_feedback_event_id']=1
    elif change=='route_cost':r['result']['cost_s']-=1.
    elif change=='incumbent':e['incumbent']['selected_channel']=1
    elif change=='counter':e['actual_vetoes_after']=0
    elif change=='expanded':e['prediction_expanded_after']+=1
    elif change=='known':e['known_channels']=[1,2,3]
    elif change=='position':e['current_position'][0]+=1.
    elif change=='channel':e['channel']=1
    elif change=='veto_channel':e['vetoed_channels_after']=[1]
    elif change=='visited':e['coverage_visited_after']-=1
    elif change=='end':e['end_actual_action_count']-=1
    elif change=='actual_kind':e['executed_kind']='source'
    elif change=='measurement':record['summary']['action_history'][20]['position'][1]+=1.
    elif change=='early_limit':p['cover_feedback_limits']['actual_vetoes']=9
    with pytest.raises((ValueError,KeyError,AssertionError)):
        audit.audit_cover_feedback_prefix(record)


def real_prediction_prefix(monkeypatch):
    from tests.test_q4_cover_feedback import make,prime
    from tests.test_audit_q4_clear_before_probe import wrap
    from planning.chain_route import solve_chain_route
    p,client=make(monkeypatch);p.actions=1;prime(p,client)
    record=wrap(p,entered=True);prefix=audit.Prefix(record);n=len(prefix.h)
    covers=[(100.,100.),(300.,0.)];current=prefix.before[n][0]
    inc=solve_chain_route(covers,[ChainSource(prefix.target(n,1),5.)],current,background_scan_s=114.,scan_source_s=0.)
    args=dict(current=current,covers=covers,ready_channels=[1],ready_positions=[prefix.target(n,1)],
        sources=audit.actual_sources(prefix,n),cleared_count=0,c=1,incumbent=inc.cost_s,allowance=70400)
    return prefix,n,args,production_prediction(args)


def test_true_prefix_model_recomputes_all_eleven_shared_routes(monkeypatch):
    prefix,n,args,log=real_prediction_prefix(monkeypatch)
    veto,expanded=audit.check_prediction(log,prefix,n,args['covers'],[1],1,args['incumbent'],70400)
    assert len(log['solve_log'])==11 and expanded==sum(s['result']['expanded'] for s in log['solve_log'])
    assert veto==log['recommend_veto']


@pytest.mark.parametrize('change',['source_prefix','source_region','latent_position','latent_radius','world_feedback',
    'world_target','world_changed','baseline_marginal','D0','D','recommend','matrix','release','service','cost','expanded',
    'missing_solve','runtime'])
def test_real_prefix_model_tampering_cannot_invent_information_or_fee(monkeypatch,change):
    prefix,n,args,log=real_prediction_prefix(monkeypatch)
    if change=='source_prefix':log['input_sources'][1]['prefix'][0]['bearing_deg']+=1.
    elif change=='source_region':log['input_sources'][1]['vertices'][0][0]+=1.
    elif change=='latent_position':log['worlds'][0]['feedbacks'][1]['latent']['position'][0]+=1.
    elif change=='latent_radius':log['worlds'][0]['feedbacks'][1]['latent']['radius_m']+=1.
    elif change=='world_feedback':log['worlds'][0]['feedbacks'][1]['bearing_deg']+=.1
    elif change=='world_target':log['worlds'][0]['updated_sources'][1]['target'][0]+=1.
    elif change=='world_changed':log['worlds'][0]['changed_channels']=[]
    elif change=='baseline_marginal':log['base_marginal_s']+=1.
    elif change=='D0':log['D0_s']-=1.
    elif change=='D':log['D_s']-=1.
    elif change=='recommend':log['recommend_veto']=not log['recommend_veto']
    elif change=='matrix':log['solve_log'][1]['source_scan_s'][0][0]=6.
    elif change=='release':log['solve_log'][1]['release_indices'][1]=0
    elif change=='service':log['solve_log'][1]['service_s']=0.
    elif change=='cost':log['solve_log'][1]['result']['cost_s']-=1.
    elif change=='expanded':log['expanded']+=1
    elif change=='missing_solve':log['solve_log'].pop()
    elif change=='runtime':log['solve_log'][1]['result']['runtime_s']=-1.
    with pytest.raises(ValueError):audit.check_prediction(log,prefix,n,args['covers'],[1],1,args['incumbent'],70400)


def test_partial_scan_cannot_be_relabelled_complete_or_get_an_actual_veto(monkeypatch):
    record=actual_record(monkeypatch,veto=True,partial=5)
    e=record['summary']['strategy_parameters']['cover_feedback_log'][0]
    e.update(executed_cover=True,actual_vetoes_after=1,vetoed_channels_after=[2],coverage_visited_after=2,status='completed')
    with pytest.raises(ValueError):audit.audit_cover_feedback_prefix(record)


@pytest.mark.parametrize('child',['audit_joint_continuation_prefix','audit_clear_before_probe_prefix','audit_range_prefix','audit_scheduling_prefix'])
def test_inherited_false_cannot_silently_pass(monkeypatch,child):
    record=actual_record(monkeypatch)
    monkeypatch.setattr(audit,child,lambda record:{'passed':False})
    with pytest.raises(ValueError,match='Inherited'):audit.audit_cover_feedback_prefix(record)


def test_full_audit_preserves_original_failed_wire_and_requires_physical_gate(monkeypatch):
    record=actual_record(monkeypatch,veto=True,reject=True);seen=[]
    def physical(record):seen.append(record);return {'passed':True}
    monkeypatch.setattr('experiments.audit_q4_cover.audit_record',physical)
    result=audit.audit_full(record)
    assert result['passed'] and seen==[record] and seen[0]['history'][-1]['response']['accepted'] is False
    monkeypatch.setattr('experiments.audit_q4_cover.audit_record',lambda record:{'passed':False,'errors':['constructed']})
    with pytest.raises(ValueError,match='Generic physical'):audit.audit_full(record)


def test_frozen_source_contract_and_configuration_fail_closed(monkeypatch):
    assert audit.verify_source_contract()==58
    record=actual_record(monkeypatch)
    record['spec']['kwargs']['config']='another_config'
    with pytest.raises(ValueError,match='specification'):audit.audit_cover_feedback_prefix(record)
    monkeypatch.setitem(audit.NEW_SOURCE_CONTRACT,'src/strategies/q4_cover_feedback.py','0'*64)
    with pytest.raises(ValueError,match='Unreviewed source contract'):audit.verify_source_contract()
