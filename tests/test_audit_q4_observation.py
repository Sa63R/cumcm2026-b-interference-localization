import copy
import math
import pytest

from localization import CandidateRegion
from experiments.audit_q4_observation import audit_observation_prefix, audit_observation_records


def histories(actions):
    phase, wire, elapsed, current, tuned = [], [], 0.,(0.,0.),1
    for kind,c,p,result,bearing,name in actions:
        elapsed += round(math.dist(current,p)/5*1e6)/1e6 + (5. if kind == 'measure' or result == 'success' else 3.)
        if kind == 'measure':
            elapsed += tuned != c
            tuned = c
        current = p
        item = dict(action=kind,channel=c,position=list(p),result=result,phase=name,virtual_time_s=elapsed)
        response = dict(accepted=True,virtual_time_s=elapsed)
        response['measure_result' if kind == 'measure' else 'clear_result'] = result
        if bearing is not None:
            item['bearing_deg'] = response['svd_deg'] = bearing
        phase.append(item)
        wire.append(dict(action='/'+kind,channel=c,position=list(p),response=response))
    return phase,wire


def decision(n,current,baseline,selected,unique=1,tuned=1):
    scores=[]
    for p,tail in [(baseline,1000.),(selected,6.)]:
        branches = [dict(outcome=['no_signal'],mass=.5,particles=2,spatial_ess=2.,
            continuation=None,continuation_cost_s=tail-1.,reason='depth_leaf'),
            dict(outcome=['bearing',0],mass=.5,particles=2,spatial_ess=2.,
            continuation=None,continuation_cost_s=tail+1.,reason='depth_leaf')]
        scores.append(dict(position=p,cost_s=math.dist(current,p)/5+5+(tuned != 1)+tail,branches=branches))
    return dict(after_actual_action_count=n,channel=1,prefix_unique=unique,
        version='finite-observation-tree-v1',depth=1,effective_depth=1,baseline=baseline,selected=selected,
        fallback=False,reason='complete_tree',cpu_s=.01,decision_wall_s=.01,
        candidate_scores=scores,predicted_local_cost_s=scores[-1]['cost_s'])


def example():
    selected,baseline=(100.,10.),(100.,-10.)
    phase,wire=histories([('measure',1,(0.,0.),'direction',0.,'coverage'),
        ('measure',1,selected,'direction',1.,'active_localization')])
    params=dict(active_probe_algorithm='finite-observation-tree-v1',observation_depth=1,
                early_service_log=[],observation_tree_log=[decision(1,(0.,0.),baseline,selected)])
    return dict(history=wire,summary=dict(action_history=phase,strategy_parameters=params))


def test_audit_and_pooled_actual_change_rate_are_read_only():
    record=example();before=copy.deepcopy(record)
    result=audit_observation_prefix(record)
    assert result['executed_decisions']==result['executed_changed']==1
    assert record==before
    pooled=audit_observation_records([record,record])
    assert pooled['decisions']==2 and pooled['actual_probe_change_rate']==1.
    assert pooled['total_decision_wall_s']==pytest.approx(.02)


@pytest.mark.parametrize('fault',['prefix','negative_prefix','missing_log','channel','nonfinite','mass','negative_cost','argmin',
    'wire','action','unique','fallback','singleton','not_shared','cost_identity','omit_baseline'])
def test_corrupt_or_clairvoyant_trace_is_rejected(fault):
    record=example();params=record['summary']['strategy_parameters'];e=params['observation_tree_log'][0]
    branch=e['candidate_scores'][0]['branches'][0]
    if fault=='prefix':e['after_actual_action_count']=0
    elif fault=='negative_prefix':e['after_actual_action_count']=-1
    elif fault=='missing_log':params['observation_tree_log']=[]
    elif fault=='channel':e['channel']=2
    elif fault=='nonfinite':e['baseline']=[float('nan'),0.]
    elif fault=='mass':branch['mass']=.1
    elif fault=='negative_cost':branch['continuation_cost_s']=-1.
    elif fault=='argmin':e['selected']=e['baseline']
    elif fault=='wire':record['history'][0]['response']['svd_deg']=2.
    elif fault=='action':
        record['summary']['action_history'][1]['position']=[101.,10.]
        record['history'][1]['position']=[101.,10.]
    elif fault=='unique':e['prefix_unique']=2
    elif fault=='fallback':e['fallback']=True;e['reason']='cpu_budget'
    elif fault=='singleton':
        params['observation_depth']=e['depth']=e['effective_depth']=2
        branch.update(particles=1,spatial_ess=1.,reason='shared_second_probe',continuation=[50.,5.])
    elif fault=='not_shared':
        params['observation_depth']=e['depth']=e['effective_depth']=2
        branch.update(reason='shared_second_probe',continuation=[[50.,5.],[-50.,5.]])
    elif fault=='cost_identity':e['candidate_scores'][0]['cost_s']+=10.
    else:e['candidate_scores'].pop(0)
    with pytest.raises(ValueError):audit_observation_prefix(record)


def interrupted_example():
    source_observations=[((-900.,0.),0.),((100.,-1000.),90.)]
    region=CandidateRegion()
    for p,b in source_observations:region.observe(p,b)
    disk=region.enclosing_disk();current=(0.,-50.);next_cover=(200.,0.)
    phase,wire=histories([('measure',1,source_observations[0][0],'direction',0.,'coverage'),
        ('measure',1,source_observations[1][0],'direction',90.,'coverage'),
        ('measure',20,current,'no_signal',None,'coverage')])
    event=dict(channel=1,after_actual_action_count=3,end_actual_action_count=3,radius_m=disk.radius,
        detour_m=math.dist(current,disk.center)+math.dist(disk.center,next_cover)-math.dist(current,next_cover),
        actual_cost_s=0.,budget_s=60.,interrupted=True,cleared=False)
    params=dict(active_probe_algorithm='finite-observation-tree-v1',observation_depth=1,
        q4_r2_scheduling='onroute',early_service_log=[event],discovery_stop_log=[],
        observation_tree_log=[decision(3,current,disk.center,(1000.,1000.),unique=2,tuned=20)])
    return dict(row={'successful':False},history=wire,summary=dict(action_history=phase,
        coverage_points=[list(source_observations[0][0]),list(source_observations[1][0]),
                         list(current),list(next_cover)],strategy_parameters=params))


def test_unexecuted_changed_proposal_requires_real_interrupted_slice_and_fee():
    record=interrupted_example()
    result=audit_observation_prefix(record)
    assert result['unexecuted_service_slice']==1 and result['executed_changed']==0
    pooled=audit_observation_records([example(),record])
    assert pooled['actual_probe_change_rate']==1 and pooled['executed_decisions']==1
    assert pooled['selected_changes']==2  # Separate from actual changes.
    record['summary']['strategy_parameters']['early_service_log'][0]['interrupted']=False
    with pytest.raises(ValueError,match='no actual action'):audit_observation_prefix(record)


def test_false_budget_expiry_is_rejected_even_with_valid_service_geometry():
    record=interrupted_example();params=record['summary']['strategy_parameters']
    original=params['observation_tree_log'][0]
    params['observation_tree_log']=[decision(3,(0.,-50.),original['baseline'],(100.,1.),unique=2,tuned=20)]
    with pytest.raises(ValueError,match='fits the claimed'):audit_observation_prefix(record)


def test_missing_action_is_not_silently_counted_as_executed_or_dropped():
    record=example();record['history'].pop();record['summary']['action_history'].pop()
    with pytest.raises(ValueError,match='no actual action'):audit_observation_prefix(record)
    record['summary']['completion_reason']='action_budget'
    result=audit_observation_prefix(record)
    assert result['unexecuted_terminal']==1 and result['executed_decisions']==0


def test_fallback_selection_and_fallback_rate():
    record=example();event=record['summary']['strategy_parameters']['observation_tree_log'][0]
    event.update(fallback=True,reason='cpu_budget',selected=event['baseline'],candidate_scores=[])
    for h in (record['summary']['action_history'][1],record['history'][1]):h['position']=list(event['baseline'])
    result=audit_observation_records([record])
    assert result['fallback_rate']==1 and result['actual_probe_change_rate']==0
    assert result['fallback_reason_counts']=={'cpu_budget':1}


def test_no_hidden_truth_access_or_particle_reconstruction(monkeypatch):
    class Guarded(dict):
        def __getitem__(self,k):
            assert k in {'summary','history'},f'Unauthorized record field: {k}'
            return super().__getitem__(k)
    monkeypatch.setattr('planning.q4_observation_tree.make_belief',lambda *a: (_ for _ in ()).throw(AssertionError('recompute')))
    assert audit_observation_prefix(Guarded(example()))['passed']


@pytest.mark.parametrize('depth',[1,2])
def test_actual_tree_log_passes_independent_prefix_arithmetic(depth):
    from planning.q4_observation_tree import ObservationTree,unique_prefix
    region=CandidateRegion().observe((0.,0.),0.)
    first=('measure',1,(0.,0.),'direction',0.,'coverage')
    h,_=histories([first]);baseline=region.enclosing_disk().center
    selected,event=ObservationTree(depth=depth,max_cpu_s=2.).choose(region,(0.,0.),baseline,
        unique_prefix(h,1),first_bearing=0.)
    event.update(channel=1,index=0,after_actual_action_count=1)
    # A hand-constructed legal noiseless continuation, without a simulator or
    # random case generator. It is only used to bind the emitted trace to wire.
    bearing=math.degrees(math.atan2(-selected[1],200.-selected[0])) % 360
    h,wire=histories([first,('measure',1,selected,'direction',bearing,'active_localization')])
    record=dict(history=wire,summary=dict(action_history=h,strategy_parameters=dict(
        active_probe_algorithm='finite-observation-tree-v1',observation_depth=depth,
        early_service_log=[],observation_tree_log=[event])))
    assert audit_observation_prefix(record)['executed_decisions']==1
