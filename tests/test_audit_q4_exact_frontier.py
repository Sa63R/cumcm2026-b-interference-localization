"""Scripted observations and complete small path enumeration, never scenarios."""
import copy
import itertools
import math
import gzip
import json

import pytest

from experiments.audit_q4_exact_frontier import audit_exact_frontier_prefix, bellman_cost, route_cost
from simulator_client.state import Position
from strategies.q4_exact_frontier import Q4ExactFrontier
from tests.test_q4_clear_before_probe import Replies
from tests.test_audit_q4_clear_before_probe import wrap


def example(monkeypatch):
    points = (Position(0,0),Position(500,50),Position(-100,400))
    monkeypatch.setattr('planning.q4_directional_cover.certified_cover_points',lambda profile:
        (points,{'passed':True,'test_only':True}))
    client = Replies(); policy = Q4ExactFrontier(client,20000,6,max_expansions=0)
    source_points = [(300.,-200.),(-200.,300.),(150.,400.),(-300.,-100.)]
    client.measure_replies = [('near',None)]*len(source_points)
    for c,p in enumerate(source_points,1):
        policy._perform('measure',Position.coerce(p),c,'constructed_observation')
    client.clear_replies = ['success']*len(source_points)
    client.measure_replies = [('no_signal',None)]*100
    policy._execute_plan()
    record = wrap(policy)
    record['spec'] = {'entrypoint':'strategies.q4_exact_frontier:run_q4_exact_frontier',
                      'kwargs':{'config':'truncated_dag8','max_expansions':0}}
    return record


def test_full_real_action_prefix_and_inherited_checks(monkeypatch):
    record = example(monkeypatch)
    old = copy.deepcopy(record)
    result = audit_exact_frontier_prefix(record)
    assert result['passed'] and result['route_calls'] >= 4 and result['dp_calls'] > 0
    assert result['inherited_r8']['passed'] and result['inherited_r12']['passed']
    assert record == old


@pytest.mark.parametrize('change',['missing','prefix','position','source','ready','covers','cap',
    'original_work','astar_total','dp_states','dp_arcs','cost','order','applied','limits','selected',
    'feedback','spec','counters_boolean','resolver_missing'])
def test_forged_route_prefix_or_cost_rejected(monkeypatch, change):
    r=example(monkeypatch); p=r['summary']['strategy_parameters']
    e=next(e for e in p['exact_frontier_log'] if e['called'])
    route=p['chain_route_log'][e['route_log_index']]
    if change=='missing': p['exact_frontier_log'].pop()
    elif change=='prefix': e['after_actual_action_count']+=1
    elif change=='position': e['current_position'][0]+=1
    elif change=='source': e['source_channels'][0]=20
    elif change=='ready': route['source_positions'][0][0]+=1
    elif change=='covers': route['remaining_covers'].reverse()
    elif change=='cap': route['discovery_count_cap']=True
    elif change=='original_work': e['original_result']['expanded']+=1
    elif change=='astar_total': e['astar_total_expanded']+=1
    elif change=='dp_states': e['dp_total_states']+=1
    elif change=='dp_arcs': e['dp_result']['generated']+=1
    elif change=='cost': e['dp_result']['cost_s']-=1
    elif change=='order': e['dp_result']['order']=list(reversed(e['dp_result']['order']))
    elif change=='applied': e['applied']=not e['applied']
    elif change=='limits': p['exact_frontier_limits']['max_states_per_call']+=1
    elif change=='selected': route['selected_channel']=20
    elif change=='feedback': r['history'][0]['response']['measure_result']='no_signal'
    elif change=='spec': r['spec']['entrypoint']='strategies.q4_joint_continuation:run_q4_joint_continuation'
    elif change=='counters_boolean': e['dp_result']['expanded']=True
    else: p['joint_visibility_resolver_log'].pop()
    with pytest.raises((ValueError,AssertionError,KeyError)):
        audit_exact_frontier_prefix(r)


@pytest.mark.parametrize('n,k',[(0,0),(0,3),(1,2),(2,3),(3,2)])
def test_backward_bellman_against_full_order_enumeration(n,k):
    covers=[(13.*j,2.-j*7.) for j in range(k)]
    sources=[(i*8.-10.,i*i*5.) for i in range(n)]
    start=(-3.,5.); background=17.
    best=math.inf
    for permutation in itertools.permutations(range(n)):
        for slots in itertools.combinations(range(n+k),n):
            chosen=set(slots); source=cover=0; order=[]
            for step in range(n+k):
                if step in chosen:
                    order.append(('source',permutation[source]));source+=1
                else: order.append(('cover',cover));cover+=1
            best=min(best,route_cost(order,covers,sources,start,background))
    exact,states,arcs=bellman_cost(covers,sources,start,background)
    assert exact==pytest.approx(best,abs=1e-10) and states>0 and arcs>=0


def test_batch_source_binding_and_preserve_prior_audit(monkeypatch,tmp_path):
    from experiments.audit_q4_exact_frontier_batch import audit
    from experiments.audit_q4_exact_frontier import SOURCE_CONTRACT
    record=example(monkeypatch);record['row'].update(seed=7,strategy='compact_exact_frontier')
    manifest={'seeds':[7],'specs':{'compact_exact_frontier':record['spec']},'source_sha256':SOURCE_CONTRACT}
    for name,value in [('manifest.json',manifest),('freeze.json',{}),('independent_audit.json',{'all_passed':True})]:
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'records').mkdir()
    with gzip.open(tmp_path/'records/compact_exact_frontier-7.json.gz','wt',encoding='utf-8') as stream: json.dump(record,stream)
    assert audit(tmp_path)==0
    result=json.loads((tmp_path/'exact_frontier_audit.json').read_bytes())
    assert result['all_passed'] and result['records']==result['passed_records']==1
    assert result['audits'][0]['inherited_r12']['passed']
    with pytest.raises(ValueError,match='Preserve'): audit(tmp_path)


def test_dp_tie_preserves_original_task_order():
    from experiments.audit_q4_exact_frontier import audit_refinement
    base=dict(order=[['source',0],['cover',0]],cost_s=7.,lower_bound_s=7.,expanded=4,
              generated=4,dominance_pruned=0,bound_pruned=0,exact=True,runtime_s=0.)
    # Co-located tasks have equal total cost; different shared ordering is real.
    old={**base,'order':[['cover',0],['source',0]],'exact':False}
    event=dict(original_result=old,selected_result=old,called=True,applied=False,reason='no_strict_improvement',
               dp_result=base,proxy_saved_s=0.,first_action_changed=False,max_sources=8,max_covers=22,
               max_states=60000,improvement_tolerance_s=1e-9)
    states,arcs=bellman_cost([(0.,0.)],[(0.,0.)],(0.,0.),2.)[1:]
    base.update(expanded=states,generated=arcs)
    assert audit_refinement(event,[(0.,0.)],[(0.,0.)],(0.,0.),2.)==(states,arcs,0)
    event['selected_result']=base
    with pytest.raises(ValueError,match='ties'): audit_refinement(event,[(0.,0.)],[(0.,0.)],(0.,0.),2.)


def test_both_route_and_refinement_logs_cannot_disappear(monkeypatch):
    record=example(monkeypatch)
    params=record['summary']['strategy_parameters']
    params['chain_route_log'].clear();params['exact_frontier_log'].clear()
    with pytest.raises(ValueError,match='provenance'):
        audit_exact_frontier_prefix(record)
