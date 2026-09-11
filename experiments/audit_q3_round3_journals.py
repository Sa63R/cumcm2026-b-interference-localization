"""Independent accounting of persisted hypothetical worlds and full branches.

Reads logs only: no planner, simulator, fitting or new policy runs. Failed and
truncated branches remain explicitly unscored. Legacy reused evidence is marked.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check_feedback(sources, cleared, action, cache, errors):
    channel=action['channel'];p=tuple(action['position']);result=action['result']
    source=sources.get(channel)
    alive=source is not None and channel not in cleared
    dist=math.dist(p,(source['x'],source['y'])) if alive else math.inf
    if action['action']=='clear':
        if (result=='success') != (alive and dist<=20+1e-7):errors.append('clear_truth_mismatch')
        if result=='success':cleared.add(channel)
        return
    expected='no_signal' if not alive or dist>source['reception_radius_m'] else 'near' if dist<=5 else 'direction'
    if result!=expected:errors.append('measurement_truth_mismatch')
    if result=='direction' and alive:
        theta=math.degrees(math.atan2(source['y']-p[1],source['x']-p[0]))%360
        if abs((theta-action['bearing_deg']+180)%360-180)>1.005+1e-7:errors.append('bearing_truth_mismatch')
    if channel not in cleared:
        key=(channel,p);feedback=(result,action.get('bearing_deg'))
        if key in cache and cache[key]!=feedback:errors.append('fixed_feedback_mismatch')
        cache[key]=feedback


def audit_branch(branch,world,history):
    errors=[];sources={s['channel']:s for s in world['sources']};cleared=set();cache={}
    for action in history:check_feedback(sources,cleared,action,cache,errors)
    start=branch['before_public_state'];end=branch['after_public_state']
    if branch['world_sha256']!=canonical(world):errors.append('world_hash_mismatch')
    if branch['complete']:
        if branch['terminal_session']!='exited' or end['session']!='exited':errors.append('complete_without_explicit_exit')
        if not math.isclose(end['virtual_time_s']-start['virtual_time_s'],branch['remaining_s'],abs_tol=1e-5):errors.append('remaining_cost_mismatch')
    elif branch['remaining_s'] is not None:
        errors.append('incomplete_has_score')
    actions=branch.get('future_action_history')
    if actions is None:return errors,0
    point=(start['position']['x'],start['position']['y']);channel=start['current_channel']
    time_us=round(start['virtual_time_s']*1e6)
    components={k:round(v*1e6) for k,v in start['time_breakdown'].items()}
    for action in actions:
        q=tuple(action['position']);move=round(math.dist(point,q)/5*1e6)
        time_us+=move;components['movement_s']+=move;point=q
        if action['action']=='measure':
            switch=int(channel!=action['channel'])*1000000
            time_us+=5000000+switch;components['detection_s']+=5000000;components['switching_s']+=switch
            channel=action['channel']
        else:
            time_us+=3000000;components['optical_s']+=3000000
            if action['result']=='success':time_us+=2000000;components['removal_s']+=2000000
        if abs(action['virtual_time_s']-time_us/1e6)>1e-5:errors.append('cumulative_action_cost_mismatch')
        check_feedback(sources,cleared,action,cache,errors)
    if abs(end['virtual_time_s']-time_us/1e6)>1e-5:errors.append('end_cost_mismatch')
    for key,value in components.items():
        if abs(end['time_breakdown'][key]-value/1e6)>1e-5:errors.append('component_cost_mismatch:'+key)
    if branch['complete'] and cleared!=set(sources):errors.append('uncleared_source_at_exit')
    return errors,len(actions)


def audit_journal(data):
    errors=[];d=data['decision'];history=data['public_history']
    if data['controller']['history_sha256']!=canonical(history):errors.append('history_digest_mismatch')
    worlds={i:w for i,w in enumerate(data['worlds'])}
    for i,w in enumerate(d.get('risk_worlds',[])):worlds[f'risk-{i}']=w
    evaluated={};actions=0;full=0;failures=0
    for branch in data['branches']:
        wi=branch['world_index'];ci=branch['candidate_index']
        if wi not in worlds:errors.append('missing_world');continue
        sub,n=audit_branch(branch,worlds[wi],history);errors.extend(sub);actions+=n
        full+=int('future_action_history' in branch);failures+=int(not branch['complete'])
        if branch['complete']:evaluated[wi,ci]=branch['remaining_s']
        if branch['proposal_sha256']!=canonical(data['proposals'][ci]):errors.append('candidate_hash_mismatch')
    if d.get('completed') and 'best_index' in d:
        best=d['best_index']
        if any((i,j) not in evaluated for i in range(12) for j in (0,best)):
            errors.append('decision_missing_complete_pairs')
        else:
            delta=[evaluated[i,best]-evaluated[i,0] for i in range(12)]
            if any(abs(a-b)>1e-5 for a,b in zip(delta,d['paired_delta_s'])):errors.append('decision_pair_cost_mismatch')
            confirm=delta[4:];mean=statistics.mean(confirm);se=statistics.stdev(confirm)/math.sqrt(8)
            if abs(mean-d['confirmation']['mean_s'])>1e-5 or abs(se-d['confirmation']['standard_error_s'])>1e-5:errors.append('confirmation_statistic_mismatch')
            if d['nominal_accept']!=(mean+se < -3):errors.append('independent_acceptance_rule_mismatch')
            if d['selected']!='baseline' and not d['nominal_accept']:errors.append('adopted_without_confirmation')
    return dict(passed=not errors,errors=sorted(set(errors)),branches=len(data['branches']),
                full_branches=full,full_branch_actions=actions,incomplete_branches=failures)


def run(paths):
    total=Counter();failures=[]
    for root in map(Path,paths):
        for path in sorted(root.glob('branches-*/*/decision-*.json.gz')):
            raw=gzip.open(path,'rb').read();data=json.loads(raw);result=audit_journal(data)
            total['journals']+=1
            for key in ('branches','full_branches','full_branch_actions','incomplete_branches'):total[key]+=result[key]
            if not result['passed']:failures.append(dict(path=path.relative_to(root).as_posix(),batch=root.name,errors=result['errors']))
    if not total['journals']:
        failures.append(dict(path=None,batch=None,errors=['no_journals_found']))
    return dict(passed=not failures,counts=dict(total),failures=failures,
                scope='Persisted hypothetical branches; history compatibility, paired worlds, independent8 adoption and full costs',
                limits='Unselected candidate branches retain complete costs/status but not full future action histories; finite worlds are not a universal certificate.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batches',nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=run(a.batches);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
