"""Independent real-prefix shared-known observation audit.

The reviewed R34 matrix checker is reused for plan arithmetic, not for macro
replay: shared measurements remain in the real history and change subsequent
canonical/negative prefixes. No source truths or shared-candidate helper are
imported. Full generic physics and original R12/R8 audits remain mandatory.
"""
import hashlib
import math
from pathlib import Path

from experiments import audit_q4_known_source as base

ROOT=Path(__file__).resolve().parents[1]
ENTRY='strategies.q4_shared_known:run_q4_shared_known'
LABEL='compact_shared_known'
CONFIG='nominal_ready_after_clear'
PHASE='shared_known_observation'
BASE_AUDIT_SHA256='107b943dddfaabb3735bce3be43c31798f5beda14883abbd6a6f4ea180a50d30'
NEW_SOURCE_SHA256='a39252682a7356ca9ee5586d33613357e62f903c50659eb32b19bab2e4c976b5'
require,point,close,same=base.require,base.point,base.close,base.same


def verify_source_contract():
    base.verify_source_contract()
    require(hashlib.sha256((ROOT/'experiments/audit_q4_known_source.py').read_bytes()).hexdigest()==BASE_AUDIT_SHA256,
            'Reviewed R34 replay changed')
    require(NEW_SOURCE_SHA256 is not None and
            hashlib.sha256((ROOT/'src/strategies/q4_shared_known.py').read_bytes()).hexdigest()==NEW_SOURCE_SHA256,
            'Shared-known source not reviewed/frozen')


def fresh_at(prefix,n,c,q):
    key=tuple(round(x,6) for x in q)
    return all(tuple(round(x,6) for x in point(a['position']))!=key
               for a in prefix.h[:n] if a['action']=='measure' and a['channel']==c)


def nominal(prefix,n,c,q):
    """A private canonical copy: this prediction is never a real certificate."""
    _,_,_,regions=prefix.snapshot(n)
    r=regions.get(c)
    if not r or not r.observations or not r.vertices: return None
    circle=r.enclosing_disk(); center=tuple(circle.center)
    if not math.isfinite(circle.radius) or circle.radius<=19.9: return None
    distance=math.dist(q,center)
    if not 5.<distance<=1500.: return None
    bearing=round(math.degrees(math.atan2(center[1]-q[1],center[0]-q[0]))%360.,2)%360.
    hypothetical=r.copy()
    hypothetical.observe(q,bearing)
    if not hypothetical.vertices: return None
    predicted=hypothetical.enclosing_disk().radius
    if not math.isfinite(predicted) or predicted>19.9: return None
    return dict(channel=c,original_radius_m=circle.radius,center=list(center),center_distance_m=distance,
                nominal_bearing_deg=bearing,predicted_radius_m=predicted,ratio=predicted/circle.radius)


def candidate_evidence(prefix,n,blocked,attempted):
    known,cleared,_,regions=prefix.snapshot(n); q=prefix.before[n][0]
    rows=[]
    for c in sorted(known-cleared):
        r=regions.get(c); circle=r.enclosing_disk() if r and r.vertices else None
        row=dict(channel=c,original_vertices=[list(p) for p in r.vertices] if r else [],
            positive_observation_count=len(r.observations) if r else 0,
            original_center=list(circle.center) if circle else None,
            original_radius_m=circle.radius if circle else None,ready=prefix.ready(n,c),blocked=c in blocked,
            attempted=c in attempted,previously_observed=not fresh_at(prefix,n,c,q),nominal_distance_m=None,
            nominal_bearing_deg=None,predicted_vertices=None,predicted_center=None,predicted_radius_m=None,
            ratio=None,eligible=False,reason=None)
        rows.append(row)
        reason=('blocked' if row['blocked'] else 'already_ready' if row['ready'] else
                'target_already_attempted' if row['attempted'] else
                'no_canonical_positive_region' if not r or not r.vertices or not r.observations else
                'position_already_observed' if row['previously_observed'] else
                'invalid_or_ready_radius' if not math.isfinite(circle.radius) or circle.radius<=19.9 else None)
        if reason:
            row['reason']=reason; continue
        center=tuple(circle.center); distance=math.dist(q,center)
        row['nominal_distance_m']=distance
        if not math.isfinite(distance) or not 5.<distance<=1500.:
            row['reason']='nominal_distance_outside_range'; continue
        bearing=round(math.degrees(math.atan2(center[1]-q[1],center[0]-q[0]))%360.,2)%360.
        row['nominal_bearing_deg']=bearing
        hypothetical=r.copy().observe(q,bearing)
        row['predicted_vertices']=[list(p) for p in hypothetical.vertices]
        if not hypothetical.vertices:
            row['reason']='empty_nominal_region'; continue
        disk=hypothetical.enclosing_disk()
        row.update(predicted_center=list(disk.center),predicted_radius_m=disk.radius,ratio=disk.radius/circle.radius)
        row['eligible']=math.isfinite(disk.radius) and disk.radius<=19.9
        row['reason']='nominal_ready' if row['eligible'] else 'nominal_not_ready'
    return rows


def check_share(event,prefix,n,trigger_index,blocked,seen,attempted,index):
    h=prefix.h; summary=prefix.record['summary']; params=summary['strategy_parameters']
    a=h[trigger_index]; c=a['channel']; q,tuned,now=prefix.before[n]
    require(a['action']=='clear' and a['result']=='success' and c not in seen
            and trigger_index+1==n and point(a['position'])==q,'Trigger is not a fresh terminal clear at the current point')
    epochs=[e for e in params['joint_visibility_resolver_log'] if e['channel']==c
            and e['after_actual_action_count']<=trigger_index<e['end_actual_action_count']]
    require(len(epochs)==1 and epochs[0]['end_actual_action_count']==n and epochs[0]['status']=='cleared',
            'Sharing entered before parent resolver finally closed')
    early=[j for j,e in enumerate(params['early_service_log']) if e['channel']==c
           and e['after_actual_action_count']<=trigger_index<e['end_actual_action_count']]
    require(len(early)<=1,'Ambiguous parent early slice')
    if early: require(params['early_service_log'][early[0]]['end_actual_action_count']==n,'Early finally still active')
    known,cleared,_,_=prefix.snapshot(n)
    expected=dict(id=index,trigger_clear_action_index=trigger_index,trigger_after_action_count=trigger_index+1,
        trigger_channel=c,parent_resolver_id=epochs[0]['id'],parent_end_action_count=n,
        early_service_index=early[0] if early else None,after_actual_action_count=n,
        current_position=list(q),current_channel=tuned,position=list(q),seen_clear_before=sorted(seen),
        attempted_before=sorted(attempted),known_channels=sorted(known),cleared_channels=sorted(cleared),
        blocked_channels=sorted(blocked),start_virtual_time_s=now)
    for k,v in expected.items(): same(event[k],v,'Shared prefix differs: '+k)
    require(len(cleared)<16,'Shared action after actual sixteen-clear termination')
    candidates=candidate_evidence(prefix,n,blocked,attempted)
    same(event['candidates'],candidates,'Missing/changed nominal candidate or exclusion')
    eligible=[e for e in candidates if e['eligible']]
    chosen=min(eligible,key=lambda e:(e['ratio'],e['channel'])) if eligible else None
    target=chosen['channel'] if chosen else None
    require(event['selected_channel']==target,'Shared selection differs from stable common ranking')
    budget=event['budget']
    same(budget['policy_actions_before'],prefix.enter_count+n,'Wrong actual action budget prefix')
    same(budget['max_actions'],prefix.max_actions,'Wrong maximum actions')
    close(budget['virtual_limit_s'],prefix.virtual_limit,'Wrong virtual duration limit')
    remaining=budget['remaining_real_s']
    if remaining is not None: base.finite(remaining)
    end=base.integer(event['end_actual_action_count'],n,min(n+1,len(h)))
    seen.add(c)
    if chosen is None:
        require(end==n and event['status']=='no_candidate' and event['actual_result'] is None
                and event['actual_bearing_deg'] is None and event['actual_radius_before_m'] is None
                and event['actual_radius_after_m'] is None and event['actual_ready_after'] is None,
                'Skipped opportunity fabricated an observation or radius')
    else:
        require(target not in attempted,'Target attempted more than once')
        attempted.add(target)
        same(event['actual_radius_before_m'],chosen['original_radius_m'],'Wrong actual pre-share radius')
        fee=5.+int(target!=tuned)
        if end==n+1:
            actual=h[n]
            require(event['status']=='measured' and actual['action']=='measure' and actual['phase']==PHASE
                    and actual['channel']==target and point(actual['position'])==q,'Shared action absent, moved, or entered a false phase')
            require(event['actual_result']==actual['result'] and event['actual_bearing_deg']==actual.get('bearing_deg'),
                    'Nominal prediction replaced true feedback')
            close(prefix.before[end][2]-now,fee,'Stationary shared measurement fee differs')
            require(prefix.enter_count+n<prefix.max_actions-1 and now+fee<=prefix.virtual_limit-1e-6
                    and (remaining is None or remaining>2.),'Accepted sharing exceeded original budget')
        else:
            require(event['status']=='interrupted' and base.terminal(summary,n,h)
                    and event['actual_result'] is None and event['actual_bearing_deg'] is None,
                    'Unexecuted sharing lacks a real terminal interruption')
            reason=event.get('interruption_reason')
            if reason=='action_budget': require(prefix.enter_count+n>=prefix.max_actions-1,'False action exhaustion')
            elif reason=='virtual_budget': require(now+fee>prefix.virtual_limit-1e-6,'False virtual exhaustion')
            elif reason=='real_deadline': require(remaining is not None and remaining<=2.,'False logged deadline')
            elif reason=='request_rejected':
                require(any(w['action']=='/measure' and w.get('channel')==target and point(w['position'])==q
                            and w['response'].get('accepted') is not True for w in prefix.record['history']),
                        'No matching rejected shared request')
        r=prefix.snapshot(end)[3].get(target)
        actual_radius=r.enclosing_disk().radius if r and r.vertices else None
        same(event['actual_radius_after_m'],actual_radius,'Actual post-share region forged')
        same(event['actual_ready_after'],prefix.ready(end,target),'Nominal readiness used as actual readiness')
    same(event['seen_clear_after'],sorted(seen),'Trigger ledger changed')
    same(event['attempted_after'],sorted(attempted),'Target attempt ledger changed')
    close(event['end_virtual_time_s'],prefix.before[end][2],'Wrong shared exit time')
    close(event['actual_cost_s'],prefix.before[end][2]-now,'Wrong shared total fee')
    require(base.finite(event['runtime_s'])>=base.finite(event['decision_wall_s'])>=0,'Bad sharing wall-time log')
    return end,end-n


def replay_macros(record):
    prefix=base.Prefix(record); h=prefix.h; summary=record['summary']; params=summary['strategy_parameters']
    points=list(map(point,summary['coverage_points']))
    chains=params['known_source_plan_log']; services=params['known_source_service_log']
    require(chains==params['chain_route_log'],'Two plan ledgers differ')
    epochs=params['joint_visibility_resolver_log']; early=params['early_service_log']
    maximum=record['spec'].get('kwargs',{}).get('max_expansions',200)
    n=visited=ci=si=ei=mi=total=broad=service_actions=0
    blocked=set(); attempted=set()
    shares=params['shared_known_observation_log']; xi=shared_actions=0
    seen=set(); shared_attempted=set()

    def consume_resolver(index,start,c):
        require(index<len(epochs),'Missing actual resolver')
        e=epochs[index]
        require(e['id']==index and e['channel']==c and e['after_actual_action_count']==start,
                'Resolver identity/prefix differs')
        end=base.integer(e['end_actual_action_count'],start,len(h))
        require(all(a['channel']==c and a['phase'] not in {'coverage',PHASE} for a in h[start:end]),
                'Source interval includes coverage, sharing or another source')
        status=e['status']
        require(status in {'cleared','unresolved','interrupted'},'Unfinished resolver')
        if status!='interrupted':
            require((c in prefix.snapshot(end)[1])==(status=='cleared'),'Resolver status contradicts actual clearance')
        return e,end

    # Zero-action resolver/early epochs are consumed by identity, never by a
    # comparison with later intervals which can start at the same prefix.
    for _ in range(len(h)+len(chains)+len(epochs)+len(early)+len(shares)+5):
        known,cleared,_,_=prefix.snapshot(n)
        if len(cleared)==16: break
        pending=[(j,a) for j,a in enumerate(h[:n]) if a['action']=='clear'
                 and a['result']=='success' and a['channel'] not in seen]
        require(len(pending)<=1,'Several clearances bypassed a required loop boundary')
        interrupted=False
        for j,a in pending:
            require(xi<len(shares),'Missing new-clear shared-observation opportunity')
            end,count=check_share(shares[xi],prefix,n,j,blocked,seen,shared_attempted,xi)
            n=end; xi+=1; shared_actions+=count
            if shares[xi-1]['status']=='interrupted':
                require(base.terminal(summary,n,h),'Sharing interrupted without real terminal')
                interrupted=True; break
        if interrupted: break
        known,cleared,_,_=prefix.snapshot(n)
        covers=[] if len(known)==16 else points[visited:]
        old=prefix.original_early(n,covers[0],blocked,attempted) if covers else None
        if old is not None:
            require(si<len(early),'Missing priority early service')
            event=early[si]
            require(event['after_actual_action_count']==n and event['channel']==old[2],'Changed original early priority')
            epoch,end=consume_resolver(ei,n,old[2])
            require(event['end_actual_action_count']==end,'Early/resolver ranges disagree')
            attempted.add(old[2]); si+=1; ei+=1; n=end
            if epoch['status']=='interrupted' and not event['interrupted']:
                require(base.terminal(summary,n,h),'Unexplained early interruption'); break
            continue
        channels=[c for c in sorted(known-cleared-blocked) if prefix.target(n,c) is not None]
        event=None
        if channels:
            require(ci<len(chains),'Missing all-known matrix decision')
            event=chains[ci]
            kind,c,total=base.check_plan(event,prefix,n,covers,channels,blocked,total,ci,maximum)
            ci+=1
            if kind=='source':
                require(mi<len(services),'Missing selected-source macro')
                service=services[mi]; region=prefix.snapshot(n)[3].get(c)
                radius=region.enclosing_disk().radius if region and region.vertices else None
                ready=prefix.ready(n,c); start=n
                expected=dict(decision_id=event['id'],selected=dict(channel=c,ready_before=ready,radius_m=radius),
                    remaining_covers_before=[list(p) for p in covers],resolver_start_action_count=n,
                    resolver_id=ei,start_virtual_time_s=prefix.before[n][2],start_position=list(prefix.before[n][0]))
                for k,v in expected.items(): same(service[k],v,'Source macro differs: '+k)
                epoch,n=consume_resolver(ei,start,c); ei+=1; mi+=1
                require(service['service_end_action_count']==n and event['end_actual_action_count']==n,'Wrong service exit')
                same(service['end_position'],list(prefix.before[n][0]),'Wrong actual exit position')
                close(service['actual_cost_s'],prefix.before[n][2]-prefix.before[start][2],'Wrong actual service cost')
                status={'cleared':'resolved','unresolved':'blocked','interrupted':'interrupted'}[epoch['status']]
                require(service['status']==status and event['status']==status
                        and service['cleared']==(c in prefix.snapshot(n)[1]),'Source completion or blockage fabricated')
                service_actions+=n-start
                if not ready and radius is not None and radius>40. and covers: broad+=n-start
                if status=='blocked': blocked.add(c)
                if status=='interrupted':
                    require(base.terminal(summary,n,h),'Full source resolver interrupted without terminal'); break
                continue
        elif not covers: break
        require(covers,'Cover selection without remaining obligation')
        q=covers[0]; expected=base.scan_channels(prefix,n,q); start=n
        for c in expected:
            if n==len(h): break
            a=h[n]
            require(a['action']=='measure' and a['phase']=='coverage' and a['channel']==c
                    and point(a['position'])==q,'Selected cover channel/position/order differs')
            n+=1
        complete=n-start==len(expected)
        if complete:
            require(bool(expected),'Fictional zero-action complete cover')
            visited+=1; blocked.clear()
        else: require(base.terminal(summary,n,h),'Incomplete real coverage scan without terminal')
        if event is not None:
            require(event['end_actual_action_count']==n and event['status']==('cover_completed' if complete else 'interrupted'),
                    'Cover plan exit/status differs')
        if not complete: break
    else: raise ValueError('Macro replay failed bounded progress')
    require(n==len(h) and ci==len(chains) and si==len(early) and ei==len(epochs) and mi==len(services) and xi==len(shares),
            'Missing/orphan actual action, plan or resolver macro')
    require(summary['coverage_points_visited']==visited,'Visited count differs from complete scans')
    require(shared_actions<=15 and len(shared_attempted)<=15,'Natural shared-attempt bound violated')
    require(shared_actions==sum(a.get('phase')==PHASE for a in h),'Orphan/misclassified shared action')
    return dict(shared_observation_actions=shared_actions,shared_opportunities=xi,shared_attempted_channels=sorted(shared_attempted),
                known_source_macros=mi,source_service_actions=service_actions,broad_service_actions=broad,
                completed_scans=visited,chain_decisions=ci,original_astar_expanded=total)


def audit_shared_known_prefix(record):
    verify_source_contract()
    spec=record['spec']; kwargs=spec.get('kwargs',{})
    require(spec['entrypoint']==ENTRY and record['row']['strategy']==LABEL,'Unreviewed shared-known entry/label')
    require(set(kwargs)<={'config','max_expansions','max_actions','max_active_probes','problem'}
            and kwargs.get('config')==CONFIG and type(kwargs.get('problem',4)) is int and kwargs.get('problem',4)==4
            and type(kwargs.get('max_expansions',200)) is int and kwargs.get('max_expansions',200)==200,
            'Unreviewed shared-known spec')
    params=record['summary']['strategy_parameters']
    require(params['known_source_config']==base.CONFIG and params['shared_known_config']==CONFIG,
            'Logged shared-known/parent configuration differs')
    result=replay_macros(record)
    view=dict(record,spec=dict(spec,entrypoint='strategies.q4_joint_continuation:run_q4_joint_continuation',
                              kwargs=dict(kwargs,config='after_active_miss_optical')))
    r12=base.audit_joint_continuation_prefix(view); r8=base.audit_clear_before_probe_prefix(view)
    accepted_view=dict(view,history=[w for w in view['history'] if w['action'] not in {'/measure','/clear'}
                                     or w['response'].get('accepted') is True])
    range_result=base.audit_range_prefix(accepted_view)
    scheduling=base.audit_scheduling_prefix(accepted_view)
    for child in (r12,r8,range_result,scheduling):
        require(child.get('passed') is True,'Inherited shared-known audit did not pass')
    result.update(passed=True,r12=r12,r8=r8,range=range_result,scheduling=scheduling,
                  source_contract_files=56,reused_matrix_audit_sha256=BASE_AUDIT_SHA256,
        boundary='Actual shared radio stays in every prefix; nominal readiness is only a selection proxy. '
                 'Range/scheduling alone receive accepted-wire views; full generic audit retains rejected requests. '
                 'No clearance or unknown coverage is inferred from the prediction.')
    return result


def audit_full(record):
    from experiments.audit_q4_cover import audit_record
    generic=audit_record(record)
    require(generic.get('passed') is True,'Generic physical/coverage audit failed: '+str(generic.get('errors')))
    return dict(passed=True,generic=generic,prefix=audit_shared_known_prefix(record))

