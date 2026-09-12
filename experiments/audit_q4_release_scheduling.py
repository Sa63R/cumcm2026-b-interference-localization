"""Independent prefix, released-task route and inherited physical safety audit.

Nominal release indices constrain only a frozen proxy schedule. A real source
macro during discovery still needs actual readiness. Forecast observations are
never put in the wire, canonical region, scan matrix or coverage ledger.
"""
from functools import lru_cache
import hashlib
import math
from pathlib import Path

from experiments import audit_q4_known_source as base

ROOT=Path(__file__).resolve().parents[1]
ENTRY='strategies.q4_release_scheduling:run_q4_release_scheduling'
LABEL='compact_release_scheduling'
CONFIG='nominal_release'
BASE_AUDIT_SHA256='107b943dddfaabb3735bce3be43c31798f5beda14883abbd6a6f4ea180a50d30'
NEW_SOURCE_CONTRACT={
    'src/planning/release_chain_route.py':'f0ea8c4438465e9876f7f967fd15de543a751a7e3261579d74160ed27b94593d',
    'src/strategies/q4_release_scheduling.py':'87a5124ca6021e0ce2df7c1ce49cacf1116699a2bed21ad5e2b592ee4ace2fe2'}
require,point,close,same=base.require,base.point,base.close,base.same


def verify_source_contract():
    base.verify_source_contract()
    require(hashlib.sha256((ROOT/'experiments/audit_q4_known_source.py').read_bytes()).hexdigest()==BASE_AUDIT_SHA256,
            'Reviewed R34 replay changed')
    require(len(NEW_SOURCE_CONTRACT)==2,'Release implementation not yet reviewed/frozen')
    for name,expected in NEW_SOURCE_CONTRACT.items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,'Reviewed release source changed: '+name)


def release_route_cost(covers,sources,start,order,background,matrix,releases):
    require(len(releases)==len(sources) and all(type(r) is int and 0<=r<=len(covers) for r in releases),
            'Invalid release-index vector')
    k=0
    for kind,j in order:
        if kind=='cover': k+=1
        else:
            require(kind=='source' and type(j) is int and 0<=j<len(sources),'Invalid source index')
            require(k>=releases[j],'Source scheduled before its predicted release')
    return base.matrix_route_cost(covers,sources,start,order,background,matrix)+5.*len(sources)


def small_optimum(covers,sources,start,background,matrix,releases):
    """Enumerate the small released DAG independently of production solvers."""
    require(len(sources)<=4 and len(releases)==len(sources),'Small released check only')
    @lru_cache(None)
    def suffix(mask,k,last):
        if mask==(1<<len(sources))-1 and k==len(covers): return 0.
        here=start if last==-1 else sources[last] if last<len(sources) else covers[last-len(sources)]
        values=[math.dist(here,p)/5.+5.+suffix(mask|(1<<j),k,j)
                for j,p in enumerate(sources) if not mask&(1<<j) and k>=releases[j]]
        if k<len(covers):
            fee=background[k]+sum(matrix[k][j] for j in range(len(sources)) if not mask&(1<<j))
            values.append(math.dist(here,covers[k])/5.+fee+suffix(mask,k+1,len(sources)+k))
        require(values,'Released DAG has no complete continuation')
        return min(values)
    return suffix(0,0,-1)


def forecast(prefix,n,c,covers):
    """Reconstruct all nominal steps from true canonical C and fixed target."""
    target=prefix.target(n,c); r=prefix.snapshot(n)[3].get(c)
    circle=r.enclosing_disk() if r and r.vertices else None
    ready=prefix.ready(n,c)
    result=dict(channel=c,target=list(target),ready_before=ready,
                initial_radius_m=circle.radius if circle else None,
                release_index=len(covers),status='planned',steps=[])
    if ready:
        result.update(release_index=0,status='actual_ready'); return result
    if not covers:
        result.update(release_index=0,status='no_remaining_covers'); return result
    if (not r or not r.vertices or not r.observations or not circle
            or not math.isfinite(circle.radius) or circle.radius<=19.9
            or any(not all(math.isfinite(x) for x in p) for p in r.vertices)):
        result['status']='fallback_invalid_canonical'; return result
    predicted=r.copy()
    for k,q in enumerate(covers):
        distance=math.dist(q,target)
        step=dict(cover_index=k,position=list(q),nominal_distance_m=distance,bearing_deg=None,
                  predicted_radius_m=predicted.enclosing_disk().radius,status='planned',stopped=False)
        result['steps'].append(step)
        if not math.isfinite(distance):
            step.update(status='invalid_distance',stopped=True)
            result['status']='fallback_invalid_prediction'; return result
        if distance<=5.:
            step['status']='skip_near_distance'; continue
        if distance>1500.:
            step['status']='skip_beyond_reception_radius'; continue
        bearing=round(math.degrees(math.atan2(target[1]-q[1],target[0]-q[0]))%360.,2)%360.
        step['bearing_deg']=bearing
        try:
            predicted.observe(q,bearing)
        except (ValueError,ArithmeticError):
            step.update(status='invalid_prediction',predicted_radius_m=None,stopped=True)
            result['status']='fallback_invalid_prediction'; return result
        if not predicted.vertices:
            step.update(status='empty_prediction',predicted_radius_m=None,stopped=True)
            result['status']='fallback_empty_prediction'; return result
        try:
            radius=predicted.enclosing_disk().radius
        except (ValueError,ArithmeticError):
            step.update(status='invalid_prediction',predicted_radius_m=None,stopped=True)
            result['status']='fallback_invalid_prediction'; return result
        step['predicted_radius_m']=radius
        if not math.isfinite(radius) or any(not all(math.isfinite(x) for x in p) for p in predicted.vertices):
            step.update(status='invalid_prediction',stopped=True)
            result['status']='fallback_invalid_prediction'; return result
        if radius<=19.9:
            step.update(status='predicted_ready',stopped=True)
            result.update(release_index=k+1,status='nominal_ready'); return result
        step['status']='predicted_not_ready'
    result['status']='cover_tail'
    return result


def check_plan(event,prefix,n,covers,channels,blocked,total,index,maximum):
    known,cleared,_,regions=prefix.snapshot(n); current,tuned,_=prefix.before[n]
    targets=[prefix.target(n,c) for c in channels]
    forecasts=[forecast(prefix,n,c,covers) for c in channels]
    releases=[f['release_index'] for f in forecasts]
    require(all(r==0 if prefix.ready(n,c) or not covers else 1<=r<=len(covers)
                for c,r in zip(channels,releases)),'Invalid actual versus nominal release')
    expected=dict(id=index,config=CONFIG,after_actual_action_count=n,current_position=list(current),current_channel=tuned,
        known_channels=sorted(known),cleared_channels=sorted(cleared),blocked_channels=sorted(blocked),
        remaining_covers=[list(q) for q in covers],source_channels=channels,source_positions=[list(p) for p in targets],
        source_services_s=[5.]*len(channels),source_evidence=[base.evidence(prefix,n,c,blocked) for c in channels],
        release_indices=releases,release_forecasts=forecasts,discovery_count_cap=len(known)==16)
    other,background,matrix=base.frozen_matrix(prefix,n,covers,channels)
    expected.update(background_channels=other,background_scan_s=background,source_scan_s=matrix,
                    background_evidence=[base.evidence(prefix,n,c,blocked) for c in other])
    cells=[]
    for q in covers:
        row=[]
        for c in range(1,21):
            if c in cleared: continue
            r=regions.get(c); lower=None; reason='measurement_proxy'
            if prefix.ready(n,c): reason='ready'
            elif c in known and r and r.observations:
                lower=base._range_lower(q,r.vertices)
                if lower>1500.+1e-5: reason='positive_region_beyond_max_reception_radius'
            row.append(dict(channel=c,fee_s=base.fee_at(prefix,n,q,c),reason=reason,distance_lower_bound_m=lower))
        cells.append(row)
    expected['matrix_evidence']=cells
    for k,v in expected.items(): same(event[k],v,'Release plan differs from actual prefix: '+k)
    budget=max(0,min(maximum,60000-total))
    require(event['max_expansions']==budget and event['total_expansions_before']==total,'Release A* budget prefix differs')
    result=event['result']; order=result['order']
    cost=release_route_cost(covers,targets,current,order,background,matrix,releases)
    close(result['cost_s'],cost,'Released complete route cost differs')
    lower=base.finite(result['lower_bound_s'])
    initial=base.root_relaxation(covers,targets,current,background)+5.*len(channels)
    require(-1e-7<=lower<=cost+1e-7 and lower+2e-6>=initial,'Invalid released-model lower bound')
    require(type(result['exact']) is bool and (not result['exact'] or abs(lower-cost)<=2e-6),'Wrong exact release gap')
    if len(channels)<=4:
        optimum=small_optimum(covers,targets,current,background,matrix,releases)
        require(lower<=optimum+2e-6 and cost>=optimum-2e-6
                and (not result['exact'] or abs(cost-optimum)<=2e-6),'Independent small released DAG disagrees')
    expanded=base.integer(result['expanded'],0,budget)
    for k in ('generated','dominance_pruned','bound_pruned'): base.integer(result[k])
    base._finite_runtime(result['runtime_s']); base._finite_runtime(event['runtime_s'])
    require(event['total_expansions_after']==total+expanded,'Release cumulative A* count differs')
    kind,j=order[0]; channel=channels[j] if kind=='source' else None
    require(event['selected_kind']==kind and event['selected_channel']==channel,'First actual macro differs from released order')
    return kind,channel,total+expanded


def replay_macros(record):
    prefix=base.Prefix(record); h=prefix.h; summary=record['summary']; params=summary['strategy_parameters']
    points=list(map(point,summary['coverage_points']))
    chains=params['known_source_plan_log']; services=params['known_source_service_log']
    require(chains==params['release_scheduling_plan_log'],'Release/parent plan ledgers differ')
    require(chains==params['chain_route_log'],'Two plan ledgers differ')
    epochs=params['joint_visibility_resolver_log']; early=params['early_service_log']
    maximum=record['spec'].get('kwargs',{}).get('max_expansions',200)
    n=visited=ci=si=ei=mi=total=broad=service_actions=forecast_macros=0
    blocked=set(); attempted=set()

    def consume_resolver(index,start,c):
        require(index<len(epochs),'Missing actual resolver')
        e=epochs[index]
        require(e['id']==index and e['channel']==c and e['after_actual_action_count']==start,
                'Resolver identity/prefix differs')
        end=base.integer(e['end_actual_action_count'],start,len(h))
        require(all(a['channel']==c and a['phase']!='coverage' for a in h[start:end]),
                'Source interval includes coverage/another source')
        status=e['status']
        require(status in {'cleared','unresolved','interrupted'},'Unfinished resolver')
        if status!='interrupted':
            require((c in prefix.snapshot(end)[1])==(status=='cleared'),'Resolver status contradicts actual clearance')
        return e,end

    # Zero-action resolver/early epochs are consumed by identity, never by a
    # comparison with later intervals which can start at the same prefix.
    for _ in range(len(h)+len(chains)+len(epochs)+len(early)+5):
        known,cleared,_,_=prefix.snapshot(n)
        if len(cleared)==16: break
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
            kind,c,total=check_plan(event,prefix,n,covers,channels,blocked,total,ci,maximum)
            ci+=1
            mixed=bool(covers) and any(prefix.ready(n,c) for c in channels) and any(not prefix.ready(n,c) for c in channels)
            decision_start=n
            if kind=='source':
                require(not covers or prefix.ready(n,c),'Actual source macro started without actual readiness during coverage')
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
                forecast_macros+=int(mixed and n>decision_start)
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
            forecast_macros+=int(mixed and n>decision_start)
            require(event['end_actual_action_count']==n and event['status']==('cover_completed' if complete else 'interrupted'),
                    'Cover plan exit/status differs')
        if not complete: break
    else: raise ValueError('Macro replay failed bounded progress')
    require(n==len(h) and ci==len(chains) and si==len(early) and ei==len(epochs) and mi==len(services),
            'Missing/orphan actual action, plan or resolver macro')
    require(summary['coverage_points_visited']==visited,'Visited count differs from complete scans')
    return dict(forecast_scheduled_macros=forecast_macros,known_source_macros=mi,source_service_actions=service_actions,broad_service_actions=broad,
                completed_scans=visited,chain_decisions=ci,original_astar_expanded=total)


def audit_release_scheduling_prefix(record):
    verify_source_contract()
    spec=record['spec']; kwargs=spec.get('kwargs',{})
    require(spec['entrypoint']==ENTRY and record['row']['strategy']==LABEL,'Unreviewed release entry/label')
    require(set(kwargs)<={'config','max_expansions','max_actions','max_active_probes','problem'}
            and kwargs.get('config')==CONFIG and type(kwargs.get('problem',4)) is int and kwargs.get('problem',4)==4
            and type(kwargs.get('max_expansions',200)) is int and kwargs.get('max_expansions',200)==200,'Unreviewed release spec')
    params=record['summary']['strategy_parameters']
    require(params['known_source_config']==base.CONFIG and params['release_scheduling_config']==CONFIG,
            'Logged release/parent configuration differs')
    result=replay_macros(record)
    view=dict(record,spec=dict(spec,entrypoint='strategies.q4_joint_continuation:run_q4_joint_continuation',
                              kwargs=dict(kwargs,config='after_active_miss_optical')))
    r12=base.audit_joint_continuation_prefix(view); r8=base.audit_clear_before_probe_prefix(view)
    accepted_view=dict(view,history=[w for w in view['history'] if w['action'] not in {'/measure','/clear'}
                                     or w['response'].get('accepted') is True])
    range_result=base.audit_range_prefix(accepted_view); scheduling=base.audit_scheduling_prefix(accepted_view)
    for child in (r12,r8,range_result,scheduling): require(child.get('passed') is True,'Inherited release audit did not pass')
    result.update(passed=True,r12=r12,r8=r8,range=range_result,scheduling=scheduling,source_contract_files=57,
        reused_matrix_audit_sha256=BASE_AUDIT_SHA256,
        boundary='Prefix-frozen optimistic release proxy; actual ready source guard remains mandatory. '
                 'Forecast-mixed executed macros are exposure counts, not causal changes or saved time. '
                 'Only <=4-source optimum is independently enumerated. No actual history is modified; '
                 'only legacy range/scheduling receive accepted-wire views; generic retains rejected requests.')
    return result


def audit_full(record):
    from experiments.audit_q4_cover import audit_record
    generic=audit_record(record)
    require(generic.get('passed') is True,'Generic physical/coverage audit failed: '+str(generic.get('errors')))
    return dict(passed=True,generic=generic,prefix=audit_release_scheduling_prefix(record))

