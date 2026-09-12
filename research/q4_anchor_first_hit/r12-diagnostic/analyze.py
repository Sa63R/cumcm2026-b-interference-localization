"""R12 observed-history accounting only: no simulator, policy or truth access.

Reads exactly the complete 140+98 reference rows. Ground-truth positions are
never selected or used; final public N is checked against accepted clear count.
A phase attribution is accounting, not a causal or recoverable-saving estimate.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

WORKSPACE=Path(__file__).resolve().parents[4]
REFERENCE=WORKSPACE/'q4-per-source-reference'
GEOMETRY_SHA={'src/geometry/__init__.py':'ab863186eed111790cf712c0cd19c741841087144a8fd9ecd37a3c10d33e46fa',
              'src/localization/__init__.py':'cda7e2f2a45faa60cb7dcef631e54cd384e3f367db96f23f415fc7eb666bb98c'}
for name,digest in GEOMETRY_SHA.items():
    assert hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest()==digest,'Changed reference geometry'
sys.path.insert(0,str(REFERENCE/'src'))
from localization import CandidateRegion

COMP=('movement_s','detection_s','switching_s','optical_s','removal_s')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dist(p,q): return math.dist(p,q)
def total(items): return {k:math.fsum(x[k] for x in items) for k in COMP}
def amount(v): return math.fsum(v.values())
def quantile(values,p):
    if not values:return None
    values=sorted(values);a=(len(values)-1)*p;i=int(a)
    return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(a-i)
def describe(values):
    return dict(count=len(values),mean=statistics.fmean(values) if values else None,
        median=quantile(values,.5),p95=quantile(values,.95),maximum=max(values) if values else None)

def blocks(h,points):
    result=[];i=0
    while i<len(h):
        a=h[i]
        if a['phase']!='coverage':i+=1;continue
        assert a['action']=='measure'
        start=i;p=tuple(a['position']);assert p==points[len(result)]
        channels=[]
        while i<len(h) and h[i]['phase']=='coverage' and tuple(h[i]['position'])==p:
            assert h[i]['action']=='measure' and h[i]['channel'] not in channels
            channels.append(h[i]['channel']);i+=1
        result.append(dict(start=start,end=i,position=p,channels=channels))
    return result

def analyze(s,row):
    h=s['action_history'];params=s['strategy_parameters'];points=list(map(tuple,s['coverage_points']))
    scans=blocks(h,points);assert len(scans)==s['coverage_points_visited']
    epochs=params['joint_visibility_resolver_log'];early=params['early_service_log'];chains=params['chain_route_log']
    requested={e['after_actual_action_count'] for e in epochs}|{b['end'] for b in scans}|{e['after_actual_action_count'] for e in params['discovery_stop_log']}
    requested|={0,len(h)}
    snapshots={};known=set();cleared=set();near=set();regions={};fees=[];phases=defaultdict(list)
    p=(0.,0.);tuned=1;elapsed=0.;first16=None
    unknown_scan=known_unready_scan=known_ready_scan=0
    counts=Counter();after_scans=[]
    end_to_visit={b['end']:i+1 for i,b in enumerate(scans)};visited=0
    for i in range(len(h)+1):
        if i in end_to_visit:visited=end_to_visit[i]
        if i in requested:
            circles={c:(tuple(r.enclosing_disk().center),r.enclosing_disk().radius) for c,r in regions.items() if r.vertices}
            snapshots[i]=dict(known=set(known),cleared=set(cleared),near=set(near),circles=circles,visited=visited,position=p)
            if i in end_to_visit:
                live=known-cleared-near
                wide=[c for c in live if c in circles and circles[c][1]>40.]
                after_scans.append(dict(prefix=i,visited=visited,known=len(known),wide_channels=wide,
                    radii={str(c):circles[c][1] for c in wide}))
        if i==len(h):break
        a=h[i];q=tuple(a['position']);c=a['channel'];kind=a['action']
        f=dict.fromkeys(COMP,0.)
        f['movement_s']=round(dist(p,q)/5.*1e6)/1e6
        if kind=='measure':
            f['detection_s']=5.;f['switching_s']=float(tuned!=c);tuned=c
            if a['phase']=='coverage':
                if c not in known:unknown_scan+=1
                elif c in near or c in regions and regions[c].vertices and regions[c].enclosing_disk().radius<=19.9: known_ready_scan+=1
                else:known_unready_scan+=1
            if a['result'] in {'direction','near'}:
                known.add(c)
                if a['result']=='near':near.add(c)
                else:regions.setdefault(c,CandidateRegion()).observe(q,a['bearing_deg'])
        else:
            f['optical_s']=3.
            if a['result']=='success':f['removal_s']=2.;known.add(c);cleared.add(c)
        elapsed+=amount(f);assert abs(a['virtual_time_s']-elapsed)<3e-5
        fees.append(f);phases[a['phase']].append(f);counts[a['phase']]+=1;p=q
        if len(known)==16 and first16 is None:first16=i+1
    assert len(cleared)==row['source_total']==row['cleared_total']
    assert abs(elapsed-row['virtual_time_s'])<3e-5
    for k in COMP:assert abs(total(fees)[k]-row[k])<3e-5
    service=[];used=set()
    for e in epochs:
        start,end,c=e['after_actual_action_count'],e['end_actual_action_count'],e['channel'];ss=snapshots[start]
        assert not used.intersection(range(start,end));used.update(range(start,end))
        radius=ss['circles'].get(c,(None,None))[1]
        if c in ss['near']:bucket='near'
        elif radius is None:bucket='missing'
        elif radius<=19.9:bucket='certified'
        elif radius<=40.:bucket='r19.9_40'
        elif radius<=120.:bucket='r40_120'
        else:bucket='r_over120'
        origin='early' if any(x['channel']==c and x['after_actual_action_count']==start and x['end_actual_action_count']==end for x in early) else 'chain'
        chain=[x for x in chains if x['after_actual_action_count']==start and x['selected_kind']=='source' and x['selected_channel']==c]
        assert origin=='early' or chain
        remaining=0 if len(ss['known'])==16 else len(points)-ss['visited']
        ft=total(fees[start:end]);acts=h[start:end]
        service.append(dict(id=e['id'],channel=c,start=start,end=end,origin=origin,entry_bucket=bucket,
            radius_m=radius,known_at_entry=len(ss['known']),remaining_cover_obligations=remaining,
            physically_unvisited_stations=len(points)-ss['visited'],components=ft,total_s=amount(ft),
            active_measurements=sum(a['action']=='measure' and a['phase']=='active_localization' for a in acts),
            first_action_phase=acts[0]['phase'] if acts else None,
            optical_grid_actions=sum(a['phase'] in {'guaranteed_clearance','joint_visibility_optical'} for a in acts),
            entry_movement_s=fees[start]['movement_s'] if end>start else 0.,
            internal_movement_s=math.fsum(f['movement_s'] for f in fees[start+1:end]),status=e['status']))
    cover_indices={j for b in scans for j in range(b['start'],b['end'])}
    assert used|cover_indices==set(range(len(h))) and not used.intersection(cover_indices)
    prefix_cover_move=0.;detour=0.;segments=[];prev_end=0;prev_point=(0.,0.)
    for b in scans:
        direct=dist(prev_point,b['position'])/5.
        actual=math.fsum(f['movement_s'] for f in fees[prev_end:b['start']+1])
        extra=actual-direct
        assert extra>=-2e-5
        segments.append(dict(start=prev_end,end=b['start']+1,direct_s=direct,actual_s=actual,excess_s=extra))
        prefix_cover_move+=direct;detour+=extra;prev_end=b['end'];prev_point=b['position']
    tail_move=math.fsum(f['movement_s'] for f in fees[prev_end:])
    assert abs(prefix_cover_move+detour+tail_move-total(fees)['movement_s'])<1e-6
    stops=params['discovery_stop_log'];assert len(stops)==int(first16 is not None)
    cap=None
    if stops:
        stop=stops[0];n=stop['after_actual_action_count'];assert len(snapshots[n]['known'])==16
        assert not any(a['phase']=='coverage' for a in h[n:])
        after=total(fees[n:]);remaining=stop['omitted_cover_stations']
        cap=dict(first16_prefix=first16,stop_prefix=n,omitted_stations=remaining,
            finish_current_scan_after16_actions=n-first16,finish_current_scan_after16_s=amount(total(fees[first16:n])),
            post_stop_components=after,post_stop_total_s=amount(after),
            post_stop_resolvers=sum(e['start']>=n for e in service),
            post_stop_wide_resolvers=sum(e['start']>=n and e['radius_m'] is not None and e['radius_m']>40 and e['entry_bucket']!='near' for e in service),
            later_coverage_actions=0)
    return dict(seed=row['seed'],N=row['source_total'],total_s=elapsed,components=total(fees),
        phase_components={k:dict(actions=len(v),**total(v)) for k,v in phases.items()},
        scans=len(scans),unknown_scan=unknown_scan,known_unready_scan=known_unready_scan,known_ready_scan=known_ready_scan,
        services=service,cover_prefix_direct_movement_s=prefix_cover_move,between_cover_excursion_excess_movement_s=detour,
        post_last_visited_cover_movement_s=tail_move,segments=segments,after_scans=after_scans,discovery_cap=cap)

def aggregate(cases):
    size=len(cases);services=[e for c in cases for e in c['services']]
    groups={}
    for key in sorted({(e['origin'],e['entry_bucket'],e['remaining_cover_obligations']>0) for e in services}):
        values=[e for e in services if (e['origin'],e['entry_bucket'],e['remaining_cover_obligations']>0)==key]
        groups['/'.join(map(str,key))]=dict(epochs=len(values),epochs_per_case=len(values)/size,
            total_s_per_case=math.fsum(e['total_s'] for e in values)/size,
            components_s_per_case={k:math.fsum(e['components'][k] for e in values)/size for k in COMP},
            active_measures_per_case=sum(e['active_measurements'] for e in values)/size,
            grid_actions_per_case=sum(e['optical_grid_actions'] for e in values)/size,
            entry_movement_s_per_case=math.fsum(e['entry_movement_s'] for e in values)/size,
            internal_movement_s_per_case=math.fsum(e['internal_movement_s'] for e in values)/size)
    phase_names=sorted({k for c in cases for k in c['phase_components']})
    phases={p:{k:math.fsum(c['phase_components'].get(p,{}).get(k,0.) for c in cases)/size for k in ('actions',*COMP)} for p in phase_names}
    cap=[c['discovery_cap'] for c in cases if c['discovery_cap']]
    broad=lambda e:e['entry_bucket'] in {'r40_120','r_over120'}
    return dict(cases=size,mean_total_s=statistics.fmean(c['total_s'] for c in cases),
        components={k:statistics.fmean(c['components'][k] for c in cases) for k in COMP},phase_components=phases,
        resolver_groups=groups,mean_scans=statistics.fmean(c['scans'] for c in cases),
        mean_unknown_scans=statistics.fmean(c['unknown_scan'] for c in cases),
        mean_known_unready_scans=statistics.fmean(c['known_unready_scan'] for c in cases),mean_known_ready_scans=statistics.fmean(c['known_ready_scan'] for c in cases),
        all_resolvers=len(services),wide_resolvers=sum(broad(e) for e in services),
        wide_resolver_cases=sum(any(broad(e) for e in c['services']) for c in cases),
        wide_resolver_first_active_cases=sum(any(broad(e) and e['active_measurements'] for e in c['services']) for c in cases),
        wide_resolver_total_s_per_case=math.fsum(e['total_s'] for e in services if broad(e))/size,
        wide_resolver_active_per_case=sum(e['active_measurements'] for e in services if broad(e))/size,
        ready_resolver_total_s_per_case=math.fsum(e['total_s'] for e in services if e['entry_bucket'] in {'near','certified'})/size,
        wide_at_coverage_cases=sum(any(b['wide_channels'] for b in c['after_scans']) for c in cases),
        wide_unique_channels_at_coverage_per_case=statistics.fmean(len(set(ch for b in c['after_scans'] for ch in b['wide_channels'])) for c in cases),
        moving_decomposition={k:statistics.fmean(c[k] for c in cases) for k in ('cover_prefix_direct_movement_s','between_cover_excursion_excess_movement_s','post_last_visited_cover_movement_s')},
        cap=dict(cases=len(cap),omitted_stations=describe([x['omitted_stations'] for x in cap]),
            post_stop_total_s=describe([x['post_stop_total_s'] for x in cap]),
            finish_current_scan_s=describe([x['finish_current_scan_after16_s'] for x in cap]),
            no_later_coverage=all(x['later_coverage_actions']==0 for x in cap),
            post_stop_wide_resolvers=sum(x['post_stop_wide_resolvers'] for x in cap)),
        mean_time_per_source_s=statistics.fmean(c['total_s']/c['N'] for c in cases),
        uniform_seconds_per_case_to_reach_mean460=(statistics.fmean(c['total_s']/c['N'] for c in cases)-460)/statistics.fmean(1/c['N'] for c in cases),
        total_minus460N_mean_s=statistics.fmean(c['total_s']-460*c['N'] for c in cases))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    case_store={}
    report=dict(schema='r12-reference-observed-phase-diagnostic-v1',runtime_commit='81aa6e1a',
        scope='Entire already completed reference; only summary/row/accepted action logs used, never ground-truth coordinates. Accounting opportunity ceilings are not recoverable savings or policy comparisons.',
        script_sha256=sha(Path(__file__)),geometry_source_sha256=GEOMETRY_SHA,inputs={},splits={})
    for split,count in (('confirmation',140),('stress',98)):
        folder=REFERENCE/'results/q4_per_source_reference'/split
        summary=json.loads((folder/'summary.json').read_bytes());audit=json.loads((folder/'independent_audit.json').read_bytes())
        assert summary['complete'] and summary['source_unchanged'] and summary['runs']==count and summary['all_clear']
        assert audit['all_passed'] and audit['records']==audit['passed_records']==count
        assert not summary['infrastructure_errors'] and not audit['errors']
        assert len(summary['rows'])==count and len({r['seed'] for r in summary['rows']})==count
        cases=[];hashes={}
        for row in sorted(summary['rows'],key=lambda r:r['seed']):
            assert row['successful'] and row['all_cleared'] and row['accepted_exit']
            name=f"records/{row['strategy']}-{row['seed']}.json.gz";path=folder/name
            digest=sha(path);assert audit['input_sha256'][name]==digest
            record=json.loads(gzip.decompress(path.read_bytes()))
            assert record['row']==row
            cases.append(analyze(record['summary'],row));hashes[name]=digest
        report['inputs'][split]=dict(directory=str(folder),summary_sha256=sha(folder/'summary.json'),audit_sha256=sha(folder/'independent_audit.json'),records=hashes)
        report['splits'][split]=dict(aggregate=aggregate(cases),by_N={str(n):aggregate([c for c in cases if c['N']==n]) for n in range(10,17)})
        case_store[split]=cases
        print(split,json.dumps(report['splits'][split]['aggregate'],ensure_ascii=False),flush=True)
    sidecar=args.output.with_name(args.output.stem+'-cases.json.gz')
    blob=gzip.compress(json.dumps(case_store,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8'),mtime=0)
    with sidecar.open('xb') as f:f.write(blob)
    report['case_accounting']=dict(path=sidecar.name,sha256=hashlib.sha256(blob).hexdigest(),records=sum(map(len,case_store.values())))
    with args.output.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':main()
