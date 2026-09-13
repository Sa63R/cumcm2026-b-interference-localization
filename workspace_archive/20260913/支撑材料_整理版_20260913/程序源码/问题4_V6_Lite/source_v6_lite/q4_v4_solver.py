"""Q4 V4: replan globally after each paired RF localization stage.

The 21-point continuous discovery certificate, conservative uncertainty sets,
feedback-only access and optical-cover safety are inherited from V3. Changes:
(1) localization is interruptible; (2) default optical cover size is 12, not 6;
(3) pairwise route distances are cached without changing the routing rule.
The official HTTP protocol is NOT included or guessed.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import math
import q4_baseline as b
from q4_v3_solver import Belief as BaseBelief,validate_layout
from q4_v4_local import LocalConfig,localize_step

@dataclass
class V4Config:
    local:LocalConfig=field(default_factory=LocalConfig)
    ring_sites:tuple=(8,12,998,1865)
    route_starts:int=8
    cached_routing:bool=True
    shared_at_stations:bool=True
    shared_at_clears:bool=True
    shared_at_probes:bool=True
    share_min_distance:float=50.
    share_min_gain:float=10.
    channel_potential_radius:float=600.
    directional_negatives:bool=True
    direction_sectors:int=16

@dataclass
class Belief(BaseBelief):
    rf_rounds:int=0
    pair_failures:int=0

def solve_v4(device:b.Device,config:V4Config|None=None)->dict:
    config=config or V4Config()
    if config.route_starts<1:raise ValueError('route_starts must be positive')
    if min(config.share_min_distance,config.share_min_gain,config.channel_potential_radius)<0:
        raise ValueError('Distance and gain settings must be nonnegative')
    if config.local.optical_cover_limit<0:raise ValueError('Invalid optical cover limit')
    n1,n2,r1,r2=config.ring_sites
    sites=[(0.,0.)]+[b.mul(r,b.unit(2*math.pi*k/n)) for n,r in [(n1,r1),(n2,r2)] for k in range(n)]
    validate_layout(tuple(sites))
    if config.cached_routing:
        from q4_route_cached import multi_route
    else:
        from q4_route import multi_route
    remaining=set(range(len(sites)));unknown=set(range(1,21))
    pending:dict[int,Belief]={};cleared:set[int]=set()
    negatives={c:[] for c in range(1,21)}
    visited=[];scanpoints=[];localizations=[];shared_count=0;replans=0;actions=0

    def discovery():
        for c in sorted(unknown,key=lambda x:(x!=device.channel,x)):
            if len(cleared)+len(pending)==16:break
            obs=device.detect(c)
            if obs.status=='none':
                negatives[c].append(device.position)
                continue
            unknown.remove(c)
            poly=b.initial_polygon(device.position,obs.theta,1500.) if obs.status=='bearing' else []
            tr=Belief(channel=c,anchor=device.position,obs=obs,poly=poly,
                      measured=[device.position],positives=[device.position],negatives=negatives[c][:],
                      negative_enabled=config.directional_negatives,direction_sectors=config.direction_sectors)
            tr.tighten();pending[c]=tr
        scanpoints.append(device.position)

    def shared(exclude=None):
        nonlocal shared_count
        from q4_information import expected_radius_gain
        for c in sorted(pending,key=lambda x:(x!=device.channel,x)):
            if c==exclude:continue
            tr=pending[c]
            if tr.obs.status=='strong':continue
            if min(b.dist(device.position,p) for p in tr.measured)<config.share_min_distance:continue
            center,radius=b.enclosing_circle(tr.poly)
            if radius<=b.CLEAR_CERT_RADIUS:continue
            if b.dist(center,device.position)>config.channel_potential_radius+radius:continue
            if config.share_min_gain>0 and expected_radius_gain(tr.poly,tr.positives,tr.negatives,device.position)<config.share_min_gain:continue
            obs=device.detect(c);shared_count+=1;tr.add(device.position,obs)

    while remaining or pending:
        actions+=1
        # A scan consumes one site; a target action consumes one of <=12 RF
        # stages or completes its clear. This guard should never be approached.
        if actions>len(sites)+16*14+1:
            raise RuntimeError('Progress invariant violated; refusing an infinite loop')
        if len(cleared)+len(pending)==16:
            remaining.clear();unknown.clear()
        if not visited:
            typ,k='site',0
        else:
            keys=[('site',i) for i in sorted(remaining)]+[('target',c) for c in sorted(pending)]
            points=[sites[k] if typ=='site' else pending[k].center() for typ,k in keys]
            order=multi_route(device.position,list(range(len(keys))),points,config.route_starts)
            typ,k=keys[order[0]]
        if typ=='site':
            device.move(sites[k]);remaining.remove(k);visited.append(k)
            discovery()
            if config.shared_at_stations:shared()
        else:
            tr=pending[k]
            stats=localize_step(device,tr,config.local,
                                on_probe=(lambda:shared(exclude=k)) if config.shared_at_probes else None)
            if stats.get('complete') is False:
                replans+=1
                continue
            localizations.append(dict(channel=k,**stats))
            cleared.add(k);del pending[k]
            if len(cleared)==16:break
            if config.shared_at_clears:shared()
    return dict(stop_certificate='source_upper_bound' if len(cleared)==16 else 'coverage_complete',
                visited_stations=visited,localizations=localizations,shared_detections=shared_count,
                scanpoints=len(scanpoints),replans_after_pairs=replans,global_actions=actions)
