"""Joint search/localization scheduling for Q4.

solve_fast() sees the Device protocol only. All target positions, radii, counts,
and directions remain hidden. Default: 21 certified scan stations, shared
positive observations, joint 8-start 2-opt routing, and robust optical clears.
"""
from __future__ import annotations
import q4_baseline as b
from q4_fast_local import LocalConfig,localize
from dataclasses import dataclass,field
import math,statistics,json,time

@dataclass
class FastConfig:
    local: LocalConfig=field(default_factory=LocalConfig)
    shared_at_stations: bool=True
    shared_at_clears: bool=True
    extra_discovery: float=0.
    joint_route: bool=True
    ring_sites: tuple | None=(8,12,998,1865)
    channel_potential_radius: float=600.
    prune_sites: bool=False
    route_starts: int=8
    use_centroid: bool=False

@dataclass
class Belief:
    channel:int
    anchor:b.Point
    obs:b.Observation
    poly:list[b.Point]
    measured:list[b.Point]=field(default_factory=list)
    def center(self):
        return self.anchor if self.obs.status=='strong' else b.enclosing_circle(self.poly)[0]
    def add(self,p,obs):
        self.measured.append(p)
        if obs.status=='none':return
        self.anchor=p;self.obs=obs
        if obs.status=='bearing':
            self.poly=b.clip_bearing(self.poly,p,obs.theta,1500.)
            if not self.poly:raise RuntimeError('inconsistent shared measurement')


def make_sites(config):
    if config.ring_sites is None:return b.coverage_sites()[0]
    n1,n2,r1,r2=config.ring_sites
    return [(0.,0.)]+[b.mul(r,b.unit(2*math.pi*k/n)) for n,r in [(n1,r1),(n2,r2)] for k in range(n)]


from functools import lru_cache

@lru_cache(maxsize=16)
def validate_layout(sites: tuple[b.Point, ...]) -> None:
    """Fail closed: a custom site layout must pass continuous coverage checks."""
    from q4_coverage import certify
    result=certify(list(sites),max_depth=22)
    if not result['ok']:
        raise ValueError(f'Uncertified coverage layout: {result}')


def solve_fast(device: b.Device, config: FastConfig | None = None) -> dict:
    config=config or FastConfig()
    if config.extra_discovery<0 or config.channel_potential_radius<0 or config.route_starts<0:
        raise ValueError('Configuration distances and route_starts must be nonnegative')
    sites=make_sites(config)
    if config.ring_sites is not None:
        validate_layout(tuple(sites))
    remaining=set(range(len(sites)))
    unknown=set(range(1,21));pending={};cleared=set();visited=[];locstats=[];scanpoints=[]
    shared_count=0
    pruned=[];certificate_checks=0
    def discovery():
        for c in sorted(unknown,key=lambda x:(x!=device.channel,x)):
            obs=device.detect(c)
            if obs.status!='none':
                unknown.remove(c)
                poly=b.initial_polygon(device.position,obs.theta,1500.) if obs.status=='bearing' else []
                pending[c]=Belief(c,device.position,obs,poly,[device.position])
        scanpoints.append(device.position)
    def shared():
        nonlocal shared_count
        for c in sorted(pending,key=lambda x:(x!=device.channel,x)):
            tr=pending[c]
            if tr.obs.status=='strong':continue
            if min(b.dist(device.position,p) for p in tr.measured)<50.:continue
            cen,rad=b.enclosing_circle(tr.poly)
            if rad<=b.CLEAR_CERT_RADIUS:continue
            if b.dist(cen,device.position)>config.channel_potential_radius+rad:continue
            obs=device.detect(c);shared_count+=1
            tr.add(device.position,obs)
    while remaining or pending:
        if not visited:
            action=('site',0)
        elif not config.joint_route and pending:
            action=('target',min(pending,key=lambda c:b.dist(device.position,pending[c].center())))
        else:
            keys=[('site',i) for i in sorted(remaining)]+[('target',c) for c in sorted(pending)]
            from q4_route import multi_route,centroid
            points=[sites[k] if typ=='site' else
                    (centroid(pending[k].poly) if config.use_centroid and pending[k].obs.status!='strong' else pending[k].center())
                    for typ,k in keys]
            order=(multi_route(device.position,list(range(len(keys))),points,config.route_starts)
                   if config.route_starts else b.route_order(device.position,list(range(len(keys))),points))
            action=keys[order[0]]
        typ,k=action
        if typ=='site':
            device.move(sites[k]);remaining.remove(k);visited.append(k);discovery()
            if config.shared_at_stations:shared()
        else:
            tr=pending[k]
            stats=localize(device,b.Track(k,tr.anchor,tr.obs),True,config.local,tr.poly)
            locstats.append(dict(channel=k,**stats));cleared.add(k);del pending[k]
            if len(cleared)==16:break
            if config.shared_at_clears:shared()
            if config.prune_sites and remaining:
                from q4_coverage import certify
                newpoint=device.position
                # Only buy an extra full scan if it can replace a planned scan.
                candidates=sorted(remaining,key=lambda i:b.dist(sites[i],newpoint))
                for i in candidates:
                    if b.dist(sites[i],newpoint)>1100:continue
                    planned=scanpoints+[newpoint]+[sites[j] for j in sorted(remaining) if j!=i]
                    certificate_checks+=1
                    cert=certify(planned,max_depth=18)
                    if cert['ok']:
                        discovery()
                        remaining.remove(i);pruned.append(i)
                        break
            if config.extra_discovery and min(b.dist(device.position,p) for p in scanpoints)>=config.extra_discovery:
                discovery()
    return dict(stop_certificate='source_upper_bound' if len(cleared)==16 else 'coverage_complete',visited_stations=visited,localizations=locstats,shared_detections=shared_count,scanpoints=len(scanpoints),pruned_stations=pruned,coverage_checks=certificate_checks)

