"""Q4 V3: bounded search, conservative negative feedback, selective sharing.

Default configuration uses the unchanged 21-site continuous coverage guarantee.
Optical covering allows timed misses and declares completion ONLY on positive
clear feedback. Disable it with LocalConfig(optical_cover_limit=0) until an
adapter's optical failure semantics are confirmed. No hidden scenario data is
read by solve_v3(). Nondefault experimental route options are not recommended.
"""
from __future__ import annotations
import q4_baseline as b
from q4_v3_local import LocalConfig,localize
from dataclasses import dataclass,field
import math,statistics,json,time

@dataclass
class V3Config:
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
    shared_at_probes: bool=True
    directional_negatives: bool=True
    direction_sectors: int=16
    share_min_distance: float=50.
    share_gain_ratio: float=10.
    rotation_choices: int=0
    defer_ready: bool=False
    information_weight:float=0.
    lookahead:int=5
    stronger_route:bool=False
    known_bound_stop:bool=True
    entry_exit_route:bool=False

@dataclass
class Belief:
    channel:int
    anchor:b.Point
    obs:b.Observation
    poly:list[b.Point]
    measured:list[b.Point]=field(default_factory=list)
    positives:list[b.Point]=field(default_factory=list)
    negatives:list[b.Point]=field(default_factory=list)
    negative_enabled:bool=False
    direction_sectors:int=16
    negative_updates:int=0
    def center(self):
        return self.anchor if self.obs.status=='strong' else b.enclosing_circle(self.poly)[0]
    def add(self,p,obs):
        self.measured.append(p)
        if obs.status=='none':
            self.negatives.append(p)
        else:
            self.positives.append(p)
            self.anchor=p;self.obs=obs
            if obs.status=='bearing':
                self.poly=b.clip_bearing(self.poly,p,obs.theta,1500.)
                if not self.poly:raise RuntimeError('inconsistent shared measurement')
        self.tighten()
    def tighten(self):
        if self.negative_enabled and self.poly and self.obs.status!='strong':
            from q4_information import tighten_directional
            self.poly,used=tighten_directional(self.poly,self.positives,self.negatives,self.direction_sectors)
            self.negative_updates+=int(used)


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


def solve_v3(device: b.Device, config: V3Config | None = None) -> dict:
    config=config or V3Config()
    if config.extra_discovery<0 or config.channel_potential_radius<0 or config.route_starts<0:
        raise ValueError('Configuration distances and route_starts must be nonnegative')
    sites=make_sites(config)
    if config.ring_sites is not None:
        validate_layout(tuple(sites))
    remaining=set(range(len(sites)))
    unknown=set(range(1,21));pending={};cleared=set();visited=[];locstats=[];scanpoints=[]
    shared_count=0
    negatives={c:[] for c in range(1,21)}
    pruned=[];certificate_checks=0
    def discovery():
        for c in sorted(unknown,key=lambda x:(x!=device.channel,x)):
            if config.known_bound_stop and len(cleared)+len(pending)==16:break
            obs=device.detect(c)
            if obs.status!='none':
                unknown.remove(c)
                poly=b.initial_polygon(device.position,obs.theta,1500.) if obs.status=='bearing' else []
                pending[c]=Belief(c,device.position,obs,poly,[device.position],[device.position],negatives[c][:],config.directional_negatives,config.direction_sectors)
                pending[c].tighten()
            else:negatives[c].append(device.position)
        scanpoints.append(device.position)
    def shared(exclude=None):
        nonlocal shared_count
        for c in sorted(pending,key=lambda x:(x!=device.channel,x)):
            if c==exclude:continue
            tr=pending[c]
            if tr.obs.status=='strong':continue
            if min(b.dist(device.position,p) for p in tr.measured)<config.share_min_distance:continue
            cen,rad=b.enclosing_circle(tr.poly)
            if rad<=b.CLEAR_CERT_RADIUS:continue
            if b.dist(cen,device.position)>config.channel_potential_radius+rad:continue
            if config.share_gain_ratio>0:
                from q4_information import expected_radius_gain
                gain=expected_radius_gain(tr.poly,tr.positives,tr.negatives,device.position)
                if gain<config.share_gain_ratio:continue
            obs=device.detect(c);shared_count+=1
            tr.add(device.position,obs)
    while remaining or pending:
        if config.known_bound_stop and len(cleared)+len(pending)==16:
            remaining.clear();unknown.clear()
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
            if config.stronger_route or config.entry_exit_route:
                from q4_v3_route import matrix_route,predicted_entries
                entries=[[sites[k]] if typ=='site' else predicted_entries(pending[k],config.local) for typ,k in keys] if config.entry_exit_route else None
                order=matrix_route(device.position,points,entries,config.route_starts)
            if config.information_weight>0:
                from q4_information import expected_radius_gain
                from q4_route import length
                oldlen=length(device.position,order,points)
                bestscore=0.;bestfirst=order[0]
                for j in order[1:config.lookahead]:
                    typ0,k0=keys[j]
                    if typ0!='site':continue
                    p=points[j]
                    gain=0.
                    for tr in pending.values():
                        if tr.obs.status=='strong':continue
                        cen,rad=b.enclosing_circle(tr.poly)
                        if rad<=b.CLEAR_CERT_RADIUS or b.dist(cen,p)>config.channel_potential_radius+rad:continue
                        gain+=expected_radius_gain(tr.poly,tr.positives,tr.negatives,p)
                    if gain<=0:continue
                    proposed=[j]+[i for i in order if i!=j]
                    cost=length(device.position,proposed,points)-oldlen
                    score=cost-config.information_weight*gain
                    if score<bestscore:bestscore=score;bestfirst=j
                action=keys[bestfirst]
            else:action=keys[order[0]]
        typ,k=action
        if typ=='site':
            device.move(sites[k]);remaining.remove(k);visited.append(k);discovery()
            if k==0 and config.rotation_choices>0 and pending:
                from q4_route import multi_route,length
                base=sites[:];targetpts=[tr.center() for tr in pending.values()]
                bestroute=float('inf');best_sites=sites
                for ri in range(config.rotation_choices):
                    a=(math.pi/2)*ri/config.rotation_choices;ca,sa=math.cos(a),math.sin(a)
                    rotated=[(p[0]*ca-p[1]*sa,p[0]*sa+p[1]*ca) for p in base]
                    pts=[rotated[j] for j in sorted(remaining)]+targetpts
                    order=multi_route(device.position,list(range(len(pts))),pts,4)
                    value=length(device.position,order,pts)
                    if value<bestroute:bestroute=value;best_sites=rotated
                sites=best_sites
            if config.shared_at_stations:shared()
        else:
            tr=pending[k]
            stats=localize(device,tr,config.local,
                           on_probe=(lambda:shared(exclude=k)) if config.shared_at_probes else None)
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

