"""Observation-only coverage candidates. No unverified site may be skipped.

A rotation is allowed only immediately after the origin scan and is certified.
An on-route replacement is committed only when the remaining fixed sites plus
actual completed scans and the new position have a continuous box certificate.
"""
from __future__ import annotations
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'q4_comparison'))
import common
import q4_baseline as b
from online import State, HistoryDevice
from q4_coverage import certify
from q4_route_cached import multi_route


def route_length(position, points, starts=8):
    order=multi_route(position,list(range(len(points))),points,starts)
    return sum(math.dist(a,z) for a,z in zip([position]+[points[i] for i in order[:-1]],[points[i] for i in order]))


class CandidateState(State):
    def __init__(self,*args,rotate_initial=False,replace_sites=True,replace_at_probes=False,
                 rotation_choices=24,max_replacement_checks=4,replacement_distance=700.,**kwargs):
        self.rotate_initial=rotate_initial
        self.replace_sites=replace_sites
        self.replace_at_probes=replace_at_probes
        self.rotation_choices=rotation_choices
        self.max_replacement_checks=max_replacement_checks
        self.replacement_distance=replacement_distance
        self.coverage_checks=0
        self.coverage_rotated=False
        self.rotation_angle=0.
        self.replacements=[]
        super().__init__(*args,**kwargs)

    def _rotate(self,device):
        if not self.rotate_initial or self.coverage_rotated or self.visited != [0]:return
        self.coverage_rotated=True
        if len(self.pending)+len(self.cleared)==16:return
        points=[self.sites[j] for j in sorted(self.remaining)]+[tr.center() for tr in self.pending.values()]
        base=route_length(device.position,points,self.config.route_starts)
        best=base;best_sites=None;best_angle=0
        for k in range(1,self.rotation_choices):
            angle=(math.pi/2)*k/self.rotation_choices
            cs,sn=math.cos(angle),math.sin(angle)
            ss=[(x*cs-y*sn,x*sn+y*cs) for x,y in self.sites]
            points=[ss[j] for j in sorted(self.remaining)]+[tr.center() for tr in self.pending.values()]
            cost=route_length(device.position,points,self.config.route_starts)
            if cost<best-1e-6:best,best_sites,best_angle=cost,ss,angle
        if best_sites is not None:
            self.coverage_checks+=1
            if certify(self.scanpoints+[best_sites[j] for j in sorted(self.remaining)],max_depth=18)['ok']:
                self.sites=best_sites;self.rotation_angle=best_angle

    def _replace(self,raw_device):
        if not self.replace_sites or not self.remaining or not self.unknown:return
        if len(self.pending)+len(self.cleared)==16:return
        pos=raw_device.position
        candidates=sorted(self.remaining,key=lambda i:b.dist(self.sites[i],pos))
        candidates=[i for i in candidates if b.dist(self.sites[i],pos)<=self.replacement_distance]
        # Outer sites cannot be replaced by points inside the target disk under
        # this single-site guarantee; cheap distance rejection catches most.
        for i in candidates[:self.max_replacement_checks]:
            planned=self.scanpoints+[pos]+[self.sites[j] for j in sorted(self.remaining) if j!=i]
            self.coverage_checks+=1
            cert=certify(planned,max_depth=18)
            if not cert['ok']:continue
            device=HistoryDevice(raw_device,self.history)
            original=self.sites[i]
            self.sites[i]=pos
            self.remaining.remove(i)
            self.visited.append(i)
            self.shifted_sites+=1
            self.replacements.append(dict(site=i,original=original,position=pos))
            self.discovery(device)
            if self.config.shared_at_stations:self.shared(device)
            break

    def execute(self,raw_device,action):
        completed=len(self.cleared)
        super().execute(raw_device,action)
        if action[0]=='site' and len(self.visited)==1:self._rotate(raw_device)
        if action[0]=='target' and (self.replace_at_probes or len(self.cleared)>completed):self._replace(raw_device)

    def report(self):
        return dict(super().report(),coverage_checks=self.coverage_checks,rotation_angle=self.rotation_angle,
                    coverage_replacements=self.replacements)


class OpportunisticState(CandidateState):
    """Development-only: buy a bounded number of scans at reached clear points.

New scans only add evidence; removal still needs a full geometric certificate.
This can combine two reached points to replace one station, which single-point
replacement cannot accomplish. The extra scan count is bounded explicitly.
"""
    def __init__(self,*args,bonus_scans=6,bonus_distance=650.,bonus_unknown=20,**kwargs):
        super().__init__(*args,replace_sites=False,**kwargs)
        self.bonus_scans=bonus_scans;self.bonus_distance=bonus_distance;self.bonus_unknown=bonus_unknown
        self.bonus_used=0;self.pruned_sites=[]

    def execute(self,raw_device,action):
        completed=len(self.cleared)
        super().execute(raw_device,action)
        if action[0]!='target' or len(self.cleared)==completed:return
        if not self.remaining or not self.unknown or len(self.pending)+len(self.cleared)==16:return
        if self.bonus_used>=self.bonus_scans or len(self.unknown)>self.bonus_unknown:return
        pos=raw_device.position
        candidates=sorted(self.remaining,key=lambda i:b.dist(self.sites[i],pos))
        if not candidates or b.dist(self.sites[candidates[0]],pos)>self.bonus_distance:return
        if min((b.dist(p,pos) for p in self.scanpoints),default=math.inf)<150:return
        # This is an actual paid scan, never an assumed observation.
        self.bonus_used+=1;self.extra_actions+=1
        device=HistoryDevice(raw_device,self.history)
        self.discovery(device)
        for i in candidates:
            if b.dist(self.sites[i],pos)>1100:continue
            planned=self.scanpoints+[self.sites[j] for j in sorted(self.remaining) if j!=i]
            self.coverage_checks+=1
            if not certify(planned,max_depth=18)['ok']:continue
            self.remaining.remove(i);self.pruned_sites.append(i)

    def report(self):
        return dict(super().report(),bonus_scans=self.bonus_used,coverage_pruned_sites=self.pruned_sites)


def relocate_route(position,order,points,passes=12):
    """Best-improvement node relocations interleaved with 2-opt reversals."""
    n=len(points);dist=[[math.dist(a,z) for z in points+[position]] for a in points+[position]]
    route=list(order)
    for _ in range(passes):
        best_gain=1e-7;best_order=None
        for i,node in enumerate(route):
            before=n if i==0 else route[i-1]
            remove=dist[before][node]
            if i+1<len(route):
                after=route[i+1];remove+=dist[node][after]-dist[before][after]
            other=route[:i]+route[i+1:]
            for j in range(len(other)+1):
                if j==i:continue
                prev=n if j==0 else other[j-1]
                insert=dist[prev][node]
                if j<len(other):
                    following=other[j];insert+=dist[node][following]-dist[prev][following]
                gain=remove-insert
                if gain>best_gain:
                    best_gain=gain;best_order=other[:j]+[node]+other[j:]
        for i in range(len(route)-1):
            prev=n if i==0 else route[i-1];a=route[i]
            for j in range(i+1,len(route)):
                z=route[j];gain=dist[prev][a]-dist[prev][z]
                if j+1<len(route):
                    following=route[j+1];gain+=dist[z][following]-dist[a][following]
                if gain>best_gain:
                    best_gain=gain;best_order=route[:i]+list(reversed(route[i:j+1]))+route[j+1:]
        if best_order is None:break
        route=best_order
    return route


class RouteState(State):
    def ordered_actions(self,device):
        if not self.visited:return [('site',0)]
        keys=[('site',i) for i in sorted(self.remaining)]+[('target',c) for c in sorted(self.pending)]
        points=[self.sites[k] if typ=='site' else self.pending[k].center() for typ,k in keys]
        order=multi_route(device.position,list(range(len(keys))),points,self.config.route_starts)
        order=relocate_route(device.position,order,points)
        return [keys[i] for i in order]


class FlexibleState(State):
    """Pull each next station toward the already planned incoming/outgoing path.

Every proposed location is certified jointly with completed scans and all other
unvisited sites. Failed or unresolved checks preserve the original station.
"""
    def __init__(self,*args,shift_trials=(1.,.5,.25,.125,.0625),**kwargs):
        self.shift_trials=shift_trials;self.flex_checks=0;self.flex_attempts=0
        super().__init__(*args,**kwargs)

    def ordered_actions(self,device):
        actions=super().ordered_actions(device)
        if not self.visited or not actions or actions[0][0]!='site':return actions
        index=actions[0][1];old=self.sites[index];start=device.position
        if len(actions)>1:
            typ,k=actions[1]
            end=self.sites[k] if typ=='site' else self.pending[k].center()
        else:end=start
        vx,vy=end[0]-start[0],end[1]-start[1];vv=vx*vx+vy*vy
        t=max(0.,min(1.,((old[0]-start[0])*vx+(old[1]-start[1])*vy)/vv)) if vv else 0.
        goal=(start[0]+t*vx,start[1]+t*vy)
        if math.dist(old,goal)<10:return actions
        self.flex_attempts+=1
        base=self.scanpoints+[self.sites[j] for j in sorted(self.remaining) if j!=index]
        for alpha in self.shift_trials:
            point=(old[0]+alpha*(goal[0]-old[0]),old[1]+alpha*(goal[1]-old[1]))
            self.flex_checks+=1
            if certify(base+[point],max_depth=16)['ok']:
                return [('shift',index,point)]+actions[1:]
        return actions

    def report(self):
        return dict(super().report(),coverage_checks=self.flex_checks,flex_attempts=self.flex_attempts)


class TangentialState(FlexibleState):
    """Keep ring radius while shifting a station's angle toward the route.

Radial inward moves destroy the tight radius guarantee quickly. Tangential
moves use the inner ring's angular slack, with the same exact certificate.
"""
    def __init__(self,*args,angles=(1,2,4,8,12),**kwargs):
        self.angles=tuple(math.radians(x) for x in angles)
        super().__init__(*args,**kwargs)

    def ordered_actions(self,device):
        actions=State.ordered_actions(self,device)
        if not self.visited or not actions or actions[0][0]!='site':return actions
        index=actions[0][1];old=self.sites[index];start=device.position
        if math.hypot(*old)>1500:return actions
        if len(actions)>1:
            typ,k=actions[1];end=self.sites[k] if typ=='site' else self.pending[k].center()
        else:end=start
        base_cost=math.dist(start,old)+math.dist(old,end)
        candidates=[]
        for angle in self.angles:
            for signed in [angle,-angle]:
                c,s=math.cos(signed),math.sin(signed)
                point=(old[0]*c-old[1]*s,old[0]*s+old[1]*c)
                cost=math.dist(start,point)+math.dist(point,end)
                if cost<base_cost-1e-7:candidates.append((cost,point))
        if not candidates:return actions
        self.flex_attempts+=1
        fixed=self.scanpoints+[self.sites[j] for j in sorted(self.remaining) if j!=index]
        for _,point in sorted(candidates):
            self.flex_checks+=1
            if certify(fixed+[point],max_depth=16)['ok']:
                return [('shift',index,point)]+actions[1:]
        return actions
