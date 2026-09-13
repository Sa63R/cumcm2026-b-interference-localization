"""Observation-only scheduling experiments; safety/localization unchanged.

CandidateState is an online.State subclass. Strategies only reorder lawful
actions. Full fixed-layout scanning (unless 16 sources found) is preserved.
"""
from __future__ import annotations
import math
import sys
from functools import lru_cache
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'q4_comparison'))
import common
import q4_baseline as b
from online import State
from q4_route_cached import multi_route
from q4_v4_local import localize_step
from q4_information import expected_radius_gain


@lru_cache(None)
def _sample_worlds():
    """Small fixed quadrature for ranking only; all actual cases may differ."""
    worlds=[]
    for ri in range(8):
        radius=1800*math.sqrt((ri+.5)/8)
        for ai in range(24):
            g=b.mul(radius,b.unit(2*math.pi*(ai+.37)/24))
            for r in (1000.,1250.,1500.):
                worlds.append((g,r,None))
                for di in range(8):worlds.append((g,r,b.unit(2*math.pi*(di+.13)/8)))
    return tuple(worlds)


@lru_cache(1024)
def _visibility_masks(point):
    omni=direct=0
    for i,(g,r,direction) in enumerate(_sample_worlds()):
        if b.dist(g,point)>r:continue
        if direction is None:omni|=1<<i
        elif b.dot(direction,b.sub(point,g))>=0:direct|=1<<i
    return omni,direct


class _FirstMove(BaseException):
    pass


def action_entry(tr, here, config):
    """Dry first movement only: stops before any feedback or belief update."""
    class Preview:
        position = here
        def move(self, p):
            raise _FirstMove(p)
    try:
        localize_step(Preview(), tr, config)
    except _FirstMove as hit:
        return hit.args[0]
    raise RuntimeError('No first action to preview')


def directed_route(here, entries, exits, starts=8):
    """Short deterministic open route for jobs with separate entry and exit."""
    n=len(entries)
    d=[[b.dist(exit,entry) for entry in entries] for exit in exits+[here]]
    def length(order):
        return sum(d[a][z] for a,z in zip([n]+order[:-1], order))
    best=None; best_cost=math.inf
    for first in sorted(range(n),key=lambda i:(d[n][i],i))[:starts]:
        order=[first]; todo=set(range(n))-{first}
        while todo:
            j=min(todo,key=lambda k:(d[order[-1]][k],k)); todo.remove(j); order.append(j)
        value=length(order)
        for _ in range(8):
            changed=False
            for i in range(n):
                for j in range(i+1,n):
                    trial=order[:i]+list(reversed(order[i:j+1]))+order[j+1:]
                    cost=length(trial)
                    if cost<value-1e-7:
                        order,value=trial,cost; changed=True
            if not changed: break
        if value<best_cost: best,best_cost=order,value
    return best or []


class ScheduleMixin:
    def __init__(self, *args, strategy='baseline', **kwargs):
        self.strategy=strategy
        self.last_target=None
        super().__init__(*args,**kwargs)
        if strategy=='share_none':
            self.config.shared_at_stations=False
            self.config.shared_at_probes=False
            self.config.shared_at_clears=False
        elif strategy.startswith('share_'):
            _,potential,gain = strategy.split('_')
            self.config.channel_potential_radius=float(potential)
            self.config.share_min_gain=float(gain)
        elif strategy.startswith('defer_'):
            _,rad,detour,potential,gain=strategy.split('_')
            self.defer_radius=float(rad)
            self.defer_detour=float(detour)
            self.config.channel_potential_radius=float(potential)
            self.config.share_min_gain=float(gain)
        elif strategy.startswith('gainwait_'):
            _,fraction,minimum,detour=strategy.split('_')
            self.wait_fraction=float(fraction)
            self.wait_minimum=float(minimum)
            self.wait_detour=float(detour)
        elif strategy.startswith('station_'):
            if strategy not in ('station_inner','station_outer'):
                _,minimum,power,maxratio,prefix=strategy.split('_')
                self.station_minimum=int(minimum)
                self.station_power=float(power)
                self.station_maxratio=float(maxratio)
                self.station_prefix=int(prefix)
            for site in self.sites:_visibility_masks(tuple(site))

    def execute(self, raw_device, action):
        super().execute(raw_device,action)
        self.last_target=action[1] if action[0]=='target' and action[1] in self.pending else None

    def ordered_actions(self,device):
        if not self.visited:return [('site',0)]
        if self.strategy=='baseline' or self.strategy.startswith('share_'):return super().ordered_actions(device)
        if self.strategy.startswith('station_'):
            actions=super().ordered_actions(device)
            if actions[0][0]!='site':return actions
            if self.strategy=='station_inner':key=min(self.remaining,key=lambda k:(b.norm(self.sites[k])>1500,b.dist(device.position,self.sites[k]),k))
            elif self.strategy=='station_outer':key=min(self.remaining,key=lambda k:(b.norm(self.sites[k])<1500,b.dist(device.position,self.sites[k]),k))
            else:
                if len(self.pending)+len(self.cleared)<self.station_minimum or len(self.visited)>self.station_prefix:return actions
                om=dr=0
                for point in self.scanpoints:
                    a,z=_visibility_masks(tuple(point));om|=a;dr|=z
                base_distance=b.dist(device.position,self.sites[actions[0][1]])
                def score(k):
                    distance=b.dist(device.position,self.sites[k])
                    if distance>self.station_maxratio*base_distance+50:return -1.
                    a,z=_visibility_masks(tuple(self.sites[k]))
                    # 50:50 source type mass despite 1 omni vs 8 angular samples.
                    gain=8*(a&~om).bit_count()+(z&~dr).bit_count()
                    cost=distance+30*len(self.unknown)+1
                    return gain/(cost**self.station_power)
                key=max(sorted(self.remaining),key=score)
                if score(key)<0:return actions
            selected=('site',key)
            return [selected]+[a for a in actions if a!=selected]
        if self.strategy.startswith('gainwait_'):
            actions=super().ordered_actions(device)
            if not self.remaining or actions[0][0]=='site':return actions
            typ,k=actions[0];tr=self.pending[k]
            if tr.obs.status=='strong':return actions
            center,radius=b.enclosing_circle(tr.poly)
            if radius<=b.CLEAR_CERT_RADIUS:return actions
            site_action=next(x for x in actions if x[0]=='site')
            site=self.sites[site_action[1]]
            gain=expected_radius_gain(tr.poly,tr.positives,tr.negatives,site)
            entry=action_entry(tr,device.position,self.config.local)
            detour=b.dist(device.position,entry)+b.dist(entry,site)-b.dist(device.position,site)
            if gain>max(self.wait_minimum,self.wait_fraction*radius) and detour>self.wait_detour:
                return [site_action]+[a for a in actions if a!=site_action]
            return actions
        if self.strategy.startswith('defer_'):
            actions=super().ordered_actions(device)
            if not self.remaining or actions[0][0]=='site':return actions
            site_action=next(x for x in actions if x[0]=='site')
            nextsite=self.sites[site_action[1]]
            allowed=[]; deferred=[]
            for action in actions:
                if action[0]=='site':allowed.append(action);continue
                tr=self.pending[action[1]]
                if tr.obs.status=='strong':allowed.append(action);continue
                center,radius=b.enclosing_circle(tr.poly)
                entry=action_entry(tr,device.position,self.config.local)
                detour=b.dist(device.position,entry)+b.dist(entry,nextsite)-b.dist(device.position,nextsite)
                if radius>self.defer_radius and detour>self.defer_detour:deferred.append(action)
                else:allowed.append(action)
            return allowed+deferred
        keys=[('site',i) for i in sorted(self.remaining)]+[('target',c) for c in sorted(self.pending)]
        centers=[self.sites[k] if typ=='site' else self.pending[k].center() for typ,k in keys]
        if self.strategy=='continue' and self.last_target in self.pending:
            return [('target',self.last_target)]+[x for x in keys if x!=('target',self.last_target)]
        if self.strategy in ('continue','nearest','stations_first','targets_first'):
            if self.strategy=='continue':return super().ordered_actions(device)
            if self.strategy=='nearest':order=sorted(range(len(keys)),key=lambda i:(b.dist(device.position,centers[i]),i))
            else:
                priority='site' if self.strategy=='stations_first' else 'target'
                order=multi_route(device.position,list(range(len(keys))),centers,self.config.route_starts)
                order=sorted(order,key=lambda i:keys[i][0]!=priority)
            return [keys[i] for i in order]
        entries=[self.sites[k] if typ=='site' else action_entry(self.pending[k],device.position,self.config.local) for typ,k in keys]
        if self.strategy=='entry_nearest':order=sorted(range(len(keys)),key=lambda i:(b.dist(device.position,entries[i]),i))
        elif self.strategy=='service_route':order=directed_route(device.position,entries,centers,self.config.route_starts)
        else:
            weight={'entry':1.,'entry_half':.5,'entry_quarter':.25,'entry_threequarter':.75}[self.strategy]
            points=[b.add(b.mul(weight,a),b.mul(1-weight,z)) for a,z in zip(entries,centers)]
            order=multi_route(device.position,list(range(len(keys))),points,self.config.route_starts)
        return [keys[i] for i in order]

    def report(self):
        return dict(super().report(),schedule_strategy=self.strategy)


class CandidateState(ScheduleMixin, State):
    pass
