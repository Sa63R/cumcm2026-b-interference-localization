"""Position quadrature with exact conditional radius/direction sampling.
Planning prior: uniform area, radius, orientation; 50/50 types conditioned on
both existing; uniform count 10..16. Zero sampled mass causes policy fallback.
"""
from __future__ import annotations
import math, random, copy
import common
import q4_baseline as b
from radius_direction import marginalize, count_and_subset_sampler

def angle_difference(a,z): return (a-z+math.pi)%(2*math.pi)-math.pi

def compatible_position(g, history):
    if b.norm(g)>1800.+1e-8: return False
    for kind,p,result,theta in history:
        d=b.dist(g,p)
        if kind=='clear':
            if bool(d<=20.+1e-8)!=result: return False
        elif result=='strong':
            if d>5.+1e-9: return False
        elif result=='bearing':
            if d<=5. or d>1500.: return False
            true=math.atan2(g[1]-p[1],g[0]-p[0])
            if abs(angle_difference(theta,true))>b.DELTA+1e-10: return False
    return True

def polygon_draw(poly, rng):
    origin=poly[0]
    triangles=list(zip(poly[1:-1],poly[2:]))
    weights=[abs((a[0]-origin[0])*(z[1]-origin[1])-(a[1]-origin[1])*(z[0]-origin[0])) for a,z in triangles]
    if not weights or sum(weights)<=1e-15: raise ValueError('Degenerate position posterior')
    a,z=rng.choices(triangles,weights)[0]
    u,v=rng.random(),rng.random()
    if u+v>1: u,v=1-u,1-v
    return (origin[0]+u*(a[0]-origin[0])+v*(z[0]-origin[0]),origin[1]+u*(a[1]-origin[1])+v*(z[1]-origin[1]))

class Pool:
    def __init__(self, points, masses, evidence):
        self.points=points;self.masses=masses;self.evidence=evidence
        self.weights=[m.mixture_evidence for m in masses]
    def target(self, c, history, rng, cleared=False):
        if not self.points: raise ValueError('No position posterior mass')
        ix=rng.choices(range(len(self.points)),self.weights)[0]
        g,m=self.points[ix],self.masses[ix]
        pos=[p for kind,p,status,_ in history if kind=='detect' and status!='none']
        neg=[p for kind,p,status,_ in history if kind=='detect' and status=='none']
        if rng.random()<m.posterior_directional:
            piece=rng.choices(m.pieces,[p.mass for p in m.pieces])[0]
            direction=b.unit(rng.uniform(piece.begin,piece.end))
            lo,hi=piece.radius_lower,min(1500.,piece.radius_upper_exclusive)
        else:
            direction=None
            lo=max([1000.]+[b.dist(g,p) for p in pos])
            hi=min([1500.]+[b.dist(g,p) for p in neg])
        if hi<=lo: raise ValueError('Empty radius interval')
        radius=rng.uniform(lo,math.nextafter(hi,lo))
        return b.Target(c,g,radius,direction,cleared)

def build_pool(history, belief, rng, particles=48, unknown_particles=192):
    pos=[p for kind,p,status,_ in history if kind=='detect' and status!='none']
    neg=[p for kind,p,status,_ in history if kind=='detect' and status=='none']
    unknown=belief is None;n=unknown_particles if unknown else particles
    points=[];masses=[];total_mass=0.;draws=0
    max_draws=n if unknown else max(512,n*64)
    for _ in range(max_draws):
        if unknown:
            g=b.mul(1800*math.sqrt(rng.random()),b.unit(rng.random()*2*math.pi))
        elif belief.poly and len(belief.poly)>=3:
            g=polygon_draw(belief.poly,rng)
        else:
            g=b.add(belief.anchor,b.mul(5*math.sqrt(rng.random()),b.unit(rng.random()*2*math.pi)))
        draws+=1
        if not compatible_position(g,history): continue
        m=marginalize(g,pos,neg);total_mass+=m.mixture_evidence
        if m.mixture_evidence>1e-15:
            points.append(g);masses.append(m)
        if not unknown and len(points)>=n: break
    if not unknown and len(points)<min(8,n): raise ValueError('Insufficient known-source posterior support')
    return Pool(points,masses,total_mass/max(1,draws))

class ConditionalSimulator(b.LocalSimulator):
    def __init__(self, targets, seed, history, position, channel):
        super().__init__(targets,seed,'hash',False)
        self.position=position;self.channel=channel
        self.old_bearings={(c,tuple(p)):theta for c,records in history.items()
                           for kind,p,status,theta in records if kind=='detect' and status=='bearing'}
    def _error(self,c):
        theta=self.old_bearings.get((c,tuple(self.position)))
        if theta is not None:
            g=self._targets[c].position
            true=math.atan2(g[1]-self.position[1],g[0]-self.position[0])
            error=angle_difference(theta,true)
            if abs(error)>b.DELTA+1e-8: raise ValueError('Sample conflicts with historic bearing')
            return error
        return super()._error(c)

class Worlds:
    def __init__(self,state,rng,particles=48,unknown_particles=192):
        self.state=state;self.pools={}
        for c,tr in {**state.archived,**state.pending}.items():
            self.pools[c]=build_pool(state.history[c],tr,rng,particles,unknown_particles)
        self.unknown=sorted(state.unknown);cache={};evidence=[]
        for c in self.unknown:
            hist=state.history.get(c,[]);key=tuple(hist)
            if key not in cache: cache[key]=build_pool(hist,None,rng,particles,unknown_particles)
            self.pools[c]=cache[key];evidence.append(self.pools[c].evidence)
        self.count_prob,self.select=count_and_subset_sampler(evidence,len(state.pending)+len(state.cleared))
    def sample(self,rng):
        for _ in range(128):
            n,ix=self.select(rng)
            channels=sorted(set(self.state.pending)|self.state.cleared)+[self.unknown[i] for i in ix]
            targets=[self.pools[c].target(c,self.state.history.get(c,[]),rng,c in self.state.cleared) for c in channels]
            n_direct=sum(t.direction is not None for t in targets)
            if 0<n_direct<n: return targets,rng.randrange(1<<60)
        raise ValueError('Cannot sample a scene containing both source types')
