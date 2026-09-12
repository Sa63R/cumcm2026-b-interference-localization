#!/usr/bin/env python3
"""Q3 improvements; self-built benchmark, NOT the official simulator.

The agent sees only the Backend interface, not simulator hidden states.
Coverage, set membership, and safe clearing remain deterministic. Candidate
measurement scoring is a heuristic and never authorizes a clear or an exit.
"""
from __future__ import annotations
import argparse, csv, itertools, json, math, time
from pathlib import Path
import numpy as np
import q3_base as b

TAU = 2*math.pi
COVER_RADIUS = 999.99  # deliberately less than the guaranteed 1000 m range


def hull(points: list[np.ndarray] | np.ndarray) -> np.ndarray:
    pts = sorted(set((float(x), float(y)) for x,y in points))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=float).reshape(-1,2)
    def cross(o,a,c):
        return (a[0]-o[0])*(c[1]-o[1])-(a[1]-o[1])*(c[0]-o[0])
    lo=[]; hi=[]
    for p in pts:
        while len(lo)>1 and cross(lo[-2],lo[-1],p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi)>1 and cross(hi[-2],hi[-1],p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1]+hi[:-1])


def outside_disk_hull(poly: np.ndarray, p: np.ndarray, radius: float=999.999) -> np.ndarray:
    """Conservative convex hull after excluding the OPEN disk.
    Near-boundary vertices are retained with outward numerical slack.

    Extreme points of this hull are outside polygon vertices or circle/edge
    intersections. The excluded disk is slightly smaller than 1000 m.
    """
    if len(poly)==0:
        return poly
    d2=np.sum((poly-p)**2,axis=1)
    threshold=(radius-1e-5)**2  # keep near-boundary vertices; avoid roundoff cuts
    if np.all(d2>=threshold):
        return poly  # its convex hull is unchanged even if edges cross the disk
    keep=[x.copy() for x,d in zip(poly,d2) if d >= threshold]
    for x,y in zip(poly,np.roll(poly,-1,axis=0)):
        v=y-x; w=x-p
        aa=float(v@v)
        if aa<1e-20: continue
        bb=2*float(v@w); cc=float(w@w)-radius*radius
        discr=bb*bb-4*aa*cc
        if discr>=0:
            root=math.sqrt(discr)
            for t in ((-bb-root)/(2*aa),(-bb+root)/(2*aa)):
                if -1e-10<=t<=1+1e-10:
                    keep.append(x+min(1.0,max(0.0,t))*v)
    if not keep:
        raise RuntimeError('Negative observation contradicts the feasible polygon.')
    return hull(keep)


def arc_inside(c: np.ndarray, r: float, d: np.ndarray, R: float) -> list[tuple[float,float]]:
    """Angles on circle (c,r) whose points are in closed disk (d,R)."""
    vec=d-c; dist=b.norm(vec)
    if dist+r <= R: return [(0.0,TAU)]
    if dist >= r+R: return []
    if r >= dist+R: return []  # at most a tangency, never an arc
    if dist < 1e-12: return []
    z=(dist*dist+r*r-R*R)/(2*dist*r)
    if z<=-1: return [(0.0,TAU)]
    if z>=1: return []
    mid=math.atan2(vec[1],vec[0])%TAU; half=math.acos(z)
    left,right=mid-half,mid+half
    if left<0: return [(0.0,right),(left+TAU,TAU)]
    if right>TAU: return [(0.0,right-TAU),(left,TAU)]
    return [(left,right)]


def merge_arcs(arcs: list[tuple[float,float]]) -> list[tuple[float,float]]:
    ans=[]
    for a,c in sorted(arcs):
        if ans and a<=ans[-1][1]+1e-12:
            ans[-1]=(ans[-1][0],max(ans[-1][1],c))
        else: ans.append((a,c))
    return ans


def arcs_contained(need: list[tuple[float,float]], have: list[tuple[float,float]]) -> bool:
    merged=merge_arcs(have)
    return all(any(a<=l+1e-12 and r<=z+1e-12 for a,z in merged) for l,r in need)


def certified_cover(positions: list[np.ndarray]) -> bool:
    """Continuous disk-union containment certificate, not a sampled grid.

    1. The whole outer circumference must be covered.
    2. No covering circle may have an exposed boundary ARC inside the domain.
    If there were an uncovered interior component, its boundary would contain
    such an exposed arc. Duplicate centers are removed first.
    0.01 m radial and 0.001 m domain margins protect roundoff.
    """
    centers=[]
    for p in positions:
        if not any(b.norm(p-q)<1e-7 for q in centers): centers.append(np.asarray(p))
    if not centers: return False
    origin=np.zeros(2); domain=1800.001; radius=COVER_RADIUS
    boundary=[]
    for p in centers: boundary.extend(arc_inside(origin,domain,p,radius))
    if not arcs_contained([(0.0,TAU)],boundary): return False
    for i,p in enumerate(centers):
        need=arc_inside(p,radius,origin,domain)
        if not need: continue
        have=[]
        for j,q in enumerate(centers):
            if i!=j: have.extend(arc_inside(p,radius,q,radius))
        if not arcs_contained(need,have): return False
    return True


def approximate_radius(poly: np.ndarray) -> float:
    """Cheap enclosing radius used ONLY to rank hypothetical measurements."""
    if not len(poly): return math.inf
    c=(poly.max(axis=0)+poly.min(axis=0))/2
    return float(np.linalg.norm(poly-c,axis=1).max())


def nearest_clear_point(p,vertices,fallback):
    R=20-1e-5
    def feasible(q):return np.max(np.linalg.norm(vertices-q,axis=1))<=R+1e-8
    if feasible(p):return p.copy()
    candidates=[fallback]
    for v in vertices:
        d=p-v;dist=b.norm(d)
        if dist>R:
            q=v+R*d/dist
            if feasible(q):candidates.append(q)
    for v,w in itertools.combinations(vertices,2):
        d=w-v;dist=b.norm(d)
        if dist<1e-10 or dist>2*R:continue
        mid=(v+w)/2;h=math.sqrt(max(0,R*R-dist*dist/4))
        off=np.array([-d[1],d[0]])*(h/dist)
        for q in (mid+off,mid-off):
            if feasible(q):candidates.append(q)
    q=min(candidates,key=lambda q:b.norm(q-p))
    if not feasible(q):raise RuntimeError('Invalid safe-clear point.')
    return q.copy()


def improved_route(start,points):
    if not points:return []
    coords=np.array([start]+points);dist=np.linalg.norm(coords[:,None]-coords[None,:],axis=2)
    n=len(points);initial=b.route_order(start,points)
    seeds=[[j+1 for j in initial]]
    nearest=np.argsort(dist[0,1:])[:min(4,n)]+1
    for first in nearest:
        remain=set(range(1,n+1));remain.remove(first);rt=[int(first)]
        while remain:
            j=min(remain,key=lambda j:dist[rt[-1],j]);rt.append(j);remain.remove(j)
        seeds.append(rt)
    def length(rt):return sum(dist[a,z] for a,z in zip([0]+rt,rt))
    best=None
    for rt in seeds:
        for _ in range(25):
            change=False
            for i in range(n):
                for j in range(i+1,n):
                    pre=0 if i==0 else rt[i-1];a,z=rt[i],rt[j]
                    delta=dist[pre,z]-dist[pre,a]
                    if j+1<n:delta+=dist[a,rt[j+1]]-dist[z,rt[j+1]]
                    if delta<-1e-7:rt[i:j+1]=reversed(rt[i:j+1]);change=True
            # Relocate one node (Or-opt of length 1).
            old=length(rt);improved=False
            for i in range(n):
                rest=rt[:i]+rt[i+1:]
                for j in range(n):
                    if j==i:continue
                    test=rest[:j]+[rt[i]]+rest[j:];new=length(test)
                    if new<old-1e-7:rt=test;improved=True;break
                if improved:break
            if not change and not improved:break
        cost=length(rt)
        if best is None or cost<best[0]:best=cost,rt.copy()
    return [j-1 for j in best[1]]


class OptimizedAgent(b.Agent):
    def __init__(self, env: b.Backend, info=True, lateral=False, dynamic=False,
                 opportunity=False, lateral_weight=0.5, cover_count=7,
                 rotate=False, commit=False, ring_radius=None, relocate=False,
                 better_route=False, exact_clear=False, opportunity_gain=0.0):
        super().__init__(env,'joint',True)
        if cover_count not in (6,7):
            raise ValueError('cover_count must be 6 or 7')
        self.info=info; self.lateral=lateral; self.dynamic=dynamic
        self.opportunity=opportunity; self.lateral_weight=lateral_weight
        self.rotate=rotate; self.commit=commit
        self.relocate=relocate; self.better_route=better_route
        self.exact_clear=exact_clear; self.opportunity_gain=opportunity_gain
        self.negative_positions={c:[] for c in b.CHANNELS}
        self.positive_positions={c:[] for c in b.CHANNELS}
        self.full_scans=[]
        self.cover=[q.copy() for q in b.COVER] if cover_count==7 else [1124*np.array([math.cos(TAU*k/6),math.sin(TAU*k/6)]) for k in range(6)]
        if ring_radius is not None:
            self.cover=[ring_radius*np.array([math.cos(TAU*k/cover_count),math.sin(TAU*k/cover_count)]) for k in range(cover_count)]
        if not certified_cover([np.zeros(2)]+self.cover):
            raise ValueError('The proposed initial scan layout does not cover the domain.')
        self.remaining_cover=list(range(len(self.cover)))
        self.pruned_covers=0; self.extra_full_scans=0; self.lateral_steps=0
        self.local_steps={c:0 for c in b.CHANNELS}

    def refine_information(self,c:int,poly:np.ndarray) -> np.ndarray:
        if not self.info: return poly
        for neg in self.negative_positions[c]:
            poly=outside_disk_hull(poly,neg)
            # R_c is unknown but FIXED: d(positive,g)<=R_c<d(negative,g).
            for pos in self.positive_positions[c]:
                normal=2*(neg-pos)
                bound=float(neg@neg-pos@pos)+1e-5
                poly=b.clip(poly,normal,bound)
                if not len(poly): raise RuntimeError('Empty positive/negative range intersection.')
        return poly

    def observe(self,c:int,cover_id:int|None=None,center_step:bool=False) -> None:
        p=self.env.pos.copy(); ans=self.env.detect(c)
        if ans.kind=='none':
            if center_step: raise RuntimeError('No signal at a certified center.')
            self.negative_positions[c].append(p)
            if c in self.tracks and self.info:
                t=self.tracks[c]; t.poly=self.refine_information(c,t.poly)
                t.center,t.radius=b.mec(t.poly)
            return
        if ans.kind=='strong': self.do_clear(c); return
        if ans.kind!='bearing' or ans.bearing is None:
            raise RuntimeError('Invalid bearing response.')
        self.positive_positions[c].append(p)
        if c in self.unseen:
            self.unseen.remove(c)
            poly=self.refine_information(c,b.first_polygon(p,ans.bearing))
            self.tracks[c]=b.Track(poly)
        elif c in self.tracks:
            t=self.tracks[c]; old_r=t.radius
            poly=b.wedge(t.poly,p,ans.bearing)
            if not len(poly): raise RuntimeError('Inconsistent bearing observations.')
            poly=self.refine_information(c,poly)
            t.poly=poly; t.center,t.radius=b.mec(poly); t.positives+=1
            if center_step:
                t.center_updates+=1
                if t.radius>old_r/(2*math.cos(b.A))+1e-3:
                    raise RuntimeError('Center contraction failed.')
        else: raise RuntimeError('Detected an absent or cleared channel.')
        self.tracks[c].measured_at.add(b.coord_key(p))
        self.count_certificate()

    def select_probe(self,c:int) -> tuple[np.ndarray,bool]:
        t=self.tracks[c]; p=self.env.pos
        # Bounded heuristic probes; the center fallback retains finite completion.
        if not self.lateral or self.local_steps[c]>=4 or t.radius<25:
            return t.center.copy(),True
        delta=t.center-p; d=b.norm(delta)
        if d<1e-7: return t.center.copy(),True
        u=delta/d; v=np.array([-u[1],u[0]])
        candidates=[t.center.copy()]
        for beta in (0.08,0.2,0.4):
            for sign in (-1,1): candidates.append(t.center+sign*beta*t.radius*v)
        # Candidate source locations are for scoring only, never a coverage proof.
        samples=[t.center,0.5*t.center+0.5*t.poly.mean(axis=0)]
        samples.extend(0.8*x+0.2*t.center for x in t.poly)
        best=None
        for q in candidates:
            if float(np.linalg.norm(t.poly-q,axis=1).max())>=999.9: continue
            if b.coord_key(q) in t.measured_at: continue
            costs=[]
            for g in samples:
                vec=g-q; dist=b.norm(vec)
                if dist<=5: costs.append(5.0); continue
                theta=math.atan2(vec[1],vec[0])
                # No independence or zero-mean assumption is used in safety logic.
                radii=[]
                for noise in (-b.A,0.0,b.A):
                    pp=b.wedge(t.poly,q,theta+noise)
                    if len(pp): radii.append(approximate_radius(pp))
                rp=float(np.mean(radii)) if radii else t.radius
                rem=max(0.0,math.log2(max(rp,19.5)/19.5))
                costs.append(max(0.0,dist-19.5)/5+6*rem+self.lateral_weight*rp/5)
            score=b.norm(q-p)/5+float(np.mean(costs))
            if best is None or score<best[0]: best=(score,q)
        if best is None: return t.center.copy(),True
        q=best[1]; center=b.norm(q-t.center)<1e-6
        return q.copy(),center

    def prune_for(self,scans:list[np.ndarray],remaining:list[int]) -> list[int]:
        keep=remaining.copy()
        # Greedy redundancy removal; certificate remains valid after each deletion.
        order=sorted(keep,key=lambda k:b.norm(self.cover[k]-self.env.pos),reverse=True)
        for k in order:
            positions=scans+[self.cover[j] for j in keep if j!=k]
            if certified_cover(positions): keep.remove(k)
        return keep

    def refresh_coverage(self) -> None:
        if not self.unseen: return
        if self.dynamic:
            new=self.prune_for(self.full_scans,self.remaining_cover)
            self.pruned_covers+=len(self.remaining_cover)-len(new)
            self.remaining_cover=new
            if not new:
                if not certified_cover(self.full_scans): raise RuntimeError('Missing union certificate.')
                self.absent|=self.unseen; self.unseen.clear()
        elif not self.remaining_cover:
            self.absent|=self.unseen; self.unseen.clear()

    def scan_stop(self,cover_id:int|None=None) -> None:
        old_active=set(self.tracks)
        for c in sorted(self.unseen,key=lambda x:(x!=self.env.channel,x)):
            if c in self.unseen: self.observe(c,cover_id)
        self.full_scans.append(self.env.pos.copy())
        for c in sorted(old_active,key=lambda x:(x!=self.env.channel,x)):
            if c in self.tracks and self.useful_bearing(c): self.observe(c)
        if cover_id is not None and cover_id in self.remaining_cover:
            self.remaining_cover.remove(cover_id)
        self.count_certificate(); self.refresh_coverage()

    def opportunistic_scan(self,exclude:int|None=None) -> None:
        if self.dynamic and self.unseen and self.remaining_cover:
            # A stop is worthwhile when its negative disk can replace >=1 planned stop.
            candidate=self.prune_for(self.full_scans+[self.env.pos.copy()],self.remaining_cover)
            if len(candidate)<len(self.remaining_cover):
                self.extra_full_scans+=1; self.scan_stop(); return
        if self.opportunity:
            for c in sorted(list(self.tracks),key=lambda x:(x!=self.env.channel,x)):
                if c!=exclude and c in self.tracks and self.useful_bearing(c):
                    t=self.tracks[c]; delta=t.center-self.env.pos
                    pp=b.wedge(t.poly,self.env.pos,math.atan2(delta[1],delta[0]))
                    rp=approximate_radius(pp)
                    if rp<=b.CLEAR_GUARD or t.radius-rp>self.opportunity_gain:
                        self.observe(c)

    def order(self,start,points):
        if not self.better_route:return b.route_order(start,points)
        return improved_route(start,points)

    def relocate_cover_points(self):
        if not self.unseen or not self.remaining_cover:return
        covers=self.remaining_cover
        nodes=[self.cover[k] for k in covers]+[t.center for t in self.tracks.values()]
        order=self.order(self.env.pos,nodes)
        for k in covers.copy():
            idx=covers.index(k); loc=order.index(idx)
            prev=self.env.pos if loc==0 else nodes[order[loc-1]]
            nxt=nodes[order[loc+1]] if loc+1<len(order) else None
            old=self.cover[k]
            candidates=[]
            if nxt is None:candidates.append(prev)
            else:
                d=nxt-prev; dd=float(d@d)
                proj=prev if dd<1e-10 else prev+np.clip(float((old-prev)@d)/dd,0,1)*d
                candidates.extend([proj,0.5*(old+proj)])
            for t in sorted(self.tracks.values(),key=lambda t:b.norm(old-t.center))[:3]:
                candidates.extend([t.center,0.5*(old+t.center)])
            def local(q):return b.norm(prev-q)+(0 if nxt is None else b.norm(q-nxt))
            best_cost=local(old);best_q=old
            for q in candidates:
                cost=local(q)
                if cost>=best_cost-1.0:continue
                proposed=self.full_scans+[q if j==k else self.cover[j] for j in covers]
                if certified_cover(proposed):best_cost,best_q=cost,q.copy()
            self.cover[k]=best_q;nodes[idx]=best_q

    def go_clear(self,c):
        if not self.exact_clear:return super().go_clear(c)
        t=self.tracks[c]
        if t.radius>b.CLEAR_GUARD:raise RuntimeError('Uncertified clear')
        q=nearest_clear_point(self.env.pos,t.poly,t.center)
        self.env.move(q);self.do_clear(c)

    def choose_rotation(self) -> None:
        if not self.rotate or not self.unseen: return
        best=None
        base=np.array(self.cover)
        for ang in np.linspace(0,TAU/len(self.cover),14,endpoint=False):
            co,si=math.cos(ang),math.sin(ang)
            pts=base@np.array([[co,si],[-si,co]])
            nodes=list(pts)+[t.center for t in self.tracks.values()]
            order=b.route_order(self.env.pos,nodes)
            coords=np.array([self.env.pos]+[nodes[k] for k in order])
            length=float(np.linalg.norm(np.diff(coords,axis=0),axis=1).sum())
            if best is None or length<best[0]:best=(length,pts)
        self.cover=list(best[1])

    def run(self) -> None:
        self.scan_stop(); self.choose_rotation()
        committed=None
        while self.unseen or self.tracks:
            self.steps+=1
            if self.steps>600: raise RuntimeError('Progress bound exceeded.')
            self.clear_at_current_position()
            if not self.unseen and not self.tracks: break
            covers=self.remaining_cover if self.unseen else []
            if self.relocate and covers:
                self.relocate_cover_points()
            nodes=[('cover',k,self.cover[k]) for k in covers]
            nodes += [('target',c,t.center) for c,t in sorted(self.tracks.items())]
            if not nodes: raise RuntimeError('No feasible next task.')
            if self.commit and committed in self.tracks:
                kind,ident,q='target',committed,self.tracks[committed].center
            else:
                idx=self.order(self.env.pos,[n[2] for n in nodes])[0]
                kind,ident,q=nodes[idx]
            if kind=='cover':
                committed=None; self.env.move(q); self.scan_stop(ident)
            else:
                committed=ident
                t=self.tracks[ident]
                if t.radius<=b.CLEAR_GUARD:
                    self.go_clear(ident)
                    self.opportunistic_scan()
                else:
                    q,center=self.select_probe(ident)
                    self.env.move(q); self.local_steps[ident]+=1
                    if not center:self.lateral_steps+=1
                    self.observe(ident,center_step=center)
                    if ident in self.tracks and self.tracks[ident].radius<=b.CLEAR_GUARD:
                        self.go_clear(ident)
                    self.opportunistic_scan(exclude=ident)
        if self.unseen or self.tracks or len(self.cleared|self.absent)!=20:
            raise RuntimeError('Incomplete channel certificate.')


CONFIGS={
    'baseline':None,
    'negative_info':dict(info=True),
    'lateral_only':dict(info=False,lateral=True),
    'info_lateral':dict(info=True,lateral=True),
    'dynamic_only':dict(info=False,dynamic=True),
    'info_dynamic':dict(info=True,dynamic=True),
    'full':dict(info=True,lateral=True,dynamic=True,opportunity=True),
    'full_no_lateral':dict(info=True,lateral=False,dynamic=True,opportunity=True),
    'full_rotate':dict(info=True,lateral=True,dynamic=True,opportunity=True,rotate=True),
    'full_commit':dict(info=True,lateral=True,dynamic=True,opportunity=True,commit=True),
    'full_six':dict(info=True,lateral=True,dynamic=True,opportunity=True,cover_count=6),
}



for rad in (1300,1450,1600,1730):
    CONFIGS[f'six_{rad}']=dict(info=True,lateral=True,dynamic=True,opportunity=True,cover_count=6,ring_radius=rad)
CONFIGS['relocate7']=dict(info=True,lateral=True,dynamic=True,opportunity=True,relocate=True)
CONFIGS['relocate6']=dict(info=True,lateral=True,dynamic=True,opportunity=True,cover_count=6,ring_radius=1450,relocate=True)
CONFIGS['route']=dict(info=True,lateral=True,dynamic=True,opportunity=True,better_route=True)
CONFIGS['exact_clear']=dict(info=True,lateral=True,dynamic=True,opportunity=True,exact_clear=True)
CONFIGS['selective']=dict(info=True,lateral=True,dynamic=True,opportunity=True,opportunity_gain=50)
CONFIGS['opportunity_only']=dict(info=False,opportunity=True)
CONFIGS['opportunity_lateral']=dict(info=False,opportunity=True,lateral=True)


CONFIGS['combined']=dict(info=True,lateral=True,dynamic=True,opportunity=True,relocate=True,better_route=True,exact_clear=True,opportunity_gain=30)
CONFIGS['combined_no_relocate']=dict(info=True,lateral=True,dynamic=True,opportunity=True,better_route=True,exact_clear=True,opportunity_gain=30)
CONFIGS['combined_no_info']=dict(info=False,lateral=True,dynamic=True,opportunity=True,relocate=True,better_route=True,exact_clear=True,opportunity_gain=30)
CONFIGS['combined_no_lateral']=dict(info=True,lateral=False,dynamic=True,opportunity=True,relocate=True,better_route=True,exact_clear=True,opportunity_gain=30)

def evaluate(seed:int,name:str,noise='hash',stress=False,sources=None,record=False):
    if sources is None:sources=b.make_case(seed,stress)
    env=b.ToySimulator(sources,seed,noise,record)
    agent=b.Agent(env) if CONFIGS[name] is None else OptimizedAgent(env,**CONFIGS[name])
    start=time.perf_counter();agent.run();runtime=time.perf_counter()-start
    if env._live:raise AssertionError(f'Uncleared sources: {env._live}')
    row=dict(seed=seed,mode=name,noise=noise,stress=stress,true_sources=len(sources),
             cleared=env.clears,virtual_seconds=env.virtual_time,
             seconds_per_source=env.virtual_time/env.clears,movement_metres=env.distance,
             detects=env.detects,switches=env.switches,failed_clears=env.failed,
             local_runtime_seconds=runtime,
             pruned_covers=getattr(agent,'pruned_covers',0),
             extra_full_scans=getattr(agent,'extra_full_scans',0),
             lateral_steps=getattr(agent,'lateral_steps',0))
    return row,env


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',type=int,default=40);ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--modes',nargs='+',choices=CONFIGS,default=['baseline','combined'])
    ap.add_argument('--out',type=Path,default=Path('results/comparison.csv'))
    ap.add_argument('--stress',action='store_true');ap.add_argument('--noise',default='hash')
    args=ap.parse_args()
    if args.cases<1:ap.error('--cases must be positive')
    rows=[];args.out.parent.mkdir(parents=True,exist_ok=True)
    for seed in range(args.start,args.start+args.cases):
        for name in args.modes:
            try:
                row,_=evaluate(seed,name,args.noise,args.stress);rows.append(row)
            except Exception:
                print(f'FAILED seed={seed} mode={name}',flush=True);raise
        if (seed-args.start+1)%10==0:print(f'{seed-args.start+1}/{args.cases}',flush=True)
    b.save_csv(args.out,rows)
    for name in args.modes:
        rr=[r for r in rows if r['mode']==name]
        print(name,{key:round(float(np.mean([r[key] for r in rr])),3) for key in
                    ['virtual_seconds','movement_metres','detects','local_runtime_seconds','pruned_covers','extra_full_scans','lateral_steps']},flush=True)

if __name__=='__main__':main()
