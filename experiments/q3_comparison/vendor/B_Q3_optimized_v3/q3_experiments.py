"""Continuation of Q3 optimization. Self-built tests, NOT official results."""
from __future__ import annotations
import math
import numpy as np
import q3_base as b
import q3_optimized as o

class PublicBackend:
    """Prevent accidental source/seed access from the decision policy."""
    def __init__(self,backend):self.__backend=backend
    @property
    def pos(self):return self.__backend.pos
    @property
    def channel(self):return self.__backend.channel
    def move(self,p):return self.__backend.move(p)
    def detect(self,c):return self.__backend.detect(c)
    def clear(self,c):return self.__backend.clear(c)

CONFIGS={'v2':{}}
for r in (1124,1170,1220,1300,1380,1450):
    CONFIGS['six'+str(r)]={'cover_count':6,'ring_radius':r}
for r in (1050,1120,1200,1300):
    CONFIGS['seven'+str(r)]={'ring_radius':r}
CONFIGS['rotate']={'rotate':True}
CONFIGS['commit']={'commit':True}
for gain in (0,10,60,100):CONFIGS['gain'+str(gain)]={'opportunity_gain':gain}

def make_agent(env,mode):
    if mode=='v1':return b.Agent(PublicBackend(env))
    opts=dict(o.CONFIGS['combined']);opts.update(CONFIGS[mode])
    return o.OptimizedAgent(PublicBackend(env),**opts)

from probe_score import rank_probes, sample_area

class AdvancedAgent(o.OptimizedAgent):
    def __init__(self,env,probe_version=0,probe_penalty=.5,probe_grid=1,route_centroid=False,
                 range_info=False,preobserve=False,cover_priority=0.,lookahead_depth=1,
                 adaptive_first=False,**kw):
        super().__init__(env,**kw)
        self.probe_version=probe_version;self.probe_penalty=probe_penalty;self.probe_grid=probe_grid
        self.route_centroid=route_centroid;self.range_info=range_info;self.preobserve=preobserve
        self.cover_priority=cover_priority;self.lookahead_depth=lookahead_depth;self.adaptive_first=adaptive_first
        self.directions={c:[] for c in b.CHANNELS}
        self.choice_log=[]

    def refine_information(self,c,poly):
        poly=super().refine_information(c,poly)
        if self.range_info:
            for pos in self.positive_positions[c]:
                # Adaptive outer tangent cuts, never an inscribed polygon.
                for _ in range(20):
                    d=poly-pos;ds=np.linalg.norm(d,axis=1);i=int(np.argmax(ds))
                    if ds[i]<=1500.0001:break
                    u=d[i]/ds[i];poly=b.clip(poly,u,float(pos@u)+1500+1e-6)
                    if not len(poly):raise RuntimeError('Positive range contradiction')
        return poly

    def route_location(self,t):
        if not self.route_centroid or t.radius<=b.CLEAR_GUARD:return t.center
        gs,weights=sample_area(t.poly)
        return np.sum(gs*weights[:,None],axis=0)

    def select_probe(self,c):
        if not self.probe_version:return super().select_probe(c)
        t=self.tracks[c];p=self.env.pos
        if self.local_steps[c]>=4 or t.radius<22:return t.center.copy(),True
        d=t.center-p;nd=b.norm(d)
        if nd<1e-7:return t.center.copy(),True
        u=d/nd;v=np.array([-u[1],u[0]]);r=t.radius
        candidates=[t.center.copy()]
        # Cartesian longitudinal / lateral candidates; reachability is certified below.
        if self.probe_grid==0:
            for off in (-.4,-.2,-.08,.08,.2,.4):candidates.append(t.center+off*r*v)
        else:
            for a in (-.7,-.35,0.,.35,.7):
                for z in (-.4,-.15,0.,.15,.4):
                    if a==0. and z==0.:continue
                    candidates.append(t.center+r*(a*u+z*v))
            candidates.append(p.copy())
        candidates=[q for q in candidates if np.max(np.linalg.norm(t.poly-q,axis=1))<999.9 and b.coord_key(q) not in t.measured_at]
        if not candidates:return t.center.copy(),True
        cc=np.array(candidates)
        scores=rank_probes(t.poly,p,cc,self.probe_version,self.probe_penalty,self.lookahead_depth,np.array(self.positive_positions[c]).reshape(-1,2),np.array(self.negative_positions[c]).reshape(-1,2),getattr(self,"range_prior",False))
        q=cc[int(np.argmin(scores))]
        return q.copy(),b.norm(q-t.center)<1e-6

    def run(self):
        self.scan_stop();self.choose_rotation()
        while self.unseen or self.tracks:
            self.steps+=1
            if self.steps>600:raise RuntimeError('Progress guard exceeded')
            self.clear_at_current_position()
            if not self.unseen and not self.tracks:break
            covers=self.remaining_cover if self.unseen else []
            if self.relocate and covers:self.relocate_cover_points()
            covers=self.remaining_cover if self.unseen else []
            nodes=[('cover',k,self.cover[k]) for k in covers]
            nodes += [('target',c,self.route_location(t)) for c,t in sorted(self.tracks.items())]
            order=self.order(self.env.pos,[n[2] for n in nodes])
            idx=order[0]
            if self.cover_priority and covers:
                # Short-horizon bias to discover unknown targets before committing to uncertain ones.
                first=nodes[idx]
                if first[0]=='target' and self.tracks[first[1]].radius>self.cover_priority:
                    near=min(range(len(covers)),key=lambda j:b.norm(nodes[j][2]-self.env.pos))
                    if b.norm(nodes[near][2]-self.env.pos)<b.norm(first[2]-self.env.pos)+self.tracks[first[1]].radius:
                        idx=near
            kind,ident,q=nodes[idx]
            self.route_successor=nodes[order[1]][2].copy() if len(order)>1 and idx==order[0] else None
            if kind=='cover':
                self.env.move(q);self.scan_stop(ident)
            else:
                if self.preobserve and self.useful_bearing(ident):
                    # New observation at our existing stop, before committing to a destination.
                    self.observe(ident)
                    if ident not in self.tracks:continue
                t=self.tracks[ident]
                if t.radius<=b.CLEAR_GUARD:
                    self.go_clear(ident);self.opportunistic_scan()
                else:
                    q,center=self.select_probe(ident)
                    self.env.move(q);self.local_steps[ident]+=1
                    if not center:self.lateral_steps+=1
                    self.observe(ident,center_step=center)
                    if ident in self.tracks and self.tracks[ident].radius<=b.CLEAR_GUARD:self.go_clear(ident)
                    self.opportunistic_scan(exclude=ident)
        if self.unseen or self.tracks or len(self.cleared|self.absent)!=20:raise RuntimeError('Incomplete channel certificate')

# All proposals below are development configurations, not validated defaults.
for ver in (1,2,3):
    for grid in (0,1):
        CONFIGS[f'probe{ver}g{grid}']={'probe_version':ver,'probe_grid':grid}
for penalty in (0.,.2,1.,2.):CONFIGS['pen'+str(penalty)]={'probe_version':2,'probe_grid':1,'probe_penalty':penalty}
CONFIGS['centroid']={'route_centroid':True}
CONFIGS['range']={'range_info':True}
CONFIGS['preobserve']={'preobserve':True}
for th in (100,200,350):CONFIGS['priority'+str(th)]={'cover_priority':th}

# Redefinition keeps unmodified v2 available for exact reproduction.
def make_agent(env,mode):
    backend=PublicBackend(env)
    if mode=='v1':return b.Agent(backend)
    if mode=='v2':return o.OptimizedAgent(backend,**o.CONFIGS['combined'])
    opts=dict(o.CONFIGS['combined']);opts.update(CONFIGS[mode])
    return AdvancedAgent(backend,**opts)

# Optional global search-stop redesign, audited by the continuous certificate.
def coverage_grid():
    axis=np.arange(-1800.,1801.,150.)
    mesh=np.array([(x,y) for x in axis for y in axis if x*x+y*y<=1800**2])
    angles=np.linspace(0,2*math.pi,240,endpoint=False)
    border=np.column_stack([np.cos(angles),np.sin(angles)])*1800
    return np.vstack([mesh,border])
GRID=coverage_grid()

class GlobalAgent(AdvancedAgent):
    def __init__(self,env,global_cover=False,adaptive_layout=False,scan_price=.7,cover_replans=8,**kw):
        super().__init__(env,**kw)
        self.global_cover=global_cover;self.adaptive_layout=adaptive_layout
        self.scan_price=scan_price;self.cover_replans=cover_replans;self.replans=0
        self.last_cover_state=None;self.accepted_replans=0

    def choose_rotation(self):
        if not self.adaptive_layout:return super().choose_rotation()
        if not self.unseen:return
        pts=[self.route_location(t) for t in self.tracks.values()]
        best=None
        # The entire ring is selected jointly, not moved one point at a time.
        for n,r in [(7,999),(7,1050),(7,1150),(7,1250),(6,1124),(6,1220),(6,1350)]:
            for angle in np.linspace(0,2*math.pi/n,10,endpoint=False):
                qs=[r*np.array([math.cos(angle+2*math.pi*k/n),math.sin(angle+2*math.pi*k/n)]) for k in range(n)]
                nodes=pts+qs;rt=b.route_order(self.env.pos,nodes)
                path=np.array([self.env.pos]+[nodes[i] for i in rt])
                cost=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())/5+6*self.scan_price*len(self.unseen)*n
                if best is None or cost<best[0]:best=cost,qs
        if not o.certified_cover(self.full_scans+best[1]):raise RuntimeError('Invalid ring design')
        self.cover=[p.copy() for p in best[1]];self.remaining_cover=list(range(len(self.cover)))

    def redesign_search(self):
        if not self.global_cover or not self.unseen or self.replans>=self.cover_replans:return
        # Redesign only after useful changes: a new full scan or a target removed/added.
        key=(len(self.full_scans),tuple(self.tracks),len(self.cleared))
        if key==self.last_cover_state:return
        self.last_cover_state=key;self.replans+=1
        old=[self.cover[i] for i in self.remaining_cover]
        targets=[self.route_location(t) for t in self.tracks.values()]
        candidates=old+targets
        for r in (1000,1125,1250,1375,1500):
            for a in np.linspace(0,2*math.pi,35,endpoint=False):
                candidates.append(r*np.array([math.cos(a),math.sin(a)]))
        cc=np.array(candidates)
        full=np.array(self.full_scans)
        covered=np.any(np.linalg.norm(GRID[:,None,:]-full[None,:,:],axis=2)<999.98,axis=1)
        masks=np.linalg.norm(cc[:,None,:]-GRID[None,:,:],axis=2)<999.98
        weights=np.ones(len(GRID));weights[-240:]=3.
        rt=self.order(self.env.pos,targets);path=[self.env.pos.copy()]+[targets[i].copy() for i in rt]
        chosen=[];price=30*self.scan_price*len(self.unseen)
        for _ in range(12):
            if covered.all():break
            # Marginal insertion distance along current open route.
            pp=np.array(path)
            if len(pp)==1:
                inc=np.linalg.norm(cc-self.env.pos,axis=1);places=np.zeros(len(cc),dtype=int)
            else:
                allcost=np.linalg.norm(cc[:,None,:]-pp[None,:,:],axis=2)
                icost=allcost[:,:-1]+allcost[:,1:]-np.linalg.norm(np.diff(pp,axis=0),axis=1)[None,:]
                icost=np.column_stack([icost,allcost[:,-1]])
                places=icost.argmin(axis=1);inc=icost.min(axis=1)
            gains=masks[:,~covered]@weights[~covered]
            score=np.where(gains>0,gains/(inc+price),-1.)
            if chosen:score[chosen]=-1
            j=int(np.argmax(score))
            if score[j]<=0:break
            chosen.append(j);covered|=masks[j]
            path.insert(int(places[j])+1,cc[j].copy())
        proposed=[cc[j].copy() for j in chosen]
        # Sampling is only a proposal mechanism. Repair and exact pruning below.
        if not o.certified_cover(self.full_scans+proposed):
            proposed+=old
        for qidx in reversed(range(len(proposed))):
            if o.certified_cover(self.full_scans+proposed[:qidx]+proposed[qidx+1:]):
                proposed.pop(qidx)
        def total(qs):
            nodes=targets+qs
            rt=self.order(self.env.pos,nodes)
            pts=np.array([self.env.pos]+[nodes[j] for j in rt])
            return np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()+price*len(qs)
        if total(proposed)+10<total(old):
            if not o.certified_cover(self.full_scans+proposed):raise RuntimeError('Invalid redesigned cover')
            self.cover=proposed;self.remaining_cover=list(range(len(proposed)));self.accepted_replans+=1

    def relocate_cover_points(self):
        self.redesign_search()
        super().relocate_cover_points()

for price in (.3,.6,1.):
    CONFIGS['layout'+str(price)]={'adaptive_layout':True,'scan_price':price}
    CONFIGS['global'+str(price)]={'global_cover':True,'scan_price':price}
CONFIGS['globalprobe']={'global_cover':True,'scan_price':.6,'probe_version':2,'probe_grid':1}

# Keep the v2 baseline isolated; higher-level classes share its geometry.
def make_agent(env,mode):
    backend=PublicBackend(env)
    if mode=='v1':return b.Agent(backend)
    if mode=='v2':return o.OptimizedAgent(backend,**o.CONFIGS['combined'])
    opts=dict(o.CONFIGS['combined']);opts.update(CONFIGS[mode])
    return GlobalAgent(backend,**opts)
from route_search import search_route

class EfficientAgent(GlobalAgent):
    def __init__(self,env,route_trials=0,extra_scan_threshold=0.,batch_clear=False,**kw):
        super().__init__(env,**kw)
        self.route_trials=route_trials;self.extra_scan_threshold=extra_scan_threshold;self.batch_clear=batch_clear

    def order(self,start,points):
        if not self.route_trials or len(points)<=1:return super().order(start,points)
        return list(search_route(np.array([start]+list(points)),self.route_trials))

    def observe(self,c,cover_id=None,center_step=False):
        super().observe(c,cover_id,center_step)
        if self.batch_clear and c in self.tracks:
            if np.linalg.norm(self.tracks[c].poly-self.env.pos,axis=1).max()<19.99999:self.do_clear(c)

    def opportunistic_scan(self,exclude=None):
        if self.extra_scan_threshold and self.unseen:
            # New coverage area is a heuristic for value of early discovery.
            # Only actual negative detections enter the exact coverage certificate.
            full=np.array(self.full_scans)
            old=np.any(np.linalg.norm(GRID[:,None,:]-full[None,:,:],axis=2)<1150,axis=1)
            new=np.linalg.norm(GRID-self.env.pos,axis=1)<1150
            gain=float(np.mean(new & ~old))
            if gain>self.extra_scan_threshold:
                self.extra_full_scans+=1;self.scan_stop();return
        super().opportunistic_scan(exclude)

for k in (15,40,100,250):CONFIGS['route'+str(k)]={'route_trials':k}
for th in (.04,.07,.10,.15,.2):CONFIGS['early'+str(th)]={'extra_scan_threshold':th}
CONFIGS['batch']={'batch_clear':True}
CONFIGS['mixprobe']={'probe_version':2,'probe_grid':1,'route_trials':40}
CONFIGS['mixglobal']={'probe_version':2,'probe_grid':1,'route_trials':40,'global_cover':True,'scan_price':.6}

def make_agent(env,mode):
    backend=PublicBackend(env)
    if mode=='v1':return b.Agent(backend)
    if mode=='v2':return o.OptimizedAgent(backend,**o.CONFIGS['combined'])
    opts=dict(o.CONFIGS['combined']);opts.update(CONFIGS[mode])
    return EfficientAgent(backend,**opts)
class StartupAgent(EfficientAgent):
    def __init__(self,env,initial_channels=20,**kw):
        super().__init__(env,**kw)
        self.initial_channels=initial_channels;self.startup_done=False
        if initial_channels<20 and not o.certified_cover(self.cover):
            raise ValueError('Without origin scan, the future plan must cover the whole domain alone')

    def scan_stop(self,cover_id=None):
        if not self.startup_done:
            self.startup_done=True
            if self.initial_channels<20:
                for c in range(1,self.initial_channels+1):
                    if c in self.unseen:self.observe(c)
                return
        super().scan_stop(cover_id)

for n in (0,3,5,10,15):CONFIGS['initial'+str(n)]={'initial_channels':n}
CONFIGS['initial0mix']={'initial_channels':0,'probe_version':2,'probe_grid':1,'route_trials':40}
CONFIGS['initial5mix']={'initial_channels':5,'probe_version':2,'probe_grid':1,'route_trials':40}
CONFIGS['initial0deep']={'initial_channels':0,'probe_version':3,'probe_grid':0,'route_trials':40}

def make_agent(env,mode):
    backend=PublicBackend(env)
    if mode=='v1':return b.Agent(backend)
    if mode=='v2':return o.OptimizedAgent(backend,**o.CONFIGS['combined'])
    opts=dict(o.CONFIGS['combined']);opts.update(CONFIGS[mode])
    return StartupAgent(backend,**opts)
class FinishingAgent(StartupAgent):
    def __init__(self,env,through_clear=False,nonuniform=0,**kw):
        super().__init__(env,**kw);self.through_clear=through_clear;self.nonuniform=nonuniform
        if nonuniform:
            from pathlib import Path
            import json
            layouts=json.loads((Path(__file__).parent/'results'/'nonuniform_layouts.json').read_text())
            layout=next(x for x in layouts if x['n']==nonuniform)
            self.cover=[np.array(p) for p in layout['points']];self.remaining_cover=list(range(nonuniform))
            if not o.certified_cover(([np.zeros(2)] if self.initial_channels==20 else [])+self.cover):raise ValueError('Nonuniform cover lacks certificate')

    def go_clear(self,c):
        if not self.through_clear or getattr(self,'route_successor',None) is None:return super().go_clear(c)
        t=self.tracks[c];p=self.env.pos.copy();nxt=self.route_successor
        if t.radius>b.CLEAR_GUARD:raise RuntimeError('Uncertified clear')
        qa=o.nearest_clear_point(p,t.poly,t.center);qb=o.nearest_clear_point(nxt,t.poly,t.center)
        def value(q):return b.norm(q-p)+b.norm(q-nxt)
        # Feasible set is convex, so every point on this segment remains certified.
        best=qa;bestval=value(qa)
        for lam in np.linspace(0,1,21):
            q=(1-lam)*qa+lam*qb;vv=value(q)
            if vv<bestval:best=q;bestval=vv
        # Direct intersection with the route segment, when possible.
        d=nxt-p;dd=float(d@d);lo=0.;hi=1.
        if dd>1e-12:
            for vertex in t.poly:
                w=p-vertex;bb=2*float(d@w);cc=float(w@w)-(20-1e-5)**2
                disc=bb*bb-4*dd*cc
                if disc<0:lo=2.;break
                root=math.sqrt(disc);lo=max(lo,(-bb-root)/(2*dd));hi=min(hi,(-bb+root)/(2*dd))
                if lo>hi:break
            if lo<=hi:
                q=p+d*min(hi,max(lo,(lo+hi)/2))
                if value(q)<bestval:best=q
        if np.linalg.norm(t.poly-best,axis=1).max()>20-1e-7:raise RuntimeError('Invalid through-clear point')
        self.env.move(best);self.do_clear(c)

CONFIGS['through']={'through_clear':True}
CONFIGS['nonuniform7']={'nonuniform':7}
CONFIGS['nonuniform6']={'nonuniform':6}
CONFIGS['initial0through']={'initial_channels':0,'through_clear':True,'probe_version':2,'probe_grid':1,'route_trials':40}
CONFIGS['initial0uniform']={'initial_channels':0,'nonuniform':7,'probe_version':2,'probe_grid':1,'route_trials':40}
CONFIGS['combo1050']={'ring_radius':1050,'probe_version':2,'probe_grid':1,'route_trials':40}
CONFIGS['combo1050deep']={'ring_radius':1050,'probe_version':3,'probe_grid':0,'route_trials':40}
CONFIGS['initial0radius']={'ring_radius':1050,'initial_channels':0,'probe_version':2,'probe_grid':1,'route_trials':40}

def make_agent(env,mode):
    backend=PublicBackend(env)
    if mode=='v1':return b.Agent(backend)
    if mode=='v2':return o.OptimizedAgent(backend,**o.CONFIGS['combined'])
    opts=dict(o.CONFIGS['combined']);opts.update(CONFIGS[mode])
    return FinishingAgent(backend,**opts)
class FinalistAgent(FinishingAgent):
    def __init__(self,env,startup_signals=0,startup_cap=8,range_prior=False,**kw):
        super().__init__(env,**kw)
        self.startup_signals=startup_signals;self.startup_cap=startup_cap;self.range_prior=range_prior
        if startup_signals and not o.certified_cover(self.cover):raise ValueError('Partial origin scan cannot certify missing center region')
    def scan_stop(self,cover_id=None):
        if not self.startup_done and self.startup_signals:
            self.startup_done=True
            for c in range(1,self.startup_cap+1):
                if c in self.unseen:self.observe(c)
                if len(self.tracks)+len(self.cleared)>=self.startup_signals:break
            return
        super().scan_stop(cover_id)

for sig,cap in [(1,4),(2,8),(3,10)]:
    CONFIGS['adaptive'+str(sig)]={'startup_signals':sig,'startup_cap':cap,'probe_version':2,'probe_grid':1,'route_trials':40}
for init in (0,5,20):
    CONFIGS['weighted'+str(init)]={'initial_channels':init,'probe_version':2,'probe_grid':1,'route_trials':40,'range_prior':True}
CONFIGS['weighted_adaptive']={'startup_signals':2,'startup_cap':8,'probe_version':2,'probe_grid':1,'route_trials':40,'range_prior':True}

def make_agent(env,mode):
    backend=PublicBackend(env)
    if mode=='v1':return b.Agent(backend)
    if mode=='v2':return o.OptimizedAgent(backend,**o.CONFIGS['combined'])
    opts=dict(o.CONFIGS['combined']);opts.update(CONFIGS[mode])
    return FinalistAgent(backend,**opts)
