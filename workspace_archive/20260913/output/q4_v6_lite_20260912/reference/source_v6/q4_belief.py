"""History-conditioned scene sampling with analytic radius/direction integration.

Assumptions for PLANNING only: area-uniform positions; bounded, independent
uniform RF errors at previously unobserved locations; a uniform count prior;
exchangeable channels; a mixture radius prior and initially equiprobable types.
The two types are jointly conditioned to both exist. Numerical position pools
are importance quadrature, NOT proofs that a source is absent. Failures simply
disable lookahead; no safety state is deleted.

Rao-Blackwellisation inspiration: Doucet et al., UAI 2000 (arXiv:1301.3853).
This is an original problem-specific implementation, not the paper's code.
"""
from __future__ import annotations
from dataclasses import dataclass
import bisect
import copy
import hashlib
import math
import random
import q4_baseline as b
from q4_state import State,RFRecord
from radius_direction import marginalize,RadiusPrior,Marginal

class PosteriorUnavailable(RuntimeError):pass


def keypoint(p):return (round(p[0],9),round(p[1],9))

def angle_error(theta,g,p):
    a=math.atan2(g[1]-p[1],g[0]-p[0])
    return abs((theta-a+math.pi)%(2*math.pi)-math.pi)

def sample_polygon(poly,rng):
    a=poly[0]
    tris=[];weights=[]
    for z,w in zip(poly[1:-1],poly[2:]):
        area=abs((z[0]-a[0])*(w[1]-a[1])-(z[1]-a[1])*(w[0]-a[0]))
        if area>1e-17:tris.append((z,w));weights.append(area)
    if not tris:raise PosteriorUnavailable('Position region is numerically degenerate')
    z,w=rng.choices(tris,weights=weights)[0]
    u,v=rng.random(),rng.random()
    if u+v>1:u,v=1-u,1-v
    return (a[0]+u*(z[0]-a[0])+v*(w[0]-a[0]),a[1]+u*(z[1]-a[1])+v*(w[1]-a[1]))

def intersect_disk_outer(poly,center,radius,n=32):
    for k in range(n):
        u=b.unit(2*math.pi*k/n)
        poly=b.clip_halfplane(poly,u,b.dot(u,center)+radius)
    return poly

def proposal_polygon(state,c):
    tr=state.pending.get(c) or state.cleared_tracks.get(c)
    if tr is None:return None
    poly=tr.poly[:] if tr.poly else [(-1801.,-1801.),(1801.,-1801.),(1801.,1801.),(-1801.,1801.)]
    for r in state.rf[c]:
        if r.obs.status=='strong':poly=intersect_disk_outer(poly,r.point,5.)
        elif r.obs.status=='bearing' and not tr.poly:
            poly=b.clip_bearing(poly,r.point,r.obs.theta,1500.)
    for r in state.optical[c]:
        if r.success:poly=intersect_disk_outer(poly,r.point,20.)
    if len(poly)<3:raise PosteriorUnavailable(f'No positive-area position proposal for channel {c}')
    return poly

def geometric_consistent(g,records,optics):
    if b.norm(g)>1800.+1e-8:return False
    for rec in records:
        d=b.dist(g,rec.point);obs=rec.obs
        if obs.status=='bearing':
            if not 5.<d<=1500.+1e-8 or angle_error(obs.theta,g,rec.point)>b.DELTA+2e-9:return False
        elif obs.status=='strong' and d>5.+1e-8:return False
    for rec in optics:
        d=b.dist(g,rec.point)
        if rec.success and d>20.+1e-8:return False
        if not rec.success and d<=20.+1e-8:return False
    return True

@dataclass
class PositionPool:
    positions:list[b.Point]
    marginals:list[Marginal]
    omni_weights:list[float]
    dir_weights:list[float]
    omni_evidence:float
    dir_evidence:float
    positives:tuple[b.Point,...]
    negatives:tuple[b.Point,...]
    known:bool

    def sample(self,kind:int,rng:random.Random,rprior:RadiusPrior):
        weights=self.omni_weights if kind==1 else self.dir_weights
        if not sum(weights)>0:raise PosteriorUnavailable('Requested zero-weight type')
        j=rng.choices(range(len(weights)),weights=weights)[0]
        g=self.positions[j];m=self.marginals[j]
        if kind==1:
            lo=max([1000.]+[b.dist(g,p) for p in self.positives])
            hi=min([math.inf]+[b.dist(g,q) for q in self.negatives])
            direction=None
        else:
            piece=rng.choices(m.pieces,weights=[p.mass for p in m.pieces])[0]
            a=piece.begin+(piece.end-piece.begin)*(1e-9+(1-2e-9)*rng.random())
            direction=b.unit(a);lo=piece.radius_lower;hi=piece.radius_upper_exclusive
        low=max(1000.,lo);high=min(1500.,hi)
        parts=[(None,rprior.uniform_weight*max(0.,high-low)/500.)]
        parts.extend((radius,w) for radius,w in rprior.atoms if lo<=radius<hi)
        part=rng.choices(parts,weights=[w for _,w in parts])[0][0]
        radius=low+(high-low)*(1e-9+(1-2e-9)*rng.random()) if part is None else part
        return g,radius,direction

    def mean(self,directional_prior=.5):
        ww=[(1-directional_prior)*o+directional_prior*d for o,d in zip(self.omni_weights,self.dir_weights)]
        tot=sum(ww)
        if tot<=0:return self.positions[0]
        return tuple(sum(w*p[j] for w,p in zip(ww,self.positions))/tot for j in [0,1])

class JointTypesCounts:
    """DP over 20 channels, source count, and the two-type presence bitmask.

    This DP is exact conditional on supplied integrated likelihoods; position
    integration itself is finite quadrature. Known-channel evidence can be
    scaled by any positive channel-specific factor without changing posterior.
    """
    def __init__(self,options,directional_prior=.5,count_prior=None):
        self.options=options
        self.p=directional_prior
        self.prior=count_prior or {n:1/7 for n in range(10,17)}
        self.tables=[{(0,0):1.}]
        for opts in options:
            tab={}
            for (n,mask),v in self.tables[-1].items():
                for kind,w in enumerate(opts):
                    nn=n+int(kind!=0)
                    if w<=0 or nn>16:continue
                    k=(nn,mask|kind)
                    tab[k]=tab.get(k,0.)+v*w
            scale=max(tab.values(),default=0.)
            if scale<=0:raise PosteriorUnavailable('Empty joint count/type posterior')
            self.tables.append({k:v/scale for k,v in tab.items()})
        self.count_weights={}
        for n in range(10,17):
            # Prior samples types iid, then conditions on both types existing.
            mixed_mass=1-self.p**n-(1-self.p)**n
            if mixed_mass<=0:continue
            w=self.tables[-1].get((n,3),0.)*self.prior.get(n,0.)/math.comb(20,n)/mixed_mass
            if w>0:self.count_weights[n]=w
        if not self.count_weights:raise PosteriorUnavailable('Cannot satisfy 10..16 sources and both types')

    def sample(self,rng):
        n=rng.choices(list(self.count_weights),weights=list(self.count_weights.values()))[0]
        total=n;mask=3;kinds=[]
        for i in range(len(self.options),0,-1):
            candidates=[];weights=[]
            for kind,w in enumerate(self.options[i-1]):
                prevn=n-int(kind!=0)
                if w<=0 or prevn<0:continue
                for pm in range(4):
                    if pm|kind!=mask:continue
                    val=self.tables[i-1].get((prevn,pm),0.)*w
                    if val>0:candidates.append((kind,prevn,pm));weights.append(val)
            kind,n,mask=rng.choices(candidates,weights=weights)[0]
            kinds.append(kind)
        if n or mask:raise RuntimeError('Joint DP backward-sampling invariant failed')
        return total,list(reversed(kinds))

@dataclass(frozen=True)
class Scenario:
    targets:tuple[b.Target,...]
    seed:int
    history:dict
    count:int

    def device(self,position,channel):
        sim=ConditionedSimulator([copy.copy(t) for t in self.targets],self.seed,self.history)
        sim.position=position;sim.channel=channel
        return sim

class ConditionedSimulator(b.LocalSimulator):
    """Independent imagined environment, with fixed observed RF prefixes.

    It is NEVER constructed from or given access to the real simulator's state.
    Historical errors are conditioned on; previously unvisited points use a
    deterministic bounded hash field. The base continuation sees only Device.
    """
    def __init__(self,targets,seed,history):
        super().__init__(targets,seed)
        self._history=history
    def detect(self,c):
        obs=super().detect(c)
        t=self._targets.get(c)
        old=self._history.get((c,keypoint(self.position)))
        if old is not None and (t is None or not t.cleared):
            if old.status!=obs.status:
                raise PosteriorUnavailable('Sample contradicts an RF prefix status')
            return old
        return obs

class SceneFactory:
    def __init__(self,draws=96,seed=37013,directional_prior=.5,atoms=True):
        if draws<8:raise ValueError('Use at least 8 position proposals')
        self.draws=draws;self.seed=seed;self.p=directional_prior
        self.radius_prior=RadiusPrior(.95,((1000.,.05),)) if atoms else RadiusPrior()
        self.cache={};self.pool_builds=0;self.skipped_positions=0

    def pool(self,state,c):
        records=tuple(dict.fromkeys(state.rf[c]))
        optics=tuple(state.optical[c])
        tr=state.pending.get(c) or state.cleared_tracks.get(c)
        key=(records,optics,tuple(tr.poly) if tr is not None else None,tr is not None)
        if key in self.cache:return self.cache[key]
        # Seeds depend only on observed history, never on real scenario identity.
        digest=int.from_bytes(hashlib.blake2b(repr(key).encode(),digest_size=8).digest(),'big')
        rng=random.Random(self.seed^digest)
        positives=tuple(dict.fromkeys(r.point for r in records if r.obs.status!='none'))
        negatives=tuple(dict.fromkeys(r.point for r in records if r.obs.status=='none'))
        poly=proposal_polygon(state,c)
        gs=[];ms=[];ows=[];dws=[];attempts=0
        target_draws=self.draws if tr is not None else self.draws*2
        # Unknown proposal is the area-uniform source prior. Known proposal is
        # uniform in an outer feasible polygon, with full bearing/range rejection.
        for _ in range(target_draws):
            attempts+=1
            g=sample_polygon(poly,rng) if poly else b.mul(1800*math.sqrt(rng.random()),b.unit(2*math.pi*rng.random()))
            if not geometric_consistent(g,records,optics):continue
            m=marginalize(g,positives,negatives,directional_prior=self.p,radius_prior=self.radius_prior)
            if m.mixture_evidence<=1e-300:continue
            gs.append(g);ms.append(m);ows.append(m.omni_evidence);dws.append(m.directional_evidence)
        if tr is not None and not gs:
            raise PosteriorUnavailable(f'Finite quadrature lost support at known channel {c}')
        # Unknown pools may carry zero mass, but that NEVER proves absence.
        denom=max(1,attempts)
        oe=sum(ows)/denom;de=sum(dws)/denom
        if tr is not None:
            scale=(1-self.p)*oe+self.p*de
            if scale<=0:raise PosteriorUnavailable('Zero mass for known source')
            oe/=scale;de/=scale
        pool=PositionPool(gs,ms,ows,dws,oe,de,positives,negatives,tr is not None)
        if len(self.cache)>500:self.cache.clear()
        self.cache[key]=pool;self.pool_builds+=1
        return pool

    def prepare(self,state):
        pools={c:self.pool(state,c) for c in range(1,21)}
        opts=[]
        for c in range(1,21):
            po=pools[c]
            opts.append((0. if po.known else 1.,(1-self.p)*po.omni_evidence,self.p*po.dir_evidence))
        joint=JointTypesCounts(opts,self.p)
        history={(c,keypoint(rec.point)):rec.obs for c,recs in state.rf.items() for rec in recs}
        return pools,joint,history

    def sample(self,state,n,rng,prepared=None):
        pools,joint,history=prepared or self.prepare(state)
        scenes=[]
        for _ in range(n):
            count,kinds=joint.sample(rng);targets=[]
            for c,kind in enumerate(kinds,1):
                if kind==0:continue
                g,r,u=pools[c].sample(kind,rng,self.radius_prior)
                targets.append(b.Target(c,g,r,u,c in state.cleared))
            scene=Scenario(tuple(targets),rng.randrange(1<<60),history,count)
            self.verify(scene,state)
            scenes.append(scene)
        return scenes

    @staticmethod
    def verify(scene,state):
        targets={t.channel:t for t in scene.targets}
        if not 10<=len(targets)<=16 or len({t.direction is None for t in scene.targets})!=2:
            raise PosteriorUnavailable('Generated scene violates count or type constraints')
        for c,records in state.rf.items():
            t=targets.get(c)
            for rec in records:
                g=t.position if t else (1e10,1e10)
                d=b.dist(g,rec.point)
                vis=t is not None and d<=t.radius+1e-9 and (t.direction is None or b.dot(t.direction,b.sub(rec.point,g))>=-1e-9)
                st='none' if not vis else ('strong' if d<=5.+1e-9 else 'bearing')
                if st!=rec.obs.status:raise PosteriorUnavailable('Scene fails prefix visibility consistency')
                if st=='bearing' and angle_error(rec.obs.theta,g,rec.point)>b.DELTA+2e-9:
                    raise PosteriorUnavailable('Scene fails bounded bearing consistency')
            for rec in state.optical[c]:
                if t is None:
                    if rec.success:raise PosteriorUnavailable('Successful optical source omitted')
                elif (b.dist(t.position,rec.point)<=20.+1e-8)!=rec.success:
                    raise PosteriorUnavailable('Scene fails optical history consistency')
