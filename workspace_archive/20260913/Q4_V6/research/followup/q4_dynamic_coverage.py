"""Costed 1-to-2 scan-site exchanges, authorized by a continuous certificate.

Only planned sites are replaced. All remaining unknown channels must still be
actually scanned at every replacement site. Actual source clearing is unrelated
to this coverage plan and always requires success feedback. Pending target
centers here are just fixed candidate scan COORDINATES, not assumed truths.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import itertools,math,time
import q4_baseline as b
from q4_coverage import certify
from q4_route_cached import multi_route

@dataclass
class CoverageConfig:
    max_updates:int=5
    max_checks:int=60
    max_wall_seconds:float=2.
    max_candidate_radius:float=300.
    min_saved_seconds:float=12.
    allow_pairs:bool=True


def proxy_cost(engine,planned):
    st=engine.state;p=engine.device.position
    # Candidate scan points that coincide with a target center can share travel.
    points=planned+[t.center() for t in st.pending.values()]
    if not points:return 0.
    order=multi_route(p,list(range(len(points))),points,4)
    travel=0.;here=p
    for j in order:travel+=b.dist(here,points[j]);here=points[j]
    return travel/5+6*len(st.unknown)*len(planned)


class CoverageExchanger:
    def __init__(self,config=None):
        self.config=config or CoverageConfig();self.updates=0;self.checks=0;self.trace=[];self.seconds=0.
    def attempt(self,engine):
        cfg=self.config;st=engine.state
        if self.updates>=cfg.max_updates or len(st.remaining)<3 or not st.unknown:return False
        start=time.perf_counter();deadline=start+cfg.max_wall_seconds
        current=engine.device.position
        remaining=sorted(st.remaining);oldpoints=[st.sites[i] for i in remaining]
        available=[current]
        for t in st.pending.values():
            if t.obs.status=='strong':available.append(t.anchor)
            else:
                cen,rad=b.enclosing_circle(t.poly)
                if rad<cfg.max_candidate_radius:available.append(cen)
        # New sites close to existing mandatory scans offer no meaningful change.
        available=[q for q in available if min((b.dist(q,p) for p in oldpoints+st.scanpoints),default=1e10)>35.]
        available=list(dict.fromkeys(available))
        if not available:return False
        available=sorted(available,key=lambda q:b.dist(q,current))[:6]
        basecost=proxy_cost(engine,oldpoints)
        best=None;bestgain=cfg.min_saved_seconds;checks=0
        # Exterior hull sites cannot usually be replaced by interior source points.
        # They are still legal candidates; rank by distance to available points.
        removals=sorted(remaining,key=lambda i:min(b.dist(st.sites[i],q) for q in available))[:6]
        proposals=[]
        for i in removals:
            near=sorted(available,key=lambda q:b.dist(st.sites[i],q))[:4]
            for n in ([1,2] if cfg.allow_pairs else [1]):
                for qs in itertools.combinations(near,n):
                    rest=[st.sites[j] for j in remaining if j!=i]
                    planned=rest+list(qs)
                    gain=basecost-proxy_cost(engine,planned)
                    if gain>bestgain:proposals.append((gain,i,qs,planned))
        proposals.sort(reverse=True,key=lambda x:x[0])
        for gain,i,qs,planned in proposals:
            if checks>=cfg.max_checks or time.perf_counter()>deadline:break
            if gain<=bestgain:continue
            checks+=1
            cert=certify(st.scanpoints+planned,max_depth=18)
            if cert['ok']:
                best=(i,qs,cert);bestgain=gain
                # Sorted proposals: first certified proposal has the largest proxy gain.
                break
        self.checks+=checks;self.seconds+=time.perf_counter()-start
        if best is None:return False
        i,qs,cert=best
        st.remaining.remove(i)
        ids=[]
        for q in qs:
            idx=len(st.sites);st.sites.append(q);st.remaining.add(idx);ids.append(idx)
        self.updates+=1
        self.trace.append(dict(removed=i,added_ids=ids,added_points=qs,proxy_saved_seconds=bestgain,
            actual_scan_points=st.scanpoints[:],planned_points=[st.sites[j] for j in sorted(st.remaining)],
            continuous_certificate={k:v for k,v in cert.items() if k not in ['wall','leaves']},checks=checks))
        return True
