"""Fast local probing with a bounded conservative fallback.

Only bounded bearing errors and feedback are used for correctness. Optional
speculative clears are OFF by default. No third-party dependencies.
"""
import q4_baseline as b
import math,statistics,time,json
BASELINE_LOCALIZE = b.localize_and_clear
from dataclasses import dataclass

@dataclass
class LocalConfig:
    axial: float = .3
    lateral: float = .025
    speculative_radius: float = 0.
    nearest_clear: bool = True
    advance_lower: bool = True
    lower_factor: float = .98

def nearest_clear_point(poly, here, radius=19.5):
    # Exact candidates for projection onto intersection of closed disks.
    feasible=lambda q: all(b.dist(q,v)<=radius+1e-7 for v in poly)
    if feasible(here): return here
    candidates=[]
    for v in poly:
        d=b.dist(here,v)
        q=b.add(v,b.mul(radius/d,b.sub(here,v))) if d>0 else here
        if feasible(q): candidates.append(q)
    for i,v in enumerate(poly):
        for w in poly[i+1:]:
            d=b.dist(v,w)
            if d<1e-9 or d>2*radius: continue
            m=b.mul(.5,b.add(v,w)); off=math.sqrt(max(0,radius**2-d**2/4))
            e=b.mul(1/d,b.sub(w,v)); n=(-e[1],e[0])
            for sgn in (-1,1):
                q=b.add(m,b.mul(sgn*off,n))
                if feasible(q): candidates.append(q)
    if not candidates: return b.enclosing_circle(poly)[0]
    return min(candidates,key=lambda q:b.dist(here,q))


def localize(device: b.Device, track: b.Track, use_intersections: bool = True,
             config: LocalConfig | None = None, init_poly: list[b.Point] | None = None) -> dict:
    config=config or LocalConfig()
    c,s,obs=track.channel,track.anchor,track.observation
    if obs.status=='strong':
        q=s
        if config.nearest_clear:
            d=b.dist(device.position,s)
            q=b.add(s,b.mul(min(1,14.5/max(d,1e-12)),b.sub(device.position,s)))
        b.checked_clear(device,c,q)
        return dict(stages=0,pair_failures=0,certificate='strong')
    theta=obs.theta
    if theta is None: raise ValueError('bearing required')
    upper=1500.
    poly=init_poly if init_poly is not None else b.initial_polygon(s,theta,upper)
    pair_fails=0; tried_optical=False
    for stage in range(12):
        center,radius=b.enclosing_circle(poly)
        if radius<=b.CLEAR_CERT_RADIUS:
            q=nearest_clear_point(poly,device.position) if config.nearest_clear else center
            b.checked_clear(device,c,q)
            return dict(stages=stage,pair_failures=pair_fails,certificate='intersection')
        if not tried_optical and radius<=config.speculative_radius:
            tried_optical=True
            device.move(center)
            if device.clear(c):
                return dict(stages=stage,pair_failures=pair_fails,certificate='optical_confirmation')
        upper=min(upper,max(b.dist(s,p) for p in poly)+1e-7)
        e=b.unit(theta); v=(-e[1],e[0]); a=config.axial; lateral=config.lateral
        t,h=upper*a,upper*lateral
        if config.advance_lower:
            lower=max(0.,min(b.dot(e,b.sub(p,s)) for p in poly))
            t=min(.95*upper,max(t,config.lower_factor*lower))
            h=max(h,t*b.TAN_D*1.05)
            a,lateral=t/upper,h/upper
        if not (lateral/a > b.TAN_D and (lateral/a)**2+2*(lateral/a)*b.TAN_D<1):
            raise ValueError('unsafe pair geometry')
        midpoint=b.add(s,b.mul(t,e))
        probes=[b.add(midpoint,b.mul(h,v)),b.add(midpoint,b.mul(-h,v))]
        probes.sort(key=lambda q:b.dist(device.position,q))
        good=False
        for q in probes:
            device.move(q); new=device.detect(c)
            if new.status=='strong':
                b.checked_clear(device,c,q)
                return dict(stages=stage+1,pair_failures=pair_fails,certificate='strong')
            if new.status=='bearing':
                # Use actual outer polygon, no loose contraction coefficient needed.
                new_upper=min(1500.,max(b.dist(q,p) for p in poly)+1e-7)
                poly=b.clip_bearing(poly,q,new.theta,new_upper)
                if not poly: raise RuntimeError('empty polygon positive')
                upper=new_upper; s=q; theta=new.theta; good=True; break
        if not good:
            pair_fails+=1
            poly=b.clip_halfplane(poly,e,b.dot(e,s)+t)
            if not poly: raise RuntimeError('empty polygon negative')
            upper=t/b.COS_D+1e-7
    fallback=BASELINE_LOCALIZE(device,b.Track(c,s,b.Observation('bearing',theta)),True)
    return dict(stages=12+fallback['stages'],pair_failures=pair_fails+fallback['pair_failures'],certificate='conservative_fallback:'+fallback['certificate'])

