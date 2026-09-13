"""Local paired probes, conservative negative updates, optional shared sensing."""
from __future__ import annotations
import math
import q4_baseline as b
from q4_fast_local import LocalConfig as BaseLocalConfig,nearest_clear_point,BASELINE_LOCALIZE
from dataclasses import dataclass

@dataclass
class LocalConfig(BaseLocalConfig):
    range_fraction:float=0.
    adaptive_lateral:float=0.
    side_information:bool=False
    optical_cover_limit:int=6


def optical_cover(poly,radius=19.5):
    """Cover a containing oriented rectangle with certified optical disks.

    Return centers, or [] if the polygon is too wide. Every candidate point
    belongs to at least one radius-19.5 disk; failed optical attempts are allowed
    and must be timed by the adapter. Completion is declared only on success.
    """
    a,c=max(((a,c) for a in poly for c in poly),key=lambda z:b.dist(*z))
    if b.dist(a,c)<1e-8:return [a]
    axes=[b.sub(c,a)]+[b.sub(z,a) for a,z in zip(poly,poly[1:]+poly[:1])]
    best=None;centers=[]
    for direction in axes:
        d=b.norm(direction)
        if d<1e-8:continue
        e=b.mul(1/d,direction);v=(-e[1],e[0])
        xs=[b.dot(e,p) for p in poly];ys=[b.dot(v,p) for p in poly]
        lo,hi=min(xs),max(xs);ym=(min(ys)+max(ys))/2;half=(max(ys)-min(ys))/2
        if half>=radius-1e-5:continue
        step=2*math.sqrt(radius**2-half**2)-1e-5
        n=max(1,math.ceil((hi-lo)/step))
        if n>100:continue
        score=(n,hi-lo,half)
        if best is None or score<best:
            best=score
            centers=[b.add(b.mul(lo+(i+.5)*(hi-lo)/n,e),b.mul(ym,v)) for i in range(n)]
    return centers


def localize(device,belief,config:LocalConfig|None=None,on_probe=None):
    config=config or LocalConfig()
    c=belief.channel;pair_fails=0
    if belief.obs.status=='strong':
        s=belief.anchor;d=b.dist(device.position,s)
        q=b.add(s,b.mul(min(1,14.5/max(d,1e-12)),b.sub(device.position,s)))
        b.checked_clear(device,c,q)
        return dict(stages=0,pair_failures=0,certificate='strong',negative_updates=belief.negative_updates)
    upper=1500.
    for stage in range(12):
        s=belief.anchor;theta=belief.obs.theta;poly=belief.poly
        center,radius=b.enclosing_circle(poly)
        if radius<=b.CLEAR_CERT_RADIUS:
            q=nearest_clear_point(poly,device.position)
            b.checked_clear(device,c,q)
            return dict(stages=stage,pair_failures=pair_fails,certificate='intersection',negative_updates=belief.negative_updates)
        if config.optical_cover_limit>0:
            cover=optical_cover(poly)
            if cover and len(cover)<=config.optical_cover_limit:
                if b.dist(device.position,cover[-1])<b.dist(device.position,cover[0]):cover.reverse()
                for j,q in enumerate(cover):
                    device.move(q)
                    if device.clear(c):
                        return dict(stages=stage,pair_failures=pair_fails,certificate='optical_cover_confirmed',optical_attempts=j+1,cover_size=len(cover),negative_updates=belief.negative_updates)
                raise RuntimeError('Entire certified optical cover failed; inspect adapter/model')
        upper=min(1500.,max(b.dist(s,p) for p in poly)+1e-7)
        e=b.unit(theta);v=(-e[1],e[0]);a=config.axial;lateral=config.lateral
        t,h=upper*a,upper*lateral
        if config.advance_lower:
            lower=max(0.,min(b.dot(e,b.sub(p,s)) for p in poly))
            advance=config.lower_factor*lower
            if lower/upper>.25:
                advance=max(advance,lower+config.range_fraction*(upper-lower))
                h=max(h,config.adaptive_lateral*upper)
            t=min(.95*upper,max(t,advance))
            h=max(h,t*b.TAN_D*1.05)
        ratio=h/t
        if not (ratio>b.TAN_D and ratio**2+2*ratio*b.TAN_D<1):raise ValueError('unsafe pair')
        mid=b.add(s,b.mul(t,e));probes=[b.add(mid,b.mul(h,v)),b.add(mid,b.mul(-h,v))]
        probes.sort(key=lambda q:b.dist(device.position,q))
        good=False
        for q in probes:
            device.move(q);obs=device.detect(c);belief.add(q,obs)
            if obs.status=='strong':
                b.checked_clear(device,c,q)
                return dict(stages=stage+1,pair_failures=pair_fails,certificate='strong',negative_updates=belief.negative_updates)
            if on_probe is not None:on_probe()
            if obs.status=='bearing':good=True;break
        if not good:
            pair_fails+=1
            belief.poly=b.clip_halfplane(belief.poly,e,b.dot(e,s)+t)
            if not belief.poly:raise RuntimeError('empty paired negative polygon')
            belief.tighten()
    fallback=BASELINE_LOCALIZE(device,b.Track(c,belief.anchor,belief.obs),True)
    return dict(stages=12+fallback['stages'],pair_failures=pair_fails+fallback['pair_failures'],certificate='conservative_fallback:'+fallback['certificate'],negative_updates=belief.negative_updates)
