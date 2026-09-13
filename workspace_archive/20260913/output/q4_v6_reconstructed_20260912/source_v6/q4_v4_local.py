"""Interruptible RF localization with persistent progress and an exact fallback.

No hypothesis about the source distribution is added. A single paired-probe
stage can return ``complete=False``. That is NOT a successful clear. Only an
actual True response from clear() permits removal of a pending source.
"""
from __future__ import annotations
from dataclasses import dataclass
import q4_baseline as b
from q4_fast_local import nearest_clear_point,BASELINE_LOCALIZE
from q4_v3_local import LocalConfig as V3LocalConfig,optical_cover

@dataclass
class LocalConfig(V3LocalConfig):
    optical_cover_limit:int=12
    pause_after_pair:bool=True

def localize_step(device:b.Device,belief,config:LocalConfig|None=None,on_probe=None)->dict:
    config=config or LocalConfig()
    if config.optical_cover_limit<0:raise ValueError('optical_cover_limit must be nonnegative')
    c=belief.channel
    start=getattr(belief,'rf_rounds',0)
    pair_fails=getattr(belief,'pair_failures',0)
    if belief.obs.status=='strong':
        s=belief.anchor;d=b.dist(device.position,s)
        q=b.add(s,b.mul(min(1,14.5/max(d,1e-12)),b.sub(device.position,s)))
        b.checked_clear(device,c,q)
        return dict(stages=start,pair_failures=pair_fails,certificate='strong',negative_updates=belief.negative_updates)
    for stage in range(start,12):
        s=belief.anchor;theta=belief.obs.theta;poly=belief.poly
        center,radius=b.enclosing_circle(poly)
        if radius<=b.CLEAR_CERT_RADIUS:
            b.checked_clear(device,c,nearest_clear_point(poly,device.position))
            return dict(stages=stage,pair_failures=pair_fails,certificate='intersection',negative_updates=belief.negative_updates)
        if config.optical_cover_limit>0:
            cover=optical_cover(poly)
            if cover and len(cover)<=config.optical_cover_limit:
                if b.dist(device.position,cover[-1])<b.dist(device.position,cover[0]):cover.reverse()
                for j,q in enumerate(cover):
                    device.move(q)
                    if device.clear(c):
                        return dict(stages=stage,pair_failures=pair_fails,certificate='optical_cover_confirmed',optical_attempts=j+1,cover_size=len(cover),negative_updates=belief.negative_updates)
                raise RuntimeError('Certified optical cover failed; inspect feedback/error assumptions')
        upper=min(1500.,max(b.dist(s,p) for p in poly)+1e-7)
        e=b.unit(theta);v=(-e[1],e[0])
        t,h=upper*config.axial,upper*config.lateral
        if config.advance_lower:
            lower=max(0.,min(b.dot(e,b.sub(p,s)) for p in poly))
            advance=config.lower_factor*lower
            if lower/upper>.25:
                advance=max(advance,lower+config.range_fraction*(upper-lower))
                h=max(h,config.adaptive_lateral*upper)
            t=min(.95*upper,max(t,advance))
            h=max(h,t*b.TAN_D*1.05)
        ratio=h/t
        if not (ratio>b.TAN_D and ratio**2+2*ratio*b.TAN_D<1):
            raise ValueError('Unsafe paired-probe geometry')
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
            if not belief.poly:raise RuntimeError('Empty paired-negative feasible polygon')
            belief.tighten()
        # Persist BEFORE yielding. Repeated activation cannot reset the limit.
        belief.rf_rounds=stage+1
        belief.pair_failures=pair_fails
        if config.pause_after_pair:
            return dict(complete=False,stages=stage+1,pair_failures=pair_fails,certificate='replan_after_pair',negative_updates=belief.negative_updates)
    fallback=BASELINE_LOCALIZE(device,b.Track(c,belief.anchor,belief.obs),True)
    return dict(stages=12+fallback['stages'],pair_failures=pair_fails+fallback['pair_failures'],certificate='conservative_fallback:'+fallback['certificate'],negative_updates=belief.negative_updates)
