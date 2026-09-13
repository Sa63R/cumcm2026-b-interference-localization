"""Exact acceleration of the unchanged adaptive continuous coverage proof.

No proof condition, threshold, box traversal, subdivision or result field is
changed. Only the wall-time diagnostic differs. Hull geometry is cached within
one certificate call and released on return. Import/build strategies before
entering the optional single-process, nonoverlapping temporary context.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
from functools import lru_cache
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'q4_comparison'))
import common
import q4_coverage

_ACTIVE=None


def _contains_exact(geometry,point,margin=1e-7):
    if geometry is None:return False
    x,y=point
    for ox,oy,dx,dy,length in geometry:
        if not (dx*(y-oy)-dy*(x-ox)>=margin*length):return False
    return True


def certify_exact(sites,max_depth=22,keep_leaves=False,*,cache_limit=512,_record=None):
    """Same certificate as q4_coverage.certify, with identical non-wall data."""
    if not isinstance(cache_limit,int) or cache_limit<1:raise ValueError('cache_limit must be a positive integer')
    @lru_cache(maxsize=cache_limit)
    def geometry(indices):
        hh=q4_coverage.hull([sites[i] for i in indices])
        if len(hh)<3:return None
        return tuple((v[0],v[1],w[0]-v[0],w[1]-v[1],math.dist(v,w)) for v,w in zip(hh,hh[1:]+hh[:1]))
    queue=[(0.,0.,1800.,0)];accepted=[];stats={'accepted':0,'outside':0,'subdivided':0,'max_depth':0};tic=time.time()
    try:
        while queue:
            x,y,h,depth=queue.pop();stats['max_depth']=max(stats['max_depth'],depth)
            near=math.hypot(max(abs(x)-h,0),max(abs(y)-h,0))
            if near>1800.+1e-7:stats['outside']+=1;continue
            rad=math.sqrt(2)*h;center=(x,y)
            # Exactly the original distance operations, evaluated once for
            # both full-box active sites and the possible point witness.
            distances=[math.dist(s,center) for s in sites]
            active=tuple(i for i,d in enumerate(distances) if d+rad<1000.-1e-6)
            hh=geometry(active)
            corners=[(x+dx*h,y+dy*h) for dx,dy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
            if all(_contains_exact(hh,c) for c in corners):
                stats['accepted']+=1
                if keep_leaves:accepted.append([x,y,h,list(active)])
                continue
            if math.hypot(x,y)<=1800.+1e-9:
                truehull=geometry(tuple(i for i,d in enumerate(distances) if d<=1000.))
                if not _contains_exact(truehull,center,0):
                    return {'ok':False,'witness':[x,y,h,depth],**stats,'wall':time.time()-tic}
            if depth>=max_depth:
                return {'ok':False,'unresolved':[x,y,h,depth],**stats,'wall':time.time()-tic}
            half=h/2
            for dx,dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:queue.append((x+dx*half,y+dy*half,half,depth+1))
            stats['subdivided']+=1
        return {'ok':True,**stats,'wall':time.time()-tic,'leaves':accepted if keep_leaves else None}
    finally:
        info=geometry.cache_info()._asdict()
        geometry.cache_clear()
        if _record is not None:_record(info)


class CoverageComputeCache(AbstractContextManager):
    """Temporary exact function aliases, independent of compute_fast context."""
    def __init__(self,*,cache_limit=512):
        if not isinstance(cache_limit,int) or cache_limit<1:raise ValueError('cache_limit must be a positive integer')
        self.cache_limit=cache_limit;self._changes=[];self._entered=False
        self._stats=dict(calls=0,hits=0,misses=0,peak_call_entries=0,limit=cache_limit)

    def _record(self,info):
        self._stats['calls']+=1;self._stats['hits']+=info['hits'];self._stats['misses']+=info['misses']
        self._stats['peak_call_entries']=max(self._stats['peak_call_entries'],info['currsize'])

    def stats(self):return dict(self._stats)

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE is not None or self._entered:raise RuntimeError('CoverageComputeCache requires a nonoverlapping context')
        _ACTIVE=self;self._entered=True
        original=q4_coverage.certify
        def replacement(sites,max_depth=22,keep_leaves=False):
            return certify_exact(sites,max_depth,keep_leaves,cache_limit=self.cache_limit,_record=self._record)
        # Substitute only already-imported aliases that still refer to this
        # exact original function. Never edit a file or alter unrelated APIs.
        names=('q4_coverage','q4_v3_solver','runtime','coverage_candidate','planner',
               'unrestricted_planner','bench','benchmark','domain_candidate')
        for name in names:
            module=sys.modules.get(name)
            if module is not None and getattr(module,'certify',None) is original:
                self._changes.append((module,'certify',original));module.certify=replacement
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        global _ACTIVE
        for module,name,original in reversed(self._changes):setattr(module,name,original)
        self._changes.clear();self._entered=False
        if _ACTIVE is self:_ACTIVE=None
        return False
