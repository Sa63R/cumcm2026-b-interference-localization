"""Optional exact computation acceleration, without editing archived vendors.

Use ``with ExactComputeCache() as cache: ... one complete case ...`` in a
single-threaded process. The context temporarily swaps selected pure functions,
restores them on every exit (including exceptions), and clears bounded caches.
It does not change search actions, geometry constants, observations or timing.
Keep geometry constants fixed inside a context. Parallel workers should each
have their own process; overlapping/nested contexts are explicitly rejected.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
from functools import lru_cache
import importlib
import itertools
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'q4_comparison'))
import common
import q4_baseline as b

_ACTIVE=None


def _circle_exact(poly):
    """Same support enumeration; stop checking candidates that cannot win."""
    if not poly:raise RuntimeError('Empty feasible polygon; cannot certify a clear')
    center=poly[0];best_radius=0.;hypot=math.hypot
    for p in poly:best_radius=max(best_radius,hypot(center[0]-p[0],center[1]-p[1]))
    def consider(c,r):
        nonlocal center,best_radius
        if r>best_radius+b.EPS:return
        actual=0.
        for p in poly:
            actual=max(actual,hypot(c[0]-p[0],c[1]-p[1]))
            # The original update requires BOTH inequalities. Once either
            # fails, remaining distances cannot restore this candidate.
            if actual>=best_radius or actual>r+1e-6:return
        center,best_radius=c,actual
    for a,z in itertools.combinations(poly,2):
        c=(.5*(a[0]+z[0]),.5*(a[1]+z[1]))
        consider(c,hypot(c[0]-a[0],c[1]-a[1]))
    for a,z,p in itertools.combinations(poly,3):
        circle=b.circumcircle(a,z,p)
        if circle is not None:consider(*circle)
    return center,best_radius+1e-7


def _clip_halfplane_exact(poly,normal,bound):
    """Inline the existing point helpers, preserving each float operation."""
    if not poly:return []
    bound+=b.EPS
    nx,ny=normal;out=[];a=poly[-1];ax,ay=a
    fa=(nx*ax+ny*ay)-bound
    for z in poly:
        zx,zy=z;fz=(nx*zx+ny*zy)-bound
        ia,iz=fa<=0.0,fz<=0.0
        if ia!=iz:
            weight=fa/(fa-fz)
            out.append((ax+weight*(zx-ax),ay+weight*(zy-ay)))
        if iz:out.append(z)
        a,fa=z,fz;ax,ay=zx,zy
    cleaned=[]
    hypot=math.hypot
    for point in out:
        if not cleaned or hypot(point[0]-cleaned[-1][0],point[1]-cleaned[-1][1])>1e-8:
            cleaned.append(point)
    if len(cleaned)>1 and hypot(cleaned[0][0]-cleaned[-1][0],cleaned[0][1]-cleaned[-1][1])<1e-8:
        cleaned.pop()
    return cleaned


def _route_exact(pos,remaining,pts,starts,small_route):
    """Same candidates, tie breaks, 2-opt reversals and floating operations."""
    if len(remaining)<3:return small_route(pos,remaining,pts)
    n=len(pts);all_points=list(pts)+[pos];distance=[[b.dist(a,z) for z in all_points] for a in all_points]
    candidates=sorted(remaining,key=lambda i:distance[n][i])[:starts]
    best=None;bestlen=float('inf')
    for first in candidates:
        order=[first];todo=set(remaining)-{first};here=first
        while todo:
            row=distance[here]
            # Retain the exact (distance, integer index) tie break.
            j=next(iter(todo));nearest=row[j]
            for k in todo:
                value=row[k]
                if value<nearest or (value==nearest and k<j):j,nearest=k,value
            todo.remove(j);order.append(j);here=j
        size=len(order)
        for _ in range(20):
            changed=False
            for i in range(size-1):
                before=n if i==0 else order[i-1];first_id=order[i]
                before_row=distance[before];first_row=distance[first_id]
                # The final j has no following edge. Splitting it out removes
                # a branch and repeated len() calls, preserving traversal order.
                for j in range(i+1,size-1):
                    last=order[j];after=order[j+1]
                    old=before_row[first_id];new=before_row[last]
                    old+=distance[last][after];new+=first_row[after]
                    if new<old-1e-7:
                        order[i:j+1]=reversed(order[i:j+1])
                        first_id=order[i];first_row=distance[first_id];changed=True
                last=order[-1]
                old=before_row[first_id];new=before_row[last]
                if new<old-1e-7:
                    order[i:]=reversed(order[i:]);changed=True
            if not changed:break
        total=sum(distance[a][z] for a,z in zip([n]+order[:-1],order))
        if total<bestlen:best=order;bestlen=total
    return best


class ExactComputeCache(AbstractContextManager):
    def __init__(self, *, polygon_limit=2048, gain_limit=512, route_limit=128,
                 optical_limit=512, fast_route=True):
        limits=(polygon_limit,gain_limit,route_limit,optical_limit)
        if any(not isinstance(v,int) or v<1 for v in limits):
            raise ValueError('Cache entry limits must be positive integers')
        self.limits=limits;self.fast_route=fast_route
        self._changes=[];self._caches={};self._last_stats={};self._entered=False

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE is not None or self._entered:
            raise RuntimeError('ExactComputeCache requires a single nonoverlapping context')
        _ACTIVE=self;self._entered=True
        try:
            info=importlib.import_module('q4_information')
            local3=importlib.import_module('q4_v3_local')
            local4=importlib.import_module('q4_v4_local')
            online=importlib.import_module('online')
            route=importlib.import_module('q4_route_cached')
            original_circle=b.enclosing_circle
            original_gain=info.expected_radius_gain
            original_optical=local3.optical_cover
            original_route=route.multi_route
            polygon_limit,gain_limit,route_limit,optical_limit=self.limits

            @lru_cache(maxsize=polygon_limit)
            def circle_cached(poly,eps):
                return _circle_exact(poly)
            def circle(poly):
                return circle_cached(tuple(poly),b.EPS)

            @lru_cache(maxsize=gain_limit)
            def gain_cached(poly,positives,negatives,p,tan_d,clear_radius):
                return original_gain(poly,positives,negatives,p)
            def gain(poly,positives,negatives,p):
                return gain_cached(tuple(poly),tuple(positives),tuple(negatives),tuple(p),b.TAN_D,b.CLEAR_CERT_RADIUS)

            @lru_cache(maxsize=optical_limit)
            def optical_cached(poly,radius):
                return tuple(original_optical(poly,radius))
            def optical(poly,radius=19.5):
                # Callers reverse returned lists: never expose the cached tuple.
                return list(optical_cached(tuple(poly),radius))

            @lru_cache(maxsize=route_limit)
            def route_cached(pos,remaining,pts,starts):
                if self.fast_route:
                    return tuple(_route_exact(pos,remaining,pts,starts,b.route_order))
                return tuple(original_route(pos,remaining,pts,starts))
            def multi_route(pos,remaining,pts,starts=4):
                return list(route_cached(tuple(pos),tuple(remaining),tuple(pts),starts))

            self._caches=dict(circle=circle_cached,gain=gain_cached,optical=optical_cached,route=route_cached)
            patches=[(b,'enclosing_circle',circle),(b,'clip_halfplane',_clip_halfplane_exact),
                     (info,'expected_radius_gain',gain),
                     (online,'expected_radius_gain',gain),(local3,'optical_cover',optical),
                     (local4,'optical_cover',optical),(route,'multi_route',multi_route),
                     (online,'multi_route',multi_route)]
            # This optional integration is an exact alias substitution only;
            # the acceleration module never imports or changes its algorithm.
            custom=sys.modules.get('local_candidate')
            if custom is not None and getattr(custom,'optical_cover',None) is original_optical:
                patches.append((custom,'optical_cover',optical))
            for module,name,replacement in patches:
                self._changes.append((module,name,getattr(module,name)))
                setattr(module,name,replacement)
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def stats(self):
        if self._entered:
            return {name:fn.cache_info()._asdict() for name,fn in self._caches.items()}
        return self._last_stats

    def clear(self):
        """Optional explicit case boundary when retaining a worker context."""
        for fn in self._caches.values():fn.cache_clear()

    def __exit__(self,exc_type,exc_value,traceback):
        global _ACTIVE
        self._last_stats=self.stats()
        for module,name,original in reversed(self._changes):setattr(module,name,original)
        self._changes.clear();self.clear();self._caches.clear();self._entered=False
        if _ACTIVE is self:_ACTIVE=None
        return False
