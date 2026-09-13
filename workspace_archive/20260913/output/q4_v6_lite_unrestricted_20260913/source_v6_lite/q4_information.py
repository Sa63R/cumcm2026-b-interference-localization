"""Conservative use of directional negative observations.

No probability model is used to remove candidate positions. Angular sectors
are OUTER approximations. Negative points qualify only if they are certainly
within the unknown receiver radius for every remaining candidate position.
"""
from __future__ import annotations
import math
import q4_baseline as b
from q4_coverage import hull
TAU=2*math.pi

def intersect_arcs(intervals, direction, halfwidth=math.pi/2):
    """Closed circular interval intersection, with a small outward tolerance."""
    lo=(direction-halfwidth-1e-10)%TAU
    width=min(TAU,2*halfwidth+2e-10)
    segments=[(lo,min(TAU,lo+width))]
    if lo+width>TAU:segments.append((0.,lo+width-TAU))
    return [(max(a,c),min(d,z)) for a,z in intervals for c,d in segments if max(a,c)<=min(d,z)+1e-12]

def point_poly_distance(p, poly):
    # Lower bound via enclosing disk is cheap and conservative.
    c,r=b.enclosing_circle(poly)
    return max(0.,b.dist(p,c)-r)

def tighten_directional(poly, positives, negatives, sectors=24):
    """Return a containing outer polygon; keep input on numerical ambiguity."""
    if not positives or not negatives or not poly:return poly,False
    center,radius=b.enclosing_circle(poly)
    lower_radius=max([1000.]+[max(0.,b.dist(p,center)-radius) for p in positives])
    # Strict numerical slack prevents mistaking just-out-of-range negatives.
    near=[q for q in negatives if max(b.dist(q,v) for v in poly)<lower_radius-1e-6]
    if not near:return poly,False
    arcs=[(0.,TAU)]
    for p in positives:
        for q in near:
            d=b.sub(p,q)
            if b.norm(d)<1e-8:return poly,False
            arcs=intersect_arcs(arcs,math.atan2(d[1],d[0]))
            if not arcs:return poly,False  # Fail conservatively on contradictory feedback.
    # u dot (g-p)<=0 for positive; u dot (g-q)>=0 for near-negative.
    constraints=[(p,1.) for p in positives]+[(q,-1.) for q in near]
    bounds=[max(b.dist(p,v) for v in poly) for p,sgn in constraints]
    pieces=[]
    total_width=sum(z-a for a,z in arcs)
    for a,z in arcs:
        n=max(1,math.ceil(sectors*(z-a)/max(total_width,1e-9)))
        for k in range(n):
            lo=a+(z-a)*k/n;hi=a+(z-a)*(k+1)/n
            u=b.unit((lo+hi)/2);err=2*math.sin((hi-lo)/4)+1e-10
            subpoly=poly[:]
            for (p,sgn),bound in zip(constraints,bounds):
                normal=b.mul(sgn,u)
                subpoly=b.clip_halfplane(subpoly,normal,b.dot(normal,p)+err*bound+1e-7)
                if not subpoly:break
            pieces.extend(subpoly)
    if not pieces:return poly,False
    out=hull(pieces)
    if len(out)<3:return poly,False
    # Convexify only outward; clip to old P to limit numerical accumulation.
    # All pieces are subsets of P; the convex hull is also a subset in exact arithmetic.
    return out,True

def angular_mass(arcs):
    return sum(max(0.,z-a) for a,z in arcs)

def heuristic_visibility(g,positives,negatives,p):
    """A HEURISTIC probability for ranking only, never a safety certificate."""
    distp=b.dist(g,p)
    if distp>1500:return 0.
    rlo=max([1000.]+[b.dist(g,s) for s in positives])
    distance_weight=1. if distp<=rlo else (1500-distp)/max(1e-9,1500-rlo)
    arcs=[(0.,TAU)]
    for s in positives:
        d=b.sub(s,g)
        if b.norm(d)>1e-8:arcs=intersect_arcs(arcs,math.atan2(d[1],d[0]))
    near=[q for q in negatives if b.dist(q,g)<rlo-1e-7]
    for q in near:
        d=b.sub(g,q)
        if b.norm(d)>1e-8:arcs=intersect_arcs(arcs,math.atan2(d[1],d[0]))
    mass=angular_mass(arcs)
    if mass<1e-10:return 0. if near else distance_weight
    d=b.sub(p,g)
    illum=intersect_arcs(arcs,math.atan2(d[1],d[0])) if b.norm(d)>1e-8 else arcs
    direct=angular_mass(illum)/mass
    # Unspecified type prior used only to rank actions; not official generator.
    return distance_weight*(direct if near else .5+.5*direct)

def expected_radius_gain(poly,positives,negatives,p):
    """Three-point deterministic quadrature; planning heuristic, not proof."""
    cen,rad=b.enclosing_circle(poly)
    if rad<=b.CLEAR_CERT_RADIUS:return 0.
    if min((b.dist(s,p) for s in positives+negatives),default=10000)<20:return 0.
    # Capture longitudinal endpoints while remaining inside the polygon.
    far=max(poly,key=lambda v:b.dist(v,cen))
    other=max(poly,key=lambda v:b.dist(v,far))
    samples=[cen,b.add(b.mul(.3,cen),b.mul(.7,far)),b.add(b.mul(.3,cen),b.mul(.7,other))]
    gain=0.
    for g in samples:
        prob=heuristic_visibility(g,positives,negatives,p)
        if prob<=0:continue
        if b.dist(p,g)<=5:
            gain+=prob*rad;continue
        theta=math.atan2(g[1]-p[1],g[0]-p[0])
        new=b.clip_bearing(poly,p,theta,1500.)
        if new:
            nr=b.enclosing_circle(new)[1]
            gain+=prob*max(0.,rad-nr)
    return gain/len(samples)
