"""Adaptive box certificate for continuous, all-direction coverage.

Every accepted box is inside the convex hull of sites that are within 1000 m
of the ENTIRE box. Boxes outside the target disk are safely excluded.
A failure or unresolved box never authorizes dropping a scan site.
"""
from __future__ import annotations
import math,json,time
import q4_baseline as b


def cross(o,a,c):return (a[0]-o[0])*(c[1]-o[1])-(a[1]-o[1])*(c[0]-o[0])
def hull(points):
    p=sorted(set(points))
    if len(p)<3:return p
    lo=[];hi=[]
    for x in p:
        while len(lo)>1 and cross(lo[-2],lo[-1],x)<=1e-9:lo.pop()
        lo.append(x)
    for x in reversed(p):
        while len(hi)>1 and cross(hi[-2],hi[-1],x)<=1e-9:hi.pop()
        hi.append(x)
    return lo[:-1]+hi[:-1]

def contains(h,pt,margin=1e-7):
    return len(h)>=3 and all(cross(v,w,pt)>=margin*math.dist(v,w) for v,w in zip(h,h[1:]+h[:1]))

def certify(sites,max_depth=22,keep_leaves=False):
    queue=[(0.,0.,1800.,0)];accepted=[];stats={'accepted':0,'outside':0,'subdivided':0,'max_depth':0};tic=time.time()
    while queue:
        x,y,h,depth=queue.pop();stats['max_depth']=max(stats['max_depth'],depth)
        # exact distance from origin to axis-aligned box
        near=math.hypot(max(abs(x)-h,0),max(abs(y)-h,0))
        if near>1800.+1e-7:stats['outside']+=1;continue
        rad=math.sqrt(2)*h
        active=[(i,s) for i,s in enumerate(sites) if math.dist(s,(x,y))+rad<1000.-1e-6]
        hh=hull([s for i,s in active])
        corners=[(x+dx*h,y+dy*h) for dx,dy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        if all(contains(hh,c) for c in corners):
            stats['accepted']+=1
            if keep_leaves:accepted.append([x,y,h,[i for i,s in active]])
            continue
        if math.hypot(x,y)<=1800.+1e-9:
            truehull=hull([s for s in sites if math.dist(s,(x,y))<=1000.])
            if not contains(truehull,(x,y),0):
                return {'ok':False,'witness':[x,y,h,depth],**stats,'wall':time.time()-tic}
        if depth>=max_depth:
            return {'ok':False,'unresolved':[x,y,h,depth],**stats,'wall':time.time()-tic}
        hh2=h/2
        for dx,dy in [(-1,-1),(1,-1),(-1,1),(1,1)]:queue.append((x+dx*hh2,y+dy*hh2,hh2,depth+1))
        stats['subdivided']+=1
    return {'ok':True,**stats,'wall':time.time()-tic,'leaves':accepted if keep_leaves else None}
