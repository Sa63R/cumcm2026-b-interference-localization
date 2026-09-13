"""Pure geometric scoring of hypothetical observations, optionally JIT compiled.
No simulator state enters these functions. Scores are NOT safety certificates.
"""
import math
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args,**kwargs):
        def decorate(fn):return fn
        return decorate

@njit(cache=True)
def clip_small(poly,nx,ny,bound):
    n=len(poly); out=np.empty((n+2,2),dtype=np.float64);k=0
    if n==0:return out[:0]
    x0,y0=poly[-1];f0=nx*x0+ny*y0-bound-1e-8
    for i in range(n):
        x1,y1=poly[i];f1=nx*x1+ny*y1-bound-1e-8
        if (f0<=0)!=(f1<=0):
            t=f0/(f0-f1);out[k,0]=x0+(x1-x0)*t;out[k,1]=y0+(y1-y0)*t;k+=1
        if f1<=0:out[k,0]=x1;out[k,1]=y1;k+=1
        x0,y0,f0=x1,y1,f1
    return out[:k]

@njit(cache=True)
def cut(poly,q,theta):
    a=math.pi/180
    nx=math.sin(theta-a);ny=-math.cos(theta-a)
    pp=clip_small(poly,nx,ny,nx*q[0]+ny*q[1])
    nx=-math.sin(theta+a);ny=math.cos(theta+a)
    return clip_small(pp,nx,ny,nx*q[0]+ny*q[1])

@njit(cache=True)
def approx_center_radius(poly):
    if len(poly)==0:return np.zeros(2),1e9
    # A diameter midpoint is typically exact for the narrow convex polygons.
    best=-1.;ci=0;cj=0
    for i in range(len(poly)):
        for j in range(i):
            dx=poly[i,0]-poly[j,0];dy=poly[i,1]-poly[j,1];d=dx*dx+dy*dy
            if d>best:best=d;ci=i;cj=j
    c=(poly[ci]+poly[cj])/2;r=0.
    for i in range(len(poly)):
        d=(poly[i,0]-c[0])**2+(poly[i,1]-c[1])**2
        if d>r:r=d
    return c,math.sqrt(r)+1e-7

@njit(cache=True)
def sample_area(poly):
    """Degree-2 triangle quadrature; weights sum to one, all points in polygon."""
    n=len(poly)
    if n<3:return poly.copy(),np.ones(n)/max(1,n)
    out=np.empty((3*(n-2),2));weights=np.empty(3*(n-2));k=0
    for i in range(1,n-1):
        a=poly[0];d=poly[i];c=poly[i+1]
        area=abs((d[0]-a[0])*(c[1]-a[1])-(d[1]-a[1])*(c[0]-a[0]))
        for j in range(3):
            z=a if j==0 else (d if j==1 else c)
            out[k]=(a+d+c)/6+z/2;weights[k]=area;k+=1
    total=weights.sum()
    if total<1e-12:return poly.copy(),np.ones(n)/n
    return out,weights/total

@njit(cache=True)
def rank_probes(poly,p,candidates,version,penalty,depth,positive=np.empty((0,2)),negative=np.empty((0,2)),range_prior=False):
    samples,weights=sample_area(poly);ns=len(samples)
    if range_prior:
        ww=weights.copy()
        for j in range(ns):
            lo=1000.;hi=1500.
            for a in positive:lo=max(lo,math.sqrt(((a-samples[j])**2).sum()))
            for a in negative:hi=min(hi,math.sqrt(((a-samples[j])**2).sum()))
            ww[j]*=max(0.,hi-lo)
            if (samples[j]**2).sum()>1800**2:ww[j]=0.
        if ww.sum()>1e-12:weights=ww/ww.sum()
    noise=np.array([-math.sqrt(3/5),0.,math.sqrt(3/5)])*math.pi/180
    probs=np.array([5/18,4/9,5/18])
    scores=np.empty(len(candidates))
    for i in range(len(candidates)):
        q=candidates[i];score=math.sqrt(((q-p)**2).sum())/5+5
        expected=0.
        for j in range(ns):
            g=samples[j];dist=math.sqrt(((g-q)**2).sum());value=0.
            if dist<=5:
                expected+=weights[j]*5;continue
            theta=math.atan2(g[1]-q[1],g[0]-q[0])
            for k in range(3):
                pp=cut(poly,q,theta+noise[k])
                if len(pp)==0:continue
                c1,r1=approx_center_radius(pp)
                d1=math.sqrt(((c1-q)**2).sum());dg=math.sqrt(((g-c1)**2).sum())
                rem=max(0.,math.log2(max(r1,19.5)/19.5))
                if r1<=19.5:
                    future=max(0.,d1-(20-r1))/5+5
                elif version==1:
                    future=max(0.,dist-19.5)/5+6*rem+penalty*r1/5+5
                elif version==2:
                    future=(d1+max(0.,dg-19.5))/5+6*rem+penalty*r1/5+5
                else:
                    # An approximate next step, based only on the first posterior.
                    vec=c1-q;dn=math.sqrt((vec**2).sum())
                    u=vec/max(dn,1e-9);v=np.array([-u[1],u[0]])
                    best=1e30
                    samples2,ws2=sample_area(pp)
                    for frac in np.array([0.,-0.2,0.2]):
                        q2=c1+frac*r1*v
                        future2=0.
                        for jj in range(len(samples2)):
                            g2=samples2[jj];d2=math.sqrt(((g2-q2)**2).sum())
                            if d2<=5:future2+=ws2[jj]*5;continue
                            th2=math.atan2(g2[1]-q2[1],g2[0]-q2[0]);vals=0.
                            for kk in range(3):
                                pp2=cut(pp,q2,th2+noise[kk]);c2,r2=approx_center_radius(pp2)
                                dd=math.sqrt(((q2-c2)**2).sum())
                                if r2<=19.5:
                                    vv=max(0.,dd-(20-r2))/5+5
                                else:
                                    rem2=max(0.,math.log2(max(r2,19.5)/19.5))
                                    vv=max(0.,d2-19.5)/5+6*rem2+penalty*r2/5+5
                                vals+=probs[kk]*vv
                            future2+=ws2[jj]*vals
                        val2=math.sqrt(((q2-q)**2).sum())/5+6+future2
                        if val2<best:best=val2
                    future=best
                value+=probs[k]*future
            expected+=weights[j]*value
        scores[i]=score+expected
    return scores
