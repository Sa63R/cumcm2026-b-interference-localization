"""Small open Euclidean routes: deterministic multi-start local search.
Pure computation, no external calls or hidden simulator state.
"""
import numpy as np
from probe_score import njit

@njit(cache=True)
def length(rt,d):
    cost=0.;last=0
    for k in rt:cost+=d[last,k];last=k
    return cost

@njit(cache=True)
def polish(rt,d):
    n=len(rt)
    for _ in range(50):
        best=-1e-7;bi=-1;bj=-1
        for i in range(n):
            pre=0 if i==0 else rt[i-1]
            for j in range(i+1,n):
                delta=d[pre,rt[j]]-d[pre,rt[i]]
                if j+1<n:delta+=d[rt[i],rt[j+1]]-d[rt[j],rt[j+1]]
                if delta<best:best=delta;bi=i;bj=j
        if bi>=0:
            for i in range((bj-bi+1)//2):
                x=rt[bi+i];rt[bi+i]=rt[bj-i];rt[bj-i]=x
            continue
        # A node relocated between two other nodes or to the open endpoint.
        best=-1e-7;bi=-1;bj=-1
        for i in range(n):
            a=0 if i==0 else rt[i-1];x=rt[i]
            removal=-d[a,x]
            if i+1<n:removal+=d[a,rt[i+1]]-d[x,rt[i+1]]
            for j in range(n+1):
                if j==i or j==i+1:continue
                pre=0 if j==0 else rt[j-1]
                delta=removal+d[pre,x]
                if j<n:delta+=d[x,rt[j]]-d[pre,rt[j]]
                if delta<best:best=delta;bi=i;bj=j
        if bi<0:break
        x=rt[bi]
        if bj<bi:
            for k in range(bi,bj,-1):rt[k]=rt[k-1]
            rt[bj]=x
        else:
            for k in range(bi,bj-1):rt[k]=rt[k+1]
            rt[bj-1]=x
    return rt

@njit(cache=True)
def search_route(coords,trials=60):
    n=len(coords)-1;d=np.empty((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):d[i,j]=np.sqrt(((coords[i]-coords[j])**2).sum())
    best=np.arange(1,n+1);bestcost=1e30
    state=1729
    for run in range(max(n,trials)):
        if run<n:
            rt=np.empty(n,dtype=np.int64);rt[0]=run+1
            used=np.zeros(n+1,dtype=np.bool_);used[run+1]=True;last=run+1
            for k in range(1,n):
                val=1e30;idx=-1
                for j in range(1,n+1):
                    if not used[j] and d[last,j]<val:idx=j;val=d[last,j]
                rt[k]=idx;used[idx]=True;last=idx
        else:
            rt=best.copy()
            # Reproducible finite kicks; never seed from the hidden environment.
            for _ in range(2+(run%3)):
                state=(state*48271)%2147483647;i=state%n
                state=(state*48271)%2147483647;j=state%n
                x=rt[i];rt[i]=rt[j];rt[j]=x
        rt=polish(rt,d);cost=length(rt,d)
        if cost<bestcost:bestcost=cost;best=rt.copy()
    return best-1
