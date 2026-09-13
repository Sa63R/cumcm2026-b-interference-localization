"""Directed entry/exit route heuristic; no optimal-tour claim."""
import math
import q4_baseline as b

def matrix_route(pos, exits, entries=None, starts=8, relocate=True):
    n=len(exits)
    if n==0:return []
    if entries is None:entries=[[p] for p in exits]
    origins=exits+[pos]
    d=[[0.]*n for _ in range(n+1)]
    for i,p in enumerate(origins):
        for j,qs in enumerate(entries):
            d[i][j]=min(b.dist(p,q)+b.dist(q,exits[j]) for q in qs)
    def length(order):
        return sum(d[a][z] for a,z in zip([n]+order[:-1],order))
    def edge(a,z):return 0. if z is None else d[a][z]
    def improve(order):
        for it in range(40):
            # Prefix of the internal forward-to-reverse differences.
            pref=[0.]
            for a,z in zip(order,order[1:]):pref.append(pref[-1]+d[z][a]-d[a][z])
            best=-1e-7;change=None
            for i in range(n-1):
                p=n if i==0 else order[i-1];a=order[i]
                for j in range(i+1,n):
                    z=order[j];after=order[j+1] if j+1<n else None
                    delta=d[p][z]-d[p][a]+edge(a,after)-edge(z,after)+pref[j]-pref[i]
                    if delta<best:best=delta;change=(i,j)
            if change:
                i,j=change;order[i:j+1]=reversed(order[i:j+1]);continue
            if not relocate:break
            best=-1e-7;change=None
            for count in (1,2):
                for i in range(n-count+1):
                    chunk=order[i:i+count];rest=order[:i]+order[i+count:]
                    p=n if i==0 else order[i-1];after=order[i+count] if i+count<n else None
                    removal=edge(p,after)-d[p][chunk[0]]-edge(chunk[-1],after)
                    for j in range(len(rest)+1):
                        if j==i:continue
                        q=n if j==0 else rest[j-1];nxt=rest[j] if j<len(rest) else None
                        delta=removal+d[q][chunk[0]]+edge(chunk[-1],nxt)-edge(q,nxt)
                        if delta<best:best=delta;change=rest[:j]+chunk+rest[j:]
            if change is None:break
            order=change
        return order
    best=None;value=float('inf')
    firsts=sorted(range(n),key=lambda k:(d[n][k],k))[:max(1,starts)]
    for first in firsts:
        order=[first];todo=set(range(n))-{first}
        while todo:
            j=min(todo,key=lambda k:(d[order[-1]][k],k));todo.remove(j);order.append(j)
        order=improve(order);cost=length(order)
        if cost<value:value=cost;best=order
    return best

def predicted_entries(tr,config):
    if tr.obs.status=='strong':return [tr.anchor]
    center,radius=b.enclosing_circle(tr.poly)
    if radius<=b.CLEAR_CERT_RADIUS:return [center]
    s=tr.anchor;e=b.unit(tr.obs.theta);v=(-e[1],e[0])
    U=min(1500.,max(b.dist(s,p) for p in tr.poly)+1e-7)
    t=U*config.axial;h=U*config.lateral
    if config.advance_lower:
        lower=max(0.,min(b.dot(e,b.sub(p,s)) for p in tr.poly))
        t=min(.95*U,max(t,config.lower_factor*lower));h=max(h,t*b.TAN_D*1.05)
    mid=b.add(s,b.mul(t,e))
    return [b.add(mid,b.mul(h,v)),b.add(mid,b.mul(-h,v))]
