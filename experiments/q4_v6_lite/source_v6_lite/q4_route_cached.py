"""Same nearest-neighbour/multi-start 2-opt decisions, cached distances.

No approximate distances or new tie-breakers: distance caching alone must not
change the route. The existing route implementation remains available as a
reference implementation for equivalence tests.
"""
from __future__ import annotations
import q4_baseline as b
from q4_route import centroid,length

def multi_route(pos,remaining,pts,starts=4):
    if len(remaining)<3:return b.route_order(pos,remaining,pts)
    # Index the origin after all points, retaining original point identifiers.
    n=len(pts);all_points=list(pts)+[pos]
    distance=[[b.dist(a,z) for z in all_points] for a in all_points]
    candidates=sorted(remaining,key=lambda i:distance[n][i])[:starts]
    best=None;bestlen=float('inf')
    for first in candidates:
        order=[first];todo=set(remaining)-{first};here=first
        while todo:
            j=min(todo,key=lambda k:(distance[here][k],k));todo.remove(j);order.append(j);here=j
        for _ in range(20):
            changed=False
            for i in range(len(order)-1):
                before=n if i==0 else order[i-1];first_id=order[i]
                for j in range(i+1,len(order)):
                    last=order[j];old=distance[before][first_id];new=distance[before][last]
                    if j+1<len(order):
                        after=order[j+1];old+=distance[last][after];new+=distance[first_id][after]
                    if new<old-1e-7:
                        order[i:j+1]=reversed(order[i:j+1]);first_id=order[i];changed=True
            if not changed:break
        total=sum(distance[a][z] for a,z in zip([n]+order[:-1],order))
        if total<bestlen:best=order;bestlen=total
    return best
