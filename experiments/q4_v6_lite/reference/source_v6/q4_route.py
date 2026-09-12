"""Pure Python open-path routing heuristics; no global-optimality claim."""
import q4_baseline as b

def opt2(pos,order,points,passes=20):
    for _ in range(passes):
        changed=False
        for i in range(len(order)-1):
            before=pos if i==0 else points[order[i-1]]
            first=points[order[i]]
            for j in range(i+1,len(order)):
                last=points[order[j]]
                old,new=b.dist(before,first),b.dist(before,last)
                if j+1<len(order):
                    after=points[order[j+1]]
                    old+=b.dist(last,after);new+=b.dist(first,after)
                if new<old-1e-7:
                    order[i:j+1]=reversed(order[i:j+1]);first=points[order[i]];changed=True
        if not changed:break
    return order

def length(pos,route,pts):
    return sum(b.dist(a,pts[j]) for a,j in zip([pos]+[pts[k] for k in route[:-1]],route))

def multi_route(pos, remaining, pts,starts=4):
    if len(remaining)<3:return b.route_order(pos,remaining,pts)
    candidates=sorted(remaining,key=lambda i:b.dist(pos,pts[i]))[:starts]
    best=None;bestlen=float('inf')
    for first in candidates:
        order=[first];todo=set(remaining)-{first};here=pts[first]
        while todo:
            j=min(todo,key=lambda k:(b.dist(here,pts[k]),k));todo.remove(j);order.append(j);here=pts[j]
        opt2(pos,order,pts)
        total=length(pos,order,pts)
        if total<bestlen:best=order;bestlen=total
    return best

def centroid(poly):
    area=0.;x=0.;y=0.
    for a,c in zip(poly,poly[1:]+poly[:1]):
        cr=a[0]*c[1]-a[1]*c[0];area+=cr;x+=(a[0]+c[0])*cr;y+=(a[1]+c[1])*cr
    if abs(area)<1e-5:return b.enclosing_circle(poly)[0]
    return x/(3*area),y/(3*area)
