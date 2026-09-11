"""Bounded optimization within a conservatively certified clear region.

For polygon C, every source in C is within 20 m of x if every vertex is.
The intersection of these vertex disks is convex. Incoming-only optimization
enumerates its circular-arc projections and vertices. Anchored ray search does
not claim a global optimum over that continuous intersection.
"""
import math
import time

from simulator_client.state import Position


POLYGON_RADIUS = 19.99998
NEAR_RADIUS = 14.99998
VERIFY_MARGIN = 0.00001
MAX_VERTICES = 64


def _incoming_candidates(original, current, centers, radius):
    """Complete boundary candidate classes in exact arithmetic for equal disks."""
    candidates=[original,current]
    for center in centers:
        dx,dy=current.x-center.x,current.y-center.y
        distance=math.hypot(dx,dy)
        if distance>0:
            candidates.append(Position(center.x+radius*dx/distance,center.y+radius*dy/distance))
    for i,a in enumerate(centers):
        for b in centers[i+1:]:
            dx,dy=b.x-a.x,b.y-a.y
            distance=math.hypot(dx,dy)
            if distance==0 or distance>2*radius:
                continue
            midpoint=Position((a.x+b.x)/2,(a.y+b.y)/2)
            height=math.sqrt(max(0.,radius*radius-distance*distance/4))
            for sign in (-1.,1.):
                candidates.append(Position(midpoint.x+sign*height*(-dy/distance),
                                           midpoint.y+sign*height*(dx/distance)))
    return candidates


def _ray_limit(origin, direction, centers, radii):
    """Positive endpoint of the disk intersection containing ray origin."""
    limit = math.inf
    for center, radius in zip(centers, radii):
        dx, dy = origin.x-center.x, origin.y-center.y
        b = dx*direction[0]+dy*direction[1]
        # Origin feasibility is independently checked before calling us.
        # Clamp only its roundoff-sensitive quadratic constant; final points
        # are checked against all actual distances, including incoming cost.
        c = min(0., dx*dx+dy*dy-radius*radius)
        disc = b*b-c
        if not math.isfinite(disc) or disc < 0:
            raise ArithmeticError("Nonfinite disk/ray intersection")
        limit = min(limit, max(0., -b+math.sqrt(disc)))
    if not math.isfinite(limit):
        raise ArithmeticError("Unbounded ray")
    return limit


def choose_certified_clear_point(original, current, *, vertices=None,
                                 near_point=None, anchor=None, config="incoming"):
    """Return (point, evidence); invalid geometry returns the original point.

    Exactly one of vertices/near_point is required to optimize. A near result
    places the source within 5 m of near_point, leaving a 14.99998 m service
    disk. A further disk enforces incoming distance no greater than original.
    """
    if config not in {"incoming", "anchored"}:
        raise ValueError("Unknown clear-region configuration")
    began = time.perf_counter()
    original, current = Position.coerce(original), Position.coerce(current)
    log = {"status": "fallback", "certificate": {"passed": False},
           "ray_count": 0, "candidate_count": 1, "proxy_saved_s": 0.}
    try:
        if (vertices is None) == (near_point is None):
            raise ValueError("Need exactly one observation certificate")
        if near_point is not None:
            centers = [Position.coerce(near_point)]
            radius, verification = NEAR_RADIUS, 15.-VERIFY_MARGIN
            certificate = {"kind": "near_disk", "near_point": [centers[0].x, centers[0].y]}
        else:
            centers = [Position.coerce(v) for v in vertices]
            if not centers:
                raise ValueError("Empty polygon")
            radius, verification = POLYGON_RADIUS, 20.-VERIFY_MARGIN
            certificate = {"kind": "polygon_vertices", "vertices": [[v.x,v.y] for v in centers]}
        certificate.update(search_radius_m=radius, verification_radius_m=verification, passed=False)
        log["certificate"] = certificate
        if len(centers)>MAX_VERTICES:
            raise ValueError("Vertex budget exceeded")
        if max(original.distance_to(v) for v in centers) > radius:
            raise ValueError("Original point lacks the optimization certificate")
        incoming = current.distance_to(original)
        endpoint = Position.coerce(anchor) if anchor is not None and config == "anchored" else None
        def objective(point):
            return current.distance_to(point)+(point.distance_to(endpoint) if endpoint is not None else 0.)
        old_score = objective(original)
        best, best_score = original, old_score
        def consider(candidate):
            nonlocal best,best_score
            if (current.distance_to(candidate) > incoming or
                    max(candidate.distance_to(v) for v in centers) > verification):
                return
            score=objective(candidate)
            if math.isfinite(score) and score < best_score-1e-10:
                best,best_score=candidate,score
        log["method"]="boundary_candidates" if endpoint is None else "finite_rays"
        if endpoint is None:
            candidates=_incoming_candidates(original,current,centers,radius)
            log["candidate_count"]=len(candidates)
            for candidate in candidates:
                consider(candidate)
        directions = [(math.cos(2*math.pi*i/64),math.sin(2*math.pi*i/64)) for i in range(64)]
        for target in (current, endpoint):
            if target is not None:
                dx,dy=target.x-original.x,target.y-original.y
                norm=math.hypot(dx,dy)
                if norm>0: directions.append((dx/norm,dy/norm))
        for direction in directions if endpoint is not None else ():
            upper = _ray_limit(original,direction,centers+[current],[radius]*len(centers)+[incoming])
            log["ray_count"] += 1
            if upper <= 0:
                continue
            def point(t):
                return Position(original.x+t*direction[0],original.y+t*direction[1])
            left,right=0.,upper
            for _ in range(32):
                one=left+(right-left)/3;two=right-(right-left)/3
                if objective(point(one)) <= objective(point(two)):
                    right=two
                else:
                    left=one
            projection=(current.x-original.x)*direction[0]+(current.y-original.y)*direction[1]
            for t in (upper,(left+right)/2,min(upper,max(0.,projection))):
                candidate=point(t)
                log["candidate_count"] += 1
                # These are the authoritative acceptance checks, not the
                # quadratic discriminant's roundoff treatment or proxy score.
                consider(candidate)
        max_distance=max(best.distance_to(v) for v in centers)
        if not (max_distance <= verification and current.distance_to(best) <= incoming
                and math.isfinite(best_score) and best_score <= old_score):
            raise ArithmeticError("Final certificate/cost check failed")
        certificate.update(passed=True,max_distance_m=max_distance)
        log.update(status="optimized" if best != original else "original_best",
            original_incoming_m=incoming,selected_incoming_m=current.distance_to(best),
            original_objective_m=old_score,selected_objective_m=best_score,
            objective="two_segment" if endpoint is not None else "incoming",
            proxy_saved_s=(old_score-best_score)/5.)
        return best, log
    except (ValueError,TypeError,ArithmeticError,OverflowError) as exc:
        log.update(status="fallback",reason=type(exc).__name__+": "+str(exc))
        return original, log
    finally:
        log["runtime_s"]=time.perf_counter()-began
