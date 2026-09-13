"""Sufficient no-signal inference for an already observed Q3 source only."""

import math

from simulator_client.state import Position


def polygon_distance(region,position):
    """Distance to the complete convex polygon, including degenerate edges."""
    q=Position.coerce(position)
    vertices=region.vertices
    if not vertices:
        return None
    if region.contains((q.x,q.y)):
        return 0.
    if len(vertices)==1:
        return q.distance_to(Position(*vertices[0]))
    best=math.inf
    for a,b in zip(vertices,vertices[1:]+vertices[:1]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        length=dx*dx+dy*dy
        fraction=max(0.,min(1.,((q.x-a[0])*dx+(q.y-a[1])*dy)/length)) if length else 0.
        best=min(best,math.hypot(q.x-a[0]-fraction*dx,q.y-a[1]-fraction*dy))
    return best


def certify_silence(region,position,margin_m=1e-5):
    """If d(q,C)>1500+margin, every x in C is outside every permitted R.

    C is a conservative outer source region. The positive numerical margin is
    engineering protection for task-scale binary64, not interval arithmetic.
    The caller MUST restrict inference to already observed source channels;
    this routine does not create physical measurements or coverage evidence.
    """
    if not math.isfinite(margin_m) or margin_m<=0:
        raise ValueError("margin_m must be positive and finite")
    if not region.vertices or not region.observations:
        return None
    q=Position.coerce(position)
    circle=region.enclosing_disk()
    lower=q.distance_to(Position(*circle.center))-circle.radius
    method="enclosing_disk"
    if lower<=1500+margin_m:
        lower=polygon_distance(region,q)
        method="polygon_edges"
    if lower is None or lower<=1500+margin_m:
        return None
    return {"method":method,"distance_lower_m":lower,"maximum_reception_radius_m":1500.,
            "margin_m":margin_m,"outer_region_vertex_count":len(region.vertices)}
