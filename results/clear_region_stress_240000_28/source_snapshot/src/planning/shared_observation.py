"""Selective local cross-channel value, shared principle with q3-geometric.

Functions are reproduced from geometric_joint.py so this research branch has
an independent runnable snapshot. No local safety region is mutated by scoring.
"""

import math

from simulator_client.state import Position


def polygon_quadrature(vertices, limit=9):
    """Equal-mass deterministic selection from triangular area quadrature.

    This defines a planning proxy on the observed polygon, not the actual
    posterior of the hidden source/radius. Degenerate segments use midpoints.
    """
    if not vertices:
        return ()
    if len(vertices) < 3:
        a, b = vertices[0], vertices[-1]
        return tuple((a[0]+(b[0]-a[0])*(i+.5)/limit,
                      a[1]+(b[1]-a[1])*(i+.5)/limit) for i in range(limit))
    anchor = vertices[0]
    samples = []
    for a, b in zip(vertices[1:-1], vertices[2:]):
        twice_area = abs((a[0]-anchor[0])*(b[1]-anchor[1])
                         -(a[1]-anchor[1])*(b[0]-anchor[0]))
        if twice_area <= 1e-12:
            continue
        for weights in ((4, 1, 1), (1, 4, 1), (1, 1, 4)):
            point = tuple((weights[0]*anchor[j]+weights[1]*a[j]+weights[2]*b[j])/6
                          for j in range(2))
            samples.append((point, twice_area/3))
    if not samples:
        return polygon_quadrature((vertices[0], vertices[-1]), limit)
    total = sum(weight for _, weight in samples)
    result, index, accumulated = [], 0, samples[0][1]
    for i in range(limit):
        mass = total*(i+.5)/limit
        while index+1 < len(samples) and accumulated < mass:
            index += 1
            accumulated += samples[index][1]
        result.append(samples[index][0])
    return tuple(result)


def shared_observation_value(region, position):
    """Estimate travel avoided by updating an uncertain destination early.

    For old centre c and hypothetical updated centre c', the nonnegative
    triangle excess |q-c|+|c-c'|-|q-c'| compares two explicit local routes.
    Averaging it over quadrature points/errors is ONLY a ranking heuristic.
    Future coverage/ordering can change the benefit; no global bound is claimed.
    """
    if not region.vertices or not region.observations:
        return None
    q = Position.coerce(position)
    circle = region.enclosing_disk()
    if circle.radius <= 19.9:
        return None
    # A maximum of a convex distance over the outer polygon is attained at a
    # vertex. This check is a sufficient guaranteed-reception certificate.
    if max(math.hypot(v[0]-q.x, v[1]-q.y) for v in region.vertices) > 1000-1e-6:
        return None
    center = Position.coerce(circle.center)
    hypotheses = polygon_quadrature(region.vertices)
    travel, radius, mass = 0., 0., 0.
    for source in hypotheses:
        distance = math.hypot(source[0]-q.x, source[1]-q.y)
        for error, weight in ((-region.error_deg, .25), (0., .5), (region.error_deg, .25)):
            if distance <= 5:
                updated, new_radius = q, 5.
            else:
                bearing = (math.degrees(math.atan2(source[1]-q.y, source[0]-q.x))+error) % 360
                hypothetical = region.copy().observe(q, bearing)
                if not hypothetical.vertices:
                    continue
                new_circle = hypothetical.enclosing_disk()
                updated, new_radius = Position.coerce(new_circle.center), new_circle.radius
            excess = max(0., q.distance_to(center)+center.distance_to(updated)-q.distance_to(updated))
            travel += weight*excess
            radius += weight*new_radius
            mass += weight
    if mass <= 0:
        return None
    return dict(travel_saving_proxy_m=travel/mass, predicted_radius_proxy_m=radius/mass,
                original_radius_m=circle.radius, hypotheses=len(hypotheses),
                guaranteed_reception=True, proxy_kind='polygon_quadrature_local_triangle_excess')

