"""Finite rotations of a seven-site full-disk discovery certificate."""

import math

from planning.coverage import omni_coverage_points
from simulator_client.state import Position


def ring_cover_radius(radius):
    """Worst nearest-site distance in the radius-1800 disk, for any phase.

    Within a 60-degree sector the only candidates are origin/two ring sites.
    Interior Voronoi vertices have distance a/sqrt(3); the outer arc peaks
    halfway between ring sites, at sqrt(R^2+a^2-sqrt(3)*R*a).
    For the accepted a range these witnesses exist in the disk. Rotating the
    sites and disk together leaves distances unchanged, hence any phase works.
    This is an analytic formula, not a sampled boundary-only assertion.
    """
    omni_coverage_points(radius)  # Public validated interval with positive margin.
    return max(radius / math.sqrt(3),
               math.sqrt(1800**2 + radius**2 - math.sqrt(3) * 1800 * radius))


def ring_sites(radius, phase_deg=0.0):
    if (isinstance(phase_deg, bool) or not isinstance(phase_deg, (float, int))
            or not math.isfinite(phase_deg)):
        raise ValueError("phase_deg must be finite")
    ring_cover_radius(radius)
    phase = math.radians(phase_deg % 60)
    return (Position(0, 0),) + tuple(
        Position(radius * math.cos(i * math.pi / 3 + phase),
                 radius * math.sin(i * math.pi / 3 + phase)) for i in range(6))
