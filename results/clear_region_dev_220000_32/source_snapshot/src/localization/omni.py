"""Convex distance-comparison updates for a stationary Q3 omnidirectional source.

All observations in one region must refer to the same channel and to the same
uncleared source, with a fixed reception radius. This module must NOT be used
for Q4: a directional source can be silent on its back side at any distance.
"""

from __future__ import annotations

import math

from geometry import HalfPlane, Point, clip_polygon, point
from . import CandidateRegion


class OmniCandidateRegion(CandidateRegion):
    """Retain Q3 sources compatible with both positive and negative observations.

    A positive bearing from p gives |s-p| <= R. No signal at n gives |s-n| > R
    for an existing omnidirectional source, hence |s-p| < |s-n|. Squaring and
    relaxing this strict inequality to its closure yields the safe half-plane

        (n-p) . s <= (|n|^2 - |p|^2) / 2.

    Each new positive observation is paired with every stored negative point,
    and each new negative point with every stored positive observation. This
    preserves all information regardless of arrival order. Negative-only data
    are stored without cutting the prior: this class does not subtract the
    nonconvex radius-1000 disks, nor incorrectly exclude radius-1500 disks.

    The result is a conservative outer region. An empty region certifies an
    inconsistency with these assumptions; a nonempty region does not prove
    consistency of all original strict inequalities. Exact positive/negative
    responses at the same coordinate are detected as inconsistent directly.
    """

    def __init__(self, prior_radius: float = 1800.0, disk_sides: int = 128,
                 error_deg: float = 1.005, reception_radius: float = 1500.0):
        super().__init__(prior_radius, disk_sides, error_deg, reception_radius)
        self.no_signal_positions: list[Point] = []
        self._no_signal_set: set[Point] = set()

    def copy(self) -> "OmniCandidateRegion":
        # CandidateRegion.copy constructs CandidateRegion explicitly; preserve
        # this subtype, its independent histories, and its immutable MEC cache.
        other = object.__new__(type(self))
        other.__dict__ = dict(self.__dict__)
        other.observations = list(self.observations)
        other.no_signal_positions = list(self.no_signal_positions)
        other._no_signal_set = set(self._no_signal_set)
        return other

    def _compare_distances(self, positive: Point, negative: Point) -> None:
        dx, dy = negative[0] - positive[0], negative[1] - positive[1]
        length = math.hypot(dx, dy)
        if length == 0:
            # With a fixed source/radius, the same point cannot give both
            # direction and no_signal. Its strict inequality would be 0 < 0;
            # the generic closed-half-plane relaxation would lose this fact.
            self.vertices = ()
            self._circle = None
            return
        if not self.vertices:
            return
        nx, ny = dx / length, dy / length
        # Normal . midpoint is algebraically identical to the difference of
        # squared norms, but avoids subtracting large nearly equal squares.
        midpoint = (positive[0] + dx / 2, positive[1] + dy / 2)
        constraint = HalfPlane(nx, ny, nx * midpoint[0] + ny * midpoint[1])
        self.vertices = clip_polygon(self.vertices, constraint)
        self._circle = None

    def observe(self, position, bearing_deg: float) -> "OmniCandidateRegion":
        super().observe(position, bearing_deg)
        positive = self.observations[-1].position
        for negative in self.no_signal_positions:
            self._compare_distances(positive, negative)
            if not self.vertices:
                break
        return self

    def observe_no_signal(self, position) -> "OmniCandidateRegion":
        """Store one Q3 no_signal position and add all available comparisons.

        Repeated negative coordinates are idempotent. No hidden radius, source
        position or source count is accepted by this method.
        """
        negative = point(position)
        if negative in self._no_signal_set:
            return self
        self.no_signal_positions.append(negative)
        self._no_signal_set.add(negative)
        for observation in self.observations:
            self._compare_distances(observation.position, negative)
            if not self.vertices:
                break
        return self


__all__ = ["OmniCandidateRegion"]
