"""v4: reveal coverage obligations to the actor; never select/prune an action.

The old 60-feature rows, candidates, action execution and cost rewards are
unchanged. Known-channel checks can disappear after that source is cleared;
unknown-channel checks still require information. This is an observed-ledger
distinction, not a claim that the network's future plan has already succeeded.
"""

import math

from .controller import FEATURE_DIMS, ROUTE_FEATURE_NAMES
from .joint_scan import JointScanRLSearch


def route_debt_features(current, candidate_points, sites, pending, detected):
    """Sixteen finite features, computed once per distinct physical candidate.

    Unknown = channels not actually detected. Even after 16 discoveries, this
    representation does not mark unmeasured pairs completed or change masks.
    Detours describe geometry only; the actor chooses whether/where to continue.
    """
    if len(sites) != 7:
        raise ValueError("route-debt schema requires seven fixed cover sites")
    candidate_points = tuple(candidate_points)
    if not all(math.isfinite(value) for p in (current, *sites, *candidate_points) for value in (p.x, p.y)):
        raise ValueError("route-debt coordinates must be finite")
    unknown = set(range(1, 21)) - detected
    counts = [len(pending(site) & unknown) for site in sites]
    active = [site for site, count in zip(sites, counts) if count]
    prefix = [count / 20 for count in counts]
    current_distances = [current.distance_to(site) for site in sites]
    result = {}
    for point in candidate_points:
        if point in result:
            continue
        travel = current.distance_to(point)
        distances = [point.distance_to(site) for site in sites]
        detours = [max(0.0, travel + distance - direct) / 5000 if count else 0.0
                   for distance, direct, count in zip(distances, current_distances, counts)]
        nearest = min((point.distance_to(site) for site in active), default=0.0)
        row = prefix + detours + [nearest / 3600, len(active) / 7]
        if len(row) != len(ROUTE_FEATURE_NAMES) or not all(math.isfinite(value) for value in row):
            raise ValueError("route-debt features must be finite")
        result[point] = row
    return result


class RouteDebtRLSearch(JointScanRLSearch):
    def __init__(self, client, policy, *, feature_version="v4", **kwargs):
        if feature_version != "v4":
            raise ValueError("route-debt controller requires v4 feature semantics")
        super().__init__(client, policy, feature_version=feature_version, **kwargs)
        self.report.learning["representation_scope"] = "v3 plus observed unknown-channel coverage debt and geometric detours"

    def _features(self, candidates, remaining):
        rows, context = super()._features(candidates, remaining)
        extra = route_debt_features(self.client.state.position, [c.point for c in candidates],
                                   self.points, self._pending, self.detected)
        for row, candidate in zip(rows, candidates):
            row += extra[candidate.point]
            assert len(row) == FEATURE_DIMS["v4"]
        return rows, context
