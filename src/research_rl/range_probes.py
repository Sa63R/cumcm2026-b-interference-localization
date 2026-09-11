"""Four receivable crossing probes along the current-to-region approach.

Only public feasible geometry creates candidates. The unchanged neural actor
selects actions: no state-search score, posterior sampling or hidden truth.
All base candidates are preserved, including potentially informative negatives.
"""
import math

from simulator_client.state import Position
from .action_sets import action_schema
from .controller import Candidate
from .joint_scan import JointScanRLSearch

OPTION_ID = 7
OFFSETS_M = (-150.0, -50.0, 50.0, 150.0)
RECEPTION_MARGIN_M = 1e-7


def range_templates(region, current, first_bearing):
    """Midpoint anchors differ from position-independent long-axis quantiles."""
    if region is None or not region.vertices:
        return ()
    try:
        current = Position.coerce(current)
    except (TypeError, ValueError):
        return ()
    center = region.enclosing_disk().center
    numbers = [current.x, current.y, first_bearing, *center]
    if not all(math.isfinite(v) for v in numbers):
        return ()
    if not all(math.isfinite(v) for p in region.vertices for v in p):
        return ()
    angle = math.radians(first_bearing)
    px, py = -math.sin(angle), math.cos(angle)
    ax, ay = (current.x+center[0])/2, (current.y+center[1])/2
    points = (Position(ax+offset*px, ay+offset*py) for offset in OFFSETS_M)
    # For a convex outer polygon, the maximum distance to any possible source
    # is attained at a vertex. 1000 m is the public minimum reception radius.
    return tuple(p for p in points
                 if max(math.hypot(x-p.x,y-p.y) for x,y in region.vertices)
                 <= 1000.0-RECEPTION_MARGIN_M)


class RangeProbeRLSearch(JointScanRLSearch):
    def __init__(self, client, policy, **kwargs):
        if hasattr(policy, "action_schema") and policy.action_schema != action_schema("range_probes"):
            raise ValueError("range controller requires a matching policy action schema")
        super().__init__(client, policy, **kwargs)
        self._range_cache = {}
        self.report.learning.update(action_schema=action_schema("range_probes"),
            range_candidate_builds=0, range_candidates_offered_sum=0,
            range_probe_measurements=0, base_probe_measurements=0)
        self.report.learning["learned_scope"].append("midpoint_crossing_probe_point")

    def _source_candidates(self, channel):
        original = super()._source_candidates(channel)
        if not any(c.kind == "probe" for c in original):
            return original
        region = self.regions[channel]
        current = self.client.state.position
        bearing = self.first_bearings[channel]
        observed = frozenset(self.observed_positions.get(channel, ()))
        signature = (current, bearing, observed)
        cached = self._range_cache.get(channel)
        if cached is None or cached[0] is not original or cached[1] is not region.vertices or cached[2] != signature:
            seen = set(observed) | {(round(c.point.x,6),round(c.point.y,6)) for c in original}
            additional = []
            for point in range_templates(region, current, bearing):
                key = (round(point.x,6),round(point.y,6))
                if key not in seen:
                    seen.add(key)
                    additional.append(Candidate("probe",channel,point,option=OPTION_ID,radius=original[0].radius))
            cached = (original, region.vertices, signature, original+tuple(additional))
            self._range_cache[channel] = cached
            self.report.learning["range_candidate_builds"] += 1
        return cached[3]

    def _candidates(self, remaining):
        choices = super()._candidates(remaining)
        self.report.learning["range_candidates_offered_sum"] += sum(c.option == OPTION_ID for c in choices)
        return choices

    def _execute_candidate(self, candidate, remaining):
        before = len(self.report.action_history)
        try:
            return super()._execute_candidate(candidate, remaining)
        finally:
            if candidate.kind == "probe":
                accepted = [a for a in self.report.action_history[before:] if a["action"] == "measure"]
                new = candidate.option == OPTION_ID
                key = "range_probe_measurements" if new else "base_probe_measurements"
                self.report.learning[key] += len(accepted)
                if new:
                    for action in accepted:
                        action["rl_probe_family"] = "range_probes"
