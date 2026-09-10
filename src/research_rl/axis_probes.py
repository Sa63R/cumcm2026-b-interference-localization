"""Observation-only axis-quantile actions; the neural actor chooses all actions.

No tree search, radius score, synthetic posterior, or simulator truth is used.
The v3 action set is retained; only additional certified-receivable probes are
offered. Legal clear/fallback/coverage and decision limits remain inherited.
"""

import math
import time

from simulator_client.state import Position
from .action_sets import action_schema
from .controller import Candidate
from .joint_scan import JointScanRLSearch


def axis_templates(region):
    """Four support quantiles x three lateral offsets, plus two center offsets."""
    vertices = region.vertices
    a, b = max(((a, b) for a in vertices for b in vertices),
               key=lambda pair: (pair[0][0]-pair[1][0])**2 + (pair[0][1]-pair[1][1])**2)
    length = math.dist(a, b)
    ux, uy = ((b[0]-a[0])/length, (b[1]-a[1])/length) if length else (1.0, 0.0)
    cx, cy = region.enclosing_disk().center
    along = [(x-cx)*ux + (y-cy)*uy for x,y in vertices]
    low, high = min(along), max(along)
    scale = min(200.0, max(20.0, (high-low)/8))
    offsets = [(low+q*(high-low), across) for q in (.25, .375, .625, .75)
               for across in (-scale, 0.0, scale)] + [(0.0, -scale), (0.0, scale)]
    proposals = [Position(cx+distance*ux-across*uy, cy+distance*uy+across*ux)
                 for distance,across in offsets]
    return tuple(p for p in proposals if max(math.hypot(x-p.x,y-p.y) for x,y in vertices) <= 1000.0-1e-7)


class AxisProbeRLSearch(JointScanRLSearch):
    def __init__(self, client, policy, **kwargs):
        if hasattr(policy, "action_schema") and policy.action_schema != action_schema("axis_quantiles"):
            raise ValueError("axis controller requires a matching policy action schema")
        super().__init__(client, policy, **kwargs)
        self._axis_template_cache = {}
        self._axis_extension_cache = {}
        self.report.learning.update(action_schema=action_schema("axis_quantiles"),
            axis_template_builds=0, axis_template_wall_s=0.0, axis_extension_builds=0,
            axis_candidates_offered_sum=0, axis_probe_measurements=0,
            base_probe_measurements=0)
        self.report.learning["learned_scope"].append("axis_quantile_probe_point")

    def _source_candidates(self, channel):
        original = super()._source_candidates(channel)
        if not any(c.kind == "probe" for c in original):
            return original
        region = self.regions[channel]
        cached = self._axis_template_cache.get(channel)
        if cached is None or cached[0] is not region.vertices:
            started = time.perf_counter()
            cached = (region.vertices, axis_templates(region))
            self._axis_template_cache[channel] = cached
            self.report.learning["axis_template_builds"] += 1
            self.report.learning["axis_template_wall_s"] += time.perf_counter()-started
        observed = frozenset(self.observed_positions.get(channel, ()))
        extension = self._axis_extension_cache.get(channel)
        if extension is None or extension[0] is not original or extension[1] != observed or extension[2] is not region.vertices:
            seen = set(observed) | {(round(c.point.x,6),round(c.point.y,6)) for c in original}
            additional = []
            for point in cached[1]:
                key = (round(point.x,6),round(point.y,6))
                if key not in seen:
                    seen.add(key)
                    # One new family ID, rather than arbitrary 6..19 ordinals.
                    # Feature18 remains option/6; all old candidate rows stay intact.
                    additional.append(Candidate("probe", channel, point, option=6, radius=original[0].radius))
            extension = (original, observed, region.vertices, original+tuple(additional))
            self._axis_extension_cache[channel] = extension
            self.report.learning["axis_extension_builds"] += 1
        return extension[3]

    def _candidates(self, remaining):
        candidates = super()._candidates(remaining)
        self.report.learning["axis_candidates_offered_sum"] += sum(c.kind == "probe" and c.option == 6 for c in candidates)
        return candidates

    def _execute_candidate(self, candidate, remaining):
        before = len(self.report.action_history)
        try:
            return super()._execute_candidate(candidate, remaining)
        finally:
            if candidate.kind == "probe":
                family = "axis_quantiles" if candidate.option == 6 else "base"
                accepted = [a for a in self.report.action_history[before:] if a["action"] == "measure"]
                self.report.learning["axis_probe_measurements" if family == "axis_quantiles" else "base_probe_measurements"] += len(accepted)
                # This annotates actual physical actions, not selected labels or
                # the interrupted_scans state counter inherited from v3.
                for action in accepted:
                    if family == "axis_quantiles": action["rl_probe_family"] = family
