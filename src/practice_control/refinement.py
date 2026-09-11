"""Bounded observation-only refinement for practice dataset collection.

The five-metre target is an aspiration, not a guaranteed localization accuracy.
Session creation and practice-only mode verification remain the caller's job.
"""

from __future__ import annotations

import math

from simulator_client.state import Position
from strategies.efficient import EfficientSearch
from strategies.search import _Search


REFINEMENT_CONFIG = {
    "version": 1,
    "target_radius_m": 5.0,
    "trigger_max_radius_m": 40.0,
    "max_measurements_per_channel": 2,
    "max_measurements_per_episode": 32,
    "transverse_offset_m": 15.0,
}


class _RefinementMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refinement = {
            **REFINEMENT_CONFIG,
            "extra_measurements": 0,
            "per_channel": {},
            "near_responses": 0,
            "no_signal_responses": 0,
            "radius_target_reached": 0,
        }
        self.report.strategy_parameters["dataset_refinement"] = self.refinement

    def _planned_clear_position(self, position, channel, phase):
        """Match EfficientSearch's edge clear once, before extra sensing."""
        position = Position.coerce(position)
        config = getattr(self, "config", None)
        if phase == "certified_clear" and config is not None and config.edge_clear:
            disk = self.regions[channel].enclosing_disk()
            center = Position.coerce(disk.center)
            current = self.client.state.position
            distance = center.distance_to(current)
            fraction = min(1.0, max(0.0, 19.9 - disk.radius) / distance) if distance else 0.0
            position = Position(center.x + fraction * (current.x - center.x),
                                center.y + fraction * (current.y - center.y))
        return position

    def _refinement_budget_available(self, channel, position):
        if (self.refinement["extra_measurements"] >= 32
                or self.refinement["per_channel"].get(str(channel), 0) >= 2
                or self.actions >= self.max_actions - 2):
            return False  # Reserve the pending clear and explicit exit.
        remaining = getattr(self.client, "remaining_real_time_s", None)
        if remaining is not None and remaining <= 3.0:
            return False
        maximum = self.client.state.max_virtual_duration_s
        maximum = min(360000.0, maximum) if maximum is not None else 360000.0
        # The extra measure, plus a conservative allowance for the nearby clear.
        cost = self.client.state.position.distance_to(position) / 5.0 + 6.0 + 20.0
        return self.client.state.virtual_time_s + cost < maximum - 1e-6

    def _clear(self, position, channel, phase):
        region = self.regions.get(channel)
        if (channel in self.near_points or channel in self.cleared
                or region is None or not region.vertices):
            return super()._clear(position, channel, phase)
        disk = region.enclosing_disk()
        planned = self._planned_clear_position(position, channel, phase)
        if (not 5.0 < disk.radius <= 40.0
                or planned.distance_to(Position.coerce(disk.center)) > 40.0
                or not self._refinement_budget_available(channel, planned)):
            return super()._clear(position, channel, phase)

        probe = planned
        for index in range(2):
            if not self._refinement_budget_available(channel, probe):
                break
            response = self._perform("measure", probe, channel,
                                     "dataset_refine_at_clear" if index == 0
                                     else "dataset_refine_transverse")
            self.refinement["extra_measurements"] += 1
            key = str(channel)
            self.refinement["per_channel"][key] = self.refinement["per_channel"].get(key, 0) + 1
            if response["measure_result"] == "near":
                self.refinement["near_responses"] += 1
                break
            if response["measure_result"] == "no_signal":
                # Directional silence does not establish a distance constraint.
                self.refinement["no_signal_responses"] += 1
                break
            region = self.regions[channel]
            if not region.vertices:
                break
            if region.enclosing_disk().radius <= 5.0:
                self.refinement["radius_target_reached"] += 1
                break
            # Use the new local bearing at the planned clear point, since an
            # earlier distant bearing need not be transverse to this local ray.
            angle = math.radians(response["svd_deg"])
            dx, dy = -15.0 * math.sin(angle), 15.0 * math.cos(angle)
            options = (Position(planned.x + dx, planned.y + dy),
                       Position(planned.x - dx, planned.y - dy))
            probe = min(options, key=lambda p: (math.hypot(p.x, p.y), p.x, p.y))

        # These custom phases prevent EfficientSearch from shifting the already
        # planned clear a second time. All actions still use its _perform path.
        if channel in self.near_points:
            return super()._clear(self.near_points[channel], channel, "dataset_refined_near_clear")
        region = self.regions.get(channel)
        if region is not None and region.vertices:
            disk = region.enclosing_disk()
            if disk.radius <= 19.9:
                planned = Position.coerce(disk.center)
        return super()._clear(planned, channel, "dataset_refined_clear")


class _RefinedEfficientSearch(_RefinementMixin, EfficientSearch):
    pass


class _RefinedTriangularSearch(_RefinementMixin, _Search):
    pass


def run_collection_search(client, *, problem, variant, max_actions=20000):
    """Run Q3 efficient or Q4 triangular with at most 32 extra measurements."""
    if (isinstance(problem, bool) or not isinstance(problem, int)
            or not isinstance(variant, str)
            or (problem, variant) not in {(3, "efficient"), (4, "triangular")}):
        raise ValueError("collection refinement requires Q3 efficient or Q4 triangular")
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 2:
        raise ValueError("max_actions must be an integer >= 2")
    if problem == 3:
        search = _RefinedEfficientSearch(client, max_actions, 6, None)
    else:
        search = _RefinedTriangularSearch(client, 4, "triangular", max_actions, 6, "center")
    return search.run()
