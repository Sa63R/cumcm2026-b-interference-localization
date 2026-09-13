"""Isolated Q3 state-search refinement; the frozen v1 entry point is unchanged."""

from planning.clear_region import nearest_clear_point, nearest_near_clear_point
from simulator_client.state import Position
from .relocating_state_search import RelocatingStateSearch


class ClearRegionStateSearch(RelocatingStateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, enabled,
                 clear_mode, history_silence):
        super().__init__(client, max_actions, max_active_probes, config, enabled)
        self.clear_mode = clear_mode
        self.history_silence = history_silence
        self.clear_region_log = []
        self.report.strategy_parameters.update(
            clear_region_mode=clear_mode, history_silence=history_silence,
            clear_region_log=self.clear_region_log,
            clear_region_scope="Nearest certified point for this clearance only; future route and measurements may change")

    def _clear(self, position, channel, phase):
        if self.clear_mode == "disabled" or phase not in {"certified_clear", "near_clear"}:
            return super()._clear(position, channel, phase)
        current = self.client.state.position
        if phase == "near_clear":
            old = self.near_points[channel]
            chosen = nearest_near_clear_point(old, current)
            worst = chosen.distance_to(old) + 5 if chosen is not None else None
        else:
            region = self.regions[channel]
            disk = region.enclosing_disk()
            center = Position.coerce(disk.center)
            distance = center.distance_to(current)
            fraction = min(1.0, max(0.0, 19.9 - disk.radius) / distance) if distance else 0.0
            old = Position(center.x + fraction * (current.x - center.x),
                           center.y + fraction * (current.y - center.y))
            chosen = nearest_clear_point(region.vertices, current, fallback=old)
            worst = (max(chosen.distance_to(Position.coerce(v)) for v in region.vertices)
                     if chosen is not None else None)
        if chosen is None:
            return super()._clear(position, channel, phase)
        # Do not pass through EfficientSearch._clear, which would overwrite
        # the selected point with the older enclosing-circle construction.
        self.clear_region_log.append({
            "channel": channel, "phase": phase,
            "after_actual_action_count": len(self.report.action_history),
            "from": [current.x, current.y], "old_point": [old.x, old.y],
            "new_point": [chosen.x, chosen.y], "worst_distance_m": worst,
            "local_movement_saved_s": (current.distance_to(old) - current.distance_to(chosen)) / 5,
        })
        return self._perform("clear", chosen, channel, phase)["clear_result"] == "success"

    def _scan(self, point):
        if not self.history_silence:
            return super()._scan(point)
        # This matches the inherited scan except for its additional, explicitly
        # recorded logical certificate. Unknown channels always receive a real
        # measurement, so discovery coverage is unchanged.
        from planning.history_silence import certify_history_silence
        channels = [c for c in range(1, 21) if c not in self.cleared]
        current = self.client.state.current_channel
        if current in channels:
            channels.remove(current)
            channels.insert(0, current)
        for channel in channels:
            region = self.regions.get(channel)
            certified = (channel in self.near_points or (
                channel in self.detected and region and region.vertices
                and region.enclosing_disk().radius <= 19.9))
            if self.state_config.skip_certified_scans and certified:
                self.skipped_measurements += 1
                continue
            certificate = (certify_history_silence(region, point)
                           if channel in self.detected and region else None)
            if certificate:
                region.observe_no_signal(point)
                self.inferred.append({
                    "inference_kind": "inferred_no_signal", "channel": channel,
                    "position": [point.x, point.y],
                    "after_actual_action_count": len(self.report.action_history),
                    "physical_measurement": False,
                    "virtual_time_s": self.client.state.virtual_time_s,
                    **certificate,
                })
                continue
            self._perform("measure", point, channel, "coverage")
        self.report.coverage_points_visited += 1
        self.discovery_stations.append(point)
        self.blocked.clear()
        self.report.strategy_parameters["skipped_certified_measurements"] = self.skipped_measurements


def run_clear_region_state_search(client, *, problem=3, max_actions=10000,
                                  max_active_probes=6, config=None, enabled=True,
                                  clear_mode="nearest", history_silence=False):
    if problem != 3:
        raise ValueError("Clear-region state optimization supports Q3 only")
    if type(enabled) is not bool or type(history_silence) is not bool:
        raise ValueError("enabled and history_silence must be boolean")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be integer >=2")
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError("max_active_probes must be in [0,30]")
    if clear_mode not in {"disabled", "nearest"}:
        raise ValueError("clear_mode must be disabled or nearest")
    return ClearRegionStateSearch(client, max_actions, max_active_probes, config,
                                  enabled, clear_mode, history_silence).run()
