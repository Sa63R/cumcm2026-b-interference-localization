"""Unexecuted Q3 prototype, archived after previous failure evidence was found.

Every physical update and safety certificate is inherited from the frozen
derived-silence policy. No hypothetical observation updates live state.
"""
from planning import clearance_grid
from simulator_client.state import Position

from .derived_silence_state_search import DerivedSilenceStateSearch


class FeedbackReplanSearch(DerivedSilenceStateSearch):
    def __init__(self, client, max_actions, max_active_probes, config, *, enabled=True,
                 relative_silence=True, stop_scan_at_16=True, replan=True):
        if type(replan) is not bool:
            raise ValueError("replan must be boolean")
        super().__init__(client, max_actions, max_active_probes, config,
                         enabled=enabled, relative_silence=relative_silence,
                         stop_scan_at_16=stop_scan_at_16)
        self.feedback_replan = replan
        self.source_probe_counts = {}
        self.feedback_events = []
        self.report.strategy_parameters['feedback_replan'] = {
            'enabled': replan, 'events': self.feedback_events,
            'source_probe_budget': max_active_probes,
            'scope': 'real observations only; unchanged clear and discovery certificates',
        }

    def _try_certified(self, channel):
        if channel in self.cleared:
            return True
        if channel in self.near_points:
            return self._clear(self.near_points[channel], channel, 'near_clear')
        region = self.regions.get(channel)
        if region and region.vertices:
            disk = region.enclosing_disk()
            if disk.radius <= 19.9:
                return self._clear(Position(*disk.center), channel, 'certified_clear')
        return None

    def _source_step(self, channel):
        """Return progress separately from actual clear success."""
        result = self._try_certified(channel)
        if result is not None:
            return result
        region = self.regions.get(channel)
        if not region or not region.vertices:
            return False
        used = self.source_probe_counts.get(channel, 0)
        if used < self.max_active_probes:
            point = self._next_probe(channel, used)
            if point is not None:
                previous_center = list(region.enclosing_disk().center)
                self._perform('measure', point, channel, 'active_localization')
                self.source_probe_counts[channel] = used + 1
                result = self._try_certified(channel)
                self.feedback_events.append({
                    'channel': channel, 'after_actual_action_count': len(self.report.action_history),
                    'previous_center': previous_center,
                    'updated_center': list(region.enclosing_disk().center) if region.vertices else None,
                    'source_cleared': channel in self.cleared,
                    'used_probes': used + 1,
                })
                return result is not False
        # The budget belongs to the source for the whole episode, so repeatedly
        # selecting a source cannot restart a probing loop indefinitely.
        region = self.regions.get(channel)
        if not region or not region.vertices:
            return False
        for point in clearance_grid(region.vertices, bearing_deg=self.first_bearings[channel],
                                    start=self.client.state.position):
            if self._clear(point, channel, 'guaranteed_clearance'):
                return True
        return False

    def _execute_plan(self):
        if not self.feedback_replan:
            return super()._execute_plan()
        self._scan(self.points[0])
        remaining = list(self.points[1:])
        while True:
            if not remaining:
                self.report.coverage_complete = True
            task = self._next_task(remaining)
            if task is None:
                break
            kind, channel, point = task
            if kind == 'cover':
                self._scan(point)
                remaining.remove(point)
            else:
                if not self._source_step(channel):
                    self.blocked.add(channel)
                elif channel in self.cleared and self.state_config.opportunistic_measurements:
                    self._share_observation()


def run_feedback_replan(client, *, problem=3, max_actions=10000, max_active_probes=6,
                        config=None, enabled=True, relative_silence=True,
                        stop_scan_at_16=True, replan=True):
    if problem != 3:
        raise ValueError('Feedback replanning supports Q3 only')
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError('max_actions must be integer >=2')
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError('max_active_probes must be integer in [0,30]')
    return FeedbackReplanSearch(client, max_actions, max_active_probes, config,
        enabled=enabled, relative_silence=relative_silence,
        stop_scan_at_16=stop_scan_at_16, replan=replan).run()
