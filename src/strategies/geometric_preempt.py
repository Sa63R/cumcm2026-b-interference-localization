"""Q3 single-probe geometry with bounded interruption of source localization.

Only real observation stops may release a source task. Frozen-target route
improvement is a scheduling proxy, not an expected-time improvement theorem.
"""
from collections import Counter
from dataclasses import asdict, field, dataclass

from planning import clearance_grid, improve_open_route, nearest_order
from simulator_client.state import Position

from .geometric_probe_cost import GeometricProbeCostSearch, ProbeCostResult


def route_length(route, start):
    points = [Position.coerce(start), *route]
    return sum(a.distance_to(b) for a, b in zip(points, points[1:]))


def frozen_route_interruption(current, primary, tasks):
    """Compare the SAME complete task multiset with/without a locked first.

    First optimize the suffix while primary remains fixed. Then let 2-opt
    reverse through the first task, retaining the complete forced route as
    incumbent. A pure suffix improvement cannot be mistaken for interruption.
    Return no proposal if the final first physical destination stays primary.
    """
    others = [task for task in tasks if task is not primary]
    if not others:
        return None
    suffix = improve_open_route(nearest_order([t[2] for t in others], start=primary[2]),
                                start=primary[2])
    forced = (primary[2], *suffix)
    free = improve_open_route(forced, start=current)
    assert Counter((p.x, p.y) for p in forced) == Counter((p.x, p.y) for p in free)
    if free[0] == primary[2]:
        return None
    task = next(t for t in tasks if t[2] == free[0])
    before, after = route_length(forced, current), route_length(free, current)
    # This is only a local two-retune allowance. The true entire trace always
    # pays its actual switching, measurement, and movement costs separately.
    score = (before-after)/5.-2.
    if score < 10.:
        return None
    return task, dict(forced_route=[[p.x, p.y] for p in forced],
                      interrupted_route=[[p.x, p.y] for p in free],
                      forced_length_m=before, interrupted_length_m=after,
                      frozen_route_gain_s=(before-after)/5., local_retune_allowance_s=2.,
                      score_s=score, task_count=len(tasks))


class _InterruptResolution(Exception):
    def __init__(self, task):
        self.task = task


@dataclass
class PreemptResult(ProbeCostResult):
    source_interruptions: dict = field(default_factory=dict)


class GeometricPreemptSearch(GeometricProbeCostSearch):
    def __init__(self, client, max_actions, max_active_probes, joint_config,
                 max_planning_s, enabled):
        if type(enabled) is not bool:
            raise ValueError('preempt must be boolean')
        super().__init__(client, max_actions, max_active_probes, joint_config,
                         'single', max_planning_s)
        self.preempt_enabled = enabled
        self.variant = 'geometric_probe_preempt'
        self.report = PreemptResult(**asdict(self.report))
        self.report.variant = self.variant
        self.report.strategy_parameters.update(preempt=enabled,
            interruption_margin_s=10., local_retune_allowance_s=2., maximum_interruptions_per_source=1)
        self.report.source_interruptions = dict(events=[], probe_counts={}, checks=0)
        self._primary_probe_counts = Counter()
        self._interrupted_channels = set()
        self._attempted_centers = {}
        self._remaining = []

    def _interruption_after_probe(self, channel):
        # This is called only immediately after one accepted primary measure
        # and all its existing same-point shared observations have returned.
        if (channel in self._interrupted_channels or channel in self.near_points
                or self._primary_probe_counts[channel] >= self.max_active_probes):
            return
        region = self.regions.get(channel)
        if not region or not region.vertices or region.enclosing_disk().radius <= 19.9:
            return
        tasks = [('source', c, p) for c in sorted(self.detected-self.cleared-self.blocked)
                 if (p := self._target(c)) is not None]
        tasks += [('cover', None, p) for p in self._remaining]
        primary = next((t for t in tasks if t[:2] == ('source', channel)), None)
        if primary is None:
            return
        self.report.source_interruptions['checks'] += 1
        proposal = frozen_route_interruption(self.client.state.position, primary, tasks)
        if proposal is None:
            return
        task, details = proposal
        self._interrupted_channels.add(channel)
        self.report.source_interruptions['events'].append(dict(channel=channel,
            primary_probes=self._primary_probe_counts[channel], accepted_actions=self.actions,
            position=[self.client.state.position.x, self.client.state.position.y],
            next_task=dict(kind=task[0], channel=task[1], position=[task[2].x, task[2].y]), **details))
        raise _InterruptResolution(task)

    def _resolve(self, channel):
        if not self.preempt_enabled:
            return super()._resolve(channel)
        if channel in self.cleared:
            return True
        attempted = self._attempted_centers.setdefault(channel, set())
        while True:
            if channel in self.near_points:
                return self._clear(self.near_points[channel], channel, 'near_clear')
            region = self.regions.get(channel)
            if region is None or not region.vertices:
                return False
            circle = region.enclosing_disk()
            if circle.radius <= 19.9:
                center = Position.coerce(circle.center)
                key = (round(center.x, 6), round(center.y, 6))
                if key not in attempted:
                    attempted.add(key)
                    if self._clear(center, channel, 'certified_clear'):
                        return True
            count = self._primary_probe_counts[channel]
            if count >= self.max_active_probes:
                break
            point = self._next_probe(channel, count)
            if point is None:
                break
            self._perform('measure', point, channel, 'active_localization')
            self._primary_probe_counts[channel] += 1
            self.report.source_interruptions['probe_counts'][str(channel)] = self._primary_probe_counts[channel]
            self._interruption_after_probe(channel)
        region = self.regions.get(channel)
        if region is None or not region.vertices:
            return False
        for point in clearance_grid(region.vertices, bearing_deg=self.first_bearings[channel],
                                    start=self.client.state.position):
            if self._clear(point, channel, 'guaranteed_clearance'):
                return True
        return False

    def _execute_plan(self):
        if not self.preempt_enabled:
            return super()._execute_plan()
        self._scan(self.points[0])
        self._remaining = list(self.points[1:])
        pending = None
        while True:
            if not self._remaining:
                self.report.coverage_complete = True
            task = pending or self._next_task(self._remaining)
            pending = None
            if task is None:
                break
            kind, channel, point = task
            if kind == 'cover':
                self._scan(point)
                self._remaining.remove(point)
            else:
                try:
                    if not self._resolve(channel):
                        self.blocked.add(channel)
                except _InterruptResolution as interrupted:
                    # No state/observations change before this selected task.
                    # It must execute a real action before any new interruption.
                    pending = interrupted.task


def run_preempt_search(client, *, problem=3, max_actions=20000, max_active_probes=6,
                       joint_config=None, max_planning_s=120., preempt=True):
    if problem not in (3, 'q3'):
        raise ValueError('Geometric source interruption is Q3-only')
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError('max_actions must be an integer >=2')
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError('max_active_probes must be an integer in 0..30')
    return GeometricPreemptSearch(client, max_actions, max_active_probes, joint_config,
                                  max_planning_s, preempt).run()
