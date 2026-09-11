"""A bounded source interruption that commits to exactly one task then resumes.

All candidates compete with the SAME forced-primary complete frozen route.
No suffix reoptimization is allowed after the proposed two-task prefix.
"""
from collections import Counter

from planning import improve_open_route, nearest_order
from .geometric_preempt import GeometricPreemptSearch, route_length
from .search import _StopSearch


def frozen_two_task_swap(current, primary, tasks):
    others = [task for task in tasks if task is not primary]
    if not others:
        return None
    suffix = improve_open_route(nearest_order([t[2] for t in others],start=primary[2]),start=primary[2])
    pending = others.copy()
    forced = [primary]
    for point in suffix:
        task = next(t for t in pending if t[2] == point)
        forced.append(task)
        pending.remove(task)
    assert not pending
    baseline = route_length([t[2] for t in forced],current)
    best = None
    for other in forced[1:]:
        if other[2] == primary[2]:
            continue
        alternate = [other,primary]+[t for t in forced if t is not other and t is not primary]
        assert Counter(alternate) == Counter(forced)
        movement_gain = (baseline-route_length([t[2] for t in alternate],current))/5.
        # A cover before primary clearance retains primary in its real atomic
        # scan. Explicitly charge this extra measure; two retunes are a LOCAL
        # allowance, not a proved whole-continuation switching bound.
        measure_extra = 5. if other[0] == 'cover' else 0.
        net = movement_gain-measure_extra-2.
        if net <= 10. or (best is not None and net <= best[1]['score_s']):
            continue
        encode = lambda route:[[t[0],t[1],t[2].x,t[2].y] for t in route]
        best = (other,dict(forced_route=encode(forced),swapped_route=encode(alternate),
            forced_length_m=baseline,swapped_length_m=baseline-5*movement_gain,
            frozen_route_gain_s=movement_gain,extra_primary_cover_measure_s=measure_extra,
            local_retune_allowance_s=2.,score_s=net,task_count=len(tasks)))
    return best


class _LockedExchange(Exception):
    def __init__(self, other, primary_channel, event):
        self.other,self.primary_channel,self.event = other,primary_channel,event


class GeometricSwapSearch(GeometricPreemptSearch):
    def __init__(self, client, max_actions, max_active_probes, joint_config,max_planning_s,enabled):
        super().__init__(client,max_actions,max_active_probes,joint_config,max_planning_s,enabled)
        self.variant = self.report.variant = 'geometric_probe_swap'
        self._locked_execution = False
        self.report.strategy_parameters.update(
            interruption_kind='one_other_then_immediate_primary_resume',
            comparator='one_shared_forced_primary_route; no alternate suffix optimization',
            extra_primary_cover_measure_s=5.,nested_interruption=False)

    def _interruption_after_probe(self, channel):
        if (self._locked_execution or channel in self._interrupted_channels
                or channel in self.near_points or self._primary_probe_counts[channel] >= self.max_active_probes):
            return
        region = self.regions.get(channel)
        if not region or not region.vertices or region.enclosing_disk().radius <= 19.9:
            return
        tasks = [('source',c,p) for c in sorted(self.detected-self.cleared-self.blocked)
                 if (p := self._target(c)) is not None]
        tasks += [('cover',None,p) for p in self._remaining]
        primary = next((t for t in tasks if t[:2] == ('source',channel)),None)
        if primary is None:
            return
        self.report.source_interruptions['checks'] += 1
        proposal = frozen_two_task_swap(self.client.state.position,primary,tasks)
        if proposal is None:
            return
        other,details = proposal
        self._interrupted_channels.add(channel)
        event = dict(channel=channel,primary_probes=self._primary_probe_counts[channel],
            accepted_actions=self.actions,position=[self.client.state.position.x,self.client.state.position.y],
            next_task=dict(kind=other[0],channel=other[1],position=[other[2].x,other[2].y]),**details)
        self.report.source_interruptions['events'].append(event)
        raise _LockedExchange(other,channel,event)

    def _atomic_task(self, task):
        kind,channel,point = task
        if kind == 'cover':
            self._scan(point)
            self._remaining.remove(point)
        elif not self._resolve(channel):
            self.blocked.add(channel)

    def _execute_exchange(self, exchange):
        self._locked_execution = True
        event = exchange.event
        event['other_start_actions'] = self.actions
        try:
            self._atomic_task(exchange.other)
            event['other_end_actions'] = self.actions
            if event['other_end_actions'] <= event['other_start_actions']:
                raise _StopSearch('locked_exchange_task_unexecuted')
            event['resume_start_actions'] = self.actions
            event['resume_primary_probes_before'] = self._primary_probe_counts[exchange.primary_channel]
            # Resolve by channel and current observed region, never by a stale
            # frozen coordinate. No global _next_task runs between the two.
            self._atomic_task(('source',exchange.primary_channel,None))
            event['resume_end_actions'] = self.actions
            event['resume_primary_probes_after'] = self._primary_probe_counts[exchange.primary_channel]
            event['primary_cleared_after_resume'] = exchange.primary_channel in self.cleared
        finally:
            event['locked_final_actions'] = self.actions
            event['primary_cleared_final'] = exchange.primary_channel in self.cleared
            self._locked_execution = False

    def _execute_plan(self):
        if not self.preempt_enabled:
            return super()._execute_plan()
        self._scan(self.points[0])
        self._remaining = list(self.points[1:])
        while True:
            if not self._remaining:
                self.report.coverage_complete = True
            task = self._next_task(self._remaining)
            if task is None:
                return
            try:
                self._atomic_task(task)
            except _LockedExchange as exchange:
                self._execute_exchange(exchange)


def run_swap_search(client, *, problem=3,max_actions=20000,max_active_probes=6,
                    joint_config=None,max_planning_s=120.,enabled=True):
    if problem not in (3,'q3'):
        raise ValueError('Locked source exchange is Q3-only')
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError('max_actions must be an integer >=2')
    if type(max_active_probes) is not int or not 0 <= max_active_probes <= 30:
        raise ValueError('max_active_probes must be an integer in 0..30')
    return GeometricSwapSearch(client,max_actions,max_active_probes,joint_config,max_planning_s,enabled).run()
