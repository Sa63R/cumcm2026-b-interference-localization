"""One real optical attempt at an already selected Q4 centre probe.

This is an action-type experiment, not a location certificate or a calibrated
success probability. All discovery, source scheduling and probe points remain
inherited from compact_combo.
"""
import math
import time

from simulator_client.state import Position
from .q4_range_scheduling import Q4RangeScheduling


class _ProbeSourceCleared(Exception):
    def __init__(self, channel):
        self.channel = channel


class Q4ClearBeforeProbe(Q4RangeScheduling):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions,
                 config="center_once"):
        if config != "center_once":
            raise ValueError("Unknown clear-before-probe configuration")
        super().__init__(client, max_actions, max_active_probes,
                         max_expansions=max_expansions, config="onroute")
        self.speculative_attempted = set()
        self.clear_before_probe_log = []
        self._probe_resolving_channel = None
        self.report.strategy_parameters.update(
            clear_before_probe_config=config,
            clear_before_probe_log=self.clear_before_probe_log,
            clear_before_probe_scope="One accepted real clear per live source, only at an inherited active centre probe with 19.9<radius<=40; failure does not change C",
        )

    def _resolve(self, channel):
        previous = self._probe_resolving_channel
        self._probe_resolving_channel = channel
        try:
            try:
                return super()._resolve(channel)
            except _ProbeSourceCleared as stopped:
                # The cooperative parent's finally has already run. No fake
                # measurement result enters its localization loop.
                if stopped.channel != channel or channel not in self.cleared:
                    raise
                return True
        finally:
            self._probe_resolving_channel = previous

    def _insertion_budget(self, point, channel):
        state = self.client.state
        movement = math.ceil(state.position.distance_to(point) / 5. * 1e6) / 1e6
        switch = int(channel != state.current_channel)
        # Failure: travel once, optical 3, then the original measure 5+switch.
        # This also dominates a successful clear's travel+5.
        cost = movement + 8. + switch
        maximum = state.max_virtual_duration_s
        maximum = min(360000., maximum) if maximum is not None else 360000.
        real = getattr(self.client, "remaining_real_time_s", None)
        details = dict(max_actions=self.max_actions, policy_action_count=self.actions,
            remaining_actions=self.max_actions-self.actions,
            movement_ceiling_s=movement, worst_failure_cost_s=cost,
            service_deadline=self.service_deadline, virtual_limit_s=maximum,
            remaining_real_s=real)
        reasons = []
        if self.actions + 3 > self.max_actions:
            reasons.append("action_budget")
        if state.virtual_time_s + cost > maximum - 1e-6:
            reasons.append("virtual_budget")
        if self.service_deadline is not None and state.virtual_time_s + cost > self.service_deadline:
            reasons.append("service_budget")
        if real is not None and real <= 2.:
            reasons.append("real_deadline")
        details["skip_reasons"] = reasons
        return not reasons, details

    def _perform(self, action, position, channel, phase):
        if (action != "measure" or phase != "active_localization"
                or self._probe_resolving_channel != channel
                or channel not in self.detected or channel in self.cleared
                or channel in self.near_points or channel in self.speculative_attempted):
            return super()._perform(action, position, channel, phase)
        began = time.perf_counter()
        region = self.regions.get(channel)
        if region is None or not region.vertices or not region.observations:
            return super()._perform(action, position, channel, phase)
        point = Position.coerce(position)
        disk = region.enclosing_disk()
        center = Position.coerce(disk.center)
        if not math.isfinite(disk.radius) or not 19.9 < disk.radius <= 40. or point != center:
            return super()._perform(action, position, channel, phase)

        prefix = len(self.report.action_history)
        allowed, budget = self._insertion_budget(point, channel)
        event = dict(channel=channel, after_actual_action_count=prefix,
            end_actual_action_count=prefix,
            current_position=[self.client.state.position.x, self.client.state.position.y],
            position=[point.x, point.y], center=[center.x, center.y],
            radius_m=disk.radius, vertices=[list(v) for v in region.vertices],
            positive_observation_count=len(region.observations),
            phase=phase, speculative_phase="speculative_clear_before_probe",
            budget=budget, executed=False, clear_result=None,
            status="pending", decision_wall_s=time.perf_counter()-began, runtime_s=0.)
        self.clear_before_probe_log.append(event)
        if not allowed:
            event.update(status="skipped_budget", runtime_s=time.perf_counter()-began)
            return super()._perform(action, position, channel, phase)
        try:
            response = super()._perform("clear", point, channel, "speculative_clear_before_probe")
            self.speculative_attempted.add(channel)
            event.update(executed=True, clear_result=response["clear_result"])
            if response["clear_result"] == "success":
                event["status"] = "cleared"
                raise _ProbeSourceCleared(channel)
            # A real miss is not converted to a bearing, freshness credit, or a
            # convex exclusion. Only this original physical measurement updates C.
            response = super()._perform(action, position, channel, phase)
            event["status"] = "failed_then_measured"
            return response
        except _ProbeSourceCleared:
            raise
        except Exception as error:
            event["interruption_type"] = type(error).__name__
            event["interruption_reason"] = str(error)
            raise
        finally:
            event["end_actual_action_count"] = len(self.report.action_history)
            event["runtime_s"] = time.perf_counter()-began
            if event["status"] == "pending":
                event["status"] = "interrupted"


def run_q4_clear_before_probe(client, *, problem=4, config="center_once",
                              max_actions=20000, max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config != "center_once":
        raise ValueError("Q4 only; config must be center_once")
    for value, low, high in ((max_actions, 2, 1_000_000), (max_active_probes, 0, 30),
                             (max_expansions, 0, 10000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid integer execution budget")
    return Q4ClearBeforeProbe(client, max_actions, max_active_probes,
                             max_expansions=max_expansions, config=config).run()
