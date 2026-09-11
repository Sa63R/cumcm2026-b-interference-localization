"""R12 source service charged against the already selected cover transit.

Only the scan-entry hook changes. The extra-cost certificate ends at the first
real measurement at the selected cover station; the remaining scan is still
the parent's complete scan. It is not a bound of 60 seconds on service alone.
"""
import math
import time

from simulator_client.state import Position
from .q4_joint_continuation import Q4JointContinuation


CONFIG = "incremental_60"
MICRO = 1_000_000


class _TransitSliceExpired(Exception):
    def __init__(self, gate_id, reasons):
        self.gate_id = gate_id
        self.reasons = list(reasons)
        super().__init__(",".join(reasons))


def _xy(p):
    p = Position.coerce(p)
    return [p.x, p.y]


def _clock_us(value):
    # The accepted virtual clock is reported at microsecond precision.
    return int(round(value * MICRO))


def _movement_up(a, b):
    return math.ceil(a.distance_to(b) / 5. * MICRO)


class Q4TransitBudget(Q4JointContinuation):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions,
                 config=CONFIG):
        if config != CONFIG:
            raise ValueError("Unknown transit-budget configuration")
        super().__init__(client, max_actions, max_active_probes, max_expansions=max_expansions)
        self.transit_attempted = set()
        self.transit_service_log = []
        self._transit_context = None
        self.report.strategy_parameters.update(
            transit_budget_config=config, transit_service_log=self.transit_service_log,
            transit_budget_limits=dict(macros=4, per_source=1, radius_min_exclusive_m=19.9,
                radius_max_m=40., projection_min=.1, projection_max=.9, detour_m=100.,
                old_first_cost_min_exclusive_s=60., incremental_limit_us=60*MICRO,
                return_first_measure_upper_us=6*MICRO, full_scan_reserve_us=120*MICRO,
                full_scan_reserve_actions=20, exit_reserve_actions=1),
            transit_budget_scope="Unchanged R12 resolver; service plus return to selected cover first actual measure minus original transit floor is <=60s. Not a whole-scan, service-only or whole-session bound.",
        )

    def _transit_candidates(self, destination):
        current = self.client.state.position
        dx, dy = destination.x-current.x, destination.y-current.y
        norm2 = dx*dx+dy*dy
        if norm2 == 0:
            return []
        distance = current.distance_to(destination)
        choices = []
        for channel in sorted(self.detected-self.cleared-self.blocked-self.early_attempted-self.transit_attempted):
            if channel in self.near_points or self._ready(channel):
                continue
            region = self.regions.get(channel)
            if region is None or not region.vertices or not region.observations:
                continue
            disk = region.enclosing_disk()
            center = Position.coerce(disk.center)
            if not math.isfinite(disk.radius) or not 19.9 < disk.radius <= 40.:
                continue
            approach = current.distance_to(center)
            fraction = ((center.x-current.x)*dx+(center.y-current.y)*dy)/norm2
            detour = approach+center.distance_to(destination)-distance
            first_cost = approach/5.+6.
            if .1 <= fraction <= .9 and detour <= 100. and first_cost > 60.:
                choices.append(dict(channel=channel, center=_xy(center), radius_m=disk.radius,
                    projection_fraction=fraction, detour_m=detour, approach_m=approach,
                    original_first_cost_s=first_cost))
        return sorted(choices, key=lambda x:(x["detour_m"],x["approach_m"],x["channel"]))

    def _transit_gate(self, action, point, channel, *, kind):
        ctx = self._transit_context
        state = self.client.state
        point = Position.coerce(point)
        start_us = ctx["start_virtual_us"]
        current_us = _clock_us(state.virtual_time_s)
        movement = _movement_up(state.position, point)
        switch = int(channel != state.current_channel)
        count = 2 if kind == "atomic_r8" else 1
        fees = (8+switch)*MICRO if kind == "atomic_r8" else (5+(switch if action == "measure" else 0))*MICRO
        action_upper = movement+fees
        return_upper = _movement_up(point, ctx["destination"])
        spent = current_us-start_us
        incremental = spent+action_upper+return_upper+6*MICRO-ctx["baseline_floor_us"]
        maximum = state.max_virtual_duration_s
        maximum = min(360000.,maximum) if maximum is not None else 360000.
        # Strictly preserve the parent's one-microsecond virtual-limit margin.
        virtual_limit_us = math.floor(maximum*MICRO)-1
        reasons = []
        if incremental > 60*MICRO:
            reasons.append("incremental_budget")
        if self.actions+count+20+1 > self.max_actions:
            reasons.append("return_scan_action_reserve")
        if current_us+action_upper+return_upper+120*MICRO > virtual_limit_us:
            reasons.append("return_scan_virtual_reserve")
        event = dict(id=len(ctx["event"]["gates"]), kind=kind,
            after_actual_action_count=len(self.report.action_history), action=action,
            point=_xy(point), channel=channel, current_position=_xy(state.position),
            current_channel=state.current_channel, current_virtual_us=current_us,
            spent_us=spent, movement_upper_us=movement, action_fee_upper_us=fees,
            action_upper_us=action_upper, predicted_actions=count,
            return_upper_us=return_upper, baseline_floor_us=ctx["baseline_floor_us"],
            incremental_upper_us=incremental, incremental_limit_us=60*MICRO,
            policy_action_count=self.actions, max_actions=self.max_actions,
            full_scan_reserve_actions=20, exit_reserve_actions=1,
            full_scan_reserve_us=120*MICRO, return_first_measure_upper_us=6*MICRO,
            virtual_limit_us=virtual_limit_us,
            remaining_real_s=getattr(self.client,"remaining_real_time_s",None),
            admitted=not reasons, reasons=reasons, executed_action_count=0)
        ctx["event"]["gates"].append(event)
        return event

    def _check_budget(self, action, position, channel):
        if self._transit_context is not None:
            began = time.perf_counter()
            gate = self._transit_gate(action, position, channel, kind="action")
            self._transit_context["event"]["decision_wall_s"] += time.perf_counter()-began
            if not gate["admitted"]:
                raise _TransitSliceExpired(gate["id"], gate["reasons"])
        # This hook is reached even by R8's cooperative super()._perform calls.
        return super()._check_budget(action, position, channel)

    def _insertion_budget(self, point, channel):
        allowed, details = super()._insertion_budget(point, channel)
        if allowed and self._transit_context is not None:
            began = time.perf_counter()
            gate = self._transit_gate("measure", point, channel, kind="atomic_r8")
            self._transit_context["event"]["decision_wall_s"] += time.perf_counter()-began
            if not gate["admitted"]:
                # No speculative optical request, nor any fictitious R8 event,
                # exists yet. A later real-deadline/rejection remains inherited.
                raise _TransitSliceExpired(gate["id"], gate["reasons"])
        return allowed, details

    def _finish_gates(self, event):
        history = self.report.action_history
        end = event["service_end_action_count"]
        for gate in event["gates"]:
            prefix = gate["after_actual_action_count"]
            if not gate["admitted"] or prefix >= end:
                continue
            actual = history[prefix]
            same = actual["channel"] == gate["channel"] and actual["position"] == gate["point"]
            if gate["kind"] == "action":
                gate["executed_action_count"] = int(same and actual["action"] == gate["action"])
            elif same and actual["action"] == "clear" and actual["phase"] == "speculative_clear_before_probe":
                gate["executed_action_count"] = 1
                if actual["result"] != "success" and prefix+1 < end:
                    second = history[prefix+1]
                    if (second["action"] == "measure" and second["phase"] == "active_localization"
                            and second["channel"] == gate["channel"] and second["position"] == gate["point"]):
                        gate["executed_action_count"] = 2

    def _scan(self, point):
        if self._transit_context is not None:
            raise ValueError("A transit service cannot recursively scan coverage")
        began = time.perf_counter()
        destination = Position.coerce(point)
        state = self.client.state
        origin = state.position
        prefix = len(self.report.action_history)
        event = dict(id=len(self.transit_service_log), after_actual_action_count=prefix,
            end_actual_action_count=prefix, origin=_xy(origin), destination=_xy(destination),
            start_virtual_us=_clock_us(state.virtual_time_s),
            baseline_floor_us=math.floor(origin.distance_to(destination)/5.*MICRO),
            known_channels=sorted(self.detected|self.cleared),
            early_attempted_before=sorted(self.early_attempted),
            transit_attempted_before=sorted(self.transit_attempted),
            coverage_visited_before=self.report.coverage_points_visited,
            old_early_candidate=None, candidates=[], selected=None, skip_reason=None,
            resolver_id=None, resolver_start_action_count=None, service_end_action_count=prefix,
            service_status="not_started", scan_start_action_count=None,
            scan_end_action_count=None, scan_status="not_started", first_coverage_action_index=None,
            gates=[], decision_wall_s=0., runtime_s=0.)
        self.transit_service_log.append(event)
        try:
            if self.service_deadline is not None or self._joint_context is not None:
                event["skip_reason"] = "parent_service_context"
            elif len(self.detected|self.cleared) >= 16:
                event["skip_reason"] = "discovery_count_cap"
            elif origin == destination:
                event["skip_reason"] = "same_position"
            elif len(self.transit_attempted) >= 4:
                event["skip_reason"] = "macro_limit"
            else:
                old = self._early_candidate(destination)
                event["old_early_candidate"] = list(old) if old is not None else None
                if old is not None:
                    event["skip_reason"] = "old_early_candidate"
                else:
                    event["candidates"] = self._transit_candidates(destination)
                    if not event["candidates"]:
                        event["skip_reason"] = "no_candidate"
                    else:
                        event["selected"] = dict(event["candidates"][0])
            event["decision_wall_s"] = time.perf_counter()-began
            if event["selected"] is not None:
                channel = event["selected"]["channel"]
                self.transit_attempted.add(channel)
                event["resolver_start_action_count"] = len(self.report.action_history)
                event["resolver_id"] = len(self.joint_resolvers)
                self._transit_context = dict(event=event, destination=destination,
                    start_virtual_us=event["start_virtual_us"], baseline_floor_us=event["baseline_floor_us"])
                try:
                    resolved = self._resolve(channel)
                    event["service_status"] = "cleared" if resolved and channel in self.cleared else "unresolved"
                except _TransitSliceExpired as stopped:
                    event.update(service_status="slice_expired", stopped_gate_id=stopped.gate_id,
                                 stopped_reasons=stopped.reasons)
                finally:
                    self._transit_context = None
                    event["service_end_action_count"] = len(self.report.action_history)
                    event["service_end_virtual_us"] = _clock_us(state.virtual_time_s)
                    event["service_end_position"] = _xy(state.position)
                    self._finish_gates(event)
            event["scan_start_action_count"] = len(self.report.action_history)
            event["scan_status"] = "running"
            # Only normal completion or our own local slice expiry reaches this
            # line. Never issue a scan from a finally after rejection/deadline.
            result = super()._scan(destination)
            event["scan_status"] = "completed"
            return result
        except Exception as error:
            if event["selected"] is not None and event["service_status"] == "not_started":
                event["service_status"] = "interrupted"
            if event["scan_status"] == "running":
                event["scan_status"] = "interrupted"
            event.update(interruption_type=type(error).__name__, interruption_reason=str(error))
            raise
        finally:
            event["end_actual_action_count"] = len(self.report.action_history)
            event["transit_attempted_after"] = sorted(self.transit_attempted)
            event["coverage_visited_after"] = self.report.coverage_points_visited
            start = event["scan_start_action_count"]
            if start is not None:
                event["scan_end_action_count"] = len(self.report.action_history)
                if start < len(self.report.action_history):
                    first = self.report.action_history[start]
                    event["first_coverage_action_index"] = start
                    event["first_coverage_virtual_us"] = _clock_us(first["virtual_time_s"])
                    if event["selected"] is not None:
                        event["actual_incremental_through_first_measure_us"] = (
                            event["first_coverage_virtual_us"]-event["start_virtual_us"]-event["baseline_floor_us"])
            event["runtime_s"] = time.perf_counter()-began


def run_q4_transit_budget(client, *, problem=4, config=CONFIG, max_actions=20000,
                          max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config != CONFIG:
        raise ValueError("Q4 only; config must be incremental_60")
    for value, low, high in ((max_actions,2,1_000_000),(max_active_probes,0,30),(max_expansions,0,10000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid integer execution budget")
    return Q4TransitBudget(client,max_actions,max_active_probes,
                           max_expansions=max_expansions,config=config).run()
