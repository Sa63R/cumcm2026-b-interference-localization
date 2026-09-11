"""Resolver-local joint visibility on top of the frozen R8 controller.

The canonical positive region, discovery routing and scan certificates remain
owned by the inherited controller. An auxiliary outer region only changes this
resolver's probe points and, optionally, its certified optical cover.
"""
import math
import time

from planning import clearance_grid
from planning.joint_visibility_region import joint_visibility_outer
from simulator_client.state import Position
from .q4_clear_before_probe import Q4ClearBeforeProbe


class _JointSourceCleared(Exception):
    def __init__(self, channel):
        self.channel = channel


def _xy(position):
    position = Position.coerce(position)
    return [position.x, position.y]


class Q4JointVisibility(Q4ClearBeforeProbe):
    def __init__(self, client, max_actions, max_active_probes, *, config, max_expansions):
        if config not in {"probe", "probe_optical"}:
            raise ValueError("Unknown joint visibility configuration")
        super().__init__(client, max_actions, max_active_probes,
                         max_expansions=max_expansions, config="center_once")
        self.joint_config = config
        self.joint_radio = {}
        self.joint_resolvers, self.joint_probes, self.joint_grids = [], [], []
        self.joint_terminal_clears = []
        self.joint_contradictions = []
        self._joint_context = None
        self.report.strategy_parameters.update(
            joint_visibility_config=config, joint_visibility_resolver_log=self.joint_resolvers,
            joint_visibility_probe_log=self.joint_probes, joint_visibility_grid_log=self.joint_grids,
            joint_visibility_terminal_clear_log=self.joint_terminal_clears,
            joint_visibility_model_contradictions=self.joint_contradictions,
            joint_visibility_scope="One helper per already scheduled resolver; subsequent actual bearings refine a separate auxiliary region; canonical routing/scans and R8 predicate unchanged",
        )

    def _start_joint(self, channel):
        began = time.perf_counter()
        prefix = len(self.report.action_history)
        event = dict(id=len(self.joint_resolvers), channel=channel,
            after_actual_action_count=prefix, end_actual_action_count=prefix,
            skip_reason=None, helper_evidence=None, initial_aux_vertices=None,
            aux_updates=[], status="running", decision_wall_s=0.)
        self.joint_resolvers.append(event)
        region = self.regions.get(channel)
        radio = self.joint_radio.get(channel, [])
        positives = [x["position"] for x in radio if x["result"] in {"direction", "near"}]
        negatives = [x["position"] for x in radio if x["result"] == "no_signal"]
        aux = None
        if channel in self.cleared:
            event["skip_reason"] = "already_cleared"
        elif channel not in self.detected or region is None or not region.vertices:
            event["skip_reason"] = "no_canonical_positive_region"
        elif channel in self.near_points or region.enclosing_disk().radius <= 19.9:
            event["skip_reason"] = "canonical_ready"
        elif not negatives:
            event["skip_reason"] = "no_negative_evidence"
        else:
            outer, evidence = joint_visibility_outer(region.vertices, positives, negatives)
            event["helper_evidence"] = evidence
            if not evidence.get("passed") or evidence["status"] == "fallback":
                event["skip_reason"] = "helper_fallback"
            else:
                aux = region.copy()
                aux.vertices = tuple(tuple(v) for v in outer)
                aux._circle = None
                event["initial_aux_vertices"] = [list(p) for p in aux.vertices]
        event["decision_wall_s"] = time.perf_counter()-began
        return dict(channel=channel, event=event, region=aux, pending_probe=None,
                    optical_attempted=False, terminal_clear_attempted=False)

    def _resolve(self, channel):
        previous = self._joint_context
        context = self._start_joint(channel)
        self._joint_context = context
        try:
            try:
                result = super()._resolve(channel)
            except _JointSourceCleared as stopped:
                if stopped.channel != channel or channel not in self.cleared:
                    raise
                result = True
            context["event"]["status"] = "cleared" if result and channel in self.cleared else "unresolved"
            return result
        except Exception as error:
            context["event"].update(status="interrupted", interruption_type=type(error).__name__,
                                     interruption_reason=str(error))
            raise
        finally:
            context["event"]["end_actual_action_count"] = len(self.report.action_history)
            self._joint_context = previous

    def _perform(self, action, position, channel, phase):
        context = self._joint_context
        pending = context["pending_probe"] if context and context["channel"] == channel else None
        is_pending = (pending is not None and action == "measure" and phase == "active_localization"
                      and _xy(position) == pending["point"])
        try:
            response = super()._perform(action, position, channel, phase)
            if action == "measure" and channel not in self.cleared:
                item = dict(after_actual_action_count=len(self.report.action_history), position=_xy(position),
                            result=response["measure_result"])
                if item["result"] == "direction":
                    item["bearing_deg"] = response["svd_deg"]
                self.joint_radio.setdefault(channel, []).append(item)
                if (context and context["channel"] == channel and context["region"] is not None
                        and item["result"] == "direction"):
                    context["region"].observe(position, response["svd_deg"])
                    update = dict(item, aux_vertices=[list(p) for p in context["region"].vertices])
                    context["event"]["aux_updates"].append(update)
                    if not context["region"].vertices:
                        self.joint_contradictions.append(dict(resolver_id=context["event"]["id"],
                            channel=channel, after_actual_action_count=len(self.report.action_history),
                            reason="accepted_bearing_emptied_auxiliary_region"))
                        context["region"] = None
            return response
        finally:
            if is_pending:
                end = len(self.report.action_history)
                accepted = self.report.action_history[pending["after_actual_action_count"]:end]
                pending["end_actual_action_count"] = end
                pending["executed_measure"] = any(a["action"] == "measure" and a["channel"] == channel
                    and a["position"] == pending["point"] for a in accepted)
                pending["r8_clear_before_measure"] = any(a["phase"] == "speculative_clear_before_probe" for a in accepted)
                pending["status"] = ("measured" if pending["executed_measure"] else
                    "cleared_by_r8" if channel in self.cleared else "interrupted")
                context["pending_probe"] = None

    def _aux_ready_clear(self, channel, context, event):
        try:
            if super()._clear(Position.coerce(event["center"]), channel, "joint_visibility_clear"):
                event["status"] = "cleared"
                return True
            event["status"] = "certified_clear_failed"
            self.joint_contradictions.append(dict(resolver_id=context["event"]["id"], channel=channel,
                after_actual_action_count=len(self.report.action_history), reason="joint_visibility_clear_failed"))
            return False
        finally:
            event["end_actual_action_count"] = len(self.report.action_history)
            if event["status"] == "planned":
                event["status"] = "interrupted"

    def _next_probe(self, channel, index):
        original = super()._next_probe(channel, index)
        context = self._joint_context
        if not context or context["channel"] != channel or context["region"] is None:
            return original
        began = time.perf_counter()
        region = context["region"]
        circle = region.enclosing_disk()
        center = Position.coerce(circle.center)
        event = dict(resolver_id=context["event"]["id"], channel=channel, index=index,
            after_actual_action_count=len(self.report.action_history),
            end_actual_action_count=len(self.report.action_history),
            canonical_point=_xy(original) if original is not None else None,
            aux_vertices=[list(p) for p in region.vertices], center=_xy(center), radius_m=circle.radius,
            candidates=[], fresh_indices=[], selected=None, point=None, kind="probe", status="planned",
            executed_measure=False, r8_clear_before_measure=False, decision_wall_s=0.)
        self.joint_probes.append(event)
        if circle.radius <= 19.9:
            event.update(kind="clear", point=_xy(center), decision_wall_s=time.perf_counter()-began)
            if self._aux_ready_clear(channel, context, event):
                raise _JointSourceCleared(channel)
            return original
        angle = math.radians(self.first_bearings[channel])
        scale = min(180., max(25., circle.radius*.5))
        perpendicular = (-math.sin(angle), math.cos(angle))
        offsets = ((0., 0.), (perpendicular[0]*scale, perpendicular[1]*scale),
            (-perpendicular[0]*scale, -perpendicular[1]*scale),
            (math.cos(angle)*scale, math.sin(angle)*scale),
            (-math.cos(angle)*scale, -math.sin(angle)*scale))
        candidates = [Position(center.x+x, center.y+y) for x, y in offsets]
        observed = self.observed_positions.get(channel, set())
        fresh = [i for i, p in enumerate(candidates) if (round(p.x, 6), round(p.y, 6)) not in observed]
        selected = 0 if 0 in fresh else min(fresh, key=lambda i: self.client.state.position.distance_to(candidates[i])) if fresh else None
        chosen = candidates[selected] if selected is not None else None
        event.update(candidates=[_xy(p) for p in candidates], fresh_indices=fresh, selected=selected,
            point=_xy(chosen) if chosen is not None else None, status="planned" if chosen is not None else "no_fresh_probe",
            decision_wall_s=time.perf_counter()-began)
        if chosen is not None:
            context["pending_probe"] = event
        return chosen

    def _run_joint_optical_once(self, original_point, channel, context):
        context["optical_attempted"] = True
        region = context["region"]
        began = time.perf_counter()
        start = self.client.state.position
        vertices = tuple(region.vertices)
        bearing = self.first_bearings[channel]
        grid = clearance_grid(vertices, bearing_deg=bearing, spacing=28., start=start)
        event = dict(resolver_id=context["event"]["id"], channel=channel,
            after_actual_action_count=len(self.report.action_history),
            end_actual_action_count=len(self.report.action_history), canonical_first_point=_xy(original_point),
            aux_vertices=[list(p) for p in vertices], bearing_deg=bearing, start=_xy(start), spacing_m=28.,
            grid=[_xy(p) for p in grid], actual_grid_actions=0, status="planned",
            decision_wall_s=time.perf_counter()-began)
        self.joint_grids.append(event)
        try:
            for point in grid:
                if super()._clear(point, channel, "joint_visibility_optical"):
                    event["status"] = "cleared"
                    return True
            event["status"] = "exhausted_without_success" if grid else "empty_grid"
            self.joint_contradictions.append(dict(resolver_id=context["event"]["id"], channel=channel,
                after_actual_action_count=len(self.report.action_history), reason=event["status"]))
            return False
        finally:
            event["end_actual_action_count"] = len(self.report.action_history)
            event["actual_grid_actions"] = event["end_actual_action_count"]-event["after_actual_action_count"]
            if event["status"] == "planned":
                event["status"] = "interrupted"

    def _clear(self, position, channel, phase):
        context = self._joint_context
        eligible = (phase == "guaranteed_clearance" and context and context["channel"] == channel
                    and context["region"] is not None)
        if eligible and not context["terminal_clear_attempted"]:
            began = time.perf_counter()
            region = context["region"]
            circle = region.enclosing_disk()
            if circle.radius <= 19.9:
                context["terminal_clear_attempted"] = True
                event = dict(resolver_id=context["event"]["id"], channel=channel,
                    after_actual_action_count=len(self.report.action_history),
                    end_actual_action_count=len(self.report.action_history), canonical_first_point=_xy(position),
                    aux_vertices=[list(p) for p in region.vertices], center=list(circle.center),
                    radius_m=circle.radius, status="planned", decision_wall_s=time.perf_counter()-began)
                self.joint_terminal_clears.append(event)
                if self._aux_ready_clear(channel, context, event):
                    return True
        if (phase == "guaranteed_clearance" and self.joint_config == "probe_optical"
                and context and context["channel"] == channel and context["region"] is not None
                and not context["optical_attempted"]):
            if self._run_joint_optical_once(position, channel, context):
                return True
        return super()._clear(position, channel, phase)


def run_q4_joint_visibility(client, *, problem=4, config="probe", max_actions=20000,
                            max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config not in {"probe", "probe_optical"}:
        raise ValueError("Q4 only; config must be probe or probe_optical")
    for value, low, high in ((max_actions, 2, 1_000_000), (max_active_probes, 0, 30),
                             (max_expansions, 0, 10000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid integer execution budget")
    return Q4JointVisibility(client, max_actions, max_active_probes,
                             config=config, max_expansions=max_expansions).run()
