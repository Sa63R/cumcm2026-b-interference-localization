"""Refresh a resolver-local visibility outer region after a real active miss.

Only the auxiliary region is replaced. Canonical positive geometry, R8 rules,
scan obligations, action/probe budgets and both optical fallbacks are inherited.
"""
import time

from planning.joint_visibility_region import joint_visibility_outer, MAX_CONSTRAINT_WORK
from .q4_joint_visibility import Q4JointVisibility, _xy


CONFIG = "after_active_miss_optical"


class Q4JointContinuation(Q4JointVisibility):
    def __init__(self, client, max_actions, max_active_probes, *, max_expansions, config=CONFIG):
        if config != CONFIG:
            raise ValueError("Unknown joint continuation configuration")
        super().__init__(client, max_actions, max_active_probes,
                         config="probe_optical", max_expansions=max_expansions)
        self.continuation_log = []
        self.continuation_calls = 0
        self.continuation_limit = 20 * max_active_probes
        self.report.strategy_parameters.update(joint_visibility_continuation_config=config,
            joint_visibility_continuation_log=self.continuation_log,
            joint_visibility_continuation_limits={"per_resolver": max_active_probes,
                "per_session": self.continuation_limit, "helper_constraint_work": MAX_CONSTRAINT_WORK},
            joint_visibility_continuation_scope="After each accepted active no_signal, at most one deterministic-budget helper on the current certified auxiliary outer (otherwise canonical C); all actual P/N, no inferred observations",
            joint_visibility_scope="R9 probe_optical plus audited recursive auxiliary refresh after accepted active misses; canonical geometry and inherited action budgets unchanged")

    def _start_joint(self, channel):
        context = super()._start_joint(channel)
        context.update(continuation_calls=0, last_applied_continuation_id=None,
                       last_applied_continuation_prefix=None)
        return context

    def _refresh_after_miss(self, channel, context):
        began = time.perf_counter()
        prefix = len(self.report.action_history)
        trigger = self.report.action_history[-1]
        if not (trigger["action"] == "measure" and trigger["phase"] == "active_localization"
                and trigger["result"] == "no_signal" and trigger["channel"] == channel):
            raise ValueError("Continuation requires its actual active-miss prefix")
        radio = self.joint_radio.get(channel, [])
        positive = [x["position"] for x in radio if x["result"] in {"direction", "near"}]
        negative = [x["position"] for x in radio if x["result"] == "no_signal"]
        previous_prefix = context["last_applied_continuation_prefix"]
        aux = context["region"]
        canonical = self.regions.get(channel)
        event = dict(id=len(self.continuation_log), resolver_id=context["event"]["id"], channel=channel,
            trigger_actual_action_index=prefix-1, after_actual_action_count=prefix,
            end_actual_action_count=prefix, trigger_position=_xy(trigger["position"]),
            status="pending", helper_called=False, helper_evidence=None, output_aux_vertices=None,
            input_source="auxiliary" if aux is not None else "canonical",
            input_vertices=[list(p) for p in (aux or canonical).vertices] if aux or canonical else None,
            canonical_vertices=[list(p) for p in canonical.vertices] if canonical else None,
            previous_aux_vertices=[list(p) for p in aux.vertices] if aux is not None else None,
            positive_positions=positive, negative_positions=negative,
            basis=dict(resolver_entry_prefix=context["event"]["after_actual_action_count"],
                previous_applied_event_id=context["last_applied_continuation_id"],
                previous_applied_prefix=previous_prefix,
                bearing_update_prefixes=[u["after_actual_action_count"] for u in context["event"]["aux_updates"]
                    if previous_prefix is None or u["after_actual_action_count"] > previous_prefix]),
            budget=dict(resolver_calls_before=context["continuation_calls"], session_calls_before=self.continuation_calls,
                resolver_limit=self.max_active_probes, session_limit=self.continuation_limit,
                helper_constraint_work=MAX_CONSTRAINT_WORK), decision_wall_s=0.)
        self.continuation_log.append(event)
        try:
            if channel in self.cleared or channel in self.near_points:
                event["status"] = "source_already_ready_or_cleared"
            elif canonical is None or not canonical.vertices or not canonical.observations:
                event["status"] = "no_canonical_positive_region"
            elif canonical.enclosing_disk().radius <= 19.9 or aux is not None and aux.enclosing_disk().radius <= 19.9:
                event["status"] = "already_certified_ready"
            elif context["continuation_calls"] >= self.max_active_probes:
                event["status"] = "resolver_work_budget"
            elif self.continuation_calls >= self.continuation_limit:
                event["status"] = "session_work_budget"
            else:
                context["continuation_calls"] += 1
                self.continuation_calls += 1
                event["helper_called"] = True
                source = aux if aux is not None else canonical
                outer, evidence = joint_visibility_outer(source.vertices, positive, negative)
                event["helper_evidence"] = evidence
                if not evidence.get("passed") or evidence.get("status") == "fallback":
                    event["status"] = "helper_fallback"
                elif evidence.get("status") != "outer_refined" or not evidence["old_vertices_excluded"]:
                    event["status"] = "no_boundary_reduction"
                else:
                    replacement = source.copy()
                    replacement.vertices = tuple(tuple(p) for p in outer)
                    replacement._circle = None
                    context["region"] = replacement
                    context["last_applied_continuation_id"] = event["id"]
                    context["last_applied_continuation_prefix"] = prefix
                    event.update(status="applied", output_aux_vertices=[list(p) for p in replacement.vertices])
        except Exception as error:
            event.update(status="interrupted", interruption_type=type(error).__name__, interruption_reason=str(error))
            raise
        finally:
            event["budget"].update(resolver_calls_after=context["continuation_calls"], session_calls_after=self.continuation_calls)
            event["decision_wall_s"] = time.perf_counter()-began

    def _perform(self, action, position, channel, phase):
        response = super()._perform(action, position, channel, phase)
        context = self._joint_context
        if (action == "measure" and phase == "active_localization" and response["measure_result"] == "no_signal"
                and context is not None and context["channel"] == channel and channel not in self.cleared):
            # R9 has already recorded the accepted radio feedback and finished
            # pending-probe bookkeeping. No new physical action is issued here.
            self._refresh_after_miss(channel, context)
        return response


def run_q4_joint_continuation(client, *, problem=4, config=CONFIG, max_actions=20000,
                               max_active_probes=6, max_expansions=200):
    if type(problem) is not int or problem != 4 or config != CONFIG:
        raise ValueError("Q4 only; use the single frozen continuation configuration")
    for value, low, high in ((max_actions, 2, 1_000_000), (max_active_probes, 0, 30), (max_expansions, 0, 10000)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid integer execution budget")
    return Q4JointContinuation(client, max_actions, max_active_probes,
                               max_expansions=max_expansions, config=config).run()
