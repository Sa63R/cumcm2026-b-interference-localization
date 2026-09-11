"""CPU-only, public-prefix full-continuation labels for a frozen greedy actor.

``collect_group(task) -> (group_or_None, metrics)`` is a spawn-safe worker.
Required task keys: seed, weights (CPU state_dict), hidden, action_seed.
Optional: deadline_epoch, max_decisions=256, alternatives=2,
alternative_sampling="uniform" (or "runner_up_uniform"), capture_evidence=False.
Only v3/base/flat MLP is supported in this experiment.
The scene seed and post-termination costs are provenance/labels, never inputs
to the actor. One group is sampled uniformly from ALL actual policy decisions,
including single-candidate decisions; a one-choice group is not resampled.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
import random
import time

import numpy as np

from .cpu_runtime import require_cpu
import torch

from .action_sets import action_schema, controller_for
from .network import CandidateActorCritic, architecture_spec, pack_observations
from simulation import LocalResearchSimulator, random_scenario


COLLECTOR_VERSION = "public-prefix-full-rollout-v1"
ALTERNATIVE_SAMPLINGS = ("uniform", "runner_up_uniform")
FAILURE_PENALTY_S = 360000.0
TRAINING_RANGES = ((3100001, 3399999), (3400001, 3699999), (3700001, 3999999))
_worker_model = None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def public_history(history):
    """Strip only independently advancing response wall-clock timestamps."""
    result = copy.deepcopy(history)
    for row in result:
        row["response"].pop("real_timestamp_ms", None)
    return result


def _physical_count(history):
    return sum(row["action"] in {"/measure", "/clear"} for row in history)


def _expired(deadline):
    return deadline is not None and time.time() >= deadline


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid collector {name}")
    return value


def _validate_task(task):
    seed = _integer(task["seed"], "seed", 0, 2**63 - 1)
    if not any(first <= seed <= last for first, last in TRAINING_RANGES):
        raise ValueError("Collector seed is outside the declared autonomy training ranges")
    _integer(task["hidden"], "hidden", 1, 4096)
    _integer(task["action_seed"], "action_seed", 0, 2**63 - 1)
    _integer(task.get("max_decisions", 256), "max_decisions", 1, 256)
    _integer(task.get("alternatives", 2), "alternatives", 1, 2)
    if task.get("alternative_sampling", "uniform") not in ALTERNATIVE_SAMPLINGS:
        raise ValueError("Invalid collector alternative_sampling")
    deadline = task.get("deadline_epoch")
    if deadline is not None and (isinstance(deadline, bool)
            or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)):
        raise ValueError("Invalid collector deadline_epoch")
    if type(task.get("capture_evidence", False)) is not bool:
        raise ValueError("Invalid collector capture_evidence")
    for key, expected in (("feature_version", "v3"), ("architecture", architecture_spec()),
                          ("action_schema", action_schema()),
                          ("action_distribution", {"version": 1, "name": "flat"})):
        if key in task and task[key] != expected:
            raise ValueError("Collector requires unchanged v3/base/flat MLP semantics")
    if not isinstance(task["weights"], dict) or not task["weights"]:
        raise ValueError("Collector requires a CPU model state_dict")
    if any(not isinstance(t, torch.Tensor) or t.device.type != "cpu"
           for t in task["weights"].values()):
        raise ValueError("Collector weights must be CPU tensors")


def _model_for(task):
    global _worker_model
    if _worker_model is None or _worker_model.hidden != task["hidden"]:
        # Constructing the cache must not consume the learner RNG for workers=0.
        with torch.random.fork_rng(devices=[]):
            _worker_model = CandidateActorCritic(task["hidden"], feature_dim=60)
    _worker_model.load_state_dict(task["weights"], strict=True)
    _worker_model.eval()
    return _worker_model


class ReplayMismatch(RuntimeError):
    """Public observations, legal choices, or executed prefix are inconsistent."""


class _PrefixPolicy:
    def __init__(self, model, history_reader, *, rng=None, expected=None, forced_index=None):
        self.model, self.history_reader = model, history_reader
        self.rng, self.expected, self.forced_index = rng, expected, forced_index
        self.architecture, self.action_distribution, self.action_schema = (
            model.architecture, model.action_distribution, model.action_schema)
        self.control = None
        self.candidates = self.regions = None
        self.decisions, self.events = [], []
        self.selected_prefix = None
        self.prefix_verified = False
        self.forced_actions = 0

    def prepare_candidates(self, candidates, regions):
        self.candidates, self.regions = candidates, regions

    def signature(self, features, context, original):
        c = self.control
        # Values below are all public observations or deterministic controller
        # bookkeeping. No scenario, seed, evaluator, or true-source access.
        state = dict(candidates=[asdict(v) for v in self.candidates],
            original_index=original, position=asdict(c.client.state.position),
            channel=c.client.state.current_channel, time_s=c.client.state.virtual_time_s,
            accepted_actions=c.client.state.accepted_actions,
            max_virtual_duration_s=c.client.state.max_virtual_duration_s,
            regions={str(k): list(v.vertices) for k, v in sorted(self.regions.items())},
            detected=sorted(c.detected), cleared=sorted(c.cleared), blocked=sorted(c.blocked),
            probe_counts=c.probe_counts, focus=c.focus,
            scan_focus=asdict(c.scan_focus) if c.scan_focus is not None else None,
            scan_ledger=[dict(point=asdict(p), channels=sorted(channels))
                         for p, channels in c.scan_ledger.items()])
        h = hashlib.sha256(_digest(state).encode("ascii"))
        for array in (features, context):
            h.update(str(array.shape).encode("ascii"))
            h.update(array.tobytes(order="C"))
        return h.hexdigest()

    @torch.no_grad()
    def __call__(self, features, context, teacher):
        # teacher is a public heuristic index, but is deliberately unused.
        features = np.asarray(features, dtype=np.float32)
        context = np.asarray(context, dtype=np.float32)
        logits, _ = self.model(*pack_observations([dict(features=features, context=context)]))
        old_logits = logits[0].cpu().numpy().copy()
        if not np.isfinite(old_logits).all():
            raise ReplayMismatch("Non-finite actor logits")
        original = int(old_logits.argmax())
        step, selected = len(self.decisions), original
        needs_signature = self.expected is None or step <= self.expected["step"]
        signature = self.signature(features, context, original) if needs_signature else None
        before = self.control.client.state.virtual_time_s
        # Histories are short (finite decisions/actions); the returned copy is
        # used for verification, never supplied to the neural network.
        history = self.history_reader()
        if self.expected is None:
            # Reservoir sampling is uniform over all decisions, not only probes
            # or cases whose future outcome looks promising.
            if self.rng.randrange(step + 1) == 0:
                prefix = public_history(history)
                self.selected_prefix = dict(step=step, features=features.copy(),
                    context=context.copy(), old_logits=old_logits, original_index=original,
                    prefix_cost_s=before, prefix=prefix, prefix_sha256=_digest(prefix),
                    prefix_action_count=len(prefix), signature=signature,
                    candidate_count=len(self.candidates))
        elif step <= self.expected["step"]:
            target = self.expected
            prior = target["decisions"][step]
            if signature != prior["signature"] or original != prior["original_index"]:
                raise ReplayMismatch("Public decision signature differs during replay")
            if step == target["step"]:
                if public_history(history) != target["prefix"]:
                    raise ReplayMismatch("Accepted public history differs at fork")
                if (not np.array_equal(old_logits, target["old_logits"])
                        or not 0 <= self.forced_index < len(self.candidates)):
                    raise ReplayMismatch("Frozen actor or forced legal action differs")
                self.prefix_verified = True
                self.forced_actions += 1
                selected = self.forced_index
        self.decisions.append(dict(step=step, original_index=original,
            selected_index=selected, signature=signature,
            before_virtual_time_s=before, before_public_action_count=len(history)))
        return selected

    def record(self, features, context, action, teacher, selection, cost):
        if not self.decisions or action != self.decisions[-1]["selected_index"]:
            raise ReplayMismatch("Recorded action differs from chosen legal action")
        self.events.append(dict(step=len(self.events), cost_s=float(cost),
            after_virtual_time_s=self.control.client.state.virtual_time_s))


def _run_trajectory(task, model, *, rng=None, expected=None, forced_index=None):
    """One fresh engine; controller's finally owns safe exit even on exception."""
    simulator = LocalResearchSimulator(random_scenario(3, task["seed"]),
                                      max_real_duration_s=300)
    policy = _PrefixPolicy(model, simulator.observation_history,
                           rng=rng, expected=expected, forced_index=forced_index)
    control = controller_for("v3", action_schema())(simulator.client(), policy,
        recorder=policy.record, max_decisions=task.get("max_decisions", 256),
        max_actions=10000, action_deadline_epoch=task.get("deadline_epoch"), feature_version="v3")
    policy.control = control
    error_code = None
    try:
        report = control.run()
    except Exception as error:
        # Do not expose tracebacks, paths, or exception messages in result logs.
        report, error_code = control.report, type(error).__name__
    # run() always attempts legal exit. A harness stop here only makes post-run
    # audit available; it can NEVER turn an incomplete run into a cost label.
    if control.client.state.session not in {"exited", "real_timeout", "virtual_timeout"}:
        simulator.finish_for_evaluation("collector_incomplete")
    evaluation = simulator.evaluation()
    history = simulator.observation_history()
    engine_stop = evaluation["simulator_stop_reason"]
    administrative = (report.completion_reason in {"training_deadline", "real_deadline"}
                      or engine_stop == "real_timeout") and engine_stop != "virtual_timeout"
    complete = (engine_stop in {"exited", "virtual_timeout"}
                and not administrative and error_code is None)
    if report.completion_reason == "protocol_error" or (
            complete and engine_stop == "exited" and control.client.pending_request is not None):
        complete, error_code = False, "ProtocolError"
    if not administrative and not complete and error_code is None:
        error_code = "IncompleteExecution"
    success = bool(complete and evaluation["all_cleared"]
                   and report.completion_certified_under_model
                   and evaluation["failed_clear_count"] == 0
                   and not report.error and not report.exit_error)
    total = float(evaluation["virtual_time_s"])
    initial = float(report.learning.get("initial_scan_virtual_time_s", 0.0))
    after = policy.events[-1]["after_virtual_time_s"] if policy.events else initial
    tail = total - after
    accounting = (tail >= -2e-6 and math.isclose(
        initial + sum(e["cost_s"] for e in policy.events) + tail, total, abs_tol=2e-6)
        and len(policy.events) == len(policy.decisions)
        and len(policy.events) == report.learning["decisions"])
    if complete and not accounting:
        complete, error_code = False, "CostAccountingMismatch"
    if complete and expected is not None and (not policy.prefix_verified or policy.forced_actions != 1):
        complete, error_code = False, "MissingVerifiedIntervention"
    if complete and expected is not None and total + 2e-6 < expected["prefix_cost_s"]:
        complete, error_code = False, "NegativeRemainingCost"
    success = success and complete
    penalty = FAILURE_PENALTY_S if complete and not success else 0.0
    replay_count = 0
    if expected is not None:
        # Count actual work spent before the intervention, also on interruption.
        replay_count = _physical_count(history[:min(len(history), expected["prefix_action_count"])])
    row = dict(index=forced_index, total_time_s=total, success=success,
        case_sha256=_digest(evaluation["ground_truth"]),
        all_cleared=bool(evaluation["all_cleared"]),
        failed_clear_count=int(evaluation["failed_clear_count"]), failure_penalty_s=penalty,
        completion_reason=report.completion_reason, simulator_stop_reason=engine_stop,
        accepted_exit=engine_stop == "exited", complete=complete,
        administrative_timeout=administrative, error_code=error_code,
        simulator_action_count=_physical_count(history), accepted_request_count=len(history),
        replay_prefix_action_count=replay_count,
        additional_completion_tail_s=max(0.0, tail), cost_accounting_verified=accounting,
        prefix_verified=policy.prefix_verified, forced_actions=policy.forced_actions)
    evidence = None
    if task.get("capture_evidence", False):
        evidence_row = dict(seed=task["seed"], case_id=evaluation["case_id"],
            case_sha256=_digest(evaluation["ground_truth"]),
            successful=success, all_cleared=evaluation["all_cleared"],
            completion_certified=report.completion_certified_under_model,
            accepted_exit=engine_stop == "exited", source_total=evaluation["source_total"],
            cleared_total=evaluation["cleared_total"], measurement_count=evaluation["measurement_count"],
            failed_clear_count=evaluation["failed_clear_count"], action_count=len(history),
            virtual_time_s=total,
            errors=[code for code in (error_code,
                    "ControllerError" if report.error else None,
                    "ExitError" if report.exit_error else None) if code],
            **evaluation["time_breakdown_s"])
        evidence = dict(row=evidence_row, evaluation=evaluation, summary=report.as_dict(),
            history=history, decisions=policy.decisions, events=policy.events,
            evaluation_phase="after_policy_termination")
    return dict(row=row, selected_prefix=policy.selected_prefix,
                decisions=policy.decisions, evidence=evidence)


def _label(row, index, prefix_cost):
    remaining = row["total_time_s"] - prefix_cost
    if remaining < -2e-6:
        raise ReplayMismatch("Terminal cost precedes the public prefix")
    return dict(row, index=index, remaining_cost_s=max(0.0, remaining),
                cost_to_go_s=max(0.0, remaining) + row["failure_penalty_s"])


def select_alternatives(old_logits, original, count, rng, mode="uniform"):
    """Use only frozen public logits; never choose by continuation outcomes.

    The uniform branch retains its original RNG call and index ordering.
    Runner-up ties use the lowest candidate index; an optional second action
    is sampled uniformly without replacement from the remaining legal actions.
    """
    if mode not in ALTERNATIVE_SAMPLINGS:
        raise ValueError("Invalid collector alternative_sampling")
    logits = np.asarray(old_logits)
    if (logits.ndim != 1 or not len(logits) or not np.isfinite(logits).all()
            or isinstance(original, bool) or not isinstance(original, (int, np.integer))
            or original != int(logits.argmax())):
        raise ValueError("Alternative selection requires valid frozen greedy logits")
    _integer(count, "alternatives", 1, 2)
    available = [i for i in range(len(logits)) if i != original]
    count = min(count, len(available))
    if mode == "uniform":
        return rng.sample(available, count)
    if count == 0:
        return []
    runner_up = min(available, key=lambda i: (-float(logits[i]), i))
    remaining = [i for i in available if i != runner_up]
    return [runner_up] + (rng.sample(remaining, count - 1) if count > 1 else [])


def collect_group(task):
    """Return one complete action-cost group and exact actual-work accounting.

    Invalid configuration raises ValueError before work. Runtime inconsistency
    returns status=invalid_execution: callers must abort, not learn partial
    labels. Administrative timeout invalidates the whole group, while actual
    terminal failures retain costs and the explicit fixed penalty.
    """
    require_cpu()
    _validate_task(task)
    sampling = task.get("alternative_sampling", "uniform")
    started, cpu_started = time.perf_counter(), time.process_time()
    metrics = dict(seed=task["seed"], collector_version=COLLECTOR_VERSION,
        status="administrative_timeout", trajectories_started=0, trajectories_completed=0,
        trajectories_interrupted=0, successful_trajectories=0, failed_trajectories=0,
        failed_clear_count=0, simulator_action_count=0, accepted_request_count=0,
        replay_prefix_action_count=0, baseline_total_time_s=None, trajectory_metrics=[],
        alternative_sampling=sampling, alternative_indices=[],
        alternative_selection_scope="frozen public logits and independent action RNG; no future outcomes")
    if task.get("capture_evidence", False):
        metrics["evidence"] = []

    def finish(group, status):
        metrics.update(status=status, wall_time_s=time.perf_counter()-started,
                       worker_cpu_s=time.process_time()-cpu_started)
        return group, metrics

    def run(**kwargs):
        metrics["trajectories_started"] += 1
        result = _run_trajectory(task, model, **kwargs)
        row = result["row"]
        metrics["trajectory_metrics"].append(row)
        metrics["trajectories_completed"] += int(row["complete"])
        metrics["trajectories_interrupted"] += int(not row["complete"])
        metrics["successful_trajectories"] += int(row["success"])
        metrics["failed_trajectories"] += int(row["complete"] and not row["success"])
        for key in ("failed_clear_count", "simulator_action_count", "accepted_request_count",
                    "replay_prefix_action_count"):
            metrics[key] += row[key]
        if result["evidence"] is not None:
            metrics["evidence"].append(result["evidence"])
        return result

    if _expired(task.get("deadline_epoch")):
        return finish(None, "administrative_timeout")
    torch.set_num_threads(1)
    model = _model_for(task)
    rng = random.Random(task["action_seed"])
    baseline = run(rng=rng)
    row = baseline["row"]
    metrics["baseline_total_time_s"] = row["total_time_s"]
    if not row["complete"]:
        return finish(None, "administrative_timeout" if row["administrative_timeout"] else "invalid_execution")
    prefix = baseline["selected_prefix"]
    if prefix is None or prefix["candidate_count"] < 2:
        return finish(None, "no_choice")
    original = prefix["original_index"]
    alternatives = select_alternatives(prefix["old_logits"], original,
                                      task.get("alternatives", 2), rng, sampling)
    metrics["alternative_indices"] = list(alternatives)
    prefix["decisions"] = baseline["decisions"][:prefix["step"]+1]
    labels = [_label(row, original, prefix["prefix_cost_s"])]
    for index in alternatives:
        if _expired(task.get("deadline_epoch")):
            return finish(None, "administrative_timeout")
        alternative = run(expected=prefix, forced_index=index)
        row = alternative["row"]
        if not row["complete"]:
            return finish(None, "administrative_timeout" if row["administrative_timeout"] else "invalid_execution")
        if row["case_sha256"] != baseline["row"]["case_sha256"]:
            row["error_code"] = "WorldIdentityMismatch"
            return finish(None, "invalid_execution")
        labels.append(_label(row, index, prefix["prefix_cost_s"]))
    group = dict(seed=task["seed"], collector_version=COLLECTOR_VERSION,
        features=prefix["features"], context=prefix["context"], old_logits=prefix["old_logits"],
        original_index=original, prefix_cost_s=prefix["prefix_cost_s"], prefix_step=prefix["step"],
        prefix_sha256=prefix["prefix_sha256"], prefix_signature=prefix["signature"],
        candidate_count=prefix["candidate_count"], baseline_decisions=len(baseline["decisions"]),
        case_sha256=baseline["row"]["case_sha256"], evaluated_candidates=labels,
        alternative_sampling=sampling, alternative_indices=list(alternatives))
    return finish(group, "complete")
