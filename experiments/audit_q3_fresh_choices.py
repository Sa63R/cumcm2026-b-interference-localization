"""Post-hoc audit of selected R/CR actions in COMPLETED synthetic traces.

The final evaluator scenario is used only by this offline diagnostic.  It is
never passed back to a live planner, used for fitting, or substituted into its
recorded predictions.  Replay follows the decisions already recorded in the
trace; it never resamples worlds or asks the online planner to make a decision.

For each selected override, compare candidate -> frozen B against the original
pending action -> frozen B, both from the identical public state and controller
stack.  These are local post-hoc counterfactuals, not the total effect of all
later rollout choices.  Failed/truncated branches have no reported cost delta.
"""

import argparse
from dataclasses import asdict
import glob
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.run_q3_fresh import StressEngine
from simulation.cases import Scenario, Source
from strategies.q3_fresh_belief import make_branch
from strategies.q3_fresh_rollout import FreshRollout
from strategies.q3_fresh_stepper import StepperQ3


class ReplayMismatch(RuntimeError):
    """The current implementation did not reproduce a recorded action."""


def _same_position(a, b):
    return len(a) == len(b) == 2 and all(math.isclose(x, y, abs_tol=1e-8, rel_tol=0)
                                        for x, y in zip(a, b))


def matches_action(proposal, recorded):
    return (proposal.kind == recorded.get("action")
            and proposal.channel == recorded.get("channel")
            and proposal.phase == recorded.get("phase")
            and _same_position(proposal.position, recorded.get("position", ())))


def resolve_selected(options, record, recorded_next_action):
    """Disambiguate duplicate labels using the action actually in the trace.

    The online shortlist may contain two ``probe_other_source`` entries.
    Selecting the first equal label would silently audit a different action.
    Only a unique label AND recorded first-action match is accepted.
    """
    matches = [(index, proposal) for index, (label, proposal) in enumerate(options)
               if label == record["selected"] and proposal
               and matches_action(proposal[0], recorded_next_action)]
    if len(matches) != 1:
        raise ReplayMismatch(f"Selected label has {len(matches)} matching recorded actions")
    return matches[0]


def assert_same_observation(actual, expected, index):
    for key in ("action", "channel", "phase", "result"):
        if actual.get(key) != expected.get(key):
            raise ReplayMismatch(f"Action {index}: {key} differs")
    if not _same_position(actual["position"], expected["position"]):
        raise ReplayMismatch(f"Action {index}: position differs")
    for key in ("virtual_time_s", "bearing_deg"):
        if key not in actual and key not in expected:
            continue
        if key not in actual or key not in expected or not math.isclose(
                actual[key], expected[key], rel_tol=0, abs_tol=1e-5 if key == "virtual_time_s" else 1e-8):
            raise ReplayMismatch(f"Action {index}: {key} differs")


def _smooth_error(_engine, position, channel):
    return math.sin(position.x / 250 + position.y / 400 + channel)


def make_actual_branch(scenario, public_state, history):
    """Offline-only fork of the completed trace's actual research environment."""
    client = make_branch(scenario, public_state, history)
    if scenario.description == "smooth_spatial_error":
        # The nominal branch factory uses location-hashed errors.  The completed
        # stress environment instead used this fixed smooth field.  Restore it
        # ONLY on the auditor-owned branch engine, without editing policy code.
        # This deliberately private harness operation must never enter a planner.
        branch_engine = client._exchange_fn.__self__
        branch_engine._error = MethodType(_smooth_error, branch_engine)
    return client


def counterfactual_cost(controller, scenario, proposal, *, timeout_s=10.0):
    """Full certified-exit remaining cost under B, or an explicit failed sample."""
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    answer = dict(complete=False, remaining_s=None, error=None)
    try:
        client = make_actual_branch(scenario, controller.client.state, controller.history)
        fork = controller.clone(client)
        # Match FreshRollout._evaluate exactly, including a currently proposed
        # C action in CR while disabling subsequent movable-tail optimization.
        fork.movable_tail = False
        fork.tail_plan = []
        if proposal != (controller.pending,):
            fork.override(proposal)
        start = client.state.virtual_time_s
        result = fork.run(deadline=wall_start + timeout_s)
        if not result["completion_certified"] or client.state.session != "exited":
            raise RuntimeError("Counterfactual did not reach a certified exit")
        answer.update(complete=True, remaining_s=client.state.virtual_time_s-start,
                      final_virtual_time_s=client.state.virtual_time_s,
                      executed_future_actions=len(fork.history)-len(controller.history),
                      confirmation_tail_s=result.get("confirmation_tail_s"))
    except Exception as exc:
        answer["error"] = f"{type(exc).__name__}: {exc}"
    answer.update(wall_s=time.perf_counter()-wall_start, cpu_s=time.process_time()-cpu_start)
    return answer


class LoggedReplayPlanner:
    """Apply only logged choices; audit forks cannot change the replay choice."""
    def __init__(self, scenario, expected, decisions, *, max_overrides=10, branch_timeout_s=10.0):
        self.scenario, self.expected = scenario, expected
        self.decisions = {record["action_index"]: record for record in decisions}
        if len(self.decisions) != len(decisions):
            raise ReplayMismatch("Multiple planner decisions at one action index")
        self.generator = FreshRollout(enabled=False)
        self.max_overrides, self.branch_timeout_s = max_overrides, branch_timeout_s
        self.seen_decisions, self.audits, self.selected_count = set(), [], 0

    def maybe_choose(self, policy):
        index = len(policy.history)
        if index >= len(self.expected):
            raise ReplayMismatch("Replay produced more actions than the trace")
        record = self.decisions.get(index)
        if record is not None:
            self.seen_decisions.add(index)
        if record is None or record.get("selected", "baseline") == "baseline":
            if not matches_action(policy.pending, self.expected[index]):
                raise ReplayMismatch(f"Action {index}: baseline pending action differs")
            return
        options = self.generator.candidates(policy)
        if [label for label, _ in options] != record["candidates"]:
            raise ReplayMismatch(f"Action {index}: candidate shortlist differs")
        option_index, proposal = resolve_selected(options, record, self.expected[index])
        self.selected_count += 1
        if len(self.audits) < self.max_overrides:
            baseline = counterfactual_cost(policy, self.scenario, (policy.pending,),
                                           timeout_s=self.branch_timeout_s)
            candidate = counterfactual_cost(policy, self.scenario, proposal,
                                            timeout_s=self.branch_timeout_s)
            complete = baseline["complete"] and candidate["complete"]
            actual_delta = candidate["remaining_s"]-baseline["remaining_s"] if complete else None
            predicted = record.get("mean_delta_s")
            self.audits.append(dict(action_index=index, selected=record["selected"],
                candidate_index=option_index, public_virtual_time_s=policy.client.state.virtual_time_s,
                pending=asdict(policy.pending), candidate_actions=[asdict(a) for a in proposal],
                recorded_predicted_mean_delta_s=predicted,
                recorded_prediction_standard_error_s=record.get("standard_error_s"),
                recorded_paired_world_count=len(record.get("paired_delta_s", ())),
                baseline_actual_branch=baseline, candidate_actual_branch=candidate,
                complete=complete, actual_delta_s=actual_delta,
                prediction_error_s=actual_delta-predicted if complete and predicted is not None else None,
                actual_improves=actual_delta < -1e-6 if complete else None,
                wrong_sign=predicted < 0 < actual_delta if complete and predicted is not None else None))
        # Independent of all counterfactual outcomes: replay the logged choice.
        policy.override(proposal)


def audit_trace(trace, *, max_overrides=10, branch_timeout_s=10.0, replay_timeout_s=60.0):
    row = trace["row"]
    if row.get("strategy") not in {"R", "CR"}:
        raise ValueError("Only completed synthetic R/CR traces are supported")
    if not row.get("success") or not trace["search"].get("completion_certified"):
        raise ValueError("Audit requires a completed and certified trace")
    if str(row.get("suite_group", "")).startswith("reconstruction"):
        raise ValueError("Reconstructed validation worlds are outside this audit's scope")
    source = trace["scenario"]
    if "reconstruction" in source.get("description", "").lower():
        raise ValueError("Reconstructed validation worlds are outside this audit's scope")
    scenario = Scenario(**dict(source, sources=tuple(Source(**s) for s in source["sources"])))
    if scenario.problem != 3:
        raise ValueError("Only Q3 is supported")
    expected = trace["search"]["action_history"]
    decisions = trace.get("planner_stats", row.get("planner_stats", {})).get("decisions", [])
    planner = LoggedReplayPlanner(scenario, expected, decisions, max_overrides=max_overrides,
                                  branch_timeout_s=branch_timeout_s)
    engine = StressEngine(scenario)
    policy = StepperQ3(engine.client(), movable_tail=row["strategy"] == "CR")
    answer = dict(case_id=row["case_id"], strategy=row["strategy"], suite_group=row.get("suite_group"),
                  replay_valid=False, replay_error=None, audits=[],
                  physical_lower_bound_s=row.get("physical_lower_bound_s"),
                  guarantee_lower_bound_s=row.get("guarantee_lower_bound_s"),
                  recorded_total_time_s=row["virtual_time_s"],
                  recorded_time_over_physical_lower_bound=row.get("time_over_physical_lower_bound"),
                  recorded_time_over_guarantee_lower_bound=row.get("time_over_guarantee_lower_bound"))
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    try:
        policy.client.enter()
        deadline = wall_start + replay_timeout_s
        while policy.next_action() is not None:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Offline replay deadline exceeded")
            planner.maybe_choose(policy)
            index = len(policy.history)
            policy.execute_pending()
            assert_same_observation(policy.history[-1], expected[index], index)
        policy.client.exit()
        if len(policy.history) != len(expected):
            raise ReplayMismatch("Replay ended before the recorded trace")
        if planner.seen_decisions != set(planner.decisions):
            raise ReplayMismatch("Some recorded planner decisions were never encountered")
        if not policy.certified or not engine.evaluation()["all_cleared"]:
            raise ReplayMismatch("Replay did not clear and certify every source")
        if not math.isclose(policy.client.state.virtual_time_s, row["virtual_time_s"], abs_tol=1e-5, rel_tol=0):
            raise ReplayMismatch("Final replay total differs")
        answer["replay_valid"] = True
    except Exception as exc:
        answer["replay_error"] = f"{type(exc).__name__}: {exc}"
        engine.finish_for_evaluation("audit_replay_error")
    answer.update(audits=planner.audits, selected_overrides=planner.selected_count,
                  decisions_replayed=len(planner.seen_decisions),
                  cpu_s=time.process_time()-cpu_start, wall_s=time.perf_counter()-wall_start)
    for audit in answer["audits"]:
        audit["usable"] = bool(answer["replay_valid"] and audit["complete"])
        for name in ("baseline_actual_branch", "candidate_actual_branch"):
            branch = audit[name]
            if branch["complete"]:
                for label in ("physical", "guarantee"):
                    bound = row.get(f"{label}_lower_bound_s")
                    branch[f"full_time_over_{label}_lower_bound"] = (
                        branch["final_virtual_time_s"] / bound if bound else None)
    return answer


def summarize(reports):
    all_audits = [audit for report in reports for audit in report.get("audits", ())]
    usable = [a for a in all_audits if a["usable"]]
    return dict(traces=len(reports), valid_replays=sum(r.get("replay_valid", False) for r in reports),
        attempted_audits=len(all_audits), usable_audits=len(usable),
        incomplete_or_invalid_audits=len(all_audits)-len(usable),
        actual_improvements=sum(a["actual_improves"] for a in usable),
        wrong_sign_choices=sum(a["wrong_sign"] is True for a in usable),
        mean_predicted_delta_s=statistics.mean(a["recorded_predicted_mean_delta_s"] for a in usable) if usable else None,
        mean_actual_delta_s=statistics.mean(a["actual_delta_s"] for a in usable) if usable else None,
        selection="First selected overrides in caller-specified trace order; diagnostic sample, not a performance estimate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", required=True, help="Files, directories, or glob patterns")
    parser.add_argument("--out", type=Path, required=True, help="New output JSON path")
    parser.add_argument("--max-overrides", type=int, default=10, help="Total cap across all supplied traces")
    parser.add_argument("--branch-timeout-s", type=float, default=10.0)
    parser.add_argument("--replay-timeout-s", type=float, default=240.0)
    args = parser.parse_args()
    if args.max_overrides < 0 or min(args.branch_timeout_s, args.replay_timeout_s) <= 0:
        parser.error("Audit cap must be nonnegative and timeouts must be positive")
    if args.out.exists():
        parser.error("Output already exists; choose a new path")
    paths = []
    for value in args.traces:
        path = Path(value)
        found = sorted(path.glob("trace-*.json.gz")) if path.is_dir() else [Path(p) for p in sorted(glob.glob(value))]
        if not found:
            parser.error("A supplied trace path or pattern has no matches")
        paths.extend(p for p in found if p not in paths)
    reports, attempted = [], 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if attempted >= args.max_overrides:
            break
        raw = gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
        try:
            trace = json.loads(raw)
            report = audit_trace(trace, max_overrides=args.max_overrides-attempted,
                branch_timeout_s=args.branch_timeout_s, replay_timeout_s=args.replay_timeout_s)
        except Exception as exc:
            report = dict(replay_valid=False, replay_error=f"{type(exc).__name__}: {exc}", audits=[])
        report.update(trace_name=path.name, trace_sha256=hashlib.sha256(raw).hexdigest())
        reports.append(report)
        attempted += len(report["audits"])
        payload = dict(scope="posthoc_completed_synthetic_traces_only", online_or_fitting_use=False,
                       status="running", summary=summarize(reports), reports=reports)
        temporary = args.out.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.replace(args.out)
        print(json.dumps({key: report.get(key) for key in ("case_id", "strategy", "replay_valid", "replay_error")}), flush=True)
    payload = dict(scope="posthoc_completed_synthetic_traces_only", online_or_fitting_use=False,
                   status="completed", summary=summarize(reports), reports=reports)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
