"""Read explicitly named local Q4 training batches; never run a policy episode.

No file discovery, database access, training, network or model selection occurs.
Checkpoint inference is optional and uses only saved public feature tensors.
Legacy batches without action/evaluation evidence are reported as incomplete.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
COST_UNIT_S = 1000.0
FAILURE_PENALTY_S = 360000.0
TOLERANCE_S = 2e-5


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    point = (len(ordered) - 1) * p
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def numeric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric field")
    return float(value)


def assert_close(a, b, message):
    if not math.isclose(numeric(a), numeric(b), rel_tol=0, abs_tol=TOLERANCE_S):
        raise ValueError(message)


def feature_key(features):
    """Exactly the information seen by the float32 model, without channel IDs."""
    if len(features) != 16:
        raise ValueError("Unsupported candidate feature dimension")
    return struct.pack("!16f", *(numeric(value) for value in features))


def explained_variance(targets, predictions):
    if len(targets) != len(predictions):
        raise ValueError("Critic target/prediction count mismatch")
    if len(targets) < 2:
        return {"count": len(targets), "value": None, "reason": "fewer_than_two_values"}
    variance = statistics.pvariance(targets)
    if variance <= 1e-12:
        return {"count": len(targets), "value": None, "reason": "constant_return_target"}
    residual = [target - prediction for target, prediction in zip(targets, predictions)]
    return {"count": len(targets), "value": 1 - statistics.pvariance(residual) / variance,
            "target_variance": variance, "reason": None}


def audit_history(episode, actual_time, fallback_cost):
    """Independently replay billing and clear outcomes from saved requests only."""
    history = episode.get("observation_history")
    evaluation = episode.get("evaluation")
    if history is None or evaluation is None:
        return {"status": "missing_evidence", "missing": [name for name in
            ("observation_history", "evaluation") if episode.get(name) is None]}
    truth = evaluation.get("ground_truth")
    if episode.get("evaluation_phase") != "after_policy_termination" or not truth:
        raise ValueError("Raw truth lacks post-termination provenance")
    if episode.get("scenario") is not None and episode["scenario"] != truth:
        raise ValueError("Saved scene differs from evaluation ground truth")
    sources = {source["channel"]: source for source in truth["sources"]}
    components, policy_components, fallback_components = Counter(), Counter(), Counter()
    cleared, missed = set(), 0
    pos, channel, previous_time = (0., 0.), 1, 0.
    fallback_start = actual_time - fallback_cost
    distance_total = 0.
    operations = Counter()
    operation_components = defaultdict(Counter)
    policy_distance, fallback_distance = 0., 0.
    cumulative_times = {0}
    physical_requests = []
    for item in history:
        response = item["response"]
        if response.get("accepted") is not True:
            raise ValueError("Observation history includes an unaccepted action")
        operation, target_channel = item["action"], item["channel"]
        target = (numeric(item["position"]["x"]), numeric(item["position"]["y"]))
        step = Counter()
        distance = 0.
        if operation in ("/measure", "/clear"):
            physical_requests.append(item)
            distance = math.dist(pos, target)
            distance_total += distance
            step["movement_s"] = round(distance / 5 * 1e6)
            if operation == "/measure":
                step["switching_s"] = int(target_channel != channel) * 1000000
                step["detection_s"] = 5000000
                channel = target_channel
            else:
                source = sources.get(target_channel)
                actual_success = (source is not None and target_channel not in cleared and
                    math.dist((source["x"], source["y"]), target) <= 20.)
                if actual_success != (response.get("clear_result") == "success"):
                    raise ValueError("Recorded clear outcome disagrees with actual source distance")
                step["optical_s"], step["removal_s"] = 3000000, int(actual_success) * 2000000
                if actual_success:
                    cleared.add(target_channel)
                else:
                    missed += 1
            pos = target
        elif operation not in ("/enter", "/exit"):
            raise ValueError("Unknown physical operation")
        components.update(step)
        operation_components[operation].update(step)
        current_time = sum(components.values()) / 1e6
        cumulative_times.add(round(current_time * 1e6))
        assert_close(current_time, response["virtual_time_s"], "Per-request billed time mismatch")
        if fallback_cost and current_time > fallback_start + TOLERANCE_S:
            if previous_time < fallback_start - TOLERANCE_S:
                raise ValueError("Fallback boundary cuts through an indivisible physical action")
            fallback_components.update(step)
            fallback_distance += distance
        else:
            policy_components.update(step)
            policy_distance += distance
        previous_time = current_time
        operations[operation] += 1
    assert_close(sum(components.values()) / 1e6, actual_time, "History does not cover complete billed time")
    assert_close(sum(fallback_components.values()) / 1e6, fallback_cost, "Fallback physical ledger mismatch")
    assert_close(evaluation["virtual_time_s"], actual_time, "Evaluation billed time mismatch")
    if bool(evaluation["all_cleared"]) != (cleared == set(sources)):
        raise ValueError("Evaluation all-clear flag disagrees with real cleared source set")
    if evaluation["source_total"] != len(sources) or evaluation["cleared_total"] != len(cleared):
        raise ValueError("Evaluation source counts disagree with real cleared source set")
    if (evaluation["cleared_channels"] != sorted(cleared) or
            evaluation["remaining_channels"] != sorted(set(sources) - cleared)):
        raise ValueError("Evaluation remaining/cleared channel ledger mismatch")
    if evaluation["action_count"] != len(history) or evaluation["measurement_count"] != operations["/measure"]:
        raise ValueError("Evaluation physical action counts mismatch")
    if missed != evaluation["failed_clear_count"]:
        raise ValueError("Evaluation failed-clear count mismatch")
    for name in COMPONENTS:
        assert_close(components[name] / 1e6, evaluation["time_breakdown_s"][name], "Evaluation cost component mismatch")
    phases = Counter()
    certified_failures = 0
    trace = episode.get("action_history")
    if trace is None or len(trace) != len(physical_requests):
        raise ValueError("Strategy action history does not cover all physical requests")
    for action, request in zip(trace, physical_requests):
        response = request["response"]
        if ("/" + action["action"] != request["action"] or action["channel"] != request["channel"] or
                action["result"] != response.get("measure_result", response.get("clear_result"))):
            raise ValueError("Strategy action trace disagrees with physical request/outcome")
        assert_close(action["position"][0], request["position"]["x"], "Strategy/physical x mismatch")
        assert_close(action["position"][1], request["position"]["y"], "Strategy/physical y mismatch")
        assert_close(action["virtual_time_s"], response["virtual_time_s"], "Strategy/physical time mismatch")
        if action["action"] == "clear" and action.get("result") == "no_target_in_range":
            phases[action.get("phase", "unclassified")] += 1
            certified_failures += action.get("phase") in ("certified_clear", "near_clear")
    if certified_failures:
        raise ValueError("A near/certified clear failed")
    if sum(phases.values()) != missed:
        raise ValueError("Strategy action history omits physical failed clears")
    boundary = 0.
    for record in episode.get("records", []):
        boundary += record["cost_s"] - record.get("fallback_cost_s", 0.)
        if not any(round(boundary * 1e6) + delta in cumulative_times for delta in (-1, 0, 1)):
            raise ValueError("A transition boundary is not a physical action boundary")
    return {"status": "passed", "actual_all_cleared": cleared == set(sources),
            "source_total": len(sources), "cleared_total": len(cleared), "failed_clear_count": missed,
            "failed_clear_phases": dict(phases), "certified_clear_failures": certified_failures,
            "operations": dict(operations), "actual_distance_m": distance_total,
            "policy_distance_m": policy_distance, "fallback_distance_m": fallback_distance,
            "channel_switches": components["switching_s"] / 1e6,
            "operation_components_s": {op: {name: value / 1e6 for name, value in parts.items()}
                                       for op, parts in operation_components.items()},
            "components_s": {name: components[name] / 1e6 for name in COMPONENTS},
            "policy_components_s": {name: policy_components[name] / 1e6 for name in COMPONENTS},
            "fallback_components_s": {name: fallback_components[name] / 1e6 for name in COMPONENTS},
            "scope": "saved physical billing and actual clears; no new continuous coverage certificate proof"}


def model_diagnostics(records, model, *, bc_labels=False):
    if model is None:
        return None
    import torch
    from torch.distributions import Categorical
    from q4_rl.network import pack_observations
    entropy, normalized, values, targets = [], [], [], []
    strict_hits = equivalent_hits = ties = 0
    with torch.no_grad():
        for offset in range(0, len(records), 32):
            subset = records[offset:offset + 32]
            logits, critic = model(*pack_observations(subset, global_dim=model.global_dim,
                                                     candidate_dim=model.candidate_dim))
            distribution = Categorical(logits=logits)
            entropies, choices = distribution.entropy().tolist(), logits.argmax(-1).tolist()
            for row, value, ent, choice in zip(subset, critic.tolist(), entropies, choices):
                count = len(row["candidate_features"])
                entropy.append(ent)
                if count > 1:
                    normalized.append(ent / math.log(count))
                if "return" in row:
                    values.append(value)
                    targets.append(row["return"])
                if bc_labels:
                    label = row["action_index"]
                    key = feature_key(row["candidate_features"][label])
                    equivalent = [i for i, feature in enumerate(row["candidate_features"]) if feature_key(feature) == key]
                    strict_hits += choice == label
                    equivalent_hits += choice in equivalent
                    ties += len(equivalent) > 1
    return {"states": len(records), "mean_entropy_nats": statistics.mean(entropy) if entropy else None,
            "mean_entropy_over_log_legal_candidates": statistics.mean(normalized) if normalized else None,
            "normalized_entropy_states": len(normalized),
            "critic_explained_variance": explained_variance(targets, values),
            "bc_label_states": len(records) if bc_labels else 0,
            "bc_exact_index_hits": strict_hits if bc_labels else None,
            "bc_feature_equivalent_hits": equivalent_hits if bc_labels else None,
            "bc_states_with_feature_equivalent_ties": ties if bc_labels else None,
            "scope": "the explicitly supplied checkpoint on saved states; not automatically the behavior checkpoint",
            "critic_targets": targets, "critic_values": values}


def analyze_episode(episode, *, batch_label, model=None, bc_labels=False, supplemental_bounds=()):
    records, metrics = episode.get("records", []), episode.get("metrics", {})
    row = {"batch": batch_label, "seed": episode.get("seed"), "administrative_skip": episode.get("administrative_skip"),
           "checks": {}, "errors": [], "warnings": []}
    if not metrics:
        row["warnings"].append("No completed episode metrics; administrative attempt retained")
        row["checks"]["billing"] = "unavailable"
        return row
    actual = numeric(metrics["actual_time_s"])
    success = metrics.get("success")
    if type(success) is not bool:
        raise ValueError("Training metrics require Boolean success, not a guessed alternative field")
    penalty = actual if success else max(actual, FAILURE_PENALTY_S)
    costs = [numeric(record["cost_s"]) for record in records]
    if any(cost < 0 for cost in costs):
        raise ValueError("Negative billed step cost")
    assert_close(sum(costs), actual, "Transitions do not conserve entire episode time")
    assert_close(metrics["penalized_time_s"], penalty, "Failure penalty inconsistent")
    assert_close(metrics["reward_cost_s"], penalty, "Reward cost differs from total billed objective")
    fallback = numeric(metrics.get("fallback_cost_s", 0.))
    if not 0 <= fallback <= actual + TOLERANCE_S:
        raise ValueError("Invalid fallback share")
    assert_close(sum(numeric(record.get("fallback_cost_s", 0.)) for record in records), fallback,
                 "Transition fallback costs differ from episode metric")
    remaining = penalty - actual
    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        terminal_penalty = penalty - actual if index == len(records) - 1 else 0.
        assert_close(record.get("terminal_penalty_s", 0.), terminal_penalty, "Incorrect terminal penalty allocation")
        remaining += costs[index]
        assert_close(record["return"], -remaining / COST_UNIT_S, "Return is not undiscounted complete billed time")
    row.update(actual_time_s=actual, penalized_time_s=penalty, reported_success=success,
               failed_clear_count=metrics["failed_clear_count"], fallback_cost_s=fallback,
               fallback_share=fallback / actual if actual else 0., decisions=len(records),
               family=metrics.get("family"), source_mode=metrics.get("source_mode"))
    row["compute"] = {key: numeric(metrics[key]) for key in
        ("wall_time_s", "worker_cpu_s", "policy_wall_s", "policy_cpu_s", "posthoc_bound_wall_s", "posthoc_bound_cpu_s")
        if metrics.get(key) is not None}
    row["checks"]["transition_billing_and_return"] = "passed"
    row["physical_audit"] = audit_history(episode, actual, fallback)
    if row["physical_audit"]["status"] != "passed":
        row["warnings"].append("Actual all-clear and physical ledger cannot be independently verified without legacy missing fields")
    elif row["physical_audit"]["failed_clear_count"] != metrics["failed_clear_count"]:
        raise ValueError("Reported miss count differs from physical history")
    elif success and not row["physical_audit"]["actual_all_cleared"]:
        raise ValueError("Reported success lacks actual all-clear")
    lower = metrics.get("common_lower_bound_s")
    lower_source = "saved_training_diagnostic"
    if lower is None:
        matches = [value for value in supplemental_bounds if value["seed"] == episode["seed"] and
                   ("family" not in value or value["family"] == metrics.get("family")) and
                   ("source_mode" not in value or value["source_mode"] == metrics.get("source_mode")) and
                   math.isclose(value["actual_time_s"], actual, rel_tol=0, abs_tol=TOLERANCE_S)]
        if len(matches) > 1:
            raise ValueError("Ambiguous explicit supplemental lower-bound rows")
        if matches:
            lower = matches[0]["common_lower_bound_s"]
            lower_source = "explicit_report_only_supplement"
    lower = numeric(lower) if lower is not None else None
    if lower is not None and lower <= 0:
        raise ValueError("Nonpositive lower bound")
    if episode.get("common_bound") is not None:
        bound = episode["common_bound"]
        if bound["version"] != "q4-common-source-edge-v1":
            raise ValueError("Unexpected lower-bound definition")
        assert_close(bound["common_lower_bound_s"], lower, "Saved bound/metric inconsistency")
        assert_close(bound["movement_lower_bound_s"] + bound["successful_clear_action_lower_s"] +
                     bound["empty_channel_action_lower_s"], lower, "Lower-bound components inconsistent")
        truth = episode.get("evaluation", {}).get("ground_truth")
        if truth is not None:
            identity = digest_bytes(json.dumps(truth, sort_keys=True, ensure_ascii=False,
                                              allow_nan=False, separators=(",", ":")).encode())
            if bound["case_sha256"] != identity:
                raise ValueError("Lower bound is not bound to the saved post-termination scene")
            count = len(truth["sources"])
            assert_close(bound["successful_clear_action_lower_s"], 5. * count, "Successful-clear lower bound mismatch")
            assert_close(bound["empty_channel_action_lower_s"], 50. * (20 - count) if count < 16 else 0.,
                         "Empty-channel lower-bound premise mismatch")
        assert_close(bound["movement_lower_bound_s"], bound["lower_route_length_m"] / 5.,
                     "Movement lower-bound units mismatch")
        if success and actual + TOLERANCE_S < lower:
            raise ValueError("Successful actual time violates the saved common lower bound")
    verified_clear = row["physical_audit"].get("status") == "passed" and row["physical_audit"].get("actual_all_cleared")
    row.update(common_lower_bound_s=lower, lower_bound_source=lower_source if lower else "missing",
        reported_success_time_over_lower_bound=actual / lower if lower and success else None,
        actual_time_over_lower_bound=actual / lower if lower and success and verified_clear else None,
        penalized_time_over_lower_bound=penalty / lower if lower else None)
    if lower is None:
        row["warnings"].append("No lower bound in the explicitly provided files; no scene was generated to fill it")
    action_counts, action_costs, proposed_distances, proposed_switches = Counter(), Counter(), Counter(), Counter()
    behavior_targets, behavior_values, surprisals = [], [], []
    for record in records:
        features = record["candidate_features"]
        if len(record["global_features"]) != 10 or not features:
            raise ValueError("Unsupported or empty public feature schema")
        for value in record["global_features"]:
            numeric(value)
        for feature in features:
            feature_key(feature)
        chosen = record["action_index"]
        if type(chosen) is not int or not 0 <= chosen < len(features):
            raise ValueError("Chosen action outside saved legal candidate set")
        feature = features[chosen]
        kind = "measure" if feature[0] == 1 and feature[1] == 0 else "service" if feature[1] == 1 and feature[0] == 0 else None
        if kind is None or record.get("action_kind", kind) != kind:
            raise ValueError("Recorded action kind disagrees with public features")
        action_counts[kind] += 1
        own_cost = record["cost_s"] - record.get("fallback_cost_s", 0.)
        if own_cost < -TOLERANCE_S:
            raise ValueError("Fallback cost exceeds the transition billed cost")
        action_costs[kind] += own_cost
        proposed_distances[kind] += feature[4] * 3600.
        proposed_switches[kind] += int(kind == "measure" and feature[7] == 0)
        if "log_prob" in record:
            if bc_labels:
                raise ValueError("Actor log-probabilities conflict with explicit BC-label batch designation")
            surprisals.append(-numeric(record["log_prob"]))
        if "value" in record:
            behavior_values.append(numeric(record["value"]))
            behavior_targets.append(numeric(record["return"]))
    row["decision_contributions"] = {"counts": dict(action_counts), "billed_cost_excluding_fallback_s": dict(action_costs),
        "proposed_entry_distance_m": dict(proposed_distances), "proposed_measure_switches": sum(proposed_switches.values()),
        "distance_scope": "candidate entry distance; service internals may travel further, actual distance is in physical_audit"}
    row["recorded_behavior"] = {"critic_explained_variance": explained_variance(behavior_targets, behavior_values),
        "mean_sampled_surprisal_nats": statistics.mean(surprisals) if surprisals else None,
        "exact_entropy": None, "entropy_scope": "a chosen-action log-probability does not determine exact policy entropy",
        "critic_targets": behavior_targets, "critic_values": behavior_values, "surprisals": surprisals}
    row["checkpoint_diagnostics"] = model_diagnostics(records, model, bc_labels=bc_labels)
    row["checks"]["public_action_features"] = "passed"
    return row


def summary_rows(rows):
    available = [row for row in rows if "actual_time_s" in row]
    result = {"attempts": len(rows), "administratively_interrupted": sum(bool(row.get("administrative_skip")) for row in rows),
              "available_episode_metrics": len(available), "analysis_error_episodes": sum(bool(row["errors"]) for row in rows)}
    if result["analysis_error_episodes"] or any("actual_time_s" not in row and not row.get("administrative_skip") for row in rows):
        result["aggregation_status"] = "withheld_incomplete_or_inconsistent_batch"
        result["reason"] = "No means or ratios over only the successfully parsed episodes; all raw-reported error metrics remain in episode rows"
        return result
    result["aggregation_status"] = "all_completed_episode_metrics_included"
    if not available:
        return result
    actual = [row["actual_time_s"] for row in available]
    penalized = [row["penalized_time_s"] for row in available]
    bounds = [row["common_lower_bound_s"] for row in available]
    bound_complete = all(value is not None for value in bounds)
    result.update(reported_full_clear=sum(row["reported_success"] for row in available),
        reported_task_failures=sum(not row["reported_success"] for row in available),
        reported_full_clear_rate=sum(row["reported_success"] for row in available) / len(available),
        physically_verified_full_clear=sum(row.get("physical_audit", {}).get("actual_all_cleared") is True for row in available),
        physical_evidence_missing=sum(row.get("physical_audit", {}).get("status") == "missing_evidence" for row in available),
        mean_actual_time_s=statistics.mean(actual), p95_actual_time_s=percentile(actual, .95), max_actual_time_s=max(actual),
        mean_penalized_time_s=statistics.mean(penalized), p95_penalized_time_s=percentile(penalized, .95),
        failed_clear_count=sum(row["failed_clear_count"] for row in available),
        sum_fallback_cost_s=sum(row["fallback_cost_s"] for row in available),
        fallback_share_of_total_actual_time=sum(row["fallback_cost_s"] for row in available) / sum(actual) if sum(actual) else 0.,
        mean_lower_bound_s=statistics.mean(bounds) if bound_complete else None,
        ratio_of_mean_penalized_time_to_mean_bound=sum(penalized) / sum(bounds) if bound_complete else None,
        actual_all_clear_ratio_of_sums=sum(actual) / sum(bounds) if bound_complete and all(
            row["reported_success"] and row.get("physical_audit", {}).get("status") == "passed" and
            row["physical_audit"].get("actual_all_cleared") for row in available) else None,
        lower_bound_rows_missing=sum(value is None for value in bounds))
    verified = [row["physical_audit"] for row in available if row.get("physical_audit", {}).get("status") == "passed"]
    if len(verified) == len(available):
        result["physical"] = {"actual_full_clear_rate": sum(item["actual_all_cleared"] for item in verified) / len(verified),
            "actual_task_failures": sum(not item["actual_all_cleared"] for item in verified),
            "actual_distance_m": sum(item["actual_distance_m"] for item in verified),
            "policy_distance_m": sum(item["policy_distance_m"] for item in verified),
            "fallback_distance_m": sum(item["fallback_distance_m"] for item in verified),
            "channel_switches": sum(item["channel_switches"] for item in verified),
            "certified_clear_failures": sum(item["certified_clear_failures"] for item in verified)}
        for key in ("components_s", "policy_components_s", "fallback_components_s", "failed_clear_phases", "operations"):
            combined = Counter()
            for item in verified:
                combined.update(item[key])
            result["physical"][key] = dict(combined)
        by_operation = defaultdict(Counter)
        for item in verified:
            for operation, parts in item["operation_components_s"].items():
                by_operation[operation].update(parts)
        result["physical"]["operation_components_s"] = {key: dict(value) for key, value in by_operation.items()}
    else:
        result["physical"] = None
    decision = {}
    for field in ("counts", "billed_cost_excluding_fallback_s", "proposed_entry_distance_m"):
        combined = Counter()
        for row in available:
            combined.update(row["decision_contributions"][field])
        decision[field] = dict(combined)
    decision["proposed_measure_switches"] = sum(row["decision_contributions"]["proposed_measure_switches"] for row in available)
    result["decisions"] = decision
    result["compute"] = {"scope": "sum of per-episode worker times, not elapsed batch makespan or PPO optimizer CPU; posthoc bound included"}
    for field in ("wall_time_s", "worker_cpu_s", "policy_wall_s", "policy_cpu_s", "posthoc_bound_wall_s", "posthoc_bound_cpu_s"):
        values = [row["compute"][field] for row in available if field in row["compute"]]
        result["compute"][field] = {"available_episodes": len(values), "sum": sum(values) if values else None,
                                    "mean": statistics.mean(values) if values else None}
    for field in ("recorded_behavior", "checkpoint_diagnostics"):
        diagnostics = [row[field] for row in available if row.get(field)]
        targets = [value for item in diagnostics for value in item.get("critic_targets", [])]
        values = [value for item in diagnostics for value in item.get("critic_values", [])]
        combined = {"critic_explained_variance": explained_variance(targets, values)}
        if field == "recorded_behavior":
            surprisals = [value for item in diagnostics for value in item.get("surprisals", [])]
            combined["mean_sampled_surprisal_nats"] = statistics.mean(surprisals) if surprisals else None
        else:
            count = sum(item["states"] for item in diagnostics)
            bc_count = sum(item["bc_label_states"] for item in diagnostics)
            normalized_count = sum(item["normalized_entropy_states"] for item in diagnostics)
            combined.update(states=count, mean_entropy_nats=sum(item["mean_entropy_nats"] * item["states"] for item in diagnostics if item["states"]) / count if count else None,
                mean_entropy_over_log_legal_candidates=sum(item["mean_entropy_over_log_legal_candidates"] * item["normalized_entropy_states"] for item in diagnostics if item["normalized_entropy_states"]) / normalized_count if normalized_count else None,
                bc_label_states=bc_count,
                bc_exact_index_accuracy=sum(item["bc_exact_index_hits"] or 0 for item in diagnostics) / bc_count if bc_count else None,
                bc_feature_equivalent_accuracy=sum(item["bc_feature_equivalent_hits"] or 0 for item in diagnostics) / bc_count if bc_count else None,
                bc_states_with_feature_equivalent_ties=sum(item["bc_states_with_feature_equivalent_ties"] or 0 for item in diagnostics))
        result[field] = combined
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append", required=True,
                        help="Explicit local batch path; repeat to include every requested batch")
    parser.add_argument("--bc-batch", type=Path, action="append", default=[],
                        help="An explicitly included batch known to contain heuristic BC labels")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bounds", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in args.batch]
    bc_paths = {path.resolve() for path in args.bc_batch}
    if len(set(paths)) != len(paths) or not bc_paths <= set(paths):
        parser.error("No duplicate batches; each BC-label batch must also be explicitly included")
    if args.output.exists():
        parser.error("Choose a new report output file")
    started, cpu_started = time.perf_counter(), time.process_time()
    analyzer_identity = {"name": Path(__file__).name, "sha256": digest_bytes(Path(__file__).read_bytes())}
    inputs, bounds = [], []
    for path in args.bounds:
        raw = path.read_bytes()
        inputs.append({"kind": "supplemental_bounds", "name": path.name, "sha256": digest_bytes(raw)})
        bounds.extend(json.loads(raw)["rows"])
    model = None
    checkpoint_identity = None
    if args.checkpoint:
        from q4_rl.network import load_policy
        raw = args.checkpoint.read_bytes()
        checkpoint_identity = {"name": args.checkpoint.name, "sha256": digest_bytes(raw)}
        model = load_policy(args.checkpoint).model
        model.eval()
    rows = []
    for index, path in enumerate(paths):
        raw = path.read_bytes()
        label = f"input-{index:03d}/{path.name}"
        inputs.append({"kind": "training_batch", "name": label, "sha256": digest_bytes(raw), "bytes": len(raw),
                       "explicit_heuristic_bc_labels": path in bc_paths})
        episodes = json.loads(gzip.decompress(raw) if path.name.endswith(".gz") else raw)
        if not isinstance(episodes, list):
            raise ValueError("Training batch must be a complete episode list")
        for episode in episodes:
            try:
                row = analyze_episode(episode, batch_label=label, model=model,
                                      bc_labels=path in bc_paths, supplemental_bounds=bounds)
            except Exception as exc:
                row = {"batch": label, "seed": episode.get("seed"), "checks": {}, "warnings": [],
                       "errors": [{"type": type(exc).__name__, "detail": str(exc)}],
                       "reported_metrics": episode.get("metrics", {})}
            rows.append(row)
    report = {"schema": "q4-local-training-review-v1", "analyzer": analyzer_identity,
              "inputs": inputs, "checkpoint": checkpoint_identity,
              "summary": summary_rows(rows), "by_batch": {label: summary_rows([row for row in rows if row["batch"] == label])
                  for label in dict.fromkeys(row["batch"] for row in rows)}, "episodes": rows,
              "by_stratum": {label: summary_rows([row for row in rows if f"{row.get('family')}/{row.get('source_mode')}" == label])
                  for label in dict.fromkeys(f"{row.get('family')}/{row.get('source_mode')}" for row in rows)},
              "scope": "diagnosis of all episodes in explicitly named training batches; no independent efficacy or model-selection claim",
              "bc_equivalence": "exact equality of all 16 public candidate features after float32 conversion; no channel identifiers",
              "limitations": ["Missing raw evidence is reported, not regenerated or fabricated.",
                  "Recorded bounds are checked for consistency but their route DP is not reoptimized.",
                  "Checkpoint entropy is on logged states; it is not necessarily behavior-policy entropy.",
                  "Training trajectories change with the policy and are correlated within batches; these descriptive summaries do not establish independent generalization or paired improvement.",
                  "No database, remote access, training, simulator rollout or case filtering occurred."],
              "analysis_wall_s": time.perf_counter() - started,
              "analysis_cpu_s": time.process_time() - cpu_started}
    # Keep JSON reasonably compact while computing global EV from every scalar.
    for row in rows:
        for field in ("recorded_behavior", "checkpoint_diagnostics"):
            if row.get(field):
                for key in ("critic_targets", "critic_values", "surprisals"):
                    row[field].pop(key, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"report": args.output.name, "summary": report["summary"]}, allow_nan=False))
    return 1 if any(row["errors"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
