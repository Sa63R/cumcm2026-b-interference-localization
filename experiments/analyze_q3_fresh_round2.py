"""Independent trace accounting and grouped paired analysis for Q3 round two.

Reads completed offline batches only. No policy, simulator, sampling, or runner
module is imported. Truth is used solely in this evaluator-side audit. The
default excludes rollout_smoke; an explicitly requested smoke batch is placed
in its own pilot group. Conflicting duplicate runs are preserved in the CSV and
conflict list, but are not silently chosen for performance summaries.
"""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ROOT = ROOT / "results" / "q3_fresh_round2"
TIME_TOLERANCE_S = 1e-5
PLANNING_BUDGET_TOLERANCE_S = 0.25  # Cooperative checks can finish one short operation.
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
BOUND_LABELS = ("physical", "certified", "guarantee", "observation_robust_physical",
                "observation_robust_certified", "observation_robust_guarantee")


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def _point(value):
    p = (value["x"], value["y"]) if isinstance(value, dict) else tuple(value)
    if len(p) != 2 or any(isinstance(v, bool) or not math.isfinite(v) for v in p):
        raise ValueError("Invalid finite position")
    return tuple(0.0 if v == 0 else float(v) for v in p)


def audit_trace(trace):
    """Recompute charged actions and check public feedback against trace truth.

    Phase labels never enter the cost calculation. This checks the reported
    completion certificate's consistency, not an independent re-proof of the
    strategy's continuous coverage theorem. Rollout world IDs/action logs were
    not persisted, so common-world reuse cannot be independently certified.
    """
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def close(actual, expected, message, tolerance=TIME_TOLERANCE_S):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and abs(actual-expected) <= tolerance, message)

    row, evaluation, search = trace["row"], trace["evaluation"], trace["search"]
    scenario, observations = trace["scenario"], trace["observations"]
    sources = {s["channel"]: s for s in scenario["sources"]}
    require(scenario["problem"] == 3, "not_question_three")
    require(10 <= len(sources) == len(scenario["sources"]) <= 16, "invalid_source_count")
    require(row["scenario_sha256"] == fingerprint(scenario), "scenario_hash_mismatch")
    require(row["case_id"] == scenario["case_id"] == evaluation["case_id"], "case_id_mismatch")
    for source in sources.values():
        require(math.hypot(source["x"], source["y"]) <= 1800+1e-6, "source_outside_arena")
        require(1000 <= source["reception_radius_m"] <= 1500, "source_radius_out_of_range")
        require(source.get("orientation_deg") is None, "directional_source_in_question_three")
    component_us = dict.fromkeys(COMPONENTS, 0)
    position, measuring_channel = (0.0, 0.0), 1
    cleared, fixed_feedback = set(), {}
    measurement_count = failed_clear_count = 0
    raw_movement_s, accepted, entered, exited = 0.0, [], False, False
    for index, item in enumerate(observations):
        response, action = item["response"], item["action"]
        require(response.get("accepted") is True, f"action_{index}:not_accepted")
        require(not exited, f"action_{index}:action_after_exit")
        if action == "/enter":
            require(index == 0 and not entered, "enter_position_or_multiplicity")
            entered = True
        elif action == "/exit":
            require(entered, f"action_{index}:exit_before_enter")
            exited = True
        elif action in {"/measure", "/clear"}:
            require(entered, f"action_{index}:action_before_enter")
            p, channel = _point(item["position"]), item["channel"]
            require(isinstance(channel, int) and not isinstance(channel, bool)
                    and 1 <= channel <= 20, f"action_{index}:invalid_channel")
            move = math.dist(position, p)/5.0
            raw_movement_s += move
            component_us["movement_s"] += round(move*1_000_000)
            position = p
            source = sources.get(channel)
            alive = source is not None and channel not in cleared
            d = math.dist((source["x"], source["y"]), p) if alive else math.inf
            if action == "/measure":
                measurement_count += 1
                component_us["detection_s"] += 5_000_000
                component_us["switching_s"] += int(channel != measuring_channel)*1_000_000
                measuring_channel = channel
                result = response["measure_result"]
                if result == "no_signal":
                    require(not alive or d > source["reception_radius_m"],
                            f"action_{index}:no_signal_contradicts_reception_radius")
                elif result == "near":
                    require(alive and d <= 5+1e-7, f"action_{index}:near_contradicts_truth")
                elif result == "direction":
                    require(alive and 5 < d <= source["reception_radius_m"]+1e-7,
                            f"action_{index}:direction_contradicts_reception")
                    if alive:
                        theta = math.degrees(math.atan2(source["y"]-p[1], source["x"]-p[0])) % 360
                        error = abs((response["svd_deg"]-theta+180) % 360-180)
                        require(error <= 1.005+1e-8, f"action_{index}:bearing_exceeds_1_005_degrees")
                else:
                    errors.append(f"action_{index}:unknown_measure_result")
                if channel not in cleared:
                    key = (channel, p)
                    feedback = (result, response.get("svd_deg"))
                    require(key not in fixed_feedback or fixed_feedback[key] == feedback,
                            f"action_{index}:fixed_coordinate_feedback_changed")
                    fixed_feedback[key] = feedback
                accepted.append(dict(action="measure", position=p, channel=channel, result=result,
                                     bearing_deg=response.get("svd_deg"), virtual_time_s=response["virtual_time_s"]))
            else:
                result = response["clear_result"]
                success = result == "success"
                require(result in {"success", "no_target_in_range"}, f"action_{index}:unknown_clear_result")
                require(success == (alive and d <= 20), f"action_{index}:clear_contradicts_truth")
                component_us["optical_s"] += 3_000_000
                component_us["removal_s"] += int(success)*2_000_000
                if success:
                    cleared.add(channel)
                else:
                    failed_clear_count += 1
                accepted.append(dict(action="clear", position=p, channel=channel, result=result,
                                     bearing_deg=None, virtual_time_s=response["virtual_time_s"]))
            # /clear deliberately does not change the measuring channel.
        else:
            errors.append(f"action_{index}:unknown_action")
        close(response["virtual_time_s"], sum(component_us.values())/1_000_000,
              f"action_{index}:cumulative_cost_mismatch")
    history = search.get("action_history", [])
    require(len(history) == len(accepted), "public_history_length_mismatch")
    for index, (reported, actual) in enumerate(zip(history, accepted)):
        for key in ("action", "channel", "result"):
            require(reported[key] == actual[key], f"history_{index}:{key}_mismatch")
        require(_point(reported["position"]) == actual["position"], f"history_{index}:position_mismatch")
        if actual["result"] == "direction":
            close(reported["bearing_deg"], actual["bearing_deg"], f"history_{index}:bearing_mismatch", 1e-10)
        close(reported["virtual_time_s"], actual["virtual_time_s"], f"history_{index}:time_mismatch")
    total = sum(component_us.values())/1_000_000
    for source_name, data in (("row", row), ("evaluation", evaluation)):
        close(data["virtual_time_s"], total, f"{source_name}:total_cost_mismatch")
        for key, value in component_us.items():
            close(data["time_breakdown_s"][key], value/1_000_000, f"{source_name}:{key}_mismatch")
        require(data["source_total"] == len(sources), f"{source_name}:source_count_mismatch")
        require(data["cleared_total"] == len(cleared), f"{source_name}:cleared_total_mismatch")
        require(set(data["cleared_channels"]) == cleared, f"{source_name}:cleared_set_mismatch")
        require(set(data["remaining_channels"]) == set(sources)-cleared, f"{source_name}:remaining_set_mismatch")
        require(bool(data["all_cleared"]) == (cleared == set(sources)), f"{source_name}:all_cleared_mismatch")
        require(data["measurement_count"] == measurement_count, f"{source_name}:measurement_count_mismatch")
        require(data["failed_clear_count"] == failed_clear_count, f"{source_name}:failed_clear_count_mismatch")
        require(data["action_count"] == len(observations), f"{source_name}:action_count_mismatch")
    close(row["time_per_source_s"], total/len(sources), "time_per_source_mismatch")
    claimed_certificate = bool(search.get("completion_certified"))
    require(claimed_certificate == bool(row["completion_certified"]), "certificate_flag_mismatch")
    reported_channels = {int(k): v for k, v in search.get("channels", {}).items()}
    require({c for c, state in reported_channels.items() if state == "cleared"} == cleared,
            "policy_cleared_set_mismatch")
    if claimed_certificate:
        require(len(reported_channels) == 20 and all(v in {"absent", "cleared"} for v in reported_channels.values()),
                "certificate_has_unresolved_channels")
    success = bool(row.get("error") is None and cleared == set(sources) and claimed_certificate and exited)
    require(bool(row["success"]) == success, "success_flag_mismatch")
    require(bool(row["correct_exit"]) == success, "correct_exit_flag_mismatch")
    require(bool(row["incorrect_exit"]) == bool(exited and not (cleared == set(sources) and claimed_certificate)),
            "incorrect_exit_flag_mismatch")
    require(not row["success"] or evaluation["simulator_stop_reason"] == "exited", "success_without_exit_session")
    stats = trace.get("planner_stats", {})
    require(stats == row.get("planner_stats", {}), "planner_stats_row_trace_mismatch")
    if stats:
        require(0 <= stats["planning_wall_s"] <= 60+PLANNING_BUDGET_TOLERANCE_S, "planning_wall_budget_exceeded")
        require(0 <= stats["planning_calls"] <= 12, "planning_call_budget_exceeded")
        require(stats["planning_wall_s"] <= row["policy_wall_s"]+TIME_TOLERANCE_S, "planning_wall_exceeds_policy_wall")
        require(stats["planning_calls"] == len(stats["decisions"]), "planning_calls_decision_log_mismatch")
        for index, decision in enumerate(stats["decisions"]):
            if decision.get("completed"):
                delta = decision.get("paired_delta_s", [])
                require(len(delta) == 12, f"decision_{index}:not_twelve_paired_deltas")
                if delta:
                    close(decision["mean_delta_s"], statistics.mean(delta), f"decision_{index}:mean_delta_mismatch")
                    close(decision["candidate_mean_remaining_s"]-decision["baseline_mean_remaining_s"],
                          statistics.mean(delta), f"decision_{index}:candidate_baseline_delta_mismatch")
    for label in BOUND_LABELS:
        key = f"{label}_lower_bound_s"
        if key in row:
            bound = row[key]
            require(math.isfinite(bound) and bound > 0, f"{label}:invalid_lower_bound")
            if bound > 0:
                close(row[f"time_over_{label}_lower_bound"], total/bound, f"{label}:ratio_mismatch")
                if success:
                    require(total+TIME_TOLERANCE_S >= bound, f"{label}:successful_time_below_bound")
    return dict(passed=not errors, errors=errors, audited_api_actions=len(observations),
                charged_actions=len(accepted), recomputed_virtual_time_s=total,
                unrounded_movement_s=raw_movement_s,
                rounded_movement_s=component_us["movement_s"]/1_000_000,
                certified_success=success and not errors,
                actual_cleared_channels=sorted(cleared))


def load_batches(batch_paths):
    """Accept complete immutable snapshots, returning raw audited occurrences."""
    rows, batches, skipped = [], [], []
    for path in batch_paths:
        path = Path(path)
        result_path = path/"results.json"
        if not result_path.is_file():
            skipped.append(dict(batch=path.name, reason="missing_results_json"))
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = data["manifest"]
        if manifest.get("status") != "completed":
            skipped.append(dict(batch=path.name, reason="batch_not_completed", status=manifest.get("status")))
            continue
        expected = manifest["cases"]*len(manifest["configs"])
        files = sorted(path.glob("trace-*-*.json.gz"))
        if not (manifest.get("completed_runs") == len(data["rows"]) == len(files) == expected
                and manifest.get("completed_cases") == manifest["cases"]):
            skipped.append(dict(batch=path.name, reason="completed_manifest_count_mismatch",
                                expected_runs=expected, trace_files=len(files), rows=len(data["rows"])))
            continue
        listed = {(r["case_id"], r["strategy"]): r for r in data["rows"]}
        expected_keys = {(case_id, strategy) for case_id in manifest["case_ids"] for strategy in manifest["configs"]}
        if len(listed) != expected or set(listed) != expected_keys:
            skipped.append(dict(batch=path.name, reason="completed_manifest_identity_mismatch"))
            continue
        staged, seen = [], set()
        for file in files:
            with gzip.open(file, "rt", encoding="utf-8") as handle:
                trace = json.load(handle)
            row = dict(trace["row"])
            key = (row["case_id"], row["strategy"])
            if key not in listed or key in seen:
                raise ValueError(f"Trace identity mismatch in batch {path.name}: {file.name}")
            seen.add(key)
            audit = audit_trace(trace)
            if row != listed[key]:
                audit["errors"].append("trace_row_differs_from_results_row")
                audit.update(passed=False, certified_success=False)
            original_suite_group = row["suite_group"]
            if path.name == "rollout_smoke":
                row["suite_group"] = "pilot_rollout_smoke/"+row["suite_group"]
            row.update(batch=path.name, trace_file=str(file.relative_to(path.parent)).replace("\\", "/"),
                       original_suite_group=original_suite_group, audit=audit,
                       analysis_success=bool(row["success"] and audit["passed"]))
            staged.append(row)
        if seen != expected_keys:
            skipped.append(dict(batch=path.name, reason="trace_identity_set_incomplete"))
            continue
        rows.extend(staged)
        batches.append(dict(batch=path.name, manifest=manifest, runs=len(staged),
                            audit_failures=sum(not row["audit"]["passed"] for row in staged)))
    return rows, batches, skipped


def deduplicate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite_group"], row["case_id"], row["strategy"])].append(row)
    unique, conflicts, duplicates = [], [], []
    for key, occurrences in sorted(grouped.items()):
        first = occurrences[0]
        differences = []
        for other in occurrences[1:]:
            for field in ("scenario_sha256", "success", "analysis_success", "original_case_group"):
                if other[field] != first[field]:
                    differences.append(field)
            if abs(other["virtual_time_s"]-first["virtual_time_s"]) > TIME_TOLERANCE_S:
                differences.append("virtual_time_s")
        details = dict(suite_group=key[0], case_id=key[1], strategy=key[2],
                       occurrences=[dict(batch=r["batch"], virtual_time_s=r["virtual_time_s"],
                                         success=r["success"], audit_passed=r["audit"]["passed"],
                                         scenario_sha256=r["scenario_sha256"], trace_file=r["trace_file"])
                                    for r in occurrences])
        if differences:
            details["conflicting_fields"] = sorted(set(differences))
            conflicts.append(details)
        else:
            if len(occurrences) > 1:
                duplicates.append(details)
            first["replicate_count"] = len(occurrences)
            unique.append(first)
        for i, row in enumerate(occurrences):
            row.update(deduplication_status="conflict_excluded" if differences else
                       "primary" if i == 0 else "identical_replicate", replicate_count=len(occurrences))
    return unique, conflicts, duplicates


def bootstrap_groups(delta_by_group, *, draws=2000, seed=20260912):
    """Cluster bootstrap: reconstruction variants first average within raw case."""
    values = [_mean(v) for _, v in sorted(delta_by_group.items())]
    if not values:
        return None
    rng = random.Random(seed)
    boot = sorted(_mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return [boot[max(0, math.ceil(draws*.025)-1)], boot[max(0, math.ceil(draws*.975)-1)]]


def paired_comparison(rows, baseline_name, candidate_name, *, bootstrap_draws=2000, worst_count=8):
    base = {r["case_id"]: r for r in rows if r["strategy"] == baseline_name}
    candidate = {r["case_id"]: r for r in rows if r["strategy"] == candidate_name}
    pair_ids = sorted(set(base) & set(candidate))
    delta_by_group, successful, excluded = defaultdict(list), [], []
    for case_id in pair_ids:
        a, b = base[case_id], candidate[case_id]
        reason = None
        if a["scenario_sha256"] != b["scenario_sha256"]:
            reason = "scenario_mismatch"
        elif a["original_case_group"] != b["original_case_group"]:
            reason = "original_case_group_mismatch"
        elif not (a["analysis_success"] and b["analysis_success"]):
            reason = "failed_or_audit_failed_run"
        if reason:
            excluded.append(dict(case_id=case_id, reason=reason,
                                 baseline_success=a["analysis_success"], candidate_success=b["analysis_success"]))
            continue
        delta = b["virtual_time_s"]-a["virtual_time_s"]
        group = b["original_case_group"]
        delta_by_group[group].append(delta)
        detail = dict(case_id=case_id, original_case_group=group, source_total=b["source_total"],
                      baseline_s=a["virtual_time_s"], candidate_s=b["virtual_time_s"], delta_s=delta,
                      relative_change_pct=100*delta/a["virtual_time_s"])
        for label in BOUND_LABELS:
            key = f"{label}_lower_bound_s"
            if key in b:
                detail[key] = b[key]
                detail[f"candidate_over_{label}_lower_bound"] = b["virtual_time_s"]/b[key]
                detail[f"baseline_over_{label}_lower_bound"] = a["virtual_time_s"]/a[key]
        successful.append(detail)
    group_deltas = [_mean(v) for v in delta_by_group.values()]
    return dict(baseline=baseline_name, candidate=candidate_name, paired_cases=len(pair_ids),
                completed_pairs=len(successful), excluded_pairs=excluded,
                missing_baseline_cases=sorted(set(candidate)-set(base)),
                missing_candidate_cases=sorted(set(base)-set(candidate)),
                original_groups=len(group_deltas), mean_completed_delta_s=_mean(d["delta_s"] for d in successful),
                mean_group_delta_s=_mean(group_deltas),
                paired_group_bootstrap_95_delta_s=bootstrap_groups(delta_by_group, draws=bootstrap_draws),
                case_wins=sum(d["delta_s"] < -TIME_TOLERANCE_S for d in successful),
                case_losses=sum(d["delta_s"] > TIME_TOLERANCE_S for d in successful),
                case_ties=sum(abs(d["delta_s"]) <= TIME_TOLERANCE_S for d in successful),
                group_wins=sum(d < -TIME_TOLERANCE_S for d in group_deltas),
                group_losses=sum(d > TIME_TOLERANCE_S for d in group_deltas),
                worst_regressions=sorted((d for d in successful if d["delta_s"] > TIME_TOLERANCE_S),
                                         key=lambda d: d["delta_s"], reverse=True)[:worst_count],
                largest_improvements=sorted((d for d in successful if d["delta_s"] < -TIME_TOLERANCE_S),
                                            key=lambda d: d["delta_s"])[:worst_count],
                original_group_deltas={g: _mean(ds) for g, ds in sorted(delta_by_group.items())})


def summarize_strategy(rows, *, source_strata=True):
    successful = [r for r in rows if r["analysis_success"]]
    failed = len(successful) != len(rows)
    times = sorted(r["virtual_time_s"] for r in successful)
    result = dict(cases=len(rows), successes=len(successful), failures=len(rows)-len(successful),
                  failure_case_ids=[r["case_id"] for r in rows if not r["analysis_success"]],
                  mean_s=None if failed else _mean(times),
                  mean_successful_completion_s=_mean(times),
                  p95_s=None if failed or not times else times[math.ceil(.95*len(times))-1],
                  worst_s=None if failed or not times else max(times),
                  mean_time_per_source_s=None if failed else _mean(r["virtual_time_s"]/r["source_total"] for r in rows),
                  mean_measurements=_mean(r["measurement_count"] for r in successful),
                  failed_clears=sum(r["failed_clear_count"] for r in rows),
                  mean_confirmation_tail_s=_mean(r["confirmation_tail_s"] for r in successful),
                  mean_policy_wall_s=_mean(r["policy_wall_s"] for r in rows),
                  max_policy_wall_s=max((r["policy_wall_s"] for r in rows), default=None),
                  total_policy_cpu_s=sum(r["policy_cpu_s"] for r in rows),
                  mean_planning_wall_s=_mean(r.get("planner_stats", {}).get("planning_wall_s", 0) for r in rows),
                  planner_calls=sum(r.get("planner_stats", {}).get("planning_calls", 0) for r in rows),
                  planner_overrides=sum(r.get("planner_stats", {}).get("overrides", 0) for r in rows),
                  complete_rollouts=sum(r.get("planner_stats", {}).get("completed_rollouts", 0) for r in rows),
                  mean_components={k: _mean(r["time_breakdown_s"][k] for r in successful) for k in COMPONENTS},
                  lower_bounds={})
    for label in BOUND_LABELS:
        key = f"{label}_lower_bound_s"
        subset = [r for r in rows if key in r]
        if subset:
            ok = all(r["analysis_success"] for r in subset)
            lb = _mean(r[key] for r in subset)
            result["lower_bounds"][label] = dict(cases=len(subset), mean_lower_bound_s=lb,
                ratio_of_mean_times=None if not ok else _mean(r["virtual_time_s"] for r in subset)/lb,
                mean_case_ratio=None if not ok else _mean(r["virtual_time_s"]/r[key] for r in subset))
    if source_strata:
        result["by_source_count"] = {str(n): summarize_strategy([r for r in rows if r["source_total"] == n], source_strata=False)
                                     for n in sorted({r["source_total"] for r in rows})}
    return result


def analyze(batch_paths, *, bootstrap_draws=2000, worst_count=8):
    raw, batches, skipped = load_batches(batch_paths)
    unique, conflicts, duplicates = deduplicate(raw)
    grouped = defaultdict(list)
    for row in unique:
        grouped[row["suite_group"]].append(row)
    summaries = {}
    for group, rows in sorted(grouped.items()):
        strategies = sorted({r["strategy"] for r in rows})
        comparisons = [("B", name) for name in strategies if name != "B" and "B" in strategies]
        comparisons += [(a, b) for a, b in (("R", "RA"), ("CR", "CRA")) if a in strategies and b in strategies]
        summaries[group] = dict(
            strategies={name: summarize_strategy([r for r in rows if r["strategy"] == name]) for name in strategies},
            paired={f"{a}_vs_{b}": paired_comparison(rows, a, b, bootstrap_draws=bootstrap_draws, worst_count=worst_count)
                    for a, b in comparisons})
    failures = [dict(batch=r["batch"], case_id=r["case_id"], strategy=r["strategy"],
                     trace_file=r["trace_file"], errors=r["audit"]["errors"]) for r in raw if not r["audit"]["passed"]]
    output = dict(schema_version=1, included_batches=batches, skipped_batches=skipped,
        raw_run_occurrences=len(raw), unique_unconflicted_runs=len(unique),
        consistent_duplicates=duplicates, duplicate_conflicts=conflicts,
        audit=dict(passed=not failures, audited_traces=len(raw),
                   audited_api_actions=sum(r["audit"]["audited_api_actions"] for r in raw), failures=failures,
                   time_tolerance_s=TIME_TOLERANCE_S, planning_budget_tolerance_s=PLANNING_BUDGET_TOLERANCE_S,
                   checks=["independent move/detect/switch/clear accounting", "all cumulative action timestamps",
                           "public history alignment", "fixed-position feedback before removal", "truth distance/bearing compatibility",
                           "cleared sets and reported completion/exit", "planning budgets and 12-delta completion logs"],
                   limitations=["No persisted per-world rollout action log: common-world reuse is not independently auditable.",
                                "The certificate flag is cross-checked against statuses and trace truth; continuous absence geometry is not re-proved.",
                                "Stored lower-bound values and ratios are checked for consistency; exponential path-bound calculation is not rerun."]),
        cpu=dict(total_completed_batch_cpu_s=sum(b["manifest"].get("total_cpu_s", 0) for b in batches),
                 total_raw_policy_cpu_s=sum(r["policy_cpu_s"] for r in raw),
                 total_unique_policy_cpu_s=sum(r["policy_cpu_s"] for r in unique)),
        statistical_method=dict(bootstrap_draws=bootstrap_draws, seed=20260912,
                                sampling_unit="original_case_group; average reconstruction deltas within group, then bootstrap groups",
                                failed_run_policy="Failed/audit-failed runs never improve mean completion time; full mean is null, pair exclusions explicit.",
                                duplicate_policy="Conflicting duplicates remain in CSV and conflict list, and are excluded from summary selection."),
        groups=summaries)
    return output, raw


def write_outputs(output, rows, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    csv_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        flat.update(audit_passed=row["audit"]["passed"], audit_errors=" | ".join(row["audit"]["errors"]))
        flat.update({f"cost_{k}": v for k, v in row["time_breakdown_s"].items()})
        flat.update({f"planner_{k}": v for k, v in row.get("planner_stats", {}).items() if not isinstance(v, (dict, list))})
        csv_rows.append(flat)
    first = ["suite_group", "case_id", "original_case_group", "strategy", "batch", "deduplication_status", "replicate_count"]
    fields = first+sorted({key for row in csv_rows for key in row}-set(first))
    with (out.parent/"all_runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--batches", nargs="+", help="Batch directory names or paths; default all except rollout_smoke")
    parser.add_argument("--out", type=Path, default=ROOT/"research"/"q3_fresh_round2"/"analysis.json")
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--worst-count", type=int, default=8)
    args = parser.parse_args()
    if args.bootstrap_draws < 1 or args.worst_count < 0:
        parser.error("bootstrap-draws must be positive and worst-count nonnegative")
    if args.batches:
        paths = [Path(name) if Path(name).is_dir() else args.batch_root/name for name in args.batches]
    else:
        paths = [p for p in sorted(args.batch_root.iterdir()) if p.is_dir() and p.name != "rollout_smoke"]
    report, rows = analyze(paths, bootstrap_draws=args.bootstrap_draws, worst_count=args.worst_count)
    write_outputs(report, rows, args.out)
    print(json.dumps(dict(output=str(args.out), included_batches=len(report["included_batches"]),
                          skipped_batches=len(report["skipped_batches"]), raw_runs=len(rows),
                          unique_runs=report["unique_unconflicted_runs"], audit_passed=report["audit"]["passed"],
                          conflicts=len(report["duplicate_conflicts"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
