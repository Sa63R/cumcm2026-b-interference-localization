"""Read terminated source-mode records; never run a policy or read world truth.

Example: python experiments/analyze_source_modes_v2.py --input results/pilot \
    --output results/pilot/source_mode_diagnostics.json

The input layout is manifest.json plus records/<variant>-<seed>.json.gz. Each
record uses research_v1_eval.run_case's row/summary/evaluation/history/spec
schema. Missing instrumentation is reported, not silently inferred as a change.
Statistics describe observed decisions, not counterfactual policy effects.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _position(value):
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y")]
    if (isinstance(value, (tuple, list)) and len(value) == 2
            and all(_number(v) for v in value)):
        return tuple(float(v) for v in value)
    return None


def _same_position(a, b):
    a, b = _position(a), _position(b)
    return a is not None and b is not None and math.dist(a, b) <= 1e-5


def _stats(values):
    values = sorted(float(v) for v in values if _number(v))
    if not values:
        return {"n": 0, "sum": 0., "mean": None, "median": None, "p95": None, "max": None}
    index = .95*(len(values)-1)
    lower = int(index)
    p95 = values[lower] + (index-lower)*(values[min(lower+1, len(values)-1)]-values[lower])
    return {"n": len(values), "sum": sum(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p95": p95, "max": values[-1]}


def _fraction(numerator, denominator):
    return numerator/denominator if denominator else None


def _physical_history(record):
    """Align report's measurement/clear indices with the accepted wire ledger."""
    actions, previous, channel, elapsed, residuals = [], (0., 0.), 1, 0., []
    for wire in record.get("history", []):
        kind = str(wire.get("action", "")).lstrip("/")
        if kind not in ("measure", "clear"):
            continue
        response = wire.get("response") or {}
        if response.get("accepted") is not True:
            raise ValueError("unaccepted action in completed wire history")
        position = _position(wire.get("position"))
        ch = wire.get("channel")
        when = response.get("virtual_time_s")
        if position is None or type(ch) is not int or not _number(when):
            raise ValueError("wire action missing finite position/channel/time")
        movement = round(math.dist(previous, position)/5*1_000_000)/1_000_000
        switching = float(kind == "measure" and ch != channel)
        detection = 5. if kind == "measure" else 0.
        result = response.get("measure_result" if kind == "measure" else "clear_result")
        optical = 3. if kind == "clear" else 0.
        removal = 2. if kind == "clear" and result == "success" else 0.
        expected = movement+switching+detection+optical+removal
        residuals.append(abs(when-elapsed-expected))
        actions.append({"action": kind, "position": position, "channel": ch,
                        "result": result, "start_time_s": elapsed, "time_s": when,
                        "movement_s": movement, "switching_s": switching,
                        "detection_s": detection, "optical_s": optical, "removal_s": removal,
                        "cost_s": when-elapsed, "phase": None})
        if kind == "measure":
            channel = ch  # Optical clear does not change the measuring channel.
        previous, elapsed = position, when
    reported = (record.get("summary") or {}).get("action_history", [])
    aligned = len(reported) == len(actions)
    if aligned:
        for actual, summary in zip(actions, reported):
            if (actual["action"] != summary.get("action") or actual["channel"] != summary.get("channel")
                    or not _same_position(actual["position"], summary.get("position"))):
                aligned = False
                break
            actual["phase"] = summary.get("phase")
    return actions, aligned, max(residuals, default=0.)


def _task(value, separate_position=None):
    if isinstance(value, dict):
        return value.get("kind"), value.get("channel"), _position(value.get("position", separate_position))
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[0], value[1], _position(value[2] if len(value) > 2 else separate_position)
    return None


def _task_change(log):
    old = _task(log.get("old_selected"), log.get("old_selected_position"))
    new = _task(log.get("selected"), log.get("selected_position"))
    if old is None or new is None:
        return "unlogged"
    if old[:2] != new[:2]:
        return "changed_kind_or_channel"
    if old[0] == "cover":
        if old[2] is None or new[2] is None:
            return "cover_position_unlogged"
        return "unchanged" if _same_position(old[2], new[2]) else "changed_cover_position"
    return "unchanged"


def _chosen(log, physical):
    """Return source info and a valid nominal mode, never a 5s point fallback."""
    info = log
    channel = log.get("channel")
    chosen = log.get("chosen")
    selection = "explicit_chosen" if isinstance(chosen, dict) else None
    if log.get("variant") == "joint":
        order = (log.get("route") or {}).get("order", [])
        groups = log.get("groups", [])
        if (not order or not isinstance(order[0], (list, tuple)) or len(order[0]) != 2):
            return None, None, None, "route_first_group_mode_unavailable"
        group, mode = order[0]
        if type(group) is not int or not 0 <= group < len(groups):
            return None, None, None, "route_group_index_invalid"
        info = groups[group]
        channel = info.get("channel")
        if info.get("kind") != "nominal" or info.get("point_fallback"):
            return channel, info, None, "non_nominal_or_point_fallback"
        candidates = [c for c in info.get("candidates", []) if c.get("valid") is True]
        if type(mode) is not int or not 0 <= mode < len(candidates):
            return channel, info, None, "route_mode_index_invalid"
        if isinstance(chosen, dict) and not _same_position(chosen.get("entry"), candidates[mode].get("entry")):
            return channel, info, None, "chosen_entry_disagrees_with_route_order"
        return channel, info, candidates[mode], "route_order_valid_filtered_index"
    if info.get("kind") != "nominal" or info.get("point_fallback"):
        return channel, info, None, "non_nominal_or_point_fallback"
    candidates = [c for c in info.get("candidates", []) if c.get("valid") is True]
    if isinstance(chosen, dict):
        matches = [c for c in candidates if _same_position(c.get("entry"), chosen.get("entry"))]
        if len(matches) != 1:
            return channel, info, None, "chosen_local_entry_not_unique_valid_candidate"
        # The compact chosen dictionary need not repeat support/measurement
        # metadata; retain that from the unambiguously matching candidate.
        chosen = {**matches[0], **chosen}
    if not isinstance(chosen, dict):
        mode = log.get("selected_mode")
        if type(mode) is int and 0 <= mode < len(candidates):
            chosen, selection = candidates[mode], "explicit_valid_filtered_index"
    if not isinstance(chosen, dict):
        prefix = log.get("after_actual_action_count")
        if type(prefix) is int and 0 <= prefix < len(physical):
            actual = physical[prefix]
            matches = [c for c in candidates if actual["action"] == "measure"
                       and actual["channel"] == channel
                       and _same_position(c.get("entry"), actual["position"])]
            if len(matches) == 1:
                chosen, selection = matches[0], "recovered_unique_immediate_actual_entry"
    if not isinstance(chosen, dict):
        return channel, info, None, "selected_local_mode_unavailable"
    return channel, info, chosen, selection


def _service(log, physical, aligned, configured_supports=None):
    channel, info, candidate, selection = _chosen(log, physical)
    result = {"status": selection, "selection_recovery": selection,
              "channel": channel, "after_actual_action_count": log.get("after_actual_action_count")}
    if candidate is None:
        return result
    result.update(predicted_cost_s=candidate.get("cost_s"), predicted_exit=candidate.get("exit"),
                  predicted_measurements=candidate.get("measurements_mean"),
                  predicted_hypotheses=candidate.get("hypotheses"), entry=candidate.get("entry"))
    old_probe = info.get("old_probe")
    result["changed_first_probe"] = (not _same_position(old_probe, candidate.get("entry"))
                                     if _position(old_probe) is not None else None)
    candidates = info.get("candidates", [])
    # The generator proposes [old, center, quartile25, quartile75], but
    # deduplication/history filtering can remove rows. Only this unambiguous
    # four-row case is used to assign the center role from historical logs.
    if (len(candidates) == 4 and _same_position(candidates[0].get("entry"), old_probe)):
        matches = [i for i, c in enumerate(candidates) if _same_position(c.get("entry"), candidate.get("entry"))]
        result["candidate_role"] = ("old", "center", "quartile25", "quartile75")[matches[0]] if len(matches) == 1 else "unavailable"
    else:
        result["candidate_role"] = "unavailable_due_to_candidate_filtering"
    result["center_atom_near_mass_at_least"] = (1/3 if result["candidate_role"] == "center"
        and configured_supports == 3 and candidate.get("hypotheses") == 3 else None)
    if not aligned:
        result["status"] = "summary_wire_action_alignment_failed"
        return result
    prefix = log.get("after_actual_action_count")
    if type(prefix) is not int or not 0 <= prefix < len(physical):
        result["status"] = "actual_prefix_unavailable"
        return result
    first = physical[prefix]
    current = physical[prefix-1]["position"] if prefix else (0., 0.)
    old_candidates = [c for c in candidates if c.get("valid") is True
                      and _same_position(c.get("entry"), old_probe)]
    if (len(old_candidates) == 1 and _number(old_candidates[0].get("cost_s"))
            and _number(candidate.get("cost_s")) and _position(candidate.get("entry")) is not None):
        result["nominal_old_minus_chosen_local_total_s"] = (old_candidates[0]["cost_s"]
            + math.dist(current, _position(old_probe))/5-candidate["cost_s"]
            - math.dist(current, _position(candidate["entry"]))/5)
    if (first["action"] != "measure" or first["channel"] != channel
            or not _same_position(first["position"], candidate.get("entry"))):
        result["status"] = "selected_first_probe_not_immediately_executed"
        return result
    end = next((i for i in range(prefix, len(physical)) if physical[i]["action"] == "clear"
                and physical[i]["channel"] == channel and physical[i]["result"] == "success"), None)
    if end is None:
        result["status"] = "same_source_successful_clear_absent"
        return result
    segment = physical[prefix:end+1]
    if any(a["channel"] != channel for a in segment):
        result["status"] = "intervening_other_source_actions_not_local_service"
        return result
    actual = segment[-1]["time_s"]-first["start_time_s"]-first["movement_s"]
    adjusted = actual + 1.-first["switching_s"]
    measurements = sum(a["action"] == "measure" for a in segment)
    result.update(status="matched_complete_same_source_service",
                  actual_start_index=prefix, actual_clear_index=end,
                  actual_cost_excluding_arrival_s=actual,
                  actual_cost_entry_switch_normalized_s=adjusted,
                  first_arrival_excluded_s=first["movement_s"],
                  actual_entry_switch_s=first["switching_s"],
                  actual_first_result=first["result"],
                  logged_entry_channel_consistent=(first["switching_s"] ==
                      float(log["logged_at_current_channel"] != channel)
                      if type(log.get("logged_at_current_channel")) is int else None),
                  actual_exit=list(segment[-1]["position"]), actual_measurements=measurements,
                  actual_failed_clears=sum(a["action"] == "clear" and a["result"] != "success" for a in segment),
                  actual_no_signal=sum(a["action"] == "measure" and a["result"] == "no_signal" for a in segment),
                  phases=dict(Counter(a["phase"] or "unlogged" for a in segment)))
    predicted = candidate.get("cost_s")
    if _number(predicted):
        result["actual_minus_predicted_s"] = actual-predicted
        result["normalized_actual_minus_predicted_s"] = adjusted-predicted
    predicted_exit = _position(candidate.get("exit"))
    if predicted_exit is not None:
        result["exit_error_m"] = math.dist(predicted_exit, segment[-1]["position"])
    if _number(candidate.get("measurements_mean")):
        result["actual_minus_predicted_measurements"] = measurements-candidate["measurements_mean"]
    return result


def analyze_record(record):
    if record.get("evaluation_phase") != "after_policy_termination":
        raise ValueError("record is not explicitly marked after_policy_termination")
    row, summary = record.get("row") or {}, record.get("summary") or {}
    physical, aligned, ledger_error = _physical_history(record)
    params = summary.get("strategy_parameters") or {}
    logs = params.get("source_mode_log") or []
    result = {"seed": row.get("seed"), "strategy": row.get("strategy"),
              "mode_variant": params.get("source_service_modes", "unlogged"),
              "successful": row.get("successful"), "virtual_time_s": row.get("virtual_time_s"),
              "penalized_time_s": row.get("penalized_time_s"),
              "failed_clear_count": row.get("failed_clear_count"),
              "billing_s": {key: row.get(key) for key in
                  ("movement_s", "detection_s", "switching_s", "optical_s", "removal_s")},
              "program_runtime_s": row.get("program_runtime_s"),
              "summary_wire_aligned": aligned, "ledger_max_residual_s": ledger_error,
              "logged_decisions": len(logs), "candidate_lookups": 0, "valid_candidate_lookups": 0,
              "invalid_candidate_lookups": 0, "cached_lookups": 0, "fresh_lookups": 0,
              "fresh_valid": 0, "fresh_invalid": 0, "cache_flag_unlogged": 0,
              "nominal_groups": 0, "nominal_groups_without_valid": 0, "point_fallback_groups": 0,
              "candidate_counts_per_nominal_group": [], "mode_runtime_s": [],
              "task_changes": Counter(), "invalid_reasons": Counter(), "group_kinds": Counter(),
              "route_gap_s": [], "route_gap_fraction": [], "route_exact_count": 0,
              "route_count": 0, "route_expanded": 0, "route_runtime_s": [],
              "services": [], "warnings": []}
    for index, log in enumerate(logs):
        if _number(log.get("runtime_s")):
            result["mode_runtime_s"].append(log["runtime_s"])
        result["task_changes"][_task_change(log)] += 1
        groups = log.get("groups", []) if log.get("variant") == "joint" else [log]
        for group in groups:
            result["group_kinds"][group.get("kind", "unlogged")] += 1
            if group.get("point_fallback"):
                result["point_fallback_groups"] += 1
            if group.get("kind") != "nominal":
                continue
            candidates = group.get("candidates", [])
            result["nominal_groups"] += 1
            result["candidate_counts_per_nominal_group"].append(len(candidates))
            result["nominal_groups_without_valid"] += not any(c.get("valid") is True for c in candidates)
            for candidate in candidates:
                valid = candidate.get("valid") is True
                result["candidate_lookups"] += 1
                result["valid_candidate_lookups" if valid else "invalid_candidate_lookups"] += 1
                if not valid:
                    result["invalid_reasons"][candidate.get("reason", "unlogged")] += 1
                cached = candidate.get("cached")
                if cached is True:
                    result["cached_lookups"] += 1
                elif cached is False:
                    result["fresh_lookups"] += 1
                    result["fresh_valid" if valid else "fresh_invalid"] += 1
                else:
                    result["cache_flag_unlogged"] += 1
        route = log.get("route")
        if isinstance(route, dict):
            cost, bound = route.get("cost_s"), route.get("lower_bound_s")
            result["route_count"] += 1
            result["route_exact_count"] += route.get("exact") is True
            if _number(route.get("expanded")):
                result["route_expanded"] += route["expanded"]
            if _number(route.get("runtime_s")):
                result["route_runtime_s"].append(route["runtime_s"])
            if _number(cost) and _number(bound):
                gap = cost-bound
                if gap < -1e-7:
                    result["warnings"].append(f"negative finite route gap at decision {index}")
                result["route_gap_s"].append(gap)
                if cost > 0:
                    result["route_gap_fraction"].append(gap/cost)
            else:
                result["warnings"].append(f"missing finite route cost/bound at decision {index}")
        service = _service(log, physical, aligned, (params.get("source_mode_limits") or {}).get("supports"))
        service["log_index"] = index
        result["services"].append(service)
    if ledger_error > 1e-5:
        result["warnings"].append("wire ledger residual exceeds 10 microseconds")
    if not aligned and logs:
        result["warnings"].append("action prefixes cannot be calibrated against wire history")
    return result


def summarize(cases):
    integers = ("logged_decisions", "candidate_lookups", "valid_candidate_lookups", "invalid_candidate_lookups",
                "cached_lookups", "fresh_lookups", "fresh_valid", "fresh_invalid", "cache_flag_unlogged",
                "nominal_groups", "nominal_groups_without_valid", "point_fallback_groups", "route_exact_count",
                "route_count", "route_expanded")
    result = {"records": len(cases), **{key: sum(c[key] for c in cases) for key in integers}}
    for key in ("candidate_counts_per_nominal_group", "mode_runtime_s", "route_gap_s", "route_gap_fraction", "route_runtime_s"):
        result[key] = _stats(v for c in cases for v in c[key])
    for key in ("task_changes", "invalid_reasons", "group_kinds"):
        result[key] = dict(sum((Counter(c[key]) for c in cases), Counter()))
    result["valid_lookup_fraction"] = _fraction(result["valid_candidate_lookups"], result["candidate_lookups"])
    result["fresh_valid_fraction"] = _fraction(result["fresh_valid"], result["fresh_lookups"])
    result["cache_hit_fraction_known_flags"] = _fraction(result["cached_lookups"], result["cached_lookups"]+result["fresh_lookups"])
    result["finite_route_exact_fraction"] = _fraction(result["route_exact_count"], result["route_count"])
    result["logged_extra_mode_cpu_s_per_record"] = _stats(sum(c["mode_runtime_s"]) for c in cases)
    services = [s for c in cases for s in c["services"]]
    complete = [s for s in services if s["status"] == "matched_complete_same_source_service"]
    result["service_match_statuses"] = dict(Counter(s["status"] for s in services))
    result["changed_first_probe"] = {"changed": sum(s.get("changed_first_probe") is True for s in services),
                                       "unchanged": sum(s.get("changed_first_probe") is False for s in services),
                                       "unavailable": sum(s.get("changed_first_probe") is None for s in services)}
    result["calibration"] = {"complete_same_source_services": len(complete)}
    for key in ("predicted_cost_s", "actual_cost_excluding_arrival_s", "actual_cost_entry_switch_normalized_s",
                "actual_minus_predicted_s", "normalized_actual_minus_predicted_s", "exit_error_m",
                "predicted_measurements", "actual_measurements", "actual_minus_predicted_measurements"):
        result["calibration"][key] = _stats(s.get(key) for s in complete)
    result["calibration"]["normalized_cost_absolute_error_s"] = _stats(
        abs(s["normalized_actual_minus_predicted_s"]) for s in complete
        if _number(s.get("normalized_actual_minus_predicted_s")))
    changed = [s for s in complete if s.get("changed_first_probe") is True]
    result["changed_probe_calibration"] = {
        "count": len(changed), "candidate_roles": dict(Counter(s.get("candidate_role") for s in changed)),
        "nominal_old_minus_chosen_local_total_s": _stats(s.get("nominal_old_minus_chosen_local_total_s") for s in changed),
        "normalized_actual_minus_predicted_s": _stats(s.get("normalized_actual_minus_predicted_s") for s in changed),
        "normalized_cost_absolute_error_s": _stats(abs(s["normalized_actual_minus_predicted_s"])
            for s in changed if _number(s.get("normalized_actual_minus_predicted_s")))}
    centers = [s for s in complete if s.get("candidate_role") == "center"]
    atom_centers = [s for s in centers if _number(s.get("center_atom_near_mass_at_least"))]
    result["center_atom_diagnostic"] = {
        "scope": "four unfiltered logged rows with row0=old_probe; role inferred from fixed generation order only",
        "identified_selected_centers": len(centers), "three_support_selected_centers": len(atom_centers),
        "sum_nominal_near_mass_lower_bounds": sum(s["center_atom_near_mass_at_least"] for s in atom_centers),
        "observed_first_near_count_on_same_subset": sum(s.get("actual_first_result") == "near" for s in atom_centers),
        "center_normalized_actual_minus_predicted_s": _stats(s.get("normalized_actual_minus_predicted_s") for s in centers),
        "interpretation": "deterministic finite quadrature atom, not a calibrated event probability; selected services are dependent and not a Bernoulli significance test"}
    return result


def paired_descriptive(cases, baseline="v1"):
    """Bookkeeping only; qualification/CI remain in the frozen pilot harness."""
    reference = {c["seed"]: c for c in cases if c["strategy"] == baseline}
    groups = defaultdict(list)
    for case in cases:
        if case["strategy"] == baseline or case["seed"] not in reference:
            continue
        old = reference[case["seed"]]
        if not (_number(old.get("virtual_time_s")) and _number(case.get("virtual_time_s"))):
            continue
        fees = {key: case["billing_s"][key]-old["billing_s"][key]
                for key in case["billing_s"] if _number(case["billing_s"][key]) and _number(old["billing_s"][key])}
        qualified = (old["successful"] is True and case["successful"] is True
                     and old.get("failed_clear_count") == case.get("failed_clear_count") == 0)
        penalty_saved = (old["penalized_time_s"]-case["penalized_time_s"]
                         if _number(old.get("penalized_time_s")) and _number(case.get("penalized_time_s")) else None)
        groups[case["strategy"]].append({"seed": case["seed"],
            "raw_saved_s": old["virtual_time_s"]-case["virtual_time_s"], "penalized_saved_s": penalty_saved,
            "both_successful_zero_failed_clear": qualified, "candidate_minus_baseline_billing_s": fees})
    result = {}
    for variant, pairs in groups.items():
        qualified = all(p["both_successful_zero_failed_clear"] for p in pairs)
        values = [p["raw_saved_s"] for p in pairs]
        result[variant] = {"baseline": baseline, "pairs": len(pairs),
            "all_pairs_successful_zero_failed_clear": qualified,
            "raw_saved_s": _stats(values), "penalized_saved_s": _stats(p["penalized_saved_s"] for p in pairs),
            "raw_wins": sum(v > 1e-9 for v in values), "raw_losses": sum(v < -1e-9 for v in values),
            "raw_ties": sum(abs(v) <= 1e-9 for v in values),
            "mean_billing_delta_s": {key: statistics.mean(p["candidate_minus_baseline_billing_s"][key]
                for p in pairs if key in p["candidate_minus_baseline_billing_s"])
                for key in sorted({key for p in pairs for key in p["candidate_minus_baseline_billing_s"]})},
            "cases_worst_first": sorted(pairs, key=lambda p: p["raw_saved_s"]),
            "scope": "descriptive paired bookkeeping; raw speed is not an improvement claim when reliability fails; no new CI or gate"}
    return result


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze_directory(directory, *, skip_invalid=False, baseline="v1"):
    directory = Path(directory).resolve()
    manifest = directory/"manifest.json"
    record_dir = directory/"records"
    if not record_dir.is_dir():
        raise ValueError(f"records directory does not exist: {record_dir}")
    paths = sorted(record_dir.glob("*.json.gz"))
    if not paths:
        raise ValueError("no records/*.json.gz files found")
    cases, provenance, errors, seen = [], [], [], set()
    for path in paths:
        item = {"path": path.relative_to(directory).as_posix(), "sha256": _digest(path)}
        provenance.append(item)
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                record = json.load(stream)
            case = analyze_record(record)
            key = (case["strategy"], case["seed"])
            if key in seen:
                raise ValueError(f"duplicate (strategy,seed): {key}")
            seen.add(key)
            case["input"] = item
            cases.append(case)
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            if not skip_invalid:
                raise ValueError(f"{path.name}: {exc}") from exc
            errors.append({"path": item["path"], "error": f"{type(exc).__name__}: {exc}"})
    variants = defaultdict(list)
    for case in cases:
        variants[case["strategy"] or case["mode_variant"]].append(case)
    return {"schema": "source_modes_v2_completed_diagnostics_v1", "input_directory": str(directory),
            "script_sha256": _digest(Path(__file__)), "manifest_sha256": _digest(manifest) if manifest.is_file() else None,
            "record_inputs": provenance, "complete_input_read": not errors, "input_errors": errors,
            "scope": {"simulations_run": 0, "world_truth_read": False,
                      "record_phase_required": "after_policy_termination",
                      "candidate_denominator": "logged candidates after duplicate/already-observed filtering; includes cache lookups",
                      "runtime": "logged extra mode planning after old scheduler; not complete decision CPU or exclusive CPU time",
                      "route_gap": "finite frozen additive mode model only; never original Q3 optimality gap",
                      "cost": "actual first-probe-through-same-source-clear cost less first arrival; also normalized to entry switch=1",
                      "exit": "distance from frozen mean exit representative to actual clear endpoint; no posterior coverage claim",
                      "selection_bias": "only actually selected executed completed local services calibrated; not causal or all-mode accuracy"},
            "overall": summarize(cases), "by_strategy": {key: summarize(value) for key, value in sorted(variants.items())},
            "paired_descriptive": paired_descriptive(cases, baseline),
            "cases": cases}


def self_test():
    """Synthetic accepted ledgers only. No simulator, source generation or I/O."""
    candidate = {"entry": [10., 0.], "valid": True, "cached": False, "reason": "nominal",
                 "cost_s": 12., "exit": [15., 0.], "hypotheses": 3, "measurements_mean": 1.}
    log = {"variant": "local", "channel": 2, "kind": "nominal", "old_probe": [20., 0.],
           "after_actual_action_count": 0, "runtime_s": .1, "candidates": [candidate], "chosen": candidate}
    record = {"evaluation_phase": "after_policy_termination", "row": {"seed": 100121, "strategy": "local"},
              "summary": {"strategy_parameters": {"source_service_modes": "local", "source_mode_log": [log]},
                          "action_history": [{"action": "measure", "channel": 2, "position": [10., 0.]},
                                             {"action": "clear", "channel": 2, "position": [15., 0.]}]},
              "history": [{"action": "/measure", "channel": 2, "position": {"x": 10., "y": 0.},
                           "response": {"accepted": True, "virtual_time_s": 8., "measure_result": "near"}},
                          {"action": "/clear", "channel": 2, "position": {"x": 15., "y": 0.},
                           "response": {"accepted": True, "virtual_time_s": 14., "clear_result": "success"}}]}
    case = analyze_record(record)
    service = case["services"][0]
    assert service["actual_cost_excluding_arrival_s"] == 12.
    assert service["exit_error_m"] == 0. and service["actual_measurements"] == 1
    assert service["changed_first_probe"] is True and service["normalized_actual_minus_predicted_s"] == 0.
    assert case["ledger_max_residual_s"] == 0.
    compact = copy.deepcopy(record)
    compact["summary"]["strategy_parameters"]["source_mode_log"][0]["chosen"] = {
        "entry": [10., 0.], "exit": [15., 0.], "cost_s": 12., "is_probe": True}
    assert analyze_record(compact)["services"][0]["predicted_measurements"] == 1.
    compact_log = compact["summary"]["strategy_parameters"]["source_mode_log"][0]
    compact["summary"]["strategy_parameters"]["source_mode_limits"] = {"supports": 3}
    compact_log["candidates"] = [{**candidate, "entry": [20., 0.]}, candidate,
                                  {**candidate, "entry": [0., 0.]}, {**candidate, "entry": [30., 0.]}]
    assert analyze_record(compact)["services"][0]["center_atom_near_mass_at_least"] == 1/3
    assert analyze_record(compact)["services"][0]["candidate_role"] == "center"
    no_switch = copy.deepcopy(record)
    no_switch["summary"]["strategy_parameters"]["source_mode_log"][0]["channel"] = 1
    for item in no_switch["history"]:
        item["channel"] = 1
        item["response"]["virtual_time_s"] -= 1
    for item in no_switch["summary"]["action_history"]:
        item["channel"] = 1
    assert analyze_record(no_switch)["services"][0]["normalized_actual_minus_predicted_s"] == 0.
    recovered = copy.deepcopy(record)
    del recovered["summary"]["strategy_parameters"]["source_mode_log"][0]["chosen"]
    assert analyze_record(recovered)["services"][0]["selection_recovery"] == "recovered_unique_immediate_actual_entry"
    joint = copy.deepcopy(record)
    joint["summary"]["strategy_parameters"]["source_mode_log"] = [{
        "variant": "joint", "after_actual_action_count": 0, "old_selected": ["cover", None],
        "selected": ["source", 2], "groups": [{"kind": "cover", "channel": None},
            {"kind": "nominal", "channel": 2, "candidates": [{"valid": False, "cached": True}, candidate]}],
        "route": {"order": [[1, 0], [0, 0]], "cost_s": 100., "lower_bound_s": 90., "exact": False,
                  "expanded": 5, "runtime_s": .02}}]
    joint_case = analyze_record(joint)
    assert joint_case["services"][0]["actual_cost_excluding_arrival_s"] == 12.
    assert joint_case["route_gap_s"] == [10.] and joint_case["route_gap_fraction"] == [.1]
    assert joint_case["invalid_candidate_lookups"] == 1 and joint_case["cached_lookups"] == 1
    assert joint_case["task_changes"]["changed_kind_or_channel"] == 1
    failed = copy.deepcopy(record)
    failed["history"][-1]["response"].update(clear_result="no_target_in_range", virtual_time_s=12.)
    failed_case = analyze_record(failed)
    assert failed_case["services"][0]["status"] == "same_source_successful_clear_absent"
    assert summarize([failed_case])["calibration"]["complete_same_source_services"] == 0
    assert _task_change({"old_selected": ["cover", None], "selected": ["cover", None]}) == "cover_position_unlogged"
    bad_phase = copy.deepcopy(record)
    bad_phase["evaluation_phase"] = "active"
    try:
        analyze_record(bad_phase)
    except ValueError:
        pass
    else:
        raise AssertionError("active record was accepted")
    return {"passed": True, "checks": "entry cost/switch/exit; valid-filtered joint index; recovery; gap; unfinished; missing cover identity; terminated-only",
            "simulations_run": 0, "files_written": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="completed pilot directory containing manifest.json and records/")
    parser.add_argument("--output", type=Path, help="new diagnostic JSON path")
    parser.add_argument("--baseline", default="v1", help="row.strategy label for descriptive paired bookkeeping")
    parser.add_argument("--skip-invalid", action="store_true", help="explicitly retain input errors and mark partial rather than stop")
    parser.add_argument("--overwrite", action="store_true", help="allow replacement of this diagnostic output only")
    parser.add_argument("--self-test", action="store_true", help="exercise synthetic records without reading or writing files")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False))
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --self-test")
    output = args.output.resolve()
    input_root = args.input.resolve()
    if output == input_root/"manifest.json" or input_root/"records" in output.parents:
        parser.error("output must not overwrite manifest or a raw record")
    if output.exists() and not args.overwrite:
        parser.error("output already exists; use a new path or explicit --overwrite")
    result = analyze_directory(input_root, skip_invalid=args.skip_invalid, baseline=args.baseline)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "records": result["overall"]["records"],
                      "complete_input_read": result["complete_input_read"],
                      "by_strategy": result["by_strategy"]}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
