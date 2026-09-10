"""Read-only independent physical/ledger audit of completed Q3 rollout traces.

Only this script's audit JSON is written. No solver, simulator or experiment
driver is imported or invoked. Ground truth is read exclusively from archived
post-session evaluation records. A running experiment can be audited as a
snapshot; --require-complete additionally requires its final complete manifest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
DISTANCE_TOLERANCE_M = 1e-7
TIME_TOLERANCE_S = 1.01e-6


def check(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def coordinates(value):
    pair = (value["x"], value["y"]) if isinstance(value, dict) else tuple(value)
    check(len(pair) == 2, "Position must have two coordinates")
    check(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
              for v in pair), "Position contains a nonfinite/non-numeric coordinate")
    check(all(abs(v) <= 2_000_000 for v in pair), "Position exceeds public coordinate bound")
    return tuple(float(v) for v in pair)


def equal_number(actual, expected, label, tolerance=TIME_TOLERANCE_S):
    check(isinstance(actual, (int, float)) and not isinstance(actual, bool)
          and math.isfinite(actual), f"{label}: nonfinite/non-numeric value")
    check(abs(actual - expected) <= tolerance,
          f"{label}: recorded {actual!r}, independently reconstructed {expected!r}")


def seven_point_cover(points):
    """Certify O + six regular ring points, allowing measured coordinate error.

    For ideal radius a the maximum nearest-station distance on the 1800 m
    arena is max(a/sqrt(3), sqrt(1800^2+a^2-sqrt(3)*1800*a)) in the accepted
    range. Moving each station by at most delta increases this by at most
    delta. This uses no sampling, source truth, or solver geometry routines.
    """
    points = [coordinates(point) for point in points]
    check(len(points) == 7, "Coverage certificate does not specify seven stations")
    center_index = min(range(7), key=lambda i: math.hypot(*points[i]))
    center = points.pop(center_index)
    check(math.hypot(*center) <= 1e-6, "Seven-point cover lacks an origin station")
    points.sort(key=lambda p: math.atan2(p[1], p[0]))
    radius = sum(math.hypot(*point) for point in points) / 6
    check(1122.0 <= radius <= 1733.0, "Ring radius lies outside the analytic-cover regime")
    angle = math.atan2(points[0][1], points[0][0])
    delta = max([math.hypot(*center)] + [
        math.dist(point, (radius * math.cos(angle + index * math.pi / 3),
                          radius * math.sin(angle + index * math.pi / 3)))
        for index, point in enumerate(points)
    ])
    worst = max(radius / math.sqrt(3),
                math.sqrt(1800**2 + radius**2 - math.sqrt(3) * 1800 * radius)) + delta
    check(worst < 1000 - DISTANCE_TOLERANCE_M,
          f"Reported stations do not have a proven 1000 m cover: bound {worst}")
    return {"ring_radius_m": radius, "coordinate_perturbation_m": delta,
            "worst_nearest_station_distance_upper_bound_m": worst,
            "coverage_margin_m": 1000 - worst}


def audit_record(record, cases, configurations, max_actions):
    """Audit one decoded record; raise ValueError at its first discrepancy."""
    check(record["data_origin"] == "synthetic_research", "Unexpected trace origin")
    check(record["evaluation_phase"] == "after_session_termination", "Evaluation phase is not post-session")
    row, summary, evaluation = record["row"], record["summary"], record["evaluation"]
    check(isinstance(summary, dict), "Trace has no strategy summary")
    truth = evaluation["ground_truth"]
    case_id, strategy = row["case_id"], row["strategy"]
    check(case_id in cases and strategy in configurations, "Unknown case/configuration identity")
    check(truth == cases[case_id], "Archived evaluation truth differs from fixed cases.json")
    check(record["config"] == configurations[strategy], "Trace configuration differs from manifest")
    check(row["case_sha256"] == canonical_hash(truth), "Case hash mismatch")
    check(row["config_sha256"] == canonical_hash(record["config"]), "Configuration hash mismatch")
    token = hashlib.sha256((case_id + "\0" + strategy).encode("utf-8")).hexdigest()
    check(row["trace"] == f"traces/{token}.json.gz", "Trace filename/identity hash mismatch")
    check(truth["problem"] == summary["problem"] == row["problem"] == 3, "Non-Q3 record")
    check(truth["case_id"] == evaluation["case_id"] == case_id, "Case identity mismatch")
    check(truth["seed"] == row["seed"], "Case seed mismatch")
    sources = {}
    for source in truth["sources"]:
        channel = source["channel"]
        check(type(channel) is int and 1 <= channel <= 20 and channel not in sources,
              "Source channels must be unique integers in 1..20")
        check(source["orientation_deg"] is None, "Q3 contains a directional source")
        point = coordinates((source["x"], source["y"]))
        check(math.hypot(*point) <= 1800 + DISTANCE_TOLERANCE_M, "Source lies outside arena")
        check(1000 <= source["reception_radius_m"] <= 1500, "Source radius outside public bounds")
        sources[channel] = source
    check(10 <= len(sources) <= 16, "True source count outside 10..16")

    history, policy_history = record["history"], summary["action_history"]
    check(history and history[0]["action"] == "/enter", "Missing initial /enter")
    check(history[-1]["action"] == "/exit", "Missing explicit final /exit")
    check(sum(item["action"] == "/enter" for item in history) == 1, "Repeated /enter")
    check(sum(item["action"] == "/exit" for item in history) == 1, "Repeated/nonfinal /exit")
    check(len(history) <= max_actions, "Accepted action budget exceeded")
    check(len(policy_history) == len(history) - 2, "Policy/engine action counts differ")

    position, current_channel, cleared, detected = (0.0, 0.0), 1, set(), set()
    components_us = dict.fromkeys(COMPONENTS, 0)
    exact_movement_s = 0.0
    clear_indices, seen_measurements = {}, {}
    measurement_count = failed_clears = clear_attempts = certified_clears = 0
    max_clear_distance = max_certified_distance = max_bearing_error = 0.0
    max_time_error, warnings, completed_cover_points = 0.0, [], 0
    for index, item in enumerate(history):
        check(item["index"] == index, f"History index mismatch at action {index}")
        response, action = item["response"], item["action"]
        check(response["accepted"] is True, f"Unaccepted response recorded at action {index}")
        target = coordinates(item["position"])
        if action in ("/enter", "/exit"):
            check(target == position and item["channel"] is None, f"{action} changes position/channel")
            if action == "/enter":
                check(response["max_virtual_duration_s"] == 360000, "Unexpected virtual budget")
            else:
                check(response["exit_reason"] == "user_exit", "Exit response lacks user_exit")
        else:
            check(action in ("/measure", "/clear"), f"Illegal action {action!r}")
            channel = item["channel"]
            check(type(channel) is int and 1 <= channel <= 20, "Action channel outside 1..20")
            policy_action = policy_history[index - 1]
            check(policy_action["action"] == action[1:] and policy_action["channel"] == channel
                  and coordinates(policy_action["position"]) == target,
                  f"Policy/engine action mismatch at index {index}")
            equal_number(policy_action["virtual_time_s"], response["virtual_time_s"], "Policy action time")
            movement_s = math.hypot(target[0] - position[0], target[1] - position[1]) / 5
            components_us["movement_s"] += round(movement_s * 1_000_000)
            exact_movement_s += movement_s
            position = target
            source = sources.get(channel) if channel not in cleared else None
            distance = math.hypot(source["x"] - target[0], source["y"] - target[1]) if source else math.inf
            if channel in cleared:
                warnings.append(f"Redundant legal {action} on already-cleared channel {channel} at {index}")
            if action == "/measure":
                measurement_count += 1
                components_us["switching_s"] += int(channel != current_channel) * 1_000_000
                components_us["detection_s"] += 5_000_000
                current_channel = channel
                observed = response["measure_result"]
                expected = ("no_signal" if source is None or distance > source["reception_radius_m"]
                            else "near" if distance <= 5 else "direction")
                check(observed == expected, f"Physical measurement mismatch at action {index}")
                check(policy_action["result"] == observed, f"Policy measurement mismatch at {index}")
                if observed in ("direction", "near"):
                    detected.add(channel)
                if observed == "direction":
                    bearing = response["svd_deg"]
                    check(math.isfinite(bearing) and 0 <= bearing < 360, "Invalid bearing")
                    equal_number(bearing * 100, round(bearing * 100), "Bearing centidegree quantization", 1e-7)
                    actual = math.degrees(math.atan2(source["y"] - target[1], source["x"] - target[0])) % 360
                    error = abs((bearing - actual + 180) % 360 - 180)
                    max_bearing_error = max(max_bearing_error, error)
                    check(error <= 1 + 1e-10, f"Bearing error exceeds one degree at {index}: {error}")
                    equal_number(policy_action["bearing_deg"], bearing, "Policy bearing", 1e-10)
                key = (channel, target)
                value = (observed, response.get("svd_deg"))
                if channel not in cleared:
                    check(key not in seen_measurements or seen_measurements[key] == value,
                          f"Repeated stationary measurement changes at action {index}")
                    seen_measurements[key] = value
            else:
                clear_attempts += 1
                observed = response["clear_result"]
                success = source is not None and distance <= 20
                check(observed == ("success" if success else "no_target_in_range"),
                      f"Physical clear mismatch at action {index}: distance={distance}")
                check(policy_action["result"] == observed, f"Policy clear result mismatch at {index}")
                components_us["optical_s"] += 3_000_000
                if success:
                    components_us["removal_s"] += 2_000_000
                    cleared.add(channel)
                    clear_indices[channel] = index - 1
                    max_clear_distance = max(max_clear_distance, distance)
                else:
                    failed_clears += 1
                if policy_action["phase"] == "certified_clear":
                    certified_clears += 1
                    check(success and distance <= 19.9 + DISTANCE_TOLERANCE_M,
                          f"Certified clear exceeds 19.9 m or fails at {index}: {distance}")
                    max_certified_distance = max(max_certified_distance, distance)
                # Deliberately leave current_channel unchanged after /clear.
        expected_time = sum(components_us.values()) / 1_000_000
        error = abs(response["virtual_time_s"] - expected_time)
        max_time_error = max(max_time_error, error)
        equal_number(response["virtual_time_s"], expected_time, f"Action {index} microsecond ledger")
        check(expected_time < 360000, "Virtual budget reached/exceeded before explicit exit")

    counts = {"source_total": len(sources), "cleared_total": len(cleared), "measurement_count": measurement_count,
              "failed_clear_count": failed_clears, "action_count": len(history)}
    for key, expected in counts.items():
        check(evaluation[key] == row[key] == expected, f"Row/evaluation {key} differs from actions")
    check(evaluation["cleared_channels"] == summary["cleared_channels"] == sorted(cleared), "Cleared-channel mismatch")
    check(evaluation["remaining_channels"] == sorted(set(sources) - cleared), "Remaining-channel mismatch")
    check(summary["detected_channels"] == sorted(detected), "Detected-channel mismatch")
    check(summary["unresolved_channels"] == sorted(detected - cleared), "Unresolved-channel mismatch")
    check(summary["accepted_actions"] == len(history), "Summary accepted-action count mismatch")
    check(summary["measurement_count"] == measurement_count and summary["clear_attempt_count"] == clear_attempts,
          "Summary measurement/clear-attempt count mismatch")
    check(summary["cleared_count"] == len(cleared) and summary["detected_count"] == len(detected),
          "Summary cleared/detected total mismatch")
    all_cleared = len(cleared) == len(sources)
    check(evaluation["all_cleared"] is all_cleared and row["all_cleared"] is all_cleared,
          "Recorded all-cleared differs from physical clear results")
    certified = summary["completion_certified_under_model"]
    check(type(certified) is bool and row["completion_certified"] is certified
          and summary["all_cleared"] is certified, "Certificate fields disagree")
    check(not certified or all_cleared, "False all-clear certificate against post-session truth")

    cover = seven_point_cover(summary["coverage_points"])
    for point in summary["coverage_points"]:
        indices = [i for i, action in enumerate(policy_history)
                   if action["phase"] == "coverage" and coordinates(action["position"]) == coordinates(point)]
        if not indices:
            continue
        check(all(policy_history[i]["action"] == "measure" for i in indices), "Coverage phase contains non-measure action")
        finish = max(indices)
        scanned = {policy_history[i]["channel"] for i in indices}
        previously_cleared = {channel for channel, clear_index in clear_indices.items() if clear_index < finish}
        if scanned | previously_cleared == set(range(1, 21)):
            completed_cover_points += 1
    check(summary["coverage_points_total"] == 7, "Coverage station count mismatch")
    check(summary["coverage_points_visited"] == completed_cover_points, "Reported completed scans differ from trace")
    if summary["coverage_complete"]:
        check(completed_cover_points == 7, "Full-coverage claim lacks seven complete channel scans")
    if certified:
        if summary["completion_reason"] == "source_count_upper_bound_reached":
            check(len(cleared) == 16, "Upper-bound certificate has fewer than sixteen clears")
            check(history[-2]["action"] == "/clear" and history[-2]["response"]["clear_result"] == "success",
                  "Upper-bound termination contains extra actions after final clear")
        else:
            check(summary["completion_reason"] == "coverage_exhausted_and_all_detected_cleared",
                  "Unrecognized completion-certificate justification")
            check(summary["coverage_complete"] and completed_cover_points == 7 and detected <= cleared,
                  "Coverage completion certificate lacks observation-only evidence")
            check(10 <= len(detected | cleared) <= 16, "Certificate violates observed source-count consistency")
    check(row["completion_reason"] == summary["completion_reason"], "Completion reason mismatch")

    time_s = sum(components_us.values()) / 1_000_000
    for label, container in (("row", row), ("evaluation", evaluation), ("summary", summary),
                             ("state", record["final_client_state"])):
        equal_number(container["virtual_time_s"], time_s, f"{label} total time")
    for component in COMPONENTS:
        expected = components_us[component] / 1_000_000
        equal_number(row[component], expected, f"Row {component}")
        equal_number(evaluation["time_breakdown_s"][component], expected, f"Evaluation {component}")
        unrounded = exact_movement_s if component == "movement_s" else expected
        equal_number(summary["time_breakdown"][component], unrounded, f"Summary {component}", 1e-6)
    equal_number(row["movement_m"], components_us["movement_s"] / 200_000, "Recorded movement distance", 1e-5)
    equal_number(row["time_per_source_s"], time_s / len(sources), "Per-source time")
    equal_number(row["cleared_fraction"], len(cleared) / len(sources), "Cleared fraction")
    equal_number(evaluation["cleared_fraction"], len(cleared) / len(sources), "Evaluation cleared fraction")
    if cleared:
        equal_number(row["mean_time_per_cleared_s"], time_s / len(cleared), "Per-cleared time")
        equal_number(evaluation["mean_time_per_cleared_s"], time_s / len(cleared), "Evaluation per-cleared time")
    state = record["final_client_state"]
    check(state["session"] == evaluation["simulator_stop_reason"] == "exited", "Final session did not exit")
    check(state["current_channel"] == current_channel, "Final radio channel differs; /clear must not retune")
    check(coordinates(state["position"]) == position, "Final client position mismatch")
    check(state["accepted_actions"] == len(history), "Final client action count mismatch")
    check({int(channel) for channel, value in state["sources"].items() if value["status"] == "cleared"} == cleared,
          "Final client cleared state differs from actions")
    check(row["accepted_exit"] is True, "Row does not acknowledge explicit accepted exit")
    expected_success = bool(all_cleared and certified and not row["error"])
    check(row["successful"] is expected_success, "Successful-run flag is inconsistent")
    if expected_success:
        check(summary["error"] is None and summary["exit_error"] is None
              and record["exception_traceback"] is None, "Successful run retains errors")
    return {"case_id": case_id, "strategy": strategy, "audit_passed": True,
            "run_successful": expected_success, "all_cleared": all_cleared, "certified": certified,
            "source_count": len(sources), "cleared_count": len(cleared), "action_count": len(history),
            "measurement_count": measurement_count, "failed_clear_count": failed_clears,
            "successful_clear_count": len(cleared), "certified_clear_count": certified_clears,
            "max_successful_clear_distance_m": max_clear_distance,
            "max_certified_clear_distance_m": max_certified_distance,
            "max_bearing_error_deg": max_bearing_error, "max_action_time_error_s": max_time_error,
            "virtual_time_s": time_s, "microsecond_rounding_delta_s":
                components_us["movement_s"] / 1_000_000 - exact_movement_s,
            "time_components_s": {key: value / 1_000_000 for key, value in components_us.items()},
            "coverage_proof": cover, "completed_coverage_scans": completed_cover_points,
            "completion_reason": summary["completion_reason"], "warnings": warnings,
            "case_sha256": canonical_hash(truth), "config_sha256": canonical_hash(record["config"])}


def current_source_hashes(declared):
    result = {}
    for name in declared:
        path = (ROOT / name).resolve()
        check(path.is_relative_to(ROOT) and path.is_file(), f"Invalid/missing source path: {name}")
        result[name] = digest(path.read_bytes())
    return result


def shared_observations(record):
    """Pre-clear observations must agree across policies at the same point."""
    result, cleared = {}, set()
    for item in record["history"]:
        if item["action"] == "/clear" and item["response"]["clear_result"] == "success":
            cleared.add(item["channel"])
        elif item["action"] == "/measure" and item["channel"] not in cleared:
            key = (record["row"]["case_id"], item["channel"], coordinates(item["position"]))
            result[key] = (item["response"]["measure_result"], item["response"].get("svd_deg"))
    return result


def audit_csv(raw, rows, baseline_name):
    """Check the final CSV and paired differences against audited trace rows."""
    table = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    check(len(table) == len(rows), "Final runs.csv count differs from completed traces")
    seen = set()
    for line in table:
        key = (line["case_id"], line["strategy"])
        check(key in rows and key not in seen, "CSV contains unknown/duplicate case-strategy pair")
        seen.add(key)
        row = rows[key]
        for field, value in row.items():
            expected = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        if isinstance(value, dict) else "" if value is None else str(value))
            check(line[field] == expected, f"CSV {key} field {field} differs from trace row")
        baseline = rows[(line["case_id"], baseline_name)]
        saved = baseline["virtual_time_s"] - row["virtual_time_s"]
        valid = baseline["successful"] and row["successful"]
        equal_number(float(line["paired_baseline_time_s"]), baseline["virtual_time_s"], "CSV paired baseline time")
        equal_number(float(line["paired_seconds_saved"]), saved, "CSV paired time saving")
        check(line["pair_successful"] == str(valid), "CSV paired success flag mismatch")
        check(line["paired_win"] == (str(saved > 1e-6) if valid else ""), "CSV paired win flag mismatch")
    return {"performed": True, "passed": True, "rows_checked": len(table),
            "paired_differences_checked": len(table)}


def audit_directory(directory, *, output=None, require_complete=False):
    directory = Path(directory).resolve()
    inputs = {name: (directory / name).read_bytes() for name in ("manifest.json", "cases.json", "configs.json")}
    for name in ("runs.csv", "summary.json"):
        if (directory / name).is_file():
            inputs[name] = (directory / name).read_bytes()
    manifest = json.loads(inputs["manifest.json"])
    cases_list = json.loads(inputs["cases.json"])
    configs_list = json.loads(inputs["configs.json"])
    check(manifest["data_origin"] == "synthetic_research" and manifest["problem"] == 3, "Only synthetic Q3 data accepted")
    check(manifest["official_practice"] is False and manifest["official_formal"] is False, "Official data not accepted")
    check(canonical_hash(cases_list) == manifest["cases_sha256"], "Fixed case archive hash mismatch")
    check(configs_list == manifest["configs"] and canonical_hash(configs_list) == manifest["config_content_sha256"],
          "Configuration archive differs from fixed manifest")
    check(digest(inputs["configs.json"]) == manifest["original_config_file_sha256"], "Original config-file hash mismatch")
    cases = {case["case_id"]: case for case in cases_list}
    check(len(cases) == len(cases_list), "Duplicate fixed case IDs")
    configurations = {manifest["baseline"]["name"]: None,
                      **{entry["name"]: entry["rollout_config"] for entry in configs_list}}
    check(len(configurations) == len(configs_list) + 1, "Duplicate strategy names")
    check(manifest["baseline"]["variant"] == "efficient"
          and manifest["baseline"]["efficient_config_override"] is None, "Baseline specification changed")
    sources_start = current_source_hashes(manifest["source_sha256_start"])
    findings = []
    if sources_start != manifest["source_sha256_start"]:
        findings.append("Current source hashes differ from the experiment's initial source snapshot")
    if manifest.get("source_sha256_end") is not None and manifest["source_sha256_end"] != manifest["source_sha256_start"]:
        findings.append("Experiment start/end source hashes differ")
    if manifest.get("source_changed_during_run") is True:
        findings.append("Manifest reports source_changed_during_run")
    traces = sorted((directory / "traces").glob("*.json.gz"))
    records, identities, rows, cross_policy_observations = [], set(), {}, {}
    shared_observation_checks = 0
    for path in traces:
        raw = path.read_bytes()
        item = {"trace": path.relative_to(directory).as_posix(), "trace_sha256": digest(raw)}
        try:
            decoded = gzip.decompress(raw)
            item["decoded_json_sha256"] = digest(decoded)
            record = json.loads(decoded)
            check(record["row"]["trace"] == item["trace"], "Stored trace path differs from file")
            identity = (record["row"]["case_id"], record["row"]["strategy"])
            check(identity not in identities, "Duplicate case/strategy record")
            identities.add(identity)
            audited = audit_record(record, cases, configurations, manifest["max_actions"])
            observations = shared_observations(record)
            for key, value in observations.items():
                if key in cross_policy_observations:
                    check(cross_policy_observations[key] == value,
                          f"Policies receive different pre-clear observations at identical case/channel/point: {key}")
                    shared_observation_checks += 1
            cross_policy_observations.update(observations)
            rows[identity] = record["row"]
            item.update(audited)
        except Exception as error:
            item.update(audit_passed=False, finding=f"{type(error).__name__}: {error}")
        records.append(item)
    sources_end = current_source_hashes(manifest["source_sha256_start"])
    if sources_end != sources_start:
        findings.append("Source files changed during this audit")
    expected_identities = {(case, config) for case in cases for config in configurations}
    missing = sorted(expected_identities - identities)
    check(manifest["expected_runs"] == len(expected_identities), "Manifest expected-runs cardinality mismatch")
    complete = (manifest.get("run_status") == "completed" and not missing
                and len(traces) == manifest["expected_runs"] == manifest.get("completed_runs")
                and manifest.get("source_changed_during_run") is False)
    changed_inputs = [name for name, raw in inputs.items() if (directory / name).read_bytes() != raw]
    if any(name in ("cases.json", "configs.json") for name in changed_inputs):
        findings.append("Fixed cases/configuration archive changed during audit")
    if require_complete and not complete:
        findings.append("Complete frozen experiment required; manifest/count/source-finalization checks incomplete")
    if require_complete and changed_inputs:
        findings.append("Inputs changed during a required-complete audit; rerun on the final snapshot")
    csv_result = {"performed": False, "reason": "Final complete experiment and all valid traces required"}
    if complete and len(rows) == len(traces):
        try:
            check("runs.csv" in inputs, "Completed experiment lacks runs.csv")
            csv_result = audit_csv(inputs["runs.csv"], rows, manifest["baseline"]["name"])
        except (ValueError, KeyError, TypeError) as error:
            csv_result = {"performed": True, "passed": False, "finding": str(error)}
            findings.append(f"Final CSV audit failed: {error}")
    passed = [record for record in records if record["audit_passed"]]
    result = {
        "audit_kind": "independent_q3_rollout_physics_ledger_and_certificate",
        "created_utc": datetime.now(timezone.utc).isoformat(), "directory": str(directory),
        "stage": manifest["stage"], "experiment_git_revision": manifest["git_revision"],
        "manifest_run_status": manifest.get("run_status"), "require_complete": require_complete,
        "complete_experiment_snapshot": complete, "expected_runs": manifest["expected_runs"],
        "traces_audited": len(records), "traces_passed": len(passed), "traces_failed": len(records) - len(passed),
        "audit_passed": bool(records) and len(passed) == len(records) and not findings,
        "all_audited_runs_successful": bool(passed) and len(passed) == len(records)
            and all(record["run_successful"] for record in passed),
        "missing_case_strategy_pairs": [list(pair) for pair in missing], "findings": findings,
        "inputs_changed_during_audit": changed_inputs,
        "input_sha256": {name: digest(raw) for name, raw in inputs.items()},
        "experiment_source_sha256": manifest["source_sha256_start"],
        "current_source_sha256_start": sources_start, "current_source_sha256_end": sources_end,
        "audit_script_sha256": digest(Path(__file__).read_bytes()),
        "source_hashes_match_experiment": sources_start == sources_end == manifest["source_sha256_start"],
        "cross_policy_same_point_observations_checked": shared_observation_checks,
        "csv_audit": csv_result,
        "scope_note": "Post-session synthetic ground truth checks the recorded actions. This audit does not prove "
            "runtime information isolation, posterior accuracy, global optimality, or official-simulator equivalence.",
        "accounting_note": "Movement is independently accumulated as L/5 and rounded per accepted action to "
            "the local engine's microsecond ledger. Measure=5s plus radio switch=1s; clear=3s plus success=2s; "
            "clear does not retune; enter/exit cost zero.",
        "tolerances": {"distance_m": DISTANCE_TOLERANCE_M, "ledger_time_s": TIME_TOLERANCE_S,
                       "bearing_error_deg": 1.0 + 1e-10},
        "totals": {key: sum(record[key] for record in passed) for key in
                   ("action_count", "measurement_count", "successful_clear_count", "certified_clear_count", "failed_clear_count")},
        "maxima": {key: max((record[key] for record in passed), default=None) for key in
                   ("max_successful_clear_distance_m", "max_certified_clear_distance_m", "max_bearing_error_deg", "max_action_time_error_s")},
        "certificate_reasons": dict(Counter(record["completion_reason"] for record in passed)),
        "records": records,
    }
    destination = Path(output).resolve() if output else directory / "audit.json"
    check(destination.suffix == ".json" and not destination.is_relative_to(directory / "traces")
          and destination not in [directory / name for name in
                                  ("manifest.json", "cases.json", "configs.json", "runs.csv", "summary.json")],
          "Audit output must not overwrite experiment inputs")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit_directory(args.directory, output=args.output, require_complete=args.require_complete)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Audit could not start: {error}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in ("audit_passed", "complete_experiment_snapshot", "traces_audited",
                     "traces_passed", "traces_failed", "all_audited_runs_successful", "findings", "maxima")},
                     ensure_ascii=False, indent=2))
    if result["traces_failed"]:
        print(json.dumps([{key: record[key] for key in ("trace", "finding")}
                          for record in result["records"] if not record["audit_passed"]], ensure_ascii=False, indent=2))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
