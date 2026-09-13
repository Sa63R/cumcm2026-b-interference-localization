"""Audit completed clear-region traces without running a policy or simulator.

Reconstruct geometry from physical responses and independently validated
deductions. Source truth is used only for separate post-session physical
cross-checks; it never supplies a clearance or discovery certificate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from localization.omni import OmniCandidateRegion
from planning.disk_cover import disk_cover_radius
from simulator_client.state import Position

GEOMETRY_TOLERANCE_M = 1e-6


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_object(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return digest_bytes(encoded)


def as_point(value):
    return Position.coerce(value)


def point_to_polygon(region, query):
    """Independent distance to every segment of the complete outer polygon."""
    check(bool(region.vertices), "Silence inference has an empty source region")
    if region.contains((query.x, query.y)):
        return 0.0
    best = math.inf
    for first, second in zip(region.vertices, region.vertices[1:] + region.vertices[:1]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        norm2 = dx * dx + dy * dy
        t = (max(0.0, min(1.0, ((query.x-first[0])*dx + (query.y-first[1])*dy) / norm2))
             if norm2 else 0.0)
        best = min(best, math.hypot(query.x-first[0]-t*dx, query.y-first[1]-t*dy))
    return best


def audit_inference(inference, regions, known, cleared, provenance, prefix_time):
    """Validate the claimed proof against prefix evidence before installing it."""
    channel = inference["channel"]
    check(channel in known and channel not in cleared,
          "Inferred silence is not on a known, uncleared Q3 channel")
    check(inference["inference_kind"] == "inferred_no_signal"
          and inference["physical_measurement"] is False,
          "A deduction was not explicitly labelled nonphysical")
    check(inference["virtual_time_s"] == prefix_time,
          "Inference virtual time is not its actual-prefix time")
    region, query = regions[channel], as_point(inference["position"])
    check(bool(region.observations) and bool(region.vertices),
          "Inference lacks a nonempty region with an actual bearing")
    check(inference["outer_region_vertex_count"] == len(region.vertices),
          "Inference vertex count does not match the reconstructed prefix")
    margin = inference["margin_m"]
    check(type(margin) in (int, float) and math.isfinite(margin) and margin > 0,
          "Invalid inference margin")
    method, witness_origin = inference["method"], None
    if method == "history_negative_bisector":
        witness = as_point(inference["witness_no_signal_position"])
        witness_key = (witness.x, witness.y)
        check(witness_key in provenance.get(channel, {}),
              "Historical inference witness has no earlier actual or proven negative")
        witness_origin = provenance[channel][witness_key]
        qx, qy, nx, ny = map(Fraction, (query.x, query.y, witness.x, witness.y))
        # Deliberately expand the two squared distances, rather than calling
        # the policy's certificate helper or trusting its reported lower bound.
        differences = []
        for x, y in region.vertices:
            x, y = Fraction(x), Fraction(y)
            differences.append((x-qx)**2 + (y-qy)**2 - (x-nx)**2 - (y-ny)**2)
        minimum = min(differences)
        norm2 = (qx-nx)**2 + (qy-ny)**2
        check(norm2 > 0 and minimum > 0,
              "Historical negative does not imply query silence on the whole region")
        check((minimum / 2)**2 > Fraction(margin)**2 * norm2,
              "Historical bisector margin fails exact rational verification")
        upper = Fraction(inference["query_witness_distance_upper_m"])
        claimed = Fraction(inference["signed_bisector_distance_lower_m"])
        check(upper > 0 and upper**2 >= norm2,
              "Reported witness-distance upper bound is not an upper bound")
        check(Fraction(margin) < claimed <= minimum / (2*upper),
              "Reported signed-distance lower bound is not conservative")
        check(0 < Fraction(inference["squared_distance_difference_lower_m2"]) <= minimum,
              "Reported squared-distance lower bound is not conservative")
    else:
        check(method in {"enclosing_disk", "polygon_edges"},
              f"Unrecognized silence certificate method: {method}")
        lower = point_to_polygon(region, query)
        check(lower > 1500 + margin, "Public-radius-cap silence is not proven")
        check(inference["maximum_reception_radius_m"] == 1500,
              "Silence inference changed the public maximum reception radius")
        claimed = inference["distance_lower_m"]
        check(1500 + margin < claimed <= lower + GEOMETRY_TOLERANCE_M,
              "Reported public-radius-cap lower bound is unsupported")
    region.observe_no_signal(query)
    provenance.setdefault(channel, {}).setdefault((query.x, query.y), "validated_inference")
    return method, witness_origin


def audit_trace(trace, manifest):
    row, report, evaluation = trace["row"], trace["report"], trace["evaluation"]
    check(trace["data_origin"] == "synthetic_research"
          and trace["evaluation_phase"] == "after_policy_termination",
          "Unexpected trace provenance")
    check(report is not None, "Policy produced no report")
    check(trace["exception_traceback"] is None and not row["errors"]
          and report["error"] is None and report["exit_error"] is None,
          "Run retains policy, exit or harness errors")
    method = row["method"]
    check(trace["spec"] == manifest["specs"][method], "Trace specification differs from manifest")
    check(row["spec_sha256"] == manifest["spec_sha256"][method] == digest_object(trace["spec"]),
          "Specification digest mismatch")
    truth = evaluation["ground_truth"]
    check(row["case_sha256"] == digest_object(truth), "Archived case digest mismatch")
    check(truth["problem"] == report["problem"] == 3
          and truth["case_id"] == row["case_id"] == evaluation["case_id"]
          and truth["seed"] == row["seed"] and row["stage"] == manifest["stage"],
          "Case identity or problem mismatch")
    sources = {source["channel"]: source for source in truth["sources"]}
    check(len(sources) == len(truth["sources"]) and 10 <= len(sources) <= 16,
          "Invalid source count or duplicate source channels")
    for channel, source in sources.items():
        check(type(channel) is int and 1 <= channel <= 20
              and source["orientation_deg"] is None
              and 1000 <= source["reception_radius_m"] <= 1500
              and math.hypot(source["x"], source["y"]) <= 1800 + GEOMETRY_TOLERANCE_M,
              "Archived source violates the Q3 public model")

    history, actual = report["action_history"], trace["observation_history"]
    state = trace["final_client_state"]
    check(len(actual) == len(history) + 2 and actual[0]["action"] == "/enter"
          and actual[-1]["action"] == "/exit", "Missing enter/exit or unreported physical actions")
    check(all(item["response"]["accepted"] is True for item in actual),
          "A physical request was not accepted")
    check([item["index"] for item in actual] == list(range(len(actual))),
          "Physical response indices are not contiguous")
    check(actual[0]["response"]["virtual_time_s"] == 0, "Nonzero starting virtual time")
    check(state["session"] == evaluation["simulator_stop_reason"] == "exited"
          and row["accepted_exit"] is True, "Session did not accept a normal exit")

    inferences = report["strategy_parameters"].get("inferred_no_signal_constraints", [])
    indices = [item["after_actual_action_count"] for item in inferences]
    check(all(type(i) is int and 0 <= i <= len(history) for i in indices)
          and indices == sorted(indices), "Invalid or unordered inference prefix indices")
    regions, physical_regions, near, provenance, physical_negatives = {}, {}, {}, {}, {}
    known, cleared = set(), set()
    previous, current_channel, charged_us, cursor = Position(0, 0), 1, 0, 0
    components = dict(movement_s=0, switching_s=0, detection_s=0, optical_s=0, removal_s=0)
    inference_methods, witness_origins = Counter(), Counter()
    measurement_count = clear_count = failed_clear_count = near_count = 0
    certified_count = deduction_dependent_clears = 0
    max_clear_bound = max_physical_clear_distance = 0.0

    for index in range(len(history) + 1):
        while cursor < len(inferences) and indices[cursor] == index:
            inferred = inferences[cursor]
            inference_method, origin = audit_inference(
                inferred, regions, known, cleared, provenance, charged_us / 1_000_000)
            inference_methods[inference_method] += 1
            if origin:
                witness_origins[origin] += 1
            # This separate post-session physics check is not the proof above.
            source, query = sources[inferred["channel"]], as_point(inferred["position"])
            check(math.hypot(source["x"]-query.x, source["y"]-query.y)
                  > source["reception_radius_m"], "A proved silence disagrees with archived physics")
            cursor += 1
        if index == len(history):
            break
        action, receipt = history[index], actual[index + 1]
        channel, point = action["channel"], as_point(action["position"])
        label = f"action {index}, channel {channel}"
        check(type(channel) is int and 1 <= channel <= 20, f"Invalid channel at {label}")
        check(action["action"] in {"measure", "clear"}
              and receipt["action"] == "/" + action["action"]
              and receipt["channel"] == channel
              and receipt["position"] == {"x": point.x, "y": point.y},
              f"Action differs from accepted receipt at {label}")
        check(channel not in cleared, f"Action revisits an already cleared channel at {label}")
        response = receipt["response"]
        result = action["result"]
        source = sources.get(channel)
        distance = (math.hypot(source["x"]-point.x, source["y"]-point.y)
                    if source is not None else math.inf)
        costs = {"movement_s": round(previous.distance_to(point) / 5 * 1_000_000)}
        if action["action"] == "measure":
            measurement_count += 1
            check(result == response["measure_result"], f"Measurement receipt mismatch at {label}")
            expected = ("no_signal" if source is None or distance > source["reception_radius_m"]
                        else "near" if distance <= 5 else "direction")
            check(result == expected, f"Measurement contradicts archived Q3 physics at {label}")
            costs.update(detection_s=5_000_000,
                         switching_s=int(current_channel != channel) * 1_000_000)
            current_channel = channel
            region = regions.setdefault(channel, OmniCandidateRegion())
            physical = physical_regions.setdefault(channel, OmniCandidateRegion())
            if result == "direction":
                bearing = action["bearing_deg"]
                check(bearing == response["svd_deg"], f"Bearing receipt mismatch at {label}")
                true_bearing = math.degrees(math.atan2(source["y"]-point.y, source["x"]-point.x)) % 360
                check(abs((bearing-true_bearing+180) % 360 - 180) <= 1.005 + 1e-7,
                      f"Observed bearing exceeds the public error envelope at {label}")
                region.observe(point, bearing)
                physical.observe(point, bearing)
                known.add(channel)
            elif result == "no_signal":
                region.observe_no_signal(point)
                physical.observe_no_signal(point)
                key = (point.x, point.y)
                physical_negatives.setdefault(channel, []).append(point)
                provenance.setdefault(channel, {})[key] = "physical_measurement"
            else:
                known.add(channel)
                near[channel] = point
        else:
            clear_count += 1
            check(result == response["clear_result"], f"Clear receipt mismatch at {label}")
            check(result == ("success" if source is not None and distance <= 20 else "no_target_in_range"),
                  f"Clear result contradicts archived physical distance at {label}")
            costs.update(optical_s=3_000_000, removal_s=int(result == "success") * 2_000_000)
            if action["phase"] == "certified_clear":
                check(channel in known and channel in regions and bool(regions[channel].vertices),
                      f"Certified clear has no prior observation region at {label}")
                bound = max(point.distance_to(Position(*v)) for v in regions[channel].vertices)
                raw_bound = max(point.distance_to(Position(*v)) for v in physical_regions[channel].vertices)
                deduction_dependent_clears += raw_bound > 19.9 + GEOMETRY_TOLERANCE_M
                certified_count += 1
            elif action["phase"] == "near_clear":
                check(channel in near, f"Near clear has no earlier actual near response at {label}")
                bound = point.distance_to(near[channel]) + 5
                near_count += 1
            else:
                raise ValueError(f"Clear phase lacks a per-action 19.9 m certificate at {label}: {action['phase']}")
            check(bound <= 19.9 + GEOMETRY_TOLERANCE_M,
                  f"Clearance bound {bound:.12g} exceeds 19.9 m at {label}")
            max_clear_bound = max(max_clear_bound, bound)
            max_physical_clear_distance = max(max_physical_clear_distance, distance)
            if result == "success":
                check(channel in known, f"Successful clear channel was not previously detected at {label}")
                cleared.add(channel)
            else:
                failed_clear_count += 1

        for name, amount in costs.items():
            components[name] += amount
        charged_us += sum(costs.values())
        check(action["virtual_time_s"] == response["virtual_time_s"] == charged_us / 1_000_000,
              f"Prefix virtual-time ledger mismatch at {label}")
        previous = point
    check(cursor == len(inferences), "Not all inferred constraints were replayed")
    check(10 <= len(known) <= 16, "Detected source count violates the public model")
    check(report["detected_channels"] == sorted(known) and report["cleared_channels"] == sorted(cleared)
          and report["unresolved_channels"] == sorted(known-cleared),
          "Reported detected, cleared or unresolved channels differ from actual responses")
    check(evaluation["cleared_channels"] == sorted(cleared)
          and evaluation["remaining_channels"] == sorted(sources.keys()-cleared),
          "Evaluation cleared or remaining channels differ from actual responses")
    check(report["cleared_count"] == row["cleared_total"] == evaluation["cleared_total"]
          == state["cleared_count"] == len(cleared), "Cleared count mismatch")
    check(row["source_total"] == evaluation["source_total"] == len(sources), "Source total mismatch")

    coverage = {}
    if len(known) < 16:
        for channel in sorted(set(range(1, 21))-known):
            radius = disk_cover_radius(physical_negatives.get(channel, []))
            check(radius <= 1000 + GEOMETRY_TOLERANCE_M,
                  f"Unknown channel {channel} lacks real no-signal full-arena coverage: {radius}")
            coverage[str(channel)] = radius
    complete_by_observations = (known == cleared and (len(known) == 16 or bool(coverage)))
    all_cleared = cleared == set(sources)
    check(report["completion_certified_under_model"] is True and complete_by_observations,
          "A complete observation-only termination certificate was not established")
    check(row["completion_certified"] is True and row["all_cleared"] == evaluation["all_cleared"]
          == all_cleared and report["all_cleared"] is True, "Completion status mismatch")
    check(all_cleared and row["success"] is True and failed_clear_count == 0,
          "Run is incomplete, unsuccessful, or has a failed clearance")
    check(report["clear_attempt_count"] == clear_count
          and row["failed_clear_count"] == evaluation["failed_clear_count"] == failed_clear_count,
          "Clear attempt/failure count mismatch")
    check(report["measurement_count"] == row["measurement_count"]
          == evaluation["measurement_count"] == measurement_count, "Measurement count mismatch")
    check(report["accepted_actions"] == row["action_count"] == evaluation["action_count"]
          == state["accepted_actions"] == len(actual), "Accepted action count mismatch")
    check(report["virtual_time_s"] == row["virtual_time_s"] == evaluation["virtual_time_s"]
          == state["virtual_time_s"] == actual[-1]["response"]["virtual_time_s"]
          == charged_us / 1_000_000, "Final virtual-time ledger mismatch")
    check(state["position"] == {"x": previous.x, "y": previous.y}
          and state["current_channel"] == current_channel, "Final client position/channel mismatch")
    for name, amount in components.items():
        expected = amount / 1_000_000
        check(row["time_breakdown_s"][name] == evaluation["time_breakdown_s"][name] == expected,
              f"Physical time component mismatch: {name}")
        tolerance = len(history)*0.5e-6 + 1e-9 if name == "movement_s" else 0
        check(math.isclose(report["time_breakdown"][name], expected, rel_tol=0, abs_tol=tolerance)
              and report["time_breakdown"][name] == state["time_breakdown"][name],
              f"Client estimated time component mismatch: {name}")
    check(row["penalized_time_s"] == row["virtual_time_s"], "Successful-run penalty mismatch")
    check(row["time_per_target_s"] == row["virtual_time_s"] / len(sources)
          and row["time_per_cleared_s"] == row["virtual_time_s"] / len(cleared),
          "Per-target or per-cleared time mismatch")
    return {"case_id": row["case_id"], "seed": row["seed"], "method": method,
            "case_sha256": row["case_sha256"], "audit_passed": True,
            "physical_actions": len(actual), "physical_measurements": measurement_count,
            "certified_clears": certified_count, "near_clears": near_count,
            "max_clearance_bound_m": max_clear_bound,
            "max_post_session_source_clearance_distance_m": max_physical_clear_distance,
            "clearances_requiring_validated_deductions": deduction_dependent_clears,
            "inferences": len(inferences), "inference_methods": dict(inference_methods),
            "historical_witness_origins": dict(witness_origins),
            "unknown_channel_actual_cover_radii_m": coverage,
            "termination_proof": "sixteen_distinct_detected_and_all_cleared" if len(known) == 16
                                 else "actual_no_signal_cover_for_each_unknown_and_all_known_cleared",
            "virtual_time_s": charged_us / 1_000_000}


def audit_directory(directory, allow_incomplete=False):
    directory = directory.resolve()
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    ledger_bytes = (directory / "runs.jsonl").read_bytes()
    rows = [json.loads(line) for line in ledger_bytes.splitlines() if line.strip()]
    expected = {(seed, method) for seed in range(manifest["seed_start"], manifest["seed_stop_exclusive"])
                for method in manifest["specs"]}
    findings, records, seen, case_hashes = [], [], set(), {}
    for relative, expected_hash in manifest["source_sha256"].items():
        for root in (ROOT, directory / "source_snapshot"):
            path = root / relative
            if not path.is_file() or digest_bytes(path.read_bytes()) != expected_hash:
                findings.append(f"Frozen source hash mismatch: {path}")
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    complete = bool(summary and summary["complete"] and summary["source_unchanged"]
                    and not summary["harness_errors"] and len(rows) == len(expected))
    if not allow_incomplete and not complete:
        findings.append("Evaluation is not a complete, source-verified snapshot")
    for row in rows:
        relative = row["trace"]
        item = {"trace": relative, "seed": row.get("seed"), "method": row.get("method")}
        try:
            key = (row["seed"], row["method"])
            check(key in expected and key not in seen, "Duplicate or unexpected seed/method record")
            seen.add(key)
            check(Path(relative).parts[0] == "traces" and ".." not in Path(relative).parts
                  and not Path(relative).is_absolute(), "Trace path escapes its trace directory")
            path = directory / relative
            payload = path.read_bytes()
            item["trace_sha256"] = digest_bytes(payload)
            trace = json.loads(gzip.decompress(payload))
            check(trace["row"] == row, "Trace row differs from the durable run ledger")
            check(relative == f"traces/{row['case_id']}--{row['method']}.json.gz",
                  "Trace filename differs from case/method identity")
            item.update(audit_trace(trace, manifest))
            prior = case_hashes.setdefault(row["seed"], row["case_sha256"])
            check(prior == row["case_sha256"], "Paired methods do not share the same immutable case")
        except Exception as error:
            item.update(audit_passed=False, finding=f"{type(error).__name__}: {error}")
        records.append(item)
    if not records:
        findings.append("No durable, completed trace records were found")
    if complete and seen != expected:
        findings.append("Missing seed/method combinations in complete evaluation")
    if complete:
        recorded_paths = {row["trace"] for row in rows}
        actual_paths = {path.relative_to(directory).as_posix()
                        for path in (directory / "traces").glob("*.json.gz")}
        if actual_paths != recorded_paths:
            findings.append("Completed trace files and durable ledger are not one-to-one")
    if (directory / "manifest.json").read_bytes() != manifest_bytes:
        findings.append("Manifest changed during audit")
    if complete and (directory / "runs.jsonl").read_bytes() != ledger_bytes:
        findings.append("Complete run ledger changed during audit")
    passed = [record for record in records if record["audit_passed"]]
    methods = {}
    for method in manifest["specs"]:
        subset = [record for record in passed if record["method"] == method]
        inference_methods, origins = Counter(), Counter()
        for record in subset:
            inference_methods.update(record["inference_methods"])
            origins.update(record["historical_witness_origins"])
        methods[method] = {"traces_passed": len(subset),
                           "certified_clears": sum(r["certified_clears"] for r in subset),
                           "near_clears": sum(r["near_clears"] for r in subset),
                           "max_clearance_bound_m": max((r["max_clearance_bound_m"] for r in subset), default=None),
                           "inference_methods": dict(inference_methods),
                           "historical_witness_origins": dict(origins)}
    result = {"audit_kind": "independent_clear_region_prefix_geometry_physics_ledger_and_discovery",
              "audited_utc": datetime.now(timezone.utc).isoformat(),
              "audit_passed": bool(records) and len(passed) == len(records) and not findings,
              "complete_experiment_snapshot": complete,
              "traces_audited": len(records), "traces_passed": len(passed),
              "traces_failed": len(records)-len(passed), "expected_traces": len(expected),
              "manifest_sha256": digest_bytes(manifest_bytes),
              "run_ledger_sha256": digest_bytes(ledger_bytes),
              "audit_script_sha256": digest_bytes(Path(__file__).read_bytes()),
              "geometry_tolerance_m": GEOMETRY_TOLERANCE_M,
              "clearance_bound_m": 19.9, "physical_clearance_radius_m": 20.0,
              "certificate_scope": "Actual response prefix plus individually reproved nonphysical deductions; no clear_region_log, final source estimate or truth used to authorize a clear or discovery completion",
              "arithmetic_scope": "Exact rational historical squared-distance inequalities; existing binary64 outer-region and disk-cover geometry with stated numerical tolerance",
              "scope_note": "Read-only replay of completed synthetic traces; no simulator/policy execution and no official evidence or unseen-case performance claim",
              "findings": findings, "methods": methods, "records": records}
    destination = directory / "audit.json"
    temporary = directory / "audit.json.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Audit only durable completed rows and explicitly report a partial snapshot")
    args = parser.parse_args()
    try:
        result = audit_directory(args.directory, args.allow_incomplete)
    except Exception as error:
        print(f"Audit could not start: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: value for key, value in result.items() if key != "records"},
                     ensure_ascii=False, indent=2))
    failures = [row for row in result["records"] if not row["audit_passed"]]
    if failures:
        print(json.dumps(failures, ensure_ascii=False, indent=2))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
