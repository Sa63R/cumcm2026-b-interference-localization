"""Read a completed Q4 stage; compare costs and trajectories without running a case.

Example: python experiments/analyze_q4_state_results.py
    --input results/q4_state_search/pilot-v1 --output research/q4-pilot-analysis.json

This tool uses only JSON/gzip/ZIP and standard-library arithmetic. It never
imports a policy, simulator, geometry implementation, or scene generator.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import zipfile


FEES = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
EPS = 1e-6


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(a, b, label):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > EPS:
        raise ValueError(f"{label}: {a!r} != {b!r}")


def percentile(values, fraction):
    values = sorted(values)
    index = fraction * (len(values)-1)
    left = math.floor(index)
    return values[left]*(1-(index-left)) + values[math.ceil(index)]*(index-left)


def stats(values):
    values = list(values)
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "p95": percentile(values, .95) if values else None,
            "max": max(values) if values else None}


def bootstrap(values, settings):
    rng = random.Random(settings["seed"])
    count = settings["samples"]
    estimates = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(count))
    return [estimates[int(.025*count)], estimates[int(.975*count)]]


def point(value):
    return (value["x"], value["y"]) if isinstance(value, dict) else tuple(value)


def silence_reason(position, source):
    # Explicitly post-termination explanatory labels; never proposed actions.
    dx, dy = position[0]-source["x"], position[1]-source["y"]
    distance = math.hypot(dx, dy)
    outside = distance > source["reception_radius_m"]
    backward = False
    if source["orientation_deg"] is not None:
        angle = math.radians(source["orientation_deg"])
        backward = math.cos(angle)*dx + math.sin(angle)*dy < -1e-12*max(1., distance)
    return ("backside_and_out_of_radius" if backward and outside else
            "directional_backside_only" if backward else "out_of_radius_only" if outside else "unexplained")


def log_stats(parameters, actions):
    routes = parameters.get("chain_route_log", [])
    route_results = [item["result"] for item in routes]
    result = {"route_calls": len(routes),
              "route_selected_kind": dict(Counter(item["selected_kind"] for item in routes)),
              "route_source_calls_with_covers": sum(item["selected_kind"] == "source" and
                  bool(item["remaining_covers"]) for item in routes),
              "route_count_cap_calls": sum(bool(item["discovery_count_cap"]) for item in routes),
              "route_closed_calls": sum(bool(item["exact"]) for item in route_results),
              "route_runtime_s": sum(item["runtime_s"] for item in routes),
              "route_proxy_relative_gaps": stats((item["cost_s"]-item["lower_bound_s"])/max(1., item["cost_s"])
                                                  for item in route_results)}
    for name in ("expanded", "generated", "dominance_pruned", "bound_pruned"):
        result["route_"+name] = sum(item[name] for item in route_results)
    for name, key in (("hull", "positive_hull_log"), ("pair", "directional_pair_log")):
        entries = parameters.get(key, [])
        outcome, roles = Counter(), Counter()
        selected = 0
        for entry in entries:
            roles[entry.get("role", "probe")] += 1
            chosen = (entry.get("selected") is not None if name == "hull" else
                      entry.get("role") == "second" or entry.get("pair") is not None)
            if not chosen:
                continue
            selected += 1
            index = entry["after_actual_action_count"]
            if index < len(actions):
                action = actions[index]
                if action["action"] == "measure" and action["channel"] == entry["channel"]:
                    outcome[action["result"]] += 1
                else:
                    outcome["no_matching_following_measurement"] += 1
            else:
                outcome["no_following_accepted_action"] += 1
        result[name] = {"calls": len(entries), "selected": selected,
                        "roles": dict(roles), "following_measurement_outcomes": dict(outcome)}
    return result


def analyze_record(record, path, *, failure_penalty_s=360000.):
    row, evaluation, summary = record["row"], record["evaluation"], record["summary"]
    if record.get("evaluation_phase") != "after_policy_termination":
        raise ValueError(f"Not a post-termination record: {path}")
    if digest(evaluation["ground_truth"]) != row["case_sha256"]:
        raise ValueError(f"Case hash mismatch: {path}")
    if evaluation["case_id"] != row["case_id"]:
        raise ValueError(f"Case ID mismatch: {path}")
    truth = {source["channel"]: source for source in evaluation["ground_truth"]["sources"]}
    physical = [item for item in record["history"] if item["action"] in ("/measure", "/clear")]
    measurement_count = sum(item["action"] == "/measure" for item in physical)
    if (row["problem"] != 4 or row["measurement_count"] != measurement_count
            or evaluation["measurement_count"] != measurement_count
            or row["action_count"] != len(record["history"])
            or evaluation["action_count"] != len(record["history"])):
        raise ValueError(f"Physical action count mismatch: {path}")
    successful = bool(row["completion_certified"] and row["accepted_exit"] and evaluation["all_cleared"] and not row["errors"])
    if bool(row["successful"]) != successful:
        raise ValueError(f"Inconsistent recorded success: {path}")
    close(row["penalized_time_s"], row["virtual_time_s"] if successful else failure_penalty_s, "failure penalty")
    reported = summary["action_history"] if summary else []
    if summary and len(reported) != len(physical):
        raise ValueError(f"Strategy/physical action length mismatch: {path}")
    covers = [point(p) for p in summary["coverage_points"]] if summary else []
    skip_logs = summary.get("strategy_parameters", {}).get("skipped_certified_scans", []) if summary else []
    skip_by_index = defaultdict(list)
    for skipped in skip_logs:
        skip_by_index[skipped["after_actual_action_count"]].append(skipped)
    used_skips = 0
    station = None
    completed = 0
    station_rows, clear_events = [], []
    cleared, near_channels, first_seen, positive_counts = set(), set(), {}, Counter()
    phases, source_costs, source_failures = defaultdict(Counter), defaultdict(Counter), Counter()
    totals, reasons = Counter(), Counter()
    previous, previous_channel, previous_time = (0., 0.), 1, 0.
    prefix_error = 0.
    actions = []
    for index, item in enumerate(physical):
        response = item["response"]
        action, position, channel = item["action"][1:], point(item["position"]), item["channel"]
        result = response["measure_result" if action == "measure" else "clear_result"]
        phase = reported[index]["phase"] if summary else "unreported_after_policy_exception"
        virtual_time = response["virtual_time_s"]
        if summary:
            claimed = reported[index]
            if ((claimed["action"], point(claimed["position"]), claimed["channel"], claimed["result"])
                    != (action, position, channel, result)):
                raise ValueError(f"Strategy/physical action mismatch: {path}:{index}")
            close(claimed["virtual_time_s"], virtual_time, "reported action time")
        if phase == "coverage":
            if station is None:
                if completed >= len(covers) or position != covers[completed]:
                    raise ValueError(f"Coverage chain order mismatch: {path}:{index}")
                skipped_here = skip_by_index.get(index, [])
                skipped_channels = set()
                for skipped in skipped_here:
                    c = skipped["channel"]
                    if (point(skipped["position"]) != position or c in skipped_channels or c in cleared
                            or c not in first_seen):
                        raise ValueError(f"Invalid skipped-scan identity/history: {path}:{index}")
                    if skipped["reason"] == "near":
                        if c not in near_channels:
                            raise ValueError(f"Skipped near without actual prior near: {path}:{index}")
                    elif skipped["reason"] == "enclosing_disk":
                        radius = skipped["radius_m"]
                        if not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 <= radius <= 19.9:
                            raise ValueError(f"Invalid reported skip radius: {path}:{index}")
                    else:
                        raise ValueError(f"Unknown skip reason: {path}:{index}")
                    skipped_channels.add(c)
                used_skips += len(skipped_here)
                station = {"index": completed, "position": list(position),
                           "start_action_index": index, "start_time_s": previous_time,
                           "cleared_before": len(cleared), "claimed_certified_skips": sorted(skipped_channels),
                           "expected": set(range(1, 21))-cleared-skipped_channels,
                           "measured": set()}
            if action != "measure" or position != point(station["position"]):
                raise ValueError(f"Interrupted or non-measure coverage transaction: {path}:{index}")
            if channel not in station["expected"] or channel in station["measured"]:
                raise ValueError(f"Repeated/cleared coverage channel: {path}:{index}")
            station["measured"].add(channel)
        elif station is not None:
            raise ValueError(f"Incomplete coverage station followed by other action: {path}:{index}")
        fees = {name: 0. for name in FEES}
        fees["movement_s"] = round(math.dist(previous, position)/5*1_000_000)/1_000_000
        if action == "measure":
            fees["switching_s"] = float(previous_channel != channel)
            fees["detection_s"] = 5.
            previous_channel = channel
            if result in {"direction", "near"}:
                first_seen.setdefault(channel, {"action_index": index, "time_s": virtual_time,
                                               "completed_cover_prefix": completed, "phase": phase})
                positive_counts[channel] += 1
                if result == "near":
                    near_channels.add(channel)
            if phase == "active_localization" and result == "no_signal":
                reasons[silence_reason(position, truth[channel])] += 1
        else:
            fees["optical_s"] = 3.
            fees["removal_s"] = 2. if result == "success" else 0.
            if result == "success":
                if channel in cleared:
                    raise ValueError(f"Repeated successful clear: {path}:{index}")
                cleared.add(channel)
                clear_events.append({"channel": channel, "action_index": index, "time_s": virtual_time,
                    "completed_cover_prefix": completed, "phase": phase, "positive_count": positive_counts[channel],
                    "first_seen": first_seen.get(channel),
                    "source_type_posthoc": "omni" if truth[channel]["orientation_deg"] is None else "directional"})
            else:
                source_failures[channel] += 1
        error = abs(virtual_time-previous_time-sum(fees.values()))
        prefix_error = max(prefix_error, error)
        if error > EPS:
            raise ValueError(f"Physical fee prefix mismatch: {path}:{index}: {error}")
        phases[phase].update(fees)
        phases[phase]["actions"] += 1
        phases[phase][action+"_"+result] += 1
        totals.update(fees)
        if phase != "coverage":
            source_costs[channel].update(fees)
        actions.append({"action": action, "position": list(position), "channel": channel, "result": result,
                        "phase": phase, "time_s": virtual_time, "completed_cover_prefix_before": completed,
                        "fees_s": fees})
        if station is not None and station["measured"] == station["expected"]:
            station_rows.append({key: value for key, value in station.items() if key not in {"expected", "measured"}}
                                | {"measurement_count": len(station["measured"]), "end_action_index": index,
                                   "end_time_s": virtual_time})
            completed += 1
            station = None
        previous, previous_time = position, virtual_time
    for name in FEES:
        close(totals[name], evaluation["time_breakdown_s"][name], name)
        close(totals[name], row[name], "row "+name)
    close(sum(totals.values()), evaluation["virtual_time_s"], "total time")
    close(evaluation["virtual_time_s"], row["virtual_time_s"], "row total time")
    if (len(cleared) != evaluation["cleared_total"] or len(cleared) != row["cleared_total"]
            or cleared != set(evaluation["cleared_channels"])
            or sum(source_failures.values()) != row["failed_clear_count"]
            or sum(source_failures.values()) != evaluation["failed_clear_count"]):
        raise ValueError(f"Clear ledger mismatch: {path}")
    if summary and completed != summary["coverage_points_visited"]:
        raise ValueError(f"Coverage prefix/report count mismatch: {path}")
    if station is not None and row["successful"]:
        raise ValueError(f"Successful record has partial coverage scan: {path}")
    if used_skips != len(skip_logs) and row["successful"]:
        raise ValueError(f"Successful record has unmatched skipped-scan logs: {path}")
    sources = [{"channel": c, "noncoverage_fees_s": dict(source_costs[c]),
                "failed_optical_attempts": source_failures[c], "first_seen": first_seen.get(c),
                "clear": next((event for event in clear_events if event["channel"] == c), None)}
               for c in sorted(set(source_costs) | set(first_seen))]
    return {"row": row, "record_path": str(path.resolve()), "record_sha256": file_hash(path),
            "physical_prefix_error_s": prefix_error, "phases": {p: dict(fees) for p, fees in phases.items()},
            "clear_events": clear_events, "first_clear": clear_events[0] if clear_events else None,
            "last_clear": clear_events[-1] if clear_events else None,
            "tail_after_last_successful_clear_s": row["virtual_time_s"]-clear_events[-1]["time_s"] if clear_events else None,
            "coverage_prefix_completed": completed, "coverage_points_total": len(covers),
            "coverage_stations": station_rows, "partial_coverage_channels": sorted(station["measured"]) if station else [],
            "cleared_channel_measurement_slots_avoided_at_visited_stations": sum(s["cleared_before"] for s in station_rows),
            "claimed_certified_measurement_slots_avoided_at_visited_stations": sum(len(s["claimed_certified_skips"]) for s in station_rows),
            "unmatched_skipped_scan_logs": len(skip_logs)-used_skips,
            "active_silence_reasons_posthoc": dict(reasons), "sources": sources,
            "planning": log_stats(summary.get("strategy_parameters", {}), reported) if summary else None,
            "actions": actions}


def aggregate(cases):
    phase_names = sorted({phase for case in cases for phase in case["phases"]})
    first = [case["first_clear"] for case in cases if case["first_clear"]]
    return {"runs": len(cases), "successful": sum(c["row"]["successful"] for c in cases),
            "penalized_time_s": stats(c["row"]["penalized_time_s"] for c in cases),
            "actual_time_s": stats(c["row"]["virtual_time_s"] for c in cases),
            "first_clear_time_s": stats(c["time_s"] for c in first),
            "first_clear_cover_prefix": stats(c["completed_cover_prefix"] for c in first),
            "completed_cover_prefix": stats(c["coverage_prefix_completed"] for c in cases),
            "clear_at_cover_prefix_counts": dict(Counter(str(e["completed_cover_prefix"]) for c in cases for e in c["clear_events"])),
            "mean_clears_before_last_executed_cover": statistics.mean(sum(e["completed_cover_prefix"] < c["coverage_prefix_completed"]
                                                                           for e in c["clear_events"]) for c in cases),
            "mean_avoided_cleared_channel_slots_at_visited_stations": statistics.mean(
                c["cleared_channel_measurement_slots_avoided_at_visited_stations"] for c in cases),
            "mean_avoided_claimed_certified_slots_at_visited_stations": statistics.mean(
                c["claimed_certified_measurement_slots_avoided_at_visited_stations"] for c in cases),
            "mean_phases": {phase: {key: statistics.mean(c["phases"].get(phase, {}).get(key, 0.) for c in cases)
                                     for key in FEES+("actions", "measure_no_signal", "clear_no_target_in_range")}
                            for phase in phase_names},
            "mean_components_s": {name: statistics.mean(c["row"][name] for c in cases) for name in FEES},
            "mean_measurements": statistics.mean(c["row"]["measurement_count"] for c in cases),
            "mean_failed_optical_attempts": statistics.mean(c["row"]["failed_clear_count"] for c in cases),
            "active_silence_reasons_posthoc": dict(sum((Counter(c["active_silence_reasons_posthoc"]) for c in cases), Counter())),
            "failures": [{"case_id": c["row"]["case_id"], "errors": c["row"]["errors"], "record": c["record_path"]}
                         for c in cases if not c["row"]["successful"]]}


def pair_detail(reference, candidate):
    a, b = reference["row"], candidate["row"]
    left, right = reference["actions"], candidate["actions"]
    index = 0
    signature = lambda action: (action["action"], action["position"], action["channel"])
    while index < min(len(left), len(right)) and signature(left[index]) == signature(right[index]):
        index += 1
    changed = index < max(len(left), len(right))
    phase_names = sorted(set(reference["phases"]) | set(candidate["phases"]))
    return {"case_id": a["case_id"], "seed": a["seed"], "case_sha256": a["case_sha256"],
            "reference_successful": a["successful"], "candidate_successful": b["successful"],
            "penalized_saved_s": a["penalized_time_s"]-b["penalized_time_s"],
            "actual_saved_s": a["virtual_time_s"]-b["virtual_time_s"],
            "saved_components_s": {name: a[name]-b[name] for name in FEES},
            "saved_phases_s": {phase: {name: reference["phases"].get(phase, {}).get(name, 0.) -
                candidate["phases"].get(phase, {}).get(name, 0.) for name in FEES} for phase in phase_names},
            "reference_first_clear": reference["first_clear"], "candidate_first_clear": candidate["first_clear"],
            "reference_cover_prefix": reference["coverage_prefix_completed"], "candidate_cover_prefix": candidate["coverage_prefix_completed"],
            "reference_tail_after_last_clear_s": reference["tail_after_last_successful_clear_s"],
            "candidate_tail_after_last_clear_s": candidate["tail_after_last_successful_clear_s"],
            "first_different_action_index": index if changed else None,
            "first_different_actions": {"reference": left[index] if index < len(left) else None,
                                        "candidate": right[index] if index < len(right) else None} if changed else None,
            "reference_clear_order": [e["channel"] for e in reference["clear_events"]],
            "candidate_clear_order": [e["channel"] for e in candidate["clear_events"]]}


def compare(reference, candidate, bootstrap_settings):
    by_seed = {c["row"]["seed"]: c for c in reference}
    pairs = [pair_detail(by_seed[c["row"]["seed"]], c) for c in candidate]
    savings = [pair["penalized_saved_s"] for pair in pairs]
    mean_ref = statistics.mean(c["row"]["penalized_time_s"] for c in reference)
    mean_can = statistics.mean(c["row"]["penalized_time_s"] for c in candidate)
    phase_names = sorted({phase for pair in pairs for phase in pair["saved_phases_s"]})
    return {"pairs": len(pairs), "all_complete": all(p["reference_successful"] and p["candidate_successful"] for p in pairs),
            "mean_saved_s": statistics.mean(savings), "saving_ci95_s": bootstrap(savings, bootstrap_settings),
            "mean_reduction_fraction": 1-mean_can/mean_ref if mean_ref else None,
            "p95_ratio": percentile([c["row"]["penalized_time_s"] for c in candidate], .95)/
                         percentile([c["row"]["penalized_time_s"] for c in reference], .95) if mean_ref else None,
            "wins": sum(v > EPS for v in savings), "losses": sum(v < -EPS for v in savings),
            "ties": sum(abs(v) <= EPS for v in savings), "worst_regression_s": max(0., -min(savings)),
            "mean_saved_components_s": {name: statistics.mean(p["saved_components_s"][name] for p in pairs) for name in FEES},
            "mean_saved_phases_s": {phase: {name: statistics.mean(p["saved_phases_s"].get(phase, {}).get(name, 0.)
                                                for p in pairs) for name in FEES} for phase in phase_names},
            "largest_regressions_or_smallest_gains": sorted(pairs, key=lambda p: (p["penalized_saved_s"], p["seed"]))[:3],
            "largest_gains": sorted(pairs, key=lambda p: (-p["penalized_saved_s"], p["seed"]))[:3],
            "paired_rows": [{k: v for k, v in p.items() if k not in {"first_different_actions", "saved_phases_s"}} for p in pairs]}


def analyze(stage):
    stage = Path(stage).resolve()
    manifest, freeze, summary = (read_json(stage/name) for name in ("manifest.json", "freeze.json", "summary.json"))
    if freeze["manifest_sha256"] != digest(manifest):
        raise ValueError("Manifest does not match freeze")
    with zipfile.ZipFile(stage/"source.zip") as source:
        if len(source.namelist()) != len(set(source.namelist())) or set(source.namelist()) != set(manifest["source_sha256"]):
            raise ValueError("Frozen source ZIP inventory mismatch")
        for name, expected in manifest["source_sha256"].items():
            if hashlib.sha256(source.read(name)).hexdigest() != expected:
                raise ValueError(f"Frozen source byte mismatch: {name}")
    labels, seeds = sorted(manifest["specs"]), manifest["seeds"]
    if len(seeds) != len(set(seeds)) or not seeds or "triangular" not in labels:
        raise ValueError("Invalid seed inventory or missing baseline")
    if seeds != list(range(manifest["protocol"][manifest["stage"]][0], manifest["protocol"][manifest["stage"]][1]+1)):
        raise ValueError("Incomplete predefined stage seeds")
    expected = {(label, seed) for label in labels for seed in seeds}
    expected_names = {f"{label}-{seed}.json.gz" for label, seed in expected}
    paths = sorted((stage/"records").glob("*.json.gz"))
    if {p.name for p in paths} != expected_names:
        raise ValueError("Stage incomplete or unexpected record files")
    summary_rows = {(row["strategy"], row["seed"]): row for row in summary["rows"]}
    if len(summary_rows) != len(summary["rows"]) or set(summary_rows) != expected:
        raise ValueError("Summary duplicate/missing/extra pairing")
    cases, seen = [], set()
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        row = record["row"]
        key = row["strategy"], row["seed"]
        if key in seen or key not in expected or path.name != f"{key[0]}-{key[1]}.json.gz":
            raise ValueError("Record duplicate/identity mismatch")
        seen.add(key)
        if row != summary_rows[key] or record["spec"] != manifest["specs"][key[0]] or row["stage"] != manifest["stage"]:
            raise ValueError("Record/summary/spec/stage mismatch")
        cases.append(analyze_record(record, path, failure_penalty_s=manifest["protocol"]["limits"]["virtual_seconds"]))
    grouped = {label: sorted((c for c in cases if c["row"]["strategy"] == label), key=lambda c:c["row"]["seed"]) for label in labels}
    for seed in seeds:
        identity = {(c["row"]["case_id"], c["row"]["case_sha256"]) for c in cases if c["row"]["seed"] == seed}
        if len(identity) != 1:
            raise ValueError(f"Unpaired case identities at seed {seed}")
    comparisons = {label+"_vs_triangular": compare(grouped["triangular"], grouped[label], manifest["protocol"]["bootstrap"])
                   for label in labels if label != "triangular"}
    if "state" in grouped:
        for label in labels:
            if label not in {"triangular", "state"}:
                comparisons[label+"_vs_state"] = compare(grouped["state"], grouped[label], manifest["protocol"]["bootstrap"])
    families = {}
    if manifest["stage"] == "stress":
        names, first = manifest["protocol"]["stress_families"], manifest["protocol"]["stress"][0]
        families = {name: {label: aggregate([c for c in group if names[(c["row"]["seed"]-first) % len(names)] == name])
                           for label, group in grouped.items()} for name in names}
    compact_cases = [{key: value for key, value in case.items() if key != "actions"} for case in cases]
    return {"scope": "read_only_completed_local_Q4_stage_post_termination_analysis", "new_cases_run": 0,
            "stage": manifest["stage"], "source_git_commit_at_freeze": freeze["git_commit"],
            "source_identity_note": "Actual frozen byte identities are source_sha256; HEAD alone may precede uncommitted prototype files.",
            "source_sha256": manifest["source_sha256"], "script_sha256": file_hash(__file__),
            "inputs_sha256": {name: file_hash(stage/name) for name in ("manifest.json", "freeze.json", "source.zip", "summary.json")},
            "python": manifest["python"], "platform": manifest["platform"],
            "notes": ["Positive savings are reference minus candidate; component/phase savings use actual fees even for penalized failures.",
                      "P95 ratio is the ratio of marginal P95 penalized times, not the P95 of paired ratios.",
                      "Phase is matched to the physical accepted-action ledger; movement is charged to the destination action phase.",
                      "Earlier clears and avoided channel slots explain mechanisms but are not independently additive total-time guarantees.",
                      "Route closure/gaps concern the frozen point proxy, whose six-second scan fee omits entry-channel savings.",
                      "Posthoc radius/orientation labels never generate actions. This is fee/identity analysis, not an independent full geometric clear-certificate audit.",
                      "Skipped certified scans are separate declared ledger entries, not physical measurements. Prior positives/near and reported radius are checked; enclosing-radius claims still require the separate geometric audit.",
                      "Runtime under concurrent workers is diagnostic, not a controlled cross-algorithm timing result."],
            "summaries": {label: aggregate(group) for label, group in grouped.items()},
            "comparisons": comparisons, "stress_families": families,
            "ledger_max_prefix_error_s": max(c["physical_prefix_error_s"] for c in cases),
            "cases": compact_cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed stage directory containing manifest/freeze/summary/records")
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; existing evidence is not overwritten")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite existing analysis")
    result = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "runs": len(result["cases"]),
                      "ledger_max_prefix_error_s": result["ledger_max_prefix_error_s"],
                      "comparisons": {name: {key: value for key, value in comparison.items() if key not in
                          {"paired_rows", "largest_regressions_or_smallest_gains", "largest_gains", "mean_saved_phases_s"}}
                          for name, comparison in result["comparisons"].items()}}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
