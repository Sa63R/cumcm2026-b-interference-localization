"""Read-only independent audit of a completed Q3 runtime-equivalence batch.

Recomputes raw-record hashes, checks every pairing and repetition, and checks
all physical-bound formulas plus independent DP samples. No policy is run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path


BASELINE_COMMIT = "760af8230a8e5bda6050663b19d871a9701c9829"
SUMMARY_RUNTIME = {"program_runtime_s", "runtime_s", "total_runtime_s"}
FIELDS = ("case_sha256", "action_history_sha256", "action_step_sha256",
          "summary_without_runtime_sha256", "evaluation_without_runtime_sha256",
          "observations_without_timestamp_sha256", "row_without_runtime_sha256",
          "successful", "all_cleared", "completion_certified", "accepted_exit",
          "source_total", "cleared_total", "measurement_count", "failed_clear_count",
          "action_count", "virtual_time_s")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def remove_times(value, ignored):
    if isinstance(value, dict):
        return {key: remove_times(child, ignored) for key, child in value.items() if key not in ignored}
    if isinstance(value, list):
        return [remove_times(child, ignored) for child in value]
    return value


def recompute_metrics(raw):
    summary = dict(raw["summary"])
    # SearchResult.source_estimates is keyed by integer channels in memory.
    # JSON converts object keys to strings, changing sort_keys order for 2/10.
    # Restore this one declared schema field; never coerce arbitrary string keys.
    estimates = summary["source_estimates"]
    assert all(str(int(key)) == key and 1 <= int(key) <= 20 for key in estimates)
    summary["source_estimates"] = {int(key): value for key, value in estimates.items()}
    actions = summary["action_history"]
    return {**raw["row"], "action_history_sha256": digest(actions),
            "action_step_sha256": [digest(action) for action in actions],
            "summary_without_runtime_sha256": digest(remove_times(summary, SUMMARY_RUNTIME)),
            "evaluation_without_runtime_sha256": digest(remove_times(raw["evaluation"], {"wall_time_s"})),
            "observations_without_timestamp_sha256": digest(remove_times(raw["history"], {"real_timestamp_ms"})),
            "row_without_runtime_sha256": digest(remove_times(raw["row"], {"program_runtime_s"}))}


def physical_ledger(raw):
    """Independent action ledger; successful /clear costs 5s, failed /clear 3s."""
    history, evaluation = raw["history"], raw["evaluation"]
    assert history[0]["action"] == "/enter" and history[-1]["action"] == "/exit"
    position, channel, microseconds = (0., 0.), 1, 0
    successful, failed, measured = 0, 0, 0
    components = Counter()
    for index, action in enumerate(history):
        assert action["index"] == index
        response, name = action["response"], action["action"]
        assert response["accepted"] is True
        point = (action["position"]["x"], action["position"]["y"])
        assert all(math.isfinite(value) for value in point)
        if name in ("/enter", "/exit"):
            assert (name == "/enter" and index == 0) or (name == "/exit" and index == len(history) - 1)
        else:
            assert name in ("/measure", "/clear") and 0 < index < len(history) - 1
            assert type(action["channel"]) is int and 1 <= action["channel"] <= 20
            costs = {"movement_s": round(math.dist(position, point) / 5 * 1_000_000)}
            position = point
            if name == "/measure":
                costs.update(detection_s=5_000_000, switching_s=int(channel != action["channel"]) * 1_000_000)
                channel = action["channel"]
                measured += 1
                assert response["measure_result"] in ("direction", "near", "no_signal")
            else:
                assert response["clear_result"] in ("success", "no_target_in_range")
                success = response["clear_result"] == "success"
                successful += success
                failed += not success
                costs.update(optical_s=3_000_000, removal_s=2_000_000 * success)
            components.update(costs)
            microseconds += sum(costs.values())
        assert math.isclose(response["virtual_time_s"], microseconds / 1_000_000, rel_tol=0, abs_tol=1e-6)
    assert evaluation["simulator_stop_reason"] == "exited"
    assert evaluation["all_cleared"] is True and not evaluation["remaining_channels"]
    assert successful == evaluation["source_total"] == evaluation["cleared_total"]
    assert failed == evaluation["failed_clear_count"] and measured == evaluation["measurement_count"]
    assert len(history) == evaluation["action_count"]
    assert math.isclose(evaluation["virtual_time_s"], microseconds / 1_000_000, rel_tol=0, abs_tol=1e-6)
    for name, value in evaluation["time_breakdown_s"].items():
        assert math.isclose(value, components[name] / 1_000_000, rel_tol=0, abs_tol=1e-6)
    assert raw["row"]["successful"] is True and raw["row"]["accepted_exit"] is True
    assert raw["row"]["completion_certified"] is True


def physical_graph(scenario):
    sources = sorted(scenario["sources"], key=lambda source: source["channel"])
    points = [(source["x"], source["y"]) for source in sources]
    first = [max(0., math.hypot(*point) - 20 - 1e-6) for point in points]
    edges = [[0. if i == j else max(0., math.dist(a, b) - 40 - 1e-6)
              for j, b in enumerate(points)] for i, a in enumerate(points)]
    return sources, first, edges


def independent_dp(first, edges):
    """Forward subset DP, independent from benchmark's predecessor-table DP."""
    n = len(first)
    costs = [[math.inf] * n for _ in range(1 << n)]
    for i, value in enumerate(first):
        costs[1 << i][i] = value
    complete = (1 << n) - 1
    for mask in range(1, complete):
        for last in range(n):
            current = costs[mask][last]
            if not math.isfinite(current):
                continue
            remaining = complete ^ mask
            while remaining:
                bit = remaining & -remaining
                target = bit.bit_length() - 1
                total = current + edges[last][target]
                if total < costs[mask | bit][target]:
                    costs[mask | bit][target] = total
                remaining ^= bit
    return min(costs[complete])


def audit(batch):
    errors, raw_hashes, comparisons = [], {}, {}

    def check(condition, message):
        if not condition:
            errors.append(message)

    manifest = read_json(batch / "manifest.json")
    cases = read_json(batch / "cases.json")
    rows = read_json(batch / "runs_with_bounds.json")
    initial_rows = [json.loads(line) for line in (batch / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    bounds, summary = read_json(batch / "lower_bounds.json"), read_json(batch / "summary.json")
    check(digest(cases) == manifest["cases_sha256"], "cases.json hash differs from frozen manifest")
    check(manifest["baseline"]["git_head"] == BASELINE_COMMIT, "Wrong frozen baseline commit")
    check(manifest["baseline"]["spec_sha256"] == manifest["candidate"]["spec_sha256"] == digest(manifest["spec"]), "Policy specs differ")
    expected_normalization = {"summary": sorted(SUMMARY_RUNTIME), "evaluation": ["wall_time_s"],
                              "observation_history": ["real_timestamp_ms"]}
    check(manifest["normalization"] == expected_normalization, "Unexpected hash normalization")
    check(summary["source_unchanged"] is True, "Harness reports source drift")
    count, repeats = manifest["case_count"], manifest["repeats"]
    check(count == len(cases), "Wrong scenario count")
    check(repeats >= 2 and repeats % 2 == 0, "Batch lacks balanced AB/BA repeats; not final timing evidence")
    case_map = {item["scenario"]["case_id"]: (index, item) for index, item in enumerate(cases)}
    check(len(case_map) == len(cases), "Duplicate scenario case IDs")
    expected = {(repeat, item["scenario"]["case_id"], side)
                for repeat in range(repeats) for item in cases for side in ("baseline", "candidate")}
    keys = [(row["repeat"], row["case_id"], row["side"]) for row in rows]
    check(len(keys) == len(expected) and set(keys) == expected and len(set(keys)) == len(keys), "Missing/duplicate/unexpected completed runs")
    check(len(initial_rows) == len(rows), "runs.jsonl and runs_with_bounds.json counts differ")
    for initial, final in zip(initial_rows, rows):
        check(initial == {key: value for key, value in final.items() if key not in {"physical_lower_bound_s", "time_over_physical_lower_bound"}},
              "Completed row differs from original append-only runs.jsonl")
    check(len({row["record_file"] for row in rows}) == len(rows), "A raw record is reused for multiple runs")
    check(set(bounds) == set(case_map), "Lower-bound case set differs")
    for row in rows:
        key = (row["repeat"], row["case_id"], row["side"])
        label = "/".join(map(str, key))
        try:
            index, item = case_map[row["case_id"]]
            expected_name = f"r{row['repeat']:02d}-c{index:03d}-{row['side']}.json.gz"
            check(row["record_file"] == expected_name, label + ": raw record name mismatch")
            path = (batch / row["record_file"]).resolve()
            if batch not in path.parents:
                raise ValueError("Raw record path leaves batch directory")
            raw_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                raw = json.load(stream)
            check(raw["evaluation_phase"] == "after_policy_termination", label + ": wrong truth-read phase")
            check(digest(raw["spec"]) == manifest["baseline"]["spec_sha256"], label + ": raw policy spec differs")
            scenario_hash = digest(item["scenario"])
            check(scenario_hash == row["case_sha256"] == digest(raw["evaluation"]["ground_truth"]), label + ": scenario truth differs")
            check(row["split"] == item["split"], label + ": partition label differs")
            actual = recompute_metrics(raw)
            for name in FIELDS:
                check(actual[name] == row[name] == raw["runtime_equivalence_metrics"][name], label + ": recomputed field differs: " + name)
            check(actual["program_runtime_s"] == row["program_runtime_s"], label + ": raw wall-clock metric differs")
            check(raw["runtime_equivalence_metrics"]["program_cpu_s"] == row["program_cpu_s"], label + ": raw CPU metric differs")
            check(row["program_runtime_s"] > 0 and row["program_cpu_s"] > 0, label + ": nonpositive runtime")
            physical_ledger(raw)
            comparisons[key] = {name: actual[name] for name in FIELDS}
            bound = bounds[row["case_id"]]["physical_lower_bound_s"]
            check(row["physical_lower_bound_s"] == bound, label + ": bound differs from lower_bounds.json")
            check(math.isclose(row["time_over_physical_lower_bound"], row["virtual_time_s"] / bound, rel_tol=0, abs_tol=1e-12), label + ": T/LB differs")
        except Exception as exc:
            errors.append(label + ": " + type(exc).__name__ + ": " + str(exc))
    for case_id in case_map:
        reference = comparisons.get((0, case_id, "baseline"))
        for repeat in range(repeats):
            for side in ("baseline", "candidate"):
                key = (repeat, case_id, side)
                check(reference is not None and comparisons.get(key) == reference,
                      "/".join(map(str, key)) + ": policy/repeat equivalence differs")
    sample_indices = sorted({0, len(cases) // 2, len(cases) - 1})
    dp_samples = []
    for index, item in enumerate(cases):
        case_id = item["scenario"]["case_id"]
        try:
            value = bounds[case_id]
            sources, first, edges = physical_graph(item["scenario"])
            channels = [source["channel"] for source in sources]
            order = value["lower_graph_order_channels"]
            assert sorted(order) == sorted(channels) and len(set(order)) == len(channels)
            indices = [channels.index(channel) for channel in order]
            route = first[indices[0]] + sum(edges[a][b] for a, b in zip(indices, indices[1:]))
            assert math.isclose(route, value["lower_route_length_m"], rel_tol=0, abs_tol=1e-8)
            expected_bound = value["lower_route_length_m"] / 5 + 5 * len(sources)
            assert value["source_total"] == len(sources)
            assert value["physical_lower_bound_s"] == expected_bound
            assert value["movement_lower_bound_s"] == value["lower_route_length_m"] / 5
            assert value["clear_action_lower_bound_s"] == 5 * len(sources)
            assert "excludes discovery" in value["bound_scope"]
            if index in sample_indices:
                optimum = independent_dp(first, edges)
                assert math.isclose(optimum, value["lower_route_length_m"], rel_tol=0, abs_tol=1e-8)
                dp_samples.append({"case_id": case_id, "source_total": len(sources), "independent_route_lower_m": optimum,
                                   "physical_lower_bound_s": optimum / 5 + 5 * len(sources)})
        except Exception as exc:
            errors.append(case_id + ": lower-bound audit: " + type(exc).__name__ + ": " + str(exc))
    expected_pairs = {(repeat, case_id) for repeat in range(repeats) for case_id in case_map}
    pair_keys = [(pair["repeat"], pair["case_id"]) for pair in summary["pairs"]]
    check(len(pair_keys) == len(expected_pairs) and set(pair_keys) == expected_pairs, "Missing/duplicate summary pairs")
    check(all(pair["identical"] is True and not pair["different_fields"] for pair in summary["pairs"]), "Summary reports nonidentical pairs")
    check(summary["all_pairs_exactly_identical"] and summary["all_runs_successful"] and summary["valid_runtime_only_result"], "Harness final acceptance flags failed")
    files = ["manifest.json", "cases.json", "runs.jsonl", "runs_with_bounds.json", "lower_bounds.json", "summary.json"]
    return {"audit_passed": not errors, "errors": errors, "case_count": len(cases), "repeats": repeats,
            "expected_runs": len(expected), "observed_runs": len(rows), "raw_records_checked": len(raw_hashes),
            "expected_pairs": len(expected_pairs), "observed_pairs": len(pair_keys),
            "split_case_counts": dict(Counter(item["split"] for item in cases)),
            "all_case_bound_formula_checked": True, "independent_dp_samples": dp_samples,
            "raw_record_sha256": raw_hashes,
            "batch_file_sha256": {name: hashlib.sha256((batch / name).read_bytes()).hexdigest() for name in files},
            "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_identity_scope": "Manifest source identities preserved; this result audit does not independently reload remote source snapshots. Harness verifies source hashes before and after the batch.",
            "runtime_scope": "Local policy plus in-memory client during original run_case timer; excludes startup, client construction, final serialization, actual HTTP/network latency. This audit validates records, not a statistical speedup claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON audit file; refuses overwrite")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse overwriting an existing audit")
    if __debug__ is False:
        parser.error("Do not disable assertions when auditing")
    result = audit(args.batch.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(canonical({key: value for key, value in result.items() if key not in {"raw_record_sha256", "batch_file_sha256"}}))
    return 0 if result["audit_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
