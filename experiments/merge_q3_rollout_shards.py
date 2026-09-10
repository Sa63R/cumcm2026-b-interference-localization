"""Validate and merge complete contiguous rollout holdout shards without filtering."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

from experiments.run_q3_rollout_comparison import (
    BASELINE, canonical, summaries, trace_path, value_hash, write_json, write_tables,
)


SAME_SETTINGS = ("data_origin", "problem", "stage", "official_practice", "official_formal",
                 "configs", "config_content_sha256", "baseline", "max_actions", "include_hard",
                 "source_sha256_start", "source_sha256_end", "git_revision", "python", "platform",
                 "belief_prior", "algorithm_note", "runtime_scope", "comparison", "bootstrap")
CONTROL_FILES = ("manifest.json", "configs.json", "cases.json", "runs.csv", "summary.json")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decoded(raw):
    return json.loads(raw.decode("utf-8-sig"))


def csv_value(key, value):
    if key == "planning_fallback_counts":
        return canonical(value)
    return "" if value is None else str(value)


def read_shard(path):
    for name in CONTROL_FILES:
        check((path / name).is_file(), f"Missing shard control file: {name}")
    controls = {name: (path / name).read_bytes() for name in CONTROL_FILES}
    manifest = decoded(controls["manifest.json"])
    check(manifest.get("data_origin") == "synthetic_research" and manifest.get("problem") == 3,
          "Expected a synthetic Q3 shard")
    check(manifest.get("official_practice") is False and manifest.get("official_formal") is False,
          "Shard must explicitly exclude official practice and formal origins")
    check(manifest.get("stage") == "holdout", "Only stage=holdout shards can be merged")
    check(manifest.get("run_status") == "completed", "Shard is incomplete or interrupted")
    check(manifest.get("source_changed_during_run") is False and bool(manifest.get("source_sha256_start"))
          and manifest.get("source_sha256_start") == manifest.get("source_sha256_end"),
          "Shard source hashes changed or are missing")
    check(manifest.get("baseline") == {"name": BASELINE, "variant": "efficient", "efficient_config_override": None},
          "Baseline must be the default efficient policy")
    interval = manifest.get("random_seed_interval_inclusive")
    check(isinstance(interval, list) and len(interval) == 2
          and all(type(seed) is int and seed >= 0 for seed in interval) and interval[0] <= interval[1],
          "Invalid shard seed interval")
    check(type(manifest.get("random_count")) is int and manifest["random_count"] == interval[1] - interval[0] + 1,
          "Random count does not match seed interval")
    configs = decoded(controls["configs.json"])
    check(configs == manifest.get("configs") and value_hash(configs) == manifest.get("config_content_sha256"),
          "Configuration contents/hash differ from manifest")
    check(digest(controls["configs.json"]) == manifest.get("original_config_file_sha256"),
          "Original configuration file hash differs")
    check(isinstance(configs, list) and configs, "Missing rollout configurations")
    configurations = {BASELINE: None}
    for config in configs:
        check(isinstance(config, dict) and set(config) == {"name", "rollout_config"}
              and isinstance(config["name"], str) and config["name"] not in configurations
              and isinstance(config["rollout_config"], dict), "Invalid or duplicate rollout configuration")
        configurations[config["name"]] = config["rollout_config"]
    cases = decoded(controls["cases.json"])
    check(isinstance(cases, list) and cases and value_hash(cases) == manifest.get("cases_sha256"),
          "Fixed case configuration hash differs")
    case_map = {}
    random_seeds = set()
    for case in cases:
        check(isinstance(case, dict) and case.get("problem") == 3 and case.get("case_id") not in case_map,
              "Invalid or duplicate fixed case")
        case_id = case["case_id"]
        check(isinstance(case_id, str) and type(case.get("seed")) is int and isinstance(case.get("sources"), list),
              "Case identity/seed/source configuration is missing")
        if "-random-" in case_id:
            check(case_id == f"q3-random-{case['seed']:04d}", "Case ID disagrees with its random seed")
            random_seeds.add(case["seed"])
        else:
            check(manifest.get("include_hard") is True and case_id.startswith("q3-hard-"),
                  "Unexpected deterministic hard case")
        case_map[case_id] = case
    check(random_seeds == set(range(interval[0], interval[1] + 1)), "Fixed cases do not cover every shard seed")
    if manifest.get("include_hard"):
        check(len(cases) == manifest["random_count"] + 7, "Expected all seven hard cases")
    else:
        check(len(cases) == manifest["random_count"], "Case count differs from random count")
    csv_rows = list(csv.DictReader(io.StringIO(controls["runs.csv"].decode("utf-8-sig"), newline="")))
    expected_runs = len(cases) * len(configurations)
    check(len(csv_rows) == expected_runs == manifest.get("completed_runs") == manifest.get("expected_runs"),
          "Row count differs from complete case/configuration product")
    shard_summary = decoded(controls["summary.json"])
    check(shard_summary.get("data_origin") == "synthetic_research" and shard_summary.get("stage") == "holdout"
          and shard_summary.get("source_consistent") is True
          and shard_summary.get("completed_runs") == expected_runs == shard_summary.get("expected_runs"),
          "Shard summary is incomplete or inconsistent")
    rows, seen, traces, compute_fallbacks = [], set(), {}, []
    for csv_row in csv_rows:
        key = (csv_row.get("case_id"), csv_row.get("strategy"))
        check(key not in seen and key[0] in case_map and key[1] in configurations, "Duplicate/unknown case-strategy row")
        seen.add(key)
        relative = trace_path(*key)
        check(csv_row.get("trace") == relative, "Trace path does not match case/configuration identity")
        check((path / relative).is_file(), f"Missing expected trace: {relative}")
        raw = (path / relative).read_bytes()
        try:
            record = json.loads(gzip.decompress(raw).decode("utf-8"))
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"Trace cannot be decoded: {relative}") from exc
        check(record.get("data_origin") == "synthetic_research"
              and record.get("evaluation_phase") == "after_session_termination", "Wrong trace origin/evaluation phase")
        row = record["row"]
        canonical(row)  # Reject nonfinite record values before creating output.
        check((row.get("case_id"), row.get("strategy")) == key and row.get("trace") == relative,
              "Trace row identity differs from CSV")
        check(row.get("problem") == 3 and row.get("seed") == case_map[key[0]]["seed"], "Trace row problem/seed differs")
        check(row.get("case_kind") == ("random" if "-random-" in key[0] else "hard"), "Row case kind differs")
        check(record.get("config") == configurations[key[1]]
              and row.get("config_sha256") == value_hash(configurations[key[1]]), "Trace configuration/hash differs")
        check(row.get("case_sha256") == value_hash(case_map[key[0]])
              and record.get("evaluation", {}).get("ground_truth") == case_map[key[0]],
              "Trace truth or case hash differs from fixed case configuration")
        check(row.get("source_total") == len(case_map[key[0]]["sources"]), "Trace source total differs from fixed case")
        check(type(row.get("successful")) is bool and row.get("interrupted") is False,
              "Interrupted/invalid run record cannot enter a completed merge")
        check(isinstance(record.get("history"), list), "Missing complete action history")
        check(isinstance(record.get("summary"), dict) or row["successful"] is False,
              "Successful trace lacks its strategy summary")
        if row["successful"]:
            check(row.get("all_cleared") is True and row.get("completion_certified") is True
                  and row.get("accepted_exit") is True and not row.get("error")
                  and record["evaluation"].get("all_cleared") is True,
                  "Successful trace has inconsistent completion flags")
        for field, value in row.items():
            check(csv_row.get(field) == csv_value(field, value), f"CSV/trace row value differs: {field}")
        # A deadline reached inside _Continuation is counted under
        # incomplete_continuation by the policy. Preserve the original row,
        # but inspect its decision detail before claiming no budget fallback.
        planning = (record.get("summary") or {}).get("planning", {})
        decisions = planning.get("decisions", [])
        budget_decisions = [decision for decision in decisions
                            if decision.get("status") == "compute_budget"
                            or decision.get("detail") == "rollout_compute_budget"]
        reported_count = row["planning_fallback_counts"].get("compute_budget", 0)
        direct_count = sum(decision.get("status") == "compute_budget" for decision in budget_decisions)
        continuation_count = sum(decision.get("detail") == "rollout_compute_budget"
                                 and decision.get("status") != "compute_budget" for decision in budget_decisions)
        # Direct decision events overlap the policy's compute_budget counter;
        # continuation events occupy its separate incomplete_continuation counter.
        count = max(reported_count, direct_count) + continuation_count
        if count:
            compute_fallbacks.append({
                "case_id": row["case_id"], "strategy": row["strategy"], "count": count,
                "reported_compute_budget_count": reported_count,
                "continuation_compute_budget_count": continuation_count,
                "decision_evidence": [{key: decision.get(key) for key in ("search_index", "status", "detail")}
                                      for decision in budget_decisions],
            })
        rows.append(row)
        traces[relative] = raw
    check(seen == {(case_id, name) for case_id in case_map for name in configurations},
          "Case-strategy coverage is incomplete")
    present_traces = {file.relative_to(path).as_posix() for file in (path / "traces").rglob("*.json.gz")}
    check(present_traces == set(traces), "Extra or unlisted compressed traces would otherwise be omitted")
    baselines = {row["case_id"]: row for row in rows if row["strategy"] == BASELINE}
    for row, csv_row in zip(rows, csv_rows):
        base = baselines[row["case_id"]]
        saved = base["virtual_time_s"] - row["virtual_time_s"]
        valid = base["successful"] and row["successful"]
        expected = {"paired_baseline_time_s": base["virtual_time_s"], "paired_seconds_saved": saved,
                    "paired_win": saved > 1e-6 if valid else None, "pair_successful": valid}
        for field, value in expected.items():
            check(csv_row.get(field) == csv_value(field, value), f"Paired CSV value differs: {field}")
    return {"path": path, "manifest": manifest, "controls": controls, "cases": cases, "rows": rows,
            "traces": traces, "compute_budget_fallback_rows": compute_fallbacks}


def merge(output, shards):
    output = Path(output)
    check(not output.exists(), "Output already exists; merging never overwrites")
    check(bool(shards), "Provide at least one completed shard")
    paths = [Path(path).resolve() for path in shards]
    check(len(paths) == len(set(paths)), "Duplicate shard directory")
    inputs = [read_shard(path) for path in paths]
    inputs.sort(key=lambda item: item["manifest"]["random_seed_interval_inclusive"][0])
    reference = inputs[0]["manifest"]
    previous_end = None
    all_cases, rows, all_traces, seen_cases, provenance = [], [], {}, set(), []
    for index, item in enumerate(inputs):
        manifest = item["manifest"]
        for key in SAME_SETTINGS:
            check(manifest.get(key) == reference.get(key), f"Shard settings differ: {key}")
        start, end = manifest["random_seed_interval_inclusive"]
        if previous_end is not None:
            check(start == previous_end + 1, "Shard seed intervals overlap or contain a gap")
        previous_end = end
        for case in item["cases"]:
            check(case["case_id"] not in seen_cases,
                  "Repeated case across shards; shared hard cases must be evaluated separately")
            seen_cases.add(case["case_id"])
            all_cases.append(case)
        rows.extend(item["rows"])
        for relative, raw in item["traces"].items():
            check(relative not in all_traces, "Duplicate trace identity across shards")
            all_traces[relative] = raw
        provenance.append({
            "archive_directory": f"shards/p{index}", "original_directory": str(item["path"]),
            "random_seed_interval_inclusive": [start, end], "completed_runs": manifest["completed_runs"],
            "control_file_sha256": {name: digest(raw) for name, raw in item["controls"].items()},
            "trace_sha256": {name: digest(raw) for name, raw in item["traces"].items()},
            "source_sha256_start": manifest["source_sha256_start"],
            "source_sha256_end": manifest["source_sha256_end"],
            "compute_budget_fallback_rows": item["compute_budget_fallback_rows"],
            "compute_budget_event_count": sum(row["count"] for row in item["compute_budget_fallback_rows"]),
        })
    # Revalidate bytes before the first destination write, avoiding a mixed
    # snapshot if a purportedly completed source folder was still changing.
    for item in inputs:
        for relative, raw in {**item["controls"], **item["traces"]}.items():
            check((item["path"] / relative).read_bytes() == raw, "Shard input changed during merge validation")
    groups = summaries(rows)
    failed = sum(not row["successful"] for row in rows)
    compute_fallbacks = [row for item in inputs for row in item["compute_budget_fallback_rows"]]
    compute_event_count = sum(row["count"] for row in compute_fallbacks)
    source = dict(reference)
    source.update(
        created_utc=datetime.now(timezone.utc).isoformat(), finished_utc=datetime.now(timezone.utc).isoformat(),
        random_count=sum(item["manifest"]["random_count"] for item in inputs),
        random_seed_interval_inclusive=[inputs[0]["manifest"]["random_seed_interval_inclusive"][0], previous_end],
        cases_sha256=value_hash(all_cases), expected_runs=len(rows), completed_runs=len(rows), failed_runs=failed,
        source_changed_during_run=False, run_status="completed", merged_from_shards=True,
        shard_count=len(inputs), shard_provenance=provenance,
        merged_trace_sha256={name: digest(raw) for name, raw in all_traces.items()},
        merge_tool_source_sha256={name: digest((Path(__file__).resolve().parents[1] / name).read_bytes())
                                  for name in ("experiments/merge_q3_rollout_shards.py",
                                               "experiments/run_q3_rollout_comparison.py",
                                               "experiments/run_q3_comparison.py")},
        parallel_runtime_note="Shard wall runtimes can include concurrent CPU/memory load; use serial development results for single-process runtime reference. Virtual-time comparisons retain all rows. Inspect planning compute-budget fallbacks before assuming identical effective policy behavior across execution loads.",
        merge_selection_note="All cases and all strategies from every supplied complete contiguous shard are retained, including failures. No best seeds, successful-only filtering, or duplicate-case dropping occurred.",
        compute_budget_fallback_rows=compute_fallbacks,
        compute_budget_event_count=compute_event_count,
        no_compute_budget_fallbacks=not compute_fallbacks,
    )
    canonical(source)
    canonical(groups)
    # Everything above is read-only validation. Only now create the new archive.
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    for relative, raw in all_traces.items():
        (output / relative).write_bytes(raw)
    for item, evidence in zip(inputs, provenance):
        archive = output / evidence["archive_directory"]
        archive.mkdir(parents=True)
        for name, raw in item["controls"].items():
            (archive / name).write_bytes(raw)
    (output / "configs.json").write_bytes(inputs[0]["controls"]["configs.json"])
    source["original_config_file_sha256"] = digest(inputs[0]["controls"]["configs.json"])
    write_json(output / "cases.json", all_cases)
    write_tables(output, rows)
    write_json(output / "manifest.json", source)
    write_json(output / "summary.json", {"data_origin": "synthetic_research", "stage": "holdout",
               "expected_runs": len(rows), "completed_runs": len(rows), "failed_runs": failed,
               "source_consistent": True, "parallel_runtime_note": source["parallel_runtime_note"],
               "no_compute_budget_fallbacks": not compute_fallbacks,
               "compute_budget_event_count": compute_event_count,
               "compute_budget_fallback_rows": compute_fallbacks, "groups": groups})
    print(json.dumps({"output": str(output), "shards": len(inputs), "cases": len(all_cases),
                      "rows": len(rows), "failed_runs": failed}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        return merge(args.output, args.shards)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
