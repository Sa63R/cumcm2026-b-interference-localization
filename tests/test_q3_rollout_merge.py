"""Archive-only rollout merge checks; these fixtures never run a policy or engine."""

from copy import deepcopy
import csv
import gzip
import hashlib
import json

import pytest

from experiments import merge_q3_rollout_shards as merger
from experiments import run_q3_rollout_comparison as driver


CONFIGS = [{"name": "rollout_test", "rollout_config": {"particles": 2}}]
SOURCE_HASHES = {"src/fixture_only.py": "a" * 64}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_trace(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def read_rows(directory):
    with (directory / "runs.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def archive_snapshot(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def make_record(case, name, config, *, failed=False):
    """Complete archive schema, deliberately not a physics-audit fixture."""
    elapsed = 1.0 if failed else 120.0 if name == driver.BASELINE else 100.0
    cleared = 0 if failed else 10
    components = {key: elapsed if key == "movement_s" else 0.0 for key in driver.COMPONENTS}
    row = {
        "problem": 3, "case_id": case["case_id"], "case_kind": "random", "seed": case["seed"],
        "strategy": name, "source_total": 10, "cleared_total": cleared,
        "cleared_fraction": cleared / 10, "all_cleared": not failed,
        "completion_certified": not failed, "successful": not failed, "accepted_exit": True,
        "interrupted": False, "completion_reason": "fixture_failure" if failed else "fixture_completed",
        "error": "fixture policy failure" if failed else "", "virtual_time_s": elapsed,
        "time_per_source_s": elapsed / 10, "mean_time_per_cleared_s": elapsed / cleared if cleared else None,
        "program_runtime_s": .25, "measurement_count": 0, "failed_clear_count": 0,
        "action_count": 2, "movement_m": 5 * elapsed, **components,
        "case_sha256": driver.value_hash(case), "planning_searches": 0,
        "planning_candidate_evaluations": 0, "planning_changed_decisions": 0,
        "planning_wall_time_s": 0.0, "planning_fallback_counts": {},
        "config_sha256": driver.value_hash(config), "trace": driver.trace_path(case["case_id"], name),
    }
    evaluation = {
        "case_id": case["case_id"], "ground_truth": deepcopy(case), "source_total": 10,
        "cleared_total": cleared, "cleared_fraction": cleared / 10, "all_cleared": not failed,
        "virtual_time_s": elapsed, "mean_time_per_cleared_s": row["mean_time_per_cleared_s"],
        "measurement_count": 0, "failed_clear_count": 0, "action_count": 2,
        "time_breakdown_s": components,
    }
    return {
        "data_origin": "synthetic_research", "row": row,
        "summary": {"problem": 3, "fixture_only": True, "action_history": [],
                    "variant": "efficient" if name == driver.BASELINE else "rollout"},
        "evaluation": evaluation, "evaluation_phase": "after_session_termination",
        "history": [{"action": "/enter"}, {"action": "/exit"}],
        "final_client_state": {"session": "exited", "cleared_count": cleared, "virtual_time_s": elapsed},
        "config": deepcopy(config), "exception_traceback": "fixture failure" if failed else None,
    }


def make_shard(parent, name, seed, *, configs=None, source_hashes=None, failed=False):
    directory = parent / name
    (directory / "traces").mkdir(parents=True)
    configs = deepcopy(CONFIGS if configs is None else configs)
    hashes = deepcopy(SOURCE_HASHES if source_hashes is None else source_hashes)
    case = {
        "case_id": f"q3-random-{seed:04d}", "problem": 3, "seed": seed,
        "sources": [{"channel": channel, "x": channel * 10.0, "y": 0.0,
                     "orientation_deg": None, "reception_radius_m": 1250.0}
                    for channel in range(1, 11)],
    }
    driver.write_json(directory / "configs.json", configs)
    driver.write_json(directory / "cases.json", [case])
    manifest = {
        "data_origin": "synthetic_research", "official_practice": False, "official_formal": False,
        "problem": 3, "created_utc": "2026-09-10T00:00:00+00:00", "git_revision": "fixture-only",
        "python": "fixture-only", "platform": "fixture-only", "stage": "holdout", "random_count": 1,
        "random_seed_interval_inclusive": [seed, seed], "include_hard": False, "configs": configs,
        "config_content_sha256": driver.value_hash(configs),
        "baseline": {"name": driver.BASELINE, "variant": "efficient", "efficient_config_override": None},
        "max_actions": 20000, "cases_sha256": driver.value_hash([case]), "source_sha256_start": hashes,
        "original_config_file_sha256": hashlib.sha256((directory / "configs.json").read_bytes()).hexdigest(),
        "belief_prior": "fixture-only", "algorithm_note": "fixture-only", "runtime_scope": "fixture-only",
        "comparison": "fixture-only", "holdout_note": "fixture-only", "resume_note": "fixture-only",
        "bootstrap": {"samples": 2000, "seed": 20260910, "unit": "paired random case", "interval": "percentile 95%"},
        "run_status": "completed", "expected_runs": 2, "completed_runs": 2,
        "finished_utc": "2026-09-10T00:01:00+00:00", "source_sha256_end": hashes,
        "source_changed_during_run": False,
    }
    driver.write_json(directory / "manifest.json", manifest)
    records = [make_record(case, driver.BASELINE, None)] + [
        make_record(case, entry["name"], entry["rollout_config"], failed=failed) for entry in configs]
    for record in records:
        driver.save_trace(directory, record)
    rows = [record["row"] for record in records]
    driver.write_tables(directory, rows)
    driver.write_json(directory / "summary.json", {
        "data_origin": "synthetic_research", "stage": "holdout", "expected_runs": len(rows),
        "completed_runs": len(rows), "source_consistent": True, "groups": driver.summaries(rows),
    })
    return directory


def change_manifest(directory, **updates):
    manifest = read_json(directory / "manifest.json")
    manifest.update(updates)
    driver.write_json(directory / "manifest.json", manifest)


def change_trace(directory, mutate):
    """Keep CSV in agreement so identity checks, not stale CSV, detect corruption."""
    path = next(path for path in (directory / "traces").glob("*.json.gz")
                if read_trace(path)["row"]["strategy"] == "rollout_test")
    record = read_trace(path)
    mutate(record)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False)
    driver.write_tables(directory, [read_trace(item)["row"]
                                   for item in (directory / "traces").glob("*.json.gz")])


def assert_rejected(output, shards):
    with pytest.raises(ValueError):
        merger.merge(output, shards)
    assert not output.exists(), "Validation must finish before creating the destination"


def test_merge_contiguous_shards_preserves_evidence_and_rebuilds_complete_tables(tmp_path, monkeypatch):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", 5001)
    originals = {shard: archive_snapshot(shard) for shard in (first, second)}

    def must_not_run(*args, **kwargs):
        pytest.fail("Archive merge must not invoke the simulator or solver")

    monkeypatch.setattr(driver, "LocalResearchSimulator", must_not_run)
    monkeypatch.setattr(driver, "run_search", must_not_run)
    output = tmp_path / "merged"
    assert merger.merge(output, [second, first]) == 0
    manifest = read_json(output / "manifest.json")
    cases = read_json(output / "cases.json")
    assert manifest["run_status"] == "completed"
    assert manifest["expected_runs"] == manifest["completed_runs"] == 4
    assert manifest["random_count"] == 2
    assert manifest["random_seed_interval_inclusive"] == [5000, 5001]
    assert manifest["stage"] == "holdout"
    assert manifest["source_sha256_start"] == manifest["source_sha256_end"] == SOURCE_HASHES
    assert manifest["source_changed_during_run"] is False
    assert manifest["cases_sha256"] == driver.value_hash(cases)
    assert sorted(case["seed"] for case in cases) == [5000, 5001]
    assert read_json(output / "configs.json") == CONFIGS
    assert manifest["original_config_file_sha256"] == hashlib.sha256((output / "configs.json").read_bytes()).hexdigest()
    assert len(read_rows(output)) == len(list((output / "traces").glob("*.json.gz"))) == 4
    groups = read_json(output / "summary.json")["groups"]
    candidate = next(group for group in groups if group["strategy"] == "rollout_test")
    assert candidate["runs"] == candidate["successful_pairs"] == 2
    assert candidate["paired_mean_seconds_saved"] == 20.0
    for shard, snapshot in originals.items():
        assert archive_snapshot(shard) == snapshot
        for relative, content in snapshot.items():
            if relative.startswith("traces/"):
                assert (output / relative).read_bytes() == content


@pytest.mark.parametrize("second_seed", [5000, 5002], ids=["overlap", "gap"])
def test_rejects_overlapping_or_gapped_seed_intervals(tmp_path, second_seed):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", second_seed)
    assert_rejected(tmp_path / "merged", [first, second])


@pytest.mark.parametrize("difference", ["configuration", "source_hashes", "max_actions"])
def test_rejects_incompatible_frozen_experiment_settings(tmp_path, difference):
    first = make_shard(tmp_path, "first", 5000)
    options = {}
    if difference == "configuration":
        options["configs"] = [{"name": "rollout_test", "rollout_config": {"particles": 3}}]
    if difference == "source_hashes":
        options["source_hashes"] = {"src/fixture_only.py": "b" * 64}
    second = make_shard(tmp_path, "second", 5001, **options)
    if difference == "max_actions":
        change_manifest(second, max_actions=10000)
    assert_rejected(tmp_path / "merged", [first, second])


@pytest.mark.parametrize("updates", [
    {"run_status": "running"}, {"completed_runs": 1}, {"expected_runs": 3},
    {"source_sha256_end": {"src/fixture_only.py": "b" * 64}},
    {"source_changed_during_run": True}, {"random_count": 2},
    {"stage": "development"}, {"official_formal": True},
], ids=["running", "incomplete_count", "incorrect_expected_count", "changed_sources",
        "source_change_flag", "wrong_seed_count", "not_holdout", "official_formal"])
def test_rejects_unfinished_or_inconsistent_manifest(tmp_path, updates):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", 5001)
    change_manifest(second, **updates)
    assert_rejected(tmp_path / "merged", [first, second])


@pytest.mark.parametrize("corruption", ["case_archive", "case_hash", "config_hash", "trace_path",
                                       "trace_config", "evaluation_truth", "missing_trace", "config_bytes"])
def test_rejects_corrupt_or_mismatched_evidence(tmp_path, corruption):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", 5001)
    if corruption == "case_archive":
        cases = read_json(second / "cases.json")
        cases[0]["sources"][0]["x"] += 1
        driver.write_json(second / "cases.json", cases)
    elif corruption == "missing_trace":
        next((second / "traces").glob("*.json.gz")).unlink()
    elif corruption == "config_bytes":
        with (second / "configs.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        def mutate(record):
            if corruption == "case_hash":
                record["row"]["case_sha256"] = "0" * 64
            elif corruption == "config_hash":
                record["row"]["config_sha256"] = "0" * 64
            elif corruption == "trace_path":
                record["row"]["trace"] = "traces/wrong.json.gz"
            elif corruption == "trace_config":
                record["config"]["particles"] = 3
            elif corruption == "evaluation_truth":
                record["evaluation"]["ground_truth"]["sources"][0]["x"] += 1
        change_trace(second, mutate)
    assert_rejected(tmp_path / "merged", [first, second])


def test_rejects_csv_that_disagrees_with_complete_trace_rows(tmp_path):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", 5001)
    rows = read_rows(second)
    rows[0]["virtual_time_s"] = "999.0"
    with (second / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert_rejected(tmp_path / "merged", [first, second])


@pytest.mark.parametrize("flag", ["official_practice", "official_formal"])
def test_rejects_official_flags_even_when_all_shards_agree(tmp_path, flag):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", 5001)
    for shard in (first, second):
        change_manifest(shard, **{flag: True})
    assert_rejected(tmp_path / "merged", [first, second])


def test_unlisted_trace_is_rejected_instead_of_silently_dropped(tmp_path):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", 5001)
    case = read_json(second / "cases.json")[0]
    extra = make_record(case, "unlisted_failed_policy", {"particles": 2}, failed=True)
    driver.save_trace(second, extra)
    assert len(list((second / "traces").glob("*.json.gz"))) == 3
    assert_rejected(tmp_path / "merged", [first, second])


def test_existing_output_is_never_overwritten(tmp_path):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", 5001)
    output = tmp_path / "merged"
    output.mkdir()
    (output / "keep.txt").write_text("already present", encoding="utf-8")
    before = archive_snapshot(output)
    with pytest.raises(ValueError):
        merger.merge(output, [first, second])
    assert archive_snapshot(output) == before


def test_completed_failed_run_is_preserved_and_not_counted_as_a_timing_win(tmp_path):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", 5001, failed=True)
    relative = driver.trace_path("q3-random-5001", "rollout_test")
    failed_bytes = (second / relative).read_bytes()
    output = tmp_path / "merged"
    assert merger.merge(output, [first, second]) == 1
    manifest = read_json(output / "manifest.json")
    assert manifest["run_status"] == "completed"
    assert manifest["expected_runs"] == manifest["completed_runs"] == 4
    assert manifest["failed_runs"] == 1
    rows = read_rows(output)
    assert len(rows) == 4
    failed_row = next(row for row in rows if row["trace"] == relative)
    assert failed_row["successful"] == failed_row["pair_successful"] == "False"
    assert failed_row["paired_win"] == ""
    assert (output / relative).read_bytes() == failed_bytes
    group = next(group for group in read_json(output / "summary.json")["groups"]
                 if group["strategy"] == "rollout_test")
    assert group["runs"] == 2 and group["failed_runs"] == 1
    assert group["successful_runs"] == group["successful_pairs"] == group["paired_wins"] == 1
    assert group["successful_mean_virtual_time_s"] == 100.0
    assert group["mean_virtual_time_s"] == 50.5
    assert group["all_rows_eligible_for_time_comparison"] is False


def test_cli_accepts_output_and_positional_shard_paths(tmp_path):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", 5001)
    output = tmp_path / "merged"
    assert merger.main(["--output", str(output), str(first), str(second)]) == 0
    assert read_json(output / "manifest.json")["completed_runs"] == 4


def test_continuation_compute_budget_fallback_is_reported_without_changing_trace(tmp_path):
    first = make_shard(tmp_path, "first", 5000)
    second = make_shard(tmp_path, "second", 5001)

    def mutate(record):
        record["row"]["planning_fallback_counts"] = {"incomplete_continuation": 1}
        record["summary"]["planning"] = {"decisions": [
            {"status": "incomplete_continuation", "detail": "rollout_compute_budget", "search_index": 1}
        ]}

    change_trace(second, mutate)
    relative = driver.trace_path("q3-random-5001", "rollout_test")
    original_trace = (second / relative).read_bytes()
    output = tmp_path / "merged"
    assert merger.merge(output, [first, second]) == 0
    manifest = read_json(output / "manifest.json")
    summary = read_json(output / "summary.json")
    assert manifest["no_compute_budget_fallbacks"] is False
    assert summary["no_compute_budget_fallbacks"] is False
    assert manifest["compute_budget_event_count"] == 1
    assert len(manifest["compute_budget_fallback_rows"]) == 1
    fallback = manifest["compute_budget_fallback_rows"][0]
    expected = {"case_id": "q3-random-5001", "strategy": "rollout_test", "count": 1,
                "reported_compute_budget_count": 0, "continuation_compute_budget_count": 1}
    assert {key: fallback[key] for key in expected} == expected
    provenance = next(shard for shard in manifest["shard_provenance"]
                      if shard["random_seed_interval_inclusive"] == [5001, 5001])
    assert provenance["compute_budget_event_count"] == 1
    assert (second / relative).read_bytes() == original_trace
    assert (output / relative).read_bytes() == original_trace
