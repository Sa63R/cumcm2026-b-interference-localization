"""Practice metrics use a GUI total and retain distinct evidence provenance."""

import argparse
import csv
import hashlib
import json

import pytest

from experiments.register_practice import main, register


def inputs(tmp_path, **changes):
    summary = {
        "data_origin": "simulator_http_session", "declared_mode": "practice",
        "problem": 3, "case_code": "PRACTICE-A", "variant": "adaptive",
        "completed": False, "pending_request": None, "program_wall_time_s": 2.5,
        "state": {"session": "exited", "cleared_count": 12, "virtual_time_s": 1200},
    }
    summary.update(changes)
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return argparse.Namespace(summary=path, source_total=15, case_code="PRACTICE-A",
                              runtime=3.0, output_dir=tmp_path / "registered")


def test_gui_total_derives_metrics_and_preserves_summary(tmp_path):
    args = inputs(tmp_path)
    raw = args.summary.read_bytes()
    result = register(args)
    assert result["clearance_ratio"] == 0.8
    assert result["average_clear_time_s"] == 100
    assert result["runtime_source"] == "official_gui_user_transcribed"
    records = list(args.output_dir.glob("practice-*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["source_total_source"] == "official_gui_user_transcribed"
    assert record["summary_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (args.output_dir / record["summary_path"]).read_bytes() == raw
    assert record["search_completed"] is False  # A partial, exited practice is valid data.
    with (args.output_dir / "总表.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["clearance_ratio"]) == 0.8


def test_reject_research_formal_and_unfinished_sessions(tmp_path):
    invalid = [
        {"data_origin": "synthetic_research"},
        {"declared_mode": "formal"},
        {"state": {"session": "active", "cleared_count": 12, "virtual_time_s": 1200}},
        {"pending_request": {"path": "/clear"}},
    ]
    for patch in invalid:
        args = inputs(tmp_path, **patch)
        with pytest.raises(ValueError):
            register(args)
        assert not args.output_dir.exists()


def test_reject_invalid_gui_totals_and_overcount(tmp_path):
    for total in (9, 17, True, 10.5, 11):
        args = inputs(tmp_path)
        args.source_total = total
        with pytest.raises(ValueError):
            register(args)
        assert not args.output_dir.exists()


def test_zero_clear_is_undefined_and_local_runtime_is_labelled(tmp_path):
    args = inputs(tmp_path, state={"session": "exited", "cleared_count": 0, "virtual_time_s": 300})
    args.runtime = None
    result = register(args)
    assert result["clearance_ratio"] == 0
    assert result["average_clear_time_s"] is None
    assert result["program_runtime_s"] == 2.5
    assert result["runtime_source"] == "local_program_wall_time"
    with (args.output_dir / "总表.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert next(csv.DictReader(stream))["average_clear_time_s"] == "undefined"


def test_case_mismatch_and_duplicate_do_not_overwrite_registry(tmp_path):
    args = inputs(tmp_path)
    args.case_code = "WRONG"
    with pytest.raises(ValueError, match="does not match"):
        register(args)
    assert not args.output_dir.exists()
    args.case_code = "PRACTICE-A"
    register(args)
    before = (args.output_dir / "总表.csv").read_bytes()
    with pytest.raises(ValueError, match="already registered"):
        register(args)
    assert (args.output_dir / "总表.csv").read_bytes() == before
    assert len(list(args.output_dir.glob("practice-*.json"))) == 1


def official_inputs(tmp_path, *, result_patch=None, summary_patch=None):
    summary_fields = {"started_at": "2026-09-10T22:20:54+08:00"}
    summary_fields.update(summary_patch or {})
    args = inputs(tmp_path, **summary_fields)
    result = {
        "version": 2, "problem_no": 3, "practice_run_no": 123456789,
        "case_code": "PRACTICE-A", "package_sha256": "a" * 64,
        "window_started_at_utc": "2026-09-10T14:20:48.264Z",
        "ended_at_utc": "2026-09-10T14:21:05.281Z",
        "jammer_count": 15, "omnidirectional_jammer_count": 15,
        "directional_jammer_count": 0,
    }
    result.update(result_patch or {})
    args.official_result_json = tmp_path / "unit-test-only.result.json"
    args.official_result_json.write_bytes((" \n" + json.dumps(result, indent=2) + "\n").encode("utf-8"))
    args.case_code, args.source_total, args.runtime = None, None, None
    return args


def test_official_result_cli_infers_metrics_and_archives_original_bytes(tmp_path):
    args = official_inputs(tmp_path)
    raw = args.official_result_json.read_bytes()
    assert main(["--summary", str(args.summary), "--official-result-json", str(args.official_result_json),
                 "--output-dir", str(args.output_dir)]) == 0
    record_path = next(args.output_dir.glob("practice-*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["case_code"] == "PRACTICE-A"
    assert record["source_total"] == 15
    assert record["source_total_source"] == "official_simulator_result_file"
    assert record["clearance_ratio"] == 0.8
    assert record["official_result_session_time_matched"] is True
    assert record["official_result_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (args.output_dir / record["official_result_path"]).read_bytes() == raw
    assert record["runtime_source"] == "local_program_wall_time"
    assert record["program_runtime_s"] == 2.5  # Never use the 17.017 s result window.
    with (args.output_dir / "总表.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["source_total_source"] == "official_simulator_result_file"
    assert row["official_result_sha256"] == record["official_result_sha256"]


def test_explicit_values_must_match_official_result(tmp_path):
    args = official_inputs(tmp_path)
    args.source_total = 16
    with pytest.raises(ValueError, match="Explicit source total"):
        register(args)
    assert not args.output_dir.exists()
    args.source_total, args.case_code = 15, "WRONG"
    with pytest.raises(ValueError, match="Explicit case code"):
        register(args)
    assert not args.output_dir.exists()
    args.case_code = "PRACTICE-A"
    assert register(args)["source_total_source"] == "official_simulator_result_file"


@pytest.mark.parametrize("result_patch,summary_patch", [
    ({"problem_no": 4}, {}),
    ({"omnidirectional_jammer_count": 14}, {}),
    ({"omnidirectional_jammer_count": 14, "directional_jammer_count": 1}, {}),
    ({"jammer_count": 9, "omnidirectional_jammer_count": 9}, {}),
    ({"directional_jammer_count": False}, {}),
    ({"case_code": "DIFFERENT"}, {}),
    ({"ended_at_utc": None}, {}),
    ({"ended_at_utc": "2026-09-10T14:19:00Z"}, {}),
    ({"window_started_at_utc": "2026-09-10T14:20:48"}, {}),
    ({}, {"started_at": "2026-09-10T14:19:00Z"}),
    ({}, {"started_at": "2026-09-10T14:22:00Z"}),
    ({}, {"started_at": None}),
    ({}, {"problem": 4}),
])
def test_official_result_rejects_wrong_case_counts_or_session_time(tmp_path, result_patch, summary_patch):
    args = official_inputs(tmp_path, result_patch=result_patch, summary_patch=summary_patch)
    with pytest.raises(ValueError):
        register(args)
    assert not args.output_dir.exists()


def test_official_result_keeps_exit_and_duplicate_checks(tmp_path):
    args = official_inputs(tmp_path, summary_patch={"pending_request": {"path": "/clear"}})
    with pytest.raises(ValueError, match="no pending action"):
        register(args)
    assert not args.output_dir.exists()
    args = official_inputs(tmp_path)
    register(args)
    before = (args.output_dir / "总表.csv").read_bytes()
    with pytest.raises(ValueError, match="already registered"):
        register(args)
    assert (args.output_dir / "总表.csv").read_bytes() == before


def test_official_q4_result_rejects_no_directional_sources(tmp_path):
    args = official_inputs(tmp_path, result_patch={"problem_no": 4}, summary_patch={"problem": 4})
    with pytest.raises(ValueError, match="at least one directional"):
        register(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("total,omni,directional", [(14, 0, 14), (15, 8, 7), (10, 0, 10), (16, 0, 16)])
def test_official_q4_accepts_mixed_and_all_directional_counts(tmp_path, total, omni, directional):
    # The ended R6YB-84GC-MB79-EHP6 practice had 14 sources, all directional.
    args = official_inputs(tmp_path,
                           result_patch={"problem_no": 4, "jammer_count": total,
                                         "omnidirectional_jammer_count": omni,
                                         "directional_jammer_count": directional},
                           summary_patch={"problem": 4, "completed": True,
                                          "state": {"session": "exited", "cleared_count": total,
                                                    "virtual_time_s": 1200}})
    original = args.official_result_json.read_bytes()
    result = register(args)
    assert result["source_total"] == total and result["clearance_ratio"] == 1.0
    record = json.loads(next(args.output_dir.glob("practice-*.json")).read_text(encoding="utf-8"))
    assert record["omnidirectional_source_total"] == omni
    assert record["directional_source_total"] == directional
    assert record["search_completed"] is True
    assert record["official_result_sha256"] == hashlib.sha256(original).hexdigest()
    assert (args.output_dir / record["official_result_path"]).read_bytes() == original


@pytest.mark.parametrize("omni,directional", [(-1, 16), (0, 14), (0, True), (0, 15.0)])
def test_official_q4_still_rejects_invalid_type_counts(tmp_path, omni, directional):
    args = official_inputs(tmp_path,
                           result_patch={"problem_no": 4, "omnidirectional_jammer_count": omni,
                                         "directional_jammer_count": directional},
                           summary_patch={"problem": 4})
    with pytest.raises(ValueError, match="source counts"):
        register(args)
    assert not args.output_dir.exists()
