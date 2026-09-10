"""Practice metrics use a GUI total and retain distinct evidence provenance."""

import argparse
import csv
import hashlib
import json

import pytest

from experiments.register_practice import register


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
