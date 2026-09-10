"""Evidence registration must not turn local research into official results."""

import argparse
import csv
import hashlib
import json

import pytest

from workflow.__main__ import audit, main, register
from workflow.evidence import confirm_upload, export_tables
from tests.fake_simulator import FakeSimulator


def inputs(tmp_path):
    ledger = tmp_path / "ledger.csv"
    fields = ["problem", "slot", "case_code", "cleared_count", "virtual_time_s",
              "average_clear_time_s", "program_runtime_s", "runtime_source",
              "summary_path", "official_log", "sha256", "uploaded_confirmed"]
    with ledger.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for problem in (3, 4):
            for slot in (1, 2, 3):
                writer.writerow({"problem": problem, "slot": slot})
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "data_origin": "simulator_http_session", "declared_mode": "formal",
        "problem": 3, "case_code": "CASE-A", "state": {"cleared_count": 10, "virtual_time_s": 2000},
    }))
    log = tmp_path / "original-example.encrypted"
    log.write_bytes(b"local-test-placeholder-not-an-official-log")
    (tmp_path / "requests.jsonl").write_text('{"event":"unit_test_fixture_not_official"}\n')
    return argparse.Namespace(ledger=ledger, summary=summary, official_log=log,
                              problem=3, slot=1, case_code="CASE-A", runtime=30.0, uploaded=False)


def test_registration_preserves_log_and_derives_virtual_average(tmp_path):
    args = inputs(tmp_path)
    assert register(args) == 0
    with args.ledger.open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["average_clear_time_s"]) == 200
    assert row["runtime_source"] == "official_gui_user_transcribed"
    assert row["sha256"] == hashlib.sha256(args.official_log.read_bytes()).hexdigest()
    assert (tmp_path / "正式日志/问题3" / args.official_log.name).read_bytes() == args.official_log.read_bytes()
    assert audit(args) == 1  # Other formal sessions and upload confirmation remain missing.
    with pytest.raises(ValueError, match="already registered"):
        register(args)


@pytest.mark.parametrize("patch", [
    {"data_origin": "synthetic_research"}, {"declared_mode": "practice"}, {"problem": 4},
])
def test_reject_nonformal_or_mismatched_evidence(tmp_path, patch):
    args = inputs(tmp_path)
    report = json.loads(args.summary.read_text())
    report.update(patch)
    args.summary.write_text(json.dumps(report))
    before = args.ledger.read_bytes()
    with pytest.raises(ValueError):
        register(args)
    assert args.ledger.read_bytes() == before


def test_zero_clear_average_is_undefined(tmp_path):
    args = inputs(tmp_path)
    report = json.loads(args.summary.read_text())
    report["state"]["cleared_count"] = 0
    args.summary.write_text(json.dumps(report))
    register(args)
    with args.ledger.open(encoding="utf-8") as stream:
        assert next(csv.DictReader(stream))["average_clear_time_s"] == "undefined"


def test_audit_detects_altered_log(tmp_path, capsys):
    args = inputs(tmp_path)
    register(args)
    archived = tmp_path / "正式日志/问题3" / args.official_log.name
    archived.write_bytes(b"changed")
    assert audit(args) == 1
    assert "official log changed" in capsys.readouterr().out


def test_action_budget_is_not_cli_success(tmp_path):
    with FakeSimulator() as simulator:
        destination = tmp_path / "session"
        assert main(["run", "--problem", "3", "--mode", "practice",
                     "--robot-id", "test-team", "--base-url", simulator.base_url,
                     "--max-actions", "2", "--output", str(destination)]) == 1
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        assert summary["completed"] is False
        assert summary["search"]["completion_reason"] == "action_budget"
        assert summary["error"] == "SearchIncomplete: action_budget"
        assert summary["state"]["session"] == "exited"
        assert [request[1] for request in simulator.requests] == ["/enter", "/exit"]


@pytest.mark.parametrize("problem,explicit_variant,expected_variant,policy", [
    (3, None, "efficient", "center"),
    (3, "adaptive", "adaptive", "center"),
    (3, "efficient", "efficient", "center"),
    (4, None, "triangular", "center"),
    (3, "deferred", "deferred", "minimax"),
    (4, "adaptive", "adaptive", "center"),
    (4, "baseline", "baseline", "center"),
    (4, "triangular", "triangular", "center"),
])
def test_cli_strategy_defaults_and_overrides_reach_search(tmp_path, problem, explicit_variant,
                                                        expected_variant, policy):
    with FakeSimulator() as simulator:
        destination = tmp_path / "session"
        arguments = ["run", "--problem", str(problem), "--mode", "practice",
                     "--robot-id", "test-team", "--base-url", simulator.base_url,
                     "--max-actions", "2", "--active-policy", policy,
                     "--output", str(destination)]
        if explicit_variant is not None:
            arguments += ["--variant", explicit_variant]
        assert main(arguments) == 1  # Deliberately exhaust the budget before any scan.
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        assert summary["variant"] == summary["search"]["variant"] == expected_variant
        assert summary["active_policy"] == summary["search"]["active_policy"] == policy


def test_q3_triangular_rejected_before_contact_or_output(tmp_path):
    with FakeSimulator() as simulator:
        destination = tmp_path / "invalid"
        with pytest.raises(SystemExit) as error:
            main(["run", "--problem", "3", "--mode", "practice", "--variant", "triangular",
                  "--robot-id", "test-team", "--base-url", simulator.base_url,
                  "--output", str(destination)])
        assert error.value.code == 2
        assert not simulator.requests
        assert not destination.exists()


def test_strategy_caught_protocol_error_reaches_cli(tmp_path):
    with FakeSimulator() as simulator:
        simulator.reject_next(200)
        destination = tmp_path / "session"
        assert main(["run", "--problem", "3", "--mode", "practice",
                     "--robot-id", "test-team", "--base-url", simulator.base_url,
                     "--output", str(destination)]) == 1
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        assert summary["search"]["completion_reason"] == "protocol_error"
        assert summary["error"]
        assert summary["completed"] is False


def test_keyboard_interrupt_saves_evidence_and_exits_known_session(tmp_path, monkeypatch):
    import strategies

    def interrupted_search(client, **kwargs):
        client.enter()
        raise KeyboardInterrupt

    monkeypatch.setattr(strategies, "run_search", interrupted_search)
    with FakeSimulator() as simulator:
        destination = tmp_path / "interrupted"
        assert main(["run", "--problem", "3", "--mode", "practice",
                     "--robot-id", "test-team", "--base-url", simulator.base_url,
                     "--output", str(destination)]) == 1
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        assert summary["completed"] is False
        assert summary["state"]["session"] == "exited"
        assert summary["error"].startswith("KeyboardInterrupt")
        assert (destination / "requests.jsonl").stat().st_size > 0


def test_successful_http_session_preserves_complete_report(tmp_path):
    # A ten-source protocol fixture, never an official performance result.
    with FakeSimulator() as simulator:
        simulator._sources = {channel: (channel * 30.0, channel * 20.0) for channel in range(1, 11)}
        destination = tmp_path / "session"
        assert main(["run", "--problem", "3", "--mode", "practice",
                     "--robot-id", "test-team", "--base-url", simulator.base_url,
                     "--output", str(destination)]) == 0
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        assert summary["completed"] is True
        assert summary["state"]["session"] == "exited"
        assert summary["state"]["cleared_count"] == 10
        assert summary["official_result_verified"] is False
        assert summary["search"]["action_history"]


@pytest.mark.parametrize("field,value", [
    ("runtime", float("nan")), ("runtime", -1),
    ("cleared_count", 17), ("cleared_count", True),
    ("virtual_time", float("inf")),
])
def test_invalid_gui_values_do_not_change_ledger(tmp_path, field, value):
    args = inputs(tmp_path)
    args.cleared_count, args.virtual_time = 10, 2000
    setattr(args, field, value)
    before = args.ledger.read_bytes()
    with pytest.raises(ValueError):
        register(args)
    assert args.ledger.read_bytes() == before


def test_unknown_action_requires_gui_metrics(tmp_path):
    args = inputs(tmp_path)
    report = json.loads(args.summary.read_text())
    report["pending_request"] = {"path": "/clear", "payload": {}}
    args.summary.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Unresolved action"):
        register(args)
    args.cleared_count, args.virtual_time = 11, 2005
    register(args)
    with args.ledger.open(encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["cleared_count"] == "11"
    assert row["result_source"] == "official_gui_user_transcribed"


def test_audit_checks_six_distinct_archived_sessions_and_portable_paths(tmp_path, monkeypatch):
    args = inputs(tmp_path)
    original_report = json.loads(args.summary.read_text())
    for problem in (3, 4):
        for slot in (1, 2, 3):
            args.problem, args.slot, args.case_code = problem, slot, f"CASE-{problem}-{slot}"
            report = dict(original_report, problem=problem, case_code=args.case_code)
            args.summary.write_text(json.dumps(report))
            args.official_log = tmp_path / f"case-{problem}-{slot}.encrypted"
            args.official_log.write_bytes(f"unit-test-not-official:{problem}:{slot}".encode())
            register(args)
            confirm_upload(args)
    # Original source files can disappear after byte-for-byte archival.
    args.summary.unlink()
    (tmp_path / "requests.jsonl").unlink()
    monkeypatch.chdir(tmp_path.parent)
    assert audit(args) == 0
    assert "当前材料审计：通过" in (tmp_path / "正式结果表.md").read_text(encoding="utf-8")
    archived_summary = tmp_path / "正式记录/问题3/slot-1/summary.json"
    archived_summary.write_text("{}")
    assert audit(args) == 1


def test_corrupt_table_numbers_are_detected(tmp_path, capsys):
    args = inputs(tmp_path)
    register(args)
    with args.ledger.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[0]["average_clear_time_s"] = "99"
    with args.ledger.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    assert audit(args) == 1
    assert "average does not equal" in capsys.readouterr().out


def test_export_empty_formal_tables_has_no_invented_results(tmp_path):
    args = inputs(tmp_path)
    assert export_tables(args) == 0
    assert (tmp_path / "正式结果表.md").read_text(encoding="utf-8").count("待正式测试") == 6
    with (tmp_path / "正式结果表.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert all(row["case_code"] == row["cleared_count"] == "" for row in rows)
