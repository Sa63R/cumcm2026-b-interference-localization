"""End-to-end offline registration of a session with different wall clocks."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pytest

from experiments.register_practice import register
from practice_control.dataset import import_episode, stats
from tests.test_practice_dataset import example, journal, load, write, write_journal


def clock_skew_episode(tmp_path):
    episode = example(tmp_path / "raw")
    summary = load(episode / "summary.json")
    summary["started_at"] = "2026-09-11T10:00:11+00:00"
    write(episode / "summary.json", summary)
    rows = journal(episode)
    base = int(datetime(2026, 9, 11, 10, 0, 4, tzinfo=timezone.utc).timestamp() * 1000)
    offset = 0
    for row in rows:
        row["run_id"] = "private-run-id"
        if row["event"] == "response":
            server_ms = base + offset
            row["response"]["real_timestamp_ms"] = server_ms
            row["recorded_at"] = datetime.fromtimestamp(
                server_ms / 1000 + 7, timezone.utc).isoformat()
            offset += 1
    write_journal(episode, rows)
    official_path = tmp_path / f"practice-p3-fixture-{summary['case_code']}.result.json"
    official_path.write_bytes((episode.parent / "registered/evidence/archived.result.json").read_bytes())
    args = argparse.Namespace(
        summary=episode / "summary.json",
        official_result_json=official_path,
        output_dir=tmp_path / "verified/registered",
        case_code=summary["case_code"], source_total=None, runtime=None,
    )
    return episode, args


def test_register_and_import_server_clock_without_changing_original_evidence(tmp_path):
    episode, args = clock_skew_episode(tmp_path)
    original = {p: p.read_bytes() for p in (args.summary, args.official_result_json,
                                           episode / "requests.jsonl")}
    registration = register(args)
    write(episode / "registration.json", registration)
    record = load(Path(registration["record"]))
    evidence = record["session_time_evidence"]
    assert evidence["basis"] == "simulator_http_enter"
    assert evidence["journal_sha256"] == hashlib.sha256(original[episode / "requests.jsonl"]).hexdigest()
    assert (args.output_dir / record["session_time_journal_path"]).read_bytes() == original[episode / "requests.jsonl"]
    database = tmp_path / "data.sqlite3"
    assert import_episode(database, episode)["complete"]
    assert stats(database)["complete_episodes"] == 1
    assert all(p.read_bytes() == content for p, content in original.items())


@pytest.mark.parametrize("tamper", ["journal", "archive", "proof", "missing_proof"])
def test_import_rejects_changed_clock_proof_or_journal(tmp_path, tamper):
    episode, args = clock_skew_episode(tmp_path)
    registration = register(args)
    write(episode / "registration.json", registration)
    record = load(Path(registration["record"]))
    if tamper == "journal":
        with (episode / "requests.jsonl").open("ab") as stream:
            stream.write(b"\n")
    elif tamper == "archive":
        with (args.output_dir / record["session_time_journal_path"]).open("ab") as stream:
            stream.write(b"\n")
    elif tamper == "proof":
        record["session_time_evidence"]["validated_accepted_count"] += 1
        write(Path(registration["record"]), record)
    else:
        record.pop("session_time_evidence")
        write(Path(registration["record"]), record)
    database = tmp_path / "data.sqlite3"
    with pytest.raises(ValueError, match="time.evidence|session time evidence"):
        import_episode(database, episode)
    assert not database.exists()
