"""Tiny local fixtures for the read-only final-benchmark integrity gate."""

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from scripts.verify_official_practice_benchmark import verify


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_official_practice_benchmark.py"


@pytest.fixture
def benchmark(tmp_path):
    database = tmp_path / "snapshot.sqlite3"
    with sqlite3.connect(database) as writer:
        writer.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, problem INTEGER NOT NULL)")
        writer.executemany("INSERT INTO episodes (problem) VALUES (?)", [(3,), (4,)])
        schema = writer.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name"
        ).fetchall()
    manifest = {
        "benchmark_id": "tiny-audit-only-v1",
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "schema_sha256": hashlib.sha256(json.dumps(
            schema, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "counts": {"episodes": 2, "by_problem": {"3": 1, "4": 1}},
        "capability": "logged_trajectory_audit_only",
        "policy_usage": "evaluation_only_no_training_or_tuning",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, database, manifest


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_cli_verifies_without_writes_or_performance_claim(benchmark):
    manifest, database, _ = benchmark
    before = {path.name: path.read_bytes() for path in database.parent.iterdir()}
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--database", str(database)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["verified"] is True
    assert result["capability"] == "logged_trajectory_audit_only"
    assert result["performance_evaluation_ready"] is False
    assert result["candidate_mean_time_s"] is None
    assert result["candidate_time_over_lower_bound"] is None
    assert {path.name: path.read_bytes() for path in database.parent.iterdir()} == before


def test_database_tampering_is_rejected_and_manifest_not_updated(benchmark):
    path, database, _ = benchmark
    frozen_manifest = path.read_bytes()
    with sqlite3.connect(database) as writer:
        writer.execute("UPDATE episodes SET problem=4 WHERE problem=3")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(path), "--database", str(database)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    result = json.loads(completed.stdout)
    assert result["verified"] is False
    assert "SHA256 mismatch" in result["error"]
    assert path.read_bytes() == frozen_manifest


@pytest.mark.parametrize("field,value", [
    ("benchmark_id", " "),
    ("database_sha256", "A" * 64),
    ("schema_sha256", "invalid"),
    ("capability", "full_policy_evaluation"),
    ("policy_usage", "training_allowed"),
    ("counts", {"episodes": True, "by_problem": {"3": 1, "4": 0}}),
    ("counts", {"episodes": 2, "by_problem": {"3": 1, "4": 2}}),
    ("counts", {"episodes": 2, "by_problem": {"3": 1, "5": 1}}),
])
def test_invalid_manifest_fields_are_rejected(benchmark, field, value):
    path, database, manifest = benchmark
    manifest[field] = value
    write_manifest(path, manifest)
    with pytest.raises(ValueError):
        verify(path, database)


def test_schema_digest_and_counts_are_independently_checked(benchmark):
    path, database, manifest = benchmark
    manifest["schema_sha256"] = "0" * 64
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="Schema SHA256 mismatch"):
        verify(path, database)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    with sqlite3.connect(database) as reader:
        schema = reader.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name"
        ).fetchall()
    manifest["schema_sha256"] = hashlib.sha256(json.dumps(
        schema, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    manifest["counts"] = {"episodes": 2, "by_problem": {"3": 2, "4": 0}}
    write_manifest(path, manifest)
    with pytest.raises(ValueError, match="Episode counts mismatch"):
        verify(path, database)


def test_nonempty_journal_is_rejected(benchmark):
    path, database, _ = benchmark
    Path(str(database) + "-wal").write_bytes(b"uncommitted snapshot changes")
    with pytest.raises(ValueError, match="standalone snapshot"):
        verify(path, database)


def test_corrupt_sqlite_is_rejected_even_if_file_digest_matches(benchmark):
    path, database, manifest = benchmark
    database.write_bytes(b"not a SQLite database")
    manifest["database_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    write_manifest(path, manifest)
    with pytest.raises(sqlite3.DatabaseError):
        verify(path, database)
