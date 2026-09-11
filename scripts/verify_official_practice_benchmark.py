"""Verify a frozen practice-trajectory audit set without evaluating any policy."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3


CAPABILITY = "logged_trajectory_audit_only"
POLICY_USAGE = "evaluation_only_no_training_or_tuning"
SCHEMA_SQL = (
    "SELECT type,name,tbl_name,sql FROM sqlite_master "
    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate manifest key: " + key)
        result[key] = value
    return result


def validate_manifest(manifest):
    require(isinstance(manifest, dict), "Manifest must be an object")
    require(isinstance(manifest.get("benchmark_id"), str)
            and bool(manifest["benchmark_id"].strip()), "Missing benchmark_id")
    for field in ("database_sha256", "schema_sha256"):
        require(isinstance(manifest.get(field), str)
                and re.fullmatch(r"[0-9a-f]{64}", manifest[field]) is not None,
                field + " must be a lowercase SHA256")
    require(manifest.get("capability") == CAPABILITY, "Unsupported capability")
    require(manifest.get("policy_usage") == POLICY_USAGE, "Unsupported policy_usage")
    counts = manifest.get("counts")
    require(isinstance(counts, dict), "Missing counts")
    require(type(counts.get("episodes")) is int and counts["episodes"] > 0,
            "counts.episodes must be a positive integer")
    by_problem = counts.get("by_problem")
    require(isinstance(by_problem, dict) and set(by_problem) == {"3", "4"},
            "counts.by_problem must contain exactly problems 3 and 4")
    require(all(type(value) is int and value >= 0 for value in by_problem.values()),
            "Problem counts must be nonnegative integers")
    require(sum(by_problem.values()) == counts["episodes"], "Manifest counts do not sum")


def require_standalone(database):
    # A main-file digest does not authenticate changes held in an active journal.
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(database) + suffix)
        require(not sidecar.exists() or sidecar.stat().st_size == 0,
                "Database has an active journal; use a consistent standalone snapshot")


def verify(manifest_path, database_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    database = Path(database_path).resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes, object_pairs_hook=unique_object)
    validate_manifest(manifest)
    require_standalone(database)
    require(file_sha256(database) == manifest["database_sha256"], "Database SHA256 mismatch")

    # Immutable mode avoids creating SQLite sidecars. The caller must supply the
    # frozen standalone snapshot, whose bytes are checked before and after reads.
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)) as reader:
        reader.execute("PRAGMA query_only=ON")
        reader.execute("BEGIN")
        require(reader.execute("PRAGMA quick_check").fetchall() == [("ok",)],
                "SQLite quick_check failed")
        schema = reader.execute(SCHEMA_SQL).fetchall()
        schema_bytes = json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        require(hashlib.sha256(schema_bytes).hexdigest() == manifest["schema_sha256"],
                "Schema SHA256 mismatch")
        episode_count = reader.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        problems = reader.execute("SELECT problem,COUNT(*) FROM episodes GROUP BY problem").fetchall()
        require(all(type(problem) is int and problem in (3, 4) for problem, _ in problems),
                "Database contains an unsupported problem")
        by_problem = {"3": 0, "4": 0}
        by_problem.update({str(problem): count for problem, count in problems})
        actual_counts = {"episodes": episode_count, "by_problem": by_problem}
        require(episode_count == manifest["counts"]["episodes"]
                and by_problem == manifest["counts"]["by_problem"], "Episode counts mismatch")

    require_standalone(database)
    require(file_sha256(database) == manifest["database_sha256"], "Database changed during verification")
    require(manifest_path.read_bytes() == manifest_bytes, "Manifest changed during verification")
    return {
        "verified": True,
        "benchmark_id": manifest["benchmark_id"],
        "capability": CAPABILITY,
        "policy_usage": POLICY_USAGE,
        "counts": actual_counts,
        "performance_evaluation_ready": False,
        "candidate_mean_time_s": None,
        "candidate_time_over_lower_bound": None,
        "scope": "Snapshot integrity only; no policy run or counterfactual performance estimate",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.manifest, args.database)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(json.dumps({"verified": False, "error": str(exc),
                          "performance_evaluation_ready": False,
                          "candidate_mean_time_s": None,
                          "candidate_time_over_lower_bound": None}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
