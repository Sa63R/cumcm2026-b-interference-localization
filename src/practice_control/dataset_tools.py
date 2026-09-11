"""Read-only statistics/audit and consistent SQLite snapshots for training."""

import argparse
import json
from pathlib import Path
import sqlite3

from .dataset import stats


def audit(path, *, expected_q3=None, expected_q4=None):
    database = Path(path).resolve(strict=True)
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=30) as connection:
        integrity = [row[0] for row in connection.execute("PRAGMA quick_check")]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        bad_counts = connection.execute("""SELECT e.case_code FROM episodes e WHERE
          e.step_count != (SELECT COUNT(*) FROM steps s WHERE s.episode_id=e.id)
          OR e.source_estimate_count != (SELECT COUNT(*) FROM source_estimates p WHERE p.episode_id=e.id)
          OR e.cleared_count != (SELECT COUNT(*) FROM source_estimates p WHERE p.episode_id=e.id AND p.cleared=1)
          OR (e.complete=1 AND e.source_total!=e.cleared_count)""").fetchall()
        bad_costs = connection.execute("""SELECT COUNT(*) FROM steps
          WHERE delta_virtual_time_s < -0.00001
          OR abs(delta_virtual_time_s + reward_time_component)>0.00001
          OR abs(virtual_time_after_s - virtual_time_before_s - delta_virtual_time_s)>0.00001""").fetchone()[0]
        bad_labels = connection.execute("""SELECT COUNT(*) FROM source_estimates
          WHERE cleared=1 AND (radius_m IS NULL OR radius_m<0 OR radius_m>20.000001
          OR x_m IS NULL OR y_m IS NULL)""").fetchone()[0]
    result = stats(database)
    result.update(integrity_ok=integrity == ["ok"] and not foreign_keys and not bad_counts
                  and bad_costs == 0 and bad_labels == 0,
                  sqlite_check=integrity, foreign_key_errors=len(foreign_keys),
                  episode_count_errors=len(bad_counts), transition_cost_errors=bad_costs,
                  source_bound_errors=bad_labels)
    expected = {"3": expected_q3, "4": expected_q4}
    result["targets_met"] = all(count is None or result["complete_by_problem"][key] >= count
                                for key, count in expected.items())
    return result


def snapshot(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("Snapshot destination must be different from the active dataset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb"):
        pass
    try:
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=30) as reader:
            with sqlite3.connect(destination) as writer:
                reader.backup(writer, pages=256, sleep=0.05)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {"snapshot": str(destination), "bytes": destination.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect or snapshot the practice training dataset")
    parser.add_argument("database", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats")
    check = sub.add_parser("audit")
    check.add_argument("--expected-q3", type=int)
    check.add_argument("--expected-q4", type=int)
    backup = sub.add_parser("snapshot")
    backup.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "snapshot":
        result = snapshot(args.database, args.output)
    elif args.command == "audit":
        result = audit(args.database, expected_q3=args.expected_q3, expected_q4=args.expected_q4)
    else:
        result = stats(args.database)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result.get("integrity_ok") is False or result.get("targets_met") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
