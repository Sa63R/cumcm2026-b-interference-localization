"""Offline Q4 holdout evidence audit; never a counterfactual simulator.

Only ``snapshot`` reads the live collection, using SQLite's read-only backup API.
``audit`` requires that immutable snapshot and an explicit algorithm-freeze file.
It opens only completed Q4 validation/test trajectories, preserves collection
refinement costs, and reports the existing conditional all-clear lower bound.
Candidate station JSON: {"candidates": [{"name": "frozen-name", "stations":
[{"position": [0, 0], "channels": [1, 2]}]}]}. Omitted channels means 1..20.
Support is exact equality of coordinates/channel before the original successful
clear. Unsupported observations are never interpolated or assigned a runtime.
Even full station support does not establish support for a whole adaptive policy.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from experiments.session_lower_bounds import analyze as analyze_bounds

VERSION = "q4-observation-holdout-audit-v1"
COMPONENTS = ("movement_s", "switching_s", "detection_s", "optical_s", "removal_s")
HELDOUT_SQL = "problem=4 AND complete=1 AND trajectory_complete=1 AND split IN ('validation','test')"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(value):
    def invalid(text):
        raise ValueError("Nonfinite JSON: " + text)
    return json.loads(value, parse_constant=invalid)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def private_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to((ROOT / "results/practice_batches").resolve()),
            "Private snapshot/audit output must be inside results/practice_batches")
    return path


def connect_readonly(path):
    path = Path(path).resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def snapshot_database(source, output):
    """Consistent backup, not a raw copy of a live database or journal files."""
    source = Path(source).resolve(strict=True)
    output = private_path(output)
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "practice_training.sqlite3"
    # Small backup steps release source read locks between steps. No PRAGMA that
    # changes the live journal, checkpoint, schema or collector is issued.
    with closing(connect_readonly(source)) as reader, closing(sqlite3.connect(destination)) as writer:
        reader.backup(writer, pages=256, sleep=0.05)
    with closing(connect_readonly(destination)) as reader:
        metadata = dict(reader.execute("SELECT key,value FROM metadata"))
        heldout = [dict(row) for row in reader.execute(
            "SELECT id,case_code,split,policy,evidence_sha256 FROM episodes WHERE " + HELDOUT_SQL + " ORDER BY id")]
        counts = [dict(row) for row in reader.execute(
            "SELECT problem,split,COUNT(*) n,SUM(complete) complete FROM episodes GROUP BY problem,split")]
    manifest = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": destination.name, "database_sha256": sha(destination), "metadata": metadata,
        "counts": counts, "selected_completed_q4_holdout": heldout,
        "source_access": "mode=ro SQLite backup; no raw copy or live mutations",
        "heldout_layouts_opened": False}
    write_json(output / "snapshot.json", manifest)
    return manifest


def position(value):
    if isinstance(value, dict):
        require(set(value) == {"x", "y"}, "Position needs exactly x,y")
        value = (value["x"], value["y"])
    require(isinstance(value, (list, tuple)) and len(value) == 2, "Invalid position")
    require(all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 2_000_000 for v in value),
            "Invalid coordinate")
    return tuple(float(v) for v in value)


def close(a, b, name):
    require(type(a) in (int, float) and math.isfinite(a) and abs(a-b) <= 2e-6, name + " mismatch")


def audit_steps(steps, summary):
    """Independent ledger from accepted wire actions, including failed optical checks.

    Movements are rounded to microseconds; clear does not retune the receiver.
    No source positions, label centers, orientations or learned model are used.
    """
    current, tuned, elapsed, active = (0., 0.), 1, 0, False
    totals = dict.fromkeys(COMPONENTS, 0)
    cleared, history = set(), []
    accepted = measurements = failures = 0
    for index, step in enumerate(steps):
        require(step["step_index"] == index, "Step index gap")
        require(type(step["accepted"]) is int and step["accepted"] in (0, 1), "Invalid acceptance flag")
        kind = step["action"]
        require(kind in ("enter", "measure", "clear", "exit"), "Unknown action")
        action, response = load_json(step["action_json"]), load_json(step["response_json"])
        require(type(response.get("accepted")) is bool and response["accepted"] == bool(step["accepted"]),
                "Accepted marker mismatch")
        close(step["virtual_time_before_s"], elapsed / 1e6, "Before time")
        parts = dict.fromkeys(COMPONENTS, 0)
        if step["accepted"]:
            accepted += 1
            if kind == "enter":
                require(index == 0 and not active, "Unexpected enter")
                active = True
            else:
                require(active, "Action outside active session")
                if kind == "exit":
                    require(index == len(steps)-1, "Exit must end the trajectory")
                    active = False
                else:
                    point, channel = position(action["position"]), action["channel"]
                    require(type(channel) is int and 1 <= channel <= 20, "Invalid channel")
                    require(channel == step["channel"] and point == position((step["x_m"], step["y_m"])),
                            "Stored action columns mismatch")
                    parts["movement_s"] = round(math.dist(current, point) / 5 * 1e6)
                    current = point
                    item = {"action": kind, "position": list(point), "channel": channel}
                    if kind == "measure":
                        measurements += 1
                        result = response.get("measure_result")
                        require(result in ("direction", "near", "no_signal"), "Invalid measure feedback")
                        require(channel not in cleared or result == "no_signal", "Signal after source removal")
                        parts["detection_s"] = 5_000_000
                        parts["switching_s"] = int(tuned != channel) * 1_000_000
                        tuned = channel
                        if result == "direction":
                            bearing = response.get("svd_deg")
                            require(type(bearing) in (int, float) and math.isfinite(bearing) and 0 <= bearing < 360,
                                    "Invalid bearing")
                            item["bearing_deg"] = bearing
                    else:
                        result = response.get("clear_result")
                        require(result in ("success", "no_target_in_range"), "Invalid clear feedback")
                        success = result == "success"
                        require(not success or channel not in cleared, "Duplicate successful clear")
                        if success:
                            cleared.add(channel)
                        else:
                            failures += 1
                        parts["optical_s"], parts["removal_s"] = 3_000_000, 2_000_000 * success
                    item["result"] = result
                    history.append(item)
        delta = sum(parts.values())
        elapsed += delta
        for key in COMPONENTS:
            totals[key] += parts[key]
        close(step["delta_virtual_time_s"], delta / 1e6, "Action cost")
        close(step["reward_time_component"], -delta / 1e6, "Time reward")
        close(step["virtual_time_after_s"], elapsed / 1e6, "After time")
        close(response["virtual_time_s"], elapsed / 1e6, "Response time")
        recorded = load_json(step["cost_components_json"])
        require(set(recorded).issubset(COMPONENTS), "Unknown cost component")
        for key in COMPONENTS:
            close(recorded.get(key, 0), parts[key] / 1e6, key)
        require(bool(step["clear_success"]) == (step["accepted"] and kind == "clear"
                and response.get("clear_result") == "success"), "Clear-success marker mismatch")
    require(bool(steps) and steps[0]["action"] == "enter" and steps[0]["accepted"] == 1
            and steps[-1]["action"] == "exit" and steps[-1]["accepted"] == 1 and not active,
            "Missing accepted enter/exit")
    require(summary.get("pending_request") is None and summary["state"]["session"] == "exited",
            "Summary has not exited")
    require(summary["state"]["cleared_count"] == len(cleared), "Summary cleared count mismatch")
    close(summary["state"]["virtual_time_s"], elapsed / 1e6, "Summary total")
    for key in COMPONENTS:
        close(summary["state"]["time_breakdown"].get(key, 0), totals[key] / 1e6, "Summary " + key)
    return {"passed": True, "virtual_time_s": elapsed / 1e6, "accepted_steps": accepted,
            "measurements": measurements, "failed_clear_count": failures, "cleared_count": len(cleared),
            "time_breakdown_s": {key: value / 1e6 for key, value in totals.items()},
            "action_history": history}


def measurement_support(steps, candidates):
    """Existence of exact recorded *pre-clear* feedback, not policy performance."""
    first_clear, observed, conflicts = {}, {}, set()
    for step in steps:
        if step["accepted"] and step["clear_success"]:
            first_clear.setdefault(step["channel"], step["step_index"])
    for step in steps:
        if (not step["accepted"] or step["action"] != "measure"
                or step["step_index"] >= first_clear.get(step["channel"], math.inf)):
            continue
        key = (step["channel"], *position((step["x_m"], step["y_m"])))
        response = load_json(step["response_json"])
        value = (response["measure_result"], response.get("svd_deg") if response["measure_result"] == "direction" else None)
        if key in observed and observed[key] != value:
            conflicts.add(key)
        observed[key] = value
    reports = []
    for candidate in candidates:
        require(isinstance(candidate.get("name"), str) and candidate["name"], "Candidate needs a name")
        require(isinstance(candidate.get("stations"), list) and candidate["stations"], "Candidate needs stations")
        requested = set()
        for station in candidate["stations"]:
            point = position(station["position"])
            channels = station.get("channels", list(range(1, 21)))
            require(isinstance(channels, list) and channels and all(type(c) is int and 1 <= c <= 20 for c in channels),
                    "Invalid station channels")
            requested.update((channel, *point) for channel in channels)
        unsupported = sorted(requested - (observed.keys() - conflicts))
        reports.append({"name": candidate["name"], "requested_station_channels": len(requested),
            "supported_station_channels": len(requested)-len(unsupported),
            "unsupported_station_channels": len(unsupported),
            "status": "unsupported" if unsupported else "station_queries_supported_only",
            "unsupported_examples": [{"channel": q[0], "position": list(q[1:]),
                "reason": "conflicting_recorded_feedback" if q in conflicts else "no_exact_preclear_measurement"}
                for q in unsupported[:5]], "exact_counterfactual_time_s": None,
            "reason": "Station support does not cover adaptive probes, clear positions or alternative histories"})
    return reports


def audit_snapshot(snapshot, freeze, output, candidates_path=None, expected_holdout=6):
    snapshot, output = private_path(snapshot), private_path(output)
    manifest_path = snapshot / "snapshot.json"
    manifest = load_json(manifest_path.read_bytes())
    require(manifest["version"] == VERSION and manifest["database"] == "practice_training.sqlite3", "Wrong snapshot")
    database = snapshot / manifest["database"]
    require(sha(database) == manifest["database_sha256"], "Snapshot database changed")
    freeze = Path(freeze).resolve(strict=True)
    freeze_bytes = freeze.read_bytes()
    require(isinstance(load_json(freeze_bytes), dict) and bool(load_json(freeze_bytes)), "Empty algorithm-freeze artifact")
    candidates_bytes = Path(candidates_path).read_bytes() if candidates_path else None
    candidates = load_json(candidates_bytes)["candidates"] if candidates_bytes is not None else []
    require(isinstance(candidates, list), "Candidate JSON needs a candidates list")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    with closing(connect_readonly(database)) as reader:
        reader.execute("BEGIN")  # Locks only the private snapshot, never the live collection.
        episodes = [dict(row) for row in reader.execute("SELECT * FROM episodes WHERE " + HELDOUT_SQL + " ORDER BY id")]
        require(len(episodes) == expected_holdout > 0, "Holdout count differs; do not silently change the selected cohort")
        identities = [{key: e[key] for key in ("id", "case_code", "split", "policy", "evidence_sha256")} for e in episodes]
        require(identities == manifest["selected_completed_q4_holdout"], "Snapshot selected cohort mismatch")
        for episode in episodes:
            require(episode["source_total"] == episode["cleared_count"] and 10 <= episode["source_total"] <= 16,
                    "Heldout episode lacks all-clear evidence")
            evidence = {}
            evidence_hashes = {}
            for ev in reader.execute("SELECT * FROM evidence WHERE episode_id=? AND name IN ('summary.json','official-result.json','after.json')", (episode["id"],)):
                raw = zlib.decompress(ev["content_zlib"])
                require(hashlib.sha256(raw).hexdigest() == ev["sanitized_sha256"], "Stored evidence hash mismatch")
                evidence[ev["name"]] = load_json(raw)
                evidence_hashes[ev["name"]] = ev["sanitized_sha256"]
            summary, official, after = (evidence[n] for n in ("summary.json", "official-result.json", "after.json"))
            require(summary.get("data_origin") == "simulator_http_session" and summary.get("declared_mode") == "practice"
                    and summary.get("problem") == official.get("problem_no") == after.get("problem_no") == 4,
                    "Not a Q4 practice")
            require(summary.get("case_code") == official.get("case_code") == after.get("case_code") == episode["case_code"],
                    "Case identity mismatch")
            require(after.get("phase") == "ended" and after.get("api_open") is False
                    and after.get("cleanup_complete") is True and after.get("log_package_status") == "ready",
                    "Missing terminal lifecycle evidence")
            require(official["jammer_count"] == episode["source_total"] and summary.get("completed") is True,
                    "Official total or completion differs")
            require(summary.get("variant") == episode["policy"], "Collection policy differs")
            steps = [dict(row) for row in reader.execute("SELECT * FROM steps WHERE episode_id=? ORDER BY step_index", (episode["id"],))]
            require(len(steps) == episode["step_count"], "Stored step count differs")
            ledger = audit_steps(steps, summary)
            close(episode["virtual_time_s"], ledger["virtual_time_s"], "Episode total")
            require(episode["accepted_step_count"] == ledger["accepted_steps"], "Accepted step count differs")
            reconstructed = {"data_origin": "simulator_http_session", "problem": 4, "pending_request": None,
                "state": summary["state"], "search": {"action_history": ledger.pop("action_history"),
                "time_breakdown": ledger["time_breakdown_s"]}}
            # Reuse the unchanged historical bound implementation on a minimal,
            # independently reconstructed evidence summary. No source-label centers.
            with tempfile.TemporaryDirectory(prefix="bounds-", dir=output) as scratch:
                summary_path = Path(scratch) / "summary.json"
                write_json(summary_path, reconstructed)
                bounds = analyze_bounds(summary_path)
            lower = bounds["conditional_guaranteed_all_clear_lower_s"]
            require(lower > 0 and lower <= ledger["virtual_time_s"] + 2e-6, "Invalid historical lower bound")
            rows.append({"case_code": episode["case_code"], "split": episode["split"], "collection_policy": episode["policy"],
                "source_total": episode["source_total"], "evidence_sha256": evidence_hashes, "ledger": ledger,
                "legacy_lower_bound_field": "conditional_guaranteed_all_clear_lower_s",
                "legacy_lower_bound_s": lower, "legacy_time_over_lower_bound": ledger["virtual_time_s"] / lower,
                "physical_oracle_lower_bound_s": bounds["observation_lower_bound_s"],
                "physical_oracle_time_over_lower_bound": ledger["virtual_time_s"] / bounds["observation_lower_bound_s"],
                "bounds": bounds, "candidate_station_support": measurement_support(steps, candidates)})
    require(sha(database) == manifest["database_sha256"], "Snapshot changed during audit")
    result = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha256": manifest["database_sha256"], "freeze_sha256": hashlib.sha256(freeze_bytes).hexdigest(),
        "candidate_stations_sha256": hashlib.sha256(candidates_bytes).hexdigest() if candidates_bytes is not None else None,
        "audit_source_sha256": sha(__file__), "bound_source_sha256": sha(ROOT / "experiments/session_lower_bounds.py"),
        "episodes": len(rows), "passed": True, "rows": rows,
        "mean_recorded_time_s": sum(row["ledger"]["virtual_time_s"] for row in rows) / len(rows),
        "mean_per_case_legacy_time_over_lower_bound": sum(row["legacy_time_over_lower_bound"] for row in rows) / len(rows),
        "ratio_of_sum_recorded_time_to_sum_legacy_lower_bound": sum(row["ledger"]["virtual_time_s"] for row in rows) / sum(row["legacy_lower_bound_s"] for row in rows),
        "notes": ["This audits old recorded trajectories, not new-policy performance.",
            "The collection policy includes bounded_refinement; every such action is charged, not subtracted.",
            "Primary ratio uses conditional_guaranteed_all_clear_lower_s; physical oracle ratio is separate.",
            "The lower bound requires all sources cleared and a policy obliged to guarantee completion across allowed scenes.",
            "Unknown position, source type, radius, orientation and unmeasured bearing error are not fitted.",
            "Exact station support alone does not license full adaptive counterfactual simulation.",
            "These holdout trajectories may only be opened after algorithm selection is frozen; no tuning on this report."]}
    write_json(output / "audit.json", result)
    return result


def self_test():
    """Artificial seven-action trajectory; no collection DB, scene or network."""
    steps, elapsed = [], 0.
    def add(action, channel=None, point=None, result=None, bearing=None, **parts):
        nonlocal elapsed
        before = elapsed
        elapsed += sum(parts.values())
        payload, response = {}, {"accepted": True, "virtual_time_s": elapsed}
        if point is not None:
            payload = {"position": dict(zip(("x", "y"), point)), "channel": channel}
            response["measure_result" if action == "measure" else "clear_result"] = result
            if bearing is not None:
                response["svd_deg"] = bearing
        steps.append({"step_index": len(steps), "action": action, "channel": channel,
            "x_m": point[0] if point else None, "y_m": point[1] if point else None,
            "accepted": 1, "clear_success": int(action == "clear" and result == "success"),
            "action_json": json.dumps(payload), "response_json": json.dumps(response),
            "cost_components_json": json.dumps(parts), "virtual_time_before_s": before,
            "virtual_time_after_s": elapsed, "delta_virtual_time_s": elapsed-before,
            "reward_time_component": before-elapsed})
    add("enter")
    add("measure", 2, (0, 0), "no_signal", switching_s=1., detection_s=5.)
    add("clear", 7, (15, 0), "no_target_in_range", movement_s=3., optical_s=3.)
    add("measure", 2, (15, 0), "direction", bearing=90., detection_s=5.)
    add("clear", 2, (15, 10), "success", movement_s=2., optical_s=3., removal_s=2.)
    add("measure", 2, (15, 10), "no_signal", detection_s=5.)
    add("exit")
    summary = {"pending_request": None, "state": {"session": "exited", "cleared_count": 1,
        "virtual_time_s": 29., "time_breakdown": {"movement_s": 5., "switching_s": 1.,
            "detection_s": 15., "optical_s": 6., "removal_s": 2.}}}
    ledger = audit_steps(steps, summary)
    assert ledger["virtual_time_s"] == 29. and ledger["failed_clear_count"] == 1
    assert ledger["time_breakdown_s"]["switching_s"] == 1.  # A clear never retunes.
    candidates = [{"name": "known", "stations": [{"position": [15, 0], "channels": [2]}]},
                  {"name": "new", "stations": [{"position": [15+1e-9, 0], "channels": [2]}]},
                  {"name": "after-clear", "stations": [{"position": [15, 10], "channels": [2]}]}]
    support = measurement_support(steps, candidates)
    assert [r["unsupported_station_channels"] for r in support] == [0, 1, 1]
    assert all(r["exact_counterfactual_time_s"] is None for r in support)
    corrupt = [dict(s) for s in steps]
    corrupt[2]["delta_virtual_time_s"] -= 3.  # Omitting a failed optical charge must fail.
    try:
        audit_steps(corrupt, summary)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing optical cost was not rejected")
    duplicate = dict(steps[3], response_json=json.dumps({"measure_result": "direction", "svd_deg": 91.}))
    assert measurement_support([steps[3], duplicate], candidates[:1])[0]["status"] == "unsupported"
    return {"passed": True, "source": "artificial seven-action example only",
            "checks": ["exact total", "clear does not retune", "failed optical fully charged",
                "cost corruption rejected", "recorded point supported", "nearby point unsupported",
                "post-clear silence excluded", "conflicting repeated feedback excluded", "no counterfactual runtime"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    snap = sub.add_parser("snapshot")
    snap.add_argument("--database", type=Path, required=True)
    snap.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--snapshot", type=Path, required=True)
    audit.add_argument("--freeze-json", type=Path, required=True)
    audit.add_argument("--candidate-stations", type=Path)
    audit.add_argument("--expected-holdout", type=int, default=6)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "self-test":
        result = self_test()
    elif args.command == "snapshot":
        result = snapshot_database(args.database, args.output)
    else:
        result = audit_snapshot(args.snapshot, args.freeze_json, args.output, args.candidate_stations, args.expected_holdout)
        result = {key: value for key, value in result.items() if key not in ("rows", "notes")}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
