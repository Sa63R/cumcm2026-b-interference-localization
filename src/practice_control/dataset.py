"""One-file, observation-only dataset of *finished* Q3/Q4 practice episodes.

This module is entirely offline: it reads saved evidence, never a simulator or
scenario store. Final source estimates are bounded labels, not ground truth and
not policy observations. Use ``iter_transitions`` to decode causal transitions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import zlib

from geometry import clip_polygon, disk_halfplanes, minimum_enclosing_circle
from localization import CandidateRegion
from simulator_client.rules import CLEAR_RADIUS_M, NEAR_RADIUS_M


SCHEMA_VERSION = 1
PROTOCOL_VERSION = "cumcm2026b-practice-observation-v1"
NUMERICAL_MARGIN_M = 1e-6
_PATHS = {"/enter", "/measure", "/clear", "/exit"}
_CASE = re.compile(r"[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}\Z")
_PRIVATE = {"robot_id", "team_id", "password", "token", "access_token", "refresh_token",
            "authorization", "cookie", "run_id", "practice_run_no", "real_deadline",
            "python_executable", "source_root", "controller_root", "base_url"}
_FILES = ("summary.json", "created.json", "before.json", "after.json", "requests.jsonl")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _load(raw):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(raw, parse_constant=invalid)


def _number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid {name}")
    return float(value)


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}")
    return value


def _position(value):
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        raise ValueError("Action position must contain x and y")
    return tuple(_number(value[c], "position", -2_000_000) for c in ("x", "y"))


def split_for_case(case):
    """Keep a whole case in one split, including when imported into another DB."""
    bucket = int(_hash((PROTOCOL_VERSION + ":" + case).encode())[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _sanitizer(documents, request_numbers):
    secrets = set()
    def scan(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in _PRIVATE and isinstance(item, str) and len(item) >= 4:
                    secrets.add(item)
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)
    for document in documents:
        scan(document)

    def sanitize(value):
        if isinstance(value, dict):
            return {key: (f"request-{request_numbers.get(item, 0):06d}" if key == "request_id"
                          else sanitize(item)) for key, item in value.items()
                    if key.lower() not in _PRIVATE and not key.lower().endswith("_path")}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, str):
            for secret in sorted(secrets, key=len, reverse=True):
                value = value.replace(secret, "[redacted]")
        return value
    return sanitize


def _registered_evidence(episode, raw_summary):
    registration = _load((episode / "registration.json").read_bytes())
    reference = Path(registration.get("record", ""))
    if not reference.name.startswith("practice-") or reference.suffix != ".json":
        raise ValueError("Expected a registered practice evidence record")
    # Portable datasets often move with their sibling registered directory.
    record_path = episode.parent / "registered" / reference.name
    if not record_path.is_file():
        record_path = reference
    if record_path.parent.name != "registered" or not record_path.is_file():
        raise ValueError("Cannot locate the registered practice evidence")
    raw_record = record_path.read_bytes()
    record = _load(raw_record)
    if (record.get("data_origin") != "registered_official_practice"
            or record.get("source_total_source") != "official_simulator_result_file"
            or record.get("official_result_session_time_matched") is not True
            or record.get("summary_sha256") != _hash(raw_summary)):
        raise ValueError("Practice registration or summary hash is invalid")
    official_ref = Path(record.get("official_result_path", ""))
    if (official_ref.is_absolute() or len(official_ref.parts) != 2
            or official_ref.parts[0] != "evidence" or not official_ref.name.endswith(".result.json")):
        raise ValueError("Expected the archived finished-practice result")
    original_name = record.get("official_result_original_name", "")
    if not original_name.startswith(f"practice-p{record.get('problem')}-"):
        raise ValueError("Result evidence is not a practice result")
    raw_official = (record_path.parent / official_ref).read_bytes()
    if _hash(raw_official) != record.get("official_result_sha256"):
        raise ValueError("Archived practice result hash mismatch")
    time_evidence = record.get("session_time_evidence", {})
    if time_evidence.get("basis") == "simulator_http_enter":
        journal_ref = Path(record.get("session_time_journal_path", ""))
        if (journal_ref.is_absolute() or len(journal_ref.parts) != 2
                or journal_ref.parts[0] != "evidence" or not journal_ref.name.endswith(".requests.jsonl")):
            raise ValueError("Expected archived request journal for simulator-clock time evidence")
        if _hash((record_path.parent / journal_ref).read_bytes()) != time_evidence.get("journal_sha256"):
            raise ValueError("Archived time-evidence request journal hash mismatch")
    return registration, record, raw_official, raw_record


def _validate_episode(summary, created, before, after, registration, record, official, *, raw_journal=None):
    case, problem = summary.get("case_code"), summary.get("problem")
    if (summary.get("declared_mode") != "practice" or type(problem) is not int or problem not in (3, 4)
            or not isinstance(case, str) or not _CASE.fullmatch(case)
            or summary.get("data_origin") != "simulator_http_session"
            or summary.get("gui_mode_verified_by_api") is not True):
        raise ValueError("Only verified simulator Q3/Q4 practice episodes are importable")
    if (created.get("mode") != "practice" or created.get("problem") != problem
            or created.get("case_code") != case):
        raise ValueError("Created practice case mismatch")
    for state in (before, after):
        if state.get("mode") != "practice" or state.get("problem_no") != problem or state.get("case_code") != case:
            raise ValueError("Practice lifecycle case mismatch")
    if before.get("phase") != "waiting_enter" or before.get("entered") is not False:
        raise ValueError("Practice entry ownership was not confirmed")
    if (after.get("phase") != "ended" or after.get("cleanup_complete") is not True
            or after.get("log_package_status") != "ready" or after.get("api_open") is not False
            or summary.get("state", {}).get("session") != "exited"
            or summary.get("pending_request") is not None):
        raise ValueError("Practice must have a confirmed exit and saved result")
    for evidence in (registration, record, official):
        if evidence.get("case_code") != case:
            raise ValueError("Registered practice case mismatch")
    if record.get("problem") != problem or official.get("problem_no") != problem:
        raise ValueError("Registered practice problem mismatch")
    total = _integer(official.get("jammer_count"), "source total", 10, 16)
    omni = _integer(official.get("omnidirectional_jammer_count"), "omni total", 0, total)
    directional = _integer(official.get("directional_jammer_count"), "directional total", 0, total)
    # Zero omnidirectional sources is valid in Q4; do not require a mixed case.
    if (omni + directional != total or (problem == 3 and directional != 0)
            or total != record.get("source_total") or total != registration.get("source_total")):
        raise ValueError("Registered practice source totals disagree")
    cleared = _integer(summary["state"].get("cleared_count"), "cleared count", 0, total)
    if cleared != record.get("cleared_count"):
        raise ValueError("Registered clearance count disagrees")
    from .evidence_time import match_session_time
    time_evidence = match_session_time(summary, official, raw_journal=raw_journal)
    recorded_time_evidence = record.get("session_time_evidence")
    if (recorded_time_evidence is not None or time_evidence["basis"] != "local_summary"):
        if recorded_time_evidence != time_evidence:
            raise ValueError("Registered session time evidence does not match the original journal")
    _number(summary["state"].get("virtual_time_s"), "virtual time")
    _number(summary.get("program_wall_time_s"), "program wall time")
    return case, problem, total, cleared


def _journal(records, summary):
    """Deduplicate retried request IDs; preserve rejected actions as non-trainable."""
    steps, observations, requests = [], [], {}
    current_request = None
    previous = None
    accepted_count = 0
    cleared = set()

    def finish():
        nonlocal current_request, previous, accepted_count
        if current_request is None:
            return
        action = requests[current_request]
        response, state = action.get("response"), action.get("state")
        if not isinstance(response, dict) or type(response.get("accepted")) is not bool:
            raise ValueError("Journal contains an unresolved request")
        accepted = response["accepted"]
        if accepted and (state is None or action.get("http_status") != 200):
            raise ValueError("Accepted request lacks a committed state snapshot")
        if not accepted and state is not None:
            raise ValueError("Rejected request unexpectedly changed state")
        path, payload = action["path"], action["payload"]
        pre_index = len(observations) - 1 if previous is not None else None
        before_time = previous["virtual_time_s"] if previous else 0.0
        after_time = _number(response.get("virtual_time_s"), "response virtual time")
        if after_time + 1e-5 < before_time:
            raise ValueError("Journal virtual time moves backwards")
        costs = {}
        if accepted:
            accepted_count += 1
            if state.get("accepted_actions") != accepted_count:
                raise ValueError("Accepted state count has gaps or duplicate actions")
            if not math.isclose(_number(state.get("virtual_time_s"), "state virtual time"), after_time, abs_tol=1e-5):
                raise ValueError("Response and committed state times disagree")
            old_costs = previous.get("time_breakdown", {}) if previous else {}
            costs = {key: _number(value, "time breakdown") - old_costs.get(key, 0)
                     for key, value in state.get("time_breakdown", {}).items()}
            if any(value < -1e-5 for value in costs.values()):
                raise ValueError("Journal cost components move backwards")
            if not math.isclose(sum(costs.values()), after_time - before_time, abs_tol=2e-5):
                raise ValueError("Time cost components disagree with virtual time")
            if path == "/enter":
                if previous is not None or state.get("session") != "active":
                    raise ValueError("Journal does not begin with one accepted enter")
            elif previous is None or previous.get("session") != "active":
                raise ValueError("Action outside an entered practice session")
            if path in {"/measure", "/clear"}:
                _position(payload.get("position"))
                _integer(payload.get("channel"), "channel", 1, 20)
                if state.get("position") != payload["position"]:
                    raise ValueError("Action and post-state positions disagree")
            if path == "/measure" and response.get("measure_result") not in {"direction", "near", "no_signal"}:
                raise ValueError("Unknown measurement result")
            if path == "/clear" and response.get("clear_result") not in {"success", "no_target_in_range"}:
                raise ValueError("Unknown clearance result")
            if path == "/clear" and response.get("clear_result") == "success":
                if payload["channel"] in cleared:
                    raise ValueError("The same source was cleared twice")
                cleared.add(payload["channel"])
            if state.get("cleared_count") != len(cleared):
                raise ValueError("Committed clear count disagrees with successful actions")
            observations.append(state)
            previous = state
        elif not math.isclose(after_time, before_time, abs_tol=1e-5):
            raise ValueError("Rejected request changed virtual time")
        steps.append({**action, "index": len(steps), "accepted": accepted,
                      "before_index": pre_index, "after_index": len(observations) - 1 if previous else None,
                      "virtual_before": before_time, "virtual_after": after_time,
                      "delta_virtual": after_time - before_time if accepted else 0.0,
                      "costs": costs,
                      "clear_success": accepted and path == "/clear" and response.get("clear_result") == "success",
                      "terminal": accepted and path == "/exit"})
        current_request = None

    for record in records:
        event = record.get("event")
        if event == "request":
            payload = record.get("payload", {})
            request_id = payload.get("request_id")
            if not isinstance(request_id, str) or record.get("path") not in _PATHS:
                raise ValueError("Journal contains an unsupported request")
            if request_id != current_request:
                finish()
                if request_id in requests:
                    raise ValueError("Nonconsecutive replay would duplicate a transition")
                current_request = request_id
                requests[request_id] = {"path": record["path"], "payload": payload,
                                        "recorded_at": record.get("recorded_at"), "attempts": 0}
            action = requests[request_id]
            if action["payload"] != payload or action["path"] != record["path"]:
                raise ValueError("Retry changed an idempotent action")
            action["attempts"] += 1
        elif event in {"response", "state"}:
            if current_request is None or record.get("request_id") != current_request:
                raise ValueError("Journal response/state is out of order")
            action = requests[current_request]
            if event == "response":
                action["response"] = record.get("response")
                action["http_status"] = record.get("http_status")
            else:
                if "state" in action:
                    raise ValueError("Duplicate committed state for one request")
                action["state"] = record.get("state")
        elif event not in {"session_created", "client_closed", "attempt_failed", "outcome_unknown"}:
            raise ValueError("Unrecognized request journal event")
    finish()
    if (not steps or steps[0]["path"] != "/enter" or not steps[0]["accepted"]
            or steps[-1]["path"] != "/exit" or not steps[-1]["accepted"]
            or previous is None or previous.get("session") != "exited"):
        raise ValueError("Journal does not contain an accepted enter-to-exit trajectory")
    if previous != summary.get("state"):
        raise ValueError("Final journal state does not match summary")
    numbers = {key: number for number, key in enumerate(requests, 1)}
    return steps, observations, numbers


def _source_labels(steps):
    """Intersect only positive observation constraints; no hidden truth input."""
    regions, constraints, disks, clear_points = {}, {}, {}, {}
    for step in steps:
        if not step["accepted"] or step["path"] not in {"/measure", "/clear"}:
            continue
        payload, response = step["payload"], step["response"]
        channel, position = payload["channel"], _position(payload["position"])
        kind = response.get("measure_result") if step["path"] == "/measure" else response.get("clear_result")
        if kind not in {"direction", "near", "success"}:
            continue  # no_signal/failed clears stay in the trajectory, never imply absence in Q4.
        region = regions.setdefault(channel, CandidateRegion())
        item = {"step_index": step["index"], "kind": kind, "position": list(position)}
        if kind == "direction":
            bearing = _number(response.get("svd_deg"), "bearing")
            if bearing >= 360:
                raise ValueError("Bearing must be in [0,360)")
            region.observe(position, bearing)
            item.update(bearing_deg=bearing, error_deg=region.error_deg,
                        reception_radius_m=region.reception_radius)
        else:
            radius = NEAR_RADIUS_M if kind == "near" else CLEAR_RADIUS_M
            item["radius_m"] = radius
            disks.setdefault(channel, []).append((radius, position, kind))
            if kind == "success":
                clear_points[channel] = (position, step["index"])
            for plane in disk_halfplanes(position, radius, 128, outer=True):
                region.vertices = clip_polygon(region.vertices, plane)
                if not region.vertices:
                    break
        constraints.setdefault(channel, []).append(item)
    labels = []
    for channel, region in sorted(regions.items()):
        inconsistent = not bool(region.vertices)
        choices = []
        if region.vertices:
            circle = minimum_enclosing_circle(region.vertices)
            choices.append((circle.radius + NUMERICAL_MARGIN_M, circle.center, "observation_intersection"))
        # Each exact disk independently encloses the source. An inconsistent
        # fusion is flagged and uses only a separately observed disk as fallback.
        choices += [(radius, position, f"{kind}_disk") for radius, position, kind in disks.get(channel, [])]
        selected = min(choices, key=lambda value: value[0]) if choices else (None, (None, None), "inconsistent")
        clear = clear_points.get(channel)
        labels.append({"channel": channel, "x_m": selected[1][0], "y_m": selected[1][1],
                       "radius_m": selected[0], "method": selected[2], "is_exact_truth": False,
                       "cleared": clear is not None, "clear_position": list(clear[0]) if clear else None,
                       "clear_step_index": clear[1] if clear else None,
                       "clear_radius_m": CLEAR_RADIUS_M if clear else None,
                       "geometry_consistent": not inconsistent, "quality_flags": ["inconsistent_geometry"] if inconsistent else [],
                       "numerical_tolerance_m": NUMERICAL_MARGIN_M,
                       "prior_radius_m": region.prior_radius, "disk_polygon_sides": region.disk_sides,
                       "constraints": constraints[channel], "outer_vertices": region.vertices,
                       "target_kind": "bounded_source_position_estimate", "source_type": "unknown"})
    return labels


_SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY, case_code TEXT NOT NULL UNIQUE, problem INTEGER NOT NULL CHECK(problem IN (3,4)),
  split TEXT NOT NULL, policy TEXT NOT NULL, code_revision TEXT, started_at TEXT NOT NULL,
  imported_at TEXT NOT NULL, evidence_sha256 TEXT NOT NULL, complete INTEGER NOT NULL,
  trajectory_complete INTEGER NOT NULL, quality_flags_json TEXT NOT NULL,
  source_total INTEGER NOT NULL, cleared_count INTEGER NOT NULL, virtual_time_s REAL NOT NULL,
  program_wall_time_s REAL NOT NULL, step_count INTEGER NOT NULL, accepted_step_count INTEGER NOT NULL,
  source_estimate_count INTEGER NOT NULL, provenance_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS observations (
  episode_id INTEGER NOT NULL REFERENCES episodes(id), observation_index INTEGER NOT NULL,
  state_zlib BLOB NOT NULL, PRIMARY KEY(episode_id,observation_index));
CREATE TABLE IF NOT EXISTS steps (
  episode_id INTEGER NOT NULL REFERENCES episodes(id), step_index INTEGER NOT NULL,
  action TEXT NOT NULL, channel INTEGER, x_m REAL, y_m REAL, accepted INTEGER NOT NULL,
  attempts INTEGER NOT NULL, observation_before_index INTEGER, observation_after_index INTEGER,
  virtual_time_before_s REAL NOT NULL, virtual_time_after_s REAL NOT NULL, delta_virtual_time_s REAL NOT NULL,
  reward_time_component REAL NOT NULL, clear_success INTEGER NOT NULL, terminal INTEGER NOT NULL,
  trainable INTEGER NOT NULL, action_json TEXT NOT NULL, response_json TEXT NOT NULL,
  cost_components_json TEXT NOT NULL, recorded_at TEXT, PRIMARY KEY(episode_id,step_index));
CREATE TABLE IF NOT EXISTS source_estimates (
  episode_id INTEGER NOT NULL REFERENCES episodes(id), channel INTEGER NOT NULL,
  x_m REAL, y_m REAL, radius_m REAL, method TEXT NOT NULL, cleared INTEGER NOT NULL,
  geometry_consistent INTEGER NOT NULL, clear_x_m REAL, clear_y_m REAL, clear_radius_m REAL,
  clear_step_index INTEGER, label_json TEXT NOT NULL, PRIMARY KEY(episode_id,channel));
CREATE TABLE IF NOT EXISTS evidence (
  episode_id INTEGER NOT NULL REFERENCES episodes(id), name TEXT NOT NULL,
  original_sha256 TEXT NOT NULL, sanitized_sha256 TEXT NOT NULL,
  format TEXT NOT NULL, content_zlib BLOB NOT NULL, PRIMARY KEY(episode_id,name));
CREATE INDEX IF NOT EXISTS episodes_problem_split ON episodes(problem,split);
CREATE INDEX IF NOT EXISTS steps_action ON steps(action);
"""


def _connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=DELETE")  # One portable file after every commit.
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(_SCHEMA)
    existing = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if existing and existing[0] != str(SCHEMA_VERSION):
        connection.close()
        raise ValueError("Unsupported practice dataset schema version")
    metadata = {"schema_version": str(SCHEMA_VERSION), "protocol_version": PROTOCOL_VERSION,
                "compression": "zlib UTF-8 JSON; use iter_transitions/decode_blob",
                "label_semantics": "bounded estimates from accepted observations; no exact hidden truth",
                "reward_semantics": "reward_time_component=-delta_virtual_time_s; clear_success separate; no combined reward",
                "split_semantics": "SHA256 case split 80/10/10; episodes never split across sets",
                "observation_semantics": "causal client belief/state snapshots; final labels never in observations"}
    with connection:
        connection.executemany("INSERT OR IGNORE INTO metadata(key,value) VALUES (?,?)", metadata.items())
    return connection


def import_episode(db_path, episode_dir):
    """Atomically import one finished practice; unchanged re-import is a no-op."""
    episode = Path(episode_dir)
    raw = {name: (episode / name).read_bytes() for name in _FILES}
    documents = {name: _load(value) for name, value in raw.items() if name != "requests.jsonl"}
    records = [_load(line) for line in raw["requests.jsonl"].splitlines() if line.strip()]
    summary = documents["summary.json"]
    # Check declared scope before following any registration reference.
    if summary.get("declared_mode") != "practice":
        raise ValueError("Only practice evidence may be imported")
    registration, record, raw_official, raw_record = _registered_evidence(episode, raw["summary.json"])
    official = _load(raw_official)
    case, problem, total, cleared = _validate_episode(
        summary, documents["created.json"], documents["before.json"], documents["after.json"],
        registration, record, official, raw_journal=raw["requests.jsonl"])
    steps, observations, request_numbers = _journal(records, summary)
    labels = _source_labels(steps)
    flags = []
    if summary.get("completed") is not True:
        flags.append("search_incomplete")
    if cleared != total:
        flags.append("not_all_sources_cleared")
    if any(not label["geometry_consistent"] for label in labels):
        flags.append("inconsistent_geometry")
    if any(not step["accepted"] for step in steps):
        flags.append("rejected_requests")
    if any(step["attempts"] > 1 for step in steps):
        flags.append("retried_requests")
    complete = summary.get("completed") is True and cleared == total
    all_documents = list(documents.values()) + records + [registration, record, official]
    sanitize = _sanitizer(all_documents, request_numbers)
    identity = _hash(_json({**{name: _hash(value) for name, value in raw.items()},
                            "official_result.json": _hash(raw_official)}).encode())
    result = {"inserted": False, "case_code": case, "problem": problem, "steps": len(steps),
              "source_estimates": len(labels), "quality_flags": flags, "complete": complete}
    connection = _connect(db_path)
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            old = connection.execute("SELECT evidence_sha256 FROM episodes WHERE case_code=?", (case,)).fetchone()
            if old:
                if old[0] != identity:
                    raise ValueError("Case already exists with conflicting evidence")
                return result
            provenance = sanitize({key: summary.get(key) for key in (
                "data_origin", "declared_mode", "mode_verification", "working_tree_dirty",
                "method_metadata", "error")})
            cursor = connection.execute("""INSERT INTO episodes
                (case_code,problem,split,policy,code_revision,started_at,imported_at,evidence_sha256,
                 complete,trajectory_complete,quality_flags_json,source_total,cleared_count,virtual_time_s,
                 program_wall_time_s,step_count,accepted_step_count,source_estimate_count,provenance_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (case, problem, split_for_case(case), summary.get("variant", "unknown"), summary.get("code_revision"),
                 summary["started_at"], datetime.now(timezone.utc).isoformat(), identity, int(complete), 1,
                 _json(flags), total, cleared, summary["state"]["virtual_time_s"], summary["program_wall_time_s"],
                 len(steps), sum(step["accepted"] for step in steps), len(labels), _json(provenance)))
            episode_id = cursor.lastrowid
            connection.executemany("INSERT INTO observations VALUES (?,?,?)",
                                   [(episode_id, index, zlib.compress(_json(sanitize(state)).encode(), 6))
                                    for index, state in enumerate(observations)])
            for step in steps:
                payload = step["payload"]
                position = payload.get("position", {})
                connection.execute("INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (episode_id, step["index"], step["path"].lstrip("/"), payload.get("channel"),
                     position.get("x"), position.get("y"), int(step["accepted"]), step["attempts"],
                     step["before_index"], step["after_index"], step["virtual_before"], step["virtual_after"],
                     step["delta_virtual"], -step["delta_virtual"], int(step["clear_success"]), int(step["terminal"]),
                     int(step["accepted"] and step["path"] in {"/measure", "/clear", "/exit"}),
                     _json(sanitize(payload)), _json(sanitize(step["response"])), _json(step["costs"]),
                     step.get("recorded_at")))
            for label in labels:
                if problem == 3:
                    label["source_type"] = "omnidirectional"
                point = label["clear_position"] or (None, None)
                connection.execute("INSERT INTO source_estimates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (episode_id, label["channel"], label["x_m"], label["y_m"], label["radius_m"], label["method"],
                     int(label["cleared"]), int(label["geometry_consistent"]), point[0], point[1],
                     label["clear_radius_m"], label["clear_step_index"], _json(label)))
            evidence = {**documents, "requests.jsonl": records, "registration.json": registration,
                        "registration-record.json": record, "official-result.json": official}
            originals = {**raw, "registration.json": (episode / "registration.json").read_bytes(),
                         "registration-record.json": raw_record, "official-result.json": raw_official}
            for name, value in evidence.items():
                cleaned = _json(sanitize(value)).encode()
                connection.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?)",
                    (episode_id, name, _hash(originals[name]), _hash(cleaned), "json+zlib", zlib.compress(cleaned, 6)))
        result["inserted"] = True
        return result
    finally:
        connection.close()


def decode_blob(blob):
    """Decode an observations.state_zlib or evidence.content_zlib value."""
    return _load(zlib.decompress(blob))


def iter_transitions(db_path, *, problem=None, split=None, case_code=None, trainable_only=True):
    """Yield chronological causal tuples; final source labels are deliberately absent.

    HTTP enter is retained in the DB but omitted with trainable_only=True.
    ``terminal`` denotes the recorded episode boundary. ``terminated`` means a
    completed all-clear exit, while ``truncated`` means an incomplete search exit.
    Episode-completion metadata must not be included in earlier policy features.
    Consumers choose discount/reward shaping themselves.
    """
    filters, parameters = [], []
    if problem is not None:
        if type(problem) is not int or problem not in (3, 4):
            raise ValueError("Only Q3 and Q4 practice are supported")
        filters.append("e.problem=?")
        parameters.append(problem)
    if split is not None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown dataset split")
        filters.append("e.split=?")
        parameters.append(split)
    if case_code is not None:
        filters.append("e.case_code=?")
        parameters.append(case_code)
    if trainable_only:
        filters.append("s.trainable=1")
    query = """SELECT s.*,e.case_code,e.problem,e.split,e.complete,
        pre.state_zlib AS before_blob,post.state_zlib AS after_blob
        FROM steps s JOIN episodes e ON e.id=s.episode_id
        LEFT JOIN observations pre ON pre.episode_id=s.episode_id AND pre.observation_index=s.observation_before_index
        LEFT JOIN observations post ON post.episode_id=s.episode_id AND post.observation_index=s.observation_after_index"""
    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += " ORDER BY e.id,s.step_index"
    connection = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        for row in connection.execute(query, parameters):
            yield {"case_code": row["case_code"], "problem": row["problem"], "split": row["split"],
                   "step_index": row["step_index"], "observation": decode_blob(row["before_blob"]) if row["before_blob"] else None,
                   "action": row["action"], "action_parameters": _load(row["action_json"]),
                   "response": _load(row["response_json"]),
                   "next_observation": decode_blob(row["after_blob"]) if row["after_blob"] else None,
                   "delta_virtual_time_s": row["delta_virtual_time_s"],
                   "reward_time_component": row["reward_time_component"],
                   "clear_success": bool(row["clear_success"]), "terminal": bool(row["terminal"]),
                   "terminated": bool(row["terminal"] and row["complete"]),
                   "truncated": bool(row["terminal"] and not row["complete"]),
                   "accepted": bool(row["accepted"]), "attempts": row["attempts"],
                   "cost_components": _load(row["cost_components_json"]), "episode_complete": bool(row["complete"])}
    finally:
        connection.close()


def stats(db_path):
    """Compact JSON-compatible collection progress; does not create absent DBs."""
    result = {"schema_version": SCHEMA_VERSION, "episodes": 0, "episodes_by_problem": {"3": 0, "4": 0},
              "complete_episodes": 0, "complete_by_problem": {"3": 0, "4": 0}, "steps": 0,
              "accepted_steps": 0, "trainable_steps": 0, "source_estimates": 0,
              "verified_clear_estimates": 0, "splits": {}, "quality_flags": {}, "bytes": 0,
              "label_precision": {"cleared_count": 0, "radius_le_5m_count": 0,
                                  "radius_median_m": None, "radius_p95_m": None,
                                  "radius_max_m": None, "inconsistent_count": 0}}
    path = Path(db_path)
    if not path.is_file():
        return result
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        for problem, count, complete in connection.execute("SELECT problem,COUNT(*),SUM(complete) FROM episodes GROUP BY problem"):
            result["episodes_by_problem"][str(problem)] = count
            result["complete_by_problem"][str(problem)] = complete
            result["episodes"] += count
            result["complete_episodes"] += complete
        for table, key in (("steps", "steps"), ("source_estimates", "source_estimates")):
            result[key] = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        result["accepted_steps"], result["trainable_steps"] = connection.execute(
            "SELECT COALESCE(SUM(accepted),0),COALESCE(SUM(trainable),0) FROM steps").fetchone()
        result["verified_clear_estimates"] = connection.execute(
            "SELECT COUNT(*) FROM source_estimates WHERE cleared=1").fetchone()[0]
        radii = sorted(row[0] for row in connection.execute(
            "SELECT radius_m FROM source_estimates WHERE cleared=1 AND radius_m IS NOT NULL"))
        precision = result["label_precision"]
        precision["cleared_count"] = len(radii)
        precision["radius_le_5m_count"] = sum(radius <= 5 + NUMERICAL_MARGIN_M for radius in radii)
        precision["inconsistent_count"] = connection.execute(
            "SELECT COUNT(*) FROM source_estimates WHERE geometry_consistent=0").fetchone()[0]
        if radii:
            precision["radius_median_m"] = statistics.median(radii)
            precision["radius_p95_m"] = radii[math.ceil(0.95 * len(radii)) - 1]
            precision["radius_max_m"] = radii[-1]
        result["splits"] = dict(connection.execute("SELECT split,COUNT(*) FROM episodes GROUP BY split"))
        for (flags,) in connection.execute("SELECT quality_flags_json FROM episodes"):
            for flag in _load(flags):
                result["quality_flags"][flag] = result["quality_flags"].get(flag, 0) + 1
        result["bytes"] = path.stat().st_size
        return result
    finally:
        connection.close()
