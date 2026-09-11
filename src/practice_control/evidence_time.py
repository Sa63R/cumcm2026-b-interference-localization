"""Match finished practice evidence without widening its official time window.

The legacy path uses the local summary timestamp. If the local and simulator
clocks differ, the fallback requires a fully validated HTTP journal and uses
the simulator timestamp of its accepted enter. No input evidence is rewritten.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import re


_CASE = re.compile(r"[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _timestamp(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO timestamp with a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("missing timezone")
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"Invalid {name}: an ISO timestamp with a timezone is required") from error


def _server_timestamp(value):
    if type(value) is not int or value < 0:
        raise ValueError("Accepted response real_timestamp_ms must be a nonnegative integer")
    try:
        return _EPOCH + timedelta(milliseconds=value)
    except (ValueError, OverflowError) as error:
        raise ValueError("Accepted response real_timestamp_ms is outside the supported timestamp range") from error


def _practice_identity(summary, official):
    problem, case = summary.get("problem"), summary.get("case_code")
    if summary.get("declared_mode") != "practice":
        raise ValueError("HTTP time fallback requires a practice session")
    if type(problem) is not int or problem not in (3, 4):
        raise ValueError("HTTP time fallback requires a Q3/Q4 practice problem")
    if not isinstance(case, str) or not _CASE.fullmatch(case):
        raise ValueError("HTTP time fallback requires a complete practice case identity")
    if official.get("case_code") != case:
        raise ValueError("Official result case code does not match the practice session")
    if type(official.get("problem_no")) is not int or official["problem_no"] != problem:
        raise ValueError("Official result problem_no does not match the practice session")
    for field in ("mode", "declared_mode"):
        if field in official and official[field] != "practice":
            raise ValueError("HTTP time fallback cannot use a non-practice result")


def match_session_time(summary, official, *, raw_journal: bytes | None = None) -> dict:
    """Return deterministic audit metadata for an unchanged evidence pair.

    Local timestamps within the exact official window retain legacy behavior;
    their callers continue to perform their existing mode/identity checks.
    The fallback checks practice identity itself, validates every journal action
    and committed state, and checks all accepted server timestamps in order.
    ``local_response_minus_server_s`` includes transport/recording delay and is
    an observable clock difference, not a calibrated clock-offset estimate.
    """
    if not isinstance(summary, dict) or not isinstance(official, dict):
        raise ValueError("Session summary and official result must be objects")
    local_start = _timestamp(summary.get("started_at"), "session started_at")
    window = _timestamp(official.get("window_started_at_utc"), "window_started_at_utc")
    ended = _timestamp(official.get("ended_at_utc"), "ended_at_utc")
    if ended < window:
        raise ValueError("Official result ended_at_utc precedes its window start")
    if window <= local_start <= ended:
        return {"basis": "local_summary", "started_at_utc": local_start.isoformat()}
    if raw_journal is None:
        raise ValueError("Session started_at is outside the official result time window")
    _practice_identity(summary, official)
    if type(raw_journal) is not bytes:
        raise ValueError("raw_journal must contain the original journal bytes")

    # Dataset import also calls this helper: delay the dependency until runtime
    # so both callers can reuse the same complete trajectory validator.
    from .dataset import _journal, _load

    try:
        records = [_load(line) for line in raw_journal.splitlines() if line.strip()]
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("Journal records must be objects")
        steps, _, _ = _journal(records, summary)
    except (TypeError, KeyError, AttributeError) as error:
        raise ValueError("Malformed journal cannot establish the practice time window") from error
    accepted = [step for step in steps if step["accepted"]]
    entry = accepted[0]  # _journal proved an accepted enter-to-exit trajectory.
    entry_ms = entry["response"].get("real_timestamp_ms")
    entry_time = _server_timestamp(entry_ms)
    entry_request_id = entry["payload"]["request_id"]
    previous_ms = None
    entry_response_record = None
    for record in records:
        if record.get("event") != "response":
            continue
        response = record.get("response")
        if not isinstance(response, dict) or response.get("accepted") is not True:
            continue
        server_ms = response.get("real_timestamp_ms")
        _server_timestamp(server_ms)
        if previous_ms is not None and server_ms < previous_ms:
            raise ValueError("Accepted response real timestamps move backwards")
        previous_ms = server_ms
        if record.get("request_id") == entry_request_id and response == entry["response"]:
            entry_response_record = record
    if not window <= entry_time <= ended:
        raise ValueError("Accepted HTTP enter is outside the official result time window")
    if entry_response_record is None:
        raise ValueError("Accepted HTTP enter response is missing from the journal")
    local_response = _timestamp(entry_response_record.get("recorded_at"), "accepted enter response recorded_at")
    return {
        "basis": "simulator_http_enter",
        "started_at_utc": entry_time.isoformat(),
        "local_started_at_utc": local_start.isoformat(),
        "enter_real_timestamp_ms": entry_ms,
        "enter_request_id": entry_request_id,
        "enter_response_recorded_at_utc": local_response.isoformat(),
        "local_response_minus_server_s": (local_response - entry_time).total_seconds(),
        "journal_sha256": hashlib.sha256(raw_journal).hexdigest(),
        "validated_accepted_count": len(accepted),
    }
