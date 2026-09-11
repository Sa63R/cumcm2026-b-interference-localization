"""Offline time evidence: exact windows, complete journals and clock differences."""

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from practice_control.evidence_time import match_session_time


CASE = "ABCD-EFGH-IJKL-MNOP"
ENTER_MS = 1789128265388


def journal_bytes(records):
    return ("\n".join(json.dumps(row) for row in records) + "\n").encode("utf-8")


def fixture():
    summary = {"declared_mode": "practice", "problem": 3, "case_code": CASE,
               "started_at": "2026-09-11T12:04:26.293488+00:00"}
    official = {"problem_no": 3, "case_code": CASE,
                "window_started_at_utc": "2026-09-11T12:04:25.107Z",
                "ended_at_utc": "2026-09-11T12:04:26.272Z"}
    records = [{"event": "session_created"}]
    state = {"session": "active", "accepted_actions": 0, "cleared_count": 0,
             "virtual_time_s": 0.0, "position": {"x": 0.0, "y": 0.0},
             "time_breakdown": {"detection_s": 0.0, "optical_s": 0.0, "removal_s": 0.0}}
    for index, (path, real_ms) in enumerate([
            ("/enter", ENTER_MS), ("/measure", ENTER_MS + 400),
            ("/clear", ENTER_MS + 600), ("/exit", ENTER_MS + 885)], 1):
        request_id = f"unit-request-{index}"
        payload = {"request_id": request_id}
        response = {"accepted": True, "real_timestamp_ms": real_ms}
        if path in {"/measure", "/clear"}:
            payload.update(position={"x": 0.0, "y": 0.0}, channel=1)
        if path == "/measure":
            state["time_breakdown"]["detection_s"] = 5.0
            response["measure_result"] = "near"
        if path == "/clear":
            state["time_breakdown"].update(optical_s=3.0, removal_s=2.0)
            state["cleared_count"] = 1
            response["clear_result"] = "success"
        if path == "/exit":
            state["session"] = "exited"
            response["exit_reason"] = "user_exit"
        state["accepted_actions"] = index
        state["virtual_time_s"] = sum(state["time_breakdown"].values())
        response["virtual_time_s"] = state["virtual_time_s"]
        local_response = (datetime(1970, 1, 1, tzinfo=timezone.utc)
                          + timedelta(milliseconds=real_ms + 946)).isoformat()
        records.extend([
            {"event": "request", "path": path, "payload": payload, "attempt": 1},
            {"event": "response", "request_id": request_id, "http_status": 200,
             "response": response, "recorded_at": local_response},
            {"event": "state", "request_id": request_id, "state": copy.deepcopy(state)},
        ])
    records.append({"event": "client_closed"})
    summary["state"] = copy.deepcopy(state)
    return summary, official, records


def test_local_path_keeps_legacy_shape_and_does_not_need_case_or_journal(monkeypatch):
    summary = {"started_at": "2026-09-11T20:04:25.388+08:00"}
    official = {"window_started_at_utc": "2026-09-11T12:04:25.107Z",
                "ended_at_utc": "2026-09-11T12:04:26.272Z"}
    import practice_control.dataset
    monkeypatch.setattr(practice_control.dataset, "_journal", lambda *unused: pytest.fail("Local path must not validate a journal"))
    result = match_session_time(summary, official, raw_journal=b"not json")
    assert result == {"basis": "local_summary", "started_at_utc": "2026-09-11T12:04:25.388000+00:00"}


@pytest.mark.parametrize("key,value", [("window_started_at_utc", "2026-09-11T12:04:25.107"),
                                      ("ended_at_utc", None), ("ended_at_utc", "bad timestamp")])
def test_official_timestamps_require_timezone_and_valid_iso(key, value):
    summary, official, records = fixture()
    official[key] = value
    with pytest.raises(ValueError):
        match_session_time(summary, official, raw_journal=journal_bytes(records))


@pytest.mark.parametrize("value", [None, 123, "2026-09-11T12:04:26.293488", "invalid"])
def test_local_start_is_strict_even_when_journal_could_match(value):
    summary, official, records = fixture()
    summary["started_at"] = value
    with pytest.raises(ValueError):
        match_session_time(summary, official, raw_journal=journal_bytes(records))


def test_reversed_official_window_is_always_rejected():
    summary, official, records = fixture()
    official["ended_at_utc"] = "2026-09-11T12:04:24Z"
    with pytest.raises(ValueError, match="precedes"):
        match_session_time(summary, official, raw_journal=journal_bytes(records))


def test_outside_local_window_without_original_journal_is_still_rejected():
    summary, official, _ = fixture()
    with pytest.raises(ValueError, match="outside.*time window"):
        match_session_time(summary, official)


def test_validated_server_enter_matches_clock_difference_and_preserves_inputs():
    summary, official, records = fixture()
    snapshot = copy.deepcopy((summary, official, records))
    raw = journal_bytes(records)
    result = match_session_time(summary, official, raw_journal=raw)
    assert result == {
        "basis": "simulator_http_enter", "started_at_utc": "2026-09-11T12:04:25.388000+00:00",
        "local_started_at_utc": "2026-09-11T12:04:26.293488+00:00",
        "enter_real_timestamp_ms": ENTER_MS, "enter_request_id": "unit-request-1",
        "enter_response_recorded_at_utc": "2026-09-11T12:04:26.334000+00:00",
        "local_response_minus_server_s": 0.946,
        "journal_sha256": hashlib.sha256(raw).hexdigest(), "validated_accepted_count": 4,
    }
    assert (summary, official, records) == snapshot
    assert raw == journal_bytes(records)
    # Exit is one millisecond after result generation; only enter must lie in
    # the exact window, and accepted timestamps must remain nondecreasing.
    assert records[-3]["response"]["real_timestamp_ms"] == ENTER_MS + 885


@pytest.mark.parametrize("target,key,value", [
    ("summary", "declared_mode", "formal"), ("summary", "declared_mode", None),
    ("summary", "problem", True), ("summary", "problem", 5),
    ("summary", "case_code", None), ("summary", "case_code", "incomplete"),
    ("official", "case_code", "ZZZZ-YYYY-XXXX-WWWW"), ("official", "problem_no", 4),
    ("official", "problem_no", True), ("official", "mode", "formal"),
])
def test_fallback_rejects_formal_or_wrong_case_and_problem(target, key, value):
    summary, official, records = fixture()
    (summary if target == "summary" else official)[key] = value
    with pytest.raises(ValueError):
        match_session_time(summary, official, raw_journal=journal_bytes(records))


@pytest.mark.parametrize("index,value", [
    (2, None), (2, True), (2, -1), (2, 1789128265388.0), (2, "1789128265388"),
    (5, False), (5, 1.5), (11, -1), (11, 10 ** 30),
])
def test_all_accepted_response_timestamps_are_nonnegative_integers(index, value):
    summary, official, records = fixture()
    records[index]["response"]["real_timestamp_ms"] = value
    with pytest.raises(ValueError, match="real_timestamp_ms"):
        match_session_time(summary, official, raw_journal=journal_bytes(records))


def test_server_timestamp_regression_is_rejected():
    summary, official, records = fixture()
    records[8]["response"]["real_timestamp_ms"] = ENTER_MS + 100
    with pytest.raises(ValueError, match="backwards"):
        match_session_time(summary, official, raw_journal=journal_bytes(records))


def test_equal_server_timestamps_are_allowed():
    summary, official, records = fixture()
    for row in records:
        if row["event"] == "response":
            row["response"]["real_timestamp_ms"] = ENTER_MS
    assert match_session_time(summary, official, raw_journal=journal_bytes(records))["validated_accepted_count"] == 4


@pytest.mark.parametrize("offset", [-5000, 5000])
def test_server_enter_outside_exact_window_is_not_tolerated(offset):
    summary, official, records = fixture()
    for row in records:
        if row["event"] == "response":
            row["response"]["real_timestamp_ms"] += offset
    with pytest.raises(ValueError, match="outside.*time window"):
        match_session_time(summary, official, raw_journal=journal_bytes(records))


@pytest.mark.parametrize("mutation", ["missing_state", "bad_count", "wrong_final", "out_of_order", "missing_exit", "bad_cost"])
def test_fallback_requires_full_trajectory_and_committed_final_state(mutation):
    summary, official, records = fixture()
    if mutation == "missing_state":
        del records[3]
    elif mutation == "bad_count":
        records[6]["state"]["accepted_actions"] = 99
    elif mutation == "wrong_final":
        summary["state"]["accepted_actions"] = 99
    elif mutation == "out_of_order":
        records[1], records[2] = records[2], records[1]
    elif mutation == "missing_exit":
        del records[-4:-1]
    elif mutation == "bad_cost":
        records[6]["state"]["time_breakdown"]["detection_s"] = 6
    with pytest.raises(ValueError):
        match_session_time(summary, official, raw_journal=journal_bytes(records))


@pytest.mark.parametrize("value", [None, "not-a-date", "2026-09-11T12:04:26.334"])
def test_enter_local_response_timestamp_is_required_for_clock_audit(value):
    summary, official, records = fixture()
    records[2]["recorded_at"] = value
    with pytest.raises(ValueError, match="recorded_at"):
        match_session_time(summary, official, raw_journal=journal_bytes(records))


@pytest.mark.parametrize("raw", [b"", b"not json\n", b"null\n", b"[]\n", bytearray(b"{}\n")])
def test_fallback_rejects_invalid_original_journal(raw):
    summary, official, _ = fixture()
    with pytest.raises(ValueError):
        match_session_time(summary, official, raw_journal=raw)
