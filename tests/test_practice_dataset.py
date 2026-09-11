"""Offline evidence integrity, causal transitions and bounded-label tests."""

import copy
import hashlib
import json
import math
import sqlite3

import pytest

from practice_control.dataset import decode_blob, import_episode, iter_transitions, split_for_case, stats


CASE = "ABCD-EFGH-IJKL-MNOP"
PRIVATE_ROBOT = "DO-NOT-EXPORT-TEAM-123456"


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_registration(episode):
    record = load(episode.parent / "registered/practice-test.json")
    record["summary_sha256"] = hashlib.sha256((episode / "summary.json").read_bytes()).hexdigest()
    write(episode.parent / "registered/practice-test.json", record)


def example(tmp_path, *, problem=3, case=CASE, complete=True, clear_count=10):
    """Small protocol-consistent saved journal; never starts any application."""
    episode = tmp_path / "run-001"
    episode.mkdir(parents=True)
    costs = dict(movement_s=0.0, switching_s=0.0, detection_s=0.0, optical_s=0.0, removal_s=0.0)
    state = {"session": "new", "position": {"x": 0.0, "y": 0.0}, "current_channel": 1,
             "virtual_time_s": 0.0, "accepted_actions": 0, "cleared_count": 0, "sources": {},
             "time_breakdown": costs, "real_deadline": 123456.0}
    journal = [{"event": "session_created", "robot_id": PRIVATE_ROBOT, "run_id": "private-run-id"}]

    def action(path, channel=None, point=None, result=None, bearing=None):
        payload = {"robot_id": PRIVATE_ROBOT, "request_id": f"private-request-{state['accepted_actions'] + 1}"}
        response = {"accepted": True, "real_timestamp_ms": 123}
        if channel is not None:
            payload.update(channel=channel, position={"x": point[0], "y": point[1]})
            costs["movement_s"] += math.dist(tuple(state["position"].values()), point) / 5
            state["position"] = payload["position"].copy()
            source = state["sources"].setdefault(str(channel), {"status": "unknown", "measurement_count": 0})
            if path == "/measure":
                costs["switching_s"] += state["current_channel"] != channel
                costs["detection_s"] += 5
                state["current_channel"] = channel
                source.update(status="detected" if result != "no_signal" else source["status"],
                              measurement_count=source["measurement_count"] + 1, last_result=result)
                response["measure_result"] = result
                if bearing is not None:
                    response["svd_deg"] = bearing
            else:
                costs["optical_s"] += 3
                costs["removal_s"] += 2 * (result == "success")
                response["clear_result"] = result
                if result == "success":
                    state["cleared_count"] += 1
                    source["status"] = "cleared"
        elif path == "/enter":
            state["session"] = "active"
        else:
            state["session"] = "exited"
            response["exit_reason"] = "user_exit"
        state["virtual_time_s"] = sum(costs.values())
        state["accepted_actions"] += 1
        response["virtual_time_s"] = state["virtual_time_s"]
        journal.extend([
            {"event": "request", "path": path, "payload": payload, "attempt": 1},
            {"event": "response", "request_id": payload["request_id"], "response": response, "http_status": 200},
            {"event": "state", "request_id": payload["request_id"], "state": copy.deepcopy(state)},
        ])

    action("/enter")
    action("/measure", 1, (0.0, 0.0), "direction", 45.0)
    action("/measure", 1, (200.0, 0.0), "direction", 135.0)
    action("/measure", 1, (101.0, 100.0), "near")
    action("/clear", 1, (100.0, 102.0), "success")
    for channel in range(2, clear_count + 1):
        action("/clear", channel, (channel * 10.0, 0.0), "success")
    action("/measure", 20, (100.0, 102.0), "no_signal")
    action("/exit")
    (episode / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in journal) + "\n", encoding="utf-8")
    summary = {"data_origin": "simulator_http_session", "declared_mode": "practice", "problem": problem,
               "case_code": case, "variant": "unit-fixture", "code_revision": "test-only",
               "started_at": "2026-09-11T10:00:05+00:00", "gui_mode_verified_by_api": True,
               "completed": complete, "state": state, "program_wall_time_s": 1.2, "pending_request": None,
               "search": {"source_estimates": {"1": {"center": [99999, 99999], "radius_m": 0}}}}
    write(episode / "summary.json", summary)
    write(episode / "created.json", {"mode": "practice", "problem": problem, "case_code": case})
    lifecycle = {"mode": "practice", "problem_no": problem, "case_code": case}
    write(episode / "before.json", {**lifecycle, "phase": "waiting_enter", "entered": False})
    write(episode / "after.json", {**lifecycle, "phase": "ended", "cleanup_complete": True,
                                   "api_open": False, "log_package_status": "ready"})
    registered = tmp_path / "registered"
    (registered / "evidence").mkdir(parents=True)
    result = {"case_code": case, "problem_no": problem, "jammer_count": 10,
              "omnidirectional_jammer_count": 10 if problem == 3 else 1,
              "directional_jammer_count": 0 if problem == 3 else 9,
              "practice_run_no": 87654321,
              "window_started_at_utc": "2026-09-11T10:00:00Z", "ended_at_utc": "2026-09-11T10:00:10Z"}
    result_path = registered / "evidence/archived.result.json"
    write(result_path, result)
    record = {"data_origin": "registered_official_practice", "problem": problem, "case_code": case,
              "source_total_source": "official_simulator_result_file", "source_total": 10,
              "cleared_count": clear_count, "official_result_session_time_matched": True,
              "official_result_original_name": f"practice-p{problem}-123-{case}.result.json",
              "summary_sha256": hashlib.sha256((episode / "summary.json").read_bytes()).hexdigest(),
              "official_result_path": "evidence/archived.result.json",
              "official_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()}
    write(registered / "practice-test.json", record)
    write(episode / "registration.json", {"record": str(registered / "practice-test.json"), "case_code": case, "source_total": 10})
    return episode


def journal(episode):
    return [json.loads(line) for line in (episode / "requests.jsonl").read_text().splitlines()]


def write_journal(episode, records):
    (episode / "requests.jsonl").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")


def test_single_file_import_causal_loader_and_costs(tmp_path):
    episode = example(tmp_path / "raw")
    database = tmp_path / "data.sqlite3"
    imported = import_episode(database, episode)
    assert imported["inserted"] and imported["complete"]
    transitions = list(iter_transitions(database))
    first = transitions[0]
    assert first["action"] == "measure"
    assert first["observation"]["sources"] == {}
    assert first["next_observation"]["sources"]["1"]["last_result"] == "direction"
    assert first["observation"]["virtual_time_s"] == 0
    assert first["reward_time_component"] == -5
    assert "source_estimates" not in json.dumps(first)
    assert transitions[-1]["terminal"]
    assert transitions[-1]["terminated"] and not transitions[-1]["truncated"]
    for left, right in zip(transitions, transitions[1:]):
        assert left["next_observation"] == right["observation"]
    total = sum(row["delta_virtual_time_s"] for row in transitions)
    assert math.isclose(total, load(episode / "summary.json")["state"]["virtual_time_s"])
    assert sum(row["clear_success"] for row in transitions) == 10
    assert set(tmp_path.glob("data.sqlite3*")) == {database}
    progress = stats(database)
    assert progress["complete_by_problem"] == {"3": 1, "4": 0}
    assert progress["accepted_steps"] == imported["steps"]
    assert progress["trainable_steps"] == imported["steps"] - 1
    assert progress["label_precision"]["cleared_count"] == 10
    assert progress["label_precision"]["radius_le_5m_count"] == 1
    assert progress["label_precision"]["radius_median_m"] == 20
    assert progress["label_precision"]["radius_p95_m"] == 20
    assert progress["label_precision"]["radius_max_m"] == 20


def test_mathematical_labels_ignore_fake_summary_truth_and_remain_bounded(tmp_path):
    episode = example(tmp_path / "raw", problem=4)
    database = tmp_path / "data.sqlite3"
    import_episode(database, episode)
    with sqlite3.connect(database) as connection:
        labels = [json.loads(row[0]) for row in connection.execute("SELECT label_json FROM source_estimates ORDER BY channel")]
    assert len(labels) == 10  # Channel 20 no_signal never creates a positive source label.
    first = labels[0]
    assert not first["is_exact_truth"]
    assert first["source_type"] == "unknown"  # Q4 totals cannot label individual types.
    assert math.dist((100, 100), (first["x_m"], first["y_m"])) <= first["radius_m"]
    assert first["radius_m"] <= 5
    assert first["geometry_consistent"]
    assert first["clear_radius_m"] == 20 and first["clear_position"] == [100, 102]
    assert len(first["constraints"]) == 4
    assert {item["kind"] for item in first["constraints"]} == {"direction", "near", "success"}
    assert all(label["radius_m"] <= 20 for label in labels)
    assert all(label["method"] == "success_disk" for label in labels[1:])


def test_q4_all_directional_case_imports_and_preserves_verified_counts(tmp_path):
    episode = example(tmp_path / "raw", problem=4)
    result_path = episode.parent / "registered/evidence/archived.result.json"
    result = load(result_path)
    result.update(omnidirectional_jammer_count=0, directional_jammer_count=10)
    write(result_path, result)
    record_path = episode.parent / "registered/practice-test.json"
    record = load(record_path)
    record["official_result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    write(record_path, record)
    database = tmp_path / "data.sqlite3"
    imported = import_episode(database, episode)
    assert imported["complete"] and imported["source_estimates"] == 10
    assert stats(database)["complete_by_problem"] == {"3": 0, "4": 1}
    with sqlite3.connect(database) as connection:
        saved = decode_blob(connection.execute(
            "SELECT content_zlib FROM evidence WHERE name='official-result.json'").fetchone()[0])
    assert saved["omnidirectional_jammer_count"] == 0
    assert saved["directional_jammer_count"] == 10


@pytest.mark.parametrize("complete,clears,expected_flags", [
    (False, 9, ["search_incomplete", "not_all_sources_cleared"]),
    (False, 10, ["search_incomplete"]),
    (True, 9, ["not_all_sources_cleared"]),
])
def test_retains_failed_search_without_counting_toward_complete_goal(tmp_path, complete, clears, expected_flags):
    episode = example(tmp_path / "raw", complete=complete, clear_count=clears)
    database = tmp_path / "data.sqlite3"
    imported = import_episode(database, episode)
    assert imported["complete"] is False
    assert imported["quality_flags"] == expected_flags
    assert stats(database)["episodes_by_problem"]["3"] == 1
    assert stats(database)["complete_by_problem"]["3"] == 0
    last = list(iter_transitions(database))[-1]
    assert last["terminal"] and last["truncated"] and not last["terminated"]


def test_private_identifiers_removed_from_states_evidence_and_actions(tmp_path):
    episode = example(tmp_path / "raw")
    database = tmp_path / "data.sqlite3"
    import_episode(database, episode)
    with sqlite3.connect(database) as connection:
        evidence = {name: decode_blob(blob) for name, blob in connection.execute("SELECT name,content_zlib FROM evidence")}
        actions = [row[0] for row in connection.execute("SELECT action_json FROM steps")]
        observations = [decode_blob(row[0]) for row in connection.execute("SELECT state_zlib FROM observations")]
    text = json.dumps([evidence, actions, observations])
    assert PRIVATE_ROBOT not in text
    assert "private-request-" not in text
    assert "private-run-id" not in text
    assert "robot_id" not in text
    assert "practice_run_no" not in text
    assert "real_deadline" not in text
    assert 'request-000001' in text


def test_identical_import_is_idempotent_and_conflict_rolls_back(tmp_path):
    episode = example(tmp_path / "raw")
    database = tmp_path / "data.sqlite3"
    assert import_episode(database, episode)["inserted"]
    assert not import_episode(database, episode)["inserted"]
    before = stats(database)
    summary = load(episode / "summary.json")
    summary["variant"] = "different-policy"
    write(episode / "summary.json", summary)
    refresh_registration(episode)
    with pytest.raises(ValueError, match="conflicting evidence"):
        import_episode(database, episode)
    assert stats(database) == before


def test_retry_is_one_transition_and_marked(tmp_path):
    episode = example(tmp_path / "raw")
    records = journal(episode)
    request_index = next(i for i, row in enumerate(records) if row.get("path") == "/measure")
    request = copy.deepcopy(records[request_index])
    request["attempt"] = 2
    records.insert(request_index + 1, request)
    write_journal(episode, records)
    database = tmp_path / "data.sqlite3"
    imported = import_episode(database, episode)
    assert "retried_requests" in imported["quality_flags"]
    assert imported["steps"] == load(episode / "summary.json")["state"]["accepted_actions"]
    assert next(iter_transitions(database))["attempts"] == 2


def test_rejected_action_retained_but_not_training_transition(tmp_path):
    episode = example(tmp_path / "raw")
    records = journal(episode)
    records[4:4] = [
        {"event": "request", "path": "/measure", "payload": {"request_id": "rejected", "channel": 1, "position": {"x": 0, "y": 0}}},
        {"event": "response", "request_id": "rejected", "http_status": 429,
         "response": {"accepted": False, "virtual_time_s": 0}},
    ]
    write_journal(episode, records)
    database = tmp_path / "data.sqlite3"
    imported = import_episode(database, episode)
    assert "rejected_requests" in imported["quality_flags"]
    rows = list(iter_transitions(database, trainable_only=False))
    rejected = rows[1]
    assert not rejected["accepted"]
    assert rejected["observation"] == rejected["next_observation"]
    assert rejected["reward_time_component"] == 0
    assert all(row["accepted"] for row in iter_transitions(database))


@pytest.mark.parametrize("file,field,value", [
    ("summary.json", "declared_mode", "formal"),
    ("summary.json", "problem", 2),
    ("summary.json", "gui_mode_verified_by_api", False),
    ("summary.json", "pending_request", {"path": "/measure"}),
    ("created.json", "mode", "formal"),
    ("before.json", "entered", True),
    ("before.json", "case_code", "DIFF-EREN-TCAS-ECOD"),
    ("after.json", "phase", "running"),
    ("after.json", "cleanup_complete", False),
    ("after.json", "log_package_status", "collecting"),
])
def test_scope_lifecycle_fail_closed_before_creating_database(tmp_path, file, field, value):
    episode = example(tmp_path / "raw")
    document = load(episode / file)
    document[field] = value
    write(episode / file, document)
    refresh_registration(episode)
    database = tmp_path / "data.sqlite3"
    with pytest.raises(ValueError):
        import_episode(database, episode)
    assert not database.exists()


def test_formal_scope_rejected_before_reading_registration(tmp_path):
    episode = example(tmp_path / "raw")
    summary = load(episode / "summary.json")
    summary["declared_mode"] = "formal"
    write(episode / "summary.json", summary)
    (episode / "registration.json").unlink()
    with pytest.raises(ValueError, match="Only practice"):
        import_episode(tmp_path / "data.sqlite3", episode)


@pytest.mark.parametrize("mutation", ["missing_state", "missing_exit", "state_gap", "virtual_backwards", "duplicate_state", "unsupported_path"])
def test_malformed_journal_is_not_silently_turned_into_training_data(tmp_path, mutation):
    episode = example(tmp_path / "raw")
    records = journal(episode)
    first_state = next(i for i, row in enumerate(records) if row["event"] == "state")
    if mutation == "missing_state":
        records.pop(first_state)
    elif mutation == "missing_exit":
        records = records[:-3]
    elif mutation == "state_gap":
        records[first_state]["state"]["accepted_actions"] = 7
    elif mutation == "virtual_backwards":
        records[-2]["response"]["virtual_time_s"] = 0
    elif mutation == "duplicate_state":
        records.insert(first_state, copy.deepcopy(records[first_state]))
    else:
        records[1]["path"] = "/unknown"
    write_journal(episode, records)
    with pytest.raises(ValueError):
        import_episode(tmp_path / "data.sqlite3", episode)
    assert not (tmp_path / "data.sqlite3").exists()


def test_modified_result_and_summary_hashes_rejected(tmp_path):
    episode = example(tmp_path / "raw")
    result = episode.parent / "registered/evidence/archived.result.json"
    result.write_text(result.read_text() + " ")
    with pytest.raises(ValueError, match="result hash"):
        import_episode(tmp_path / "data.sqlite3", episode)
    record_path = episode.parent / "registered/practice-test.json"
    record = load(record_path)
    record["official_result_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
    record["summary_sha256"] = "bad-hash"
    write(record_path, record)
    with pytest.raises(ValueError, match="summary hash"):
        import_episode(tmp_path / "data.sqlite3", episode)


def test_inconsistent_geometry_is_flagged_and_uses_observed_disk_fallback(tmp_path):
    episode = example(tmp_path / "raw")
    records = journal(episode)
    bearing_response = next(row["response"] for row in records if row["event"] == "response" and row["response"].get("svd_deg") == 45)
    bearing_response["svd_deg"] = 225
    write_journal(episode, records)
    database = tmp_path / "data.sqlite3"
    assert "inconsistent_geometry" in import_episode(database, episode)["quality_flags"]
    with sqlite3.connect(database) as connection:
        label = json.loads(connection.execute("SELECT label_json FROM source_estimates WHERE channel=1").fetchone()[0])
    assert not label["geometry_consistent"]
    assert label["method"] == "near_disk"
    assert label["radius_m"] == 5
    assert label["outer_vertices"] == []


def test_empty_stats_and_whole_case_splits(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    assert stats(missing)["episodes_by_problem"] == {"3": 0, "4": 0}
    assert not missing.exists()
    episode = example(tmp_path / "raw")
    database = tmp_path / "data.sqlite3"
    import_episode(database, episode)
    split = split_for_case(CASE)
    assert {row["split"] for row in iter_transitions(database)} == {split}
    assert list(iter_transitions(database, problem=4)) == []
    assert list(iter_transitions(database, split=split))
    for invalid in ("all", "eval", ""):
        with pytest.raises(ValueError):
            list(iter_transitions(database, split=invalid))
