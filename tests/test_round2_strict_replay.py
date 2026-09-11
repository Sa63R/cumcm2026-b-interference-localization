"""Artificial fixtures only. Never load the reserved official validation DB."""

import copy
import json
from pathlib import Path
import sqlite3

import pytest

from experiments.round2_strict_replay import (
    PrefixReplayClient, RecordedRejection, ReplayDivergence, StrictPrefixTape,
    _make_client, canonical_hash, file_hash, main, sqlite_steps, validate_policy,
    verify_manifest,
)


def step(index, action, *, time_before, time_after, position=None, channel=1,
         kind=None, costs=None, accepted=True, attempts=1):
    parameters = ({"position": {"x": position[0], "y": position[1]}, "channel": channel}
                  if position is not None else {})
    response = {"accepted": accepted, "virtual_time_s": time_after, "real_timestamp_ms": 1}
    if accepted and action == "enter":
        response.update(max_virtual_duration_s=360000, max_real_duration_s=1200,
                        remaining_real_duration_s=1200)
    elif accepted and action == "measure":
        response["measure_result"] = kind
        if kind == "direction":
            response["svd_deg"] = 0.0
    elif accepted and action == "clear":
        response["clear_result"] = kind
    elif accepted and action == "exit":
        response["exit_reason"] = "user_exit"
    return {"step_index": index, "action": action, "action_parameters": parameters,
            "response": response, "accepted": accepted, "attempts": attempts,
            "virtual_time_before_s": time_before, "virtual_time_after_s": time_after,
            "delta_virtual_time_s": time_after - time_before, "cost_components": costs or {}}


def trajectory():
    return [step(0, "enter", time_before=0, time_after=0),
            step(1, "measure", time_before=0, time_after=5, position=(0, 0),
                 kind="direction", costs={"detection_s": 5}),
            step(2, "measure", time_before=5, time_after=30, position=(100, 0),
                 kind="near", costs={"detection_s": 5, "movement_s": 20}),
            step(3, "clear", time_before=30, time_after=35, position=(100, 0),
                 kind="success", costs={"optical_s": 3, "removal_s": 2}),
            step(4, "exit", time_before=35, time_after=35)]


def same_policy(client):
    assert client.state.sources == {}
    client.enter()
    client.measure((-0.0, 0.0), 1)
    client.measure((100.0, 0), 1)
    client.clear((100, 0), 1)
    client.exit()


def test_exact_trajectory_costs_and_geometry_without_final_labels():
    result = validate_policy(trajectory(), same_policy, clock=lambda: 0)
    assert result["status"] == "recorded_trajectory_matched"
    assert result["matched_steps"] == result["loaded_steps"] == 5
    assert result["matched_prefix_virtual_time_s"] == 35
    assert result["matched_prefix_costs"] == dict(
        movement_s=20.0, switching_s=0.0, detection_s=10.0, optical_s=3.0, removal_s=2.0)
    assert result["geometry_audit"]["clear_attempts_with_prior_certificate"] == 1
    assert result["geometry_audit"]["empty_outer_region_updates"] == 0
    assert result["performance_comparison_supported"] is False


@pytest.mark.parametrize("action,position,channel", [
    ("measure", (0, 0), 2), ("clear", (0, 0), 1),
    ("measure", (1e-15, 0), 1), ("measure", (0, 1e-15), 1),
    ("measure", (100, 0), 1),
])
def test_exact_coordinate_channel_and_order_mismatch_locks_cleanup(action, position, channel):
    loaded = []
    def records():
        for record in trajectory():
            loaded.append(record["step_index"])
            yield record
    def policy(client):
        client.enter()
        try:
            getattr(client, action)(position, channel)
        finally:
            client.exit()  # Real strategies often do this; it must not read onward.
    result = validate_policy(records(), policy, clock=lambda: 0)
    assert result["status"] == "unsupported_counterfactual_action"
    assert result["matched_steps"] == 1
    assert loaded == [0, 1]
    assert result["matched_prefix_virtual_time_s"] == 0
    assert result["truncation_is_policy_failure"] is False
    assert "position" not in result["first_unsupported"]


def test_no_future_read_even_when_policy_catches_divergence_and_retries():
    rows = iter(trajectory())
    tape = StrictPrefixTape(rows)
    client = _make_client(tape, lambda: 0)
    client.enter()
    with pytest.raises(ReplayDivergence):
        client.measure((0, 0), 2)
    for action in (lambda: client.measure((0, 0), 1), client.exit):
        with pytest.raises(ReplayDivergence):
            action()
    assert tape.loaded_steps == 2
    assert next(rows)["step_index"] == 2


def test_policy_state_is_a_prefix_copy_and_whitelist_hides_tape_and_labels():
    tape = StrictPrefixTape(trajectory())
    raw = _make_client(tape, lambda: 0)
    client = PrefixReplayClient(raw)
    for name in ("tape", "source_estimates", "episode_complete", "case_code", "_exchange_fn"):
        with pytest.raises(AttributeError):
            getattr(client, name)
    client.enter()
    saved = client.state
    saved.virtual_time_s = 999
    saved.sources[20] = "future"
    assert raw.state.virtual_time_s == 0
    assert raw.state.sources == {}
    client.measure((0, 0), 1)
    assert set(client.state.sources) == {1}
    assert client.state.cleared_count == 0
    assert tape.loaded_steps == 2


def test_exhaustion_is_unsupported_not_synthetic_exit_success():
    result = validate_policy(trajectory()[:2], same_policy, clock=lambda: 0)
    assert result["status"] == "unsupported_counterfactual_action"
    assert result["matched_steps"] == 2
    assert result["first_unsupported"]["reason"] == "recorded_trajectory_exhausted"


def test_early_policy_return_is_not_full_trajectory_match():
    result = validate_policy(trajectory(), lambda client: client.enter(), clock=lambda: 0)
    assert result["status"] == "policy_returned_before_recorded_exit"
    assert result["loaded_steps"] == 1


@pytest.mark.parametrize("field", ["component", "response_time", "accepted", "step_index"])
def test_corrupt_matching_evidence_is_rejected(field):
    rows = trajectory()
    if field == "component":
        rows[1]["cost_components"]["detection_s"] = 0
    elif field == "response_time":
        rows[1]["response"]["virtual_time_s"] = 7
    elif field == "accepted":
        rows[1]["accepted"] = False
    else:
        rows[1]["step_index"] = 17
    result = validate_policy(rows, same_policy, clock=lambda: 0)
    assert result["status"] == "invalid_recorded_evidence"
    assert result["matched_steps"] == 1
    assert result["matched_prefix_virtual_time_s"] == 0


def test_saved_rejection_and_retry_count_do_not_apply_state_or_duplicate_cost():
    rows = trajectory()
    rejected = step(1, "measure", time_before=0, time_after=0,
                    position=(0, 0), accepted=False, attempts=3)
    rows.insert(1, rejected)
    for index, row in enumerate(rows):
        row["step_index"] = index
    def policy(client):
        client.enter()
        with pytest.raises(RecordedRejection) as rejection:
            client.measure((0, 0), 1)
        assert rejection.value.response["accepted"] is False
        assert client.state.sources == {}
        assert client.state.virtual_time_s == 0
        client.measure((0, 0), 1)
        client.measure((100, 0), 1)
        client.clear((100, 0), 1)
        client.exit()
    result = validate_policy(rows, policy, clock=lambda: 0)
    assert result["status"] == "recorded_trajectory_matched"
    assert result["matched_steps"] == 6
    assert result["accepted_steps"] == 5
    assert result["rejected_steps"] == 1
    assert result["recorded_retry_attempts_not_resimulated"] == 2
    assert result["matched_prefix_virtual_time_s"] == 35


def test_clear_does_not_switch_the_measurement_channel():
    rows = [step(0, "enter", time_before=0, time_after=0),
            step(1, "clear", time_before=0, time_after=3, position=(0, 0), channel=2,
                 kind="no_target_in_range", costs={"optical_s": 3}),
            step(2, "measure", time_before=3, time_after=8, position=(0, 0), channel=1,
                 kind="no_signal", costs={"detection_s": 5}),
            step(3, "exit", time_before=8, time_after=8)]
    def policy(client):
        client.enter()
        client.clear((0, 0), 2)
        assert client.state.current_channel == 1
        client.measure((0, 0), 1)
        client.exit()
    result = validate_policy(rows, policy, clock=lambda: 0)
    assert result["status"] == "recorded_trajectory_matched"
    assert result["matched_prefix_costs"]["switching_s"] == 0
    assert result["geometry_audit"]["recorded_failed_clears_in_prefix"] == 1


def test_signal_after_removal_is_invalid_evidence_not_a_new_source():
    rows = trajectory()[:-1]
    rows.append(step(4, "measure", time_before=35, time_after=40, position=(100, 0),
                     kind="near", costs={"detection_s": 5}))
    def policy(client):
        client.enter()
        client.measure((0, 0), 1)
        client.measure((100, 0), 1)
        client.clear((100, 0), 1)
        client.measure((100, 0), 1)
    result = validate_policy(rows, policy, clock=lambda: 0)
    assert result["status"] == "invalid_recorded_evidence"
    assert result["matched_steps"] == 4
    assert result["matched_prefix_virtual_time_s"] == 35


@pytest.mark.parametrize("corrupt", ["missing_action", "bad_attempts", "decode_failure"])
def test_malformed_next_record_locks_without_consuming_later_records(corrupt):
    rows = trajectory()
    if corrupt == "missing_action":
        del rows[1]["action"]
    elif corrupt == "bad_attempts":
        rows[1]["attempts"] = 0
    loaded = []
    def records():
        for index, record in enumerate(rows):
            loaded.append(index)
            if index == 1 and corrupt == "decode_failure":
                raise ValueError("Artificial decode error")
            yield record
    def policy(client):
        client.enter()
        try:
            client.measure((0, 0), 1)
        finally:
            client.exit()
    result = validate_policy(records(), policy, clock=lambda: 0)
    assert result["status"] == "invalid_recorded_evidence"
    assert loaded == [0, 1]
    assert result["matched_steps"] == 1


def write_db(path):
    """Deliberately poison final metadata; these fields must never be selected."""
    with sqlite3.connect(path) as connection:
        connection.executescript("""
          CREATE TABLE episodes(id INTEGER, problem INTEGER, split TEXT, complete INTEGER);
          CREATE TABLE steps(episode_id INTEGER, step_index INTEGER, action TEXT,
            action_json TEXT, response_json TEXT, accepted INTEGER, attempts INTEGER,
            virtual_time_before_s REAL,virtual_time_after_s REAL,delta_virtual_time_s REAL,
            cost_components_json TEXT);
          CREATE TABLE source_estimates(SECRET_FINAL_TRUTH TEXT);
          INSERT INTO source_estimates VALUES ('DO NOT READ');
        """)
        for episode_id, problem, split, complete in ((1, 3, "train", 0),
                                                     (2, 4, "validation", 1),
                                                     (3, 3, "test", 1)):
            connection.execute("INSERT INTO episodes VALUES (?,?,?,?)",
                               (episode_id, problem, split, complete))
            for row in trajectory():
                connection.execute("INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    episode_id, row["step_index"], row["action"],
                    json.dumps(row["action_parameters"]), json.dumps(row["response"]),
                    row["accepted"], row["attempts"], row["virtual_time_before_s"],
                    row["virtual_time_after_s"], row["delta_virtual_time_s"],
                    json.dumps(row["cost_components"])))


def test_sqlite_loader_never_selects_labels_or_future_state(tmp_path):
    database = tmp_path / "artificial.sqlite3"
    write_db(database)
    statements = []
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.set_trace_callback(statements.append)
        rows = sqlite_steps(connection, 1)
        first = next(rows)
        assert first["action"] == "enter"
        assert "episode_complete" not in first
        assert "case_code" not in first
    assert len(statements) == 1
    assert "source_estimates" not in statements[0]
    assert "observations" not in statements[0]
    assert "episodes" not in statements[0]


def make_manifest(tmp_path, database):
    root = Path(__file__).resolve().parents[1]
    spec = {"entrypoint": "strategies.relocating_state_search:run_relocating_state_search",
            "kwargs": {}}
    return {"validation_role": "validation_only", "database_sha256": file_hash(database),
            "source_root": str(root),
            "source_sha256": {p.relative_to(root).as_posix(): file_hash(p)
                              for p in (root / "src").rglob("*.py")},
            "spec": spec, "spec_sha256": canonical_hash(spec)}


def test_manifest_fails_closed_for_changed_source_spec_or_database(tmp_path):
    database = tmp_path / "artificial.sqlite3"
    write_db(database)
    manifest = make_manifest(tmp_path, database)
    assert verify_manifest(manifest, database)[1] == manifest["spec"]
    changed = copy.deepcopy(manifest)
    changed["source_sha256"].pop(next(iter(changed["source_sha256"])))
    with pytest.raises(ValueError, match="source"):
        verify_manifest(changed, database)
    changed = copy.deepcopy(manifest)
    changed["spec"]["kwargs"]["injected"] = True
    with pytest.raises(ValueError, match="spec"):
        verify_manifest(changed, database)
    changed = copy.deepcopy(manifest)
    changed["database_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="snapshot"):
        verify_manifest(changed, database)
    changed = copy.deepcopy(manifest)
    changed["validation_role"] = "train"
    with pytest.raises(ValueError, match="validation-only"):
        verify_manifest(changed, database)


def test_cli_includes_all_q3_splits_and_incomplete_rows_without_scoring_them(tmp_path):
    database = tmp_path / "artificial.sqlite3"
    write_db(database)
    manifest = make_manifest(tmp_path, database)
    manifest_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "output.json"
    assert main(["--database", str(database), "--manifest", str(manifest_path),
                 "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(result["episodes"]) == 2
    assert all(row["status"] == "unsupported_counterfactual_action" for row in result["episodes"])
    assert all(row["performance_comparison_supported"] is False for row in result["episodes"])
    assert "source_estimates" not in output.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        main(["--database", str(database), "--manifest", str(manifest_path), "--output", str(output)])
