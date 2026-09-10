"""Protocol/state tests against real localhost HTTP, not official practice."""

from concurrent.futures import ThreadPoolExecutor
import json
import math

import pytest

from simulator_client import (
    DeadlineExceeded, OutcomeUnknown, PendingActionError,
    Position, RequestRejected, SessionError, SimulatorClient,
)
from tests.fake_simulator import FakeSimulator
from simulator_client.__main__ import main, smoke_practice


@pytest.fixture
def simulator():
    with FakeSimulator() as instance:
        yield instance


@pytest.fixture
def client(simulator, tmp_path):
    with SimulatorClient(
        simulator.expected_robot_id, base_url=simulator.base_url,
        log_path=tmp_path / "session.jsonl", retry_backoff_s=0, timeout_s=1,
    ) as instance:
        yield instance


def test_attachment_timing_example_and_clear_does_not_switch(client, simulator):
    client.enter()
    assert client.state.position == Position(0, 0)
    assert client.state.current_channel == 1
    assert client.measure((300, 400), 1)["virtual_time_s"] == 105
    assert client.measure((300, 400), 2)["virtual_time_s"] == 111
    assert client.clear((300, 0), 3)["virtual_time_s"] == 194
    assert client.state.current_channel == 2
    assert client.measure((300, 0), 2)["virtual_time_s"] == 199
    assert client.exit()["virtual_time_s"] == 199
    costs = client.state.time_breakdown
    assert (costs.movement_s, costs.switching_s, costs.detection_s, costs.optical_s, costs.removal_s) == (
        180, 1, 15, 3, 0,
    )
    assert costs.total_s == client.state.virtual_time_s == simulator.state["virtual_time_s"]


def test_all_measure_and_clear_results_preserve_source_meaning(client):
    client.enter()
    result = client.measure((0, 0), 1)
    assert result["measure_result"] == "direction"
    assert result["svd_deg"] == pytest.approx(53.13)
    assert client.state.sources[1].status == "detected"
    result = client.measure((1800, 0), 1)
    assert result["measure_result"] == "no_signal"
    assert client.state.sources[1].status == "detected"  # signal loss is not absence
    result = client.measure((0, 0), 2)
    assert result["measure_result"] == "near" and "svd_deg" not in result
    assert client.clear((0, 0), 2)["clear_result"] == "success"
    assert client.clear((0, 0), 2)["clear_result"] == "no_target_in_range"
    assert client.measure((0, 0), 2)["measure_result"] == "no_signal"
    assert client.state.sources[2].status == "cleared"
    assert client.state.cleared_count == 1
    assert client.state.sources[2].failed_clear_count == 1
    client.exit()


def test_lost_clear_response_reuses_request_and_applies_state_once(client, simulator):
    client.enter()
    simulator.drop_once_paths.add("/clear")
    result = client.clear((0, 0), 2)
    assert result["clear_result"] == "success"
    requests = [payload for _, path, payload in simulator.requests if path == "/clear"]
    assert len(requests) == 2 and requests[0] == requests[1]
    assert simulator.state["clear_count"] == client.state.cleared_count == 1
    assert client.state.virtual_time_s == client.state.time_breakdown.total_s == 5
    assert client.state.accepted_actions == 2  # enter + one clear
    client.exit()


@pytest.mark.parametrize("status", [200, 400, 409, 429, 500])
def test_definite_rejection_does_not_reset_time_or_move(client, simulator, status):
    client.enter()
    client.measure((300, 400), 1)
    before = client.state.snapshot()
    simulator.reject_next(status)
    with pytest.raises(RequestRejected) as error:
        client.measure((1500, 1500), 8)
    assert error.value.status_code == status
    assert error.value.response["virtual_time_s"] == 0
    assert client.state.snapshot() == before
    assert client.pending_request is None
    client.measure((300, 400), 2)
    assert client.state.virtual_time_s == 111
    client.exit()


def test_unknown_outcome_blocks_new_actions_and_can_resume(simulator, tmp_path):
    with SimulatorClient(simulator.expected_robot_id, base_url=simulator.base_url,
                         log_path=tmp_path / "pending.jsonl", max_attempts=1) as client:
        client.enter()
        simulator.drop_once_paths.add("/clear")
        with pytest.raises(OutcomeUnknown):
            client.clear((0, 0), 2)
        original = client.pending_request
        assert simulator.state["clear_count"] == 1 and client.state.cleared_count == 0
        with pytest.raises(PendingActionError):
            client.measure((10, 10), 1)
        with pytest.raises(PendingActionError):
            client.exit()
        # Exposed request data cannot modify the actual pending body.
        original["payload"]["channel"] = 20
        response = client.retry_pending()
        assert response["clear_result"] == "success"
        assert client.state.cleared_count == 1
        assert client.pending_request is None
        client.exit()


def test_lost_enter_anchors_deadline_to_first_send(simulator, tmp_path):
    now = [100.0]
    simulator.remaining_real_duration_s = 20
    simulator.drop_once_paths.add("/enter")
    with SimulatorClient(simulator.expected_robot_id, base_url=simulator.base_url,
                         log_path=tmp_path / "enter.jsonl", max_attempts=1,
                         clock=lambda: now[0]) as client:
        with pytest.raises(OutcomeUnknown):
            client.enter()
        now[0] = 107.0
        client.retry_pending()
        assert client.state.real_deadline == 120.0
        assert client.remaining_real_time_s == 13.0
        now[0] = 120.0
        before = len(simulator.requests)
        with pytest.raises(DeadlineExceeded):
            client.measure((0, 0), 1)
        assert len(simulator.requests) == before


@pytest.mark.parametrize("status", [200, 429, 500])
def test_rejected_retry_cannot_erase_an_earlier_unknown_execution(simulator, tmp_path, status):
    with SimulatorClient(simulator.expected_robot_id, base_url=simulator.base_url,
                         log_path=tmp_path / "retry-rejected.jsonl", max_attempts=1) as client:
        client.enter()
        simulator.drop_once_paths.add("/clear")
        with pytest.raises(OutcomeUnknown):
            client.clear((300, 400), 1)
        simulator.reject_next(status)
        with pytest.raises(OutcomeUnknown):
            client.retry_pending()
        assert client.pending_request is not None
        assert client.state.virtual_time_s == 0
        with pytest.raises(PendingActionError):
            client.clear((300, 400), 1)
        assert client.retry_pending()["clear_result"] == "success"
        assert client.state.virtual_time_s == simulator.state["virtual_time_s"] == 105
        assert client.state.cleared_count == simulator.state["clear_count"] == 1
        client.exit()


@pytest.mark.parametrize("position,channel", [
    ((math.nan, 0), 1), ((math.inf, 0), 1), ((2_000_001, 0), 1),
    ((0, 0), 0), ((0, 0), 21), ((0, 0), 1.5), ((0, 0), True),
])
def test_invalid_actions_are_rejected_before_http(client, simulator, position, channel):
    client.enter()
    before = len(simulator.requests)
    with pytest.raises(ValueError):
        client.measure(position, channel)
    assert len(simulator.requests) == before


def test_lifecycle_and_new_request_ids(client, simulator):
    with pytest.raises(SessionError):
        client.measure((0, 0), 1)
    client.enter()
    with pytest.raises(SessionError):
        client.enter()
    client.measure((0, 0), 1)
    client.measure((0, 0), 1)
    client.exit()
    with pytest.raises(SessionError):
        client.measure((0, 0), 1)
    ids = [payload["request_id"] for _, _, payload in simulator.requests]
    assert len(ids) == len(set(ids)) == 4


def test_calls_from_threads_are_serialized(client, simulator):
    client.enter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(client.measure, (0, 0), channel) for channel in (1, 2)]
        assert all(f.result()["accepted"] for f in futures)
    assert client.state.virtual_time_s == simulator.state["virtual_time_s"]
    assert client.state.accepted_actions == 3
    client.exit()


def test_proxy_environment_is_not_used_for_local_simulator(simulator, tmp_path, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "")
    with SimulatorClient(simulator.expected_robot_id, base_url=simulator.base_url,
                         log_path=tmp_path / "direct.jsonl") as client:
        client.enter()
        client.exit()


def test_journal_records_attempts_responses_and_final_state(client, simulator):
    client.enter()
    simulator.drop_once_paths.add("/measure")
    client.measure((0, 0), 1)
    client.exit()
    client.close()
    events = [json.loads(line) for line in client.log_path.read_text().splitlines()]
    assert any(event["event"] == "attempt_failed" for event in events)
    assert events[-1]["event"] == "client_closed"
    assert events[-1]["state"]["session"] == "exited"
    assert events[-1]["state"]["virtual_time_s"] == 5
    requests = [event for event in events if event["event"] == "request" and event["path"] == "/measure"]
    assert requests[0]["payload"] == requests[1]["payload"]


@pytest.mark.parametrize("bad_result", ["unexpected", [], {}])
def test_malformed_accepted_response_is_not_applied(client, monkeypatch, bad_result):
    client.enter()
    before = client.state.snapshot()
    real_exchange = client._exchange

    def corrupt_exchange(action, timeout):
        status, response = real_exchange(action, timeout)
        response["measure_result"] = bad_result
        return status, response

    monkeypatch.setattr(client, "_exchange", corrupt_exchange)
    with pytest.raises(OutcomeUnknown):
        client.measure((0, 0), 1)
    assert client.state.snapshot() == before
    assert client.pending_request is not None
    monkeypatch.setattr(client, "_exchange", real_exchange)
    assert client.retry_pending()["measure_result"] == "direction"
    assert client.state.virtual_time_s == 5
    client.exit()


def test_smoke_cli_exercises_all_four_endpoints(simulator, tmp_path, capsys):
    journal = tmp_path / "cli.jsonl"
    assert main(["smoke-practice", "--robot-id", simulator.expected_robot_id,
                 "--base-url", simulator.base_url, "--log", str(journal)]) == 0
    report = json.loads(journal.with_suffix(".summary.json").read_text())
    assert report["completed"] is True
    assert report["state"]["virtual_time_s"] == 199
    assert {entry["path"] for entry in report["responses"]} == {
        "/enter", "/measure", "/clear", "/exit",
    }
    assert json.loads(capsys.readouterr().out)["completed"] is True


def test_connection_check_does_not_send_any_action(simulator, capsys):
    assert main(["check", "--base-url", simulator.base_url]) == 0
    assert simulator.requests == []
    assert "未发送任何动作" in capsys.readouterr().out


def test_smoke_does_not_send_exit_after_an_unknown_action(client, simulator):
    client.max_attempts = 1
    simulator.drop_once_paths.add("/measure")
    report = smoke_practice(client)
    assert report["completed"] is False
    assert report["pending_request"]["path"] == "/measure"
    assert all(path != "/exit" for _, path, _ in simulator.requests)
