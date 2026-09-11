"""Offline regression tests for practice lifecycle and HTTP ownership guards."""

import json
from types import SimpleNamespace

import pytest

from practice_control import BridgeError
from practice_control import runner
from simulator_client import SimulatorClient


CASE = "ABCD-EFGH-IJKL-MNOP"


def state(**changes):
    value = {"active": True, "mode": "practice", "problem_no": 3,
             "case_code": CASE, "phase": "waiting_enter", "entered": False,
             "api_open": True, "cleanup_complete": False, "log_package_status": ""}
    value.update(changes)
    return value


class FakeBridge:
    def __init__(self, current=None):
        self.state = current if current is not None else state()
        self.starts = []
        self.cleared = []

    def current_test(self):
        if isinstance(self.state, Exception):
            raise self.state
        return dict(self.state)

    def start_practice(self, problem):
        self.starts.append(problem)
        return dict(self.state)

    def clear_finished_test(self, case):
        assert self.state["phase"] == "ended"
        assert self.state["cleanup_complete"] is True
        assert self.state["case_code"] == case
        self.cleared.append(case)


@pytest.mark.parametrize("robot_id", ["x" * 65, "中" * 22, "abc\n123", "abc\x00123", "abc\u200b123", "   ", None])
def test_robot_validation_rejects_invalid_client_ids_before_start(robot_id):
    with pytest.raises(ValueError):
        runner.validate_run(3, None, 1, 100, robot_id)


@pytest.mark.parametrize("problem, variant, repeats, max_actions", [
    (True, None, 1, 100), (2, None, 1, 100), (3, "triangular", 1, 100),
    (4, "efficient", 1, 100), (3, None, True, 100), (3, None, 0, 100),
    (3, None, 1, True), (3, None, 1, 1),
])
def test_invalid_batch_arguments_are_rejected(problem, variant, repeats, max_actions):
    with pytest.raises(ValueError):
        runner.validate_run(problem, variant, repeats, max_actions, "TEST-ROBOT")


@pytest.mark.parametrize("existing", [
    state(mode="formal"), state(problem_no=4), state(case_code="DIFF-CASE-CODE-HERE"),
    state(active=False), state(api_open=False), state(phase="ended"),
    state(cleanup_complete=True), BridgeError("bridge disconnected"),
])
@pytest.mark.parametrize("path", ["/enter", "/measure", "/clear", "/exit"])
def test_every_http_action_checks_case_and_practice_state_before_sending(monkeypatch, tmp_path, existing, path):
    requests = []
    monkeypatch.setattr(SimulatorClient, "_exchange", lambda *args: requests.append(args))
    with runner.GuardedPracticeClient("TEST-ROBOT", bridge=FakeBridge(existing), problem=3,
                                      case=CASE, log_path=tmp_path / "journal.jsonl") as client:
        with pytest.raises(runner.PracticeOwnershipError):
            client._exchange(SimpleNamespace(path=path, had_unknown_attempt=False), 3)
    assert requests == []


def test_fresh_enter_refuses_an_already_entered_case(monkeypatch, tmp_path):
    requests = []
    monkeypatch.setattr(SimulatorClient, "_exchange", lambda *args: requests.append(args))
    with runner.GuardedPracticeClient("TEST-ROBOT", bridge=FakeBridge(state(entered=True, phase="running")),
                                      problem=3, case=CASE, log_path=tmp_path / "journal.jsonl") as client:
        with pytest.raises(runner.PracticeOwnershipError):
            client._exchange(SimpleNamespace(path="/enter", had_unknown_attempt=False), 3)
    assert requests == []


def test_own_pending_enter_can_resolve_uncertainty_with_same_action(monkeypatch, tmp_path):
    action = SimpleNamespace(path="/enter", had_unknown_attempt=True,
                             payload={"request_id": "same-id"})
    requests = []

    def exchange(self, pending, timeout):
        requests.append(pending)
        return 200, {"accepted": True}

    monkeypatch.setattr(SimulatorClient, "_exchange", exchange)
    with runner.GuardedPracticeClient("TEST-ROBOT", bridge=FakeBridge(state(entered=True, phase="running")),
                                      problem=3, case=CASE, log_path=tmp_path / "journal.jsonl") as client:
        client._pending = action
        assert client._exchange(action, 3) == (200, {"accepted": True})
        client._pending = None
    assert requests == [action]


def test_reused_http_transport_still_checks_state_before_every_request(monkeypatch, tmp_path):
    from practice_control import transport
    requests = []
    closed = []

    class Transport:
        def exchange(self, action, timeout):
            requests.append((action, timeout))
            return 200, {"accepted": True}

        def close(self):
            closed.append(True)

    monkeypatch.setattr(transport, "LoopbackHTTPTransport", Transport)
    control = FakeBridge(state(entered=True, phase="running"))
    action = SimpleNamespace(path="/measure", had_unknown_attempt=False)
    with runner.GuardedPracticeClient("TEST-ROBOT", bridge=control, problem=3,
                                      case=CASE, log_path=tmp_path / "journal.jsonl",
                                      keep_alive_http=True) as client:
        assert client._exchange(action, 3) == (200, {"accepted": True})
        control.state = state(mode="formal", entered=True, phase="running")
        with pytest.raises(runner.PracticeOwnershipError):
            client._exchange(action, 3)
    assert requests == [(action, 3)]
    assert closed


def test_direct_run_once_rejects_invalid_robot_before_creating_practice(tmp_path):
    control = FakeBridge()
    with pytest.raises(ValueError):
        runner.run_once(control, problem=3, robot_id="X" * 65, variant="efficient", max_actions=100,
                        output=tmp_path / "run", simulator_dir=tmp_path)
    assert control.starts == []


def test_preparation_without_stable_identity_stops_without_start_retry(tmp_path):
    control = FakeBridge(state(phase="preparing", case_code="", api_open=False))
    with pytest.raises(BridgeError, match="stable identifier"):
        runner.run_once(control, problem=3, robot_id="TEST-ROBOT", variant="efficient", max_actions=100,
                        output=tmp_path / "run", simulator_dir=tmp_path)
    assert control.starts == [3]
    assert (tmp_path / "run" / "preparation-unresolved.json").is_file()
    assert control.cleared == []


def test_preparation_does_not_adopt_another_run_of_same_problem(tmp_path):
    class ReplacedBridge(FakeBridge):
        def start_practice(self, problem):
            self.starts.append(problem)
            return state(phase="preparing", case_code="", practice_run_no=1, api_open=False)

    control = ReplacedBridge(state(practice_run_no=2))
    with pytest.raises(runner.PracticeOwnershipError):
        runner.run_once(control, problem=3, robot_id="TEST-ROBOT", variant="efficient", max_actions=100,
                        output=tmp_path / "run", simulator_dir=tmp_path)
    assert control.starts == [3]
    assert control.cleared == []


def test_controller_lock_blocks_parallel_controller_and_releases_cleanly(tmp_path):
    path = tmp_path / "controller.lock"
    with runner.controller_lock(path):
        with pytest.raises(BridgeError, match="Another practice controller"):
            with runner.controller_lock(path):
                pytest.fail("A second controller acquired the simulator lock")
    with runner.controller_lock(path):
        assert path.is_file()


def test_unexpected_solver_exception_saves_summary_and_finishes_owned_practice(monkeypatch, tmp_path):
    control = FakeBridge()

    class ClientState:
        session = "new"

        def snapshot(self):
            return {"session": self.session, "cleared_count": 0, "virtual_time_s": 0.0}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.state = ClientState()
            self.pending_request = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def exit(self):
            self.state.session = "exited"
            control.state = state(phase="ended", entered=True, api_open=False,
                                  cleanup_complete=True, log_package_status="ready")

    def broken_solver(client, **kwargs):
        client.state.session = "active"
        raise RuntimeError("solver regression")

    log_dir = tmp_path / "JammersSimulatorData" / "behavior-logs"
    log_dir.mkdir(parents=True)
    (log_dir / f"practice-p3-123-{CASE}.result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "GuardedPracticeClient", FakeClient)
    monkeypatch.setattr(runner, "revision", lambda: "test-revision")
    monkeypatch.setattr(runner, "working_tree_dirty", lambda: False)
    import experiments.register_practice
    monkeypatch.setattr(experiments.register_practice, "register", lambda args: {"source_total": 1})

    result = runner.run_once(control, problem=3, robot_id="TEST-ROBOT", variant="efficient", max_actions=100,
                             output=tmp_path / "run", simulator_dir=tmp_path, solver=broken_solver)
    report = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert report["completed"] is False
    assert report["error"] == "RuntimeError: solver regression"
    assert report["state"]["session"] == "exited"
    assert result["completed"] is False
    assert control.starts == [3]
    assert control.cleared == [CASE]
