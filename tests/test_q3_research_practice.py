"""Practice-only preflight and an actual in-memory legal client session."""

from argparse import Namespace
from datetime import datetime
import json
from types import SimpleNamespace

import pytest

from experiments import run_q3_practice
from experiments.run_q3_practice import DeadlineClient, main, run
from simulation import LocalResearchSimulator, random_scenario
from simulator_client.errors import DeadlineExceeded


@pytest.fixture(autouse=True)
def fixed_predeadline_time(monkeypatch):
    monkeypatch.setattr(run_q3_practice.time, "time", lambda: 1789059600.)


def options(tmp_path, **overrides):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(dict(name="efficient", entrypoint="strategies:run_search",
                                   kwargs={"variant": "efficient"})), encoding="utf-8")
    return Namespace(spec=spec, max_actions=10000, robot_id="test-team", case_code="LOCAL-TEST",
                     dry_run=False, gui_practice_confirmed=False, output=tmp_path / "session",
                     base_url="http://127.0.0.1:2026", **overrides)


def test_unconfirmed_mode_is_rejected_before_client_creation(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("No client may be created without current practice evidence")
    args = options(tmp_path)
    with pytest.raises(ValueError, match="GUI confirmation"):
        run(args, forbidden)
    assert not args.output.exists()


def test_dry_run_requires_no_gui_and_sends_no_requests(tmp_path):
    args = options(tmp_path)
    args.dry_run = True
    result = run(args, lambda *a, **k: pytest.fail("dry run must not construct a client"))
    assert result["simulator_requests_sent"] is False
    assert not args.output.exists()


def test_runner_rejects_attempt_to_override_problem(tmp_path):
    args = options(tmp_path)
    spec = json.loads(args.spec.read_text())
    spec["kwargs"]["problem"] = 4
    args.spec.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="fixed at Q3"):
        run(args)
    with pytest.raises(SystemExit):
        main(["--spec", str(args.spec), "--mode", "formal"])


def test_local_client_completes_and_preserves_unverified_official_status(tmp_path):
    args = options(tmp_path)
    args.gui_practice_confirmed = True
    sim = LocalResearchSimulator(random_scenario(3, 100019))
    result = run(args, lambda *a, **k: sim.client())
    assert result["completed"] and sim.evaluation()["all_cleared"]
    assert result["official_result_verified"] is False
    saved = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert saved["case_code"] == "LOCAL-TEST"
    assert saved["identity"]["source_sha256"]
    assert saved["state"]["session"] == "exited"


def test_bounded_incomplete_run_is_not_reported_successful(tmp_path):
    args = options(tmp_path)
    args.gui_practice_confirmed = True
    args.max_actions = 2
    sim = LocalResearchSimulator(random_scenario(3, 100020))
    result = run(args, lambda *a, **k: sim.client())
    assert not result["completed"]
    assert result["state"]["session"] == "exited"
    assert not sim.evaluation()["all_cleared"]


def test_hard_cutoff_stops_actions_but_allows_exit(tmp_path):
    sim = LocalResearchSimulator(random_scenario(3, 100021))
    client = sim.client()
    client.enter()
    wrapped = DeadlineClient(client, 0)
    with pytest.raises(DeadlineExceeded):
        wrapped.measure((0, 0), 1)
    with pytest.raises(DeadlineExceeded):
        wrapped.clear((0, 0), 1)
    wrapped.exit()
    assert client.state.session == "exited"
    assert sim.evaluation()["measurement_count"] == 0


def test_expired_default_rejects_before_client_creation(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.gui_practice_confirmed = True
    monkeypatch.setattr(run_q3_practice.time, "time", lambda: datetime.fromisoformat(
        "2026-09-11T16:12:00+08:00").timestamp())
    with pytest.raises(ValueError, match="First-version execution deadline reached"):
        run(args, lambda *a, **k: pytest.fail("Expired default must not create a client"))
    assert not args.output.exists()
    args.dry_run = True
    result = run(args, lambda *a, **k: pytest.fail("Dry run must remain offline"))
    assert result["action_deadline_source"] == "protocol_minus_30_seconds"
    assert result["action_deadline"] == "2026-09-11T15:59:30+08:00"


@pytest.mark.parametrize("value, message", [
    ("2026-09-11T16:45:00", "explicit timezone"),
    ("2026-09-11T16:00:00+08:00", "in the future"),
    ("2026-09-11T16:12:00+08:00", "in the future"),
    ("not-a-date", "ISO8601"),
])
def test_invalid_explicit_deadline_rejects_without_client(tmp_path, monkeypatch, value, message):
    args = options(tmp_path, action_deadline=value)
    args.gui_practice_confirmed = True
    monkeypatch.setattr(run_q3_practice.time, "time", lambda: datetime.fromisoformat(
        "2026-09-11T16:12:00+08:00").timestamp())
    with pytest.raises(ValueError, match=message):
        run(args, lambda *a, **k: pytest.fail("Invalid deadline must not create a client"))
    assert not args.output.exists()


def test_explicit_new_window_runs_mock_and_logs_original_deadline(tmp_path, monkeypatch):
    new_deadline = "2026-09-11T16:45:00+08:00"
    args = options(tmp_path, action_deadline=new_deadline)
    args.gui_practice_confirmed = True
    monkeypatch.setattr(run_q3_practice.time, "time", lambda: datetime.fromisoformat(
        "2026-09-11T16:12:00+08:00").timestamp())

    class MockClient:
        def __init__(self):
            self.state = SimpleNamespace(session="idle")
            self.state.snapshot = lambda: {"session": self.state.session}
            self.pending_request = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def enter(self):
            self.state.session = "active"

        def exit(self):
            self.state.session = "exited"

    def callback(client, **kwargs):
        assert client.deadline == datetime.fromisoformat(new_deadline).timestamp()
        client.enter()
        client.exit()
        return SimpleNamespace(as_dict=lambda: {"mock": True}, error=None, exit_error=None,
                               completion_certified_under_model=True)

    monkeypatch.setattr(run_q3_practice, "prepare", lambda path: (
        {"name": "mock", "kwargs": {}}, callback, {"mock_identity": True}))
    result = run(args, lambda *a, **k: MockClient())
    assert result["completed"]
    for filename in ("preflight.json", "summary.json"):
        saved = json.loads((args.output / filename).read_text(encoding="utf-8"))
        assert saved["protocol_hard_deadline"] == "2026-09-11T16:00:00+08:00"
        assert saved["default_action_deadline"] == "2026-09-11T15:59:30+08:00"
        assert saved["action_deadline"] == new_deadline
        assert saved["action_deadline_override"] == new_deadline
        assert saved["action_deadline_source"] == "explicit_cli_override"


def test_explicit_window_expiring_during_prepare_never_creates_client(tmp_path, monkeypatch):
    deadline = datetime.fromisoformat("2026-09-11T16:45:00+08:00").timestamp()
    args = options(tmp_path, action_deadline="2026-09-11T16:45:00+08:00")
    args.gui_practice_confirmed = True
    monkeypatch.setattr(run_q3_practice.time, "time", lambda: deadline - 1)

    def prepare(path):
        monkeypatch.setattr(run_q3_practice.time, "time", lambda: deadline)
        return {}, None, {}

    monkeypatch.setattr(run_q3_practice, "prepare", prepare)
    with pytest.raises(ValueError, match="deadline reached during preflight"):
        run(args, lambda *a, **k: pytest.fail("Preparation must not bypass the new cutoff"))
    assert not args.output.exists()
