"""Practice-only preflight and an actual in-memory legal client session."""

from argparse import Namespace
import json

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
