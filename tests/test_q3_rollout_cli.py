"""Experimental Q3 configuration must be checked before entering a session."""

import json

import pytest

from tests.fake_simulator import FakeSimulator
from workflow.__main__ import main


def test_cli_rejects_rollout_for_q4_before_any_request(tmp_path):
    destination = tmp_path / "should-not-exist"
    with FakeSimulator() as simulator:
        with pytest.raises(SystemExit) as error:
            main(["run", "--problem", "4", "--mode", "practice", "--robot-id", "test-team",
                  "--variant", "rollout", "--base-url", simulator.base_url,
                  "--output", str(destination)])
        assert error.value.code == 2
        assert simulator.requests == []
    assert not destination.exists()


def test_cli_rejects_invalid_planning_parameters_before_any_request(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"max_planning_s": -1}', encoding="utf-8")
    with FakeSimulator() as simulator:
        with pytest.raises(SystemExit) as error:
            main(["run", "--problem", "3", "--mode", "practice", "--robot-id", "test-team",
                  "--variant", "rollout", "--rollout-config", str(config),
                  "--base-url", simulator.base_url, "--output", str(tmp_path / "session")])
        assert error.value.code == 2
        assert simulator.requests == []


def test_cli_preserves_explicit_rollout_parameters_and_bounded_exit(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"particles": 4, "candidates": 5, "max_searches": 0}', encoding="utf-8")
    destination = tmp_path / "session"
    with FakeSimulator() as simulator:
        code = main(["run", "--problem", "3", "--mode", "practice", "--robot-id", "test-team",
                     "--variant", "rollout", "--rollout-config", str(config),
                     "--base-url", simulator.base_url, "--max-actions", "2",
                     "--output", str(destination)])
        assert code == 1  # Safe exit with incomplete work is not a success.
        assert [request[1] for request in simulator.requests] == ["/enter", "/exit"]
    report = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert report["variant"] == report["search"]["variant"] == "rollout"
    assert report["search"]["strategy_parameters"]["particles"] == 4
    assert report["search"]["strategy_parameters"]["candidates"] == 5
    assert report["search"]["planning"]["candidate_evaluations"] == 0
    assert report["state"]["session"] == "exited"
