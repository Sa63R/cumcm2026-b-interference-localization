"""Offline collector tests: quotas, recovery, throttling and safe stopping."""

import json
from pathlib import Path

import pytest

from practice_control import collect as module
from practice_control.bridge import BridgeError, MutationOutcomeUnknown
from practice_control.runner import controller_lock


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Bridge:
    def __init__(self, clock):
        self.clock = clock
        self.starts = []
        self.state = {"active": False, "mode": "", "phase": "", "case_code": ""}

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        pass

    def current_test(self):
        return dict(self.state)

    def start_practice(self, problem):
        self.starts.append((problem, self.clock.now))
        return {"problem_no": problem}

    def clear_finished_test(self, case):
        raise AssertionError("Collector must not clear cases outside run_once")


class Store:
    def __init__(self):
        self.records = {}
        self.import_calls = []

    def import_episode(self, db, folder):
        self.import_calls.append(folder)
        data = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        inserted = data["case_code"] not in self.records
        self.records[data["case_code"]] = data
        Path(db).touch(exist_ok=True)
        return {"inserted": inserted, "complete": data["completed"],
                "case_code": data["case_code"], "problem": data["problem"]}

    def stats(self, db):
        return {"episodes": len(self.records),
                "episodes_by_problem": {str(p): sum(r["problem"] == p for r in self.records.values()) for p in (3, 4)},
                "complete_by_problem": {
                    str(p): sum(r["problem"] == p and r["completed"] for r in self.records.values()) for p in (3, 4)
                }}


def saved_episode(folder, case, problem, *, complete=True, error=None):
    folder.mkdir(parents=True)
    (folder / "registration.json").write_text("{}", encoding="utf-8")
    summary = {"problem": problem, "case_code": case, "completed": complete, "error": error}
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


@pytest.fixture
def setup(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    bridge = Bridge(clock)
    store = Store()
    simulator = tmp_path / "simulator"
    simulator.mkdir()
    (simulator / "jammers-simulator-full.exe").touch()
    db = tmp_path / "training.sqlite3"
    runs = []

    def runner(control, **kwargs):
        control.start_practice(kwargs["problem"])
        runs.append(kwargs)
        # This must be the same lock used by all standalone practice commands.
        with pytest.raises(BridgeError, match="Another practice controller"):
            with controller_lock(simulator / ".practice-control" / "controller.lock"):
                pytest.fail("Collector released its simulator lock between cases")
        return saved_episode(kwargs["output"], f"CASE-{len(runs)}", kwargs["problem"])

    return dict(clock=clock, bridge=bridge, store=store, db=db, simulator=simulator,
                runs=runs, runner=runner)


def execute(setup, **kwargs):
    defaults = dict(output=setup["db"], simulator_dir=setup["simulator"], robot_id="TEST-ROBOT",
                    q3=2, q4=2, bridge_factory=lambda port: setup["bridge"],
                    episode_runner=setup["runner"], importer=setup["store"].import_episode,
                    statistics=setup["store"].stats)
    defaults.update(kwargs)
    return module.collect(**defaults)


def test_balanced_complete_quotas_spacing_and_lock_held(setup):
    result = execute(setup)
    assert result["status"] == "complete"
    assert result["counts"] == {"3": 2, "4": 2}
    assert [p for p, when in setup["bridge"].starts] == [3, 4, 3, 4]
    times = [when for p, when in setup["bridge"].starts]
    assert all(b - a >= 5 for a, b in zip(times, times[1:]))
    assert [r["variant"] for r in setup["runs"]] == ["efficient", "triangular", "efficient", "triangular"]
    progress = json.loads((setup["db"].parent / "training-raw" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete" and progress["pid"] > 0
    assert setup["store"].stats(setup["db"])["episodes"] == 4


def test_resume_deduplicates_saved_import_and_fills_only_remaining_quota(setup):
    saved_episode(setup["db"].parent / "training-raw" / "episodes" / "old-case", "CASE-old", 3)
    result = execute(setup, resume=True, q3=1, q4=1)
    assert result["counts"] == {"3": 1, "4": 1}
    assert [p for p, when in setup["bridge"].starts] == [4]
    assert result["recovered_episodes"] == 1
    execute(setup, resume=True, q3=1, q4=1,
            bridge_factory=lambda port: pytest.fail("No simulator connection needed when all quotas are filled"))
    assert setup["store"].stats(setup["db"])["episodes"] == 2


def test_resume_can_explicitly_raise_targets_without_overcollection(setup):
    execute(setup, q3=1, q4=1)
    result = execute(setup, resume=True, q3=2, q4=2)
    assert result["counts"] == {"3": 2, "4": 2}
    assert len(setup["runs"]) == 4
    with pytest.raises(ValueError, match="smaller"):
        execute(setup, resume=True, q3=1, q4=1)
    assert len(setup["runs"]) == 4


def test_existing_output_requires_explicit_resume_before_simulator_contact(setup):
    setup["db"].touch()
    with pytest.raises(ValueError, match="--resume"):
        execute(setup)
    assert setup["bridge"].starts == []


@pytest.mark.parametrize("state", [
    {"active": True, "mode": "practice", "phase": "running", "case_code": "OTHER"},
    {"active": False, "mode": "formal", "phase": "", "case_code": ""},
    {"active": False, "mode": "practice", "phase": "ended", "case_code": "OTHER"},
])
def test_existing_cases_are_never_adopted_cleared_or_replaced(setup, state):
    setup["bridge"].state = state
    with pytest.raises(BridgeError):
        execute(setup)
    assert setup["bridge"].starts == []


def test_stop_file_finishes_and_imports_current_case_before_pausing(setup):
    stop = setup["db"].parent / "please-stop"

    def runner(control, **kwargs):
        result = setup["runner"](control, **kwargs)
        stop.touch()
        return result

    result = execute(setup, episode_runner=runner, stop_file=stop)
    assert result["status"] == "paused" and result["stop_reason"] == "stop_file"
    assert result["counts"] == {"3": 1, "4": 0}
    assert len(setup["runs"]) == 1


def test_time_budget_finishes_current_case_and_does_not_start_next(setup):
    def runner(control, **kwargs):
        result = setup["runner"](control, **kwargs)
        setup["clock"].now += 10
        return result

    result = execute(setup, episode_runner=runner, max_hours=6 / 3600)
    assert result["status"] == "paused" and result["stop_reason"] == "max_hours"
    assert len(setup["runs"]) == 1


def test_uncertain_start_is_not_retried_and_failure_is_saved(setup):
    def runner(control, **kwargs):
        control.start_practice(kwargs["problem"])
        kwargs["output"].mkdir()
        raise MutationOutcomeUnknown("request may have executed")

    with pytest.raises(MutationOutcomeUnknown):
        execute(setup, episode_runner=runner)
    assert len(setup["bridge"].starts) == 1
    progress = json.loads((setup["db"].parent / "training-raw" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["errors"][-1]["type"] == "MutationOutcomeUnknown"


def test_incomplete_data_is_kept_outside_quota_and_three_consecutive_stop(setup):
    sequence = []

    def runner(control, **kwargs):
        control.start_practice(kwargs["problem"])
        sequence.append(kwargs["problem"])
        return saved_episode(kwargs["output"], f"incomplete-{len(sequence)}", kwargs["problem"],
                             complete=False, error="SearchIncomplete: action budget reached")

    with pytest.raises(BridgeError, match="Three consecutive incomplete"):
        execute(setup, episode_runner=runner)
    assert len(sequence) == 3
    assert setup["store"].stats(setup["db"])["episodes"] == 3
    assert setup["store"].stats(setup["db"])["complete_by_problem"] == {"3": 0, "4": 0}


def test_solver_regression_is_saved_then_stops_without_second_scene(setup):
    def runner(control, **kwargs):
        control.start_practice(kwargs["problem"])
        return saved_episode(kwargs["output"], "broken", kwargs["problem"],
                             complete=False, error="ValueError: invalid model")

    with pytest.raises(BridgeError, match="Solver failure"):
        execute(setup, episode_runner=runner)
    assert len(setup["bridge"].starts) == 1
    assert setup["store"].stats(setup["db"])["episodes"] == 1


def test_native_readiness_countdown_is_waited_without_lifecycle_retry(setup):
    native = setup["bridge"]
    old_current = native.current_test

    def current():
        return {**old_current(), "countdown_remaining_ms": max(0, 8000 - setup["clock"].now * 1000)}

    native.current_test = current
    control = module.SpacedPracticeBridge(native)
    control.start_practice(3)
    assert native.starts == [(3, 8.0)]


def test_stop_during_throttle_prevents_next_start(setup):
    native = setup["bridge"]
    control = module.SpacedPracticeBridge(native, should_stop=lambda: bool(native.starts) and setup["clock"].now >= 7)
    control.start_practice(3)
    with pytest.raises(module.CollectionStopped):
        control.start_practice(4)
    assert native.starts == [(3, 5.0)]


@pytest.mark.parametrize("arguments", [{"q3": True}, {"q4": -1}, {"q3": 0, "q4": 0},
                                      {"max_hours": float("nan")}, {"max_hours": 0}])
def test_invalid_limits_never_start_practice(setup, arguments):
    with pytest.raises(ValueError):
        execute(setup, **arguments)
    assert setup["bridge"].starts == []
