"""Offline collector tests: quotas, recovery, throttling and safe stopping."""

import json
from pathlib import Path

import pytest

from practice_control import collect as module
from practice_control.bridge import BridgeError, MutationOutcomeUnknown, PracticeRequestFailed, UnsafeSimulatorState
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
        self.state = {"active": False, "mode": "", "phase": "", "case_code": "",
                      "api_open": False, "entered": False}

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


def reject_start(control, kwargs, *, code="server_timeout", extra_file=None, structured=True):
    control.start_practice(kwargs["problem"])
    error = PracticeRequestFailed(f"Practice start rejected: {code}", error_code=code if structured else None)
    folder = kwargs["output"]
    folder.mkdir()
    (folder / "start_error.json").write_text(
        json.dumps({"type": type(error).__name__, "error": str(error)}), encoding="utf-8")
    if extra_file:
        (folder / extra_file).write_text("{}", encoding="utf-8")
    raise error


def test_rejected_server_timeout_recovery_is_bounded_and_preserves_each_attempt(setup):
    folders = []

    def runner(control, **kwargs):
        folders.append(kwargs["output"])
        return reject_start(control, kwargs)

    with pytest.raises(PracticeRequestFailed):
        execute(setup, episode_runner=runner, q3=1, q4=0)
    assert setup["bridge"].starts == [(3, 5.0), (3, 15.0), (3, 35.0), (3, 75.0)]
    assert len(set(folders)) == 4
    assert all({p.name for p in folder.iterdir()} == {"start_error.json"} for folder in folders)
    progress = json.loads((setup["db"].parent / "training-raw" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "failed"
    assert progress["counts"] == {"3": 0, "4": 0}
    assert progress["start_recovery_count"] == 3
    assert len(progress["errors"]) == 4
    assert [r["wait_seconds"] for r in progress["recovery_events"]] == [10, 20, 40]
    assert setup["store"].import_calls == []


def test_successful_run_resets_consecutive_start_recovery_delay(setup):
    failed_problems = set()

    def runner(control, **kwargs):
        if kwargs["problem"] not in failed_problems:
            failed_problems.add(kwargs["problem"])
            return reject_start(control, kwargs)
        return setup["runner"](control, **kwargs)

    result = execute(setup, episode_runner=runner, q3=1, q4=1)
    assert result["status"] == "complete"
    assert result["counts"] == {"3": 1, "4": 1}
    assert result["consecutive_start_recoveries"] == 0
    assert [e["wait_seconds"] for e in result["recovery_events"]] == [10, 10]
    assert len(result["errors"]) == 2
    assert len(setup["bridge"].starts) == 4
    assert len(setup["store"].records) == 2


@pytest.mark.parametrize("extra", ["created.json", "requests.jsonl", "summary.json", "registration.json", "unknown.txt"])
def test_recovery_refuses_any_evidence_beyond_rejected_start(setup, extra):
    def runner(control, **kwargs):
        return reject_start(control, kwargs, extra_file=extra)

    with pytest.raises(PracticeRequestFailed):
        execute(setup, episode_runner=runner, q3=1, q4=0)
    assert len(setup["bridge"].starts) == 1


@pytest.mark.parametrize("code,structured", [("server_timeout", False), ("login_required", True),
                                            ("deadline_exceeded", True), ("request_failed", True)])
def test_recovery_allowlist_requires_exact_structured_server_timeout(setup, code, structured):
    def runner(control, **kwargs):
        return reject_start(control, kwargs, code=code, structured=structured)

    with pytest.raises(PracticeRequestFailed):
        execute(setup, episode_runner=runner, q3=1, q4=0)
    assert len(setup["bridge"].starts) == 1


@pytest.mark.parametrize("state", [
    {"active": True, "mode": "practice", "phase": "running", "case_code": "OTHER"},
    {"active": False, "mode": "practice", "phase": "ended", "case_code": "OTHER"},
    {"active": False, "mode": "formal", "phase": "", "case_code": ""},
    {"active": False, "mode": "", "phase": "", "case_code": "", "api_open": True},
    {"active": False, "mode": "", "phase": "", "case_code": "", "entered": True},
    {"active": False, "mode": "", "phase": "", "case_code": "", "api_open": False},
    {"active": False, "mode": "", "phase": "", "case_code": "", "entered": False},
    {"active": False, "mode": "", "phase": "", "case_code": "", "api_open": None, "entered": False},
    {"active": False, "mode": "", "phase": "", "case_code": "", "api_open": False, "entered": None},
])
def test_recovery_requires_idle_immediately_after_rejection(setup, state):
    def runner(control, **kwargs):
        try:
            return reject_start(control, kwargs)
        finally:
            setup["bridge"].state = state

    with pytest.raises(BridgeError):
        execute(setup, episode_runner=runner, q3=1, q4=0)
    assert len(setup["bridge"].starts) == 1
    assert setup["clock"].now == 5


def test_recovery_checks_idle_again_after_waiting(setup, monkeypatch):
    clock = setup["clock"]
    original_sleep = clock.sleep

    def sleep(seconds):
        original_sleep(seconds)
        if clock.now >= 8:
            setup["bridge"].state = {"active": True, "mode": "practice", "phase": "running", "case_code": "OTHER"}

    monkeypatch.setattr(module.time, "sleep", sleep)
    with pytest.raises(BridgeError):
        execute(setup, episode_runner=lambda control, **kwargs: reject_start(control, kwargs), q3=1, q4=0)
    assert len(setup["bridge"].starts) == 1
    progress = json.loads((setup["db"].parent / "training-raw" / "progress.json").read_text(encoding="utf-8"))
    assert progress["recovery_events"][-1]["status"] == "aborted"


def test_stop_file_interrupts_recovery_wait_without_new_case(setup, monkeypatch):
    stop = setup["db"].parent / "stop-recovery"
    clock = setup["clock"]
    original_sleep = clock.sleep

    def sleep(seconds):
        original_sleep(seconds)
        if clock.now >= 7:
            stop.touch()

    monkeypatch.setattr(module.time, "sleep", sleep)
    result = execute(setup, episode_runner=lambda control, **kwargs: reject_start(control, kwargs),
                     q3=1, q4=0, stop_file=stop)
    assert result["status"] == "paused"
    assert result["stop_reason"] == "stop_file"
    assert result["recovery_events"][-1]["status"] == "paused"
    assert len(setup["bridge"].starts) == 1


@pytest.mark.parametrize("exception", [MutationOutcomeUnknown("server_timeout"), BridgeError("connection timeout")])
def test_transport_uncertainty_never_uses_rejected_start_recovery(setup, exception):
    def runner(control, **kwargs):
        control.start_practice(kwargs["problem"])
        kwargs["output"].mkdir()
        (kwargs["output"] / "start_error.json").write_text(
            json.dumps({"type": type(exception).__name__, "error": str(exception)}), encoding="utf-8")
        raise exception

    with pytest.raises(type(exception)):
        execute(setup, episode_runner=runner, q3=1, q4=0)
    assert len(setup["bridge"].starts) == 1


def test_cooperative_holds_writer_lock_but_yields_simulator_between_episodes(setup, monkeypatch):
    original_sleep = setup["clock"].sleep
    yielded = []
    raw = setup["db"].parent / "training-raw"
    lock = setup["simulator"] / ".practice-control" / "controller.lock"

    def sleep(seconds):
        if len(setup["runs"]) == 1:
            with controller_lock(lock):
                yielded.append(setup["clock"].now)
            with pytest.raises(BridgeError):
                with controller_lock(raw / "collector.lock"):
                    pytest.fail("Dataset writer lock was released while yielding")
        original_sleep(seconds)

    monkeypatch.setattr(module.time, "sleep", sleep)
    result = execute(setup, cooperative=True, q3=1, q4=1)
    assert result["status"] == "complete"
    assert setup["bridge"].starts == [(3, 5.0), (4, 10.0)]
    assert len(yielded) == 10  # Exactly five seconds; no second redundant wait.
    assert result["cooperative"] is True
    events = [json.loads(line) for line in (raw / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e["status"] == "waiting_for_simulator" and e["waiting"]["reason"] == "yield_after_episode" for e in events)


def test_cooperative_simulator_lock_is_held_through_dataset_import(setup):
    original_import = setup["store"].import_episode

    def importer(db, episode):
        with pytest.raises(BridgeError):
            with controller_lock(setup["simulator"] / ".practice-control" / "controller.lock"):
                pytest.fail("Another process could acquire the simulator during import")
        return original_import(db, episode)

    result = execute(setup, cooperative=True, importer=importer, q3=1, q4=0)
    assert result["status"] == "complete"


def test_cooperative_waits_for_other_practice_then_requires_idle_interval(setup, monkeypatch):
    setup["bridge"].state = {"active": True, "mode": "practice", "problem_no": 4,
                             "phase": "running", "case_code": "ABCD-EFGH-IJKL-MNOP"}
    original_sleep = setup["clock"].sleep

    def sleep(seconds):
        original_sleep(seconds)
        if setup["clock"].now >= 10:
            setup["bridge"].state = {"active": False, "mode": "", "case_code": "", "phase": "",
                                     "api_open": False, "entered": False}

    monkeypatch.setattr(module.time, "sleep", sleep)
    result = execute(setup, cooperative=True, q3=1, q4=0)
    assert result["status"] == "complete"
    assert setup["bridge"].starts == [(3, 15.0)]


def test_same_dataset_writer_lock_refuses_a_second_collector_without_simulator_contact(setup):
    with controller_lock(setup["db"].parent / "training-raw" / "collector.lock"):
        with pytest.raises(BridgeError):
            execute(setup, cooperative=True, resume=True)
    assert setup["bridge"].starts == []


def test_cooperative_backoff_releases_simulator_and_rechecks_idle_before_restart(setup, monkeypatch):
    failed = False
    original_sleep = setup["clock"].sleep
    unlocked_waits = []

    def runner(control, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            return reject_start(control, kwargs)
        return setup["runner"](control, **kwargs)

    def sleep(seconds):
        if failed and 5 <= setup["clock"].now < 15:
            with controller_lock(setup["simulator"] / ".practice-control" / "controller.lock"):
                unlocked_waits.append(setup["clock"].now)
        original_sleep(seconds)

    monkeypatch.setattr(module.time, "sleep", sleep)
    result = execute(setup, cooperative=True, episode_runner=runner, q3=1, q4=0)
    assert result["status"] == "complete"
    assert setup["bridge"].starts == [(3, 5.0), (3, 20.0)]
    assert len(unlocked_waits) == 20
    assert result["recovery_events"][0]["status"] == "ready"


def test_cooperative_unknown_metadata_before_episode_is_fatal(setup):
    def invalid_state():
        raise UnsafeSimulatorState("The simulator returned an unrecognized state")

    setup["bridge"].current_test = invalid_state
    with pytest.raises(UnsafeSimulatorState):
        execute(setup, cooperative=True, q3=1, q4=0)
    assert setup["clock"].now == 0
    assert setup["bridge"].starts == []


@pytest.mark.parametrize("message,extra_file", [
    ("The simulator returned an unrecognized state", None),
    ("A session is already active; it will not be replaced", "created.json"),
])
def test_cooperative_unknown_or_started_evidence_does_not_become_retry(setup, message, extra_file):
    def runner(control, **kwargs):
        error = UnsafeSimulatorState(message)
        kwargs["output"].mkdir()
        (kwargs["output"] / "start_error.json").write_text(
            json.dumps({"type": type(error).__name__, "error": str(error)}), encoding="utf-8")
        if extra_file:
            (kwargs["output"] / extra_file).write_text("{}", encoding="utf-8")
        raise error

    with pytest.raises(UnsafeSimulatorState):
        execute(setup, cooperative=True, episode_runner=runner, q3=1, q4=0)
    assert setup["clock"].now == 0


def test_cooperative_definite_prestart_guard_race_yields_without_counting_a_case(setup):
    attempts = []

    def runner(control, **kwargs):
        attempts.append(kwargs["output"])
        if len(attempts) == 1:
            error = UnsafeSimulatorState("The simulator state changed before practice could start")
            kwargs["output"].mkdir()
            (kwargs["output"] / "start_error.json").write_text(
                json.dumps({"type": type(error).__name__, "error": str(error)}), encoding="utf-8")
            raise error
        return setup["runner"](control, **kwargs)

    result = execute(setup, cooperative=True, episode_runner=runner, q3=1, q4=0)
    assert result["status"] == "complete"
    assert result["counts"] == {"3": 1, "4": 0}
    assert setup["bridge"].starts == [(3, 15.0)]
    assert attempts[0] != attempts[1]
    assert result["coordination_events"][0]["action"] == "yield_after_start_race"
