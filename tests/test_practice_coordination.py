"""Offline cooperative turns: real temporary OS locks, fake lifecycle metadata."""

from copy import deepcopy

import pytest

from practice_control import coordination as module
from practice_control.bridge import BridgeError
from practice_control.runner import controller_lock


CASE_A = "AAAA-BBBB-CCCC-DDDD"
CASE_B = "EEEE-FFFF-GGGG-HHHH"


def idle():
    return {"active": False, "mode": "", "phase": "", "case_code": "",
            "api_open": False, "entered": False}


def practice(**changes):
    state = {"active": True, "mode": "practice", "problem_no": 3,
             "practice_run_no": 7, "case_code": CASE_A, "phase": "running",
             "api_open": True, "entered": True, "cleanup_complete": False,
             "log_package_status": "pending", "log_file_name": "",
             "log_package_bytes": 0}
    state.update(changes)
    return state


def finished(**changes):
    state = practice(active=False, phase="ended", api_open=False, entered=False,
                     cleanup_complete=True, log_package_status="ready",
                     log_file_name="practice-p3-example.jlog", log_package_bytes=1234)
    state.update(changes)
    return state


def assert_free(path):
    with controller_lock(path):
        pass


def assert_held(path):
    with pytest.raises(BridgeError, match="Another practice controller holds"):
        with controller_lock(path):
            pytest.fail("The simulator lock was unexpectedly available")


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        assert 0 < seconds <= 0.5
        self.sleeps.append(seconds)
        self.now += seconds
        assert self.now <= 120, "Unexpected endless coordination wait"


class MetadataBridge:
    """Deliberately has no methods to start, solve or read another case's data."""

    def __init__(self, clock, lock_path):
        self.clock = clock
        self.lock_path = lock_path
        self.state = idle()
        self.queries = []
        self.clears = []
        self.query_error = None
        self.after_clear = None

    def current_test(self):
        assert_held(self.lock_path)
        self.queries.append(self.clock.now)
        if self.query_error:
            raise self.query_error
        return deepcopy(self.state)

    def clear_finished_test(self, expected_case_code):
        assert_held(self.lock_path)
        assert expected_case_code == self.state["case_code"]
        assert self.state["mode"] == "practice"
        assert self.state["phase"] == "ended"
        assert self.state["cleanup_complete"] is True
        assert self.state["log_package_status"] == "ready"
        self.clears.append((self.clock.now, expected_case_code))
        self.state = deepcopy(self.after_clear) if self.after_clear is not None else idle()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    path = tmp_path / "controller.lock"
    bridge = MetadataBridge(clock, path)
    stopped = [False]
    waits, events = [], []

    def on_wait(reason, seconds, state):
        # Waiting must leave the shared simulator available to another task.
        assert_free(path)
        waits.append((clock.now, reason, seconds, deepcopy(state)))

    turns = module.CooperativeTurns(
        bridge, path, should_stop=lambda: stopped[0], on_wait=on_wait,
        on_event=lambda event: events.append(deepcopy(event)))
    return dict(clock=clock, path=path, bridge=bridge, stopped=stopped,
                waits=waits, events=events, turns=turns, on_wait=on_wait)


@pytest.mark.parametrize("body_error", [False, True])
def test_turn_holds_real_lock_until_body_exits(rig, body_error):
    def body():
        with rig["turns"].turn():
            assert_held(rig["path"])
            assert rig["turns"].idle_wait_required is False
            rig["clock"].sleep(0.5)
            assert_held(rig["path"])
            if body_error:
                raise RuntimeError("solver failed")

    if body_error:
        with pytest.raises(RuntimeError, match="solver failed"):
            body()
    else:
        body()
    assert_free(rig["path"])
    assert rig["bridge"].queries == [0]
    assert not rig["bridge"].clears
    assert not rig["waits"]


def test_busy_controller_is_not_queried_and_gets_ten_seconds_to_release(rig):
    other = controller_lock(rig["path"])
    other.__enter__()

    def release_other(reason, seconds, state):
        assert (reason, seconds, state) == ("controller_lock_busy", 10, None)
        assert_held(rig["path"])
        assert rig["bridge"].queries == []
        other.__exit__(None, None, None)
        rig["on_wait"](reason, seconds, state)

    rig["turns"].on_wait = release_other
    with rig["turns"].turn():
        assert rig["clock"].now == 10
        assert_held(rig["path"])
        assert rig["turns"].idle_wait_required is True
    assert_free(rig["path"])
    assert rig["bridge"].queries == [10]


@pytest.mark.parametrize("problem", [3, 4])
@pytest.mark.parametrize("phase", ["preparing", "countdown", "waiting_enter", "running", "ending"])
def test_other_active_practice_releases_lock_while_waiting(rig, problem, phase):
    state = practice(problem_no=problem, phase=phase)
    rig["bridge"].state = state

    def become_idle(reason, seconds, observed):
        rig["on_wait"](reason, seconds, observed)
        assert (reason, seconds, observed) == ("other_practice_in_progress", 10, state)
        rig["bridge"].state = idle()

    rig["turns"].on_wait = become_idle
    with rig["turns"].turn():
        assert rig["clock"].now == 10
        assert rig["turns"].idle_wait_required is True
    assert not rig["bridge"].clears
    assert rig["bridge"].queries == [0, 10]


@pytest.mark.parametrize("state", [
    finished(cleanup_complete=False),
    finished(log_package_status="pending"),
    finished(log_package_status="failed"),
    finished(api_open=True),
    finished(api_open=None),
    {key: value for key, value in finished().items() if key != "api_open"},
    finished(case_code="invalid-case"),
    finished(case_code=""),
])
def test_incomplete_or_unidentified_finished_practice_is_never_cleared(rig, state):
    rig["bridge"].state = state

    def stop_waiting(reason, seconds, observed):
        rig["on_wait"](reason, seconds, observed)
        assert reason == "other_practice_in_progress"
        rig["stopped"][0] = True

    rig["turns"].on_wait = stop_waiting
    with pytest.raises(module.CollectionStopped):
        with rig["turns"].turn():
            pytest.fail("An incomplete finished case was adopted")
    assert not rig["bridge"].clears
    assert_free(rig["path"])


@pytest.mark.parametrize("state", [
    {**idle(), "mode": "formal"},
    practice(mode="formal"),
    finished(mode="formal"),
])
def test_formal_metadata_only_waits_until_stopped_and_is_never_cleared(rig, state):
    rig["bridge"].state = state
    rig["turns"].finished = ("previous",)
    rig["turns"].finished_since = -100

    def stop_after_poll(reason, seconds, observed):
        rig["on_wait"](reason, seconds, observed)
        assert reason == "formal_session_present"
        if rig["clock"].now >= 10:
            rig["stopped"][0] = True

    rig["turns"].on_wait = stop_after_poll
    with pytest.raises(module.CollectionStopped):
        with rig["turns"].turn():
            pytest.fail("Formal metadata was accepted for a practice turn")
    assert rig["bridge"].queries == [0, 10]
    assert not rig["bridge"].clears
    assert rig["turns"].finished is None
    assert rig["turns"].finished_since is None
    assert_free(rig["path"])


@pytest.mark.parametrize("problem", [3, 4])
def test_finished_practice_waits_fifteen_seconds_and_clears_exact_case_metadata(rig, problem):
    state = finished(problem_no=problem)
    rig["bridge"].state = state
    with rig["turns"].turn():
        assert_held(rig["path"])
        assert rig["clock"].now >= 15
        assert rig["turns"].idle_wait_required is True
    assert rig["bridge"].clears == [(20, CASE_A)]
    assert [(when, reason, seconds) for when, reason, seconds, _ in rig["waits"]] == [
        (0, "finished_practice_stabilizing", 10),
        (10, "finished_practice_stabilizing", 10),
    ]
    assert [event["status"] for event in rig["events"]] == ["requested", "confirmed"]
    for event in rig["events"]:
        assert event == {"action": "clear_stable_finished_practice", "state": state,
                         "stable_seconds": 20, "status": event["status"]}
    assert rig["bridge"].state == idle()
    assert rig["turns"].finished is None
    assert rig["turns"].finished_since is None
    assert {path.name for path in rig["path"].parent.iterdir()} == {"controller.lock"}


def test_ended_entered_flag_can_be_true_once_api_is_closed(rig):
    rig["bridge"].state = finished(entered=True)
    with rig["turns"].turn():
        assert rig["bridge"].state == idle()
    assert rig["bridge"].clears == [(20, CASE_A)]


def test_idle_wait_flag_is_reset_for_next_uncontested_turn(rig):
    rig["bridge"].state = finished()
    with rig["turns"].turn():
        assert rig["turns"].idle_wait_required is True
    with rig["turns"].turn():
        assert rig["turns"].idle_wait_required is False
    assert rig["bridge"].clears == [(20, CASE_A)]


@pytest.mark.parametrize("changes", [
    {"case_code": CASE_B},
    {"problem_no": 4},
    {"log_file_name": "practice-p3-new.jlog"},
    {"log_package_bytes": 5678},
    {"practice_run_no": 8},
])
def test_changed_case_or_log_metadata_restarts_stability_window(rig, changes):
    rig["bridge"].state = finished()

    def change_metadata_once(reason, seconds, state):
        rig["on_wait"](reason, seconds, state)
        if rig["clock"].now == 0:
            rig["bridge"].state.update(changes)
        if rig["clock"].now == 20:
            assert not rig["bridge"].clears

    rig["turns"].on_wait = change_metadata_once
    with rig["turns"].turn():
        assert rig["clock"].now == 30
    assert rig["bridge"].clears == [(30, changes.get("case_code", CASE_A))]
    assert [event["stable_seconds"] for event in rig["events"]] == [20, 20]


def test_busy_lock_resets_observed_finished_stability(rig):
    rig["bridge"].state = finished()
    other = None

    def introduce_lock_contention(reason, seconds, state):
        nonlocal other
        if reason == "controller_lock_busy":
            assert rig["clock"].now == 10
            assert rig["turns"].finished is None
            assert rig["turns"].finished_since is None
            other.__exit__(None, None, None)
            other = None
        rig["on_wait"](reason, seconds, state)
        if rig["clock"].now == 0:
            other = controller_lock(rig["path"])
            other.__enter__()

    rig["turns"].on_wait = introduce_lock_contention
    try:
        with rig["turns"].turn():
            assert rig["clock"].now == 40
    finally:
        if other is not None:
            other.__exit__(None, None, None)
    assert rig["bridge"].clears == [(40, CASE_A)]
    assert rig["bridge"].queries == [0, 20, 30, 40, 40]
    assert [reason for _, reason, _, _ in rig["waits"]] == [
        "finished_practice_stabilizing", "controller_lock_busy",
        "finished_practice_stabilizing", "finished_practice_stabilizing",
    ]


def test_stop_before_turn_does_not_query_or_acquire(rig):
    rig["stopped"][0] = True
    with pytest.raises(module.CollectionStopped, match="before acquiring"):
        with rig["turns"].turn():
            pytest.fail("A turn began despite a pause request")
    assert not rig["bridge"].queries
    assert not rig["path"].exists()


def test_stop_interrupts_wait_with_lock_available(rig):
    rig["bridge"].state = practice()
    rig["turns"].should_stop = lambda: rig["clock"].now >= 1.5
    with pytest.raises(module.CollectionStopped, match="while sharing"):
        with rig["turns"].turn():
            pytest.fail("The occupied simulator was acquired")
    assert rig["clock"].now == 1.5
    assert rig["bridge"].queries == [0]
    assert not rig["bridge"].clears
    assert_free(rig["path"])


def test_stop_in_current_body_preserves_body_and_interrupts_next_yield(rig):
    with rig["turns"].turn():
        rig["stopped"][0] = True
        assert_held(rig["path"])
        rig["clock"].sleep(0.5)
        assert_held(rig["path"])
    assert_free(rig["path"])
    with pytest.raises(module.CollectionStopped):
        rig["turns"].yield_turn()
    assert rig["clock"].now == 0.5


def test_yield_between_episodes_leaves_lock_free_for_five_seconds(rig):
    with rig["turns"].turn():
        assert_held(rig["path"])
    rig["turns"].yield_turn()
    assert rig["clock"].now == 5
    assert rig["waits"] == [(0, "yield_after_episode", 5, None)]
    assert_free(rig["path"])
    with rig["turns"].turn():
        assert_held(rig["path"])
    assert rig["bridge"].queries == [0, 5]


def test_failed_metadata_query_releases_lock_and_does_not_retry(rig):
    rig["bridge"].query_error = BridgeError("metadata unavailable")
    with pytest.raises(BridgeError, match="metadata unavailable"):
        with rig["turns"].turn():
            pytest.fail("A turn was granted without lifecycle metadata")
    assert_free(rig["path"])
    assert rig["bridge"].queries == [0]
    assert not rig["waits"]
    assert not rig["bridge"].clears


def test_clear_followed_by_nonidle_state_does_not_grant_turn(rig):
    rig["bridge"].state = finished()
    rig["bridge"].after_clear = practice(case_code=CASE_B)
    with pytest.raises(BridgeError):
        with rig["turns"].turn():
            pytest.fail("The simulator was not idle after clearing metadata")
    assert_free(rig["path"])
    assert rig["bridge"].clears == [(20, CASE_A)]
    assert [event["status"] for event in rig["events"]] == ["requested"]


@pytest.mark.parametrize("field", ["api_open", "entered"])
@pytest.mark.parametrize("value", [True, None, "missing"])
@pytest.mark.parametrize("after_clear", [False, True])
def test_idle_requires_explicit_closed_and_unentered_metadata(rig, field, value, after_clear):
    state = idle()
    if value == "missing":
        state.pop(field)
    else:
        state[field] = value
    if after_clear:
        rig["bridge"].state = finished()
        rig["bridge"].after_clear = state
    else:
        rig["bridge"].state = state
    with pytest.raises(BridgeError, match="explicit"):
        with rig["turns"].turn():
            pytest.fail("Incomplete idle metadata granted a turn")
    assert_free(rig["path"])
    if after_clear:
        assert rig["bridge"].clears == [(20, CASE_A)]
        assert [event["status"] for event in rig["events"]] == ["requested"]
    else:
        assert not rig["bridge"].clears
        assert not rig["waits"]
