"""Cooperative ownership of one simulator; no foreign trajectory access."""

from contextlib import contextmanager
import re
import time

from .bridge import BridgeError, _require_idle
from .runner import ControllerBusyError, controller_lock

POLL_SECONDS = 10.0
YIELD_SECONDS = 5.0
FINISHED_STABLE_SECONDS = 15.0


class CollectionStopped(Exception):
    """A requested pause was reached outside a running case."""


class CooperativeTurns:
    def __init__(self, bridge, lock_path, *, should_stop, on_wait, on_event):
        self.bridge = bridge
        self.lock_path = lock_path
        self.should_stop = should_stop
        self.on_wait = on_wait
        self.on_event = on_event
        self.finished = None
        self.finished_since = None
        self.idle_wait_required = False

    def wait(self, seconds, reason, state=None):
        self.on_wait(reason, seconds, state)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.should_stop():
                raise CollectionStopped("Pause requested while sharing the simulator")
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        if self.should_stop():
            raise CollectionStopped("Pause requested while sharing the simulator")

    def yield_turn(self):
        self.wait(YIELD_SECONDS, "yield_after_episode")

    def _clear_stability(self):
        self.finished = None
        self.finished_since = None

    def _ready(self, state):
        if not isinstance(state, dict) or type(state.get("active")) is not bool:
            raise BridgeError("Cooperative control requires explicit simulator lifecycle metadata")
        if state.get("mode") == "formal":
            self._clear_stability()
            return False, "formal_session_present"
        try:
            _require_idle(state)
        except BridgeError:
            pass
        else:
            if state.get("api_open") is not False or state.get("entered") is not False:
                raise BridgeError("Idle sharing state must explicitly close the API and clear entry")
            self._clear_stability()
            return True, "idle"
        if state.get("mode") != "practice" or state.get("problem_no") not in (3, 4):
            raise BridgeError("Unrecognized simulator state while sharing; no action taken")
        if (state.get("phase") == "ended" and state.get("cleanup_complete") is True
                and state.get("api_open") is False
                and state.get("log_package_status") == "ready"
                and isinstance(state.get("case_code"), str)
                and re.fullmatch(r"[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}", state["case_code"])):
            signature = tuple(state.get(key) for key in (
                "case_code", "problem_no", "phase", "cleanup_complete", "log_package_status",
                "log_file_name", "log_package_bytes", "practice_run_no"))
            if signature != self.finished:
                self.finished, self.finished_since = signature, time.monotonic()
            if time.monotonic() - self.finished_since >= FINISHED_STABLE_SECONDS:
                # Metadata only: the other task's source counts, observations,
                # results and saved files are neither fetched nor imported.
                event = {"action": "clear_stable_finished_practice", "state": dict(state),
                         "stable_seconds": time.monotonic() - self.finished_since, "status": "requested"}
                self.on_event(event)
                self.bridge.clear_finished_test(state["case_code"])
                after = self.bridge.current_test()
                _require_idle(after)
                if after.get("api_open") is not False or after.get("entered") is not False:
                    raise BridgeError("Finished practice did not return an explicit closed, unentered idle state")
                self.on_event({**event, "status": "confirmed"})
                self._clear_stability()
                return True, "idle"
            return False, "finished_practice_stabilizing"
        self._clear_stability()
        return False, "other_practice_in_progress"

    @contextmanager
    def turn(self):
        self.idle_wait_required = False
        while True:
            if self.should_stop():
                raise CollectionStopped("Pause requested before acquiring the simulator")
            lock = controller_lock(self.lock_path)
            try:
                lock.__enter__()
            except ControllerBusyError:
                self._clear_stability()
                self.idle_wait_required = True
                self.wait(POLL_SECONDS, "controller_lock_busy")
                continue
            state = None
            try:
                state = self.bridge.current_test()
                ready, reason = self._ready(state)
                if ready:
                    # The caller retains this lock through solving, registration
                    # and DB import. Its practice bridge rechecks before start.
                    yield
                    return
                self.idle_wait_required = True
            finally:
                lock.__exit__(None, None, None)
            self.wait(POLL_SECONDS, reason, state)
