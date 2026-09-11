"""Serial Q3/Q4 practice runs, with a case check before every HTTP request."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import unicodedata

from simulator_client import SimulatorClient
from simulator_client.errors import SessionError, SimulatorError
from strategies import run_search
from workflow.__main__ import revision, working_tree_dirty

from .bridge import BridgeError


class PracticeOwnershipError(SessionError):
    pass


def assert_owned(state, problem, case, *, api=False):
    if (not isinstance(state, dict) or state.get("mode") != "practice"
            or state.get("problem_no") != problem or state.get("case_code") != case
            or not re.fullmatch(r"[A-Z0-9]{4}(?:-[A-Z0-9]{4}){3}", case)):
        raise PracticeOwnershipError("The current test is not the practice case created by this controller")
    if api and (state.get("active") is not True or state.get("api_open") is not True
                or state.get("phase") not in {"waiting_enter", "running"}
                or state.get("cleanup_complete") is not False):
        raise PracticeOwnershipError("The owned practice API is not open")
    return state


class GuardedPracticeClient(SimulatorClient):
    def __init__(self, robot_id, *, bridge, problem, case, keep_alive_http=False, **kwargs):
        self.bridge, self.problem, self.case = bridge, problem, case
        self._practice_transport = None
        # Port 2026 is the original simulator HTTP API, not the CDP port.
        super().__init__(robot_id, base_url="http://127.0.0.1:2026", **kwargs)
        if keep_alive_http:
            from .transport import LoopbackHTTPTransport
            self._practice_transport = LoopbackHTTPTransport()

    def _exchange(self, action, timeout):
        try:
            state = assert_owned(self.bridge.current_test(), self.problem, self.case, api=True)
            if (action.path == "/enter" and state.get("entered") is not False
                    and not action.had_unknown_attempt):
                raise PracticeOwnershipError("Another client has entered this practice case")
        except BridgeError as exc:
            raise PracticeOwnershipError("Cannot verify the practice case; no HTTP request sent") from exc
        if self._practice_transport is not None:
            return self._practice_transport.exchange(action, timeout)
        return super()._exchange(action, timeout)

    def close(self):
        try:
            super().close()
        finally:
            if self._practice_transport is not None:
                self._practice_transport.close()


@contextmanager
def controller_lock(path):
    """OS lock is released on a crash; a stale filename never implies ownership."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        import os
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BridgeError("Another practice controller holds the simulator lock") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def wait_state(bridge, problem, case, predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        state = assert_owned(bridge.current_test(), problem, case)
        if predicate(state):
            return state
        if time.monotonic() >= deadline:
            raise BridgeError("Practice transition timed out; preserve the current case and inspect status")
        time.sleep(0.25)


def validate_run(problem, variant, repeats, max_actions, robot_id):
    if type(problem) is not int or problem not in (3, 4):
        raise ValueError("Only problem 3 and 4 practice are supported")
    if type(repeats) is not int or not 1 <= repeats <= 1000:
        raise ValueError("Repeat count must be 1..1000")
    if type(max_actions) is not int or max_actions < 2:
        raise ValueError("max_actions must be at least 2")
    if (not isinstance(robot_id, str) or not robot_id.strip()
            or not 1 <= len(robot_id.encode("utf-8")) <= 64
            or any(unicodedata.category(c) in {"Cc", "Cf"} for c in robot_id)):
        raise ValueError("Provide --robot-id or CUMCM_ROBOT_ID; never provide a password")
    variant = variant or ("efficient" if problem == 3 else "triangular")
    allowed = {3: {"baseline", "adaptive", "deferred", "efficient", "rollout"},
               4: {"baseline", "adaptive", "deferred", "triangular"}}
    if variant not in allowed[problem]:
        raise ValueError("Variant does not support the selected problem")
    return variant


def run_once(bridge, *, problem, robot_id, variant, max_actions, output,
             simulator_dir, solver=run_search, method_label=None, method_metadata=None):
    """The injected solver receives only a guarded official HTTP client."""
    variant = validate_run(problem, variant, 1, max_actions, robot_id)
    output.mkdir(parents=True, exist_ok=False)
    try:
        initial = bridge.start_practice(problem)
    except BridgeError as exc:
        write_json(output / "start_error.json", {"type": type(exc).__name__, "error": str(exc)})
        raise
    case = initial.get("case_code", "")
    # Preparation can precede assignment of a case code. Follow it only if the
    # initial response already contains a stable run identifier.
    if not case:
        run_no = initial.get("practice_run_no")
        if not run_no:
            write_json(output / "preparation-unresolved.json", initial)
            raise BridgeError("Practice started without a stable identifier; inspect status, do not retry start")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            current = bridge.current_test()
            if (not isinstance(current, dict) or current.get("mode") != "practice"
                    or current.get("problem_no") != problem
                    or current.get("practice_run_no") != run_no):
                raise PracticeOwnershipError("Practice preparation lost its expected problem")
            case = current.get("case_code", "")
            if case:
                break
            time.sleep(0.25)
    assert_owned(bridge.current_test(), problem, case)
    write_json(output / "created.json", {"mode": "practice", "problem": problem,
               "case_code": case, "initial_state": initial})
    before = wait_state(bridge, problem, case,
                        lambda s: s.get("phase") == "waiting_enter" and s.get("api_open") is True)
    if before.get("entered") is not False:
        raise PracticeOwnershipError("The new practice case has already been entered")
    report = {
        "data_origin": "simulator_http_session", "declared_mode": "practice",
        "problem": problem, "case_code": case, "variant": method_label or variant,
        "code_revision": revision(), "working_tree_dirty": working_tree_dirty(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "gui_mode_verified_by_api": True, "mode_verification": "Wails CurrentTest before each HTTP request",
        "official_result_verified": False,
    }
    if method_metadata is not None:
        report["method_metadata"] = method_metadata
    write_json(output / "before.json", before)
    error = None
    started = time.monotonic()
    with GuardedPracticeClient(robot_id, bridge=bridge, problem=problem, case=case,
                               log_path=output / "requests.jsonl", keep_alive_http=True) as client:
        try:
            result = solver(client, problem=problem, variant=variant, max_actions=max_actions)
            report["search"] = result.as_dict()
            if result.error or result.exit_error:
                error = result.error or result.exit_error
            elif not result.completion_certified_under_model:
                error = f"SearchIncomplete: {result.completion_reason}"
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if client.state.session == "active" and client.pending_request is None:
                try:
                    client.exit()
                except (SimulatorError, BridgeError, OSError, ValueError) as exc:
                    error = error or f"{type(exc).__name__}: {exc}"
            report["state"] = client.state.snapshot()
            report["pending_request"] = client.pending_request
    report["program_wall_time_s"] = time.monotonic() - started
    if report["state"]["session"] != "exited" or report["pending_request"] is not None:
        error = error or "SessionIncomplete: accepted exit and no pending request are required"
    report.update(completed=error is None, error=error)
    write_json(output / "summary.json", report)
    if report["state"]["session"] != "exited" or report["pending_request"] is not None:
        raise BridgeError("Practice did not exit cleanly; batch stopped with evidence preserved")
    after = wait_state(bridge, problem, case,
                       lambda s: s.get("phase") == "ended" and s.get("cleanup_complete") is True
                       and s.get("log_package_status") == "ready")
    write_json(output / "after.json", after)
    # Read only the matching finished practice result. Never consult active
    # source counts, scenario seeds, hidden positions, or any other test store.
    log_dir = simulator_dir / "JammersSimulatorData" / "behavior-logs"
    matches = list(log_dir.glob(f"practice-p{problem}-*-{case}.result.json"))
    if len(matches) != 1:
        raise BridgeError("Expected exactly one saved result for the completed practice case")
    from experiments.register_practice import register
    from argparse import Namespace
    registration = register(Namespace(summary=output / "summary.json", official_result_json=matches[0],
                                      output_dir=output.parent / "registered", runtime=None,
                                      case_code=case, source_total=None))
    write_json(output / "registration.json", registration)
    bridge.clear_finished_test(case)
    return {"case_code": case, "problem": problem, "variant": method_label or variant,
            "cleared_count": report["state"]["cleared_count"],
            "source_total": registration["source_total"],
            "virtual_time_s": report["state"]["virtual_time_s"],
            "program_wall_time_s": report["program_wall_time_s"],
            "completed": report["completed"], "summary": str(output / "summary.json")}
