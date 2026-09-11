"""A narrow, loopback-only adapter for three simulator lifecycle bindings.

The browser receives fixed expressions rather than user-supplied JavaScript.
Run data is projected inside the browser, before crossing the CDP connection;
source locations, source counts, event payloads and other scenario data are
never returned by this adapter. Solver observations still use the simulator's
normal HTTP interface.
"""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class BridgeError(RuntimeError):
    """The local bridge cannot safely complete the requested operation."""


class UnsafeSimulatorState(BridgeError):
    """The simulator state does not permit the requested practice operation."""


class MutationOutcomeUnknown(BridgeError):
    """A lifecycle request may have executed; inspect state before proceeding."""


class PracticeRequestFailed(BridgeError):
    """The simulator explicitly rejected a practice lifecycle request."""

    def __init__(self, message: str, *, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BridgeError("The local debug endpoint redirected unexpectedly")


_TITLE = "无线电干扰源环境模拟器"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
_PAGE_HOSTS = _LOOPBACK_HOSTS | {"wails.localhost"}
_PHASES = frozenset({"preparing", "countdown", "waiting_enter", "running", "ending", "ended"})
_SAFE_FIELDS = (
    "active", "mode", "problem_no", "practice_run_no", "case_code", "phase",
    "api_open", "cleanup_complete", "countdown_remaining_ms", "window_remaining_ms",
    "program_remaining_ms", "entered", "end_reason", "log_package_status",
    "log_file_name", "log_package_bytes",
)
_BOOL_FIELDS = frozenset({"active", "api_open", "cleanup_complete", "entered"})
_NUMBER_FIELDS = frozenset({
    "problem_no", "practice_run_no", "countdown_remaining_ms", "window_remaining_ms",
    "program_remaining_ms", "log_package_bytes",
})

# These are the only simulator bindings this module can invoke. Keep the
# projection in the page: filtering a full run after receiving it is too late.
_PROJECT_JS = """
const fields = %s;
function project(run) {
  if (run === null || run === undefined) return null;
  if (typeof run !== 'object' || Array.isArray(run))
    throw new Error('Unexpected simulator state');
  const result = {};
  for (const key of fields) {
    if (Object.prototype.hasOwnProperty.call(run, key)) {
      const value = run[key];
      if (value !== null && !['string', 'number', 'boolean'].includes(typeof value))
        throw new Error('Unexpected simulator state field');
      result[key] = value;
    }
  }
  return result;
}
function errorCode(value) {
  return typeof value?.code === 'string' ? value.code.slice(0, 120) : 'request_failed';
}
""" % json.dumps(_SAFE_FIELDS)
_IMPORT_JS = "const {Call} = await import('/wails/runtime.js');"
_CURRENT_JS = "(async () => {" + _IMPORT_JS + _PROJECT_JS + "return project(await Call.ByID(3522211836, 1000));})()"
_SHOW_JS = "(async () => {const {Window} = await import('/wails/runtime.js'); await Window.Show(); return true;})()"
_START_RECEIPT_SLOT = "__codexPracticeStartReceiptV1"
_RECEIPT_ATTEMPTS = 3
_RECEIPT_INTERVAL_S = 5.0
_RECEIPT_WINDOW_S = 15.0
_RECEIPT_READ_TIMEOUT_S = 2.0


def _valid_port(value: int) -> int:
    if type(value) is not int or not 1024 <= value <= 65535:
        raise ValueError("debug_port must be an integer between 1024 and 65535")
    return value


def _validate_case(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("expected_case_code must be a nonempty case code")
    if any(not (character.isascii() and (character.isalnum() or character == "-")) for character in value):
        raise ValueError("expected_case_code contains invalid characters")
    return value


def _safe_state(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise UnsafeSimulatorState("The simulator returned an unrecognized state")
    if set(value) - set(_SAFE_FIELDS):
        raise UnsafeSimulatorState("The bridge returned fields outside its metadata allowlist")
    for key, item in value.items():
        if item is None:
            continue
        if key in _BOOL_FIELDS and type(item) is not bool:
            raise UnsafeSimulatorState(f"Invalid simulator metadata: {key}")
        if key in _NUMBER_FIELDS and type(item) not in (int, float):
            raise UnsafeSimulatorState(f"Invalid simulator metadata: {key}")
        if key not in _BOOL_FIELDS | _NUMBER_FIELDS and not isinstance(item, str):
            raise UnsafeSimulatorState(f"Invalid simulator metadata: {key}")
    active = value.get("active")
    if type(active) is not bool:
        raise UnsafeSimulatorState("The simulator did not identify whether a session is active")
    mode, phase = value.get("mode"), value.get("phase")
    if mode not in (None, "", "practice", "formal"):
        raise UnsafeSimulatorState("The simulator returned an unknown test mode")
    if phase not in (None, "") and phase not in _PHASES:
        raise UnsafeSimulatorState("The simulator returned an unknown test phase")
    if active and (mode not in ("practice", "formal") or phase not in _PHASES):
        raise UnsafeSimulatorState("Active simulator metadata is incomplete")
    problem = value.get("problem_no")
    if active and (type(problem) is not int or problem not in (3, 4)):
        raise UnsafeSimulatorState("The simulator returned an unknown problem number")
    return dict(value)


def _require_idle(state: dict[str, Any] | None) -> None:
    if state is None:
        raise UnsafeSimulatorState("The simulator did not provide an explicit idle state")
    if state.get("mode") == "formal":
        raise UnsafeSimulatorState("A formal-test state is present; practice control stopped")
    if state.get("active") is not False:
        raise UnsafeSimulatorState("A session is already active; it will not be replaced")
    # A finished run must be explicitly cleared by case code, preserving its
    # lifecycle and preventing a start from silently discarding that run.
    if state.get("case_code") or state.get("phase"):
        raise UnsafeSimulatorState("Clear the completed practice case explicitly before starting")


def _require_finished(state: dict[str, Any] | None, expected_case_code: str) -> None:
    if not state or state.get("mode") != "practice":
        raise UnsafeSimulatorState("Only a completed practice session can be cleared")
    if state.get("problem_no") not in (3, 4):
        raise UnsafeSimulatorState("The completed session has an unknown problem number")
    if state.get("case_code") != expected_case_code:
        raise UnsafeSimulatorState("The current practice case does not match the expected case")
    if state.get("phase") != "ended" or state.get("cleanup_complete") is not True:
        raise UnsafeSimulatorState("The practice session has not finished saving and cleaning up")


class PracticeBridge:
    """Control the existing simulator through a local WebView2 debug port.

    No lifecycle operation is retried automatically. A transport failure after
    sending a mutation raises :class:`MutationOutcomeUnknown`; callers must
    obtain fresh read-only state and reconcile it instead of repeating it.
    """

    def __init__(self, debug_port: int, *, timeout: float = 20.0):
        self.debug_port = _valid_port(debug_port)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 120:
            raise ValueError("timeout must be greater than zero and at most 120 seconds")
        self.timeout = float(timeout)
        self._connection = None
        self._reuse_connection = False
        self._request_id = 0

    def __enter__(self) -> PracticeBridge:
        # A scoped, serial controller keeps its verified WebView connection.
        # The lifecycle query still executes afresh before every robot action.
        self._reuse_connection = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release a scoped connection; unscoped calls remain one-shot."""
        self._reuse_connection = False
        self._disconnect()

    def _disconnect(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def _discover_target(self, *, timeout: float | None = None) -> str:
        url = f"http://127.0.0.1:{self.debug_port}/json/list"
        try:
            # Explicitly disable environment proxies, including localhost
            # proxies, so metadata cannot leave this machine.
            with build_opener(ProxyHandler({}), _NoRedirect()).open(
                    Request(url), timeout=self.timeout if timeout is None else timeout) as response:
                if response.geturl() != url:
                    raise BridgeError("The local debug endpoint redirected unexpectedly")
                raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                raise BridgeError("The local debug target list is too large")
            targets = json.loads(raw)
        except BridgeError:
            raise
        except Exception as exc:
            raise BridgeError("Cannot read the simulator's local debug target list") from exc
        if not isinstance(targets, list):
            raise BridgeError("Invalid local debug target list")
        matching = [target for target in targets if isinstance(target, dict)
                    and target.get("type") == "page" and target.get("title") == _TITLE]
        if len(matching) != 1:
            raise BridgeError("Expected exactly one simulator page in the local debug endpoint")
        target = matching[0]
        try:
            page = urlsplit(target.get("url", ""))
            socket = urlsplit(target.get("webSocketDebuggerUrl", ""))
            if (page.scheme != "http" or page.hostname not in _PAGE_HOSTS
                    or page.username is not None or page.password is not None):
                raise BridgeError("The simulator page does not have an approved local origin")
            if (socket.scheme != "ws" or socket.hostname not in _LOOPBACK_HOSTS
                    or socket.port != self.debug_port or socket.username is not None
                    or socket.password is not None or socket.query or socket.fragment
                    or not socket.path.startswith("/devtools/page/")):
                raise BridgeError("The simulator websocket is not on the approved local debug port")
        except (TypeError, ValueError) as exc:
            raise BridgeError("Invalid simulator target address") from exc
        return target["webSocketDebuggerUrl"]

    def _evaluate(self, expression: str, *, mutation: bool = False, timeout: float | None = None) -> Any:
        # Receipt reads have a separate small budget shared by discovery,
        # connection establishment and response waiting. Normal calls retain
        # their existing timeout behavior.
        if timeout is not None and (type(timeout) not in (int, float) or not 0 < timeout <= self.timeout):
            raise ValueError("A bounded evaluation timeout must be positive and at most the bridge timeout")
        operation_deadline = time.monotonic() + timeout if timeout is not None else None

        def remaining_timeout():
            if operation_deadline is None:
                return self.timeout
            remaining = operation_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Bounded bridge evaluation timed out")
            return remaining

        try:
            from websocket import create_connection
        except ImportError as exc:
            raise BridgeError("Install the practice-control extra to use the simulator bridge") from exc
        sent = False
        try:
            if self._connection is None:
                target = (self._discover_target() if operation_deadline is None
                          else self._discover_target(timeout=remaining_timeout()))
                self._connection = create_connection(
                    target, timeout=remaining_timeout(), suppress_origin=True,
                    http_no_proxy=["127.0.0.1", "localhost"],
                    redirect_limit=0,
                )
            connection = self._connection
            self._request_id += 1
            request_id = self._request_id
            # A reused page target could navigate. Check its current origin
            # and title before evaluating any binding, including mutations.
            guarded_expression = (
                "(() => {if (location.protocol !== 'http:' || "
                "!['127.0.0.1','localhost','wails.localhost'].includes(location.hostname) || "
                "document.title !== " + json.dumps(_TITLE) + ") "
                "throw new Error('Unexpected simulator page'); return (" + expression + ");})()"
            )
            request = {
                "id": request_id, "method": "Runtime.evaluate",
                "params": {"expression": guarded_expression, "awaitPromise": True, "returnByValue": True},
            }
            sent = True  # A partial send is also an uncertain mutation.
            connection.settimeout(remaining_timeout())
            connection.send(json.dumps(request, ensure_ascii=True))
            deadline = operation_deadline if operation_deadline is not None else time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("CDP response timed out")
                connection.settimeout(remaining)
                response = json.loads(connection.recv())
                if operation_deadline is not None and time.monotonic() > operation_deadline:
                    raise TimeoutError("Bounded bridge evaluation timed out")
                if response.get("id") != request_id:
                    continue
                if response.get("error") or response.get("result", {}).get("exceptionDetails"):
                    raise BridgeError("The simulator bridge evaluation failed")
                remote_value = response.get("result", {}).get("result", {})
                if "value" not in remote_value:
                    raise BridgeError("The simulator bridge returned no JSON value")
                return remote_value["value"]
        except Exception as exc:
            # Drop failed sockets without replaying a query or mutation.
            # A caller may explicitly reconcile state on a fresh connection.
            self._disconnect()
            if mutation and sent:
                raise MutationOutcomeUnknown(
                    "The practice request may have executed. Read current state before taking any further action."
                ) from exc
            if isinstance(exc, BridgeError):
                raise
            raise BridgeError("The local simulator bridge connection failed") from exc
        finally:
            if not self._reuse_connection:
                self._disconnect()

    def current_test(self) -> dict[str, Any] | None:
        """Read lifecycle metadata only; never retrieve scenario data."""
        return _safe_state(self._evaluate(_CURRENT_JS))

    def show_window(self) -> None:
        """Show the existing simulator window for manual login or inspection."""
        if self._evaluate(_SHOW_JS) is not True:
            raise BridgeError("The simulator window did not confirm it was shown")

    def _reconcile_start_receipt(self, nonce: str) -> dict[str, Any]:
        """Inspect only this operation's cached reply; never repeat a binding.

        At most three reads, five seconds apart, fit a fifteen-second scheduling
        budget. Each read has at most two seconds of network budget. Standard
        socket cleanup may add time; no further start request is issued.
        """
        expression = "(() => {" + _PROJECT_JS + """
const nonce = %s;
const receipt = window[%s];
if (!receipt || receipt.nonce !== nonce) return {nonce, status: 'missing_or_mismatch'};
if (receipt.status !== 'completed') return {nonce, status: receipt.status === 'pending' ? 'pending' : 'evaluation_error'};
const reply = receipt.reply;
if (reply?.guard_error === 'not_idle')
  return {nonce, status: 'completed', reply: {guard_error: 'not_idle'}};
return {nonce, status: 'completed', reply: {
  ok: typeof reply?.ok === 'boolean' ? reply.ok : null,
  run: project(reply?.run),
  error_code: typeof reply?.error_code === 'string' ? reply.error_code.slice(0, 120) : null
}};
})()""" % (json.dumps(nonce), json.dumps(_START_RECEIPT_SLOT))
        self._disconnect()  # Lost mutation replies are reconciled on a fresh socket.
        deadline = time.monotonic() + _RECEIPT_WINDOW_S
        for attempt in range(_RECEIPT_ATTEMPTS):
            if attempt:
                remaining = deadline - time.monotonic()
                if remaining <= _RECEIPT_INTERVAL_S:
                    break
                time.sleep(_RECEIPT_INTERVAL_S)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                receipt = self._evaluate(expression, timeout=min(_RECEIPT_READ_TIMEOUT_S, self.timeout, remaining))
            except Exception as error:
                raise MutationOutcomeUnknown("Cannot verify this practice-start receipt; the start must not be repeated") from error
            if not isinstance(receipt, dict) or receipt.get("nonce") != nonce:
                raise MutationOutcomeUnknown("Practice-start receipt identity was not confirmed")
            if receipt.get("status") == "pending" and set(receipt) == {"nonce", "status"}:
                continue
            if receipt.get("status") != "completed" or set(receipt) != {"nonce", "status", "reply"}:
                raise MutationOutcomeUnknown("Practice-start receipt is missing, failed or incomplete")
            reply = receipt["reply"]
            if reply == {"guard_error": "not_idle"}:
                return reply
            if (not isinstance(reply, dict) or set(reply) != {"ok", "run", "error_code"}
                    or type(reply.get("ok")) is not bool
                    or not isinstance(reply.get("error_code"), str) or not reply["error_code"]
                    or len(reply["error_code"]) > 120):
                raise MutationOutcomeUnknown("Completed practice-start receipt lacks a valid reply")
            return reply
        raise MutationOutcomeUnknown("Practice-start receipt remained pending after bounded read-only checks")

    def start_practice(self, problem_no: int) -> dict[str, Any]:
        """Start Q3 or Q4 practice only, refusing to replace an existing run."""
        if type(problem_no) is not int or problem_no not in (3, 4):
            raise ValueError("problem_no must be integer 3 or 4")
        _require_idle(self.current_test())
        nonce = uuid4().hex
        # Repeat the check inside the same fixed browser expression, reducing
        # the gap between observing idle state and requesting a practice run.
        expression = "(async () => {" + _IMPORT_JS + _PROJECT_JS + """
const nonce = %s;
const slot = %s;
const prior = window[slot];
if (prior && prior.status !== 'completed') return {guard_error: 'unresolved_start_receipt'};
const receipt = {nonce, status: 'pending'};
window[slot] = receipt;
function complete(reply) {
  if (window[slot] === receipt) {
    receipt.reply = reply;
    receipt.status = 'completed';
  }
  return reply;
}
try {
const before = project(await Call.ByID(3522211836, 1000));
if (before === null || before.active !== false || before.mode === 'formal' ||
    ![undefined, null, '', 'practice'].includes(before.mode) || before.phase || before.case_code)
  return complete({guard_error: 'not_idle'});
const reply = await Call.ByID(31672008, %d, 1000);
return complete({ok: typeof reply?.ok === 'boolean' ? reply.ok : null, run: project(reply?.run), error_code: errorCode(reply?.error)});
} catch (_) {
  if (window[slot] === receipt) receipt.status = 'evaluation_error';
  throw new Error('Practice start evaluation failed');
}
})()""" % (json.dumps(nonce), json.dumps(_START_RECEIPT_SLOT), problem_no)
        try:
            reply = self._evaluate(expression, mutation=True)
        except MutationOutcomeUnknown:
            reply = self._reconcile_start_receipt(nonce)
        if not isinstance(reply, dict):
            raise MutationOutcomeUnknown("Unexpected practice-start response; read current state")
        if reply.get("guard_error"):
            if reply["guard_error"] == "unresolved_start_receipt":
                raise MutationOutcomeUnknown("A previous practice-start receipt remains unresolved; no new start was sent")
            raise UnsafeSimulatorState("The simulator state changed before practice could start")
        if type(reply.get("ok")) is not bool:
            raise MutationOutcomeUnknown("Unexpected practice-start response; read current state")
        if reply.get("ok") is not True:
            code = reply.get("error_code", "request_failed")
            raise PracticeRequestFailed(f"Practice start rejected: {code}", error_code=code)
        try:
            state = _safe_state(reply.get("run"))
        except UnsafeSimulatorState as exc:
            raise MutationOutcomeUnknown("Practice-start metadata was unexpected; read current state") from exc
        if not state or state.get("mode") != "practice" or state.get("problem_no") != problem_no or state.get("active") is not True:
            raise MutationOutcomeUnknown("Practice-start metadata was unexpected; read current state")
        return state

    def clear_finished_test(self, expected_case_code: str) -> None:
        """Clear only the named, completely saved, finished practice run."""
        expected_case_code = _validate_case(expected_case_code)
        _require_finished(self.current_test(), expected_case_code)
        expression = "(async () => {" + _IMPORT_JS + _PROJECT_JS + """
const before = project(await Call.ByID(3522211836, 1000));
if (!before || before.mode !== 'practice' || ![3, 4].includes(before.problem_no) ||
    before.phase !== 'ended' || before.cleanup_complete !== true || before.case_code !== %s)
  return {guard_error: 'not_finished_expected_practice'};
const reply = await Call.ByID(2343202190);
return {ok: typeof reply?.ok === 'boolean' ? reply.ok : null, error_code: errorCode(reply?.error)};
})()""" % json.dumps(expected_case_code)
        reply = self._evaluate(expression, mutation=True)
        if not isinstance(reply, dict):
            raise MutationOutcomeUnknown("Unexpected clear response; read current state")
        if reply.get("guard_error"):
            raise UnsafeSimulatorState("The simulator state changed before the completed practice could be cleared")
        if type(reply.get("ok")) is not bool:
            raise MutationOutcomeUnknown("Unexpected clear response; read current state")
        if reply.get("ok") is not True:
            raise PracticeRequestFailed(f"Practice cleanup rejected: {reply.get('error_code', 'request_failed')}")
