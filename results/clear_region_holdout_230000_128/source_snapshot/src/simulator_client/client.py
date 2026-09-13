"""Serial, idempotent simulator requests with conservative runtime accounting."""

from dataclasses import dataclass
from datetime import datetime, timezone
import copy
import http.client
import json
import math
from pathlib import Path
import threading
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import uuid

from .errors import (
    DeadlineExceeded, OutcomeUnknown, PendingActionError,
    RequestRejected, SessionError,
)
from .rules import (
    CHANNELS, DEFAULT_BASE_URL, MEASURE_TIME_S, OPTICAL_TIME_S,
    REMOVAL_TIME_S, SPEED_M_S, SWITCH_TIME_S,
)
from .state import ClientState, Position, SourceState


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass
class _PendingAction:
    path: str
    payload: dict
    body: bytes
    first_sent_at: float
    had_unknown_attempt: bool = False


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _no_json_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


class SimulatorClient:
    """One object per test session; create a new object for the next test.

    Public methods return the official JSON response dictionary. Only a validated
    accepted response changes local state. Unknown outcomes retain the exact
    pending request, and block new actions until retry_pending() resolves it.
    Closing this object only closes the local journal; exit() is explicit.
    """

    def __init__(
        self, robot_id: str, *, base_url: str = DEFAULT_BASE_URL,
        log_path: str | Path | None = None, timeout_s: float = 3.0,
        max_attempts: int = 3, retry_backoff_s: float = 0.1,
        clock=time.monotonic, sleep_fn=time.sleep,
    ):
        if not isinstance(robot_id, str) or not 1 <= len(robot_id.encode("utf-8")) <= 64:
            raise ValueError("robot_id must be a UTF-8 string of 1 to 64 bytes")
        if any(unicodedata.category(c) in {"Cc", "Cf"} for c in robot_id):
            raise ValueError("robot_id must not contain control or format characters")
        parts = urlsplit(base_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username or parts.password or parts.query or parts.fragment
                or parts.path not in {"", "/"}):
            raise ValueError("base_url must be a plain http(s) origin, without credentials")
        if _number(timeout_s, "timeout_s") <= 0:
            raise ValueError("timeout_s must be positive")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if _number(retry_backoff_s, "retry_backoff_s") < 0:
            raise ValueError("retry_backoff_s must be nonnegative")
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.retry_backoff_s = retry_backoff_s
        self.state = ClientState()
        self._clock = clock
        self._sleep = sleep_fn
        self._lock = threading.Lock()
        self._pending: _PendingAction | None = None
        self._closed = False
        self._run_id = uuid.uuid4().hex
        self._sequence = 0
        # The official simulator is local. Environment proxy settings must not
        # turn localhost actions into proxy requests or redirect them elsewhere.
        self._opener = build_opener(ProxyHandler({}), _NoRedirects())
        self.log_path = Path(log_path) if log_path else Path("results/t01") / f"{self._run_id}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._journal = self.log_path.open("x", encoding="utf-8")
        self._log("session_created", base_url=self.base_url, robot_id=self.robot_id)

    @property
    def pending_request(self) -> dict | None:
        if self._pending is None:
            return None
        return {"path": self._pending.path, "payload": copy.deepcopy(self._pending.payload)}

    @property
    def remaining_real_time_s(self) -> float | None:
        if self.state.real_deadline is None:
            return None
        return max(0.0, self.state.real_deadline - self._clock())

    def enter(self) -> dict:
        with self._lock:
            self._check_available()
            if self.state.session != "new":
                raise SessionError("This client has already entered a test")
            return self._new_action("/enter", {})

    def measure(self, position: Position | tuple[float, float], channel: int) -> dict:
        return self._position_action("/measure", position, channel)

    def clear(self, position: Position | tuple[float, float], channel: int) -> dict:
        return self._position_action("/clear", position, channel)

    def exit(self) -> dict:
        with self._lock:
            self._check_available(active=True)
            return self._new_action("/exit", {})

    def retry_pending(self) -> dict:
        """Retry precisely the unresolved request, with its original ID and body."""
        with self._lock:
            if self._closed:
                raise SessionError("Client is closed")
            if self._pending is None:
                raise SessionError("There is no unresolved request")
            return self._send_pending()

    def _position_action(self, path, position, channel):
        pos = Position.coerce(position)
        if isinstance(channel, bool) or not isinstance(channel, int) or channel not in CHANNELS:
            raise ValueError("channel must be an integer from 1 to 20")
        with self._lock:
            self._check_available(active=True)
            return self._new_action(path, {"position": {"x": pos.x, "y": pos.y}, "channel": channel})

    def _check_available(self, active=False):
        if self._closed:
            raise SessionError("Client is closed")
        if self._pending is not None:
            raise PendingActionError("Resolve the previous action with retry_pending() first")
        if active and self.state.session != "active":
            raise SessionError("An active test is required")
        if active:
            self._request_timeout()
            if (self.state.max_virtual_duration_s is not None
                    and self.state.virtual_time_s >= self.state.max_virtual_duration_s):
                raise DeadlineExceeded("The virtual time limit has been reached")

    def _new_action(self, path, fields):
        self._sequence += 1
        payload = {
            "arena_id": "default", "robot_id": self.robot_id,
            "request_id": f"{self._run_id}-{self._sequence}", **fields,
        }
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._pending = _PendingAction(path, payload, body, self._clock())
        return self._send_pending()

    def _request_timeout(self):
        remaining = self.remaining_real_time_s
        if remaining is None:
            return self.timeout_s
        # A small margin also covers dispatch overhead; it does not extend the
        # simulator's deadline and does not assume a fresh 1200 s after a retry.
        if remaining <= 0.05:
            raise DeadlineExceeded("The remaining real-time budget is exhausted")
        return min(self.timeout_s, remaining - 0.05)

    def _exchange(self, action, timeout):
        request = Request(
            self.base_url + action.path, data=action.body, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        try:
            response = self._opener.open(request, timeout=timeout)
        except HTTPError as error:
            response = error  # An HTTP error can still carry a definite rejection.
        with response:
            status = response.code
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError("Response exceeds the client response-size limit")
        data = json.loads(raw.decode("utf-8"), parse_constant=_no_json_constant)
        return status, data

    def _send_pending(self):
        action = self._pending
        assert action is not None
        request_id = action.payload["request_id"]
        last_error = "No response"
        for attempt in range(1, self.max_attempts + 1):
            try:
                timeout = self._request_timeout()
            except DeadlineExceeded as error:
                self._log("outcome_unknown", request_id=request_id, detail=str(error))
                raise OutcomeUnknown(request_id, str(error)) from error
            self._log("request", attempt=attempt, path=action.path, payload=action.payload)
            try:
                status, response = self._exchange(action, timeout)
                self._validate_common(response)
                self._log("response", request_id=request_id, attempt=attempt,
                          http_status=status, response=response)
                if response["accepted"] is False:
                    if action.had_unknown_attempt:
                        # This rejection describes only THIS retry. It cannot
                        # disprove an earlier execution whose response was lost.
                        detail = f"Retry rejected with HTTP {status}; earlier execution remains unresolved"
                        self._log("outcome_unknown", request_id=request_id, detail=detail)
                        raise OutcomeUnknown(request_id, detail)
                    self._pending = None
                    raise RequestRejected(status, response, request_id)
                if status != 200:
                    raise ValueError(f"Conflicting HTTP {status} and accepted=true")
                self._validate_accepted(action, response)
            except RequestRejected:
                raise
            except (OSError, URLError, http.client.HTTPException, ValueError) as error:
                action.had_unknown_attempt = True
                last_error = f"{type(error).__name__}: {error}"
                self._log("attempt_failed", request_id=request_id,
                          attempt=attempt, detail=last_error)
                if attempt < self.max_attempts:
                    delay = self.retry_backoff_s * (2 ** (attempt - 1))
                    remaining = self.remaining_real_time_s
                    if remaining is not None:
                        delay = min(delay, max(0.0, remaining - 0.05))
                    if delay:
                        self._sleep(delay)
                    continue
                break
            self._apply_accepted(action, response)
            self._pending = None
            self._log("state", request_id=request_id, state=self.state.snapshot())
            return response
        self._log("outcome_unknown", request_id=request_id, detail=last_error)
        raise OutcomeUnknown(request_id, last_error)

    @staticmethod
    def _validate_common(response):
        if not isinstance(response, dict) or not isinstance(response.get("accepted"), bool):
            raise ValueError("Response requires a boolean accepted field")
        for name in ("virtual_time_s", "real_timestamp_ms"):
            if _number(response.get(name), name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    def _validate_accepted(self, action, response):
        if action.path == "/enter":
            virtual_max = _number(response.get("max_virtual_duration_s"), "max_virtual_duration_s")
            real_max = _number(response.get("max_real_duration_s"), "max_real_duration_s")
            remaining = _number(response.get("remaining_real_duration_s"), "remaining_real_duration_s")
            if virtual_max <= 0 or real_max <= 0 or not 0 <= remaining <= real_max:
                raise ValueError("Invalid duration limits in enter response")
            expected_time = 0.0
        else:
            if action.path == "/measure":
                kind = response.get("measure_result")
                if not isinstance(kind, str) or kind not in {"direction", "near", "no_signal"}:
                    raise ValueError("Unknown measure_result")
                if kind == "direction":
                    bearing = _number(response.get("svd_deg"), "svd_deg")
                    if not 0 <= bearing < 360:
                        raise ValueError("svd_deg must be in [0, 360)")
            elif action.path == "/clear":
                kind = response.get("clear_result")
                if not isinstance(kind, str) or kind not in {"success", "no_target_in_range"}:
                    raise ValueError("Unknown clear_result")
            elif response.get("exit_reason") != "user_exit":
                raise ValueError("Unexpected exit_reason")
            expected_time = self.state.virtual_time_s + sum(self._increments(action, response).values())
        if not math.isclose(response["virtual_time_s"], expected_time, rel_tol=1e-10, abs_tol=1e-5):
            raise ValueError(f"Virtual time inconsistent with action: expected {expected_time}, "
                             f"received {response['virtual_time_s']}")

    def _increments(self, action, response):
        if action.path not in {"/measure", "/clear"}:
            return {}
        pos = Position(**action.payload["position"])
        costs = {"movement_s": self.state.position.distance_to(pos) / SPEED_M_S}
        if action.path == "/measure":
            costs["switching_s"] = SWITCH_TIME_S * (action.payload["channel"] != self.state.current_channel)
            costs["detection_s"] = MEASURE_TIME_S
        else:
            costs["optical_s"] = OPTICAL_TIME_S
            costs["removal_s"] = REMOVAL_TIME_S * (response["clear_result"] == "success")
        return costs

    def _apply_accepted(self, action, response):
        if action.path == "/enter":
            self.state.session = "active"
            self.state.max_virtual_duration_s = float(response["max_virtual_duration_s"])
            # A lost enter response may be replayed with its ORIGINAL remaining
            # duration. Anchor at the FIRST request, not at replay arrival time.
            self.state.real_deadline = action.first_sent_at + response["remaining_real_duration_s"]
        elif action.path == "/exit":
            self.state.session = "exited"
        else:
            for key, value in self._increments(action, response).items():
                setattr(self.state.time_breakdown, key, getattr(self.state.time_breakdown, key) + value)
            pos = Position(**action.payload["position"])
            channel = action.payload["channel"]
            self.state.position = pos
            source = self.state.sources.setdefault(channel, SourceState())
            if action.path == "/measure":
                self.state.current_channel = channel
                source.measurement_count += 1
                source.last_result = response["measure_result"]
                source.last_bearing_deg = response.get("svd_deg") if source.last_result == "direction" else None
                source.last_measurement_position = pos
                source.last_measurement_virtual_time_s = float(response["virtual_time_s"])
                if source.last_result in {"direction", "near"} and source.status != "cleared":
                    source.status = "detected"
            elif response["clear_result"] == "success":
                source.status = "cleared"
            else:
                source.failed_clear_count += 1
        self.state.virtual_time_s = float(response["virtual_time_s"])
        self.state.accepted_actions += 1

    def _log(self, event, **fields):
        record = {"event": event, "run_id": self._run_id,
                  "recorded_at": datetime.now(timezone.utc).isoformat(), **fields}
        self._journal.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._journal.flush()

    def close(self):
        with self._lock:
            if not self._closed:
                self._log("client_closed", state=self.state.snapshot(), pending=self.pending_request)
                self._journal.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
