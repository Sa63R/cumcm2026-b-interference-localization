"""Small HTTP test double for the attachment 2 protocol.

This is NOT the official simulator or an official practice run. Two public,
fixed omnidirectional sources make protocol tests deterministic. It deliberately
does not model hidden scenarios, official limits, directional coverage or search
quality. Zero bearing error is used for predictable response assertions.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import socket
from threading import Lock, Thread
import time
from typing import Any
import unicodedata


_PATHS = {"/enter", "/measure", "/clear", "/exit"}
_BASE_FIELDS = {"arena_id", "robot_id", "request_id"}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def _valid_identifier(value: Any, maximum: int) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return 1 <= len(value.encode("utf-8")) <= maximum and all(
            unicodedata.category(character) not in {"Cc", "Cf"}
            for character in value
        )
    except UnicodeEncodeError:
        return False


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class FakeSimulator:
    """Context-managed localhost fixture with inspectable requests and state.

    ``drop_once_paths.add('/clear')`` executes and caches the next accepted
    clear action, then closes its connection before writing the response.
    Retrying the identical payload returns the cached response without mutation.

    ``reject_next(409)`` injects a JSON rejection before executing the next
    well-formed request. Status 200 represents a business rejection. Such
    responses always report virtual_time_s=0 and do not reserve request IDs.
    """

    expected_robot_id = "test-team"

    def __init__(self, *, remaining_real_duration_s: int = 1200) -> None:
        self.requests: list[tuple[str, str, Any]] = []
        self.state: dict[str, Any] = {
            "position": {"x": 0.0, "y": 0.0},
            "current_channel": 1,
            "virtual_time_s": 0.0,
            "clear_count": 0,
        }
        self.remaining_real_duration_s = remaining_real_duration_s
        self.drop_once_paths: set[str] = set()
        self.fault_queue: deque[int] = deque()
        self._sources = {1: (300.0, 400.0), 2: (0.0, 0.0)}
        self._cleared: set[int] = set()
        self._cache: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
        self._lifecycle = "new"
        self._virtual_time_us = 0
        self._lock = Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.fixture = self  # type: ignore[attr-defined]
        self.base_url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread: Thread | None = None

    def __enter__(self) -> FakeSimulator:
        if self._thread is not None:
            raise RuntimeError("FakeSimulator may only be started once")
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)
        self._server.server_close()

    def reject_next(self, status: int = 200) -> None:
        """Queue a non-executing rejection for the next parsed POST request."""
        with self._lock:
            self.fault_queue.append(status)

    def response(self, accepted: bool) -> dict[str, Any]:
        return {
            "accepted": accepted,
            "real_timestamp_ms": time.time_ns() // 1_000_000,
            "virtual_time_s": self.state["virtual_time_s"] if accepted else 0,
        }

    def _validate(self, path: str, payload: Any) -> int | None:
        if not isinstance(payload, dict):
            return 400
        expected = _BASE_FIELDS | (
            {"position", "channel"} if path in {"/measure", "/clear"} else set()
        )
        if not expected.issubset(payload):
            return 400
        if not isinstance(payload["arena_id"], str):
            return 400
        if not _valid_identifier(payload["robot_id"], 64):
            return 400
        if not _valid_identifier(payload["request_id"], 128):
            return 400
        if path in {"/measure", "/clear"}:
            position = payload["position"]
            channel = payload["channel"]
            if not isinstance(position, dict) or not {"x", "y"}.issubset(position):
                return 400
            for axis in ("x", "y"):
                if not _number(position[axis]) or abs(position[axis]) > 2_000_000:
                    return 400
            if not _number(channel) or not 1 <= channel <= 20 or int(channel) != channel:
                return 400
            if set(position) != {"x", "y"}:
                return 200
        if set(payload) != expected:
            return 200
        if payload["arena_id"] != "default" or payload["robot_id"] != self.expected_robot_id:
            return 200
        return None

    def dispatch(self, path: str, payload: Any) -> tuple[int, dict[str, Any], bool]:
        """Return HTTP status, JSON response and whether to drop this connection."""
        with self._lock:
            self.requests.append(("POST", path, deepcopy(payload)))
            if path not in _PATHS:
                return 404, self.response(False), False
            invalid = self._validate(path, payload)
            if invalid is not None:
                return invalid, self.response(False), False
            if self.fault_queue:
                return self.fault_queue.popleft(), self.response(False), False
            request_id = payload["request_id"]
            if request_id in self._cache:
                old_path, old_payload, old_response = self._cache[request_id]
                if path != old_path or payload != old_payload:
                    return 409, self.response(False), False
                return 200, deepcopy(old_response), False
            if (path == "/enter" and self._lifecycle != "new") or (
                path != "/enter" and self._lifecycle != "entered"
            ):
                return 200, self.response(False), False

            extra: dict[str, Any] = {}
            if path == "/enter":
                self._lifecycle = "entered"
                extra = {
                    "max_virtual_duration_s": 360000,
                    "max_real_duration_s": 1200,
                    "remaining_real_duration_s": self.remaining_real_duration_s,
                }
            elif path == "/exit":
                self._lifecycle = "exited"
                extra = {"exit_reason": "user_exit"}
            else:
                position = payload["position"]
                channel = int(payload["channel"])
                previous = self.state["position"]
                movement_s = math.hypot(
                    position["x"] - previous["x"], position["y"] - previous["y"]
                ) / 5
                self._virtual_time_us += round(movement_s * 1_000_000)
                source = self._sources.get(channel)
                distance = (
                    math.hypot(source[0] - position["x"], source[1] - position["y"])
                    if source is not None and channel not in self._cleared
                    else math.inf
                )
                if path == "/measure":
                    duration = 5 + int(channel != self.state["current_channel"])
                    self.state["current_channel"] = channel
                    if distance > 1000:
                        extra = {"measure_result": "no_signal"}
                    elif distance <= 5:
                        extra = {"measure_result": "near"}
                    else:
                        assert source is not None
                        bearing = math.degrees(
                            math.atan2(source[1] - position["y"], source[0] - position["x"])
                        ) % 360
                        extra = {"measure_result": "direction", "svd_deg": round(bearing, 2) % 360}
                elif distance <= 20:
                    self._cleared.add(channel)
                    self.state["clear_count"] += 1
                    duration = 5
                    extra = {"clear_result": "success"}
                else:
                    duration = 3
                    extra = {"clear_result": "no_target_in_range"}
                self._virtual_time_us += duration * 1_000_000
                self.state["virtual_time_s"] = self._virtual_time_us / 1_000_000
                self.state["position"] = {"x": float(position["x"]), "y": float(position["y"])}

            response = self.response(True) | extra
            self._cache[request_id] = (path, deepcopy(payload), deepcopy(response))
            drop = path in self.drop_once_paths
            if drop:
                self.drop_once_paths.remove(path)
            return 200, response, drop


class _Handler(BaseHTTPRequestHandler):
    """Only the transport needed by client protocol tests."""

    @property
    def fixture(self) -> FakeSimulator:
        return self.server.fixture  # type: ignore[attr-defined]

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, status: int, response: dict[str, Any]) -> None:
        raw = json.dumps(response, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        content_type = [part.strip().lower() for part in self.headers.get("Content-Type", "").split(";")]
        valid_type = content_type[0] == "application/json" and (
            len(content_type) == 1
            or (len(content_type) == 2 and content_type[1] == "charset=utf-8")
        )
        if not valid_type or self.headers.get("Content-Encoding", "identity").lower() != "identity":
            self._send(415, self.fixture.response(False))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Negative content length")
        except ValueError:
            self._send(400, self.fixture.response(False))
            return
        if length > 65536:
            self._send(413, self.fixture.response(False))
            return
        try:
            payload = json.loads(
                self.rfile.read(length).decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except (ValueError, UnicodeDecodeError, RecursionError):
            self._send(400, self.fixture.response(False))
            return
        status, response, drop = self.fixture.dispatch(self.path, payload)
        if drop:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return
        self._send(status, response)

    def do_GET(self) -> None:
        self.fixture.requests.append((self.command, self.path, None))
        self._send(405 if self.path in _PATHS else 404, self.fixture.response(False))

    do_PUT = do_GET
    do_DELETE = do_GET
    do_PATCH = do_GET
    do_HEAD = do_GET
