"""Reusable, serial HTTP transport to the original loopback simulator API.

Ownership checks and request-id retry decisions belong to the caller. This
transport sends exactly one request per exchange and never retries internally.
"""

from __future__ import annotations

from http.client import HTTPConnection, IncompleteRead
import json
import math


_PATHS = frozenset({"/enter", "/measure", "/clear", "/exit"})
_MAX_RESPONSE_BYTES = 1_048_576


def _reject_json_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


class LoopbackHTTPTransport:
    """One reusable connection, fixed to 127.0.0.1:2026; no redirects."""

    def __init__(self):
        self._connection = None
        self._closed = False

    def _disconnect(self):
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def close(self):
        """Release the socket. Repeated closes are harmless."""
        self._closed = True
        self._disconnect()

    def exchange(self, action, timeout):
        if self._closed:
            raise RuntimeError("The loopback transport is closed")
        if not isinstance(action.path, str) or action.path not in _PATHS:
            raise ValueError("Only enter, measure, clear and exit paths are allowed")
        if not isinstance(action.body, bytes):
            raise ValueError("The request body must be already-serialized bytes")
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("timeout must be a finite positive number")
        try:
            if self._connection is None:
                self._connection = HTTPConnection("127.0.0.1", 2026, timeout=float(timeout))
            connection = self._connection
            connection.timeout = float(timeout)
            if connection.sock is not None:
                connection.sock.settimeout(float(timeout))
            connection.request("POST", action.path, body=action.body,
                               headers={"Content-Type": "application/json; charset=utf-8"})
            with connection.getresponse() as response:
                status = response.status
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    raise ValueError("Response exceeds the client response-size limit")
                if response.length is not None and response.length > 0:
                    raise IncompleteRead(raw, response.length)
            data = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
            return status, data
        except Exception:
            # A send may already have executed. The caller must preserve its
            # request_id/body and resolve that outcome rather than create one.
            self._disconnect()
            raise
