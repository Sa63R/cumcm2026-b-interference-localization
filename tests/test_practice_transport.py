"""Real local HTTP-server tests; never connect to the simulator's port."""

from collections import deque
from http.client import HTTPConnection as RealHTTPConnection, IncompleteRead, RemoteDisconnected
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
from types import SimpleNamespace

import pytest

import practice_control.transport as transport_module
from practice_control.transport import LoopbackHTTPTransport


@pytest.fixture
def local_server(monkeypatch):
    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            # Tests deliberately close/reset sockets while handlers are active.
            pass

        def get_request(self):
            result = super().get_request()
            self.connections += 1
            return result

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            self.server.requests.append((self.path, body))
            mode = self.server.responses.popleft() if self.server.responses else "valid"
            if mode == "disconnect":
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if mode == "timeout":
                time.sleep(0.15)
            status = 302 if mode == "redirect" else 429 if mode == "rejected" else 200
            payloads = {
                "invalid": b"{broken", "nan": b'{"x": NaN}',
                "infinity": b'{"x": Infinity}', "negative_infinity": b'{"x": -Infinity}',
                "utf8": b'"\xff"', "large": b"x" * 1_048_577,
                "rejected": b'{"accepted": false}', "redirect": b'{"accepted": false}',
            }
            payload = payloads.get(mode, b'{"accepted": true}')
            try:
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload) + (10 if mode == "truncated" else 0)))
                if mode == "redirect":
                    self.send_header("Location", "/never-follow")
                if mode in {"close", "truncated"}:
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = Server(("127.0.0.1", 0), Handler)
    server.connections = 0
    server.requests = []
    server.responses = deque()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    constructors = []

    def connection_factory(host, port, *, timeout):
        assert (host, port) == ("127.0.0.1", 2026)
        connection = RealHTTPConnection("127.0.0.1", server.server_port, timeout=timeout)
        constructors.append(connection)
        return connection

    monkeypatch.setattr(transport_module, "HTTPConnection", connection_factory)
    yield server, constructors
    for connection in constructors:
        connection.close()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def action(path="/measure", request_id="same-id"):
    return SimpleNamespace(path=path, body=json.dumps({"request_id": request_id}).encode())


def test_all_four_paths_reuse_one_socket_and_set_each_timeout(local_server):
    server, constructors = local_server
    transport = LoopbackHTTPTransport()
    for index, path in enumerate(("/enter", "/measure", "/clear", "/exit")):
        timeout = 1.0 + index / 10
        assert transport.exchange(action(path), timeout) == (200, {"accepted": True})
        assert constructors[0].sock.gettimeout() == pytest.approx(timeout)
    assert server.connections == len(constructors) == 1
    assert [path for path, body in server.requests] == ["/enter", "/measure", "/clear", "/exit"]
    transport.close()
    transport.close()
    assert constructors[0].sock is None
    with pytest.raises(RuntimeError, match="closed"):
        transport.exchange(action(), 1)


@pytest.mark.parametrize("mode", ["invalid", "nan", "infinity", "negative_infinity", "utf8", "large", "truncated"])
def test_bad_response_closes_connection_without_retry(mode, local_server):
    server, constructors = local_server
    server.responses.append(mode)
    transport = LoopbackHTTPTransport()
    with pytest.raises(IncompleteRead if mode == "truncated" else ValueError):
        transport.exchange(action(), 1)
    assert len(server.requests) == 1
    assert constructors[0].sock is None
    assert transport.exchange(action(), 1) == (200, {"accepted": True})
    assert len(server.requests) == len(constructors) == server.connections == 2
    transport.close()


@pytest.mark.parametrize("mode,status", [("redirect", 302), ("rejected", 429)])
def test_redirect_not_followed_and_rejection_body_returned(mode, status, local_server):
    server, _ = local_server
    server.responses.append(mode)
    transport = LoopbackHTTPTransport()
    assert transport.exchange(action(), 1) == (status, {"accepted": False})
    assert len(server.requests) == 1
    assert server.requests[0][0] == "/measure"
    transport.close()


def test_disconnect_does_not_replay_and_caller_can_retry_same_bytes(local_server):
    server, constructors = local_server
    server.responses.append("disconnect")
    transport = LoopbackHTTPTransport()
    pending = action()
    with pytest.raises(RemoteDisconnected):
        transport.exchange(pending, 1)
    assert server.requests == [(pending.path, pending.body)]
    assert constructors[0].sock is None
    assert transport.exchange(pending, 1) == (200, {"accepted": True})
    assert server.requests == [(pending.path, pending.body)] * 2
    transport.close()


def test_timeout_closes_socket_without_retry(local_server):
    server, constructors = local_server
    server.responses.append("timeout")
    transport = LoopbackHTTPTransport()
    with pytest.raises(TimeoutError):
        transport.exchange(action(), 0.03)
    assert len(server.requests) == 1
    assert constructors[0].sock is None
    transport.close()


def test_server_requested_close_reconnects_for_next_distinct_call(local_server):
    server, _ = local_server
    server.responses.append("close")
    transport = LoopbackHTTPTransport()
    assert transport.exchange(action(request_id="one"), 1) == (200, {"accepted": True})
    assert transport.exchange(action(request_id="two"), 1) == (200, {"accepted": True})
    assert server.connections == 2
    assert len(server.requests) == 2
    transport.close()


@pytest.mark.parametrize("path", ["/start", "/measure?x=1", "//example.com/measure", "http://example.com/measure"])
def test_invalid_path_never_opens_connection(path, local_server):
    server, constructors = local_server
    with pytest.raises(ValueError, match="paths"):
        LoopbackHTTPTransport().exchange(action(path), 1)
    assert not constructors and not server.requests


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_timeout_never_opens_connection(timeout, local_server):
    server, constructors = local_server
    with pytest.raises(ValueError, match="timeout"):
        LoopbackHTTPTransport().exchange(action(), timeout)
    assert not constructors and not server.requests
