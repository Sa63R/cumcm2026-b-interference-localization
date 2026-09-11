"""Offline safety checks for the practice-only bridge; never contact a simulator."""

import json
import re
import sys
import types

import pytest

from practice_control import (
    BridgeError,
    MutationOutcomeUnknown,
    PracticeBridge,
    PracticeRequestFailed,
    UnsafeSimulatorState,
)
from practice_control import bridge


def state(**changes):
    result = {
        "active": True, "mode": "practice", "problem_no": 3,
        "phase": "running", "case_code": "ABCD-EFGH-IJKL-MNOP",
        "cleanup_complete": False,
    }
    result.update(changes)
    return result


class MockBridge(PracticeBridge):
    def __init__(self, replies):
        super().__init__(9223)
        self.replies = iter(replies)
        self.calls = []

    def _evaluate(self, expression, *, mutation=False):
        self.calls.append((expression, mutation))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.mark.parametrize("problem", [True, False, 1, 2, 5, 0, -3, 3.0, "3", None])
def test_invalid_problem_is_rejected_without_any_connection(problem):
    client = MockBridge([])
    with pytest.raises(ValueError):
        client.start_practice(problem)
    assert client.calls == []


@pytest.mark.parametrize("problem", [3, 4])
@pytest.mark.parametrize("idle", [{"active": False}, {"active": False, "mode": "", "phase": "", "problem_no": 0, "case_code": ""}])
def test_only_supported_practice_binding_can_start(problem, idle):
    expected = state(problem_no=problem, phase="preparing")
    client = MockBridge([idle, {"ok": True, "run": expected}])
    assert client.start_practice(problem) == expected
    expression, mutation = client.calls[1]
    assert mutation is True
    assert f"Call.ByID(31672008, {problem}, 1000)" in expression
    assert "const before = project(await Call.ByID(3522211836, 1000))" in expression
    assert "before.active !== false" in expression
    assert "before.mode === 'formal'" in expression
    assert "ok: typeof reply?.ok === 'boolean' ? reply.ok : null" in expression
    assert set(re.findall(r"Call.ByID\((\d+)", expression)) == {"3522211836", "31672008"}


@pytest.mark.parametrize("existing", [
    None,
    state(), state(phase="preparing"), state(phase="countdown"),
    state(phase="waiting_enter"), state(phase="ending"),
    state(phase="ended", cleanup_complete=True),
    state(active=False, phase="ended", cleanup_complete=True),
    state(mode="formal"), {"active": False, "mode": "formal"},
    state(mode="future_mode"), state(phase="future_phase"),
    {"active": False, "phase": "future_phase"}, {}, {"active": None},
    state(problem_no=True), state(problem_no=3.0),
])
def test_existing_or_unknown_state_never_sends_start(existing):
    client = MockBridge([existing])
    with pytest.raises(UnsafeSimulatorState):
        client.start_practice(3)
    assert len(client.calls) == 1
    assert client.calls[0][1] is False


def test_changed_state_in_browser_blocks_start():
    client = MockBridge([{"active": False}, {"guard_error": "not_idle"}])
    with pytest.raises(UnsafeSimulatorState):
        client.start_practice(3)
    assert len(client.calls) == 2


def test_explicit_practice_rejection_is_reported_without_retry():
    client = MockBridge([{"active": False}, {"ok": False, "error_code": "test_preparation_canceled"}])
    with pytest.raises(PracticeRequestFailed, match="test_preparation_canceled") as caught:
        client.start_practice(3)
    assert caught.value.error_code == "test_preparation_canceled"
    assert len(client.calls) == 2


@pytest.mark.parametrize("reply", [None, [], {}, {"ok": 1}, {"ok": True, "run": state(mode="formal")},
                                   {"ok": True, "run": state(problem_no=4)},
                                   {"ok": True, "run": state(phase="unknown")},
                                   {"ok": True, "run": None}])
def test_unexpected_mutation_response_requires_reconciliation(reply):
    client = MockBridge([{"active": False}, reply])
    with pytest.raises(MutationOutcomeUnknown):
        client.start_practice(3)
    assert len(client.calls) == 2


def test_timeout_does_not_repeat_mutation():
    client = MockBridge([{"active": False}, MutationOutcomeUnknown("lost response")])
    with pytest.raises(MutationOutcomeUnknown):
        client.start_practice(3)
    assert len(client.calls) == 2


@pytest.mark.parametrize("existing", [None, state(), state(mode="formal", phase="ended", cleanup_complete=True),
                                     state(phase="ended", cleanup_complete=False),
                                     state(phase="ended", cleanup_complete=True, case_code="OTHER-CASE")])
def test_clear_refuses_active_formal_unsaved_and_wrong_case(existing):
    client = MockBridge([existing])
    with pytest.raises(UnsafeSimulatorState):
        client.clear_finished_test("ABCD-EFGH-IJKL-MNOP")
    assert len(client.calls) == 1
    assert client.calls[0][1] is False


def test_clear_checks_named_completed_practice_again_in_page():
    client = MockBridge([state(phase="ended", cleanup_complete=True), {"ok": True}])
    assert client.clear_finished_test("ABCD-EFGH-IJKL-MNOP") is None
    expression, mutation = client.calls[1]
    assert mutation is True
    assert "before.mode !== 'practice'" in expression
    assert "before.cleanup_complete !== true" in expression
    assert 'before.case_code !== "ABCD-EFGH-IJKL-MNOP"' in expression
    assert "ok: typeof reply?.ok === 'boolean' ? reply.ok : null" in expression
    assert set(re.findall(r"Call.ByID\((\d+)", expression)) == {"3522211836", "2343202190"}


@pytest.mark.parametrize("case", ["", None, "A';alert(1)//", "中文", "A" * 129])
def test_invalid_case_code_does_not_connect(case):
    client = MockBridge([])
    with pytest.raises(ValueError):
        client.clear_finished_test(case)
    assert client.calls == []


def test_projection_is_applied_in_browser_and_contains_no_truth_fields():
    client = MockBridge([None])
    assert client.current_test() is None
    expression = client.calls[0][0]
    assert "return project(await Call.ByID(3522211836, 1000))" in expression
    assert "for (const key of fields)" in expression
    assert "const result = {}" in expression
    for field in ("events", "jammers", "sources", "jammer_count", "source_count", "latitude", "longitude", "token"):
        assert field not in bridge._SAFE_FIELDS
        assert f'"{field}"' not in expression
    with pytest.raises(UnsafeSimulatorState, match="allowlist"):
        bridge._safe_state({**state(), "jammers": [{"x": 12, "y": 42}]})


def test_show_window_uses_only_fixed_visibility_call():
    client = MockBridge([True])
    assert client.show_window() is None
    assert client.calls == [("(async () => {const {Window} = await import('/wails/runtime.js'); await Window.Show(); return true;})()", False)]
    assert "Call.ByID" not in client.calls[0][0]


class Response:
    def __init__(self, targets, *, returned_url="http://127.0.0.1:9223/json/list"):
        self.data = json.dumps(targets).encode()
        self.url = returned_url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def geturl(self):
        return self.url

    def read(self, limit):
        return self.data[:limit]


def target(**changes):
    item = {"type": "page", "title": "无线电干扰源环境模拟器", "url": "http://localhost/",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9223/devtools/page/1"}
    item.update(changes)
    return item


def fake_discovery(monkeypatch, targets):
    handlers_seen = []
    requests = []

    def opener(*handlers):
        handlers_seen.extend(handlers)

        def open_request(request, timeout):
            requests.append(request.full_url)
            return Response(targets)

        return types.SimpleNamespace(open=open_request)

    monkeypatch.setattr(bridge, "build_opener", opener)
    return handlers_seen, requests


def test_discovery_disables_proxies_and_requires_named_local_page(monkeypatch):
    handlers, requests = fake_discovery(monkeypatch, [target(url="http://wails.localhost/")])
    assert PracticeBridge(9223)._discover_target() == "ws://127.0.0.1:9223/devtools/page/1"
    assert requests == ["http://127.0.0.1:9223/json/list"]
    assert next(item for item in handlers if isinstance(item, bridge.ProxyHandler)).proxies == {}
    assert any(isinstance(item, bridge._NoRedirect) for item in handlers)


@pytest.mark.parametrize("targets", [[], [target(), target()], [target(title="Other application")],
    [target(type="worker")], [target(url="https://example.com/")],
    [target(url="http://localhost.evil.test/")], [target(url="file:///C:/app.html")],
    [target(url="http://user@localhost/")],
    [target(webSocketDebuggerUrl="ws://example.com:9223/devtools/page/1")],
    [target(webSocketDebuggerUrl="ws://127.0.0.1:9224/devtools/page/1")],
    [target(webSocketDebuggerUrl="wss://127.0.0.1:9223/devtools/page/1")],
    [target(webSocketDebuggerUrl="ws://127.0.0.1:9223/devtools/browser/1")],
    [target(webSocketDebuggerUrl="ws://user:secret@localhost:9223/devtools/page/1")],
])
def test_discovery_rejects_ambiguous_or_nonlocal_targets(monkeypatch, targets):
    fake_discovery(monkeypatch, targets)
    with pytest.raises(BridgeError):
        PracticeBridge(9223)._discover_target()


def test_http_redirect_is_rejected_before_following():
    with pytest.raises(BridgeError, match="redirected"):
        bridge._NoRedirect().redirect_request(None, None, 302, "", {}, "http://example.com/")


class Socket:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(json.loads(payload))

    def settimeout(self, value):
        pass

    def recv(self):
        response = next(self.replies)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response)

    def close(self):
        self.closed = True


def fake_socket(monkeypatch, replies):
    socket = Socket(replies)
    connections = []

    def connect(url, **kwargs):
        connections.append((url, kwargs))
        return socket

    monkeypatch.setitem(sys.modules, "websocket", types.SimpleNamespace(create_connection=connect))
    monkeypatch.setattr(PracticeBridge, "_discover_target", lambda self: "ws://127.0.0.1:9223/devtools/page/1")
    return socket, connections


def test_cdp_awaits_json_result_and_ignores_unrelated_events(monkeypatch):
    socket, connections = fake_socket(monkeypatch, [
        {"method": "Runtime.consoleAPICalled"},
        {"id": 1, "result": {"result": {"type": "object", "value": None}}},
    ])
    assert PracticeBridge(9223).current_test() is None
    assert len(connections) == len(socket.sent) == 1
    assert connections[0][1]["suppress_origin"] is True
    assert connections[0][1]["http_no_proxy"] == ["127.0.0.1", "localhost"]
    assert connections[0][1]["redirect_limit"] == 0
    params = socket.sent[0]["params"]
    assert params["awaitPromise"] is True and params["returnByValue"] is True
    assert socket.closed is True


@pytest.mark.parametrize("reply", [TimeoutError("timeout"), ConnectionError("disconnected"),
                                   {"id": 1, "result": {"exceptionDetails": {"text": "failed"}}}])
def test_transport_failure_after_mutation_is_uncertain_and_never_retried(monkeypatch, reply):
    socket, connections = fake_socket(monkeypatch, [reply])
    with pytest.raises(MutationOutcomeUnknown):
        PracticeBridge(9223)._evaluate("fixed test expression", mutation=True)
    assert len(connections) == len(socket.sent) == 1
    assert socket.closed is True


def test_read_timeout_is_not_misreported_as_a_mutation(monkeypatch):
    socket, _ = fake_socket(monkeypatch, [TimeoutError("timeout")])
    with pytest.raises(BridgeError) as caught:
        PracticeBridge(9223).current_test()
    assert not isinstance(caught.value, MutationOutcomeUnknown)
    assert socket.closed is True


def test_scoped_connection_reuses_transport_but_queries_fresh_state(monkeypatch):
    first, second = state(), state(case_code="ZYXW-VUTS-RQPO-NMLK")
    socket, connections = fake_socket(monkeypatch, [
        {"id": 1, "result": {"result": {"value": first}}},
        {"id": 1, "result": {"result": {"value": first}}},  # stale reply
        {"id": 2, "result": {"result": {"value": second}}},
    ])
    with PracticeBridge(9223) as client:
        assert client.current_test() == first
        assert not socket.closed
        assert client.current_test() == second
        assert not socket.closed
    assert socket.closed
    assert len(connections) == 1
    assert [request["id"] for request in socket.sent] == [1, 2]
    for request in socket.sent:
        expression = request["params"]["expression"]
        assert "location.protocol !== 'http:'" in expression
        assert "location.hostname" in expression and "document.title" in expression
        assert "Call.ByID(3522211836, 1000)" in expression


def test_scoped_failed_mutation_closes_socket_and_requires_explicit_reconciliation(monkeypatch):
    first = Socket([TimeoutError("reply lost")])
    second = Socket([{"id": 2, "result": {"result": {"value": state()}}}])
    sockets = iter([first, second])
    connections = []

    def connect(url, **kwargs):
        connections.append(url)
        return next(sockets)

    monkeypatch.setitem(sys.modules, "websocket", types.SimpleNamespace(create_connection=connect))
    monkeypatch.setattr(PracticeBridge, "_discover_target", lambda self: "ws://127.0.0.1:9223/devtools/page/1")
    with PracticeBridge(9223) as client:
        with pytest.raises(MutationOutcomeUnknown):
            client._evaluate("fixed test expression", mutation=True)
        assert first.closed and len(connections) == len(first.sent) == 1
        assert client.current_test() == state()
        assert len(connections) == 2 and len(second.sent) == 1
        assert "fixed test expression" not in second.sent[0]["params"]["expression"]
    assert second.closed


def test_scoped_page_navigation_rejection_drops_transport(monkeypatch):
    socket, connections = fake_socket(monkeypatch, [
        {"id": 1, "result": {"exceptionDetails": {"text": "Unexpected simulator page"}}},
    ])
    with PracticeBridge(9223) as client:
        with pytest.raises(BridgeError, match="evaluation failed"):
            client.current_test()
        assert socket.closed
    assert len(connections) == len(socket.sent) == 1
