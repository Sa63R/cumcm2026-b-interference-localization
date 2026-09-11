"""Offline bounded receipt recovery; an uncertain start is never replayed."""

import json
import re
import sys
from types import SimpleNamespace

import pytest

from practice_control import bridge
from practice_control.bridge import BridgeError, MutationOutcomeUnknown, PracticeBridge, PracticeRequestFailed, UnsafeSimulatorState


NONCE = "a" * 32


def state(**changes):
    return {"active": True, "mode": "practice", "problem_no": 3, "phase": "preparing",
            "practice_run_no": 17, "case_code": "ABCD-EFGH-IJKL-MNOP", **changes}


def receipt(reply=None, *, status="completed", nonce=NONCE):
    result = {"nonce": nonce, "status": status}
    if status == "completed":
        result["reply"] = reply if reply is not None else {"ok": True, "run": state(), "error_code": "request_failed"}
    return result


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ReceiptBridge(PracticeBridge):
    def __init__(self, replies):
        super().__init__(19226)
        self.replies = iter(replies)
        self.calls = []

    def _evaluate(self, expression, *, mutation=False, timeout=None):
        self.calls.append({"expression": expression, "mutation": mutation, "timeout": timeout})
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def clock(monkeypatch):
    timer = Clock()
    monkeypatch.setattr(bridge.time, "monotonic", timer.monotonic)
    monkeypatch.setattr(bridge.time, "sleep", timer.sleep)
    monkeypatch.setattr(bridge, "uuid4", lambda: SimpleNamespace(hex=NONCE))
    return timer


def lost_start(*receipts):
    return ReceiptBridge([{"active": False}, MutationOutcomeUnknown("lost start reply"), *receipts])


def assert_start_sent_once(client):
    mutations = [call for call in client.calls if call["mutation"]]
    assert len(mutations) == 1
    assert mutations[0]["expression"].count("Call.ByID(31672008,") == 1
    reads = client.calls[2:]
    assert all(call["mutation"] is False and "Call.ByID" not in call["expression"] for call in reads)
    assert all(0 < call["timeout"] <= 2.0 for call in reads)
    assert all("project(reply?.run)" in call["expression"] for call in reads)


def test_pending_then_complete_recovers_only_original_start(clock):
    client = lost_start(receipt(status="pending"), receipt())
    assert client.start_practice(3) == state()
    assert clock.sleeps == [5.0]
    assert_start_sent_once(client)
    mutation = client.calls[1]["expression"]
    assert mutation.index("window[slot] = receipt") < mutation.index("await Call.ByID(3522211836")
    assert "window[slot] === receipt" in mutation
    assert "prior.status !== 'completed'" in mutation
    assert "run: project(reply?.run)" in mutation


def test_confirmed_rejection_retains_structured_server_error(clock):
    client = lost_start(receipt({"ok": False, "run": None, "error_code": "server_timeout"}))
    with pytest.raises(PracticeRequestFailed) as caught:
        client.start_practice(3)
    assert caught.value.error_code == "server_timeout"
    assert_start_sent_once(client)


def test_confirmed_original_idle_guard_refusal_is_not_a_start_retry(clock):
    client = lost_start(receipt({"guard_error": "not_idle"}))
    with pytest.raises(UnsafeSimulatorState):
        client.start_practice(3)
    assert_start_sent_once(client)


def test_three_pending_receipts_stop_with_one_start_and_bounded_wait(clock):
    client = lost_start(*(receipt(status="pending") for _ in range(3)))
    with pytest.raises(MutationOutcomeUnknown, match="pending"):
        client.start_practice(3)
    assert len(client.calls[2:]) == 3
    assert clock.sleeps == [5.0, 5.0]
    assert clock.now == 10.0
    assert_start_sent_once(client)


@pytest.mark.parametrize("value", [
    None, {}, [], {"active": False}, receipt(nonce="b" * 32),
    receipt(status="missing_or_mismatch"), receipt(status="evaluation_error"),
    receipt(status="unexpected"), {**receipt(status="pending"), "reply": {}},
    {"nonce": NONCE, "status": "completed"},
    receipt({}), receipt({"ok": True}), receipt({"ok": False, "run": None}),
    receipt({"ok": 1, "run": state(), "error_code": "request_failed"}),
    receipt({"ok": False, "run": None, "error_code": None}),
    receipt({"ok": False, "run": None, "error_code": ""}),
    receipt({"ok": False, "run": None, "error_code": "x" * 121}),
    receipt({"ok": True, "run": state(mode="formal"), "error_code": "request_failed"}),
    receipt({"ok": True, "run": state(problem_no=4), "error_code": "request_failed"}),
    receipt({"ok": True, "run": None, "error_code": "request_failed"}),
    receipt({"ok": True, "run": state(sources=[1]), "error_code": "request_failed"}),
    {**receipt(), "extra_field": "unapproved"},
])
def test_missing_mismatched_or_incomplete_receipt_remains_unknown(clock, value):
    client = lost_start(value)
    with pytest.raises(MutationOutcomeUnknown):
        client.start_practice(3)
    assert len(client.calls) == 3
    assert clock.sleeps == []
    assert_start_sent_once(client)


@pytest.mark.parametrize("error", [BridgeError("read timeout"), ConnectionError("socket lost"),
                                 TimeoutError("read deadline"), ValueError("malformed response")])
def test_receipt_read_failure_does_not_retry_read_or_start(clock, error):
    client = lost_start(error)
    with pytest.raises(MutationOutcomeUnknown, match="must not be repeated"):
        client.start_practice(3)
    assert len(client.calls) == 3
    assert clock.sleeps == []
    assert_start_sent_once(client)


def test_previous_pending_or_failed_slot_blocks_new_start_without_adoption(clock):
    client = ReceiptBridge([{"active": False}, {"guard_error": "unresolved_start_receipt"}])
    with pytest.raises(MutationOutcomeUnknown, match="previous"):
        client.start_practice(3)
    assert len(client.calls) == 2
    mutation = client.calls[1]["expression"]
    assert mutation.index("prior.status !== 'completed'") < mutation.index("window[slot] = receipt")
    assert mutation.index("prior.status !== 'completed'") < mutation.index("Call.ByID(31672008")


def test_successful_normal_reply_needs_no_receipt_read(clock):
    client = ReceiptBridge([{"active": False}, {"ok": True, "run": state()}])
    assert client.start_practice(3) == state()
    assert len(client.calls) == 2
    assert clock.sleeps == []


def test_nonce_is_unique_per_start_operation():
    client = ReceiptBridge([{"active": False}, {"ok": True, "run": state()},
                            {"active": False}, {"ok": True, "run": state()}])
    client.start_practice(3)
    client.start_practice(3)
    nonces = [re.search(r'const nonce = "([0-9a-f]{32})";', call["expression"]).group(1)
              for call in client.calls if call["mutation"]]
    assert len(set(nonces)) == 2


def test_receipt_scheduling_budget_is_shared_across_checks(clock):
    class SlowReads(ReceiptBridge):
        def _evaluate(self, expression, *, mutation=False, timeout=None):
            if timeout is not None:
                clock.now += timeout
            return super()._evaluate(expression, mutation=mutation, timeout=timeout)

    client = SlowReads([{"active": False}, MutationOutcomeUnknown("lost"),
                       receipt(status="pending"), receipt(status="pending"), receipt(status="pending")])
    with pytest.raises(MutationOutcomeUnknown):
        client.start_practice(3)
    assert [call["timeout"] for call in client.calls[2:]] == [2.0, 2.0, 1.0]
    assert clock.now == 15.0
    assert_start_sent_once(client)


class Socket:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []
        self.timeouts = []
        self.closed = False

    def send(self, data):
        self.sent.append(json.loads(data))

    def settimeout(self, value):
        self.timeouts.append(value)

    def recv(self):
        value = next(self.replies)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)

    def close(self):
        self.closed = True


def cdp(number, value):
    return {"id": number, "result": {"result": {"value": value}}}


def test_real_evaluate_uses_new_socket_and_same_page_guard_for_receipt(clock, monkeypatch):
    first = Socket([cdp(1, {"active": False}), TimeoutError("start reply lost")])
    second = Socket([cdp(3, receipt(status="pending")), cdp(4, receipt())])
    sockets = iter((first, second))
    connected = []

    def connect(url, **kwargs):
        connected.append(kwargs)
        return next(sockets)

    monkeypatch.setitem(sys.modules, "websocket", SimpleNamespace(create_connection=connect))
    monkeypatch.setattr(PracticeBridge, "_discover_target", lambda self, **kwargs: "ws://127.0.0.1:19226/devtools/page/test")
    with PracticeBridge(19226) as client:
        assert client.start_practice(3) == state()
        assert first.closed and len(connected) == 2
    assert second.closed
    assert connected[1]["timeout"] <= 2
    all_requests = first.sent + second.sent
    assert sum("Call.ByID(31672008" in r["params"]["expression"] for r in all_requests) == 1
    for request in second.sent:
        expression = request["params"]["expression"]
        assert "document.title" in expression and "location.hostname" in expression
        assert "Call.ByID" not in expression
        assert "project(reply?.run)" in expression


def test_bounded_read_does_not_spend_a_fresh_budget_after_discovery(clock, monkeypatch):
    socket = Socket([cdp(1, receipt())])
    observed = []

    def discover(self, *, timeout=None):
        assert timeout == 2.0
        clock.now += 1.5
        return "ws://127.0.0.1:19226/devtools/page/test"

    def connect(url, **kwargs):
        observed.append(kwargs["timeout"])
        return socket

    monkeypatch.setattr(PracticeBridge, "_discover_target", discover)
    monkeypatch.setitem(sys.modules, "websocket", SimpleNamespace(create_connection=connect))
    assert PracticeBridge(19226)._evaluate("fixed read-only expression", timeout=2.0) == receipt()
    assert observed == [0.5]
    assert max(socket.timeouts) == 0.5
