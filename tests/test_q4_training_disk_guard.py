import importlib.util
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location("q4_disk_supervisor", Path(__file__).resolve().parents[1] / "scripts/q4_cpu_supervisor.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_disabled_guard_never_queries_disk(monkeypatch):
    usage = Mock(side_effect=OSError("fixture"))
    monkeypatch.setattr(module.shutil, "disk_usage", usage)
    assert module.disk_guard_snapshot(Path("unused"), 0)["stop_reason"] is None
    usage.assert_not_called()


@pytest.mark.parametrize("free,stop", [(20*1024**3, False), (20*1024**3-1, True), (30*1024**3, False)])
def test_threshold_uses_gib_and_stops_strictly_below(monkeypatch, free, stop):
    usage = Mock(return_value=SimpleNamespace(free=free))
    monkeypatch.setattr(module.shutil, "disk_usage", usage)
    root = Path("task-root")
    state = module.disk_guard_snapshot(root, 20.)
    usage.assert_called_once_with(root)
    assert state["free_bytes"] == free
    assert state["minimum_free_bytes"] == 20*1024**3
    assert bool(state["stop_reason"]) is stop


def test_enabled_observation_failure_is_explicit_and_requests_own_child_stop(monkeypatch):
    monkeypatch.setattr(module.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(module.shutil, "disk_usage", Mock(side_effect=OSError("fixture private path")))
    state = module.disk_guard_snapshot(Path("task-root"), 20.)
    assert state["stop_reason"] == "disk_space_observation_failed"
    assert state["error_type"] == "OSError"
    assert "fixture private path" not in json.dumps(state)
    child = SimpleNamespace(pid=123, poll=lambda: None, send_signal=Mock())
    kill = Mock()
    monkeypatch.setattr(module, "signal_group", kill)
    began = module.termination_step(child, now=10., terminate_at=None, reason=state["stop_reason"])
    assert began == 10.
    child.send_signal.assert_called_once_with(signal.SIGTERM)
    module.termination_step(child, now=99., terminate_at=began, reason=state["stop_reason"])
    kill.assert_not_called()
    module.termination_step(child, now=101., terminate_at=began, reason=None)
    kill.assert_called_once_with(child.pid, signal.SIGKILL)
    child.send_signal.assert_called_once()


def test_low_disk_refuses_before_any_child_launch_and_saves_reason(tmp_path, monkeypatch):
    root = tmp_path/"q4-fixture"
    root.mkdir()
    monkeypatch.setattr(module, "BASE", tmp_path)
    monkeypatch.setattr(module.os, "umask", lambda mask: 0)
    monkeypatch.setattr(module.shutil, "disk_usage", Mock(return_value=SimpleNamespace(free=10*1024**3)))
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None))
    monkeypatch.setattr(sys, "argv", ["supervisor", "--root", str(root), "--run", "fixture", "--minimum-free-gib", "20",
                                      "--deadline", "2099-01-01T00:00:00+00:00", "--", "never-launch"])
    launch = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    assert module.main() == 1
    launch.assert_not_called()
    status = json.loads((root/"runs"/"fixture"/"supervisor.json").read_text())
    assert status["startup_refused"]
    assert status["stop_reason"] == "disk_free_below_minimum_before_launch"
    assert status["child_pid"] is None


@pytest.mark.parametrize("value", [-1., float("inf"), float("nan"), True])
def test_bad_threshold_rejected(value):
    with pytest.raises(ValueError):
        module.disk_guard_snapshot(Path("unused"), value)
