import importlib.util
import json
from pathlib import Path
import signal
import subprocess

import pytest


spec = importlib.util.spec_from_file_location("q4_supervisor", Path(__file__).resolve().parents[1] / "scripts/q4_cpu_supervisor.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cpu_allowance_respects_shared_quota_and_existing_work():
    assert module.cpu_allowance(50, 160, 96) == 50
    assert module.cpu_allowance(60, 160, 96, other_cores=40) == 48
    assert module.cpu_allowance(50, 160, 96, other_cores=80) == 8
    assert module.cpu_allowance(50, 6, 96) == 6
    assert module.cpu_allowance(50, 160, 4) == 1
    for request in (0, 61, True):
        with pytest.raises(ValueError):
            module.cpu_allowance(request, 160, 96)


def write_v1(directory, cores, usage=0):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cpu.cfs_quota_us").write_text(str(int(cores * 100000)) if cores is not None else "-1")
    (directory / "cpu.cfs_period_us").write_text("100000")
    (directory / "cpuacct.usage").write_text(str(usage))


def write_v2(directory, cores, usage=0):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cpu.max").write_text((str(int(cores * 100000)) if cores is not None else "max") + " 100000")
    (directory / "cpu.stat").write_text(f"usage_usec {usage}\nuser_usec {usage}\nsystem_usec 0\n")


def mount_line(mount, root="/", *, kind="cgroup2", controllers=""):
    escaped = mount.as_posix().replace(" ", r"\040")
    return f"31 20 0:1 {root} {escaped} rw - {kind} cgroup rw{',' + controllers if controllers else ''}\n"


def test_v1_uses_current_membership_and_tightest_visible_ancestor(tmp_path):
    mount = tmp_path / "cpu"
    write_v1(mount, 96, 1000000000)
    write_v1(mount / "job", 24, 800000000)
    write_v1(mount / "job" / "workers", 64, 400000000)
    hierarchy = module.resolve_cpu_hierarchy("4:cpu,cpuacct:/tenant/job/workers\n",
        mount_line(mount, "/tenant", kind="cgroup", controllers="cpu,cpuacct"))
    assert module.quota_cores(hierarchy, affinity_count=160) == 24
    assert hierarchy["visible_ancestor_count"] == 3
    assert hierarchy["mount_root_is_hierarchy_root"] is False
    snapshot = module.cpu_scope_snapshot(hierarchy)
    assert [x["quota"] for x in snapshot] == [64, 24, 96]
    assert [x["cpu_s"] for x in snapshot] == [.4, .8, 1.]


def test_v2_hidden_host_boundary_is_not_traversed_or_claimed_known(tmp_path):
    mount = tmp_path / "group mount"
    write_v2(tmp_path, 1)  # Outside the visible mount, deliberately tighter.
    write_v2(mount, 48, 1200000)
    write_v2(mount / "child", 80, 500000)
    hierarchy = module.resolve_cpu_hierarchy("0::/container/child\n",
        mount_line(mount, "/container"))
    assert module.quota_cores(hierarchy, affinity_count=160) == 48
    assert len(hierarchy["scopes"]) == 2
    assert "unobservable" in hierarchy["hidden_ancestor_limits"]
    assert module.cpu_scope_snapshot(hierarchy)[0]["cpu_s"] == .5


def test_v1_separate_cpuacct_mount_maps_matching_ancestor_usage(tmp_path):
    cpu, acct = tmp_path / "cpu", tmp_path / "acct"
    write_v1(cpu, 96)
    write_v1(cpu / "job", 48)
    acct.mkdir()
    (acct / "cpuacct.usage").write_text("2000000000")
    (acct / "job").mkdir()
    (acct / "job" / "cpuacct.usage").write_text("1000000000")
    mounts = (mount_line(cpu, kind="cgroup", controllers="cpu") +
              mount_line(acct, kind="cgroup", controllers="cpuacct"))
    hierarchy = module.resolve_cpu_hierarchy("4:cpu:/job\n5:cpuacct:/job\n", mounts)
    assert [x["cpu_s"] for x in module.cpu_scope_snapshot(hierarchy)] == [1., 2.]


def test_unresolved_cgroup_fails_closed_instead_of_guessing_host_quota(tmp_path):
    with pytest.raises(RuntimeError, match="resolve"):
        module.resolve_cpu_hierarchy("0::/outside\n", mount_line(tmp_path, "/other"))


def test_unreadable_visible_ancestor_quota_is_not_assumed_unlimited(tmp_path):
    mount = tmp_path / "group"
    write_v2(mount, 48)
    write_v2(mount / "child", 40)
    (mount / "cpu.max").write_text("not-a-quota")
    hierarchy = module.resolve_cpu_hierarchy("0::/child\n", mount_line(mount))
    with pytest.raises(RuntimeError, match="unreadable"):
        module.quota_cores(hierarchy, affinity_count=160)


def test_real_v2_root_without_cpu_max_keeps_child_limit_and_explicit_boundary(tmp_path):
    mount = tmp_path / "group"
    write_v2(mount, None)
    (mount / "cpu.max").unlink()
    write_v2(mount / "child", 24)
    hierarchy = module.resolve_cpu_hierarchy("0::/child\n", mount_line(mount))
    assert module.quota_cores(hierarchy, affinity_count=160) == 24
    assert module.cpu_scope_snapshot(hierarchy)[-1]["quota_observation"] == "absent_at_visible_root_boundary"


class FakeChild:
    pid = 1234
    returncode = None
    def __init__(self):
        self.signals = []
    def poll(self):
        return self.returncode
    def send_signal(self, value):
        self.signals.append(value)
        self.returncode = -int(value)
    def wait(self, timeout):
        return self.returncode


def test_monitor_disk_error_always_terminates_launched_child(monkeypatch):
    child = FakeChild()
    groups = []
    monkeypatch.setattr(module, "signal_group", lambda pid, sig: groups.append((pid, sig)) or False)
    def fail():
        raise OSError("simulated full result volume")
    with pytest.raises(OSError):
        module.guarded_monitor(child, fail)
    assert child.signals == [signal.SIGTERM]
    assert groups == [(child.pid, signal.SIGTERM)]
    assert child.poll() is not None


def test_leader_exit_still_cleans_remaining_worker_session(monkeypatch):
    child = FakeChild()
    child.returncode = 0
    monkeypatch.setattr(module.signal, "SIGKILL", 9, raising=False)
    groups = []
    monkeypatch.setattr(module, "signal_group", lambda pid, sig: groups.append((pid, sig)) or True)
    module.guarded_monitor(child, lambda: "complete")
    assert groups == [(child.pid, signal.SIGTERM), (child.pid, 9)]


def test_exited_process_group_race_is_not_a_supervisor_crash(monkeypatch):
    def missing(pid, sig):
        raise ProcessLookupError()
    monkeypatch.setattr(module.os, "killpg", missing, raising=False)
    assert module.signal_group(123, signal.SIGTERM) is False


def test_final_second_sync_failure_is_explicit_and_never_returns_success(tmp_path):
    calls = []
    def attempt(root, remote):
        calls.append(1)
        return {"ok": len(calls) == 1, "returncode": 0 if len(calls) == 1 else 1,
                "error_type": None}
    path = tmp_path / "supervisor.json"
    status = {"child_returncode": 0}
    assert module.finalize_sync(tmp_path, "unused", path, status, attempt=attempt) is False
    saved = json.loads(path.read_text())
    assert len(saved["final_sync_attempts"]) == 2
    assert saved["final_sync_attempts"][1]["returncode"] == 1
    assert saved["final_sync_ok"] is False
    assert saved["remote_status_requires_readback"] is True


def test_sync_timeout_becomes_safe_explicit_failure(monkeypatch):
    def timeout(*_):
        raise subprocess.TimeoutExpired("rclone", 180)
    monkeypatch.setattr(module, "sync_results", timeout)
    assert module.sync_attempt(Path("unused"), "unused") == {
        "ok": False, "returncode": None, "error_type": "TimeoutExpired"}
