"""Supervise one CPU-only Q4 job and sync results through the approved S3 prefix.

The entire process tree shares a restricted affinity set, so sampler, learner,
evaluation and sync processes cannot collectively exceed the requested cores.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import threading
import time

BASE = Path("/home/dataset-assist-0/usr/lh/ysh/bwc/shumo")
REMOTE = "jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/"


def read_number(path, default=None):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return default


def _unescape_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), value)


def resolve_cpu_hierarchy(cgroup_text=None, mountinfo_text=None):
    """Resolve current membership against visible mounts, never host guesses.

    Only ancestors up to the mounted root are visible. Even a '/' mount can be
    the root of a cgroup namespace; hidden host ancestors remain unobservable.
    Paths are internal only: public status reports depth/limits without names.
    """
    if cgroup_text is None:
        cgroup_text = Path("/proc/self/cgroup").read_text()
    if mountinfo_text is None:
        mountinfo_text = Path("/proc/self/mountinfo").read_text()
    memberships = {}
    for line in cgroup_text.splitlines():
        _, controllers, path = line.split(":", 2)
        for name in controllers.split(",") if controllers else ("v2",):
            memberships[name] = PurePosixPath(path)
    mounts = []
    for line in mountinfo_text.splitlines():
        fields = line.split()
        separator = fields.index("-")
        kind = fields[separator + 1]
        if kind not in {"cgroup", "cgroup2"}:
            continue
        mounts.append({"root": PurePosixPath(_unescape_mount(fields[3])),
                       "mount": Path(_unescape_mount(fields[4])), "kind": kind,
                       "controllers": set(fields[separator + 3].split(","))})

    def mapping(controller):
        member = memberships.get(controller)
        if member is None or not member.is_absolute() or ".." in member.parts:
            return None
        choices = [m for m in mounts if
                   ((controller == "v2" and m["kind"] == "cgroup2") or
                    (m["kind"] == "cgroup" and controller in m["controllers"]))
                   and member.is_relative_to(m["root"])]
        if not choices:
            return None
        # A less restrictive bind mount exposes additional ancestors.
        chosen = min(choices, key=lambda m: len(m["root"].parts))
        return {**chosen, "member": member,
                "leaf": chosen["mount"] / member.relative_to(chosen["root"])}

    cpu = mapping("cpu")
    version = 1 if cpu else 2
    cpu = cpu or mapping("v2")
    if cpu is None:
        raise RuntimeError("Cannot resolve the current visible CPU cgroup hierarchy")
    accounting = mapping("cpuacct") if version == 1 else cpu
    scopes = []
    directory, logical, depth = cpu["leaf"], cpu["member"], 0
    while True:
        usage_path = None
        if version == 2:
            usage_path = directory / "cpu.stat"
        elif accounting is not None and cpu["member"] == accounting["member"] and logical.is_relative_to(accounting["root"]):
            usage_path = accounting["mount"] / logical.relative_to(accounting["root"]) / "cpuacct.usage"
        scopes.append({"directory": directory, "usage_path": usage_path, "depth": depth,
                       "allow_absent_root_quota": directory == cpu["mount"] and logical == PurePosixPath("/")})
        if directory == cpu["mount"]:
            break
        directory, logical, depth = directory.parent, logical.parent, depth + 1
    return {"version": version, "scopes": scopes,
            "visible_ancestor_count": len(scopes),
            "mount_root_is_hierarchy_root": cpu["root"] == PurePosixPath("/"),
            "hidden_ancestor_limits": "unobservable beyond visible mount/cgroup namespace boundary"}


def _scope_quota(scope, version):
    directory = scope["directory"]
    if version == 1:
        quota = read_number(directory / "cpu.cfs_quota_us")
        period = read_number(directory / "cpu.cfs_period_us")
        if quota == -1 and period and period > 0:
            return None
        if quota is not None and quota > 0 and period and period > 0:
            return quota / period
    else:
        try:
            quota, period = (directory / "cpu.max").read_text().split()
            if int(period) <= 0:
                raise ValueError("nonpositive quota period")
            if quota == "max":
                return None
            if int(quota) > 0:
                return int(quota) / int(period)
        except FileNotFoundError:
            # Linux documents cpu.max on non-root cgroups. A real v2 root may
            # lack it; hidden host limits remain explicitly unobservable.
            if scope.get("allow_absent_root_quota"):
                return None
        except (OSError, ValueError):
            pass
    raise RuntimeError("An effective visible CPU quota is unreadable")


def cpu_scope_snapshot(hierarchy):
    return [{"depth": scope["depth"], "quota": _scope_quota(scope, hierarchy["version"]),
             "quota_observation": "absent_at_visible_root_boundary" if hierarchy["version"] == 2 and
                 scope.get("allow_absent_root_quota") and not (scope["directory"] / "cpu.max").exists()
                 else "read_from_visible_control_file",
             "cpu_s": scope_cpu_s(scope, hierarchy["version"])} for scope in hierarchy["scopes"]]


def quota_cores(hierarchy=None, affinity_count=None):
    hierarchy = hierarchy or resolve_cpu_hierarchy()
    values = [_scope_quota(scope, hierarchy["version"]) for scope in hierarchy["scopes"]]
    allowed = affinity_count if affinity_count is not None else len(os.sched_getaffinity(0))
    return min([float(allowed)] + [value for value in values if value is not None])


def cpu_allowance(requested, affinity_count, quota, other_cores=0., reserve=8):
    if type(requested) is not int or not 1 <= requested <= 60:
        raise ValueError("CPU request outside 1..60")
    return max(1, min(requested, affinity_count, int(quota),
                      int(max(1., quota - max(0., other_cores) - reserve))))


def scope_cpu_s(scope, version):
    path = scope["usage_path"]
    if path is None:
        return None
    if version == 1:
        value = read_number(path)
        return value / 1e9 if value is not None else None
    try:
        values = dict(line.split() for line in path.read_text().splitlines())
        return int(values["usage_usec"]) / 1e6
    except (OSError, KeyError, ValueError):
        return None


def process_tree(root_pid):
    records = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            records[int(directory.name)] = (int(fields[1]), int(fields[11]) + int(fields[12]))
        except (OSError, ValueError, IndexError):
            pass
    selected = {root_pid}
    while True:
        more = {pid for pid, (parent, _) in records.items() if parent in selected}
        if more <= selected:
            break
        selected |= more
    return selected, sum(records.get(pid, (0, 0))[1] for pid in selected) / os.sysconf("SC_CLK_TCK")


def restrict_tree(pids, cpus):
    for pid in pids:
        try:
            # Existing native threads also need their affinity updated.
            for tid in (Path("/proc") / str(pid) / "task").iterdir():
                try:
                    os.sched_setaffinity(int(tid.name), cpus)
                except (OSError, ValueError):
                    pass
        except OSError:
            pass


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def sync_results(root, remote):
    # Explicit result-only source; no dependencies, credentials, code or deletions.
    return subprocess.run(["rclone", "copy", str(root / "runs"), remote + "/runs",
        "--s3-no-check-bucket", "--transfers", "1", "--checkers", "1",
        "--include", "*.json", "--include", "*.jsonl", "--include", "*.log",
        "--include", "*.pt", "--include", "*.json.gz", "--include", "*.zip", "--exclude", "*",
        "--retries", "2", "--low-level-retries", "2", "--contimeout", "10s",
        "--timeout", "30s", "--stats", "0", "--log-level", "ERROR"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180).returncode


def signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
        return True
    except ProcessLookupError:
        return False


def cleanup_child(child, *, grace_s=90, clock=time.monotonic, sleep=time.sleep):
    """Always stop our own launched session, including surviving worker children."""
    started = clock()
    if child.poll() is None:
        try:
            # Give the trainer time to checkpoint before signalling its workers.
            child.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        while child.poll() is None and clock() - started < grace_s:
            sleep(min(1.0, max(.01, grace_s - (clock() - started))))
    live_group = signal_group(child.pid, signal.SIGTERM)
    if child.poll() is None:
        signal_group(child.pid, signal.SIGKILL)
    elif live_group:
        # The session leader has exited: remaining members are orphaned workers,
        # so no checkpointing leader remains to justify leaving them running.
        signal_group(child.pid, signal.SIGKILL)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        signal_group(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
    return {"child_returncode": child.returncode, "cleanup_completed": True,
            "cleanup_elapsed_s": clock() - started}


def guarded_monitor(child, monitor, *, cleanup=cleanup_child):
    """An exception in monitoring must not orphan the detached child session."""
    try:
        return monitor()
    finally:
        cleanup(child)


def sync_attempt(root, remote):
    try:
        code = sync_results(root, remote)
        return {"ok": code == 0, "returncode": code, "error_type": None}
    except Exception as exc:
        return {"ok": False, "returncode": None, "error_type": type(exc).__name__}


def finalize_sync(root, remote, status_path, status, *, attempt=sync_attempt):
    """Record both final transfers locally; never equate child success to sync.

    The second transfer publishes the terminal status from the first. Its own
    outcome is then saved locally, so external readback is explicitly required
    instead of claiming that a status write proves it uploaded itself.
    """
    first = attempt(root, remote)
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    status["final_sync_attempts"] = [first]
    status["remote_status_requires_readback"] = True
    save_json(status_path, status)
    second = attempt(root, remote)
    status["final_sync_attempts"].append(second)
    status["final_sync_ok"] = first["ok"] and second["ok"]
    save_json(status_path, status)
    return status["final_sync_ok"]


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--cpu-budget", type=int, default=50)
    parser.add_argument("--deadline", default="2026-09-12T02:00:00+00:00")
    parser.add_argument("--sync-seconds", type=int, default=120)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.parent != BASE or not root.name.startswith("q4-"):
        raise ValueError("Use the independent Q4 directory under the approved base")
    if not args.run or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run):
        raise ValueError("Invalid run name")
    if not 1 <= args.cpu_budget <= 60:
        raise ValueError("CPU budget must be within 1..60")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A child command is required")
    parsed_deadline = datetime.fromisoformat(args.deadline)
    if parsed_deadline.tzinfo is None:
        raise ValueError("Supervisor deadline requires an explicit timezone")
    deadline = parsed_deadline.timestamp()
    if time.time() >= deadline:
        raise ValueError("Training deadline has already passed")
    os.umask(0o077)
    run = root / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    lock = (root / ".q4-supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    allowed = sorted(os.sched_getaffinity(0))
    hierarchy = resolve_cpu_hierarchy()
    quota = quota_cores(hierarchy, len(allowed))
    maximum = cpu_allowance(args.cpu_budget, len(allowed), quota)
    active = maximum
    os.sched_setaffinity(0, allowed[:active])
    os.nice(10)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        PYTHONPATH=str(root / "src") + ":" + str(root), TMPDIR=str(root / ".tmp"))
    (root / ".tmp").mkdir(exist_ok=True)
    logfile = (run / "console.log").open("ab", buffering=0)
    started = time.time()
    child = None
    stop_sync = threading.Event()
    sync_status = {"attempts": 0, "successes": 0, "last_error_type": None}
    remote = REMOTE + root.name

    def synchronize():
        while not stop_sync.is_set():
            sync_status["attempts"] += 1
            result = sync_attempt(root, remote)
            sync_status["last_returncode"] = result["returncode"]
            sync_status["last_error_type"] = result["error_type"]
            if result["ok"]:
                sync_status["successes"] += 1
                sync_status["last_success_utc"] = datetime.now(timezone.utc).isoformat()
            stop_sync.wait(max(15, args.sync_seconds))

    sync_thread = threading.Thread(target=synchronize, daemon=True)
    terminate_at = None
    requested_stop = False

    def stop(signum, frame):
        nonlocal requested_stop
        requested_stop = True

    old_term = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)
    status = {}

    def monitor():
        nonlocal active, quota, terminate_at, status
        prior = None
        while True:
            now = time.time()
            pids, own_cpu = process_tree(os.getpid())
            scopes = cpu_scope_snapshot(hierarchy)
            quota = min([float(len(allowed))] + [s["quota"] for s in scopes if s["quota"] is not None])
            target = cpu_allowance(args.cpu_budget, len(allowed), quota)
            if prior is not None:
                elapsed = max(now - prior[0], 1e-6)
                own_delta = max(0., own_cpu - prior[2])
                for scope, previous in zip(scopes, prior[1]):
                    if scope["cpu_s"] is not None and previous["cpu_s"] is not None:
                        other = max(0., (scope["cpu_s"] - previous["cpu_s"] - own_delta) / elapsed)
                        scope["other_cores_estimate"] = other
                        # An unlimited root still exposes contention. Use our
                        # allowed processor count as a conservative capacity.
                        capacity = scope["quota"] if scope["quota"] is not None else len(allowed)
                        target = min(target, cpu_allowance(args.cpu_budget, len(allowed), capacity, other))
            prior = (now, scopes, own_cpu)
            active = min(target, active + 2) if target > active else target
            restrict_tree(pids, allowed[:active])
            status = {"schema": 2, "run": args.run, "server_role": "q4-port-42222",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "supervisor_pid": os.getpid(), "child_pid": child.pid,
                "child_returncode": child.poll(), "elapsed_s": now - started,
                "cpu_quota_cores": quota, "requested_cpu_budget": args.cpu_budget,
                "active_affinity_cores": active, "cpu_scopes": scopes,
                "cgroup_version": hierarchy["version"],
                "visible_ancestor_count": hierarchy["visible_ancestor_count"],
                "hidden_ancestor_limits": hierarchy["hidden_ancestor_limits"],
                "mount_root_is_hierarchy_root": hierarchy["mount_root_is_hierarchy_root"],
                "process_count": len(pids), "gpu_enabled": False,
                "deadline_utc": args.deadline, "termination_requested": terminate_at is not None,
                "sync": dict(sync_status)}
            save_json(run / "supervisor.json", status)
            # Preserve the resource trajectory as well as the latest snapshot.
            # This uses only our own job's metadata and is synced with results.
            with (run / "resources.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(status, separators=(",", ":")) + "\n")
            if child.poll() is not None:
                return
            if (now >= deadline or requested_stop) and terminate_at is None:
                terminate_at = now
                try:
                    child.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif terminate_at is not None and now - terminate_at > 90:
                signal_group(child.pid, signal.SIGKILL)
            time.sleep(5)

    failed = False
    try:
        child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
            stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        # Guard covers thread startup and every subsequent monitoring operation.
        def start_and_monitor():
            sync_thread.start()
            monitor()
        guarded_monitor(child, start_and_monitor)
    except BaseException as exc:
        failed = True
        status["supervision_error_type"] = type(exc).__name__
    finally:
        stop_sync.set()
        if sync_thread.ident is not None:
            sync_thread.join(timeout=190)
        status["child_returncode"] = child.poll() if child is not None else None
        status["supervisor_failed"] = failed
        try:
            synced = finalize_sync(root, remote, run / "supervisor.json", status)
            failed = failed or not synced
        except Exception as exc:
            failed = True
            # A full/unwritable filesystem may prevent even terminal JSON. Emit
            # only a safe class name so the detached launch log retains evidence.
            print(json.dumps({"terminal_status_error_type": type(exc).__name__,
                              "final_sync_ok": False}), flush=True)
        logfile.close()
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        lock.close()
    return child.returncode if child is not None and child.returncode else (1 if failed else 0)


if __name__ == "__main__":
    raise SystemExit(main())
