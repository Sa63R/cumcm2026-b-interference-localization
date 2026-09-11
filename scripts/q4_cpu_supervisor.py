"""Supervise one CPU-only Q4 job and sync results through the approved S3 prefix.

The entire process tree shares a restricted affinity set, so sampler, learner,
evaluation and sync processes cannot collectively exceed the requested cores.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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


def quota_cores():
    # Both cgroup v1 and v2 are in use on the available servers.
    for directory in (Path("/sys/fs/cgroup/cpu"), Path("/sys/fs/cgroup/cpu,cpuacct")):
        quota = read_number(directory / "cpu.cfs_quota_us")
        period = read_number(directory / "cpu.cfs_period_us")
        if quota and quota > 0 and period:
            return quota / period
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    return float(len(os.sched_getaffinity(0)))


def cpu_allowance(requested, affinity_count, quota, other_cores=0., reserve=8):
    if type(requested) is not int or not 1 <= requested <= 60:
        raise ValueError("CPU request outside 1..60")
    return max(1, min(requested, affinity_count, int(quota),
                      int(max(1., quota - max(0., other_cores) - reserve))))


def scope_cpu_s():
    for p in ("/sys/fs/cgroup/cpuacct/cpuacct.usage",
              "/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage", "/sys/fs/cgroup/cpu/cpuacct.usage"):
        value = read_number(p)
        if value is not None:
            return value / 1e9
    try:
        values = dict(line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
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
        "--include", "*.pt", "--include", "*.json.gz", "--exclude", "*",
        "--retries", "2", "--low-level-retries", "2", "--contimeout", "10s",
        "--timeout", "30s", "--stats", "0", "--log-level", "ERROR"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180).returncode


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
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    if time.time() >= deadline:
        raise ValueError("Training deadline has already passed")
    os.umask(0o077)
    run = root / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    lock = (root / ".q4-supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    allowed = sorted(os.sched_getaffinity(0))
    quota = quota_cores()
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
    child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
        stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
    stop_sync = threading.Event()
    sync_status = {"attempts": 0, "successes": 0, "last_error_type": None}
    remote = REMOTE + root.name

    def synchronize():
        while not stop_sync.is_set():
            sync_status["attempts"] += 1
            try:
                result = sync_results(root, remote)
                sync_status["last_returncode"] = result
                if result == 0:
                    sync_status["successes"] += 1
                    sync_status["last_success_utc"] = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                sync_status["last_error_type"] = type(exc).__name__
            stop_sync.wait(max(15, args.sync_seconds))

    sync_thread = threading.Thread(target=synchronize, daemon=True)
    sync_thread.start()
    terminate_at = None
    requested_stop = False

    def stop(signum, frame):
        nonlocal requested_stop
        requested_stop = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    prior = None
    status = {}
    while True:
        now = time.time()
        pids, own_cpu = process_tree(os.getpid())
        total_cpu = scope_cpu_s()
        other_cores = None
        if prior is not None and total_cpu is not None and prior[1] is not None:
            elapsed = max(now - prior[0], 1e-6)
            # Exited processes make the own-tree cumulative count decrease;
            # clamp conservatively and never overestimate spare capacity.
            own_delta = max(0., own_cpu - prior[2])
            other_cores = max(0., (total_cpu - prior[1] - own_delta) / elapsed)
            available = max(1, int(quota - other_cores - 8))
            target = min(maximum, available)
            if target < active:
                active = target
            elif target > active:
                active = min(target, active + 2)
        prior = (now, total_cpu, own_cpu)
        restrict_tree(pids, allowed[:active])
        status = {"schema": 1, "run": args.run, "server_role": "q4-port-42222",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "supervisor_pid": os.getpid(), "child_pid": child.pid,
            "child_returncode": child.poll(), "elapsed_s": now - started,
            "cpu_quota_cores": quota, "requested_cpu_budget": args.cpu_budget,
            "active_affinity_cores": active, "other_scope_cores_estimate": other_cores,
            "process_count": len(pids), "gpu_enabled": False,
            "deadline_utc": args.deadline, "termination_requested": terminate_at is not None,
            "sync": dict(sync_status)}
        save_json(run / "supervisor.json", status)
        if child.poll() is not None:
            break
        if (now >= deadline or requested_stop) and terminate_at is None:
            terminate_at = now
            os.killpg(child.pid, signal.SIGTERM)
        elif terminate_at is not None and now - terminate_at > 90:
            os.killpg(child.pid, signal.SIGKILL)
        time.sleep(5)
    stop_sync.set()
    sync_thread.join(timeout=190)
    status["final_sync_returncode"] = sync_results(root, remote)
    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(run / "supervisor.json", status)
    sync_results(root, remote)
    logfile.close()
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
