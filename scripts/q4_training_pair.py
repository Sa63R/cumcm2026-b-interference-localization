"""Launch macro/micro trainers inside one existing Q4 CPU supervisor session.

JSON: {"run":"pair-id","jobs":[{"name":"macro","module":"q4_rl.train",
"argv":["--output","runs/pair-id/macro/training", ...]}, {"name":"micro",
"module":"q4_rl.micro_train","argv":["--output","runs/pair-id/micro/training", ...]}]}.
No shell, detached child session, affinity change, or additional resource budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


MODULES = frozenset(("q4_rl.train", "q4_rl.micro_train"))
INTEGER_FLAGS = frozenset(("--workers", "--cpu-budget", "--hidden", "--epochs",
    "--minibatch-size", "--batch-episodes", "--warmstart-episodes", "--max-decisions",
    "--random-seed", "--scenario-start", "--scenario-end", "--max-batches"))
FLOAT_FLAGS = frozenset(("--learning-rate", "--entropy-coefficient",
    "--max-wall-seconds", "--progress-interval-seconds"))
ALLOWED_FLAGS = INTEGER_FLAGS | FLOAT_FLAGS | {"--output", "--deadline"}
GRACE_SECONDS = 80.0


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("run/job identifiers must be short ASCII names")
    return value


def _inside(root, value):
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError("invalid task-relative path")
    path = Path(value)
    resolved = (root/path).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError("path must remain strictly inside the current task root")
    return resolved


def validate_config(config, root):
    if not isinstance(config, dict) or set(config) != {"run", "jobs"}:
        raise ValueError("configuration requires exactly run and jobs")
    run = _inside(root, "runs/"+_identifier(config["run"]))
    jobs = config["jobs"]
    if not isinstance(jobs, list) or len(jobs) != 2:
        raise ValueError("exactly two trainer jobs are required")
    normalized, names, modules, outputs = [], set(), set(), []
    for job in jobs:
        if not isinstance(job, dict) or set(job) != {"name", "module", "argv"}:
            raise ValueError("each job requires exactly name, module and argv")
        name = _identifier(job["name"])
        module = job["module"]
        if name in names or not isinstance(module, str) or module not in MODULES or module in modules:
            raise ValueError("one unique macro and one unique micro trainer are required")
        argv = job["argv"]
        if (not isinstance(argv, list) or not argv or len(argv) % 2 or
                any(not isinstance(v, str) or not v or any(ord(c) < 32 for c in v) for v in argv)):
            raise ValueError("argv must contain explicit flag/value string pairs")
        flags = {}
        for flag, value in zip(argv[::2], argv[1::2]):
            if flag not in ALLOWED_FLAGS or flag in flags:
                raise ValueError("unknown/duplicate trainer flag; resume and shell commands are forbidden")
            if flag in INTEGER_FLAGS:
                if not re.fullmatch(r"[0-9]+", value):
                    raise ValueError("trainer integer option is invalid")
                if int(value) == 0 and flag not in {"--warmstart-episodes", "--random-seed"}:
                    raise ValueError("trainer integer option must be positive")
            elif flag in FLOAT_FLAGS:
                number = float(value)
                if not math.isfinite(number) or number < 0 or (number == 0 and flag != "--entropy-coefficient"):
                    raise ValueError("trainer numeric option is invalid")
            elif flag == "--deadline":
                if datetime.fromisoformat(value).tzinfo is None:
                    raise ValueError("trainer deadline requires an explicit timezone")
            flags[flag] = value
        if "--output" not in flags:
            raise ValueError("each trainer must have an explicit empty output directory")
        output = _inside(root, flags["--output"])
        expected = _inside(root, (run.relative_to(root)/name/"training").as_posix())
        if output != expected or not output.is_relative_to(run):
            raise ValueError("trainer output must be runs/<run>/<job>/training for supervisor synchronization")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("trainer output must be absent or an empty directory")
        if any(output == other or output.is_relative_to(other) or other.is_relative_to(output) for other in outputs):
            raise ValueError("trainer output directories must be disjoint")
        # Do not log an absolute machine/user path or let a later cwd change
        # alter where this validated trainer writes.
        relative = output.relative_to(root).as_posix()
        canonical = [item for flag, value in zip(argv[::2], argv[1::2])
                     for item in (flag, relative if flag == "--output" else value)]
        normalized.append(dict(name=name, module=module, argv=canonical, output=relative))
        names.add(name)
        modules.add(module)
        outputs.append(output)
    return run, normalized


def require_outer_supervisor(root):
    if os.name != "posix" or not hasattr(os, "getpgrp"):
        raise ValueError("this launcher requires the Linux Q4 CPU supervisor")
    # q4_cpu_supervisor starts this wrapper as its session leader and performs
    # process-group cleanup even after this wrapper exits. Without that owner,
    # an 80-second timeout could orphan still-checkpointing worker processes.
    command = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    tokens = [v.decode("utf-8") for v in command if v]
    script = root/"scripts/q4_cpu_supervisor.py"
    owned = any((root/word).resolve() == script.resolve() for word in tokens[1:] if word.endswith("q4_cpu_supervisor.py"))
    owned = owned or any(a == "-m" and b == "scripts.q4_cpu_supervisor" for a, b in zip(tokens, tokens[1:]))
    if not owned or os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        raise ValueError("launch directly under q4_cpu_supervisor; do not detach the trainers")


def _save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def run_pair(run, jobs, root):
    run.mkdir(parents=True, exist_ok=True)
    requested = {"signal": None}
    status = dict(run=run.name, jobs=[dict(job, pid=None, returncode=None, started=False) for job in jobs],
        stop_reason=None, stop_signal=None, grace_seconds=GRACE_SECONDS, supervisor_cleanup_required=False)
    # A reused run identifier cannot overwrite another pair's logs/status.
    with (run/"training-pair-launch.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(run=run.name, jobs=jobs), stream, ensure_ascii=False, indent=2)
    processes, logs = [], []
    stopping_at = None
    began = time.monotonic()
    result = 0

    def handle(signum, frame):
        if requested["signal"] is None:
            requested["signal"] = signum

    old = {s: signal.signal(s, handle) for s in (signal.SIGTERM, signal.SIGINT)}

    def stop(reason):
        nonlocal stopping_at
        if stopping_at is None:
            stopping_at = time.monotonic()
            status["stop_reason"] = reason
            status["stop_signal"] = requested["signal"]
            for process in processes:
                if process.poll() is None:
                    try:
                        # Signal learners only; each controls its checkpoint and
                        # pool shutdown. The outer owner later reaps the group.
                        process.send_signal(signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    def snapshot():
        for row, process in zip(status["jobs"], processes):
            row["returncode"] = process.poll()
        status["elapsed_s"] = time.monotonic()-began
        _save(run/"training-pair.json", status)

    try:
        for job in jobs:
            logs.append((run/(job["name"]+".log")).open("xb", buffering=0))
        for job, row, log in zip(jobs, status["jobs"], logs):
            if requested["signal"] is not None:
                stop("signal")
                break
            process = subprocess.Popen([sys.executable, "-B", "-m", job["module"], *job["argv"]],
                cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            row.update(pid=process.pid, started=True)
        while True:
            snapshot()
            if requested["signal"] is not None:
                stop("signal")
            failures = [(row["name"], p.returncode) for row, p in zip(status["jobs"], processes)
                        if p.returncode not in (None, 0)]
            if failures:
                if stopping_at is None:
                    status["failure_job"], status["failure_returncode"] = failures[0]
                stop("peer_failure")
            if all(p.poll() is not None for p in processes):
                break
            if stopping_at is not None and time.monotonic()-stopping_at >= GRACE_SECONDS:
                status["supervisor_cleanup_required"] = True
                result = 124
                break
            time.sleep(.25)
    except BaseException as exc:
        status["launcher_error_type"] = type(exc).__name__
        result = 1
        stop("launcher_error")
        while any(p.poll() is None for p in processes) and time.monotonic()-stopping_at < GRACE_SECONDS:
            time.sleep(.25)
        status["supervisor_cleanup_required"] = any(p.poll() is None for p in processes)
    finally:
        for row, process in zip(status["jobs"], processes):
            row["returncode"] = process.poll()
        if not result:
            failed = status.get("failure_returncode", next(
                (p.returncode for p in processes if p.returncode not in (None, 0)), None))
            result = (failed if failed > 0 else 128-failed) if failed is not None else (
                128+requested["signal"] if requested["signal"] is not None else 0)
        status["returncode"] = result
        status["elapsed_s"] = time.monotonic()-began
        try:
            _save(run/"training-pair.json", status)
            print(json.dumps(status, allow_nan=False), flush=True)
        finally:
            for log in logs:
                log.close()
            for signum, handler in old.items():
                signal.signal(signum, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    path = _inside(root, args.config)
    run, jobs = validate_config(json.loads(path.read_text(encoding="utf-8-sig")), root)
    require_outer_supervisor(root)
    return run_pair(run, jobs, root)


if __name__ == "__main__":
    raise SystemExit(main())
