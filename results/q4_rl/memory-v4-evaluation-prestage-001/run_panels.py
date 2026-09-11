"""Explicitly launch the two frozen development panels, sequentially."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

TASK = "q4-rl-memory-v4-eval-20260912"
ROOT = Path("/home/dataset-assist-0/usr/lh/ysh/bwc/shumo") / TASK
CONTROL = ROOT / "launch/eval-memory-v4"
PYTHON = "/home/dataset-assist-0/usr/lh/ysh/bwc/shumo/q4-deep-rl-20260911/.venv-cpu/bin/python"
HOST = "ide-376f3dcbf2424192b9d1abca3872afc8-445523"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(plan):
    if socket.gethostname() != HOST or Path.cwd().resolve() != ROOT:
        raise ValueError("Wrong evaluation target")
    if not set(range(50)) <= os.sched_getaffinity(0):
        raise ValueError("The shared CPU 0..49 affinity pool is unavailable")
    os.sched_setaffinity(0, set(range(50)))
    if plan["task"] != TASK or plan["workers"] != 10 or plan["bootstrap_samples"] != 5000:
        raise ValueError("Frozen resource or inference configuration changed")
    for relative, expected in plan["control_sha256"].items():
        if sha(CONTROL / relative) != expected:
            raise ValueError("Frozen evaluation control changed")
    source = json.loads((ROOT / "EVAL_SOURCE_MANIFEST.json").read_text())
    if sha(ROOT / "EVAL_SOURCE_MANIFEST.json") != plan["source_manifest_sha256"]:
        raise ValueError("Evaluation source binding changed")
    for relative, item in source["files"].items():
        if sha(ROOT / relative) != item["sha256"]:
            raise ValueError("Frozen production source changed: " + relative)
    expected_python = {p for p in source["files"] if p.startswith("src/") and p.endswith(".py")}
    actual_python = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")}
    if actual_python != expected_python:
        raise ValueError("Unexpected production source addition/removal")
    for item in plan["models"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Frozen model changed: " + item["alias"])
    specs = json.loads((CONTROL / "specs.json").read_text())
    if len(specs) != 13 or set(specs) != set(plan["arms"]):
        raise ValueError("Expected all thirteen unchanged paired methods")
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Explicit root-reviewed launch; otherwise preflight only")
    args = parser.parse_args()
    plan = json.loads((CONTROL / "plan.json").read_text())
    verify_inputs(plan)
    if not args.launch:
        print(json.dumps({"preflight_passed": True, "launched": False, "panels": plan["panels"], "arms": plan["arms"]}))
        return 0
    # The existing supervisor protects its own root against concurrent launches.
    # Require both panel directories to be new before starting the first panel.
    if any((ROOT / "runs" / p["run"]).exists() for p in plan["panels"]):
        raise ValueError("A declared panel output already exists; never overwrite/retry silently")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    stopped, child = False, None
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for panel in plan["panels"]:
        if stopped:
            break
        verify_inputs(plan)
        deadline = datetime.fromtimestamp(time.time() + panel["maximum_wall_seconds"], timezone.utc).isoformat()
        command = [PYTHON, "scripts/q4_cpu_supervisor.py", "--root", str(ROOT), "--run", panel["run"],
            "--cpu-budget", "50", "--deadline", deadline, "--sync-seconds", "120", "--minimum-free-gib", "20", "--",
            PYTHON, "launch/eval-memory-v4/run_panel.py", "--panel", panel["run"]]
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL)
        child.wait()
        status_path = ROOT / "runs" / panel["run"] / "supervisor.json"
        status = json.loads(status_path.read_text())
        output = ROOT / "runs" / panel["run"] / "evaluation"
        # A fully recorded failed strategy may return 1. Administrative partial
        # panels, sync failures or missing evidence must not advance the queue.
        complete = (status.get("child_returncode") in (0, 1) and not status.get("supervisor_failed")
            and not status.get("termination_requested") and status.get("stop_reason") is None
            and status.get("final_sync_ok") is True and (output / "summary.json").is_file()
            and (output / "evidence.json").is_file() and (output / "panel_complete.json").is_file())
        print(json.dumps({"panel": panel["run"], "complete_and_synced": complete, "supervisor_returncode": child.returncode}), flush=True)
        if not complete or stopped:
            return 1
        child = None
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
