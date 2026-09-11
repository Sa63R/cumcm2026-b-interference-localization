"""Private-job CPU limits and provenance checks; never touches GPU or other jobs."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import time


def _text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_quota():
    limits = []
    value = _text("/sys/fs/cgroup/cpu.max")
    if value:
        quota, period = value.split()
        if quota != "max":
            limits.append(int(quota)/int(period))
    for directory in ("/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"):
        quota, period = _text(directory+"/cpu.cfs_quota_us"), _text(directory+"/cpu.cfs_period_us")
        if quota and period and int(quota) > 0:
            limits.append(int(quota)/int(period))
    return min(limits) if limits else None


def _core_order(allowed):
    """One hardware thread per physical core first, spread across sockets."""
    text = _text("/proc/cpuinfo")
    if not text:
        return sorted(allowed)
    sockets = {}
    for block in text.split("\n\n"):
        values = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
        values = {k.strip(): v.strip() for k,v in values.items()}
        if "processor" not in values or int(values["processor"]) not in allowed:
            continue
        sockets.setdefault(values.get("physical id", "0"), {}).setdefault(
            values.get("core id", values["processor"]), []).append(int(values["processor"]))
    socket_rows = [sorted((min(cpus) for cpus in cores.values())) for _, cores in sorted(sockets.items())]
    first = [row[i] for i in range(max(map(len, socket_rows), default=0)) for row in socket_rows if i < len(row)]
    return first + sorted(set(allowed)-set(first))


def constrain(root, slots=50, nice_increment=10):
    if not 1 <= slots <= 60:
        raise ValueError("Total CPU budget must be in 1..60; default is 50")
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("The remote job requires Linux inherited CPU affinity")
    root = Path(root).resolve()
    allowed = os.sched_getaffinity(0)
    quota = cpu_quota()
    cap = min(slots, len(allowed), int(quota) if quota is not None else len(allowed))
    if cap < 1:
        raise RuntimeError("No CPU slots available")
    selected = _core_order(allowed)[:cap]
    os.sched_setaffinity(0, set(selected))
    if os.sched_getaffinity(0) != set(selected):
        raise RuntimeError("CPU affinity was not applied")
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < nice_increment:
        os.nice(nice_increment-current_nice)
    runtime = root/"runtime"
    for name in ("tmp", "cache", "torch", "pycache"):
        (runtime/name).mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        GOMAXPROCS="1", TMPDIR=str(runtime/"tmp"), TMP=str(runtime/"tmp"), TEMP=str(runtime/"tmp"),
        XDG_CACHE_HOME=str(runtime/"cache"), TORCH_HOME=str(runtime/"torch"),
        PYTHONPYCACHEPREFIX=str(runtime/"pycache"), PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    return dict(cpu_slots=cap, requested_slots=slots, visible_quota_slots=quota,
                affinity_cpus=selected, nice=os.getpriority(os.PRIO_PROCESS, 0),
                scope="Current process and all descendants, including learner, workers, evaluators and sync",
                platform=platform.system(), python=platform.python_version(), captured_unix=time.time())


def inside(root, relative):
    root, value = Path(root).resolve(), Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Use a relative path inside the private job directory")
    result = (root/value).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Path escaped the private job directory")
    return result


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)
