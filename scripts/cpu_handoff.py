"""Portable CPU-only operator workflow. Uses local research simulation only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def write_gzip_json(path, value):
    """A stopped evaluation must never leave a readable-looking partial case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def hardware():
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        q, period = quota.read_text().split()
        if q != "max":
            available = min(available, max(1, int(q)//int(period)))
    memory = None
    proc = Path("/proc/meminfo")
    if proc.exists():
        memory = int(next(x.split()[1] for x in proc.read_text().splitlines() if x.startswith("MemAvailable:")))*1024
    memmax, memnow = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if memmax.exists() and memnow.exists() and memmax.read_text().strip() != "max":
        remaining = max(0, int(memmax.read_text())-int(memnow.read_text()))
        memory = min(memory, remaining) if memory is not None else remaining
    return {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
            "logical_cpus": os.cpu_count(), "available_cpu_slots": available,
            "available_memory_gib": memory/2**30 if memory is not None else None,
            "gpu_policy": "CPU-only PyTorch required; accelerator visibility disabled"}


def verify_package():
    path = ROOT / "PACKAGE_MANIFEST.json"
    if not path.exists():
        raise ValueError("Use the built handoff package, or explicitly build it first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        file = (ROOT/name).resolve()
        if not file.is_relative_to(ROOT) or not file.is_file() or sha256(file) != expected:
            raise ValueError(f"Package file missing or modified: {name}")
    os.environ["Q3_SOURCE_COMMIT"] = manifest["git_commit"]
    return manifest


def choose_worker_counts(info, maximum=96):
    cap = min(maximum, info["available_cpu_slots"])
    memory = info["available_memory_gib"]
    if memory is not None:
        # Conservative initial allowance for Torch processes and trajectory arrays.
        cap = min(cap, max(0, int((memory-3)/.75)))
    return sorted({min(cap, n) for n in (16, 32, 64, 96) if min(cap, n) >= 1})


def validate_cached_benchmark(result, info, maximum, smoke):
    cap = max(choose_worker_counts(info, maximum), default=0)
    if (not isinstance(result.get("selected_workers"), int)
            or not 1 <= result["selected_workers"] <= cap
            or not isinstance(result.get("learner_threads"), int)
            or not 1 <= result["learner_threads"] <= min(4, info["available_cpu_slots"])
            or result.get("episodes_per_update") != (2 if smoke else 128)):
        raise ValueError("Cached benchmark exceeds current CPU/memory limits or differs from this run; use a new output directory")
    return result


def training_command(out, *, seed, scenario, workers, threads, episodes, updates,
                     seconds, checkpoint, resume=False, learning_rate=1e-4, smoke=False,
                     scenario_end=None, max_attempted_episodes=None, deadline_epoch=None):
    command = [sys.executable, "-m", "research_rl.train", "--output", str(out),
               "--device", "cpu", "--feature-version", "v3", "--hidden", "96",
               "--architecture", "mlp", "--probe-candidates", "base", "--group-alpha", "0",
               "--seed", str(seed), "--scenario-start", str(scenario), "--bc-episodes", "0",
               "--workers", str(workers), "--num-threads", str(threads),
               "--episodes-per-update", str(episodes), "--updates", str(updates),
               "--lr", str(learning_rate), "--gae-lambda", ".95", "--entropy-coef", ".005",
               "--aux-bc-coef", "0", "--epochs", "1" if smoke else "4",
               "--minibatch", "128", "--max-wall-s", str(max(.1, seconds)),
               "--checkpoint-seconds", "600"]
    if scenario_end is not None:
        command += ["--scenario-end", str(scenario_end)]
    if max_attempted_episodes is not None:
        command += ["--max-attempted-episodes", str(max_attempted_episodes)]
    if deadline_epoch is not None:
        command += ["--deadline-utc", datetime.fromtimestamp(deadline_epoch, timezone.utc).isoformat()]
    command += ["--resume" if resume else "--initialize-from", str(checkpoint)]
    return command


def stop_process_group(process, grace_seconds=5):
    """Reap the learner and stop its pool even if its leader exits first."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    finally:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def run_training(command, log_path, *, deadline_epoch):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"src"), PYTHONUNBUFFERED="1")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\nCOMMAND "+json.dumps(command)+"\n")
        stream.flush()
        remaining = deadline_epoch - time.time()
        if remaining <= 0:
            stream.write("Training was not launched: outer process deadline reached.\n")
            return 124
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
        try:
            return process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            stream.write("Outer process deadline reached; stopping learner and workers.\n")
            stream.flush()
            stop_process_group(process)
            return 124
        except BaseException:
            stop_process_group(process)
            raise


def read_training_rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # A partial last line from interrupted output is diagnostic, never an update.
            result.append({"stage": "incomplete_log_line"})
    return result


def benchmark(output, info, checkpoint, max_workers, smoke, deadline_epoch):
    target = output/"benchmark.json"
    if target.exists():
        return validate_cached_benchmark(json.loads(target.read_text()), info, max_workers, smoke)
    cap = max(choose_worker_counts(info, max_workers), default=0)
    if cap < 1:
        raise ValueError("Insufficient CPU or memory allowance for even one training worker")
    counts = sorted({min(cap, n) for n in (1, 2)}) if smoke else choose_worker_counts(info, max_workers)
    batch = 2 if smoke else 128
    entries = []
    for workers in counts:
        remaining = deadline_epoch-time.time()-120
        if remaining <= 5:
            raise ValueError("Deadline reached before CPU benchmark completion")
        folder = output/"benchmark"/f"workers-{workers}"
        if folder.exists():
            raise ValueError("Incomplete benchmark directory; use a new output for a clean rerun")
        seconds = min(remaining, 120 if smoke else 600)
        command = training_command(folder, seed=9112190, scenario=1900001,
            workers=workers, threads=1 if smoke else min(4, info["available_cpu_slots"]),
            episodes=batch, updates=2 if smoke else 3, seconds=seconds,
            checkpoint=checkpoint, learning_rate=0., smoke=smoke,
            scenario_end=1900384, max_attempted_episodes=batch*(2 if smoke else 3),
            deadline_epoch=deadline_epoch-120)
        print(f"Benchmark: {workers} CPU workers", flush=True)
        code = run_training(command, output/"logs"/f"benchmark-{workers}.log",
                            deadline_epoch=min(deadline_epoch-60, time.time()+seconds+30))
        rows = [r for r in read_training_rows(folder/"training.jsonl") if r.get("stage")=="ppo"]
        steady = rows[1:]
        elapsed = sum(r["collect_wall_s"]+r["optimize_wall_s"] for r in steady)
        valid = (code==0 and len(rows)==(2 if smoke else 3)
                 and all(e.get("success") for r in rows for e in r["episodes"]))
        entries.append({"workers": workers, "returncode": code, "valid": valid,
                        "updates": len(rows), "steady_episode_rate_s":
                        sum(len(r["episodes"]) for r in steady)/elapsed if elapsed else 0,
                        "steady_collect_s": sum(r["collect_wall_s"] for r in steady),
                        "steady_optimize_s": sum(r["optimize_wall_s"] for r in steady)})
        write_json(output/"benchmark-progress.json", entries)
    good = [r for r in entries if r["valid"]]
    if not good:
        raise ValueError("No complete CPU benchmark; return logs before launching long training")
    best = max(good, key=lambda r: (r["steady_episode_rate_s"], -r["workers"]))
    result = {"configurations": entries, "selected_workers": best["workers"],
              "learner_threads": 1 if smoke else min(4, info["available_cpu_slots"]),
              "episodes_per_update": batch,
              "scope": "Fixed parent weights, learning rate zero, first update excluded; throughput only, no performance claim"}
    write_json(target, result)
    return result


def evaluate(checkpoint, output, smoke, deadline_epoch):
    import torch
    torch.set_num_threads(1)
    from experiments.research_v1_eval import run_case, summarize
    limits = {"limits": {"real_seconds_per_case": 300, "virtual_seconds_per_case": 360000, "max_actions": 10000}}
    from simulation import random_scenario
    spec = {"name": output.name, "entrypoint": "research_rl:run_rl_search",
            "kwargs": {"checkpoint": str(checkpoint), "device": "cpu", "num_threads": 1}}
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    checkpoint_hash = sha256(checkpoint)
    for seed in range(2100001, 2100003 if smoke else 2100049):
        remaining = deadline_epoch-time.time()-10
        file = output/f"case-{seed}.json.gz"
        if file.exists():
            with gzip.open(file, "rt", encoding="utf-8") as stream:
                record = json.load(stream)
            if record.get("checkpoint_sha256") != checkpoint_hash:
                raise ValueError("Evaluation checkpoint changed; preserve old records")
        else:
            if remaining <= 2:
                break
            record = run_case(random_scenario(3, seed), spec, limits, real_budget=min(300, remaining))
            record["checkpoint_sha256"] = checkpoint_hash
            write_gzip_json(file, record)
        rows.append(record["row"])
    result = summarize(rows, 2 if smoke else 48)
    result["scope"] = "Development evaluation; no final-test or statistical selection claim"
    result["checkpoint_sha256"] = checkpoint_hash
    write_json(output/"summary.json", result)
    return result


def export_results(output):
    if not output.is_dir() or not (output/"run_manifest.json").is_file():
        raise ValueError("Expected a CPU run directory containing run_manifest.json")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("Result symlinks are not exported")
    write_json(output/"RESULT_FILES.json", {p.relative_to(output).as_posix(): sha256(p)
                                           for p in files if p.name != "RESULT_FILES.json"})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output.parent/f"{output.name}-results-{stamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(output, arcname=output.name)
    archive.with_suffix(archive.suffix+".sha256").write_text(sha256(archive)+"  "+archive.name+"\n")
    print("RESULT_ARCHIVE "+str(archive), flush=True)
    return archive


def enforce_package_mode(manifest, smoke):
    if manifest.get("dirty_source_smoke_only") and not smoke:
        raise ValueError("This package contains uncommitted source and is restricted to --smoke")


def request_shutdown(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}; stopping CPU workers and exporting retained results")


def trial_status(state, *, scenario, scenario_end, max_attempted, max_updates,
                 trial_seconds, deadline_epoch, now=None, returncode=None):
    """Derive completion from saved progress, including a lost parent marker."""
    now = time.time() if now is None else now
    saved = state or {}
    updates = int(saved.get("update", 0))
    steps = int(saved.get("optimizer_steps", 0))
    next_seed = int(saved.get("next_seed", scenario))
    attempted = max(int(saved.get("attempted_episodes", saved.get("episodes", 0))),
                    next_seed-scenario)
    elapsed = float(saved.get("elapsed_training_s", 0.))
    violation = (next_seed < scenario or next_seed > scenario_end+1 or attempted > max_attempted
                 or attempted < 0 or not math.isfinite(elapsed) or elapsed < 0
                 or ("attempted_episodes" in saved and saved["attempted_episodes"] != next_seed-scenario))
    sample_limit = attempted >= max_attempted or next_seed > scenario_end
    update_limit = updates >= max_updates
    time_limit = elapsed >= trial_seconds
    budget_exhausted = sample_limit or update_limit or time_limit
    global_deadline = now >= deadline_epoch-122 or saved.get("stop_reason") == "global_deadline"
    complete = bool(state and steps > 0 and budget_exhausted and not violation
                    and returncode in (None, 0))
    if violation:
        reason, status = "budget_or_partition_violation", "invalid"
    elif returncode not in (None, 0):
        reason, status = "process_timeout" if returncode == 124 else "process_failure", "interrupted"
    elif complete:
        reason = "update_limit" if update_limit else "attempted_episode_limit" if sample_limit else "wall_time_limit"
        status = "complete"
    elif state and steps == 0 and budget_exhausted:
        reason, status = "no_optimizer_progress", "incomplete"
    elif global_deadline:
        reason, status = "global_deadline", "incomplete"
    elif not state:
        reason, status = "not_started", "not_started"
    else:
        reason, status = saved.get("stop_reason") or "interrupted", "incomplete"
    return {"status": status, "stop_reason": reason,
            "training_budget_complete": complete,
            "can_resume": not violation and not budget_exhausted and not global_deadline,
            "updates": updates, "optimizer_steps": steps, "attempted_episodes": attempted,
            "elapsed_training_s": elapsed, "next_seed": next_seed,
            "scenario_start": scenario, "scenario_end_inclusive": scenario_end,
            "max_attempted_episodes": max_attempted, "max_updates": max_updates,
            "trial_seconds": trial_seconds, "returncode": returncode,
            "completion_scope": "Declared training budget exhausted with optimizer progress; evaluation completeness is reported separately"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "run", "export"))
    parser.add_argument("--output", type=Path, default=ROOT/"cpu_runs"/"run01")
    parser.add_argument("--max-workers", type=int, default=96)
    parser.add_argument("--trial-minutes", type=float, default=60)
    parser.add_argument("--max-episodes-per-trial", type=int, default=199936)
    parser.add_argument("--deadline", default="2026-09-12T10:00:00+08:00")
    parser.add_argument("--smoke", action="store_true", help="Short functional test; never a performance experiment")
    args = parser.parse_args()
    if args.max_workers < 1 or not math.isfinite(args.trial_minutes) or args.trial_minutes <= 0 or not 128 <= args.max_episodes_per_trial <= 199999:
        parser.error("Invalid resource or episode limits")
    manifest = verify_package()
    info = hardware()
    if args.command == "inspect":
        print(json.dumps(info, indent=2))
        return 0
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/"cpu_runs"):
        raise ValueError("Output must be inside this package's cpu_runs directory")
    if args.command == "export":
        export_results(output)
        return 0
    enforce_package_mode(manifest, args.smoke)
    from research_rl.cpu_runtime import require_cpu
    require_cpu()
    import torch
    info["torch_version"] = torch.__version__
    info["torch_cuda_build"] = torch.version.cuda
    cutoff = datetime.fromisoformat(args.deadline)
    if cutoff.tzinfo is None or cutoff.timestamp() <= time.time():
        raise ValueError("Explicit future deadline with timezone required")
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = {"package_manifest_sha256": sha256(ROOT/"PACKAGE_MANIFEST.json"),
                    "protocol_sha256": sha256(ROOT/"research/cpu_v2_protocol.json"),
                    "smoke_only": args.smoke, "trial_minutes": args.trial_minutes,
                    "max_episodes_per_trial": args.max_episodes_per_trial,
                    "max_workers": args.max_workers, "deadline": args.deadline}
    rp = output/"run_manifest.json"
    if rp.exists() and json.loads(rp.read_text()) != run_manifest:
        raise ValueError("Run settings changed; use a new output directory")
    write_json(rp, run_manifest)
    write_json(output/"environment.json", info)
    parent = ROOT/"models/parent.pt"
    error = None
    failure_code = 1
    active_training = None
    summaries, statuses = {}, {}
    expected = 1 if args.smoke else 3

    def write_summary():
        complete = sum(row["training_budget_complete"] for row in statuses.values())
        write_json(output/"summary.json", {"smoke_only": args.smoke, "results": summaries,
                   "training_status": statuses, "expected_training_seeds": expected,
                   "completed_training_seeds": complete,
                   "training_complete": complete == expected,
                   "evaluation_complete": (len(summaries) == expected+1
                                           and all(row.get("complete") for row in summaries.values())),
                   "error": error,
                   "scope": "CPU handoff development endpoints; incomplete and interrupted trials are retained; return all seeds/checkpoints for independent analysis"})

    previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    try:
        selected = benchmark(output, info, parent, args.max_workers, args.smoke, cutoff.timestamp())
        summaries = {"parent": evaluate(parent, output/"evaluation/parent", args.smoke, cutoff.timestamp())}
        for index in range(expected):
            name, seed, scenario = f"trial-{index+1}", 9112101+index, 1000001+index*200000
            folder = output/name
            latest = folder/"latest.pt"
            completed = folder/"finished.json"
            state = None
            if latest.exists():
                state = torch.load(latest, map_location="cpu", weights_only=False)["state"]
            max_attempted = 4 if args.smoke else args.max_episodes_per_trial
            max_updates = (max_attempted+selected["episodes_per_update"]-1)//selected["episodes_per_update"]
            status_args = dict(scenario=scenario, scenario_end=scenario+199998,
                               max_attempted=max_attempted, max_updates=max_updates,
                               trial_seconds=args.trial_minutes*60, deadline_epoch=cutoff.timestamp())
            status = trial_status(state, **status_args)
            limit = min(args.trial_minutes*60-status["elapsed_training_s"], cutoff.timestamp()-time.time()-120)
            code = None
            if status["can_resume"] and limit > 2:
                seconds = min(limit, 120) if args.smoke else limit
                command = training_command(folder, seed=seed, scenario=scenario,
                    workers=selected["selected_workers"], threads=selected["learner_threads"],
                    episodes=selected["episodes_per_update"], updates=max_updates,
                    seconds=seconds, scenario_end=scenario+199998,
                    max_attempted_episodes=max_attempted, deadline_epoch=cutoff.timestamp()-120,
                    checkpoint=latest if latest.exists() else parent, resume=latest.exists(), smoke=args.smoke)
                print(f"Train {name}: CPU only, at most {limit:.0f} seconds", flush=True)
                active_training = (name, folder, latest, status_args)
                code = run_training(command, output/"logs"/(name+".log"),
                                    deadline_epoch=min(cutoff.timestamp()-60, time.time()+seconds+30))
                active_training = None
                if latest.exists():
                    state = torch.load(latest, map_location="cpu", weights_only=False)["state"]
                status = trial_status(state, returncode=code, **status_args)
            if latest.exists():
                status["checkpoint_sha256"] = sha256(latest)
            statuses[name] = status
            write_json(folder/"status.json", status)
            if status["training_budget_complete"]:
                write_json(completed, {**status, "state": state})
            write_summary()
            if code:
                raise RuntimeError(f"{name} returned {code}; latest checkpoint and logs retained")
            if status["status"] == "invalid":
                raise ValueError(f"{name} checkpoint violates its declared sampling budget or partition")
            if latest.exists() and cutoff.timestamp()-time.time() > 60:
                summaries[name] = evaluate(latest, output/"evaluation"/name, args.smoke, cutoff.timestamp())
            write_summary()
    except KeyboardInterrupt as exc:
        error, failure_code = f"KeyboardInterrupt: {exc}", 130
        if active_training is not None:
            name, folder, latest, status_args = active_training
            try:
                state = torch.load(latest, map_location="cpu", weights_only=False)["state"] if latest.exists() else None
                statuses[name] = trial_status(state, returncode=130, **status_args)
                if latest.exists():
                    statuses[name]["checkpoint_sha256"] = sha256(latest)
                write_json(folder/"status.json", statuses[name])
            except Exception as checkpoint_error:
                error += f"; interrupted checkpoint could not be inspected: {checkpoint_error}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            write_json(output/"error.json", {"error": error, "active": bool(error)})
            write_summary()
            export_results(output)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    if error:
        print(error, file=sys.stderr)
        return failure_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
