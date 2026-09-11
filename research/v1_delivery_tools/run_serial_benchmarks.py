"""Serial, externally deadline-limited timing of the four selected identities.

No scenario generation is implemented here. The independently reviewed worker
runs only the opened development seeds 6000..6015, never final evaluation cases.
Training processes are observed, never signalled by this launcher.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def in_directory(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def is_training_command(command):
    return any(arg == "research_rl.train" or arg.replace("\\", "/").endswith("/research_rl/train.py")
               for arg in command)


def read_process(pid, proc_root=Path("/proc")):
    """Read one Linux process, rejecting ambiguous access instead of guessing."""
    directory = Path(proc_root) / str(pid)
    try:
        state = (directory / "stat").read_text().rsplit(")", 1)[1].split()[0]
        command = [part.decode("utf-8", errors="replace")
                   for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
        # Zombie processes no longer train; their cwd often cannot be resolved.
        try:
            cwd = str((directory / "cwd").resolve(strict=True)) if state != "Z" else None
        except (FileNotFoundError, PermissionError):
            if is_training_command(command):
                raise
            # Non-training command lines are already excluded. Linux may hide
            # their cwd even from the same uid (e.g. non-dumpable services).
            # Missing/inaccessible cwd of any training process remains fatal.
            cwd = None
        return dict(pid=pid, state=state, command=command, cwd=cwd)
    except (FileNotFoundError, ProcessLookupError):
        if directory.exists():
            # A live process with inaccessible cwd is not established inactive.
            raise RuntimeError(f"Cannot establish process state/cwd for existing PID {pid}")
        return None


def training_quiescence(record_paths, research_root, proc_root=Path("/proc")):
    require(bool(record_paths), "Supply at least one explicit --training-process-record; missing evidence is not completion")
    records, active = [], {}
    for path in record_paths:
        raw = read_json(path)  # Missing or malformed records deliberately stop launch.
        pid = raw.get("pid")
        command, cwd = raw.get("command"), raw.get("cwd")
        require(type(pid) is int and pid > 0, f"Invalid training PID in {path}")
        require(isinstance(command, list) and all(isinstance(x, str) for x in command)
                and is_training_command(command) and isinstance(cwd, str),
                f"Training record lacks its original command/cwd: {path}")
        require(in_directory(cwd, research_root), f"Training record is outside the registered research root: {path}")
        current = read_process(pid, proc_root)
        same = bool(current and current["command"] == command
                    and current["cwd"] == str(Path(cwd).resolve()))
        running_training = bool(current and current["state"] != "Z" and is_training_command(current["command"])
                                and current["cwd"] and in_directory(current["cwd"], research_root))
        if running_training:
            active[pid] = current
        records.append(dict(path=str(Path(path).resolve()), sha256=sha(path), recorded_pid=pid,
                            same_command_and_cwd=same, observed=current,
                            status="active_training" if running_training else
                                   "not_present" if current is None else
                                   "zombie" if current["state"] == "Z" else "pid_reused_or_different_process"))
    # Also catch relevant training that is not among the supplied process files.
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
        except FileNotFoundError:
            continue
        current = read_process(int(directory.name), proc_root)
        if (current and current["state"] != "Z" and is_training_command(current["command"])
                and current["cwd"] and in_directory(current["cwd"], research_root)):
            active[current["pid"]] = current
    return dict(research_root=str(research_root), record_checks=records,
                active_training=list(active.values()), checked_epoch=time.time(),
                scope="Current-user /proc scan and exact recorded PID command/cwd checks; no training process is terminated.")


def signal_group(child, force):
    try:
        os.killpg(child.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def stop_launcher(signum, frame):
    # Turn a supervisor TERM/INT into Python cleanup, rather than orphaning a
    # worker that intentionally lives in its own session/process group.
    raise KeyboardInterrupt(f"Launcher received signal {signum}")


def supervise(command, *, cwd, env, log, hard_deadline, reserve=15.,
              term_grace=5., kill_grace=5., stop_group=signal_group):
    """Start one isolated worker; injectable group stopper supports fake tests."""
    started, child = time.time(), None
    result = dict(command=command, cwd=str(cwd), log=str(log), started_epoch=started,
                  started_process=False, timed_out=False, not_started_due_to_deadline=False,
                  returncode=None, error=None)
    remaining = hard_deadline - started - reserve
    if remaining <= 5:
        result.update(not_started_due_to_deadline=True, elapsed_wall_s=time.time() - started)
        return result
    try:
        with Path(log).open("a", encoding="utf-8") as stream:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            result.update(started_process=True, pid=child.pid)
            try:
                result["returncode"] = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                result["timed_out"] = True
                stop_group(child, False)
                try:
                    result["returncode"] = child.wait(timeout=term_grace)
                except subprocess.TimeoutExpired:
                    stop_group(child, True)
                    result["returncode"] = child.wait(timeout=kill_grace)
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["interrupted"] = not isinstance(exc, Exception)
        if child is not None and child.poll() is None:
            try:
                stop_group(child, True)
                result["returncode"] = child.wait(timeout=kill_grace)
            except Exception as cleanup:
                result["cleanup_error"] = f"{type(cleanup).__name__}: {cleanup}"
    result["elapsed_wall_s"] = time.time() - started
    return result


def load_selection(tool_path, registry_path, selection_path):
    spec = importlib.util.spec_from_file_location("serial_benchmark_selection_tool", tool_path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    registry = tool.verify_registry(registry_path)
    chosen = tool.read_sealed(selection_path, "selection_decision")
    require(chosen["registry_file_sha256"] == tool.file_hash(registry_path)
            and chosen["registry_payload_sha256"] == registry["payload_sha256"]
            and chosen["rule_sha256"] == registry["rule_sha256"], "Selection belongs to another registry/rule")
    require(tool.timestamp(chosen["finalized_at"]) >= tool.timestamp(registry["registered_at"]),
            "Selection predates registration")
    previous = tool.load_partition({key: value["directory"] for key, value in chosen["evaluations"].items()},
                                  registry, "validation_extended", registry["entries"])
    require(tool.evidence(previous) == chosen["evaluations"] and tool.select(registry, previous) == chosen["decisions"],
            "Selection evidence or deterministic decision changed")
    ids = [registry["baseline"]] + [chosen["decisions"][name]["selected"] for name in tool.DIRECTIONS]
    return tool, registry, chosen, ids


def check_environment(expected):
    require(platform.system() == "Linux", "This process-group supervisor requires Linux")
    require(Path(expected["python_executable"]).resolve() == Path(sys.executable).resolve(),
            "Use the registered Python interpreter")
    for key, value in (("system", platform.system()), ("python_version", platform.python_version()),
                       ("release", platform.release()), ("machine", platform.machine())):
        if key in expected:
            require(expected[key] == value, f"Actual environment differs: {key}")
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]}
    require(packages == expected["packages"], "Installed packages changed since registration")


def verify_benchmark(path, entry):
    benchmark = read_json(path)
    require(benchmark["identity"] == entry["identity"] and benchmark["git_commit"] == entry["git_commit"],
            "Benchmark returned another frozen identity")
    expected_env_sha = hashlib.sha256(json.dumps(entry["environment_value"], sort_keys=True).encode()).hexdigest()
    require(benchmark["environment_sha256"] == expected_env_sha, "Benchmark environment digest differs")
    expected = {(-1, 6000)} | {(repeat, seed) for repeat in range(2) for seed in range(6000, 6016)}
    rows = benchmark["rows"]
    keys = [(row["repeat"], row["seed"]) for row in rows]
    require(len(keys) == len(set(keys)) and set(keys) <= expected,
            "Benchmark contains duplicate or non-public case identities")
    require(all(bool(row["warmup"]) == (row["repeat"] == -1) for row in rows), "Wrong warmup labels")
    if benchmark["complete"]:
        require(set(keys) == expected and benchmark["repeats"] == 2, "Complete benchmark lacks the fixed repetitions")
    return dict(path=str(path), sha256=sha(path), complete=benchmark["complete"],
                all_successful=benchmark["all_successful"], timed_cases=benchmark["timed_cases"],
                repeated_action_histories_identical=benchmark["repeated_action_histories_identical"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tool", "registry", "selection", "worker-script", "run-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--training-process-record", type=Path, action="append", default=[])
    args = parser.parse_args()
    require(not sys.flags.optimize, "Do not run audited timing with Python assertions disabled")
    require(bool(args.training_process_record), "Supply explicit training process records")
    require(platform.system() == "Linux", "This supervisor runs on the registered Linux host")
    signal.signal(signal.SIGTERM, stop_launcher)
    signal.signal(signal.SIGINT, stop_launcher)
    args.tool, args.registry, args.selection = (p.resolve() for p in (args.tool, args.registry, args.selection))
    worker, run_root = args.worker_script.resolve(), args.run_root.resolve()
    worker_sha, launcher_sha = sha(worker), sha(__file__)
    tool, registry, chosen, ids = load_selection(args.tool, args.registry, args.selection)
    environment = next(iter(registry["entries"].values()))["environment_value"]
    check_environment(environment)
    research_root = Path(os.path.commonpath([entry["source_dir"] for entry in registry["entries"].values()]))
    require(research_root != Path(research_root.anchor), "Registered research root is too broad")
    deadline = datetime.fromisoformat(registry["protocol_value"]["hard_deadline"]).timestamp()
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    started, results, error = time.time(), [], None
    final_identity_checked = False
    run_root.mkdir(parents=True, exist_ok=True)
    logs, outputs = run_root / "logs", run_root / "benchmarks"
    logs.mkdir(exist_ok=True)
    outputs.mkdir(exist_ok=True)
    stamp = time.time_ns()
    ledger_path = run_root / f"serial-benchmark-execution-{stamp}.jsonl"
    summary_path = run_root / f"serial-benchmark-execution-{stamp}.json"
    stopped = None
    with ledger_path.open("x", encoding="utf-8") as ledger:
        try:
            for key in ids:
                require(Path(key).name == key and key not in (".", ".."), "Unsafe candidate id for output path")
                entry = registry["entries"][key]
                output, log = outputs / (key + ".json"), logs / (key + ".log")
                command = [sys.executable, str(worker), "--source", entry["source_dir"], "--spec", entry["spec"],
                           "--freeze-record", entry["freeze"], "--environment", entry["environment"],
                           "--output", str(output), "--repeats", "2"]
                job = dict(id=key, output=str(output), command=command, cwd=entry["source_dir"], started_process=False)
                try:
                    if stopped:
                        job.update(not_started_reason=stopped)
                    elif deadline - time.time() <= 20:
                        stopped = "hard_deadline"
                        job.update(not_started_reason=stopped, not_started_due_to_deadline=True)
                    else:
                        require(sha(worker) == worker_sha and sha(__file__) == launcher_sha, "Launcher/worker changed during timing")
                        check_environment(environment)
                        observation = training_quiescence(args.training_process_record, research_root)
                        job["training_preflight"] = observation
                        require(not observation["active_training"], "Registered-root research_rl.train is still active; benchmark refused")
                        require(not output.exists() and not output.with_suffix(".jsonl").exists(),
                                "Existing benchmark evidence is retained; choose a new run root for an intentional rerun")
                        job.update(supervise(command, cwd=entry["source_dir"], env=env, log=log, hard_deadline=deadline))
                        if output.exists():
                            job["benchmark"] = verify_benchmark(output, entry)
                        if job.get("timed_out") or job.get("not_started_due_to_deadline"):
                            stopped = "hard_deadline"
                        if job.get("cleanup_error"):
                            stopped = "worker_cleanup_failed"
                        if job.get("interrupted"):
                            stopped = "launcher_interrupted"
                except Exception as exc:
                    job["error"] = f"{type(exc).__name__}: {exc}"
                    stopped = "preflight_or_evidence_error"
                job["finished_epoch"] = time.time()
                results.append(job)
                ledger.write(json.dumps(job, ensure_ascii=False, allow_nan=False) + "\n")
                ledger.flush()
                print(json.dumps({k: v for k, v in job.items() if k not in ("training_preflight", "command")}), flush=True)
            # No heavy evidence re-audit near the cutoff; incomplete provenance
            # checks remain explicit and cannot be mistaken for a completed run.
            if deadline - time.time() > 60 and stopped != "launcher_interrupted":
                tool.verify_registry(args.registry)
                require(sha(worker) == worker_sha and sha(__file__) == launcher_sha, "Timing source changed")
                final_identity_checked = True
            else:
                final_identity_checked = False
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            good = lambda job: (job.get("returncode") == 0 and not job.get("timed_out") and not job.get("error")
                                and job.get("benchmark", {}).get("complete") and job["benchmark"]["all_successful"])
            complete = bool(error is None and len(results) == len(ids) and all(good(job) for job in results)
                            and final_identity_checked)
            summary = dict(kind="serial_public_runtime_execution", planned_ids=ids, execution_order="baseline,state,rl,geo",
                registry_sha256=sha(args.registry), selection_sha256=sha(args.selection),
                tool_sha256=sha(args.tool), launcher_sha256=launcher_sha, worker_sha256=worker_sha,
                started_epoch=started, elapsed_wall_s=time.time() - started, hard_deadline_epoch=deadline,
                cpu_environment={key: env[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                complete=complete, error=error, final_identity_checked=final_identity_checked,
                jobs=results, ledger=str(ledger_path),
                scope="Serial public 6000..6015 whole-policy timing; no independent final evidence. Other non-training evaluation jobs must also be stopped operationally.")
            with summary_path.open("x", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
    print(json.dumps(dict(summary=str(summary_path), complete=summary["complete"])), flush=True)
    return 0 if summary["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
