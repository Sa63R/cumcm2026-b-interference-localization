"""One-use GAE v5 resume sidecar. Default: read-only preflight, never train.

Only --supervise may start the unchanged supervisor; --worker is its owned child.
This is bound to the four preserved 19:41 UTC disk-stop transactions, not a
generic launcher. No source edits, state edits, transfers or artifact deletion.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

TASK = "q4-rl-gae-v5-20260912"
REMOTE_ROOT = "/home/dataset-assist-0/usr/lh/ysh/bwc/shumo/" + TASK
OLD_RUN = "train-gae-v5"
NEW_RUN = "train-gae-v5-resume-001"
COMMIT = "df36f0ce43a8b1e23d6ebf1cb1b0bba6f16ef01d"
DEADLINE = "2026-09-11T22:00:00+00:00"
MANIFEST_SHA = "72def29c976d1c076de5c221140ce3bd1df6572dce209a25d90a0a87cbe454f0"
CONFIG_SHA = "022316a9f4797c3af5aad4e86f9eefd130dfc3a2307ec09d5b15f92013405154"
ARCHIVE_SHA = "ab3a7d6e312306700e9a79b9b14092ce53f084c6e5a8e3fb39847e8c6802c464"
INITIAL_SHA = {
    "g1": "a3170afc53ae48d5b03618872400a424ce27d9c1110f7f2b342fdea13837688d",
    "g3": "61d4ac338cdfc0b62fcbb0cf9c4f2baaf9f832c8d203ac8d33a0f9b308f5ce49",
}
# completed batches, latest, complete checkpoint, terminal status SHA256.
EXPECTED = {
    "g1_h128_mc": (13, "ba253fe13d5b9ea98b9db8d96d38f1c65d9b3c26c9df83a59ff802de635ca657", "2870173ae856d8ed85963612a36bb66598d73dae6f33cecf45f6831f64fbbea6", "03d6176fd318668fc86788696a887f130aa16beeef967ccda9c08efdf6506bc6"),
    "g1_h128_gae097": (13, "e8693bb863cf559ec4299255a9de630b07b2b47cec3edd01dee0d64be7c489ab", "f236acabd5e62bbcd2ed80dbf3588d35913eaa830005df9702a4d66aa604ca49", "5add64f620f74b84ed169ccefbf3ecf96648122ce12c4ceeea1dee145c8a3470"),
    "g3_h128_mc": (12, "ce4b39f779b12bfee48accb25d23c6e8c87edd3bb567f16c05445c35e05be215", "46022d77b009dcf550ff7a385cedfddb67853c1da6d168513d3a2414713d4fb9", "9bd400d216478d4cb1fc746041dcfd374b1f3c3e843ec124827b195bf07ddf30"),
    "g3_h128_gae097": (13, "85569cdd5682c90cf1752e1d5a9b0d206f9e9acd0e7902c1693221ef883fa54b", "ff1c306182684392772f10896cc4202252aaa745ee0e64ccfdbf9bd2c1eb7f1c", "96a5eabb616b685fc9ff978350c528119eb4b670787936d613768ea9e2dbe237"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, relative):
    relative = PurePosixPath(relative)
    require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in str(relative), "unsafe relative path")
    path = root.joinpath(*relative.parts)
    require(path.resolve() == path.absolute() and path.resolve().is_relative_to(root), "symlink or path escape")
    return path


def exact_bytes(path, expected):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "artifact SHA256 mismatch: " + path.name)
    return raw


def flag_map(argv):
    require(len(argv) % 2 == 0 and all(isinstance(v, str) for v in argv), "invalid argv pairs")
    flags = dict(zip(argv[::2], argv[1::2]))
    require(len(flags) * 2 == len(argv), "duplicate flags")
    return flags


def remaining_seconds(wall, now):
    require(isinstance(wall, (int, float)) and math.isfinite(wall) and wall >= 0, "invalid cumulative wall time")
    require(now < datetime.fromisoformat(DEADLINE).timestamp(), "original absolute deadline expired")
    remaining = math.floor(3600 - wall)
    require(remaining > 0, "original cumulative wall budget exhausted")
    return remaining


def resume_job(job, saved, root, now):
    """Pure argv derivation; no mutation of source config, saved state or RNG."""
    name = job["name"]
    require(name in EXPECTED, "unknown resume job")
    group = name[:2]
    module = "q4_rl.micro_train" if group == "g1" else "q4_rl.memory_train"
    require(job["module"] == module, "wrong algorithm/module")
    flags = flag_map(job["argv"])
    output = f"runs/{OLD_RUN}/{name}/training"
    require(flags.get("--output") == output, "wrong original output directory")
    inside(root, output)
    fixed = {"--workers": "5", "--learner-threads": "4", "--cpu-budget": "9",
        "--warmstart-episodes": "0", "--max-batches": "32", "--max-wall-seconds": "3600",
        "--scenario-start": "8022000", "--scenario-end": "8031999", "--random-seed": "424555"}
    require(all(flags.get(k) == v for k, v in fixed.items()), "changed original resource/experiment configuration")
    require(datetime.fromisoformat(flags["--deadline"]).timestamp() == datetime.fromisoformat(DEADLINE).timestamp(), "changed absolute deadline")
    require(float(flags["--gae-lambda"]) == (1. if name.endswith("_mc") else .97), "wrong GAE arm")
    initial_flag = "--initialize-micro-warmstart" if group == "g1" else "--initialize-memory-warmstart"
    require(flags.get(initial_flag) == f"models/initialization/{group}_h128_bc256.pt" and flags.get("--initialize-sha256") == INITIAL_SHA[group], "changed BC provenance")
    require("--resume" not in flags, "source configuration is not the original fresh experiment")
    state = saved["state"]
    batches = EXPECTED[name][0]
    require(state["batches"] == state["ppo_batches"] == batches and state["episodes"] == 16 * batches and state["warmstart_completed"] == 0, "changed committed counters")
    pending = state.get("pending_batch")
    require(isinstance(pending, dict) and pending["mode"] == "ppo" and pending["seeds"] == list(range(8022000 + 16 * batches, 8022000 + 16 * (batches + 1))) and len(pending["action_seeds"]) == 16, "pending reservation missing or changed")
    require(state["next_seed"] == 8022000 + 16 * (batches + 1) and state["next_attempt"] == batches + 1, "changed reservation cursor")
    remaining = remaining_seconds(state["wall_time_s"], now)
    flags["--max-wall-seconds"] = str(remaining)
    flags["--resume"] = output + "/latest.pt"
    if group == "g3":
        # G3 reads the existing binding from latest; its CLI forbids init+resume.
        del flags[initial_flag]
        del flags["--initialize-sha256"]
    # G1 retains init flags for config equality. main reads BC for validation,
    # then restore_checkpoint(latest) restores trained model/Adam/RNG instead.
    argv = [item for pair in flags.items() for item in pair]
    trainer = importlib.import_module(module)
    _, args = trainer._arguments(argv)
    requested = trainer.configuration(args)
    if group == "g3":
        requested["initialization"] = trainer.validate_initialization_binding(saved["config"]["initialization"])
    require(requested == saved["config"], "resume checkpoint/config mismatch")
    return {"name": name, "module": module, "argv": argv, "output": output}


def validate_budget(jobs, other_compute=11):
    require(type(other_compute) is int and other_compute >= 0, "invalid other-task CPU reservation")
    total = 0
    for job in jobs:
        flags = flag_map(job["argv"])
        required = int(flags["--workers"]) + int(flags["--learner-threads"])
        require(required <= int(flags["--cpu-budget"]), "per-job CPU budget exceeded")
        total += required
    require(total + other_compute <= 50, "combined compute budget exceeds 50")
    return {"training_compute": total, "reserved_other_compute": other_compute, "combined_compute": total + other_compute, "outer_cpu_budget": 50}


def verify_source(root):
    manifest = json.loads(exact_bytes(inside(root, "RELEASE_MANIFEST.json"), MANIFEST_SHA))
    require(manifest["git_commit"] == COMMIT and len(manifest["files"]) == 131, "wrong original source release")
    for relative, item in manifest["files"].items():
        raw = exact_bytes(inside(root, relative), item["sha256"])
        require(len(raw) == item["bytes"], "source size mismatch")
    exact_bytes(inside(root, "launch/train-gae-v5/q4-gae-v5-source-20260912-r2.tar.gz"), ARCHIVE_SHA)
    config = json.loads(exact_bytes(inside(root, "research/q4_rl/train_gae_v5.json"), CONFIG_SHA))
    require(config["run"] == OLD_RUN and [j["name"] for j in config["jobs"]] == list(EXPECTED), "wrong original job set/order")
    for group, sha in INITIAL_SHA.items():
        exact_bytes(inside(root, f"models/initialization/{group}_h128_bc256.pt"), sha)
    return config


def check_journal(output, name):
    """Check small committed index and referenced paths; no raw payload rewrite."""
    require(Path(name).name == name, "unsafe journal index")
    path = inside(output, name)
    with gzip.open(path, "rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    require(len(raw) <= 1024 * 1024, "oversized journal index")
    index = json.loads(raw)
    require(index.get("format") == "q4-training-episode-index-v1" and isinstance(index.get("episodes"), list) and len(index["episodes"]) <= 16, "invalid original journal")
    paths = [row["path"] for row in index["episodes"]]
    require(len(paths) == len(set(paths)), "duplicate journal reference")
    for row in index["episodes"]:
        require(Path(row["path"]).name == row["path"] and inside(output, row["path"]).is_file(), "missing original episode payload")
        require(len(row["sha256"]) == 64, "missing episode hash")


def check_old_journals(output, state):
    check_journal(output, state["last_progress"]["raw_attempt"])
    old = f"batch-{state['batches']:06d}-attempt-{state['next_attempt'] - 1:06d}.json.gz"
    if (output / old).exists():
        check_journal(output, old)
    # Zero completed episodes can legitimately leave no pending index. Existing
    # pending files are retained; replay gets a new, collision-free attempt.
    prefix = f"batch-{state['batches']:06d}-attempt-{state['next_attempt']:06d}"
    require(not any(output.glob(prefix + "*")), "next attempt already exists; do not overwrite/reuse")


def fresh_run(root):
    run = inside(root, "runs/" + NEW_RUN)
    require(not run.exists(), "supervisor run directory already exists")
    return run


def preflight(root, *, worker=False):
    require(os.name == "posix" and str(root) == REMOTE_ROOT, "resume only in the original Linux task root")
    require(not root.is_symlink(), "task root symlink forbidden")
    run = inside(root, "runs/" + NEW_RUN)
    if not worker:
        fresh_run(root)
    require(shutil.disk_usage(root).free > 20 * 1024**3, "free disk must exceed 20 GiB")
    allowed = set(os.sched_getaffinity(0))
    require(allowed and allowed <= set(range(50)), "launch within common taskset 0-49")
    config = verify_source(root)  # Validate source before importing any trainer.
    sys.path[:0] = [str(root / "src"), str(root)]
    import torch
    from q4_rl.micro_network import configure_cpu
    configure_cpu()
    jobs, details = [], []
    for job in config["jobs"]:
        name = job["name"]
        count, latest_sha, complete_sha, status_sha = EXPECTED[name]
        output = inside(root, f"runs/{OLD_RUN}/{name}/training")
        raw = exact_bytes(output / "latest.pt", latest_sha)
        exact_bytes(output / f"checkpoint-{count:06d}.pt", complete_sha)
        status = json.loads(exact_bytes(output / "status.json", status_sha))
        saved = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
        trainer = importlib.import_module(job["module"])
        trainer.validate_checkpoint(saved)
        require(saved["state"] == status, "latest/status state mismatch")
        check_old_journals(output, saved["state"])
        jobs.append(resume_job(job, saved, root, time.time()))
        details.append({"job": name, "completed_batches": count, "latest_sha256": latest_sha,
            "cumulative_wall_s": status["wall_time_s"], "remaining_wall_s": int(flag_map(jobs[-1]["argv"])["--max-wall-seconds"]),
            "pending_count": len(status["pending_batch"]["seeds"]), "pending_action_seeds_preserved": True})
    budget = validate_budget(jobs)
    return run, jobs, {"source_commit": COMMIT, "original_config_sha256": CONFIG_SHA,
        "resume_run": NEW_RUN, "deadline_utc": DEADLINE, "jobs": details, "budget": budget,
        "journal_check": "indexes and referenced paths only; no raw payload download/hash",
        "execution": "not started by preflight"}


def supervisor_command(root, script):
    return [sys.executable, "-B", str(root / "scripts/q4_cpu_supervisor.py"),
        "--root", str(root), "--run", NEW_RUN, "--cpu-budget", "50",
        "--minimum-free-gib", "20", "--deadline", DEADLINE, "--",
        sys.executable, "-B", str(script), "--worker"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--supervise", action="store_true", help="explicitly start supervised resume after preflight")
    mode.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    script = Path(__file__).resolve()
    require(script.is_relative_to(root), "sidecar must reside inside original task")
    if args.worker:
        sys.path[:0] = [str(root / "src"), str(root)]
        # This import contains process ownership checks, not trainer execution.
        verify_source(root)
        from scripts.q4_training_pair import require_outer_supervisor, run_pair
        require_outer_supervisor(root)
        run = inside(root, "runs/" + NEW_RUN)
        claim = json.loads((run / "resume-claim.json").read_bytes())
        require(claim["supervisor_pid"] == os.getppid() and claim["run"] == NEW_RUN and claim["sidecar_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest(), "resume directory was not freshly claimed by this supervisor")
        parent = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
        require([p.decode() for p in parent if p] == supervisor_command(root, script), "unexpected supervisor command/budget/deadline")
        require(not (run / "training-pair-launch.json").exists(), "resume was already launched")
        run, jobs, report = preflight(root, worker=True)
        with (run / "resume-preflight.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
        return run_pair(run, jobs, root)
    run, jobs, report = preflight(root)
    print(json.dumps(report, indent=2), flush=True)
    if not args.supervise:
        return 0
    # Atomic absent->created transition occurs BEFORE the unchanged supervisor
    # starts. exec retains this PID, proving ownership to its direct worker.
    run.mkdir(exist_ok=False)
    claim = {"run": NEW_RUN, "supervisor_pid": os.getpid(),
        "sidecar_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    with (run / "resume-claim.json").open("x", encoding="utf-8") as stream:
        json.dump(claim, stream, indent=2)
    command = supervisor_command(root, script)
    os.execv(sys.executable, command)
    raise AssertionError("exec returned")


if __name__ == "__main__":
    raise SystemExit(main())
