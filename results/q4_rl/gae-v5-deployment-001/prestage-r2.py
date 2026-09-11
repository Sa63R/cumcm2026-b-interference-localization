"""Install and test the frozen GAE task only; this file never launches training."""
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from datetime import datetime

TASK = "q4-rl-gae-v5-20260912"
ROOT = Path("/home/dataset-assist-0/usr/lh/ysh/bwc/shumo") / TASK
CONTROL = ROOT / "launch/train-gae-v5"
REMOTE = "jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/" + TASK


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exchange(*args):
    subprocess.run(["rclone", *args, "--s3-no-check-bucket", "--immutable", "--transfers", "1", "--checkers", "1",
                    "--contimeout", "10s", "--timeout", "60s", "--retries", "2", "--log-level", "ERROR"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)


def save(name, value):
    target = CONTROL / name
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    exchange("copyto", str(target), REMOTE + "/launch/train-gae-v5/" + name)


def main():
    if socket.gethostname() != "ide-376f3dcbf2424192b9d1abca3872afc8-445523" or Path.cwd().resolve() != ROOT:
        raise ValueError("Unexpected Q4 task target")
    allowed = os.sched_getaffinity(0)
    if not set(range(50)) <= allowed:
        raise ValueError("Shared CPU0..49 pool unavailable")
    os.sched_setaffinity(0, {0})
    os.environ.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "src")+":"+str(ROOT))
    sys.dont_write_bytecode = True
    (ROOT / ".tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(ROOT / ".tmp")
    plan = json.loads((CONTROL / "plan-r2.json").read_text())
    if plan["task"] != TASK or plan["git_commit"] != "df36f0ce43a8b1e23d6ebf1cb1b0bba6f16ef01d":
        raise ValueError("Unexpected immutable plan")
    if sha(Path(__file__)) != plan["prestage_sha256"]:
        raise ValueError("Prestage script identity mismatch")
    free_before = shutil.disk_usage(ROOT).free
    if free_before < 20*1024**3 + 100*1024**2:
        raise ValueError("Insufficient disk for task prestage")
    archive_path = CONTROL / plan["source_archive"]
    if sha(archive_path) != plan["source_archive_sha256"]:
        raise ValueError("Source archive identity mismatch")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or any(not member.isfile() or PurePosixPath(member.name).is_absolute()
                or ".." in PurePosixPath(member.name).parts for member in members):
            raise ValueError("Unsafe source archive")
        manifest_raw = archive.extractfile("RELEASE_MANIFEST.json").read()
        manifest = json.loads(manifest_raw)
        if (manifest["git_commit"] != plan["git_commit"] or hashlib.sha256(manifest_raw).hexdigest() != plan["source_manifest_sha256"]
                or set(names) != set(manifest["files"]) | {"RELEASE_MANIFEST.json"}):
            raise ValueError("Source manifest identity mismatch")
        for member in members:
            raw = archive.extractfile(member).read()
            if member.name != "RELEASE_MANIFEST.json" and hashlib.sha256(raw).hexdigest() != manifest["files"][member.name]["sha256"]:
                raise ValueError("Source member checksum mismatch")
            target = ROOT / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != hashlib.sha256(raw).hexdigest():
                    raise ValueError("Existing task source differs")
            else:
                with target.open("xb") as stream:
                    stream.write(raw)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    from scripts.q4_cpu_supervisor import resolve_cpu_hierarchy, cpu_scope_snapshot, quota_cores, cpu_allowance
    hierarchy = resolve_cpu_hierarchy()
    quota = quota_cores(hierarchy, len(allowed))
    resource = dict(visible_quota_cores=quota, original_affinity_count=len(allowed), common_pool=list(range(50)),
        test_affinity=list(os.sched_getaffinity(0)), cpu_scopes=cpu_scope_snapshot(hierarchy),
        planned_supervisor_allowance=cpu_allowance(50, 50, quota), reserve_cores=8,
        current_load_average=list(os.getloadavg()), free_bytes_before=free_before)
    from q4_rl.network import configure_cpu
    configure_cpu()
    models = []
    for item in plan["models"]:
        target = ROOT / item["path"]
        if item["path"] not in {"models/initialization/g1_h128_bc256.pt", "models/initialization/g3_h128_bc256.pt"}:
            raise ValueError("Unexpected initialization path")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            exchange("copyto", REMOTE + "/" + item["path"], str(target))
        if sha(target) != item["sha256"]:
            raise ValueError("Initialization byte checksum mismatch")
        module_name, function_name = item["loader"].split(":")
        if item["loader"] not in {"q4_rl.micro_initialization:load_micro_warmstart", "q4_rl.memory_initialization:load_memory_warmstart"}:
            raise ValueError("Unexpected initialization loader")
        saved = getattr(importlib.import_module(module_name), function_name)(target, item["sha256"])
        if (saved["network"]["global_dim"] != 13 or saved["network"]["candidate_dim"] != item["candidate_dim"]
                or saved["network"]["hidden"] != 128 or saved["state"]["warmstart_completed"] != 256
                or saved["state"]["episodes"] != 256 or saved["state"]["ppo_batches"] != 0
                or saved["state"]["pending_batch"] is not None):
            raise ValueError("Initialization dimensions/BC transaction differ")
        models.append(dict(path=item["path"], sha256=item["sha256"], network=saved["network"],
                           warmstart_completed=256, ppo_batches=0, strict_loader_passed=True))
    from scripts.q4_training_bundle import validate_config
    config_path = ROOT / "research/q4_rl/train_gae_v5.json"
    if sha(config_path) != plan["config_sha256"]:
        raise ValueError("Training config checksum mismatch")
    config = json.loads(config_path.read_text())
    run, jobs = validate_config(config, ROOT)
    if len(jobs) != 4 or run != ROOT / "runs/train-gae-v5":
        raise ValueError("Unexpected training job group")
    parsed_jobs = []
    for job in jobs:
        trainer = importlib.import_module(job["module"])
        _, args = trainer._arguments(job["argv"])
        if (args.workers != 5 or args.learner_threads != 4 or args.cpu_budget != 9
                or args.warmstart_episodes != 0 or args.max_batches != 32
                or args.max_wall_seconds != 3600 or args.scenario_start != 8022000
                or args.scenario_end != 8031999 or args.random_seed != 424555
                or args.max_decisions != 512 or datetime.fromisoformat(args.deadline).timestamp() != datetime.fromisoformat("2026-09-11T22:00:00+00:00").timestamp()):
            raise ValueError("Preregistered training configuration changed")
        parsed_jobs.append(dict(name=job["name"], module=job["module"], gae_lambda=args.gae_lambda,
            cpu_budget=9, workers=5, learner_threads=4, config=trainer.configuration(args)))
    import torch
    from q4_rl.network import configure_cpu
    configure_cpu()
    testing_started = time.monotonic()
    test_cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(ROOT / ".tmp/pytest"), *plan["test_modules"]]
    with (CONTROL / "pytest.log").open("xb") as stream:
        tests = subprocess.run(test_cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=240)
    exchange("copyto", str(CONTROL / "pytest.log"), REMOTE + "/launch/train-gae-v5/pytest.log")
    test_wall = time.monotonic()-testing_started
    if tests.returncode != 0:
        save("PRESTAGE_FAILURE.json", dict(task=TASK, step="pytest", returncode=tests.returncode,
            source_sha256=sha(archive_path), pytest_sha256=sha(CONTROL / "pytest.log"), training_started=False))
        raise ValueError("Frozen server tests failed")
    log = (CONTROL / "pytest.log").read_text()
    if "142 passed" not in log:
        raise ValueError("Unexpected frozen test count")
    mismatches = [name for name, item in manifest["files"].items() if sha(ROOT / name) != item["sha256"]]
    if mismatches:
        raise ValueError("Source changed during prestage tests")
    free_after = shutil.disk_usage(ROOT).free
    if free_after < 20*1024**3:
        raise ValueError("Disk no longer has twenty GiB headroom")
    result = dict(task=TASK, git_commit=plan["git_commit"], source_archive_sha256=sha(archive_path),
        source_manifest_sha256=plan["source_manifest_sha256"], source_file_count=len(manifest["files"]),
        all_source_hashes_unchanged=True, config_sha256=sha(config_path), plan_sha256=sha(CONTROL / "plan-r2.json"),
        models=models, jobs=parsed_jobs, tests=dict(passed=142, returncode=0, wall_s=test_wall,
            affinity_cpus=1, log_sha256=sha(CONTROL / "pytest.log")), resource=resource,
        free_bytes_after=free_after, total_declared_training_compute=36, concurrent_eval_compute=11,
        combined_compute=47, outer_cpu_budget=50, minimum_free_gib=20,
        python_version=sys.version.split()[0], torch_version=torch.__version__,
        torch_intraop=torch.get_num_threads(), torch_interop=torch.get_num_interop_threads(),
        cuda_disabled=os.environ["CUDA_VISIBLE_DEVICES"] == "", training_started=False,
        launch_command=plan["launch_command"], next_action="Await explicit root review and authorization before launch")
    save("PRESTAGE.json", result)
    print("GAE task prestaged; 142 tests and two BC checks passed; no training launched", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("prestage failed: " + type(error).__name__ + (": " + str(error) if isinstance(error, ValueError) else ""), flush=True)
        raise SystemExit(1)
