"""Install a new isolated evaluation task and verify it without any rollout."""
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

TASK = "q4-rl-memory-v4-eval-20260912"
ROOT = Path("/home/dataset-assist-0/usr/lh/ysh/bwc/shumo") / TASK
CONTROL = ROOT / "launch/eval-memory-v4"
REMOTE = "jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rclone(*args):
    subprocess.run(["rclone", *args, "--s3-no-check-bucket", "--immutable", "--transfers", "1", "--checkers", "1",
                    "--contimeout", "10s", "--timeout", "60s", "--retries", "2", "--log-level", "ERROR"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)

def main():
    if socket.gethostname() != "ide-376f3dcbf2424192b9d1abca3872afc8-445523" or Path.cwd().resolve() != ROOT:
        raise ValueError("Unexpected server/task target")
    allowed = os.sched_getaffinity(0)
    if not set(range(50)) <= allowed:
        raise ValueError("Shared CPU0..49 pool unavailable")
    os.sched_setaffinity(0, {min(allowed)})
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    (ROOT / ".tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(ROOT / ".tmp")
    plan = json.loads((CONTROL / "plan.json").read_text())
    if plan["task"] != TASK or plan["model_source_task"] != "q4-rl-memory-v4-20260912":
        raise ValueError("Unexpected immutable task plan")
    free_before = shutil.disk_usage(ROOT).free
    if free_before < 20 * 1024**3 + sum(m["bytes"] for m in plan["models"]) + 10 * 1024**2:
        raise ValueError("Insufficient disk headroom for immutable prestage")
    source = CONTROL / plan["source_archive"]
    if sha(source) != plan["source_archive_sha256"]:
        raise ValueError("Source archive checksum mismatch")
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        if any(not m.isfile() or PurePosixPath(m.name).is_absolute() or ".." in PurePosixPath(m.name).parts for m in members):
            raise ValueError("Unsafe source archive member")
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicated source archive member")
        binding_raw = archive.extractfile("EVAL_SOURCE_MANIFEST.json").read()
        if hashlib.sha256(binding_raw).hexdigest() != plan["source_manifest_sha256"]:
            raise ValueError("Source binding mismatch")
        binding = json.loads(binding_raw)
        if set(names) != set(binding["files"]) | {"EVAL_SOURCE_MANIFEST.json"}:
            raise ValueError("Undeclared source archive content")
        for member in members:
            data = archive.extractfile(member).read()
            if member.name != "EVAL_SOURCE_MANIFEST.json" and hashlib.sha256(data).hexdigest() != binding["files"][member.name]["sha256"]:
                raise ValueError("Source member mismatch")
            target = ROOT / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != hashlib.sha256(data).hexdigest():
                    raise ValueError("Existing source file differs")
            else:
                with target.open("xb") as stream:
                    stream.write(data)
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    models = []
    for item in plan["models"]:
        name = item["path"]
        if name != "models/eval-memory-v4/" + item["alias"] + ".pt":
            raise ValueError("Unexpected model alias")
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            rclone("copyto", REMOTE + plan["model_source_task"] + "/" + name, str(target))
        if sha(target) != item["sha256"]:
            raise ValueError("Model checksum mismatch")
        if item["loader"] not in {"q4_rl.network:load_policy", "q4_rl.micro_network:load_policy", "q4_rl.memory_network:load_policy"}:
            raise ValueError("Unexpected frozen model loader")
        module, function = item["loader"].split(":")
        policy = getattr(importlib.import_module(module), function)(target, deterministic=True)
        saved = torch.load(target, map_location="cpu", weights_only=True)
        network, state = saved["network"], saved["state"]
        if any(network[k] != item["expected_" + k] for k in ("global_dim", "candidate_dim", "hidden")):
            raise ValueError("Loaded model dimensions changed")
        if state["pending_batch"] is not None:
            raise ValueError("Evaluation alias is not a completed endpoint")
        if item["is_BC256"] and not (state["episodes"] == state["warmstart_completed"] == 256 and state["batches"] == 16 and state["ppo_batches"] == 0):
            raise ValueError("BC endpoint state mismatch")
        if "expected_episodes" in item and not all(state[k] == item["expected_" + k] for k in ("episodes", "batches", "ppo_batches")):
            raise ValueError("PPO endpoint state mismatch")
        rclone("copyto", str(target), REMOTE + TASK + "/" + name)
        models.append({"alias":item["alias"],"sha256":item["sha256"],"network":network,"strict_loader_passed":True})
        del policy, saved
        print("verified " + item["alias"], flush=True)
    os.sched_setaffinity(0, set(range(50)))
    check = subprocess.run([sys.executable, str(CONTROL / "run_panels.py")], cwd=ROOT, check=True, capture_output=True, text=True)
    preflight = json.loads(check.stdout)
    if preflight["launched"] is not False:
        raise ValueError("Unexpected launch during prestage")
    os.sched_setaffinity(0, {min(allowed)})
    result = {"task":TASK,"source_archive_sha256":sha(source),"source_manifest_sha256":plan["source_manifest_sha256"],
        "base_git_commit":binding["base_git_commit"],"r9_qualification_commit":binding["r9_qualification_commit"],
        "identical_common_R9_files":len(binding["identical_common_production_files"]),"only_added_production_files":binding["only_added_production_files"],
        "plan_sha256":sha(CONTROL / "plan.json"),"models":models,"strict_model_loading_cpu_cores":1,
        "free_bytes_before":free_before,"free_bytes_after":shutil.disk_usage(ROOT).free,"minimum_free_gib":20,
        "preflight":preflight,"evaluation_started":False,"training_started":False,
        "next_action":"Await root's B raw eligibility audit and launch review; explicit --launch required"}
    path = CONTROL / "PRESTAGE.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    rclone("copyto", str(path), REMOTE + TASK + "/launch/eval-memory-v4/PRESTAGE.json")
    print("new isolated evaluation task prestaged; all thirteen arms ready; nothing launched", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("prestage failed: " + type(exc).__name__, flush=True)
        raise SystemExit(1)
