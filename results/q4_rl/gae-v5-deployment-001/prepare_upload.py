"""Publish the exact new GAE source, initial weights and prestage plan through S3."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path.cwd().resolve()
OUT = Path(__file__).resolve().parent
TASK = "q4-rl-gae-v5-20260912"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = OUT / "q4-gae-v5-source-20260912-r2.tar.gz"
    with tarfile.open(archive, "r:gz") as source:
        manifest_raw = source.extractfile("RELEASE_MANIFEST.json").read()
    manifest = json.loads(manifest_raw)
    if manifest["git_commit"] != "df36f0ce43a8b1e23d6ebf1cb1b0bba6f16ef01d":
        raise ValueError("Unexpected source commit")
    config_path = ROOT / "research/q4_rl/train_gae_v5.json"
    if sha(config_path) != manifest["files"]["research/q4_rl/train_gae_v5.json"]["sha256"]:
        raise ValueError("Config/source mismatch")
    models = [dict(path="models/initialization/g1_h128_bc256.pt", candidate_dim=50,
                   loader="q4_rl.micro_initialization:load_micro_warmstart",
                   sha256="a3170afc53ae48d5b03618872400a424ce27d9c1110f7f2b342fdea13837688d"),
              dict(path="models/initialization/g3_h128_bc256.pt", candidate_dim=58,
                   loader="q4_rl.memory_initialization:load_memory_warmstart",
                   sha256="61d4ac338cdfc0b62fcbb0cf9c4f2baaf9f832c8d203ac8d33a0f9b308f5ce49")]
    for model in models:
        path = ROOT / model["path"]
        if sha(path) != model["sha256"]:
            raise ValueError("BC identity mismatch")
        model["bytes"] = path.stat().st_size
    plan = dict(task=TASK, git_commit=manifest["git_commit"], source_archive=archive.name,
        source_archive_sha256=sha(archive), source_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        prestage_sha256=sha(OUT / "prestage.py"), config_sha256=sha(config_path), models=models,
        superseded_prelaunch_source=dict(filename="q4-gae-v5-source-20260912-r1.tar.gz",
            sha256="fbd4c56f3723078b4cb441b9401c679447eee1ac3bf79f549298398ca47d5262",
            git_commit="4e7584cbc9318a386498cdd76e34e72709de8351", uploaded=False, launched=False),
        test_modules=["tests/test_q4_rl_gae.py", "tests/test_q4_rl_memory_initialization.py",
            "tests/test_q4_rl_micro_training.py", "tests/test_q4_rl_memory_training.py",
            "tests/test_q4_rl_learner_threads.py", "tests/test_q4_training_bundle.py",
            "tests/test_q4_rl_micro_initialization.py", "tests/test_q4_rl_micro_attention.py",
            "tests/test_q4_rl_training_journal.py"],
        launch_command=["taskset", "--cpu-list", "0-49", "{CPU_PYTHON}", "scripts/q4_cpu_supervisor.py",
            "--root", "{TASK_ROOT}", "--run", "train-gae-v5", "--cpu-budget", "50",
            "--deadline", "2026-09-11T22:00:00+00:00", "--sync-seconds", "120",
            "--minimum-free-gib", "20", "--", "{CPU_PYTHON}", "-m", "scripts.q4_training_bundle",
            "--config", "research/q4_rl/train_gae_v5.json"],
        training_started=False, evaluation_started=False, launch_requires_root_review=True)
    with (OUT / "plan.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    uploads = [(archive, "launch/train-gae-v5/"+archive.name),
               (OUT / "prestage.py", "launch/train-gae-v5/prestage.py"),
               (OUT / "plan.json", "launch/train-gae-v5/plan.json")]
    uploads.extend((ROOT / model["path"], model["path"]) for model in models)
    receipts = []
    for path, name in uploads:
        result = subprocess.run([sys.executable, "scripts/q4_object_exchange.py", "--credentials-document", "../AGENTS.md",
            "--task", TASK, "upload", "--file", str(path), "--name", name], check=True, capture_output=True, text=True)
        receipts.append(json.loads(result.stdout))
        print(json.dumps(dict(name=name, sha256=sha(path), object_readback_verified=True)), flush=True)
    with (OUT / "UPLOAD_RECEIPTS.json").open("x", encoding="utf-8") as stream:
        json.dump(receipts, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("prepare upload failed: " + type(error).__name__, flush=True)
        raise SystemExit(1)
