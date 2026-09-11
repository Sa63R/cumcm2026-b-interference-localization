"""Retrieve prestage evidence only through the task's approved object prefix."""
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
    plan = json.loads((OUT / "plan-r2.json").read_text())
    receipts = []
    for name in ("PRESTAGE.json", "pytest.log", "prestage-r2.log"):
        destination = OUT / ("server-"+name)
        result = subprocess.run([sys.executable, "scripts/q4_object_exchange.py", "--credentials-document", "../AGENTS.md",
            "--task", TASK, "download", "--file", str(destination), "--name", "launch/train-gae-v5/"+name],
            check=True, capture_output=True, text=True)
        receipts.append(json.loads(result.stdout))
    server = json.loads((OUT / "server-PRESTAGE.json").read_text())
    if (server["source_archive_sha256"] != plan["source_archive_sha256"]
            or server["source_manifest_sha256"] != plan["source_manifest_sha256"]
            or server["plan_sha256"] != sha(OUT / "plan-r2.json")
            or server["config_sha256"] != plan["config_sha256"]
            or server["tests"]["log_sha256"] != sha(OUT / "server-pytest.log")
            or server["tests"]["passed"] != 142 or server["tests"]["returncode"] != 0
            or not server["all_source_hashes_unchanged"] or server["training_started"] is not False):
        raise ValueError("Server prestage receipt identity mismatch")
    for item, expected in zip(server["models"], plan["models"]):
        if item["sha256"] != expected["sha256"] or not item["strict_loader_passed"]:
            raise ValueError("Server BC receipt mismatch")
    archive = OUT / plan["source_archive"]
    with tarfile.open(archive, "r:gz") as source:
        for name, target in (("RELEASE_MANIFEST.json", "RELEASE_MANIFEST.json"),
                             ("research/q4_rl/train_gae_v5.json", "train_gae_v5.json")):
            with (OUT / target).open("xb") as stream:
                stream.write(source.extractfile(name).read())
    config = json.loads((OUT / "train_gae_v5.json").read_text())
    for job in config["jobs"]:
        flags = dict(zip(job["argv"][::2], job["argv"][1::2]))
        if flags["--max-batches"] != "32" or flags["--max-wall-seconds"] != "3600":
            raise ValueError("Wrong bounded training plan")
    result = dict(task=TASK, git_commit=server["git_commit"], source_archive=archive.name,
        source_archive_sha256=sha(archive), source_config_sha256=sha(OUT / "train_gae_v5.json"),
        source_manifest_sha256=sha(OUT / "RELEASE_MANIFEST.json"),
        prestage_receipt_sha256=sha(OUT / "server-PRESTAGE.json"), pytest_sha256=sha(OUT / "server-pytest.log"),
        verified_object_readbacks=receipts, tests=server["tests"], models=server["models"],
        max_batches_per_job=32, max_wall_seconds_per_job=3600, compute_budget=36,
        combined_with_declared_C_evaluation=47, common_cpu_pool=list(range(50)), minimum_free_gib=20,
        free_bytes_after=server["free_bytes_after"], training_started=False,
        next_action="Explicit root review and launch authorization required")
    with (OUT / "DEPLOYMENT_RECEIPT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(task=TASK, tests_passed=142, source_sha256=sha(archive),
        free_gib=server["free_bytes_after"]/1024**3, training_started=False)))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("readback failed: " + type(error).__name__, flush=True)
        raise SystemExit(1)
