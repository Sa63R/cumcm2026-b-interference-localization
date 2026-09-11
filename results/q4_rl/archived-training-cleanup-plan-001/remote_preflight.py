"""Read-only exact-file audit. This program has no deletion mode."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import time

BASE = Path("/home/dataset-assist-0/usr/lh/ysh/bwc/shumo")
OWN = BASE / "q4-rl-gae-v5-20260912"
EXPECTED = "ee3448179cd5cf1db269ea6989918533ee480c37d7343321c047d41e5db18330"
TASKS = {"q4-rl-memory-v4-20260912": "train-memory-v4", "q4-rl-bundle-v3-20260912": "train-bundle-v3"}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def dependencies(roots):
    references, unreadable = [], []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / "cmdline").read_bytes()
            if any(str(root).encode() in command for root in roots):
                references.append({"pid": int(proc.name), "kind": "command"})
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if any(target.startswith(str(root) + "/") for root in roots):
                    references.append({"pid": int(proc.name), "kind": "open_file"})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unreadable.append(int(proc.name))
    return dict(references=references, unreadable_process_count=len(unreadable))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert Path.cwd().resolve() == OWN.resolve() == OWN
    assert args.output.resolve().is_relative_to(OWN) and not args.output.exists()
    assert sha(args.manifest) == EXPECTED
    plan = json.loads(args.manifest.read_bytes())
    assert {t["task"]: t["run"] for t in plan["tasks"]} == TASKS
    roots = [BASE / t / "runs" / run for t, run in TASKS.items()]
    before = dependencies(roots)
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=EXPECTED, read_only=True, deleted_files=0,
        dependency_before=before, tasks=[], disk_free_before=shutil.disk_usage(OWN).free)
    started = time.monotonic()
    for task in plan["tasks"]:
        root = BASE / task["task"] / "runs" / task["run"]
        assert root.resolve() == root and root.is_dir()
        assert task["remote_run_root"] == str(root)
        listing = subprocess.run(["rclone", "lsjson", task["object_archive_prefix"], "--recursive",
            "--files-only", "--s3-no-check-bucket", "--checkers", "1", "--retries", "1",
            "--low-level-retries", "1", "--stats", "0", "--log-level", "ERROR"],
            capture_output=True, check=True, timeout=120)
        objects = {r["Path"]: r["Size"] for r in json.loads(listing.stdout)}
        count = size = blocks = 0
        for item in task["files"]:
            rel = PurePosixPath(item["name"])
            assert not rel.is_absolute() and ".." not in rel.parts
            assert ("-episode-" in rel.name or "-leg-" in rel.name) and rel.name.endswith(".json.gz")
            path = root.joinpath(*rel.parts)
            assert path.resolve() == path and path.is_relative_to(root)
            metadata = path.lstat()
            assert stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1
            assert metadata.st_size == item["bytes"] == objects[item["name"]]
            assert sha(path) == item["sha256"]
            count += 1
            size += metadata.st_size
            blocks += metadata.st_blocks * 512
        assert count == task["file_count"] and size == task["logical_payload_bytes"]
        result["tasks"].append(dict(task=task["task"], files=count, bytes=size,
            allocated_bytes=blocks, all_current_remote_sha256_match=True,
            all_object_archive_names_and_sizes_present=True))
        print(json.dumps(result["tasks"][-1]), flush=True)
    result["dependency_after"] = dependencies(roots)
    result["disk_free_after"] = shutil.disk_usage(OWN).free
    result["elapsed_s"] = time.monotonic() - started
    result["ready_for_explicit_approval"] = all(not d["references"] and d["unreadable_process_count"] == 0
        for d in (before, result["dependency_after"]))
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"audit_complete": True, "ready_for_explicit_approval": result["ready_for_explicit_approval"],
        "deleted_files": 0, "elapsed_s": result["elapsed_s"]}), flush=True)

if __name__ == "__main__":
    main()
