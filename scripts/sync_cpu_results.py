"""Periodically copy immutable result snapshots to the authorized S3 task prefix.

Uses the operator's existing rclone configuration. Never copies credentials or
source directories, never deletes remote objects, and never imports Torch.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
REMOTE_BASE = "jiangsu10:bucket-c20250204-pool01/lianghao/bwc/shumo/"


def validate_remote(remote):
    remote = remote.rstrip("/")
    suffix = remote.removeprefix(REMOTE_BASE)
    if not remote.startswith(REMOTE_BASE) or not re.fullmatch(r"[A-Za-z0-9_-]+", suffix):
        raise ValueError("Use one dedicated task folder under the agreed shumo prefix")
    return remote


def result_files(root):
    """Explicit allowlist: only generated results under cpu_runs, no symlinks."""
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root.parent):
            continue
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            continue
        if len(rel.parts) == 1:
            if path.name in {"startup.log", "preflight.json"} or re.fullmatch(
                    r"[A-Za-z0-9_-]+-results-\d{8}T\d{6}Z\.tar\.gz(?:\.sha256)?", path.name):
                yield path
            continue
        if not (root/rel.parts[0]/"run_manifest.json").is_file():
            continue
        if path.suffix in {".json", ".jsonl", ".log", ".pt"} or path.name.endswith(".json.gz"):
            # Training outputs are machine generated; avoid arbitrary operator configs.
            if path.name in {"run_manifest.json", "environment.json", "benchmark.json",
                             "benchmark-progress.json", "summary.json", "error.json", "RESULT_FILES.json"}:
                yield path
            elif rel.parts[1] in {"logs", "benchmark", "evaluation", "trial-1", "trial-2", "trial-3"}:
                yield path


def stable_copy(source, destination):
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("wb") as dst:
        for block in iter(lambda: src.read(1024*1024), b""):
            dst.write(block)
            digest.update(block)
    after = source.stat()
    signature = lambda s: (s.st_size, s.st_mtime_ns, s.st_ino)
    if signature(before) != signature(after):
        destination.unlink()
        return None
    return {"sha256": digest.hexdigest(), "bytes": after.st_size, "source_mtime_ns": after.st_mtime_ns}


def upload(local, remote, timeout=90):
    # No shell expansion, no bucket checks, no deletion or directory sync.
    result = subprocess.run(["rclone", "copyto", str(local), remote,
        "--s3-no-check-bucket", "--retries", "2", "--low-level-retries", "2",
        "--contimeout", "10s", "--timeout", "30s", "--transfers", "1",
        "--checkers", "1", "--stats", "0"], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"rclone copy failed (exit {result.returncode}); check connectivity and existing configuration")


class ResultSync:
    def __init__(self, root, remote, upload_fn=upload):
        self.root, self.remote, self.upload_fn = root.resolve(), validate_remote(remote), upload_fn
        self.index = {}
        self.session = uuid.uuid4().hex[:12]
        self.root.mkdir(parents=True, exist_ok=True)

    def cycle(self, include_models=False, final=False):
        # A manifest is published only after all referenced content objects exist.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        sent, skipped = 0, []
        with tempfile.TemporaryDirectory(prefix=".sync-stage-", dir=self.root) as directory:
            stage = Path(directory)
            for source in result_files(self.root):
                name = source.relative_to(self.root).as_posix()
                if source.suffix == ".pt" and not include_models:
                    continue
                snapshot = stage/"content"
                try:
                    meta = stable_copy(source, snapshot)
                except FileNotFoundError:
                    meta = None
                if meta is None:
                    skipped.append(name)
                    continue
                previous = self.index.get(name)
                if previous and previous["sha256"] == meta["sha256"]:
                    continue
                key = "live/objects/"+meta["sha256"]
                self.upload_fn(snapshot, self.remote+"/"+key)
                sent += 1
                # Return ready archives at conventional names as well.
                if "-results-" in source.name and len(source.relative_to(self.root).parts) == 1:
                    self.upload_fn(snapshot, self.remote+"/results/"+source.name)
                self.index[name] = dict(meta, object_key=key, captured_utc=stamp)
            manifest = {"schema": 1, "captured_utc": stamp, "session": self.session,
                        "final_sync": final, "files": self.index, "changed_during_read": skipped,
                        "scope": "Per-file snapshots; log/model times may differ. Not a completed-run declaration."}
            target = stage/"manifest.json"
            target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
            key = f"live/manifests/{stamp}-{self.session}.json"
            self.upload_fn(target, self.remote+"/"+key)
            # Pointer is last; older pointer stays usable if any preceding upload fails.
            pointer = stage/"LATEST.json"
            pointer.write_text(json.dumps({"manifest_key": key, "captured_utc": stamp,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})+"\n", encoding="utf-8")
            self.upload_fn(pointer, self.remote+"/live/LATEST.json")
        print(json.dumps({"sync_ok": True, "new_files": sent, "known_files": len(self.index),
                          "final_sync": final, "captured_utc": stamp}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--model-interval", type=float, default=600)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 10 or args.model_interval < args.interval:
        parser.error("Use interval >= 10 seconds and model-interval >= interval")
    if shutil.which("rclone") is None:
        parser.error("rclone is required; use the server's existing configured installation")
    sync = ResultSync(ROOT/"cpu_runs", args.remote)
    stop = threading.Event()
    for kind in (signal.SIGINT, signal.SIGTERM):
        signal.signal(kind, lambda *_: stop.set())
    last_models = -float("inf")
    failed = False
    while True:
        if args.watch_pid:
            try:
                os.kill(args.watch_pid, 0)
            except ProcessLookupError:
                stop.set()
        final = args.once or stop.is_set()
        models = final or time.monotonic()-last_models >= args.model_interval
        try:
            sync.cycle(include_models=models, final=final)
            failed = False
            if models:
                last_models = time.monotonic()
        except Exception as exc:
            # Never print rclone config, command environments or raw provider errors.
            print(json.dumps({"sync_ok": False, "error_type": type(exc).__name__,
                              "retry_next_cycle": not final}), flush=True)
            failed = True
        if final:
            return int(failed)
        stop.wait(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
