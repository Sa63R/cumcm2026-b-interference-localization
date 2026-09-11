"""Read back a completed result subtree using object storage only."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from q4_object_exchange import connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-document", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"q4-[A-Za-z0-9_-]+", args.task):
        raise ValueError("Independent Q4 prefix required")
    rel = PurePosixPath(args.prefix)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.parts[0] != "runs":
        raise ValueError("Result prefix must be under runs")
    client = connection(args.credentials_document)
    bucket = "bucket-c20250204-pool01"
    prefix = "lianghao/bwc/shumo/" + args.task + "/" + args.prefix.rstrip("/") + "/"
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    objects = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(page.get("Contents", []))

    def fetch(item):
        relative = PurePosixPath(item["Key"][len(prefix):])
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe remote result name")
        target = root.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(root) or target.is_symlink():
            raise ValueError("Result escaped output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".download")
        client.download_file(bucket, item["Key"], str(temp))
        raw = temp.read_bytes()
        if len(raw) != item["Size"]:
            raise ValueError("Result byte size changed; fetch only completed runs")
        sha = hashlib.sha256(raw).hexdigest()
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != sha:
                raise ValueError("Existing local result differs; choose a new output directory")
            temp.unlink()
        else:
            temp.replace(target)
        return {"name": relative.as_posix(), "bytes": len(raw), "sha256": sha}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = sorted(pool.map(fetch, objects), key=lambda x: x["name"])
    manifest = {"task": args.task, "prefix": args.prefix, "objects": results,
                "files": len(results), "bytes": sum(x["bytes"] for x in results)}
    (root / "OBJECT_READBACK.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("files", "bytes")}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))
        raise SystemExit(1)
