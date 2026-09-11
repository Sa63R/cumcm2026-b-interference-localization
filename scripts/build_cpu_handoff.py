"""Build a credential-free, versioned CPU handoff tarball with exact hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "8985b7fc8d1709c5a54bb1aaf8f21df66f56782119eb901163f723a03281734f"


def source_paths():
    """Use one input inventory for both provenance checking and the archive."""
    required = ["scripts/build_cpu_handoff.py", "scripts/cpu_handoff.py",
                "scripts/object_exchange.py", "scripts/START_CPU.sh",
                "scripts/sync_cpu_results.py", "scripts/RUN_WITH_SYNC.sh",
                "research/cpu_v2_protocol.json", "research/v1_protocol.json",
                "research/cpu_v2/README_OPERATOR.md", "pyproject.toml",
                "experiments/__init__.py", "experiments/research_v1_eval.py",
                "experiments/run_q3_comparison.py"]
    paths = {ROOT/name for name in required}
    paths.update((ROOT/"src").rglob("*.py"))
    paths.update(p for p in (ROOT/"research/cpu_v2").rglob("*") if p.is_file())
    for path in paths:
        if (not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(ROOT.resolve())):
            raise ValueError(f"Required package source is missing or is a symlink: {path}")
    return sorted(paths)


def git_output(*args):
    return subprocess.check_output(["git", "-c", "safe.directory="+ROOT.as_posix(), *args],
                                   cwd=ROOT, text=True)


def source_status(paths):
    names = [p.relative_to(ROOT).as_posix() for p in paths]
    dirty = git_output("status", "--porcelain", "--untracked-files=all", "--", *names)
    tracked = set(git_output("ls-files", "--cached", "-z", "--", *names).split("\0"))
    # Ignored source files are also uncommitted, although git status omits them.
    untracked = sorted(set(names)-tracked)
    return dirty.strip(), untracked


def build(checkpoint, output, allow_dirty=False):
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError("Parent checkpoint does not match the frozen selected model")
    paths = source_paths()
    dirty, untracked = source_status(paths)
    dirty_source = bool(dirty or untracked)
    if dirty_source and not allow_dirty:
        raise ValueError("Commit the source before creating a delivery package")
    commit = git_output("rev-parse", "HEAD").strip()
    members = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    for name in list(members):
        if name.endswith(".sh"):
            members[name] = members[name].replace(b"\r\n", b"\n")
    members["START_CPU.sh"] = members["scripts/START_CPU.sh"]
    members["RUN_WITH_SYNC.sh"] = members["scripts/RUN_WITH_SYNC.sh"]
    members["README_OPERATOR.md"] = members["research/cpu_v2/README_OPERATOR.md"]
    members["models/parent.pt"] = checkpoint.read_bytes()
    identity = {"schema": 1, "git_commit": commit, "dirty_source_smoke_only": dirty_source,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "parent_checkpoint_sha256": PARENT_SHA,
                "source_paths": [p.relative_to(ROOT).as_posix() for p in paths],
                "files": {k: hashlib.sha256(v).hexdigest() for k, v in members.items()}}
    members["PACKAGE_MANIFEST.json"] = (json.dumps(identity, indent=2, ensure_ascii=False)+"\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw:
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            for name, content in members.items():
                item = tarfile.TarInfo("q3-cpu-handoff/"+name)
                item.size = len(content)
                item.mode = 0o755 if name.endswith(".sh") else 0o644
                archive.addfile(item, io.BytesIO(content))
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+".sha256").write_text(sha+"  "+output.name+"\n", encoding="utf-8")
    print(json.dumps({"archive": str(output), "sha256": sha, "files": len(members), "git_commit": commit}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty-smoke-only", action="store_true")
    args = parser.parse_args()
    build(args.checkpoint.resolve(), args.output.resolve(), args.allow_dirty_smoke_only)
