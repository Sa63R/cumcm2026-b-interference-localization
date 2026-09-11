"""Build an allowlisted, credential-free Q4 source release with byte hashes."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new immutable release filename")
    files = [ROOT / "scripts/q4_setup_cpu.sh"]
    files.extend((ROOT / "bootstrap").glob("*.whl"))
    if not args.setup_only:
        for package in ("simulation", "simulator_client", "strategies", "planning", "geometry", "localization", "q4_rl"):
            files.extend((ROOT / "src" / package).rglob("*.py"))
        files.extend((ROOT / "experiments").glob("*.py"))
        files.extend((ROOT / "scripts").glob("q4_*.py"))
        files.extend((ROOT / "research/q4_rl").glob("*.json"))
        files.extend((ROOT / "tests").glob("test_q4_rl*.py"))
        files.append(ROOT / "pyproject.toml")
    contents = {}
    for path in sorted(set(files)):
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Release path escapes source root")
        raw = path.read_bytes()
        if path.suffix == ".whl":
            if not path.name.startswith("pip-25.2-"):
                raise ValueError("Unexpected bootstrap wheel")
            contents[path.relative_to(ROOT).as_posix()] = raw
            continue
        content = raw.decode("utf-8-sig")
        if re.search(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----", content):
            raise ValueError("Private key marker in release input")
        if re.search(r"(?i)(?:secret_access_key|access_key_id)\s*[=:]\s*['\"]?[a-z0-9]{24,}", content):
            raise ValueError("Credential assignment in release input")
        contents[path.relative_to(ROOT).as_posix()] = raw
    manifest = {"schema": 1, "task": "q4-deep-rl", "setup_only": args.setup_only,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                  for name, raw in contents.items()}}
    contents["RELEASE_MANIFEST.json"] = json.dumps(manifest, indent=2).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for name, raw in contents.items():
            info = tarfile.TarInfo(name)
            # A deterministic ZIP-compatible timestamp lets the evaluator archive
            # the extracted source with the standard zipfile implementation.
            info.size, info.mode, info.mtime = len(raw), 0o600, 946684800
            archive.addfile(info, io.BytesIO(raw))
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        sha + "  " + args.output.name + "\n", encoding="utf-8")
    print(json.dumps({"file": args.output.name, "sha256": sha, "file_count": len(contents)}))


if __name__ == "__main__":
    main()
