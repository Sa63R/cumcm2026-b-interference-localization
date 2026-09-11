"""Prepare a version-locked local simulator copy for the practice bridge.

The installed executable is read only. The only binary change replaces the
environment-variable name that Wails clears before constructing WebView2.
Simulator logic, assets, bindings and test modes are not patched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ORIGINAL_SHA256 = "110542502401761b81f701ca1a866092e38d4ac290fa025bf18aed73ff8e9628"
PATCHED_SHA256 = "aeb8827e6c5f318e31f5f7514348422ce479d8a4059b3a8c674d760c08940eaf"
ORIGINAL_NAME = "jammers-simulator-full.exe"
ORIGINAL_ENV_KEY = b"WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"
UNUSED_ENV_KEY = b"JAMMERS_PRACTICE_BROWSER_UNUSED_ARGS_"
RUNTIME_DIRECTORY = ".practice-control"


class RuntimePreparationError(RuntimeError):
    """The installed version or local copy does not match the audited version."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _patch_bytes(data: bytes, *, expected_sha256: str = ORIGINAL_SHA256) -> tuple[bytes, int]:
    """Private pure helper; the public preparer always pins the original hash."""
    if _sha256(data) != expected_sha256:
        raise RuntimePreparationError("Original simulator SHA-256 does not match the audited version")
    if len(ORIGINAL_ENV_KEY) != len(UNUSED_ENV_KEY):
        raise RuntimePreparationError("Environment-key replacement must preserve byte length")
    if data.count(ORIGINAL_ENV_KEY) != 1:
        raise RuntimePreparationError("Expected exactly one environment-key occurrence")
    if UNUSED_ENV_KEY in data:
        raise RuntimePreparationError("Replacement environment key already exists")
    offset = data.index(ORIGINAL_ENV_KEY)
    end = offset + len(ORIGINAL_ENV_KEY)
    patched = data[:offset] + UNUSED_ENV_KEY + data[end:]
    if (len(patched) != len(data) or patched[:offset] != data[:offset]
            or patched[offset:end] != UNUSED_ENV_KEY or patched[end:] != data[end:]):
        raise RuntimePreparationError("Unexpected change outside the environment-key replacement")
    return patched, offset


def _normal_path(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _write_or_verify(path: Path, expected: bytes) -> None:
    if path.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(path)):
        raise RuntimePreparationError(f"Refusing redirected local artifact: {path.name}")
    try:
        with path.open("xb") as stream:
            stream.write(expected)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != expected:
            raise RuntimePreparationError(f"Existing local artifact differs: {path.name}") from None


def prepare_runtime(original_exe: str | Path) -> dict[str, Any]:
    """Create or verify only the patched executable and private audit manifest.

    This function never starts processes, reads simulator data, or creates data
    links. The PowerShell launcher verifies process/port state and directory
    junctions separately before starting the copy.
    """
    original = Path(original_exe).resolve(strict=True)
    if original.name.casefold() != ORIGINAL_NAME.casefold() or not original.is_file():
        raise RuntimePreparationError(f"Expected installed executable named {ORIGINAL_NAME}")
    if original.parent.name.casefold() == RUNTIME_DIRECTORY.casefold():
        raise RuntimePreparationError("Pass the original installed simulator, not its local copy")
    source = original.read_bytes()
    patched, offset = _patch_bytes(source)
    if _sha256(patched) != PATCHED_SHA256:
        raise RuntimePreparationError("Patched simulator SHA-256 differs from the audited copy")

    destination = original.parent / RUNTIME_DIRECTORY
    if destination.is_symlink() or (hasattr(os.path, "isjunction") and os.path.isjunction(destination)):
        raise RuntimePreparationError("Local runtime directory must not be a symbolic link or junction")
    if _normal_path(destination.resolve()) != _normal_path(destination):
        raise RuntimePreparationError("Local runtime directory resolves outside its expected location")
    destination.mkdir(exist_ok=True)
    copied = destination / original.name
    manifest = {
        "schema_version": 1,
        "purpose": "practice_control_webview2_debug_environment",
        "original_exe": str(original),
        "copy_exe": str(copied),
        "original_sha256": ORIGINAL_SHA256,
        "copy_sha256": PATCHED_SHA256,
        "changed_offset": offset,
        "replacement_length": len(ORIGINAL_ENV_KEY),
        "original_environment_key": ORIGINAL_ENV_KEY.decode("ascii"),
        "replacement_environment_key": UNUSED_ENV_KEY.decode("ascii"),
    }
    _write_or_verify(copied, patched)
    _write_or_verify(
        destination / "preparation-manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    if _sha256(original.read_bytes()) != ORIGINAL_SHA256:
        raise RuntimePreparationError("Installed simulator changed during preparation")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the audited local practice-control runtime copy")
    parser.add_argument("--simulator-exe", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        result = prepare_runtime(arguments.simulator_exe)
    except (OSError, RuntimePreparationError) as error:
        parser.exit(2, f"Practice runtime preparation refused: {error}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
