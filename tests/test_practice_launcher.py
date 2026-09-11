"""Offline checks: no real simulator execution or installed-data access."""

import hashlib
import json
from pathlib import Path

import pytest

from practice_control import runtime


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixture_exe(tmp_path, monkeypatch):
    original = tmp_path / runtime.ORIGINAL_NAME
    content = b"MZ\x00fixture-prefix" + runtime.ORIGINAL_ENV_KEY + b"\x00fixture-suffix"
    patched = content.replace(runtime.ORIGINAL_ENV_KEY, runtime.UNUSED_ENV_KEY)
    original.write_bytes(content)
    monkeypatch.setattr(runtime, "ORIGINAL_SHA256", digest(content))
    monkeypatch.setattr(runtime, "PATCHED_SHA256", digest(patched))
    original_patch = runtime._patch_bytes
    monkeypatch.setattr(runtime, "_patch_bytes", lambda data: original_patch(data, expected_sha256=digest(content)))
    return original, content, patched


def test_patch_changes_only_one_equal_length_key():
    source = b"prefix\x00" + runtime.ORIGINAL_ENV_KEY + b"\x00suffix"
    patched, offset = runtime._patch_bytes(source, expected_sha256=digest(source))
    end = offset + len(runtime.ORIGINAL_ENV_KEY)
    assert len(source) == len(patched)
    assert len(runtime.ORIGINAL_ENV_KEY) == len(runtime.UNUSED_ENV_KEY) == 37
    assert patched[:offset] == source[:offset]
    assert patched[end:] == source[end:]
    assert patched[offset:end] == runtime.UNUSED_ENV_KEY


@pytest.mark.parametrize("source", [b"missing", runtime.ORIGINAL_ENV_KEY * 2,
                                         runtime.ORIGINAL_ENV_KEY + runtime.UNUSED_ENV_KEY])
def test_patch_rejects_ambiguous_or_prepatched_bytes(source):
    with pytest.raises(runtime.RuntimePreparationError):
        runtime._patch_bytes(source, expected_sha256=digest(source))


def test_unknown_binary_hash_fails_before_any_copy(tmp_path):
    original = tmp_path / runtime.ORIGINAL_NAME
    original.write_bytes(b"MZ" + runtime.ORIGINAL_ENV_KEY)
    before = original.read_bytes()
    with pytest.raises(runtime.RuntimePreparationError, match="SHA-256"):
        runtime.prepare_runtime(original)
    assert original.read_bytes() == before
    assert not (tmp_path / runtime.RUNTIME_DIRECTORY).exists()


def test_preparation_preserves_original_and_reuses_exact_copy(tmp_path, monkeypatch):
    original, content, patched = fixture_exe(tmp_path, monkeypatch)
    manifest = runtime.prepare_runtime(original)
    copied = Path(manifest["copy_exe"])
    assert original.read_bytes() == content
    assert copied.name == original.name
    assert copied.parent == original.parent / runtime.RUNTIME_DIRECTORY
    assert copied.read_bytes() == patched
    assert manifest["changed_offset"] == content.index(runtime.ORIGINAL_ENV_KEY)
    assert json.loads((copied.parent / "preparation-manifest.json").read_text()) == manifest
    assert runtime.prepare_runtime(original) == manifest
    assert not (copied.parent / "JammersSimulatorData").exists()


def test_existing_different_copy_is_not_overwritten(tmp_path, monkeypatch):
    original, content, _ = fixture_exe(tmp_path, monkeypatch)
    copied = tmp_path / runtime.RUNTIME_DIRECTORY / original.name
    copied.parent.mkdir()
    copied.write_bytes(b"unrecognized local file")
    with pytest.raises(runtime.RuntimePreparationError, match="Existing local artifact differs"):
        runtime.prepare_runtime(original)
    assert copied.read_bytes() == b"unrecognized local file"
    assert original.read_bytes() == content


def test_preparation_rejects_wrong_patched_hash_before_writes(tmp_path, monkeypatch):
    original, content, _ = fixture_exe(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, "PATCHED_SHA256", "0" * 64)
    with pytest.raises(runtime.RuntimePreparationError, match="Patched simulator SHA-256"):
        runtime.prepare_runtime(original)
    assert original.read_bytes() == content
    assert not (tmp_path / runtime.RUNTIME_DIRECTORY).exists()


def test_alternate_original_basename_is_rejected(tmp_path):
    original = tmp_path / "other.exe"
    original.write_bytes(b"MZ")
    with pytest.raises(runtime.RuntimePreparationError, match="named"):
        runtime.prepare_runtime(original)


def test_launcher_has_no_process_termination_or_persistent_environment():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "start_practice_control.ps1").read_text()
    assert "Start-Process" in script and "-WindowStyle Hidden" in script
    assert "Stop-Process" not in script and "taskkill" not in script.lower()
    assert "'Machine'" not in script and "'User'" not in script
    assert "--remote-debugging-address=127.0.0.1" in script
    assert script.index("Assert-SimulatorStopped\nAssert-DebugPortFree") < script.index("practice_control.runtime")
