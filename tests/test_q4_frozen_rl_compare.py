import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from experiments.run_q4_frozen_rl_compare import verify_archive
from experiments.q4_frozen_rl_worker import configure


def sha(data):
    return hashlib.sha256(data).hexdigest()


def test_exact_frozen_archive(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("src/example.py", b"x=1\n")
    verify_archive(path, {"src/example.py": sha(b"x=1\n")})


@pytest.mark.parametrize("member", ["../x.py", "C:/x.py", "/x.py", "src\\x.py"])
def test_archive_rejects_escaping_members(tmp_path, member):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, b"x=1\n")
    with pytest.raises(ValueError, match="Unsafe"):
        verify_archive(path, {member: sha(b"x=1\n")})


def test_rejects_unmanifested_code(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("src/a.py", b"a")
        archive.writestr("src/b.py", b"b")
    with pytest.raises(ValueError, match="membership"):
        verify_archive(path, {"src/a.py": sha(b"a")})


def test_rejects_changed_source(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("src/a.py", b"changed")
    with pytest.raises(ValueError, match="source mismatch"):
        verify_archive(path, {"src/a.py": sha(b"original")})


def test_worker_rejects_modified_checkpoint_before_import(tmp_path):
    source = tmp_path / "src/a.py"
    source.parent.mkdir()
    source.write_bytes(b"a")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"changed")
    reference = {"source_sha256": {"src/a.py": sha(b"a")}, "checkpoint_sha256": sha(b"original")}
    with pytest.raises(ValueError, match="checkpoint changed"):
        configure(tmp_path, reference, checkpoint)
