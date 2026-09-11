import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("sync_cpu_results", Path(__file__).parents[1]/"scripts/sync_cpu_results.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)
REMOTE = sync.REMOTE_BASE+"q3-cpu-test"


def run_tree(tmp_path):
    root = tmp_path/"cpu_runs"
    trial = root/"run01"/"trial-1"
    trial.mkdir(parents=True)
    (root/"run01/run_manifest.json").write_text("{}")
    (trial/"latest.pt").write_bytes(b"cpu-checkpoint")
    (trial/"training.jsonl").write_text('{"update": 1}\n')
    (root/".credentials.json").write_text("should never upload")
    (root/"other-config.json").write_text("should never upload")
    return root


def test_sync_pointer_refers_to_complete_hashed_content_and_models_are_deferred(tmp_path):
    root = run_tree(tmp_path)
    remote = {}
    uploader = sync.ResultSync(root, REMOTE, lambda p, r: remote.__setitem__(r, p.read_bytes()))
    uploader.cycle()
    assert "run01/trial-1/latest.pt" not in uploader.index
    uploader.cycle(include_models=True, final=True)
    pointer = json.loads(remote[REMOTE+"/live/LATEST.json"])
    manifest_bytes = remote[REMOTE+"/"+pointer["manifest_key"]]
    assert hashlib.sha256(manifest_bytes).hexdigest() == pointer["sha256"]
    manifest = json.loads(manifest_bytes)
    assert manifest["final_sync"]
    assert len(manifest["files"]) == 3
    for entry in manifest["files"].values():
        assert hashlib.sha256(remote[REMOTE+"/"+entry["object_key"]]).hexdigest() == entry["sha256"]


def test_failed_upload_does_not_publish_new_pointer_and_retries(tmp_path):
    root = run_tree(tmp_path)
    remote = {}
    fail = False
    def upload(p, r):
        if fail and r.endswith("/"+hashlib.sha256(b"updated").hexdigest()):
            raise OSError("network unavailable")
        remote[r] = p.read_bytes()
    uploader = sync.ResultSync(root, REMOTE, upload)
    uploader.cycle()
    before = remote[REMOTE+"/live/LATEST.json"]
    (root/"run01/trial-1/training.jsonl").write_bytes(b"updated")
    fail = True
    with pytest.raises(OSError):
        uploader.cycle()
    assert remote[REMOTE+"/live/LATEST.json"] == before
    fail = False
    uploader.cycle()
    assert remote[REMOTE+"/live/LATEST.json"] != before


@pytest.mark.parametrize("remote", ["jiangsu10:bucket-c20250204-pool01", sync.REMOTE_BASE,
    sync.REMOTE_BASE+"../other", sync.REMOTE_BASE+"job/child", "https://example.com"])
def test_sync_rejects_wrong_or_broad_destinations(remote):
    with pytest.raises(ValueError):
        sync.validate_remote(remote)


def test_rclone_command_does_not_check_bucket_or_delete(monkeypatch, tmp_path):
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd)
        return type("Result", (), {"returncode": 0})()
    monkeypatch.setattr(sync.subprocess, "run", run)
    sync.upload(tmp_path/"file", REMOTE+"/live/test")
    assert commands[0][1] == "copyto"
    assert "--s3-no-check-bucket" in commands[0]
    assert not {"sync", "delete", "purge"}.intersection(commands[0])


def test_result_archive_return_retries_if_conventional_copy_failed(tmp_path):
    root = run_tree(tmp_path)
    archive = root/"run01-results-20260911T140000Z.tar.gz"
    archive.write_bytes(b"archive")
    attempted = []
    def upload(p, remote):
        attempted.append(remote)
        if "/results/" in remote and attempted.count(remote) == 1:
            raise OSError("temporary failure")
    uploader = sync.ResultSync(root, REMOTE, upload)
    with pytest.raises(OSError):
        uploader.cycle()
    uploader.cycle()
    assert attempted.count(REMOTE+"/results/"+archive.name) == 2
