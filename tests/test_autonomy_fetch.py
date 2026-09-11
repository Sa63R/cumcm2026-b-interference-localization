"""Fake-object-store verification; no credentials or network are accessed."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("autonomy_status_fetch_test", ROOT / "scripts/fetch_autonomy_status.py")
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
STAMP = "20260912T010203.123456Z"
SESSION = "012345abcdef"


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


class FakeClient:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []
        self.lengths = {}

    def get_object(self, *, Bucket, Key):
        assert Bucket == "test-bucket"
        assert Key.startswith(fetch.PREFIX + "/")
        self.calls.append(Key)
        data = self.objects[Key]
        return {"Body": io.BytesIO(data), "ContentLength": self.lengths.get(Key, len(data))}


def snapshot(contents=None, mutate_manifest=None, mutate_pointer=None):
    if contents is None:
        contents = {
            "base-r1/summary.json": encoded({"status": "running", "pid": 987654, "start_ticks": "private",
                "training_complete": False, "update": 7, "completed_blocks": [1], "blocks": {"2": {"phase": "training"}},
                "secret_access_key": "DO-NOT-PRINT-SECRET", "operator": "DO-NOT-PRINT-OPERATOR"}),
            "base-r1/environment.json": encoded({"cpu_slots": 50, "requested_slots": 50, "nice": 10,
                "hostname": "DO-NOT-PRINT-HOST", "credential": "DO-NOT-PRINT-SECRET"}),
            "base-r1/evaluation/block-01/summary.json": encoded({"runs": 64, "successful_runs": 64,
                "complete": True, "raw_mean_total_time_s": 3100, "physical_mean_lower_s": 1700,
                "ratio_of_sums": 3100 / 1700, "failed_clear_count": 0, "audit_passed_records": 64}),
            "base-r1/evaluation/block-01/comparisons.json": encoded({"state_search": {"pairs": 64,
                "mean_seconds_saved": 10, "paired_mean_savings_ci95_s": [-2, 22],
                "strict_upgrade_on_supplied_cases": False, "debug": "DO-NOT-PRINT-SECRET"}}),
            "base-r1/trial-1/training.jsonl": b"large private training log",
            "base-r1/trial-1/latest.pt": b"opaque model, never unpickled",
            "base-r1/evaluation/block-01/case-5300001.json.gz": b"opaque compressed case",
            "cost-r1/trial-1/evidence/group-3100001.json.gz": b"opaque evidence DO-NOT-PRINT-EVIDENCE",
            "cost-r1/trial-1/evidence/group-3100002.json.gz": b"unselected evidence",
        }
    entries, objects = {}, {}
    for name, data in contents.items():
        sha = hashlib.sha256(data).hexdigest()
        entries[name] = {"sha256": sha, "bytes": len(data), "object_key": "live/objects/" + sha,
                         "captured_utc": STAMP}
        objects[fetch.PREFIX + "/live/objects/" + sha] = data
    manifest = {"schema": 1, "captured_utc": STAMP, "session": SESSION, "final_sync": False,
                "files": entries, "scope": "per-file snapshots"}
    if mutate_manifest:
        mutate_manifest(manifest)
    key = f"live/manifests/{STAMP}-{SESSION}.json"
    raw = encoded(manifest)
    objects[fetch.PREFIX + "/" + key] = raw
    pointer = {"manifest_key": key, "captured_utc": STAMP, "sha256": hashlib.sha256(raw).hexdigest()}
    if mutate_pointer:
        mutate_pointer(pointer)
    objects[fetch.PREFIX + "/live/LATEST.json"] = encoded(pointer)
    return FakeClient(objects), manifest, pointer


def test_default_fetches_only_verified_metadata_and_printable_summary_is_anonymous(tmp_path):
    client, manifest, pointer = snapshot()
    output = tmp_path / "status"
    result = fetch.fetch_status(client, "test-bucket", output=output)
    assert result["ok"] and result["verified_objects"] == 4
    assert len(client.calls) == 6  # pointer + manifest + four selected objects
    assert result["verified_pointer"]["manifest_sha256"] == pointer["sha256"]
    assert result["verified_pointer"]["manifest_key"] == pointer["manifest_key"]
    assert result["process_liveness"] == "not_observed"
    task = result["tasks"][0]
    assert task["reported_status"] == "running" and task["training"]["update"] == 7
    assert task["block_phases"] == {"2": "training"}
    assert task["environment"]["cpu_slots"] == 50
    evaluation = task["evaluations"]["block-01"]
    assert evaluation["ratio_of_sums"] == pytest.approx(3100 / 1700)
    assert evaluation["comparisons"]["state_search"]["paired_mean_savings_ci95_s"] == [-2, 22]
    assert "DO-NOT-PRINT" not in json.dumps(result) and "987654" not in json.dumps(result)
    assert "base-r1" not in json.dumps(result)
    assert not (output / "base-r1/trial-1").exists()
    assert result["optional_evidence"] == 0 and not (output / "cost-r1").exists()
    for name in manifest["files"]:
        assert (output / name).exists() == fetch.metadata_path(name)
    assert json.loads((output / "verified_status.json").read_text()) == result


def test_optional_models_and_cases_are_exactly_selected_and_not_executed(tmp_path):
    client, _, _ = snapshot()
    model = "base-r1/trial-1/latest.pt"
    case = "base-r1/evaluation/block-01/case-5300001.json.gz"
    result = fetch.fetch_status(client, "test-bucket", output=tmp_path / "selected", models=[model], cases=[case])
    assert result["verified_objects"] == 6 and result["optional_models"] == result["optional_cases"] == 1
    assert (tmp_path / "selected" / model).read_bytes() == b"opaque model, never unpickled"
    assert (tmp_path / "selected" / case).read_bytes() == b"opaque compressed case"


def test_optional_evidence_is_exact_selected_opaque_and_verified(tmp_path):
    client, manifest, _ = snapshot()
    name = "cost-r1/trial-1/evidence/group-3100001.json.gz"
    output = tmp_path / "evidence"
    result = fetch.fetch_status(client, "test-bucket", output=output, evidence=[name, name])
    assert result["verified_objects"] == 5 and result["optional_evidence"] == 1
    assert result["optional_cases"] == result["optional_models"] == 0
    assert len(client.calls) == 7
    assert (output / name).read_bytes() == b"opaque evidence DO-NOT-PRINT-EVIDENCE"
    assert not (output / "cost-r1/trial-1/evidence/group-3100002.json.gz").exists()
    assert not (output / "base-r1/trial-1").exists()
    index = json.loads((output / "fetch_index.json").read_text())["files"]
    assert index[name] == {key: manifest["files"][name][key] for key in ("sha256", "bytes")}
    assert "DO-NOT-PRINT" not in json.dumps(result) and "group-3100001" not in json.dumps(result)
    assert all(key.startswith(fetch.PREFIX + "/") for key in client.calls)


@pytest.mark.parametrize("name", [
    "cost-r1/trial-2/evidence/group-3100001.json.gz",
    "cost-r1/trial-1/group-3100001.json.gz",
    "cost-r1/trial-1/evidence/nested/group-3100001.json.gz",
    "cost-r1/trial-1/evidence/case-3100001.json.gz",
    "cost-r1/trial-1/evidence/group-3100001.json",
    "cost-r1/trial-1/evidence/group-private.json.gz",
    "cost-r1/trial-1/evidence/group-3100001.json.gz.part",
    "cost-r1/trial-1/training.jsonl",
    "cost-r1/trial-1/latest.pt",
    "cost-r1/evaluation/block-01/group-3100001.json.gz",
])
def test_evidence_selector_rejects_other_manifest_artifacts(tmp_path, name):
    client, _, _ = snapshot({name: b"must not download"})
    with pytest.raises(ValueError, match="wrong type"):
        fetch.fetch_status(client, "test-bucket", output=tmp_path / "bad", evidence=[name])
    assert len(client.calls) == 2 and not (tmp_path / "bad").exists()


@pytest.mark.parametrize("name", [
    "cost-r1/trial-1/evidence/group-*.json.gz",
    "cost-r1/trial-1/evidence/../group-3100001.json.gz",
    "cost-r1/trial-1/evidence/group-3100999.json.gz",
    "other-prefix/cost-r1/trial-1/evidence/group-3100001.json.gz",
    "live/objects/" + "0" * 64,
])
def test_evidence_selector_requires_exact_present_safe_relative_path(tmp_path, name):
    client, _, _ = snapshot()
    with pytest.raises(ValueError):
        fetch.fetch_status(client, "test-bucket", output=tmp_path / "bad", evidence=[name])
    assert len(client.calls) == 2 and not (tmp_path / "bad").exists()


@pytest.mark.parametrize("kind", ["models", "cases"])
def test_evidence_cannot_be_selected_using_other_optional_types(tmp_path, kind):
    name = "cost-r1/trial-1/evidence/group-3100001.json.gz"
    client, _, _ = snapshot()
    with pytest.raises(ValueError, match="wrong type"):
        fetch.fetch_status(client, "test-bucket", output=tmp_path / "bad", **{kind: [name]})
    assert len(client.calls) == 2 and not (tmp_path / "bad").exists()


@pytest.mark.parametrize("damage", ["hash", "bytes", "stream_length", "oversize"])
def test_selected_evidence_integrity_and_size_failure_never_publishes_artifact(tmp_path, damage):
    name = "cost-r1/trial-1/evidence/group-3100001.json.gz"
    def mutate(manifest):
        if damage == "bytes":
            manifest["files"][name]["bytes"] = 1
        elif damage == "oversize":
            manifest["files"][name]["bytes"] = fetch.LIMITS["evidence"] + 1
    client, manifest, _ = snapshot({name: b"opaque evidence"}, mutate_manifest=mutate)
    key = fetch.PREFIX + "/" + manifest["files"][name]["object_key"]
    if damage == "hash":
        client.objects[key] = b"changed evidence"
    elif damage == "stream_length":
        client.lengths[key] = len(client.objects[key]) + 1
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        fetch.fetch_status(client, "test-bucket", output=output, evidence=[name])
    assert not (output / name).exists() and not list(output.rglob("*.part"))
    assert not (output / "verified_status.json").exists()
    assert not (output / "fetch_index.json").exists()
    if damage == "oversize":
        assert len(client.calls) == 2


def test_unselected_large_evidence_is_not_downloaded(tmp_path):
    name = "cost-r1/trial-1/evidence/group-3100001.json.gz"
    client, _, _ = snapshot(mutate_manifest=lambda m: m["files"][name].update(bytes=20 << 30))
    result = fetch.fetch_status(client, "test-bucket", output=tmp_path / "metadata")
    assert result["verified_objects"] == 4 and result["optional_evidence"] == 0
    assert len(client.calls) == 6


def test_cli_evidence_option_is_independent_exact_and_stdout_stays_anonymous(monkeypatch, tmp_path, capsys):
    client, _, _ = snapshot()
    monkeypatch.setattr(fetch, "client_from_config", lambda _: (client, "test-bucket"))
    name = "cost-r1/trial-1/evidence/group-3100001.json.gz"
    output = tmp_path / "selected"
    assert fetch.main(["--config", str(tmp_path / "unused.json"), "--output", str(output),
                       "--evidence", name]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["optional_evidence"] == 1 and result["optional_models"] == result["optional_cases"] == 0
    assert captured.err == "" and "DO-NOT-PRINT" not in captured.out and str(tmp_path) not in captured.out
    assert (output / name).is_file()


@pytest.mark.parametrize("pointer_change", [
    {"manifest_key": "../unrelated.json"}, {"manifest_key": "live/objects/elsewhere.json"},
    {"sha256": "not-a-hash"}, {"captured_utc": "../../outside"},
])
def test_untrusted_pointer_cannot_escape_manifest_namespace(tmp_path, pointer_change):
    client, _, _ = snapshot(mutate_pointer=lambda p: p.update(pointer_change))
    with pytest.raises(ValueError):
        fetch.fetch_status(client, "test-bucket", output=tmp_path / "bad")
    assert len(client.calls) == 1 and not (tmp_path / "bad").exists()


def test_manifest_hash_and_declared_identity_are_both_verified(tmp_path):
    client, _, _ = snapshot(mutate_pointer=lambda p: p.update(sha256="0" * 64))
    with pytest.raises(ValueError, match="Manifest SHA256"):
        fetch.fetch_status(client, "test-bucket", output=tmp_path / "hash")
    client, _, _ = snapshot(mutate_manifest=lambda m: m.update(captured_utc="20260911T000000Z"))
    with pytest.raises(ValueError, match="identity"):
        fetch.fetch_status(client, "test-bucket", output=tmp_path / "identity")


@pytest.mark.parametrize("name", ["../outside", "/absolute", "job/../escape", "job//summary.json",
    "job\\summary.json", "C:/outside.json", "job/a:b.json", "job/CON/summary.json",
    "job/evaluation/a%2fb/summary.json", "job./summary.json"])
def test_every_manifest_path_is_validated_even_if_not_selected(tmp_path, name):
    def inject(manifest):
        manifest["files"][name] = copy.deepcopy(next(iter(manifest["files"].values())))
    client, _, _ = snapshot(mutate_manifest=inject)
    with pytest.raises(ValueError):
        fetch.fetch_status(client, "test-bucket", output=tmp_path / "unsafe")
    assert len(client.calls) == 2 and not (tmp_path / "unsafe").exists()


@pytest.mark.parametrize("field,value", [("object_key", "../private"), ("object_key", "live/objects/" + "0" * 64),
                                         ("bytes", -1), ("bytes", True), ("sha256", "invalid")])
def test_content_object_keys_are_locked_to_the_exact_sha(tmp_path, field, value):
    def mutate(manifest):
        next(iter(manifest["files"].values()))[field] = value
    client, _, _ = snapshot(mutate_manifest=mutate)
    with pytest.raises(ValueError):
        fetch.fetch_status(client, "test-bucket", output=tmp_path / "bad")
    assert len(client.calls) == 2


@pytest.mark.parametrize("damage", ["hash", "bytes", "stream_length"])
def test_corrupt_object_never_becomes_a_verified_local_artifact(tmp_path, damage):
    name = "job/environment.json"
    client, manifest, _ = snapshot({name: encoded({"cpu_slots": 50})},
        mutate_manifest=(lambda m: m["files"][name].update(bytes=1)) if damage == "bytes" else None)
    key = fetch.PREFIX + "/" + manifest["files"][name]["object_key"]
    if damage == "hash":
        client.objects[key] = encoded({"cpu_slots": 60})
    if damage == "stream_length":
        client.lengths[key] = len(client.objects[key]) + 1
    with pytest.raises(ValueError):
        fetch.fetch_status(client, "test-bucket", output=tmp_path / "bad")
    assert not (tmp_path / "bad" / name).exists()
    assert not list((tmp_path / "bad").rglob("*.part"))
    assert not (tmp_path / "bad/verified_status.json").exists()


def test_large_unselected_log_does_not_block_small_metadata_fetch(tmp_path):
    def enlarge(manifest):
        manifest["files"]["base-r1/trial-1/training.jsonl"]["bytes"] = 20 << 30
    client, _, _ = snapshot(mutate_manifest=enlarge)
    assert fetch.fetch_status(client, "test-bucket", output=tmp_path / "small")["verified_objects"] == 4


def test_wrong_prefix_and_existing_output_rejected_without_requests(tmp_path):
    client, _, _ = snapshot()
    with pytest.raises(ValueError):
        fetch.fetch_status(client, "test-bucket", prefix="lianghao/bwc/shumo/other", output=tmp_path / "new")
    with pytest.raises(ValueError):
        fetch.fetch_status(client, "test-bucket", output=tmp_path)
    assert client.calls == []


def test_low_level_reader_also_rejects_unscoped_object_keys():
    client, _, _ = snapshot()
    for key in ("../unrelated", "other-task/live/LATEST.json", "live/manifests/../private.json"):
        with pytest.raises(ValueError):
            fetch.read_object(client, "test-bucket", key, 100)
    assert client.calls == []


def test_missing_or_wrong_type_optional_selection_does_not_download_other_objects(tmp_path):
    for kind, name in (("models", "base-r1/trial-1/missing.pt"),
                       ("models", "base-r1/trial-1/training.jsonl"),
                       ("cases", "../case-1.json.gz")):
        client, _, _ = snapshot()
        with pytest.raises(ValueError):
            fetch.fetch_status(client, "test-bucket", output=tmp_path / "bad", **{kind: [name]})
        assert len(client.calls) == 2 and not (tmp_path / "bad").exists()


def test_stream_limit_is_enforced_without_content_length_and_body_is_closed():
    body = io.BytesIO(b"x" * 100)
    client = SimpleNamespace(get_object=lambda **kwargs: {"Body": body})
    with pytest.raises(ValueError, match="stream exceeds"):
        fetch.read_object(client, "test-bucket", "live/LATEST.json", 20)
    assert body.closed


def test_default_output_uses_manifest_time_and_never_overwrites(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    client, _, _ = snapshot()
    fetch.fetch_status(client, "test-bucket")
    fetch.fetch_status(client, "test-bucket")
    outputs = sorted((tmp_path / "handoff").iterdir())
    assert len(outputs) == 2 and all(STAMP in p.name for p in outputs)
    assert all((p / "verified_status.json").is_file() for p in outputs)


def test_duplicate_json_members_fail_closed():
    with pytest.raises(ValueError, match="Duplicate"):
        fetch.parsed_json(b'{"files":{},"files":{"escape":1}}')


def test_cli_failure_never_prints_provider_secrets_or_private_paths(monkeypatch, tmp_path, capsys):
    def fail(path):
        raise RuntimeError("secret_access_key=PRIVATE-CREDENTIAL /home/private-user/q3-object-storage.json")
    monkeypatch.setattr(fetch, "client_from_config", fail)
    assert fetch.main(["--config", str(tmp_path / "private.json")]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"ok": False, "error_type": "RuntimeError"}
    assert captured.err == "" and "PRIVATE" not in captured.out and str(tmp_path) not in captured.out


def test_external_config_prefix_is_not_allowed_to_redirect_task(monkeypatch, tmp_path, capsys):
    client, _, _ = snapshot()
    calls = []
    def create(service, **kwargs):
        calls.append((service, kwargs))
        return client
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=create))
    monkeypatch.setitem(sys.modules, "botocore.config", SimpleNamespace(Config=lambda **kwargs: kwargs))
    config = tmp_path / "q3-object-storage.json"
    config.write_text(json.dumps({"endpoint": "https://storage.example.invalid", "bucket": "test-bucket",
        "access_key_id": "FAKE-ACCESS", "secret_access_key": "FAKE-SECRET", "prefix": "old-task"}))
    assert fetch.main(["--config", str(config), "--output", str(tmp_path / "status")]) == 0
    assert calls[0][1]["aws_secret_access_key"] == "FAKE-SECRET"
    assert all(key.startswith(fetch.PREFIX + "/") for key in client.calls)
    output = capsys.readouterr().out
    assert "FAKE-SECRET" not in output and "storage.example" not in output and "test-bucket" not in output
    assert "DO-NOT-PRINT" not in output
