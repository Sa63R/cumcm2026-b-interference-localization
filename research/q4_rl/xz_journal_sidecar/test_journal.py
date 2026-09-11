"""Synthetic serialization fixtures only: no simulation or real training data."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("xz_journal_prototype", HERE / "journal.py")
journal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(journal)


def put_gzip(path, value):
    path.write_bytes(gzip.compress(journal.canonical_bytes(value), compresslevel=6, mtime=0))


def sample(seed=1):
    return {"seed": seed, "records": [{"text": "公开观测→频道", "floats": [-0.0, 1.25, 1e-20],
        "candidate_features": [[.125, -.875], [0., 1.]]}], "bool": True, "none": None}


def index_value(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def failure(path):
    return json.loads(next(path.parent.glob(path.name + ".failure-*.json")).read_bytes())


def test_byte_exact_json_roundtrip_and_unchanged_row(tmp_path):
    path = tmp_path / "batch-000000-attempt-000000.json.gz"
    row = sample()
    row["multichunk"] = "汉字 and ascii " * 30000
    original = copy.deepcopy(row)
    writer = journal.EpisodeJournal(path)
    writer.append(row)
    assert row == original
    index = index_value(path)
    assert index["format"] == journal.FORMAT
    entry = index["episodes"][0]
    assert entry["codec"] == "xz" and entry["path"].endswith(".json.xz")
    raw = journal.decode_bytes(tmp_path / entry["path"], expected_sha256=entry["compressed_sha256"])
    assert raw == json.dumps([row], ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == entry["decoded_sha256"]
    assert len(raw) == entry["decoded_bytes"]
    assert journal.read_batch(path) == [row]
    assert not list(tmp_path.glob("*.partial"))


def test_legacy_monolithic_v1_and_mixed_v2_without_modifying_old(tmp_path):
    legacy = tmp_path / "legacy-episode.json.gz"
    row = sample(10)
    put_gzip(legacy, [row])
    original = legacy.read_bytes()
    old_entry = {"path": legacy.name, "seed": 10, "sha256": hashlib.sha256(original).hexdigest()}
    v1 = tmp_path / "old-index.json.gz"
    put_gzip(v1, {"format": journal.LEGACY_FORMAT, "episodes": [old_entry]})
    old_index_bytes = v1.read_bytes()
    assert journal.read_batch(legacy) == journal.read_batch(v1) == [row]
    path = tmp_path / "new-index.json.gz"
    writer = journal.EpisodeJournal(path, prior_entries=[old_entry])
    writer.append(sample(11))
    assert [e["codec"] for e in index_value(path)["episodes"]] == ["gzip", "xz"]
    assert journal.read_batch(path) == [row, sample(11)]
    assert legacy.read_bytes() == original and v1.read_bytes() == old_index_bytes
    # Frozen v1 reader must clearly reject a v2 index, not treat XZ as gzip.
    root = HERE.parents[2]
    spec = importlib.util.spec_from_file_location("legacy_journal", root / "src/q4_rl/training_journal.py")
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)
    with pytest.raises(ValueError, match="unsupported"):
        legacy_module.read_batch(path)


@pytest.mark.parametrize("field,bad", [
    ("compressed_sha256", "0" * 64), ("decoded_sha256", "0" * 64),
    ("compressed_bytes", 1), ("decoded_bytes", 1), ("codec", "gzip"), ("seed", 999),
    ("path", "../outside.json.xz"), ("path", "C:\\outside.json.xz"),
    ("path", "sub/file.json.xz"), ("path", "/outside.json.xz"),
])
def test_metadata_hash_and_path_tampering_rejected(tmp_path, field, bad):
    original = tmp_path / "original.json.gz"
    journal.EpisodeJournal(original).append(sample())
    index = index_value(original)
    index["episodes"][0][field] = bad
    altered = tmp_path / "altered.json.gz"
    put_gzip(altered, index)
    with pytest.raises((ValueError, journal.OutputLimit)):
        journal.read_batch(altered)


def test_duplicate_and_symlink_rejected(tmp_path):
    original = tmp_path / "original.json.gz"
    journal.EpisodeJournal(original).append(sample())
    index = index_value(original)
    index["episodes"] *= 2
    altered = tmp_path / "duplicate.json.gz"
    put_gzip(altered, index)
    with pytest.raises(ValueError, match="duplicate"):
        journal.read_batch(altered)
    link = tmp_path / "link.json.gz"
    try:
        link.symlink_to(original)
    except OSError:
        pytest.skip("host cannot create symlinks; basename/escape tests still run")
    with pytest.raises(ValueError, match="symlink"):
        journal.read_batch(link)


def test_output_cap_preserves_previous_index_and_episode(tmp_path):
    path = tmp_path / "attempt.json.gz"
    writer = journal.EpisodeJournal(path)
    writer.append(sample())
    old_index = path.read_bytes()
    old_episode = tmp_path / index_value(path)["episodes"][0]["path"]
    old_payload = old_episode.read_bytes()
    writer.episode_byte_cap = 5
    with pytest.raises(journal.OutputLimit):
        writer.append(sample(2))
    assert path.read_bytes() == old_index and old_episode.read_bytes() == old_payload
    partial = next(tmp_path.glob("*.json.xz.partial"))
    assert partial.stat().st_size == 5
    assert journal.read_batch(path) == [sample()]
    receipt = failure(path)
    assert receipt["error_type"] == "OutputLimit"
    assert receipt["eligible_for_training_update"] is False
    assert receipt["original_row_completeness_claimed"] is False
    with pytest.raises(RuntimeError, match="failed attempt"):
        writer.append(sample(3))
    with pytest.raises(FileExistsError):
        journal.EpisodeJournal(path)


def test_administrative_interrupt_retains_partial_and_no_index(tmp_path):
    path = tmp_path / "interrupt.json.gz"
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        return calls == 2
    writer = journal.EpisodeJournal(path, stop_check=stop)
    row = sample()
    row["large"] = "abc" * journal.CHUNK
    with pytest.raises(journal.AdministrativeStop):
        writer.append(row)
    assert not path.exists()
    assert next(tmp_path.glob("*.json.xz.partial")).stat().st_size > 0
    assert failure(path)["stage"] == "compress_episode"


def test_index_publish_interrupt_keeps_old_index_and_complete_orphan(tmp_path, monkeypatch):
    path = tmp_path / "attempt.json.gz"
    writer = journal.EpisodeJournal(path)
    writer.append(sample())
    original = path.read_bytes()
    def interrupt(partial):
        # Before atomic publication, the complete next index references files
        # that already exist and can be hash-validated.
        decoded = json.loads(gzip.decompress(partial.read_bytes()))
        assert len(decoded["episodes"]) == 2
        assert path.read_bytes() == original
        raise KeyboardInterrupt()
    monkeypatch.setattr(writer, "_publish_index", interrupt)
    with pytest.raises(KeyboardInterrupt):
        writer.append(sample(2))
    assert path.read_bytes() == original and len(writer.index["episodes"]) == 1
    orphan = next(tmp_path.glob("*-episode-0001.json.xz"))
    assert json.loads(journal.decode_bytes(orphan)) == [sample(2)]
    assert failure(path)["stage"] == "publish_index"
    assert path.with_name(path.name + ".partial").is_file()


def test_atomic_replace_only_after_complete_episode_and_small_index(tmp_path, monkeypatch):
    path = tmp_path / "attempt.json.gz"
    writer = journal.EpisodeJournal(path)
    writer.append(sample())
    original = path.read_bytes()
    replace = journal.os.replace
    seen = []
    def checked_replace(partial, destination):
        assert destination == path and path.read_bytes() == original
        index = json.loads(gzip.decompress(partial.read_bytes()))
        for entry in index["episodes"]:
            raw = journal.decode_bytes(tmp_path / entry["path"], expected_sha256=entry["compressed_sha256"])
            assert hashlib.sha256(raw).hexdigest() == entry["decoded_sha256"]
        seen.append(True)
        return replace(partial, destination)
    monkeypatch.setattr(journal.os, "replace", checked_replace)
    writer.append(sample(2))
    assert seen == [True] and journal.read_batch(path) == [sample(), sample(2)]


def test_no_clobber_atomic_episode_publication(tmp_path, monkeypatch):
    path = tmp_path / "attempt.json.gz"
    writer = journal.EpisodeJournal(path)
    link = journal.os.link
    def race(partial, destination):
        if destination.name.endswith(".json.xz"):
            destination.write_bytes(b"existing competing immutable evidence")
        return link(partial, destination)
    monkeypatch.setattr(journal.os, "link", race)
    with pytest.raises(FileExistsError):
        writer.append(sample())
    assert next(tmp_path.glob("*.json.xz")).read_bytes() == b"existing competing immutable evidence"
    assert next(tmp_path.glob("*.json.xz.partial")).exists() and not path.exists()


def test_bounded_decode_truncation_trailing_data_and_index_cap(tmp_path):
    path = tmp_path / "attempt.json.gz"
    row = sample()
    row["large"] = "x" * 200000
    writer = journal.EpisodeJournal(path)
    writer.append(row)
    episode = tmp_path / index_value(path)["episodes"][0]["path"]
    with pytest.raises(journal.OutputLimit, match="decoded"):
        journal.decode_bytes(episode, max_decoded_bytes=100)
    with pytest.raises(journal.OutputLimit, match="compressed"):
        journal.decode_bytes(episode, max_compressed_bytes=2)
    truncated = tmp_path / "truncated.json.xz"
    truncated.write_bytes(episode.read_bytes()[:-3])
    with pytest.raises(ValueError, match="truncated"):
        journal.decode_bytes(truncated)
    trailing = tmp_path / "trailing.json.xz"
    trailing.write_bytes(episode.read_bytes() + b"extra")
    with pytest.raises(ValueError, match="trailing"):
        journal.decode_bytes(trailing)
    other = tmp_path / "index-capped.json.gz"
    with pytest.raises(journal.OutputLimit):
        journal.EpisodeJournal(other, index_byte_cap=2).append(sample())
    assert not other.exists()
    assert other.with_name(other.name + ".partial").stat().st_size == 2
    assert failure(other)["stage"] == "compress_index"


def test_serialization_error_does_not_change_input_or_old_data(tmp_path):
    path = tmp_path / "attempt.json.gz"
    writer = journal.EpisodeJournal(path)
    writer.append(sample())
    original = path.read_bytes()
    with pytest.raises(ValueError):
        writer.append({"seed": 2, "invalid": float("nan")})
    assert path.read_bytes() == original and journal.read_batch(path) == [sample()]
    assert failure(path)["stage"] == "serialize"


def test_compressor_error_preserves_partial_without_leaking_error_message(tmp_path, monkeypatch):
    path = tmp_path / "compressor-error.json.gz"
    original = journal.lzma.LZMACompressor
    class BrokenCompressor:
        def __init__(self, **kwargs):
            self.compressor = original(**kwargs)
            self.calls = 0
        def compress(self, block):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("do-not-copy-exception-content")
            return self.compressor.compress(block)
        def flush(self):
            return self.compressor.flush()
    monkeypatch.setattr(journal.lzma, "LZMACompressor", BrokenCompressor)
    row = sample()
    row["large"] = "x" * (journal.CHUNK * 3)
    with pytest.raises(RuntimeError, match="do-not-copy"):
        journal.EpisodeJournal(path).append(row)
    receipt = failure(path)
    assert receipt["stage"] == "compress_episode"
    assert receipt["error_type"] == "RuntimeError"
    assert "do-not-copy" not in json.dumps(receipt)
    assert next(tmp_path.glob("*.json.xz.partial")).stat().st_size > 0
    assert not path.exists()
