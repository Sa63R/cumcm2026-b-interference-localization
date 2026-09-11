import gzip
import json

import pytest

from q4_rl.train import _write_batch
from q4_rl.training_journal import EpisodeJournal, read_batch


def test_journal_preserves_worker_objects_and_legacy_batches(tmp_path):
    rows = [{"seed": 8006000+i, "records": [{"cost_s": 12.345, "features": [i, .125]}]}
            for i in range(3)]
    path = tmp_path/"batch-000000-attempt-000000.json.gz"
    writes = []
    def writer(target, value):
        writes.append((target.name, isinstance(value, list)))
        _write_batch(target, value)
    journal = EpisodeJournal(path, writer)
    for i, row in enumerate(rows):
        journal.append(row)
        assert read_batch(path) == rows[:i+1]
    assert len([name for name, episode in writes if episode]) == len(rows)
    assert len(set(name for name, episode in writes if episode)) == len(rows)
    legacy = tmp_path/"legacy.json.gz"
    _write_batch(legacy, rows)
    assert read_batch(legacy) == read_batch(path)
    with pytest.raises(FileExistsError):
        EpisodeJournal(path, writer)


def test_partial_index_does_not_rewrite_prior_episode_and_detects_corruption(tmp_path):
    path = tmp_path/"attempt.json.gz"
    journal = EpisodeJournal(path, _write_batch)
    journal.append({"seed": 8006000, "records": []})
    episode = tmp_path/journal.index["episodes"][0]["path"]
    original = episode.read_bytes()
    journal.append({"seed": 8006001, "records": []})
    assert episode.read_bytes() == original
    episode.write_bytes(original[:-3]+b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_batch(path)


def test_reader_rejects_path_escape(tmp_path):
    path = tmp_path/"attempt.json.gz"
    _write_batch(path, {"format": "q4-training-episode-index-v1", "episodes": [
        {"path": "../outside.json.gz", "seed": 8006000, "sha256": "unused"}]})
    with pytest.raises(ValueError, match="path"):
        read_batch(path)
