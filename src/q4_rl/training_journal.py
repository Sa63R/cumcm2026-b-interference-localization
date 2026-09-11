"""Linear-size immutable episode evidence with a small atomic batch index.

Only storage changes: decoded episode objects are the original worker results.
The reader also accepts historical gzip JSON-list batches.
"""
import gzip
import hashlib
import json
from pathlib import Path


FORMAT = "q4-training-episode-index-v1"


class EpisodeJournal:
    def __init__(self, path, writer):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError("refusing to replace a recorded training attempt")
        self.writer = writer
        self.index = {"format": FORMAT, "episodes": []}

    def append(self, row):
        number = len(self.index["episodes"])
        name = self.path.name.removesuffix(".json.gz") + f"-episode-{number:04d}.json.gz"
        episode = self.path.with_name(name)
        if episode.exists():
            raise FileExistsError("refusing to replace immutable episode evidence")
        # Write once, then publish its hash. A crash between these operations
        # leaves an identifiable episode file beside the last valid index.
        self.writer(episode, [row])
        self.index["episodes"].append({"path": name, "seed": row["seed"],
            "sha256": hashlib.sha256(episode.read_bytes()).hexdigest()})
        self.writer(self.path, self.index)


def read_batch(path):
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if isinstance(value, list):
        return value
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise ValueError("unsupported training evidence format")
    result, seen = [], set()
    for entry in value["episodes"]:
        name = entry["path"]
        if not isinstance(name, str) or Path(name).name != name or name in seen:
            raise ValueError("invalid or duplicate episode evidence path")
        episode = path.with_name(name)
        if episode.resolve().parent != path.resolve().parent:
            raise ValueError("episode evidence escapes its attempt directory")
        if hashlib.sha256(episode.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("episode evidence hash mismatch")
        with gzip.open(episode, "rt", encoding="utf-8") as stream:
            rows = json.load(stream)
        if not isinstance(rows, list) or len(rows) != 1 or rows[0]["seed"] != entry["seed"]:
            raise ValueError("episode evidence identity mismatch")
        result.extend(rows)
        seen.add(name)
    return result
