"""Offline XZ3 episode / gzip6 index prototype; not a production resume driver.

The append/read_batch object contract is retained with an explicit v2 index.
Compression is owned here (not delegated to the old unbounded writer callback).
Only a fresh attempt may be created; existing evidence is never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import uuid
import zlib

LEGACY_FORMAT = "q4-training-episode-index-v1"
FORMAT = "q4-training-episode-index-v2"
CHUNK = 64 * 1024
DEFAULT_EPISODE_CAP = 32 * 1024 * 1024
DEFAULT_DECODE_CAP = 128 * 1024 * 1024
INDEX_CAP = 1024 * 1024


class OutputLimit(RuntimeError):
    pass


class AdministrativeStop(RuntimeError):
    pass


def canonical_bytes(value):
    # Exact parameters used by the frozen train._write_batch, including default
    # separators and insertion order. This intentionally bounds output, not the
    # already materialized worker row or json.dumps allocation.
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _positive_cap(value):
    if type(value) is not int or value < 0:
        raise ValueError("byte cap must be a nonnegative integer")
    return value


def _basename(name):
    if (not isinstance(name, str) or not name or name in (".", "..")
            or any(c in name for c in "/\\:") or any(ord(c) < 32 for c in name)
            or Path(name).name != name):
        raise ValueError("unsafe evidence basename")
    return name


def _path(directory, name):
    path = directory / _basename(name)
    if path.is_symlink() or path.resolve().parent != directory:
        raise ValueError("evidence path escapes directory or is a symlink")
    return path


def _directory(path):
    path = Path(path).absolute()
    if path.parent.resolve() != path.parent or not path.parent.is_dir() or path.is_symlink():
        raise ValueError("use an existing nonsymlink attempt directory")
    _basename(path.name)
    return path


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(CHUNK), b""):
            value.update(part)
    return value.hexdigest()


def _sync_directory(directory):
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _publish_new(partial, destination):
    # Hard-link creation is atomic and fails if destination already exists on
    # Windows/Linux. If unsupported, fail closed; never use overwriting rename.
    os.link(partial, destination)
    partial.unlink()  # Only this newly written, successfully published temp.
    _sync_directory(destination.parent)


def _write_compressed(partial, payload, *, codec, byte_cap, stop_check):
    byte_cap = _positive_cap(byte_cap)
    compressor = (lzma.LZMACompressor(format=lzma.FORMAT_XZ,
                    check=lzma.CHECK_CRC64, preset=3) if codec == "xz"
                  else zlib.compressobj(level=6, wbits=31))
    written = 0
    with partial.open("xb") as stream:
        def emit(part):
            nonlocal written
            room = byte_cap - written
            if len(part) > room:
                stream.write(part[:room])
                written += room
                stream.flush()
                os.fsync(stream.fileno())
                raise OutputLimit("compressed output cap reached")
            stream.write(part)
            written += len(part)

        try:
            for offset in range(0, len(payload), CHUNK):
                if stop_check is not None and stop_check():
                    raise AdministrativeStop("compression interrupted")
                emit(compressor.compress(payload[offset:offset + CHUNK]))
            if stop_check is not None and stop_check():
                raise AdministrativeStop("compression interrupted")
            emit(compressor.flush())
        finally:
            stream.flush()
            os.fsync(stream.fileno())
    return written


def decode_bytes(path, *, expected_sha256=None, max_compressed_bytes=DEFAULT_EPISODE_CAP,
                 max_decoded_bytes=DEFAULT_DECODE_CAP):
    """Stream-decode a single gzip/XZ stream, with input/output and XZ memory caps.

    Return bytes only after complete stream + optional compressed-byte SHA pass.
    Concatenated streams/trailing bytes are rejected; original writers use one.
    """
    path = _directory(path)
    if not path.is_file():
        raise ValueError("evidence must be a regular file")
    max_compressed_bytes = _positive_cap(max_compressed_bytes)
    max_decoded_bytes = _positive_cap(max_decoded_bytes)
    if expected_sha256 is not None and (not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
        raise ValueError("invalid evidence SHA256")
    if path.name.endswith(".json.xz"):
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=64 * 1024 * 1024)
        is_xz = True
    elif path.name.endswith(".json.gz"):
        decoder = zlib.decompressobj(wbits=31)
        is_xz = False
    else:
        raise ValueError("unsupported evidence codec")
    digest, output, compressed = hashlib.sha256(), bytearray(), 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(min(CHUNK, max_compressed_bytes - compressed + 1))
            if not block:
                break
            compressed += len(block)
            if compressed > max_compressed_bytes:
                raise OutputLimit("compressed input cap reached")
            digest.update(block)
            if decoder.eof:
                raise ValueError("trailing compressed data")
            pending = block
            while True:
                allowance = min(CHUNK, max_decoded_bytes - len(output) + 1)
                plain = decoder.decompress(pending, max_length=allowance)
                output.extend(plain)
                if len(output) > max_decoded_bytes:
                    raise OutputLimit("decoded output cap reached")
                if decoder.eof:
                    if decoder.unused_data:
                        raise ValueError("trailing or concatenated stream")
                    break
                if is_xz:
                    if decoder.needs_input:
                        break
                    pending = b""
                else:
                    pending = decoder.unconsumed_tail
                    if not pending:
                        break
    if not decoder.eof:
        raise ValueError("truncated compressed evidence")
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError("evidence hash mismatch")
    return bytes(output)


def _read_entry(directory, entry, *, version, max_compressed_bytes, max_decoded_bytes):
    legacy = version == LEGACY_FORMAT
    fields = ({"path", "seed", "sha256"} if legacy else
        {"path", "seed", "codec", "compressed_sha256", "compressed_bytes", "decoded_sha256", "decoded_bytes"})
    if not isinstance(entry, dict) or set(entry) != fields or type(entry["seed"]) is not int:
        raise ValueError("invalid episode entry")
    path = _path(directory, entry["path"])
    if not legacy:
        codec = "xz" if path.name.endswith(".json.xz") else "gzip" if path.name.endswith(".json.gz") else None
        if entry["codec"] != codec or codec is None:
            raise ValueError("codec metadata mismatch")
        for key in ("compressed_bytes", "decoded_bytes"):
            _positive_cap(entry[key])
        if path.stat().st_size != entry["compressed_bytes"] or entry["decoded_bytes"] > max_decoded_bytes:
            raise ValueError("episode byte count mismatch or cap exceeded")
    raw = decode_bytes(path, expected_sha256=entry["sha256"] if legacy else entry["compressed_sha256"],
        max_compressed_bytes=max_compressed_bytes, max_decoded_bytes=max_decoded_bytes)
    if not legacy and (len(raw) != entry["decoded_bytes"] or hashlib.sha256(raw).hexdigest() != entry["decoded_sha256"]):
        raise ValueError("decoded evidence hash/size mismatch")
    rows = json.loads(raw)
    if (not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict)
            or type(rows[0].get("seed")) is not int or rows[0]["seed"] != entry["seed"]):
        raise ValueError("episode identity mismatch")
    return rows[0], raw


def _new_entry(path, seed, raw):
    return {"path": path.name, "seed": seed,
        "codec": "xz" if path.name.endswith(".json.xz") else "gzip",
        "compressed_sha256": _digest(path), "compressed_bytes": path.stat().st_size,
        "decoded_sha256": hashlib.sha256(raw).hexdigest(), "decoded_bytes": len(raw)}


def _entry_rows(directory, entries, *, version, max_compressed_bytes, max_decoded_bytes):
    if not isinstance(entries, list):
        raise ValueError("invalid episode index")
    result, seen, total = [], set(), 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid episode entry")
        name = _basename(entry["path"])
        if name in seen:
            raise ValueError("duplicate episode path")
        row, raw = _read_entry(directory, entry, version=version,
            max_compressed_bytes=max_compressed_bytes, max_decoded_bytes=max_decoded_bytes - total)
        total += len(raw)
        result.append(row)
        seen.add(name)
    return result


def read_batch(path, *, expected_sha256=None, max_compressed_bytes=DEFAULT_EPISODE_CAP,
               max_decoded_bytes=DEFAULT_DECODE_CAP):
    path = _directory(path)
    value = json.loads(decode_bytes(path, expected_sha256=expected_sha256,
        max_compressed_bytes=max_compressed_bytes, max_decoded_bytes=max_decoded_bytes))
    if isinstance(value, list):
        return value  # Historical monolithic gzip-list batch.
    if not isinstance(value, dict) or set(value) != {"format", "episodes"} or value["format"] not in {FORMAT, LEGACY_FORMAT}:
        raise ValueError("unsupported episode index")
    return _entry_rows(path.parent, value["episodes"], version=value["format"], max_compressed_bytes=max_compressed_bytes,
                       max_decoded_bytes=max_decoded_bytes)


class EpisodeJournal:
    def __init__(self, path, *, episode_byte_cap=DEFAULT_EPISODE_CAP,
                 index_byte_cap=INDEX_CAP, stop_check=None, prior_entries=()):
        self.path = _directory(path)
        if not self.path.name.endswith(".json.gz") or self.path.exists():
            raise FileExistsError("new attempt requires an absent .json.gz index")
        self.episode_byte_cap = _positive_cap(episode_byte_cap)
        self.index_byte_cap = _positive_cap(index_byte_cap)
        self.stop_check, self._failed, self._index_sha = stop_check, False, None
        entries, seen = [], set()
        # A fresh index may reference immutable legacy files in this directory.
        for entry in prior_entries:
            version = LEGACY_FORMAT if "sha256" in entry else FORMAT
            row, raw = _read_entry(self.path.parent, entry, version=version,
                max_compressed_bytes=DEFAULT_EPISODE_CAP, max_decoded_bytes=DEFAULT_DECODE_CAP)
            if entry["path"] in seen:
                raise ValueError("duplicate prior episode path")
            entries.append(_new_entry(_path(self.path.parent, entry["path"]), row["seed"], raw))
            seen.add(entry["path"])
        self.index = {"format": FORMAT, "episodes": entries}
        self.owner = self.path.with_name(self.path.name + ".xz-owner")
        with self.owner.open("xb") as stream:
            stream.write(b"offline-xz-prototype-v1\n")
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(self.path.parent)

    def _publish_index(self, partial):
        if self._index_sha is None:
            _publish_new(partial, self.path)
        else:
            if self.path.is_symlink() or _digest(self.path) != self._index_sha:
                raise ValueError("owned index changed externally")
            os.replace(partial, self.path)  # Atomic replacement of our own index.
            _sync_directory(self.path.parent)

    def _failure(self, stage, error, artifacts):
        # Keep messages/data/absolute paths out of the administrative receipt.
        record = {"kind": "administrative_storage_partial", "stage": stage,
            "error_type": type(error).__name__, "last_acknowledged_entries": len(self.index["episodes"]),
            "previous_index_sha256": self._index_sha, "original_row_completeness_claimed": False,
            "observed_index_sha256": _digest(self.path) if self.path.is_file() and not self.path.is_symlink() else None,
            "eligible_for_training_update": False, "artifacts": []}
        for path in artifacts:
            if path.is_file() and not path.is_symlink():
                record["artifacts"].append({"path": path.name, "bytes": path.stat().st_size,
                                            "sha256": _digest(path)})
        name = self.path.name + ".failure-" + uuid.uuid4().hex + ".json"
        with _path(self.path.parent, name).open("xb") as stream:
            stream.write(canonical_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, row):
        if self._failed:
            raise RuntimeError("failed attempt retained; do not append or reuse it")
        number = len(self.index["episodes"])
        name = self.path.name.removesuffix(".json.gz") + f"-episode-{number:04d}.json.xz"
        episode = _path(self.path.parent, name)
        partial = episode.with_name(episode.name + ".partial")
        index_partial = self.path.with_name(self.path.name + ".partial")
        stage = "serialize"
        try:
            if not isinstance(row, dict) or type(row.get("seed")) is not int:
                raise ValueError("episode row requires seed identity")
            if episode.exists() or partial.exists() or index_partial.exists():
                raise FileExistsError("immutable episode or partial already exists")
            payload = canonical_bytes([row])
            stage = "compress_episode"
            _write_compressed(partial, payload, codec="xz", byte_cap=self.episode_byte_cap,
                              stop_check=self.stop_check)
            stage = "publish_episode"
            _publish_new(partial, episode)
            entry = _new_entry(episode, row["seed"], payload)
            next_index = {"format": FORMAT, "episodes": [*self.index["episodes"], entry]}
            stage = "compress_index"
            _write_compressed(index_partial, canonical_bytes(next_index), codec="gz",
                              byte_cap=self.index_byte_cap, stop_check=self.stop_check)
            stage = "publish_index"
            self._publish_index(index_partial)
            self._index_sha = _digest(self.path)
            self.index = next_index  # In-memory cursor changes only after commit.
        except BaseException as error:
            self._failed = True
            try:
                self._failure(stage, error, [partial, episode, index_partial])
            except BaseException:
                # ENOSPC or hard interruption can prevent this small receipt;
                # exclusive partial/orphan names still retain physical evidence.
                pass
            raise
