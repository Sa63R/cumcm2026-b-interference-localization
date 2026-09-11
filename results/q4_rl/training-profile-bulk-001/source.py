"""Single-CPU offline microbenchmark on one explicitly supplied saved batch.

Writes three compression variants with identical decoded JSON. Optional profiling
runs exactly one original BC update on disposable in-memory model/optimizer state;
it never collects a scenario or writes a model/checkpoint.
"""
from __future__ import annotations

import argparse
import cProfile
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def one_cpu():
    if hasattr(os, "sched_getaffinity"):
        selected = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {selected})
        return {"method": "sched_setaffinity", "logical_cpus": 1}
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        handle = kernel.GetCurrentProcess()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process), ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        selected = process.value & -process.value
        if not selected or not kernel.SetProcessAffinityMask(handle, selected):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"method": "SetProcessAffinityMask", "logical_cpus": 1}
    raise RuntimeError("Cannot enforce requested single-CPU affinity")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def timed(function):
    wall, cpu = time.perf_counter(), time.process_time()
    value = function()
    return value, {"wall_s": time.perf_counter() - wall, "cpu_s": time.process_time() - cpu}


def frame_name(frame):
    file, line, function = frame
    path = Path(file)
    try:
        file = path.resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        file = path.name
    return {"file": file, "line": line, "function": function}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--profile-imitation", action="store_true")
    parser.add_argument("--mode", choices=("stream", "bulk", "all"), default="stream",
                        help="stream matches json.dump; bulk tests one C-encoder JSON bytes write")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new benchmark directory")
    if args.profile_imitation and not args.checkpoint:
        parser.error("Explicit checkpoint required for disposable BC profiling")
    affinity = one_cpu()
    source = args.batch.read_bytes()
    decoded, read_time = timed(lambda: gzip.decompress(source))
    episodes, parse_time = timed(lambda: json.loads(decoded))
    if not isinstance(episodes, list):
        raise ValueError("Expected complete training batch list")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "source.py").write_bytes(Path(__file__).read_bytes())
    report = {"schema": "q4-training-offline-microbenchmark-v1", "affinity": affinity,
              "source": {"name": args.batch.name, "sha256": sha(source), "bytes": len(source),
                         "decompressed_bytes": len(decoded), "episodes": len(episodes)},
              "script_sha256": sha(Path(__file__).read_bytes()), "gzip_read": read_time, "json_parse": parse_time,
              "compression": [], "scope": "one local fixed-order microbenchmark; no new scene or retained model update; not strategy evidence"}
    # Match train._write_raw exactly apart from the explicitly varied level.
    for level in ((1, 6, 9) if args.mode in ("stream", "all") else ()):
        path = args.output_dir / f"compression-level-{level}.json.gz"
        def write_json():
            with gzip.open(path, "wt", encoding="utf-8", compresslevel=level) as stream:
                json.dump(episodes, stream, ensure_ascii=False, allow_nan=False)
        _, elapsed = timed(write_json)
        compressed = path.read_bytes()
        def verify():
            unpacked = gzip.decompress(compressed)
            if json.loads(unpacked) != episodes:
                raise ValueError("Compression changed the saved batch object")
            return sha(unpacked)
        decoded_hash, verify_time = timed(verify)
        result = {"level": level, **elapsed, "compressed_bytes": len(compressed),
                  "decoded_sha256": decoded_hash, "decoded_object_equal": True,
                  "verify": verify_time}
        report["compression"].append(result)
        print(json.dumps({"compression_complete": result}), flush=True)
    if report["compression"] and len({item["decoded_sha256"] for item in report["compression"]}) != 1:
        raise ValueError("Compression levels did not receive identical JSON bytes")
    if args.mode in ("bulk", "all"):
        encoded, encode_time = timed(lambda: json.dumps(episodes, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        if encoded != decoded:
            raise ValueError("Bulk encoder differs from the actual saved JSON bytes")
        report["bulk_serialization"] = {"encode": encode_time, "bytes": len(encoded),
            "decoded_sha256": sha(encoded), "byte_identical_to_original": True,
            "memory_tradeoff": "holds one additional full JSON string then UTF-8 byte buffer before writing", "compression": []}
        for level in (6, 9):
            path = args.output_dir / f"bulk-level-{level}.json.gz"
            def bulk_write():
                with gzip.open(path, "wb", compresslevel=level) as stream:
                    stream.write(encoded)
            _, elapsed = timed(bulk_write)
            compressed = path.read_bytes()
            if gzip.decompress(compressed) != decoded or json.loads(gzip.decompress(compressed)) != episodes:
                raise ValueError("Bulk write changed the saved batch object")
            result = {"level": level, **elapsed, "compressed_bytes": len(compressed),
                "encode_plus_write_wall_s": encode_time["wall_s"] + elapsed["wall_s"],
                "encode_plus_write_cpu_s": encode_time["cpu_s"] + elapsed["cpu_s"],
                "decoded_object_equal": True, "byte_identical_to_original": True}
            report["bulk_serialization"]["compression"].append(result)
            print(json.dumps({"bulk_complete": result}), flush=True)
    if args.profile_imitation:
        from q4_rl.train import imitation_update, restore_checkpoint
        model, optimizer, _, _ = restore_checkpoint(args.checkpoint)
        records = [record for episode in episodes for record in episode.get("records", [])]
        if any("log_prob" in record for record in records):
            raise ValueError("Explicit BC benchmark requires saved heuristic-label records")
        checkpoint_before = sha(args.checkpoint.read_bytes())
        profiler = cProfile.Profile()
        def disposable_update():
            profiler.enable()
            try:
                return imitation_update(model, optimizer, records, epochs=1, minibatch_size=128)
            finally:
                profiler.disable()
        result, elapsed = timed(disposable_update)
        stats = pstats.Stats(profiler)
        entries = []
        for frame, data in stats.stats.items():
            primitive, total, exclusive, cumulative, _ = data
            entries.append({**frame_name(frame), "primitive_calls": primitive, "total_calls": total,
                            "exclusive_s": exclusive, "cumulative_s": cumulative})
        entries.sort(key=lambda row: row["cumulative_s"], reverse=True)
        if sha(args.checkpoint.read_bytes()) != checkpoint_before:
            raise ValueError("Input checkpoint changed during the disposable benchmark")
        report["imitation_profile"] = {**elapsed, "settings": {"epochs": 1, "minibatch_size": 128},
            "result": result, "checkpoint": {"name": args.checkpoint.name, "sha256": checkpoint_before},
            "profile_total_exclusive_s": stats.total_tt, "functions_by_cumulative": entries[:80],
            "scope": "original update including packing, forward/backward and optimizer on disposable state; cumulative function times overlap"}
        print(json.dumps({"imitation_profile_complete": {**elapsed, "result": result}}), flush=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": "report.json", "compression_verified": len(report["compression"])}))


if __name__ == "__main__":
    main()
