"""Four fixed existing TRAIN samples; compression only, no rollouts/network.

Run with a local Python containing the standard-library lzma module. Each codec
measurement uses a fresh, single-core process. Original gzip files are read-only.
"""
import argparse
import ctypes
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE = REPO.parent / "q4-rl-negative-memory/results/q4_rl/server-memory-v4-training-001"
MANIFEST_SHA = "2c5e8e8adeaa828a2f80b4c802f7221a3e64995fc47bc7f877dfa3341eed55fc"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def single_cpu():
    if os.name == "nt":
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        proc, system = ctypes.c_size_t(), ctypes.c_size_t()
        handle = k.GetCurrentProcess()
        assert k.GetProcessAffinityMask(handle, ctypes.byref(proc), ctypes.byref(system))
        assert k.SetProcessAffinityMask(handle, proc.value & -proc.value)
    else:
        allowed = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(allowed)})


def memory():
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize",
                "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = ctypes.c_void_p
        ps = ctypes.WinDLL("psapi", use_last_error=True)
        ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        assert ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return {"rss_bytes": counters.WorkingSetSize, "lifetime_peak_rss_bytes": counters.PeakWorkingSetSize}
    import resource
    return {"rss_bytes": None, "lifetime_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def worker(job, phase, codec):
    single_cpu()
    manifest_bytes = (SOURCE / "OBJECT_READBACK.json").read_bytes()
    assert sha(manifest_bytes) == MANIFEST_SHA
    objects = {x["name"]: x for x in json.loads(manifest_bytes)["objects"]}
    batch = 0 if phase == "BC" else 16
    seed = 8012000 + batch * 16
    prefix = f"{job}/training/"
    stem = f"batch-{batch:06d}-attempt-{batch:06d}"
    name = prefix + stem + "-episode-0000.json.gz"
    index_bytes = (SOURCE / (prefix+stem+".json.gz")).read_bytes()
    assert sha(index_bytes) == objects[prefix+stem+".json.gz"]["sha256"]
    entry = json.loads(gzip.decompress(index_bytes))["episodes"][0]
    assert entry["seed"] == seed and entry["sha256"] == objects[name]["sha256"]
    source = (SOURCE / name).read_bytes()
    assert sha(source) == entry["sha256"]
    payload = gzip.decompress(source)
    source_size, source_hash = len(source), sha(source)
    del source
    baseline_memory = memory()
    wall, cpu = time.perf_counter(), time.process_time()
    packed = (gzip.compress(payload, compresslevel=6, mtime=0) if codec == "gzip6" else
              lzma.compress(payload, format=lzma.FORMAT_XZ, preset=3))
    compress_wall, compress_cpu = time.perf_counter()-wall, time.process_time()-cpu
    compressed_memory = memory()
    wall, cpu = time.perf_counter(), time.process_time()
    restored = gzip.decompress(packed) if codec == "gzip6" else lzma.decompress(packed)
    decompress_wall, decompress_cpu = time.perf_counter()-wall, time.process_time()-cpu
    roundtrip_memory = memory()
    assert restored == payload
    return {"job": job, "phase": phase, "seed": seed, "codec": codec,
            "source_relative_to_readback": name, "source_sha256": source_hash,
            "existing_gzip_bytes": source_size, "original_json_bytes": len(payload),
            "original_json_sha256": sha(payload), "output_bytes": len(packed),
            "output_sha256": sha(packed), "size_ratio_to_existing_gzip": len(packed)/source_size,
            "size_ratio_to_original_json": len(packed)/len(payload),
            "compress_cpu_s": compress_cpu, "compress_wall_s": compress_wall,
            "decompress_cpu_s": decompress_cpu, "decompress_wall_s": decompress_wall,
            "cpu_affinity_count": 1, "json_bytes_strictly_equal_after_roundtrip": True,
            "baseline_memory": baseline_memory, "after_compression_memory": compressed_memory,
            "after_roundtrip_memory": roundtrip_memory}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=3, metavar=("JOB", "PHASE", "CODEC"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker)), flush=True)
        return
    samples = [(job, phase) for phase in ["BC", "PPO"] for job in ["g1_h64", "g3_h64"]]
    results = []
    for job, phase in samples:
        for codec in ["gzip6", "xz3"]:
            done = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                   "--worker", job, phase, codec], check=True,
                                  capture_output=True, text=True, timeout=60)
            results.append(json.loads(done.stdout))
    document = {
        "schema": "q4-lossless-episode-codec-probe-v1",
        "scope": "Compression-only probe on four fixed existing synthetic TRAIN episodes; no training, scenario execution, network or policy-performance comparison.",
        "selection": "First episode of batch0 (BC) and batch16 (first PPO after256 BC), in both G1/G3 h64 jobs; fixed before inspecting compression results.",
        "source_readback_workspace_relative": "q4-rl-negative-memory/results/q4_rl/server-memory-v4-training-001",
        "source_OBJECT_READBACK_sha256": MANIFEST_SHA,
        "script_sha256": sha(Path(__file__).read_bytes()), "python_version": sys.version.split()[0],
        "timing_scope": "One fresh single-core process per sample/codec, one compression and one decompression; timers exclude initial reading/source decompression/JSON encoding. Descriptive single-pass timings, not stable performance estimates.",
        "memory_scope": "Operating-system process lifetime peak working set and snapshots. Includes Python plus input and restored JSON buffers; not isolated codec allocation or server worker RSS. No dense JSON deserialization in this probe.",
        "gzip_header_note": "gzip.compress baseline has a53-byte smaller header than the existing writer's filename-bearing gzip; ratios use existing on-disk gzip size, not original JSON size.",
        "results": results,
    }
    target = OUT / "measurements.json"
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"file": target.relative_to(REPO).as_posix(), "sha256": sha(target.read_bytes()),
        "xz": [{k:r[k] for k in ["job", "phase", "existing_gzip_bytes", "output_bytes", "size_ratio_to_existing_gzip",
            "compress_cpu_s", "decompress_cpu_s", "after_roundtrip_memory"]} for r in results if r["codec"]=="xz3"]}, indent=2))


if __name__ == "__main__":
    main()
