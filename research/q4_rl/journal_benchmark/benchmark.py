"""Single-episode, sequential CPU gzip-level benchmark; no strategy imports."""
import argparse
import ast
import ctypes
from ctypes import wintypes
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TRAINING = ROOT.parent/"q4-rl-micro-attention/results/q4_rl/server-bundle-v3-training-001/scst_initialized/training"
HANDOFF = ROOT/"handoff/journal-benchmark-8007000"
SCHEMES = ("current_function", "level1", "level3", "level6_equivalence", "level9")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]
    info = Counters(); info.cb = ctypes.sizeof(info)
    handle = wintypes.HANDLE(-1)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(info), info.cb):
        raise ctypes.WinError()
    return dict(rss_bytes=info.WorkingSetSize, lifetime_peak_working_set_bytes=info.PeakWorkingSetSize)


def pin_single_cpu():
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    handle = wintypes.HANDLE(-1)
    if not ctypes.windll.kernel32.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
        raise ctypes.WinError()
    selected = process_mask.value & -process_mask.value
    if not selected or not ctypes.windll.kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(selected)):
        raise ctypes.WinError()


def writer_function():
    source = (ROOT/"src/q4_rl/train.py").read_text(encoding="utf-8-sig")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_write_batch")
    code = ast.get_source_segment(source, node)
    namespace = dict(Path=Path, json=json, gzip=gzip)
    exec(compile(code, "current_write_batch_extracted", "exec"), namespace)
    return namespace["_write_batch"], code


def worker(scheme, source):
    pin_single_cpu()
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    assert value["seed"] == 8007000 and value["leg"] == "sample"
    assert value["metrics"]["success"] and not value.get("administrative_skip")
    original, code = writer_function()
    def alternative(path, value):
        temporary = Path(str(path)+".tmp")
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        level = 6 if scheme == "level6_equivalence" else int(scheme.removeprefix("level"))
        with gzip.open(temporary, "wb", compresslevel=level) as stream:
            stream.write(payload)
        temporary.replace(path)
    writer = original if scheme == "current_function" else alternative
    trials = []
    for iteration in range(4):
        path = HANDOFF/f"{scheme}-{iteration}.json.gz"
        before = memory()
        wall, cpu = time.perf_counter(), time.process_time()
        writer(path, value)
        elapsed, cpu_used = time.perf_counter()-wall, time.process_time()-cpu
        after = memory()
        raw = gzip.decompress(path.read_bytes())
        decoded = json.loads(raw)
        assert decoded == value, "Full JSON structure changed"
        trials.append(dict(warmup=iteration == 0, wall_s=elapsed, cpu_s=cpu_used,
            output_bytes=path.stat().st_size, uncompressed_bytes=len(raw),
            decoded_json_sha256=hashlib.sha256(raw).hexdigest(), full_structure_equal=True,
            before_memory=before, after_write_memory=after, file=path.relative_to(ROOT).as_posix()))
        del raw, decoded
    measured = trials[1:]
    print(json.dumps(dict(scheme=scheme, cpu_affinity_count=1, trials=trials,
        mean_wall_s=statistics.mean(t["wall_s"] for t in measured),
        mean_cpu_s=statistics.mean(t["cpu_s"] for t in measured),
        min_cpu_s=min(t["cpu_s"] for t in measured), max_cpu_s=max(t["cpu_s"] for t in measured),
        mean_output_bytes=statistics.mean(t["output_bytes"] for t in measured),
        peak_working_set_bytes=max(t["after_write_memory"]["lifetime_peak_working_set_bytes"] or 0 for t in trials),
        function_sha256=hashlib.sha256(code.encode()).hexdigest())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=SCHEMES)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.source)
        return
    if HANDOFF.exists() or (HERE/"timing.json").exists():
        raise ValueError("Fresh benchmark output required")
    source = sorted((TRAINING/"raw").glob("*-leg-*.json.gz"))[0]
    stem = source.name.split("-leg-")[0]
    index_path = source.with_name(stem+".jsonl")
    entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    entry = next(row for row in entries if row["file"] == source.name)
    assert entry["sha256"] == digest(source) and entry["bytes"] == source.stat().st_size
    assert entry["seed"] == 8007000 and entry["leg"] == "sample" and entry["administrative_skip"] is None
    with (TRAINING/"progress.jsonl").open(encoding="utf-8") as stream:
        committed = json.loads(next(stream))
    assert committed["batch"] == 1 and committed["raw_index"] == "raw/"+index_path.name
    original, code = writer_function()
    assert 'json.dumps' in code and 'compresslevel=6' in code
    (HERE/"current_write_batch.py").write_text("from pathlib import Path\nimport gzip\nimport json\n\n"+code+"\n", encoding="utf-8")
    HANDOFF.mkdir(parents=True)
    results = []
    for scheme in SCHEMES:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", scheme,
                                  "--source", str(source)], text=True, capture_output=True, check=True)
        result = json.loads(process.stdout)
        results.append(result)
        print(json.dumps({key:result[key] for key in ("scheme", "mean_cpu_s", "mean_wall_s", "mean_output_bytes", "peak_working_set_bytes")}), flush=True)
    payload_hashes = {trial["decoded_json_sha256"] for result in results for trial in result["trials"]}
    assert len(payload_hashes) == 1
    baseline = results[0]
    for result in results:
        result["cpu_speedup_vs_current"] = baseline["mean_cpu_s"]/result["mean_cpu_s"]
        result["relative_output_size"] = result["mean_output_bytes"]/baseline["mean_output_bytes"]
    report = dict(scope="I/O microbenchmark only, one prespecified complete synthetic TRAIN episode. No policy/training/validation execution or checkpoint changes.",
        source=dict(file=source.relative_to(ROOT.parent).as_posix(), sha256=digest(source), compressed_bytes=source.stat().st_size,
            index=index_path.relative_to(ROOT.parent).as_posix(), index_sha256=digest(index_path), index_entry=entry,
            committed_batch_verified=True, selection="lexicographically first raw episode; complete and index verified"),
        current_function_sha256=hashlib.sha256(code.encode()).hexdigest(),
        method="Five sequential subprocesses, each pinned to one permitted CPU; one warmup plus three measured complete writes per scheme. Atomic temporary.replace retained; no fsync, same as production. Input parsing and full decode/equality verification excluded from write timing.",
        memory_scope="Windows process-lifetime peak working set, including input parsing, warmup and prior verification; not incremental per-write allocation. Separate subprocess per scheme limits cross-scheme contamination.",
        json_semantics="Same json.dumps ensure_ascii=False, allow_nan=False and default separators; every decoded field preserved, and all 20 decompressed byte strings have identical SHA256.",
        measured_repetitions=3, current_is_already_native_dumps_level6=True, results=results)
    (HERE/"timing.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Journal compression-level microbenchmark", "", report["scope"], "",
        "Current production already uses one native json.dumps/UTF-8 encoding followed by gzip level 6; native serialization is not a new change here.", "",
        "| Scheme | Mean CPU s | Mean wall s | Output MB | CPU speedup | Size / current | Peak working set MB |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        lines.append(f"| {result['scheme']} | {result['mean_cpu_s']:.6f} | {result['mean_wall_s']:.6f} | {result['mean_output_bytes']/1e6:.3f} | {result['cpu_speedup_vs_current']:.3f} | {result['relative_output_size']:.4f} | {result['peak_working_set_bytes']/1e6:.2f} |")
    lines += ["", report["method"], "", report["memory_scope"], "", report["json_semantics"], "",
        "This single episode can establish a compression tradeoff, not explain all 1190.6 seconds of SCST parent overhead. Multiprocessing transfer, model/state copies, checkpoint writes, index fsync, parsing and other bookkeeping were not measured. No production change is made; validate other episode sizes and real end-to-end overhead before deployment.", "",
        "Reproduce with `python research/q4_rl/journal_benchmark/benchmark.py` in the feature-cache worktree after selecting a fresh benchmark output path. Large temporary gzip outputs remain under handoff/journal-benchmark-8007000 and need not be committed.", ""]
    (HERE/"README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
