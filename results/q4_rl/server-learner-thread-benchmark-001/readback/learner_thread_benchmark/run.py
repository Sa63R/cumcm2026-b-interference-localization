"""Bounded CPU-only pack + BC update timing; never exports learned weights."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT/"handoff/v4-bc-fit-readback/g3_h128"
RNG = 424445
RECORD_COUNT = 512
EPOCHS, MINIBATCH = 3, 128
EXPECTED_MODEL_SHA = "61d4ac338cdfc0b62fcbb0cf9c4f2baaf9f832c8d203ac8d33a0f9b308f5ce49"
ATOL, RTOL = 2e-4, 1e-4
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_records():
    index_bytes = (DATA/"batch-000000-attempt-000000.json.gz").read_bytes()
    index = json.loads(gzip.decompress(index_bytes))
    assert index["format"] == "q4-training-episode-index-v1"
    records, inputs = [], []
    for item in index["episodes"]:
        assert Path(item["path"]).name == item["path"]
        raw = (DATA/item["path"]).read_bytes()
        assert digest(raw) == item["sha256"]
        episode = json.loads(gzip.decompress(raw))[0]
        assert episode["seed"] == item["seed"]
        assert episode["metrics"]["split"] == "train"
        count = min(RECORD_COUNT-len(records), len(episode["records"]))
        # Only already generated training features/actions/returns reach update.
        records.extend({key: row[key] for key in (
            "global_features", "candidate_features", "action_index", "return")}
            for row in episode["records"][:count])
        inputs.append(dict(file=item["path"], sha256=item["sha256"], selected_prefix_count=count))
        if len(records) == RECORD_COUNT:
            break
    assert len(records) == RECORD_COUNT
    return records, dict(index_sha256=digest(index_bytes), episodes=inputs,
        selected_records_sha256=digest(json.dumps(records, separators=(",", ":"), allow_nan=False).encode()))


def worker(threads):
    worker_cpu_start, worker_wall_start = time.process_time(), time.perf_counter()
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        assert kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask))
        affinity = [i for i in range(64) if process_mask.value & (1 << i)][:4]
        assert len(affinity) == 4
        assert kernel.SetProcessAffinityMask(handle, sum(1 << i for i in affinity))
    else:
        affinity = sorted(os.sched_getaffinity(0))[:4]
        assert len(affinity) == 4
        os.sched_setaffinity(0, affinity)
    import torch
    from q4_rl.memory_network import model_from_metadata, validate_checkpoint, pack_observations
    from q4_rl.train import imitation_update
    torch.set_num_interop_threads(1)
    torch.set_num_threads(threads)
    model_bytes = (DATA/"warmstart.pt").read_bytes()
    assert digest(model_bytes) == EXPECTED_MODEL_SHA
    import io
    saved = torch.load(io.BytesIO(model_bytes), map_location="cpu", weights_only=True)
    validate_checkpoint(saved)
    assert saved["network"]["hidden"] == 128
    assert saved["state"]["warmstart_completed"] == 256
    assert saved["state"]["ppo_batches"] == 0 and saved["state"]["pending_batch"] is None
    records, source = read_records()
    counts = [len(row["candidate_features"]) for row in records]
    assert all(len(row["global_features"]) == 13 for row in records)
    assert all(len(candidate) == 58 for row in records for candidate in row["candidate_features"])
    model = model_from_metadata(saved["network"])

    def reset():
        model.load_state_dict(saved["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        random.seed(RNG)
        torch.manual_seed(RNG)
        return optimizer

    def update():
        return imitation_update(model, reset(), records, epochs=EPOCHS, minibatch_size=MINIBATCH,
            stop_check=lambda: time.process_time()-worker_cpu_start > 95.)

    # Full same-work warmup, followed by three independent equal-work trials.
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    warmup = update()
    warmup.update(wall_s=time.perf_counter()-start_wall, cpu_s=time.process_time()-start_cpu)
    rounds = []
    for trial in range(3):
        optimizer = reset()
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        result = imitation_update(model, optimizer, records, epochs=EPOCHS, minibatch_size=MINIBATCH,
            stop_check=lambda: time.process_time()-worker_cpu_start > 95.)
        result.update(trial=trial, wall_s=time.perf_counter()-start_wall,
                      cpu_s=time.process_time()-start_cpu)
        assert math.isfinite(result["imitation_loss"])
        assert result["updates"] == 12 and result["records"] == RECORD_COUNT
        assert torch.get_num_threads() == threads and torch.get_num_interop_threads() == 1
        rounds.append(result)
    assert all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        packed = pack_observations(records[:4])
        logits, values = model(*packed)
    # Output probes are not trainable model parameters or a usable checkpoint.
    probe = dict(logits=logits[packed[2]].tolist(), values=values.tolist())
    assert all(math.isfinite(value) for values in probe.values() for value in values)
    return dict(threads=threads, interop_threads=1, affinity=affinity,
        blas_environment_threads=1, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        schema="q4-micro-g3-negative-v1", model_sha256=EXPECTED_MODEL_SHA,
        network=saved["network"], source=source, record_count=RECORD_COUNT,
        candidate_count=dict(min=min(counts), max=max(counts), mean=statistics.mean(counts), total=sum(counts)),
        random_seed=RNG, epochs=EPOCHS, minibatch_size=MINIBATCH, warmup=warmup, rounds=rounds,
        numerical_probe=probe, median_wall_s=statistics.median(row["wall_s"] for row in rounds),
        median_cpu_s=statistics.median(row["cpu_s"] for row in rounds),
        process_cpu_total_s=time.process_time()-worker_cpu_start,
        process_wall_total_s=time.perf_counter()-worker_wall_start)


def comparison(reference, result):
    assert result["source"] == reference["source"] and result["model_sha256"] == reference["model_sha256"]
    maximum = {}
    consistent = True
    for field in ("logits", "values"):
        a, b = reference["numerical_probe"][field], result["numerical_probe"][field]
        assert len(a) == len(b)
        maximum[field] = max(abs(x-y) for x,y in zip(a,b))
        consistent &= all(abs(x-y) <= ATOL+RTOL*abs(x) for x,y in zip(a,b))
    loss_delta = max(abs(a["imitation_loss"]-b["imitation_loss"])
        for a,b in zip(reference["rounds"], result["rounds"]))
    consistent &= all(abs(a["imitation_loss"]-b["imitation_loss"]) <= 1e-5+1e-5*abs(a["imitation_loss"])
        for a,b in zip(reference["rounds"], result["rounds"]))
    return dict(threads=result["threads"], approximately_equal=consistent,
        max_absolute_output_difference=maximum, max_absolute_loss_difference=loss_delta,
        output_atol=ATOL, output_rtol=RTOL, loss_atol=1e-5, loss_rtol=1e-5,
        median_wall_speedup=reference["median_wall_s"]/result["median_wall_s"],
        median_cpu_ratio=result["median_cpu_s"]/reference["median_cpu_s"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-threads", type=int, choices=(1,2,4))
    args = parser.parse_args()
    if args.worker_threads:
        print(json.dumps(worker(args.worker_threads), allow_nan=False), flush=True)
        return
    if (HERE/"timings.json").exists():
        raise FileExistsError("Existing timing evidence must not be overwritten")
    results = []
    for threads in (1,2,4):
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker-threads", str(threads)],
            cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, timeout=180)
        if proc.returncode:
            # No stack trace with local account paths in portable evidence.
            raise RuntimeError(f"Bounded worker {threads} failed with code {proc.returncode}")
        result = json.loads(proc.stdout)
        results.append(result)
        (HERE/f"threads-{threads}.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("threads", "median_wall_s", "median_cpu_s", "process_cpu_total_s")}), flush=True)
    output = dict(scope="Fixed training-record CPU kernel/update timing only; no rollout or policy efficacy claim",
        script_sha256=digest(Path(__file__).read_bytes()),
        source_sha256={name: digest((ROOT/name).read_bytes()) for name in (
            "src/q4_rl/network.py", "src/q4_rl/memory_network.py", "src/q4_rl/train.py")},
        total_worker_cpu_s=sum(result["process_cpu_total_s"] for result in results),
        results=results, comparisons=[comparison(results[0], result) for result in results])
    (HERE/"timings.json").write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(comparisons=output["comparisons"], total_worker_cpu_s=output["total_worker_cpu_s"])), flush=True)


if __name__ == "__main__":
    main()
