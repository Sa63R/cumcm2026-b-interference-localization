"""Frozen, CPU-only paired replay of returned RL development worlds.

No fitting, official simulator connections, or checkpoint selection. Each
strategy imports its own committed source in a separate process. The physical
engine and evaluation helper must be byte-identical across all strategies.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(2100001, 2100049))
CHECKPOINT_SHA = "3d21ba7931143f281d2e7554261b001867c65fec9c2bd7465a79f91e95b7cef4"
COMMON = ["experiments/research_v1_eval.py", "experiments/run_q3_comparison.py"]
COMMON += ["src/"+p for p in (
    "simulation/__init__.py", "simulation/cases.py", "simulation/engine.py",
    "simulator_client/__init__.py", "simulator_client/rules.py",
    "simulator_client/client.py", "simulator_client/state.py",
    "simulator_client/errors.py", "geometry/__init__.py", "localization/omni.py")]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(tree):
    paths = set((tree/"src").rglob("*.py")) | {tree/p for p in COMMON}
    return {p.relative_to(tree).as_posix(): sha(p) for p in sorted(paths)}


def record(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def prepare(args):
    evidence = args.evidence.resolve()
    output = args.output.resolve()
    checkpoint = evidence/"run01/trial-1/latest.pt"
    assert sha(checkpoint) == CHECKPOINT_SHA
    recipes = {
        "state_v1": (ROOT.parent/"q3-state-search", "8c624d08c61c0e411f93345d63a932b01ead4868",
                     "experiments/state_search_candidate_relocating_cover_v1.json"),
        "state_certified_tail": (ROOT.parent/"q3-r2-certified-tail", "eb0b5a822a61f9077bbc403f87549b148349b5bd",
                     "experiments/state_search_candidate_certified_tail_v2.json"),
        "rl_trial1": (ROOT, "4e2ba11b9cebdaf8adb92ec76f0ddfb36ed43851", None),
    }
    policies, common = {}, None
    for name, (tree, expected, config) in recipes.items():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip()
        assert commit == expected, (name, commit)
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", *([config] if config else [])], cwd=tree, text=True)
        assert not dirty.strip(), (name, "Policy sources/config are dirty")
        source_hashes = hashes(tree)
        physical = {p: source_hashes[p] for p in COMMON}
        if common is None:
            common = physical
        assert physical == common, "Physical engine/evaluation helper mismatch"
        spec = read(tree/config) if config else {
            "name": "rl_trial1", "entrypoint": "research_rl:run_rl_search",
            "kwargs": {"checkpoint": str(checkpoint), "device": "cpu", "num_threads": 1}}
        policies[name] = dict(tree=str(tree), commit=commit, spec=spec, source_sha256=source_hashes)
    output.mkdir(parents=True, exist_ok=False)
    for name, policy in policies.items():
        with zipfile.ZipFile(output/(name+"-source.zip"), "w", zipfile.ZIP_DEFLATED) as archive:
            for rel in policy["source_sha256"]:
                archive.write(Path(policy["tree"])/rel, rel)
        policy["archive_sha256"] = sha(output/(name+"-source.zip"))
    # Copy previously disclosed worlds; do not regenerate a different case set.
    originals = {}
    for seed in SEEDS:
        src = evidence/f"run01/evaluation/trial-1/case-{seed}.json.gz"
        dst = output/f"remote_trial1/case-{seed}.json.gz"
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, dst)
        originals[str(seed)] = dict(record_sha256=sha(dst), case_sha256=record(dst)["row"]["case_sha256"])
    shutil.copyfile(evidence/"physical_bound_cache.json", output/"physical_bound_cache.json")
    manifest = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), seeds=SEEDS,
        scope="Previously used synthetic development worlds; no official counterfactual replay or final-test claim",
        selection="Preselected trained trial-1 endpoint versus independently confirmed certified-tail; protected v1 reference",
        policies=policies, common_source_sha256=common, remote_records=originals,
        checkpoint_sha256=CHECKPOINT_SHA, runner_sha256=sha(__file__),
        limits=dict(max_actions=10000, real_seconds_per_case=300, virtual_seconds_per_case=360000),
        hardware=dict(platform=platform.platform(), python=sys.version, device="cpu", threads_per_process=1),
        lower_bounds=dict(primary="L/5 + 5*N (original state-search physical oracle relaxation)",
                          previous_rl="primary + 30*(20-N) for N<16; primary for N=16",
                          aggregation="sum(actual_time)/sum(case_lower_bound); not mean of individual ratios"))
    write(output/"manifest.json", manifest)
    shutil.copyfile(__file__, output/"runner.py")
    print(json.dumps({"frozen": str(output), "cases": len(SEEDS), "policies": list(policies)}), flush=True)


def worker(args):
    output = args.output.resolve()
    manifest = read(output/"manifest.json")
    assert sha(__file__) == manifest["runner_sha256"]
    policy = manifest["policies"][args.policy]
    tree = Path(policy["tree"])
    assert hashes(tree) == policy["source_sha256"], "Source changed after freeze"
    assert sha(Path(manifest["policies"]["rl_trial1"]["spec"]["kwargs"]["checkpoint"])) == CHECKPOINT_SHA
    sys.path[:0] = [str(tree), str(tree/"src")]
    from experiments.research_v1_eval import run_case, digest, summarize
    from simulation import Scenario, Source
    import socket
    def no_network(*args, **kwargs):
        raise RuntimeError("Offline paired evaluation forbids network connections")
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    socket.create_connection = no_network
    rows = []
    target = output/args.policy
    target.mkdir(exist_ok=False)
    for seed in manifest["seeds"]:
        src = output/f"remote_trial1/case-{seed}.json.gz"
        assert sha(src) == manifest["remote_records"][str(seed)]["record_sha256"]
        remote = record(src)
        config = remote["evaluation"]["ground_truth"]
        case = Scenario(**{**config, "sources": tuple(Source(**s) for s in config["sources"])})
        assert digest(case.evaluation_config()) == remote["row"]["case_sha256"]
        result = run_case(case, policy["spec"], {"limits": manifest["limits"]})
        assert result["row"]["case_sha256"] == remote["row"]["case_sha256"]
        if args.policy == "rl_trial1":
            # Do not assume the remote acceleration/platform is equivalent.
            result["remote_reproduction"] = dict(
                virtual_time_equal=result["row"]["virtual_time_s"] == remote["row"]["virtual_time_s"],
                observation_history_equal=result["history"] == remote["history"])
        dst = target/f"case-{seed}.json.gz"
        dst.write_bytes(gzip.compress(json.dumps(result, ensure_ascii=False, allow_nan=False).encode(), mtime=0))
        rows.append(result["row"])
        print(json.dumps({"policy": args.policy, "completed": len(rows), "successful": result["row"]["successful"]}), flush=True)
    write(target/"summary.json", summarize(rows, len(manifest["seeds"])))
    assert hashes(tree) == policy["source_sha256"], "Source changed during evaluation"


def run(args):
    manifest = read(args.output/"manifest.json")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
    def launch(name):
        with (args.output/(name+".log")).open("w", encoding="utf-8") as stream:
            result = subprocess.run([sys.executable, __file__, "worker", "--output", str(args.output.resolve()),
                                     "--policy", name], env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed; read {name}.log")
        return name
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(launch, name) for name in manifest["policies"]]):
            print(json.dumps({"completed_policy": future.result()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--policy")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
