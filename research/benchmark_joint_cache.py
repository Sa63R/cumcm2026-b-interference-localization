"""Exact v3 cache audit against a frozen source snapshot, training seeds only.

Run from the repository root with PYTHONPATH=src. No PyTorch/GPU or official
simulator is needed. --reference-root permits use without a local .git directory.
"""

import argparse
import cProfile
import importlib.util
import json
import hashlib
from pathlib import Path
import pstats
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import types

from research_rl.joint_scan import JointScanRLSearch
from simulation import LocalResearchSimulator, random_scenario


class ObservationOnlyClient:
    __slots__ = ("_client",)
    def __init__(self, client):
        self._client = client
    def __getattr__(self, name):
        if name not in {"state", "remaining_real_time_s", "pending_request", "enter", "measure", "clear", "exit"}:
            raise AssertionError(f"non-observation access: {name}")
        return getattr(self._client, name)


def load_reference(directory):
    package = types.ModuleType("joint_cache_reference")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    for name in ("controller", "joint_scan"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", directory / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package.__name__}.joint_scan"].JointScanRLSearch


def run(cls, seed, *, capture=False, selector="teacher"):
    observations, candidate_sets = [], []
    rng = random.Random(seed + 310000)
    def policy(features, context, teacher):
        if capture:
            observations.append((features, context, teacher))
        return teacher if selector == "teacher" else rng.randrange(len(features))
    simulator = LocalResearchSimulator(random_scenario(3, seed))
    controller = cls(ObservationOnlyClient(simulator.client()), policy)
    if capture:
        original = controller._candidates
        def candidates(remaining):
            result = original(remaining)
            candidate_sets.append([(c.kind, c.channel, c.point.x, c.point.y, c.option, c.radius) for c in result])
            return result
        controller._candidates = candidates
    started = time.perf_counter()
    report = controller.run()
    elapsed = time.perf_counter() - started
    evaluation = simulator.evaluation()
    row = dict(seed=seed, wall_s=elapsed, feature_s=report.learning["feature_wall_time_s"],
               virtual_s=report.virtual_time_s, decisions=report.learning["decisions"],
               candidate_sum=report.learning["candidate_count_sum"],
               complete=report.completion_certified_under_model and evaluation["all_cleared"],
               failed_clear_count=evaluation["failed_clear_count"])
    return row, observations, candidate_sets, report.action_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-commit", default="fe6cf25")
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/rl/joint-cache-audit"))
    parser.add_argument("--seed-start", type=int, default=100501)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile-count", type=int, default=8)
    args = parser.parse_args()
    if args.count < 32 or not 100001 <= args.seed_start < args.seed_start + args.count <= 200000:
        parser.error("audit requires at least 32 training-only seeds in 100001..199999")
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="q3-frozen-cache-") as temporary:
        folder = Path(temporary)
        for name in ("controller", "joint_scan"):
            if args.reference_root:
                contents = (args.reference_root / "src" / "research_rl" / f"{name}.py").read_text(encoding="utf-8")
            else:
                contents = subprocess.check_output(["git", "show", f"{args.reference_commit}:src/research_rl/{name}.py"], text=True)
            (folder / f"{name}.py").write_text(contents, encoding="utf-8")
        reference = load_reference(folder)
        exact_checks = []
        for seed in range(args.seed_start, args.seed_start + args.count):
            for selector in ("teacher", "random"):
                before = run(reference, seed, capture=True, selector=selector)
                after = run(JointScanRLSearch, seed, capture=True, selector=selector)
                assert before[1] == after[1], (seed, selector, "actor inputs/teacher")
                assert before[2] == after[2], (seed, selector, "candidate order or values")
                assert before[3] == after[3], (seed, selector, "complete action history")
                assert before[0]["virtual_s"] == after[0]["virtual_s"]
                exact_checks.append(dict(seed=seed, selector=selector, exact=True,
                                         virtual_s=after[0]["virtual_s"], complete=after[0]["complete"]))
            if (seed - args.seed_start + 1) % 8 == 0:
                print(json.dumps(dict(stage="exact_audit", scenarios=seed - args.seed_start + 1)), flush=True)
        rows = {"reference": [], "cached": []}
        for repeat in range(args.repeats):
            for seed in range(args.seed_start, args.seed_start + args.count):
                order = ("reference", "cached") if (repeat + seed) % 2 == 0 else ("cached", "reference")
                for name in order:
                    cls = reference if name == "reference" else JointScanRLSearch
                    row = run(cls, seed)[0]
                    row["repeat"] = repeat
                    rows[name].append(row)
        summary = {}
        for name, values in rows.items():
            summary[name] = {key: statistics.mean(v[key] for v in values)
                             for key in ("wall_s", "feature_s", "virtual_s", "decisions", "candidate_sum")}
            summary[name]["episodes"] = len(values)
            summary[name]["complete"] = sum(v["complete"] for v in values)
        summary["wall_speedup"] = summary["reference"]["wall_s"] / summary["cached"]["wall_s"]
        summary["wall_reduction_fraction"] = 1 - 1 / summary["wall_speedup"]
        for name, cls in (("reference", reference), ("cached", JointScanRLSearch)):
            profiler = cProfile.Profile()
            profiler.enable()
            for seed in range(args.seed_start, args.seed_start + args.profile_count):
                run(cls, seed)
            profiler.disable()
            profiler.dump_stats(str(args.output / f"{name}.prof"))
            with (args.output / f"{name}_profile.txt").open("w", encoding="utf-8") as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(40)
        result = dict(reference_commit=args.reference_commit, seed_start=args.seed_start,
                      count=args.count, repeats=args.repeats, exact_checks=exact_checks,
                      summary=summary, timings=rows, python_version=platform.python_version(),
                      platform=platform.platform(), candidate_source_sha256={
                          name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("src/research_rl/controller.py", "src/research_rl/joint_scan.py")})
        (args.output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
