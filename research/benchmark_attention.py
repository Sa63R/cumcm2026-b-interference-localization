"""Synthetic CPU throughput ablation; does not train or inspect test scenarios.

Run with PYTHONPATH=src python research/benchmark_attention.py --output result.json.
Alternates MLP/one/two-block model order, uses candidate counts
seen in the joint-scan controller, and separates prepacked inference from packing.
"""

import argparse
import json
import platform
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from research_rl.controller import CONTEXT_DIM
from research_rl.network import CandidateActorCritic, architecture_spec, pack_observations
from research_rl.train import git_version, source_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.repeats < 1 or args.num_threads < 1:
        parser.error("iterations, repeats and num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(914)
    generator = np.random.default_rng(914)
    specs = [architecture_spec(), architecture_spec("attention", 1), architecture_spec("attention", 2)]
    models = [CandidateActorCritic(args.hidden, 60, spec).eval() for spec in specs]
    # Nonzero gates exercise the trained path; zero gates deliberately do not
    # skip computation either, since their gradients need branch activations.
    with torch.no_grad():
        for model in models:
            for block in model.relations:
                block.attention_gate.fill_(0.1)
                block.feedforward_gate.fill_(0.1)
    rows = []
    for count in (16, 32, 80, 140, 220):
        records = [dict(features=generator.normal(size=(count, 60)).astype(np.float32),
                        context=generator.normal(size=CONTEXT_DIM).astype(np.float32))]
        packed = pack_observations(records)
        times = {i: {"prepacked_ms": [], "pack_and_forward_ms": []} for i in range(3)}
        with torch.no_grad():
            for model in models:
                for _ in range(10):
                    model(*packed)
            for repeat in range(args.repeats):
                for index in (range(3) if repeat % 2 == 0 else range(2, -1, -1)):
                    for include_pack in (False, True):
                        begin = time.perf_counter()
                        for _ in range(args.iterations):
                            models[index](*(pack_observations(records) if include_pack else packed))
                        elapsed = 1000 * (time.perf_counter() - begin) / args.iterations
                        key = "pack_and_forward_ms" if include_pack else "prepacked_ms"
                        times[index][key].append(elapsed)
        for index, spec in enumerate(specs):
            row = dict(architecture=spec, candidate_count=count,
                       parameters=sum(p.numel() for p in models[index].parameters()),
                       repeats_ms=times[index],
                       **{key: statistics.median(values) for key, values in times[index].items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = dict(kind="synthetic_cpu_inference_only", scenarios_used=[],
                  git_commit=git_version(), source_manifest=source_manifest(),
                  torch_version=torch.__version__, platform=platform.platform(),
                  python=platform.python_version(), args=vars(args), rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
