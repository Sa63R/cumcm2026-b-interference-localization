"""CPU-only synthetic alpha=0/1 cost and conditional-diagnostic microbenchmark."""

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from research_rl.controller import CONTEXT_DIM
from research_rl.distributions import distribution_spec, sample_probe_diagnostics
from research_rl.network import CandidateActorCritic, pack_observations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("iterations must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(923)
    rng = np.random.default_rng(923)
    features = rng.uniform(-1, 1, size=(140, 60)).astype(np.float32)
    features[:, :4] = 0
    features[:112, 0] = 1
    for index in range(112):
        features[index, 4:6] = ((index // 16) / 7, (index // 16) / 14)
        features[index, 45] = (index % 16 + 1) / 20
    features[112:, 1] = 1
    for index in range(112, 140):
        features[index, 45] = ((index - 112) // 4 + 1) / 20
    record = dict(features=features, context=np.zeros(CONTEXT_DIM, dtype=np.float32))
    tensors = pack_observations([record])
    models = [CandidateActorCritic(96, 60, action_distribution=distribution_spec(a)).eval() for a in (0, 1)]
    models[1].load_state_dict(models[0].state_dict())
    center_mask = [index >= 112 and (index-112) % 4 == 0 for index in range(140)]
    times = {alpha: [] for alpha in (0, 1)}
    diagnostic_times = []
    with torch.no_grad():
        for model in models:
            for _ in range(10):
                model(*tensors)
        for repeat in range(4):
            for alpha in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                started = time.perf_counter()
                for _ in range(args.iterations):
                    models[alpha](*tensors)
                times[alpha].append(1000*(time.perf_counter()-started)/args.iterations)
            logits = models[0](*tensors)[0][0].numpy()
            started = time.perf_counter()
            for _ in range(args.iterations):
                sample_probe_diagnostics(logits, features, center_mask, 112)
            diagnostic_times.append(1000*(time.perf_counter()-started)/args.iterations)
    result = dict(kind="synthetic_cpu_only_no_scenarios", candidate_count=140,
                  group_structure="7 cover groups x16 channels + 7 sources x4 probes",
                  iterations=args.iterations, torch_version=torch.__version__, platform=platform.platform(),
                  policy_forward_ms={str(a):statistics.median(t) for a,t in times.items()},
                  forward_repeats_ms=times, behavior_diagnostic_ms=statistics.median(diagnostic_times),
                  diagnostic_excludes="legal disk-center metadata preparation and environment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
