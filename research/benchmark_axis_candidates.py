"""Paired CPU teacher replay with/without axis candidates and diagnostics.

Weights, world, physical trajectory and teacher are equal. Only candidate-set
preparation, actor/critic evaluation and diagnostic costs differ. The benchmark
does not estimate RL optimizer padding cost or learned-policy performance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from research_rl import run_rl_search
from research_rl.action_sets import action_schema
from research_rl.network import CandidateActorCritic, TorchPolicy
from research_rl.portable_checkpoint import model_tensor_digest
from simulation import LocalResearchSimulator, random_scenario
from tests.test_strategy import ObservationOnlyClient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=110001)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/rl/axis_cpu_benchmark.json"))
    args = parser.parse_args(argv)
    if args.count < 32 or args.repeats < 2 or not 100001 <= args.seed_start <= args.seed_start+args.count-1 <= 199999:
        parser.error("Use at least 32 training scenarios and two alternating repeats")
    torch.set_num_threads(1)
    torch.manual_seed(9911)
    models = {name: CandidateActorCritic(96,60,action_schema=action_schema(name)).eval()
              for name in ("base","axis_quantiles")}
    models["axis_quantiles"].load_state_dict(models["base"].state_dict())
    rows = []
    for repeat in range(args.repeats):
        for seed in range(args.seed_start,args.seed_start+args.count):
            results = {}
            order = ("base","axis_quantiles") if (seed+repeat)%2 else ("axis_quantiles","base")
            for name in order:
                simulator = LocalResearchSimulator(random_scenario(3,seed))
                policy = TorchPolicy(models[name],teacher=True,capture_diagnostics=True)
                started = time.perf_counter()
                report = run_rl_search(ObservationOnlyClient(simulator.client()),policy=policy)
                elapsed = time.perf_counter()-started
                assert simulator.evaluation()["all_cleared"] and report.completion_certified_under_model
                results[name] = report
                rows.append(dict(seed=seed,repeat=repeat,action_schema=name,elapsed_s=elapsed,
                    inference_s=report.learning["inference_wall_time_s"],
                    geometry_features_s=report.learning["feature_wall_time_s"],
                    decisions=report.learning["decisions"],candidate_sum=report.learning["candidate_count_sum"]))
            assert results["base"].action_history==results["axis_quantiles"].action_history
            assert results["base"].virtual_time_s==results["axis_quantiles"].virtual_time_s
    means = {name:statistics.mean(row["elapsed_s"] for row in rows if row["action_schema"]==name)
             for name in models}
    totals = {name: {key:statistics.mean(r[key] for r in rows if r["action_schema"]==name)
                   for key in ("inference_s","geometry_features_s","decisions","candidate_sum")}
              for name in models}
    result = dict(schema=1,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        platform=platform.platform(),torch_version=torch.__version__,num_threads=1,
        seed_range=[args.seed_start,args.seed_start+args.count-1],repeats=args.repeats,
        weight_seed=9911,model_tensor_digest=model_tensor_digest(models["base"].state_dict()),
        teacher_trajectory_pairs=len(rows)//2,all_physical_histories_exact=True,
        mean_wall_s=means,wall_ratio=means["axis_quantiles"]/means["base"],components=totals,
        limitations="CPU teacher policy forward with diagnostics; not PPO optimization or learned trajectory performance",
        rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
