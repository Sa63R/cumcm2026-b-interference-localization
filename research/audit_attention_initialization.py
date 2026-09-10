"""Audit neutral strong-MLP to attention initialization on training-only worlds."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from research_rl.joint_scan import JointScanRLSearch
from research_rl.network import CandidateActorCritic, TorchPolicy, architecture_spec, pack_observations
from research_rl.portable_checkpoint import model_tensor_digest
from research_rl.train import initialize_from, source_manifest
from simulation import LocalResearchSimulator, random_scenario
from tests.test_strategy import ObservationOnlyClient


def trial(model, seed, deterministic):
    torch.manual_seed(seed + 9951)
    np.random.seed(seed + 9951)
    random.seed(seed + 9951)
    simulator = LocalResearchSimulator(random_scenario(3, seed))
    records = []
    def record(f,c,a,t,s,cost):
        records.append(dict(features=np.asarray(f,dtype=np.float32), context=np.asarray(c,dtype=np.float32),
                            action=a, teacher=t, log_prob=s[1], value=s[2], cost=cost))
    controller = JointScanRLSearch(ObservationOnlyClient(simulator.client()),
        TorchPolicy(model, deterministic=deterministic, capture_diagnostics=True), recorder=record)
    start = time.perf_counter()
    report = controller.run()
    wall = time.perf_counter()-start
    result = simulator.evaluation()
    assert result["all_cleared"] and not result["failed_clear_count"] and report.completion_certified_under_model
    return records, report, wall


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--output",type=Path,default=Path("research/attention_strong_evidence/initialization.json"))
    parser.add_argument("--seed-start",type=int,default=110601)
    parser.add_argument("--count",type=int,default=8)
    args = parser.parse_args(argv)
    if args.count < 8 or not 100001 <= args.seed_start <= args.seed_start+args.count-1 <= 199999:
        parser.error("Use at least eight training worlds in 100001..199999")
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    assert payload["feature_schema"]["version"] == "v3" and payload["hidden"] == 96
    torch.manual_seed(9112037)
    old = CandidateActorCritic(96,60).eval()
    old_rng = torch.get_rng_state().clone()
    old.load_state_dict(payload["model"])
    torch.manual_seed(9112037)
    new = CandidateActorCritic(96,60,architecture_spec("attention",1,4)).eval()
    assert torch.equal(old_rng,torch.get_rng_state())
    initialize_from(new,payload)
    assert torch.equal(old_rng,torch.get_rng_state())
    rows = []
    models = {"mlp":old,"attention":new}
    for seed in range(args.seed_start,args.seed_start+args.count):
        for deterministic in (True,False):
            order = ["mlp","attention"] if (seed+int(deterministic))%2 else ["attention","mlp"]
            runs = {name:trial(models[name],seed,deterministic) for name in order}
            a, b = runs["mlp"][0],runs["attention"][0]
            assert len(a)==len(b)
            assert runs["mlp"][1].action_history == runs["attention"][1].action_history
            for left,right in zip(a,b):
                assert np.array_equal(left["features"],right["features"])
                assert np.array_equal(left["context"],right["context"])
                assert all(left[key]==right[key] for key in ("action","teacher","log_prob","value","cost"))
            tensors = pack_observations(a)
            with torch.no_grad():
                assert all(torch.equal(x,y) for x,y in zip(old(*tensors),new(*tensors)))
            rows.append(dict(seed=seed,deterministic=deterministic,order=order,
                virtual_time_s=runs["mlp"][1].virtual_time_s, decisions=len(a),
                mlp_wall_s=runs["mlp"][2],attention_wall_s=runs["attention"][2],exact=True))
    result = dict(schema=1,scope="CPU float32 initial-policy parity, not a performance improvement or GPU cost claim",
        checkpoint_name=args.checkpoint.name,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        model_tensor_digest=model_tensor_digest(payload["model"]),source_manifest=source_manifest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        constructor_and_transfer_rng_exact=True,all_actions_logits_values_exact=True,
        cases=args.count,seed_range=[args.seed_start,args.seed_start+args.count-1],torch_version=torch.__version__,
        mean_wall_s={name:statistics.mean(r[f"{name}_wall_s"] for r in rows) for name in models},rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in {"rows","source_manifest"}},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
