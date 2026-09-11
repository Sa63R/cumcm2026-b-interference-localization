"""Same-information induced-set ablation contracts and isolated CPU benchmark.

Optional microbenchmark: python tests/test_q4_rl_attention.py --benchmark induced
--output results/q4_rl/attention-microbench/induced.json. It uses fabricated
feature tensors, never scenarios, official actions, or validation examples.
"""
from copy import deepcopy
import random

import pytest

torch = pytest.importorskip("torch")

from q4_rl.network import (ARCHITECTURE, INDUCED_ARCHITECTURE, CandidateActorCritic,
    InducedCandidateActorCritic, configure_cpu, feature_schema, model_from_metadata,
    pack_observations, TorchPolicy, load_policy)
from q4_rl.train import (attach_returns, ppo_update, save_checkpoint, restore_checkpoint,
                         _configuration)


@pytest.fixture(autouse=True)
def cpu_only():
    configure_cpu()


def observation(n=5):
    return {"global_features": [.1*i for i in range(10)],
            "candidate_features": [[(i*3+j)/40 for j in range(16)] for i in range(n)]}


@pytest.mark.parametrize("training",[False,True])
def test_induced_candidate_permutation_and_padding(training):
    torch.manual_seed(401)
    model=InducedCandidateActorCritic(hidden=16,inducing_points=4,attention_heads=2)
    model.train(training)
    row=observation(5)
    order=[3,0,4,1,2]
    permuted={**row,"candidate_features":[row["candidate_features"][i] for i in order]}
    logits,value=model(*pack_observations([row]))
    actual,other_value=model(*pack_observations([permuted,observation(12)]))
    torch.testing.assert_close(actual[0,:5],logits[0,order],atol=1e-7,rtol=1e-5)
    torch.testing.assert_close(other_value[:1],value,atol=1e-6,rtol=1e-5)
    assert torch.isneginf(actual[0,5:]).all()


def test_padding_nan_values_and_gradients_cannot_affect_valid_candidates():
    torch.manual_seed(402)
    model=InducedCandidateActorCritic(hidden=16,inducing_points=4,attention_heads=2)
    g,x,mask=pack_observations([observation(2),observation(9)])
    reference=model(g,x,mask)
    x=x.masked_fill(~mask[...,None],float("nan")).requires_grad_()
    logits,value=model(g,x,mask)
    torch.testing.assert_close(logits[mask],reference[0][mask])
    torch.testing.assert_close(value,reference[1])
    (logits[mask].square().sum()+value.square().sum()).backward()
    assert torch.isfinite(x.grad).all()
    assert torch.equal(x.grad[~mask],torch.zeros_like(x.grad[~mask]))


@pytest.mark.parametrize("non_cpu",["global","candidate","mask","model"])
def test_cpu_only_rejection_without_initializing_a_gpu(non_cpu):
    model=InducedCandidateActorCritic(hidden=16,inducing_points=4,attention_heads=2)
    g,x,mask=pack_observations([observation()])
    if non_cpu=="global": g=g.to("meta")
    elif non_cpu=="candidate": x=x.to("meta")
    elif non_cpu=="mask": mask=mask.to("meta")
    else: model=model.to("meta")
    with pytest.raises(ValueError,match="CPU-only"):
        model(g,x,mask)


def test_new_metadata_and_legacy_mlp_are_distinct_and_reloadable():
    legacy=CandidateActorCritic(hidden=16)
    assert legacy.metadata()==dict(architecture=ARCHITECTURE,global_dim=10,candidate_dim=16,hidden=16)
    copy=model_from_metadata(legacy.metadata())
    copy.load_state_dict(legacy.state_dict())
    assert type(copy) is CandidateActorCritic
    model=InducedCandidateActorCritic(hidden=16,inducing_points=8,attention_heads=2,attention_blocks=2,attention_refinement="norm_ff")
    metadata=model.metadata()
    assert metadata["architecture"]==INDUCED_ARCHITECTURE
    restored=model_from_metadata(metadata)
    restored.load_state_dict(model.state_dict())
    assert restored.metadata()==metadata
    bad=deepcopy(metadata)
    bad.pop("attention_heads")
    with pytest.raises(ValueError,match="incomplete"):
        model_from_metadata(bad)
    with pytest.raises(ValueError,match="divisible"):
        InducedCandidateActorCritic(hidden=16,attention_heads=3)
    assert len(feature_schema()["global_features"])==10
    assert len(feature_schema()["candidate_features"])==16


def _records(model):
    policy=TorchPolicy(model,deterministic=False)
    for n in (3,5,2,4):
        policy(**observation(n))
    records=policy.records
    for i,row in enumerate(records):
        row["cost_s"]=20.+i*7
    attach_returns(records,actual_time_s=sum(r["cost_s"] for r in records),success=True)
    return records


def test_attention_checkpoint_restores_rng_optimizer_and_next_update(tmp_path):
    torch.manual_seed(403)
    random.seed(403)
    model=InducedCandidateActorCritic(hidden=16,inducing_points=4,attention_heads=2)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    records=_records(model)
    ppo_update(model,optimizer,records,epochs=1,minibatch_size=2)
    state={"next_seed":8000100,"pending_batch":{"seeds":[8000100],"action_seeds":[100]}}
    config={"architecture":"induced","inducing_points":4,"attention_heads":2,
            "attention_blocks":1,"learning_rate":.001}
    checkpoint=tmp_path/"induced.pt"
    save_checkpoint(checkpoint,model,optimizer,state,config)
    expected_python,expected_torch=random.random(),torch.rand(3)
    ppo_update(model,optimizer,records,epochs=2,minibatch_size=2)
    restored,restored_optimizer,restored_state,restored_config=restore_checkpoint(checkpoint)
    assert random.random()==expected_python
    assert torch.equal(torch.rand(3),expected_torch)
    assert restored_state==state and restored_config==config
    assert restored.metadata()["architecture"]==INDUCED_ARCHITECTURE
    ppo_update(restored,restored_optimizer,records,epochs=2,minibatch_size=2)
    assert all(torch.equal(value,restored.state_dict()[name]) for name,value in model.state_dict().items())
    assert load_policy(checkpoint)(**observation(3)) in range(3)


def test_legacy_config_shape_remains_compatible_with_resume():
    from types import SimpleNamespace
    values={name:1 for name in ("hidden","learning_rate","epochs","minibatch_size","batch_episodes",
        "warmstart_episodes","max_decisions","random_seed","scenario_start","scenario_end","entropy_coefficient")}
    args=SimpleNamespace(**values,architecture="mlp",inducing_points=16,attention_heads=2,attention_blocks=1,attention_dim=16,attention_refinement="residual")
    assert _configuration(args)==values
    args.architecture="induced"
    assert _configuration(args)=={**values,"architecture":"induced","inducing_points":16,
                                 "attention_heads":2,"attention_blocks":1,"attention_dim":16,"attention_refinement":"residual"}


def _peak_process_bytes():
    import sys
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss if sys.platform=="darwin" else rss*1024)
    except ImportError:
        # Windows process-lifetime peak working set. No third-party dependency.
        import ctypes
        from ctypes import wintypes
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(key,ctypes.c_size_t) for key in
                ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage",
                 "QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL("psapi",use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessMemoryCounters),wintypes.DWORD]
        counters=ProcessMemoryCounters()
        counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise OSError("Cannot query process peak working set")
        return int(counters.PeakWorkingSetSize)


def benchmark(argv=None):
    """Isolated tensor/gradient timings, not a scene or policy performance run."""
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import statistics
    import time
    parser=argparse.ArgumentParser()
    parser.add_argument("--benchmark",choices=("mlp","induced"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--inducing-points",type=int,default=16)
    parser.add_argument("--attention-heads",type=int,default=2)
    parser.add_argument("--attention-dim",type=int,default=16)
    parser.add_argument("--attention-refinement",choices=("residual","norm_ff"),default="residual")
    parser.add_argument("--hidden",type=int,default=64)
    parser.add_argument("--candidates",type=int,default=440)
    parser.add_argument("--repetitions",type=int,default=30)
    args=parser.parse_args(argv)
    configure_cpu()
    torch.manual_seed(409)
    peak_before=_peak_process_bytes()
    model=(CandidateActorCritic(hidden=args.hidden) if args.benchmark=="mlp" else
        InducedCandidateActorCritic(hidden=args.hidden,inducing_points=args.inducing_points,
                                   attention_heads=args.attention_heads,attention_dim=args.attention_dim,
                                   attention_refinement=args.attention_refinement))
    torch.manual_seed(410)  # Identical fabricated inputs across architectures.
    g=torch.randn(args.batch_size,10)
    x=torch.randn(args.batch_size,args.candidates,16)
    mask=torch.ones(args.batch_size,args.candidates,dtype=torch.bool)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    with torch.no_grad():
        logits,_=model(g,x,mask)
        old_log=torch.log_softmax(logits,-1)[:,0]
    returns=torch.linspace(-5.,-8.,args.batch_size)
    advantage=torch.linspace(-1.,1.,args.batch_size)
    timings={"forward_batch1_s":[],"ppo_tensor_update_s":[]}
    # Update includes actor/critic/entropy, backward, gradient clipping and Adam;
    # it excludes Python record packing, scenario generation and all simulation.
    def update():
        logits,values=model(g,x,mask)
        distribution=torch.distributions.Categorical(logits=logits)
        logp=distribution.log_prob(torch.zeros(args.batch_size,dtype=torch.long))
        ratio=(logp-old_log).exp()
        loss=-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
        loss=loss+.5*torch.nn.functional.smooth_l1_loss(values,returns)-.005*distribution.entropy().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
        optimizer.step()
    for _ in range(5):
        with torch.no_grad(): model(g[:1],x[:1],mask[:1])
        update()
    peak_after_warmup=_peak_process_bytes()
    for _ in range(args.repetitions):
        start=time.perf_counter()
        with torch.no_grad(): model(g[:1],x[:1],mask[:1])
        timings["forward_batch1_s"].append(time.perf_counter()-start)
        start=time.perf_counter()
        update()
        timings["ppo_tensor_update_s"].append(time.perf_counter()-start)
    from q4_rl import network
    output={"kind":"isolated-fabricated-tensor-cpu-benchmark-not-policy-performance",
        "network":model.metadata(),"torch_version":torch.__version__,"native_threads":torch.get_num_threads(),
        "source_sha256":hashlib.sha256(Path(network.__file__).read_bytes()).hexdigest(),
        "batch_size":args.batch_size,"candidates":args.candidates,"repetitions":args.repetitions,
        "parameter_count":sum(p.numel() for p in model.parameters()),"raw_times":timings,
        "summary":{name:{"median":statistics.median(times),"mean":statistics.mean(times),
                            "max":max(times)} for name,times in timings.items()},
        "peak_process_bytes_before_model":peak_before,"peak_process_bytes_after_warmup":peak_after_warmup,
        "peak_process_bytes_end":_peak_process_bytes(),
        "memory_scope":"Fresh process peak working set including interpreter/PyTorch; not allocator-exact model memory",
        "score_scope":"No physical actions/scenarios, no virtual time or theoretical task lower-bound ratio applies"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in output.items() if k!="raw_times"}))


if __name__=="__main__":
    benchmark()
