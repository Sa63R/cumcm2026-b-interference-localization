"""Separate generic Torch construction state from the actual PPO RNG streams."""

from argparse import Namespace
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research_rl.controller import ALGORITHM_VERSIONS, CONTEXT_DIM, feature_schema
from research_rl.network import CandidateActorCritic, GatedSetAttention, architecture_spec, load_policy
from research_rl.train import initialize_from, episode, main, update


def checkpoint(model):
    return dict(algorithm=ALGORITHM_VERSIONS["v3"], feature_schema=feature_schema("v3"),
                hidden=model.hidden, architecture=model.architecture, model=model.state_dict())


@pytest.mark.parametrize("layers", [1, 2])
def test_constructor_and_transfer_preserve_mlp_rng_and_old_attention_weights(layers):
    torch.set_num_threads(1)
    torch.manual_seed(9112037)
    old = CandidateActorCritic(96, 60)
    before_relations = torch.get_rng_state().clone()
    # This exactly reproduces the old relation-initialization random stream.
    expected_relations = [GatedSetAttention(96, 4) for _ in range(layers)]
    torch.set_rng_state(before_relations)
    expected_generic_torch_draw = torch.randperm(512)  # Not the PPO minibatch shuffle.
    torch.manual_seed(9112037)
    expanded = CandidateActorCritic(96, 60, architecture_spec("attention", layers, 4))
    assert torch.equal(torch.get_rng_state(), before_relations)
    assert all(torch.equal(value, expanded.state_dict()[key]) for key,value in old.state_dict().items())
    for expected, actual in zip(expected_relations, expanded.relations):
        assert all(torch.equal(value, actual.state_dict()[key]) for key,value in expected.state_dict().items())
    initialize_from(expanded, checkpoint(old))
    assert torch.equal(torch.get_rng_state(), before_relations)
    assert torch.equal(torch.randperm(512), expected_generic_torch_draw)


def test_trained_old_attention_checkpoint_loads_and_preserves_restored_rng(tmp_path):
    torch.manual_seed(1011)
    previous = CandidateActorCritic(16, 60, architecture_spec("attention")).eval()
    with torch.no_grad():
        previous.relations[0].attention_gate.fill_(.25)
        previous.relations[0].feedforward_gate.fill_(-.15)
    saved = checkpoint(previous)
    saved["torch_rng"] = torch.get_rng_state().clone()
    expected_permutation = torch.randperm(256)
    path = tmp_path / "old_attention.pt"
    torch.save(saved, path)
    loaded = load_policy(path).model
    assert all(torch.equal(value, loaded.state_dict()[key]) for key,value in previous.state_dict().items())
    tensors = (torch.randn(2, 19, 60), torch.randn(2, CONTEXT_DIM), torch.ones(2, 19, dtype=torch.bool))
    with torch.no_grad():
        assert all(torch.equal(a,b) for a,b in zip(previous(*tensors), loaded(*tensors)))
    # Resume restores saved RNG after construction/loading, so the saved stream
    # has the same meaning under the old and new constructors.
    torch.set_rng_state(saved["torch_rng"])
    assert torch.equal(torch.randperm(256), expected_permutation)


def test_neutral_attention_transfer_matches_sampled_complete_training_trajectory():
    torch.set_num_threads(1)
    torch.manual_seed(812)
    old = CandidateActorCritic(16, 60)
    new = CandidateActorCritic(16, 60, architecture_spec("attention"))
    initialize_from(new, checkpoint(old))
    records = []
    metrics = []
    for model in (old, new):
        trajectory, status = episode((110631, model.state_dict(), 16, 992, False, 48,
                                       None, "v3", model.architecture))
        records.append(trajectory)
        metrics.append(status)
        assert status["success"]
    assert metrics[0]["virtual_time_s"] == metrics[1]["virtual_time_s"]
    assert len(records[0]) == len(records[1])
    for a,b in zip(*records):
        assert np.array_equal(a["features"], b["features"])
        assert np.array_equal(a["context"], b["context"])
        assert all(a[key] == b[key] for key in ("action", "teacher", "log_prob", "value", "reward"))


def test_explicit_mlp_attention_cli_initialization_keeps_checkpoint_rng_identical(tmp_path):
    source = tmp_path / "source.pt"
    torch.save(checkpoint(CandidateActorCritic(16,60)), source)
    initial = []
    for name in ("mlp", "attention"):
        output = tmp_path / name
        args = ["--output",str(output),"--device","cpu","--hidden","16", "--seed","9112037",
                "--feature-version","v3","--architecture",name,"--initialize-from",str(source),
                "--scenario-start","110632","--updates","0","--workers","0","--max-wall-s","30"]
        assert main(args) == 0
        initial.append(torch.load(output/"initialized.pt",weights_only=False))
    assert torch.equal(initial[0]["torch_rng"],initial[1]["torch_rng"])
    assert initial[0]["python_rng"] == initial[1]["python_rng"]
    assert np.array_equal(initial[0]["numpy_rng"][1],initial[1]["numpy_rng"][1])
    assert all(torch.equal(value,initial[1]["model"][key]) for key,value in initial[0]["model"].items())


def test_old_and_forked_constructors_already_match_actual_numpy_shuffle_and_python_action_seeds(monkeypatch):
    """Replay the pre-fork constructor and call actual update(), without a world.

    The old extra Torch draws change generic Torch state, but do not by themselves
    change Python task seeds or NumPy's optimizer shuffle in the shipped trainer.
    """
    torch.set_num_threads(1)
    original_permutation = np.random.permutation
    results = []
    for kind in ("mlp", "old_attention", "forked_attention"):
        torch.manual_seed(9112037); np.random.seed(9112037); random.seed(9112037)
        if kind == "forked_attention":
            model = CandidateActorCritic(16,60,architecture_spec("attention"))
        else:
            model = CandidateActorCritic(16,60)
            if kind == "old_attention":
                model.relations = torch.nn.ModuleList([GatedSetAttention(16,4)])
                model.architecture = architecture_spec("attention")
        state = torch.get_rng_state().clone()
        action_seeds = [random.randrange(2**31) for _ in range(32)]
        permutations = []
        def capture(count):
            result = original_permutation(count)
            permutations.append(result.tolist())
            return result
        monkeypatch.setattr(np.random,"permutation",capture)
        records = [dict(features=np.full((2,60),i/10,dtype=np.float32),
                        context=np.zeros(CONTEXT_DIM,dtype=np.float32),teacher=0,return_=1.)
                   for i in range(6)]
        for record in records:
            record["return"] = record.pop("return_")
        args = Namespace(bc_epochs=2,minibatch=3,value_coef=.5,max_grad_norm=.5)
        actual = update(model,torch.optim.Adam(model.parameters(),lr=1e-4),records,args,bc=True)
        assert actual["optimizer_steps"] == 4 and len(permutations) == 2
        results.append((state,action_seeds,permutations))
    assert not torch.equal(results[0][0],results[1][0])
    assert torch.equal(results[0][0],results[2][0])
    assert all(item[1:] == results[0][1:] for item in results[1:])
