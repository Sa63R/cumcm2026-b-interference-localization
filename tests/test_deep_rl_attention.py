"""Architecture-only ablation: legal set inputs, strict transfer, actual learning."""

from argparse import Namespace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research_rl.controller import ALGORITHM_VERSIONS, CONTEXT_DIM, feature_schema
from research_rl.network import (CandidateActorCritic, architecture_spec,
    architecture_from_args, checkpoint_architecture, load_policy, pack_observations)
from research_rl.train import (compute_returns, episode, initialize_from, main,
    source_manifest, update, validate_resume)


def payload(model, version="v3", *, legacy=False):
    result = dict(algorithm=ALGORITHM_VERSIONS[version], hidden=model.hidden,
                  feature_schema=feature_schema(version), model=model.state_dict(),
                  source_manifest=source_manifest())
    if not legacy:
        result["architecture"] = model.architecture
    return result


def inputs():
    torch.manual_seed(910)
    return (torch.randn(2, 7, 60), torch.randn(2, CONTEXT_DIM),
            torch.tensor([[True] * 7, [True] * 4 + [False] * 3]))


@pytest.mark.parametrize("layers", [1, 2])
def test_zero_gate_transfer_preserves_legacy_predictions(layers):
    torch.set_num_threads(1)
    torch.manual_seed(911)
    previous = CandidateActorCritic(16, 60).eval()
    expanded = CandidateActorCritic(16, 60, architecture_spec("attention", layers)).eval()
    initialize_from(expanded, payload(previous, legacy=True))
    assert not any(key.startswith("relations.") for key in previous.state_dict())
    tensors = inputs()
    with torch.no_grad():
        for expected, actual in zip(previous(*tensors), expanded(*tensors)):
            assert torch.equal(expected, actual)
    assert all(block.attention_gate.item() == block.feedforward_gate.item() == 0
               for block in expanded.relations)


@pytest.mark.parametrize("layers", [1, 2])
def test_attention_padding_mask_and_candidate_permutation(layers):
    model = CandidateActorCritic(16, 60, architecture_spec("attention", layers)).eval()
    with torch.no_grad():
        for block in model.relations:
            block.attention_gate.fill_(0.3)
            block.feedforward_gate.fill_(-0.2)
    features, context, mask = inputs()
    order = [6, 2, 4, 1, 0, 5, 3]
    with torch.no_grad():
        logits, values = model(features, context, mask)
        permuted, same_values = model(features[:, order], context, mask[:, order])
        changed_padding = features.clone()
        changed_padding[~mask] = 1e6
        masked_logits, masked_values = model(changed_padding, context, mask)
        alone_logits, alone_value = model(features[1:2, :4], context[1:2], mask[1:2, :4])
    assert torch.allclose(permuted, logits[:, order], atol=1e-6, rtol=1e-5)
    assert torch.allclose(values, same_values, atol=1e-6, rtol=1e-5)
    assert torch.allclose(masked_logits, logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(masked_values, values, atol=1e-6, rtol=1e-5)
    assert torch.allclose(alone_logits, logits[1:2, :4], atol=1e-6, rtol=1e-5)
    assert torch.allclose(alone_value, values[1:2], atol=1e-6, rtol=1e-5)
    assert (logits[~mask] < -1e8).all()


def test_zero_gate_receives_gradient_then_attention_weights_learn():
    torch.manual_seed(913)
    model = CandidateActorCritic(16, 60, architecture_spec("attention"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tensors = inputs()
    gate = model.relations[0].attention_gate
    branch = model.relations[0].attention.in_proj_weight
    before = branch.detach().clone()
    logits, values = model(*tensors)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([2, 1])) + values.square().mean()
    loss.backward()
    assert torch.isfinite(gate.grad) and gate.grad.abs().item() > 1e-10
    assert torch.equal(branch.grad, torch.zeros_like(branch.grad))
    optimizer.step()
    assert gate.item() != 0
    assert torch.equal(before, branch)
    optimizer.zero_grad(set_to_none=True)
    logits, values = model(*tensors)
    (torch.nn.functional.cross_entropy(logits, torch.tensor([2, 1])) + values.square().mean()).backward()
    assert torch.isfinite(branch.grad).all() and branch.grad.abs().sum().item() > 1e-10
    optimizer.step()
    assert not torch.equal(before, branch)


def test_adding_second_layer_preserves_trained_first_layer():
    previous = CandidateActorCritic(16, 60, architecture_spec("attention", 1)).eval()
    with torch.no_grad():
        previous.relations[0].attention_gate.fill_(0.3)
        previous.relations[0].feedforward_gate.fill_(0.2)
    expanded = CandidateActorCritic(16, 60, architecture_spec("attention", 2)).eval()
    initialize_from(expanded, payload(previous))
    with torch.no_grad():
        for expected, actual in zip(previous(*inputs()), expanded(*inputs())):
            assert torch.equal(expected, actual)
    assert expanded.relations[1].attention_gate.item() == 0
    for target in (CandidateActorCritic(16, 60),
                   CandidateActorCritic(16, 60, architecture_spec("attention", heads=2))):
        with pytest.raises(ValueError, match="discard/change"):
            initialize_from(target, payload(previous))
    with pytest.raises(ValueError, match="discard/change"):
        initialize_from(previous, payload(expanded))


def test_incomplete_or_mislabeled_attention_checkpoint_is_rejected():
    model = CandidateActorCritic(16, 60, architecture_spec("attention"))
    with pytest.raises(ValueError, match="missing architecture"):
        checkpoint_architecture(payload(model, legacy=True))
    with pytest.raises(ValueError, match="contradict MLP"):
        checkpoint_architecture(dict(payload(model), architecture=architecture_spec()))
    incomplete = payload(model)
    incomplete["model"] = dict(incomplete["model"])
    del incomplete["model"]["relations.0.attention_gate"]
    with pytest.raises(ValueError, match="missing existing parameter"):
        initialize_from(CandidateActorCritic(16, 60, architecture_spec("attention", 2)), incomplete)
    two = CandidateActorCritic(16, 60, architecture_spec("attention", 2))
    with pytest.raises(ValueError, match="layer count"):
        checkpoint_architecture(dict(payload(two), architecture=architecture_spec("attention", 1)))


def test_architecture_metadata_load_resume_and_legacy_namespace(tmp_path):
    legacy_args = Namespace(feature_version="v3", hidden=16)
    assert architecture_from_args(legacy_args) == architecture_spec()
    legacy = CandidateActorCritic(16, 60)
    validate_resume(payload(legacy, legacy=True), legacy_args)
    model = CandidateActorCritic(16, 60, architecture_spec("attention"))
    path = tmp_path / "attention.pt"
    torch.save(payload(model), path)
    policy = load_policy(path)
    assert policy.architecture == model.architecture and policy.feature_version == "v3"
    args = Namespace(feature_version="v3", hidden=16, architecture="attention",
                     attention_layers=1, attention_heads=4)
    validate_resume(payload(model), args)
    with pytest.raises(ValueError, match="architecture differs"):
        validate_resume(payload(model), legacy_args)
    with pytest.raises(ValueError, match="architecture differs"):
        validate_resume(payload(legacy, legacy=True), args)


@pytest.mark.parametrize("kwargs", [dict(layers=0), dict(layers=3), dict(layers=True),
    dict(layers=1.0), dict(heads=0), dict(heads=9), dict(heads=True)])
def test_invalid_attention_architecture_rejected(kwargs):
    with pytest.raises(ValueError):
        architecture_spec("attention", **kwargs)


def test_invalid_width_and_empty_candidates_rejected():
    with pytest.raises(ValueError, match="divisible"):
        CandidateActorCritic(15, 60, architecture_spec("attention"))
    for records in ([], [dict(features=[], context=[0.0] * CONTEXT_DIM)]):
        with pytest.raises(ValueError, match="legal candidate"):
            pack_observations(records)


def test_attention_worker_reuse_and_real_joint_scan_ppo_update():
    import research_rl.train as training
    torch.set_num_threads(1)
    model = CandidateActorCritic(16, 60, architecture_spec("attention"))
    task = (100601, model.state_dict(), 16, 67, False, 20, None, "v3", model.architecture)
    training._worker_model = None
    first, first_metrics = episode(task)
    reused, reused_metrics = episode(task)
    assert first_metrics["success"] and reused_metrics["success"]
    assert first_metrics["virtual_time_s"] == reused_metrics["virtual_time_s"]
    assert first_metrics["reward_cost_s"] == pytest.approx(first_metrics["virtual_time_s"])
    assert first_metrics["learning"]["network_architecture"] == model.architecture
    assert len(first) == len(reused) == 20
    for a, b in zip(first, reused):
        for key in ("features", "context"):
            assert np.array_equal(a[key], b[key])
        for key in ("action", "teacher", "log_prob", "value", "reward"):
            assert a[key] == b[key]
    compute_returns(first)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    args = Namespace(bc_epochs=1, epochs=2, minibatch=10, clip=0.2,
                     value_coef=0.5, entropy_coef=0.01, aux_bc_coef=0,
                     max_grad_norm=0.5, target_kl=0.03)
    losses = update(model, torch.optim.Adam(model.parameters(), lr=3e-4), first, args)
    assert losses["optimizer_steps"] >= 2 and np.isfinite(losses["loss"])
    assert model.relations[0].attention_gate.item() != 0
    assert not torch.equal(before["relations.0.attention.in_proj_weight"],
                           model.state_dict()["relations.0.attention.in_proj_weight"])


def test_attention_training_checkpoint_and_resume(tmp_path):
    arguments = ["--output", str(tmp_path), "--device", "cpu", "--hidden", "16",
        "--feature-version", "v3", "--architecture", "attention", "--attention-layers", "1",
        "--scenario-start", "100611", "--bc-episodes", "2", "--bc-epochs", "1",
        "--updates", "1", "--episodes-per-update", "2", "--workers", "0",
        "--epochs", "1", "--max-wall-s", "90"]
    assert main(arguments) == 0
    checkpoint = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert checkpoint["architecture"] == architecture_spec("attention")
    assert checkpoint["state"]["update"] == 1
    assert checkpoint["state"]["optimizer_steps"] > 0
    before = checkpoint["model"]["relations.0.attention.in_proj_weight"].clone()
    assert main(arguments + ["--resume", str(tmp_path / "latest.pt"), "--updates", "2"]) == 0
    resumed = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert resumed["state"]["update"] == 2
    assert not torch.equal(before, resumed["model"]["relations.0.attention.in_proj_weight"])
    entries = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert [entry["stage"] for entry in entries] == ["bc", "ppo", "ppo"]
    assert load_policy(tmp_path / "latest.pt").architecture == architecture_spec("attention")
