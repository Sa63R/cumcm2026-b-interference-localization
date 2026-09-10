"""Meaningful optimization/resumption tests; skipped on non-training hosts."""

from argparse import Namespace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research_rl.controller import CONTEXT_DIM, FEATURE_DIM, ALGORITHM_VERSIONS, feature_schema
from research_rl.network import CandidateActorCritic, pack_observations, load_policy
from research_rl.train import compute_returns, episode, main, update, initialize_from, validate_resume, source_manifest


def test_undiscounted_returns_and_terminal_failure_cost():
    trajectory = [dict(reward=-2.0, value=10), dict(reward=-5.0, value=20)]
    compute_returns(trajectory)
    assert [r["return"] for r in trajectory] == [-7.0, -5.0]
    assert [r["advantage"] for r in trajectory] == [-17.0, -25.0]


def test_masked_actor_is_permutation_equivariant():
    torch.manual_seed(4)
    model = CandidateActorCritic(16).eval()
    features = torch.randn(1, 4, FEATURE_DIM)
    context = torch.randn(1, CONTEXT_DIM)
    mask = torch.tensor([[True, True, True, False]])
    logits, value = model(features, context, mask)
    order = [2, 0, 3, 1]
    permuted, other_value = model(features[:, order], context, mask[:, order])
    assert torch.allclose(permuted, logits[:, order], atol=1e-6)
    assert torch.allclose(value, other_value, atol=1e-6)
    assert logits[0, 3] < -1e8


def test_episode_reward_includes_decision_limit_fallback():
    model = CandidateActorCritic(16)
    records, metrics = episode((100005, model.state_dict(), 16, 33, True, 1))
    assert metrics["success"]
    assert len(records) == 1
    assert metrics["reward_cost_s"] + metrics["initial_scan_virtual_time_s"] == pytest.approx(metrics["virtual_time_s"])
    assert metrics["learning"]["fallback_virtual_time_s"] > 0


def test_training_refuses_validation_seed():
    model = CandidateActorCritic(16)
    with pytest.raises(ValueError, match="training ranges"):
        episode((6000, model.state_dict(), 16, 33, True, 256))


def test_expired_administrative_deadline_is_excluded_from_training():
    model = CandidateActorCritic(16)
    records, metrics = episode((100007, model.state_dict(), 16, 33, True, 256, 0.0))
    assert records == []
    assert metrics["deadline_skipped"]


def test_real_ppo_update_changes_parameters():
    torch.set_num_threads(1)
    model = CandidateActorCritic(16)
    records, _ = episode((100006, model.state_dict(), 16, 34, False, 256))
    compute_returns(records)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    args = Namespace(bc_epochs=1, epochs=1, minibatch=32, clip=0.2,
                     value_coef=0.5, entropy_coef=0.01, aux_bc_coef=0,
                     max_grad_norm=0.5, target_kl=0.03)
    losses = update(model, torch.optim.Adam(model.parameters(), lr=3e-4), records, args)
    assert np.isfinite(losses["loss"])
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert losses["optimizer_steps"] > 0


def test_expired_update_does_not_claim_optimizer_steps():
    model = CandidateActorCritic(16)
    records = [dict(features=[[0.0] * FEATURE_DIM], context=[0.0] * CONTEXT_DIM,
                    action=0, teacher=0, log_prob=0.0, value=0.0, reward=-1.0)]
    compute_returns(records)
    args = Namespace(bc_epochs=1, epochs=1, minibatch=32)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    losses = update(model, torch.optim.Adam(model.parameters()), records, args, stop_at=0.0)
    assert losses["optimizer_steps"] == 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_explicit_v1_to_v2_transfer_preserves_logits_and_values():
    torch.manual_seed(19)
    previous = CandidateActorCritic(16, feature_dim=24)
    expanded = CandidateActorCritic(16, feature_dim=FEATURE_DIM)
    initialize_from(expanded, dict(algorithm=ALGORITHM_VERSIONS["v1"], model=previous.state_dict()))
    features = torch.randn(2, 7, FEATURE_DIM)
    context = torch.randn(2, CONTEXT_DIM)
    mask = torch.ones((2, 7), dtype=torch.bool)
    expected = previous(features[..., :24], context, mask)
    actual = expanded(features, context, mask)
    for first, second in zip(expected, actual):
        assert torch.allclose(first, second, atol=1e-6)


def test_legacy_v1_checkpoint_loads_24_feature_policy(tmp_path):
    previous = CandidateActorCritic(16, feature_dim=24)
    path = tmp_path / "legacy.pt"
    torch.save(dict(algorithm=ALGORITHM_VERSIONS["v1"], hidden=16,
                    model=previous.state_dict()), path)
    policy = load_policy(path)
    assert policy.feature_version == "v1"
    action, log_prob, value = policy([[0.0] * 24, [0.1] * 24], [0.0] * CONTEXT_DIM, 0)
    assert action in (0, 1)
    assert np.isfinite(log_prob) and np.isfinite(value)


def test_resume_rejects_semantic_and_source_drift():
    args = Namespace(feature_version="v2", hidden=16)
    valid = dict(algorithm=ALGORITHM_VERSIONS["v2"], hidden=16,
                 feature_schema=feature_schema("v2"), source_manifest=source_manifest())
    validate_resume(valid, args)
    with pytest.raises(ValueError, match="semantics"):
        validate_resume(dict(valid, feature_schema={}), args)
    with pytest.raises(ValueError, match="source manifest"):
        validate_resume(dict(valid, source_manifest={}), args)


def test_nonempty_output_without_resume_is_rejected_before_overwrite(tmp_path):
    (tmp_path / "random.pt").write_text("preserve this", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path), "--device", "cpu"])
    assert (tmp_path / "random.pt").read_text() == "preserve this"


def test_tiny_training_saves_distinct_ablation_files_and_resumes(tmp_path):
    arguments = ["--output", str(tmp_path), "--device", "cpu", "--hidden", "16",
                 "--bc-episodes", "2", "--bc-epochs", "1", "--updates", "1",
                 "--episodes-per-update", "2", "--workers", "0", "--epochs", "1",
                 "--max-wall-s", "60"]
    assert main(arguments) == 0
    assert (tmp_path / "random.pt").is_file()
    assert (tmp_path / "bc_only.pt").is_file()
    checkpoint = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert checkpoint["state"]["update"] == 1
    previous_seed = checkpoint["state"]["next_seed"]
    assert main(arguments + ["--resume", str(tmp_path / "latest.pt"), "--updates", "2"]) == 0
    resumed = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert resumed["state"]["update"] == 2
    assert resumed["state"]["next_seed"] > previous_seed
    entries = [json.loads(line) for line in (tmp_path / "training.jsonl").read_text().splitlines()]
    assert [entry["stage"] for entry in entries] == ["bc", "ppo", "ppo"]
    assert load_policy(tmp_path / "latest.pt") is load_policy(tmp_path / "latest.pt")
