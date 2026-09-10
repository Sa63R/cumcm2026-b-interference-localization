"""Meaningful optimization/resumption tests; skipped on non-training hosts."""

from argparse import Namespace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research_rl.controller import CONTEXT_DIM, FEATURE_DIM
from research_rl.network import CandidateActorCritic, pack_observations, load_policy
from research_rl.train import compute_returns, episode, main, update


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
