"""Mechanism tests, not policy performance claims or held-out evaluation."""
from argparse import Namespace
import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from research_rl.controller import ALGORITHM_VERSIONS, feature_schema
from research_rl.network import CandidateActorCritic, pack_observations
from research_rl.recurrent_network import (RecurrentCandidateActorCritic, RecurrentPolicy,
    pack_episodes, initialize_from_mlp, model_from_checkpoint)
from research_rl import train_recurrent as trainer


def trajectory(length=3, offset=0):
    records = []
    for i in range(length):
        features = np.random.default_rng(offset + i).normal(size=(2 + i % 2, 60)).astype(np.float32)
        records.append(dict(features=features, context=np.full(12, i / 10, dtype=np.float32),
            action=i % 2, log_prob=-.7, value=.3, reward=-.1 * (i + 1),
            episode_start=i == 0, terminal=i == length - 1))
    return trainer.compute_gae(records)


def update_args():
    return Namespace(epochs=2, episodes_per_minibatch=2, clip=.2, value_coef=.5,
                     entropy_coef=.005, max_grad_norm=.5, target_kl=0)


def test_full_sequence_matches_causal_step_execution_and_resets():
    torch.manual_seed(8)
    model = RecurrentCandidateActorCritic(16, 8).eval()
    torch.nn.init.normal_(model.memory_readout.weight, std=.2)
    episodes = [trajectory(4), trajectory(2, 41)]
    batch = pack_episodes(episodes)
    logits, values, final = model.forward_sequence(batch["features"], batch["context"],
                                                  batch["candidate_mask"], batch["time_mask"])
    for b, records in enumerate(episodes):
        hidden = None
        for t, record in enumerate(records):
            actual, value, hidden = model.forward_step(*pack_observations([record]), hidden)
            assert torch.allclose(actual[0], logits[t, b, :len(record["features"])], atol=2e-6)
            assert torch.allclose(value[0], values[t, b], atol=2e-6)
        assert torch.allclose(hidden[0, 0], final[0, b], atol=2e-6)
    tensors = pack_observations([episodes[0][0]])
    fresh = model.forward_step(*tensors)
    reset = model.forward_step(*tensors, hidden=torch.ones(1, 1, 8), episode_start=torch.tensor([True]))
    assert all(torch.equal(x, y) for x, y in zip(fresh, reset))


def test_padding_and_candidate_order_do_not_change_real_outputs():
    torch.manual_seed(9)
    model = RecurrentCandidateActorCritic(16, 8).eval()
    torch.nn.init.normal_(model.memory_readout.weight, std=.1)
    batch = pack_episodes([trajectory(3), trajectory(1, 14)])
    before = model.forward_sequence(batch["features"], batch["context"], batch["candidate_mask"], batch["time_mask"])
    mutated = batch["features"].clone()
    mutated[~batch["candidate_mask"]] = 999
    context = batch["context"].clone()
    context[~batch["time_mask"]] = 999
    after = model.forward_sequence(mutated, context, batch["candidate_mask"], batch["time_mask"])
    valid = batch["time_mask"]
    assert torch.equal(before[0][valid], after[0][valid])
    assert torch.equal(before[1][valid], after[1][valid])
    assert torch.equal(before[2], after[2])
    order = list(reversed(range(mutated.shape[2])))
    permuted = model.forward_sequence(batch["features"][:, :, order], batch["context"],
                                      batch["candidate_mask"][:, :, order], batch["time_mask"])
    assert torch.allclose(permuted[0][valid], before[0][:, :, order][valid], atol=2e-6)
    assert torch.allclose(permuted[1][valid], before[1][valid], atol=2e-6)


def test_memory_affects_same_observation_and_gradient_crosses_earlier_steps():
    torch.manual_seed(10)
    model = RecurrentCandidateActorCritic(16, 8)
    torch.nn.init.normal_(model.memory_readout.weight, std=.4)
    sequence = trajectory(3)
    batch = pack_episodes([sequence])
    features = batch["features"].clone().requires_grad_()
    logits, values, _ = model.forward_sequence(features, batch["context"], batch["candidate_mask"], batch["time_mask"])
    values[-1].sum().backward()
    assert features.grad[0].abs().sum() > 0
    assert model.memory.weight_hh_l0.grad.abs().sum() > 0
    alternate = features.detach().clone()
    alternate[0] += 2
    altered = model.forward_sequence(alternate, batch["context"], batch["candidate_mask"], batch["time_mask"])
    assert not torch.allclose(values[-1], altered[1][-1], atol=1e-7)
    assert not torch.equal(logits[-1].softmax(-1), altered[0][-1].softmax(-1))
    # The last current observation is identical; only earlier observation differs.
    assert torch.equal(alternate[-1], features[-1])


def test_mlp_warm_start_preserves_logits_and_values_on_identical_histories():
    torch.manual_seed(11)
    original = CandidateActorCritic(16, 60)
    recurrent = RecurrentCandidateActorCritic(16, 8)
    payload = dict(algorithm=ALGORITHM_VERSIONS["v3"], hidden=16,
                   feature_schema=feature_schema("v3"), model=original.state_dict())
    initialize_from_mlp(recurrent, payload)
    hidden = None
    for record in trajectory(4):
        inputs = pack_observations([record])
        expected = original(*inputs)
        logits, values, hidden = recurrent.forward_step(*inputs, hidden)
        assert torch.equal(logits, expected[0])
        assert torch.equal(values, expected[1])


def test_terminal_gae_and_malformed_boundaries():
    records = trajectory(3)
    for r in records:
        r["value"] = 10
    trainer.compute_gae(records, gae_lambda=1)
    assert [r["return"] for r in records] == pytest.approx([-.6, -.5, -.3])
    records[1]["terminal"] = True
    with pytest.raises(ValueError, match="boundaries"):
        trainer.compute_gae(records)
    records = trajectory(2)
    records[-1]["terminal"] = False
    with pytest.raises(ValueError, match="complete episode"):
        trainer.compute_gae(records)


def test_two_real_optimizer_passes_train_memory_weights():
    torch.manual_seed(12)
    torch.set_num_threads(1)
    model = RecurrentCandidateActorCritic(16, 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.memory.weight_hh_l0.detach().clone()
    episodes = [trajectory(4), trajectory(3, 81)]
    result = trainer.update(model, optimizer, episodes, update_args())
    assert result["optimizer_steps"] == 2
    assert result["transitions"] == 7
    assert not torch.equal(model.memory.weight_hh_l0, before)
    assert model.memory_readout.weight.abs().sum() > 0
    assert np.isfinite(result["loss"])


def test_policy_reset_clears_memory():
    policy = RecurrentPolicy(RecurrentCandidateActorCritic(16, 8))
    sample = trajectory(1)[0]
    first = policy(sample["features"], sample["context"], 0)
    assert policy.hidden is not None and policy.steps == 1
    policy.reset()
    assert policy.hidden is None and policy.steps == 0
    assert policy(sample["features"], sample["context"], 0) == first


def test_real_episode_reset_deadline_and_terminal_fallback_accounting():
    model = RecurrentCandidateActorCritic(16, 8)
    task = dict(seed=1600001, weights=model.state_dict(), hidden=16, memory_hidden=8,
                action_seed=72, max_decisions=2, deadline=None)
    records, metrics = trainer.episode(task)
    assert metrics["success"] and metrics["memory_reset_verified"]
    assert len(records) == 2 and records[-1]["terminal"]
    assert metrics["reward_cost_s"] + metrics["initial_scan_virtual_time_s"] == pytest.approx(metrics["virtual_time_s"])
    records2, other = trainer.episode(task)
    assert [r["action"] for r in records2] == [r["action"] for r in records]
    assert other["virtual_time_s"] == metrics["virtual_time_s"]
    assert trainer.episode(dict(task, deadline=0))[0] == []
    with pytest.raises(ValueError, match="training-only"):
        trainer.episode(dict(task, seed=2200001))


def cli(output, updates):
    return ["--output", str(output), "--hidden", "16", "--memory-hidden", "8", "--workers", "0",
            "--updates", str(updates), "--episodes-per-update", "2", "--episodes-per-minibatch", "2",
            "--epochs", "2", "--max-decisions", "2", "--max-attempted-episodes", "4",
            "--scenario-start", "1600001", "--scenario-end", "1600004", "--max-wall-s", "60"]


def test_real_updates_resume_equal_uninterrupted_weights_and_budget(tmp_path):
    first = tmp_path / "resumed"
    continuous = tmp_path / "continuous"
    assert trainer.main(cli(first, 1)) == 0
    initial = torch.load(first / "latest.pt", weights_only=False)
    assert initial["state"]["update"] == 1
    assert trainer.main(cli(first, 2) + ["--resume", str(first / "latest.pt")]) == 0
    assert trainer.main(cli(continuous, 2)) == 0
    a = torch.load(first / "latest.pt", weights_only=False)
    b = torch.load(continuous / "latest.pt", weights_only=False)
    assert a["state"]["attempted_episodes"] == a["state"]["episodes"] == 4
    assert a["state"]["next_seed"] == 1600005
    assert a["state"]["stop_reason"] == "attempted_episode_limit"
    assert a["state"]["optimizer_steps"] == 4
    assert all(torch.equal(a["model"][key], value) for key, value in b["model"].items())
    assert model_from_checkpoint(a).memory_hidden == 8
    bad = dict(a, memory_spec={})
    with pytest.raises(ValueError, match="semantics"):
        model_from_checkpoint(bad)
    assert trainer.main(cli(first, 2) + ["--resume", str(first / "latest.pt")]) == 0
    assert torch.load(first / "latest.pt", weights_only=False)["state"]["attempted_episodes"] == 4


def test_spawn_sampling_matches_inline_rng_and_weights(tmp_path):
    inline, spawned = tmp_path / "inline", tmp_path / "spawned"
    trainer.main(cli(inline, 1))
    trainer.main(cli(spawned, 1) + ["--workers", "2"])
    a = torch.load(inline / "latest.pt", weights_only=False)
    b = torch.load(spawned / "latest.pt", weights_only=False)
    assert a["state"]["attempted_episodes"] == b["state"]["attempted_episodes"] == 2
    assert all(torch.equal(a["model"][key], value) for key, value in b["model"].items())
    assert np.array_equal(a["numpy_rng"][1], b["numpy_rng"][1])
    assert a["python_rng"] == b["python_rng"]


def test_full_real_episode_update_and_fresh_evaluator_load(tmp_path):
    from research_rl.recurrent import run_recurrent_search
    from simulation import LocalResearchSimulator, random_scenario
    model = RecurrentCandidateActorCritic(16, 8)
    task = dict(seed=1600021, weights=model.state_dict(), hidden=16, memory_hidden=8,
                action_seed=72, max_decisions=256, deadline=None)
    records, metrics = trainer.episode(task)
    assert metrics["success"] and len(records) > 2
    trainer.compute_gae(records)
    result = trainer.update(model, torch.optim.Adam(model.parameters()), [records], update_args())
    assert result["optimizer_steps"] == 2 and result["transitions"] == len(records)
    # The shared evaluator callback can reset a reused policy for each episode.
    policy = RecurrentPolicy(model.eval())
    outcomes = []
    for _ in range(2):
        simulator = LocalResearchSimulator(random_scenario(3, 1600021))
        report = run_recurrent_search(simulator.client(), policy=policy, max_decisions=2)
        outcomes.append(report.virtual_time_s)
        assert policy.hidden is None and report.learning["memory_reset_at_episode_start"]
    assert outcomes[0] == outcomes[1]


def test_soft_interrupt_saves_completed_optimizer_count(tmp_path, monkeypatch):
    original = trainer.update

    def interrupted(*args, **kwargs):
        callback = kwargs["on_optimizer_step"]
        def stop_after_step():
            callback()
            raise KeyboardInterrupt("test soft interrupt after committed step")
        kwargs["on_optimizer_step"] = stop_after_step
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "update", interrupted)
    with pytest.raises(KeyboardInterrupt):
        trainer.main(cli(tmp_path, 1))
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["state"]["optimizer_steps"] == 1
    assert saved["state"]["stop_reason"] == "interrupted"
    assert saved["state"]["attempted_episodes"] == 2


def test_sampling_reservation_is_durable_before_dispatch_and_not_reused(tmp_path, monkeypatch):
    original = trainer.episode

    def interrupt_first_task(task):
        # Inspect the on-disk checkpoint before any episode returns. Even a
        # hard kill at this point must retain the entire dispatched reservation.
        saved = torch.load(tmp_path / "latest.pt", weights_only=False)
        assert saved["state"]["attempted_episodes"] == 2
        assert saved["state"]["next_seed"] == 1600003
        assert saved["state"]["episodes"] == 0
        raise KeyboardInterrupt("test interrupt during sampling")

    monkeypatch.setattr(trainer, "episode", interrupt_first_task)
    with pytest.raises(KeyboardInterrupt):
        trainer.main(cli(tmp_path, 1))
    seeds = []

    def record_seed(task):
        seeds.append(task["seed"])
        return original(task)

    monkeypatch.setattr(trainer, "episode", record_seed)
    trainer.main(cli(tmp_path, 1) + ["--resume", str(tmp_path / "latest.pt")])
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert seeds == [1600003, 1600004]
    assert saved["state"]["attempted_episodes"] == 4
    assert saved["state"]["episodes"] == 2
    assert saved["state"]["next_seed"] == 1600005
    assert saved["state"]["stop_reason"] == "attempted_episode_limit"


def test_zero_update_initialization_and_runner_cli_aliases(tmp_path, monkeypatch):
    def no_sampling(task):
        raise AssertionError("Zero-update initialization must not generate a scenario")

    monkeypatch.setattr(trainer, "episode", no_sampling)
    args = ["--output", str(tmp_path), "--updates", "0", "--hidden", "16", "--memory-hidden", "8",
            "--scenario-range", "1600001", "1600004", "--max-attempted-episodes", "4",
            "--max-wall-seconds", "60", "--checkpoint-interval", "10"]
    assert trainer.main(args) == 0
    saved = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert saved["state"]["attempted_episodes"] == saved["state"]["optimizer_steps"] == 0
    assert saved["state"]["stop_reason"] == "update_limit"
    assert saved["args"]["scenario_start"] == 1600001
    assert saved["args"]["scenario_end"] == 1600004
    assert saved["args"]["max_wall_s"] == 60 and saved["args"]["checkpoint_seconds"] == 10
    assert trainer.main(args + ["--resume", str(tmp_path / "latest.pt")]) == 0
