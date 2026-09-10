"""Analytic policy-gradient, baseline independence and real training checks."""

from argparse import Namespace
import copy
import time
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research_rl.controller import CONTEXT_DIM
from research_rl.network import CandidateActorCritic, pack_observations
from research_rl.train_paired import (TRAINER, legal_seed, main, paired_episode,
                                     penalized_cost, reinforce_loss, update)


def test_expected_reinforce_gradient_matches_exact_variable_length_tree():
    """Three leaves, two trainable decisions and path lengths 1/2/2."""
    theta = torch.tensor([0.3, -0.7], dtype=torch.float64, requires_grad=True)
    p, q = theta.sigmoid()
    probabilities = torch.stack((1 - p, p * (1 - q), p * q))
    costs = torch.tensor([2., 5., 9.], dtype=torch.float64)
    exact = torch.autograd.grad((probabilities * costs).sum(), theta, retain_graph=True)[0]
    estimates = []
    for probability, cost, logs in zip(probabilities, costs,
            ([torch.log1p(-p)], [p.log(), torch.log1p(-q)], [p.log(), q.log()])):
        logs = torch.stack(logs)
        loss = reinforce_loss(logs, torch.ones_like(logs) * (4.2 - cost),
                              torch.zeros_like(logs), torch.ones_like(logs) * len(logs), 1, 0)
        estimates.append(probability.detach() * loss)
    estimated = torch.autograd.grad(sum(estimates), theta)[0]
    assert torch.allclose(estimated, exact, atol=1e-12)


def make_records(model):
    rows = []
    for count, advantage in ((1, .3), (3, -.4)):
        for step in range(count):
            rows.append(dict(features=np.full((2, 24), .1 + step, dtype=np.float32),
                             context=np.arange(CONTEXT_DIM, dtype=np.float32) / 20,
                             action=step % 2, advantage=advantage, trajectory_length=count))
            rows[-1]["features"][1, 0] += .4
    with torch.no_grad():
        logits, _ = model(*pack_observations(rows))
        dist = torch.distributions.Categorical(logits=logits)
        old = dist.log_prob(torch.tensor([r["action"] for r in rows]))
    for row, value in zip(rows, old.tolist()):
        row["log_prob"] = value
    return rows


def test_gradient_accumulation_does_not_reweight_long_episodes():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    first = CandidateActorCritic(16, 24)
    second = copy.deepcopy(first)
    rows = make_records(first)
    args = Namespace(minibatch=1, entropy_coef=.01, max_grad_norm=1e8)
    one = update(first, torch.optim.SGD(first.parameters(), lr=.1), rows, 2, args, time.monotonic() + 30)
    args.minibatch = 100
    all_at_once = update(second, torch.optim.SGD(second.parameters(), lr=.1), rows, 2, args, time.monotonic() + 30)
    assert one["optimizer_steps"] == all_at_once["optimizer_steps"] == 1
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.allclose(a, b, atol=1e-7)


def test_expired_update_has_no_parameter_change_or_optimizer_step():
    model = CandidateActorCritic(16, 24)
    rows = make_records(model)
    before = copy.deepcopy(model.state_dict())
    result = update(model, torch.optim.Adam(model.parameters()), rows, 2,
                    Namespace(minibatch=2), 0)
    assert result["optimizer_steps"] == 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_greedy_baseline_is_independent_of_sampled_action_rng_and_fallback_is_charged():
    model = CandidateActorCritic(16, 24)
    outputs = [paired_episode((100025, model.state_dict(), 16, seed, "v1", 1,
                               time.time() + 60)) for seed in (12, 93)]
    assert outputs[0][1]["baseline"]["cost_s"] == outputs[1][1]["baseline"]["cost_s"]
    for rows, metrics in outputs:
        assert metrics["sampled"]["success"] and metrics["baseline"]["success"]
        assert len(rows) == 1
        assert metrics["sampled"]["fallback_time_s"] > 0
        assert rows[0]["advantage"] == pytest.approx(
            (metrics["baseline"]["cost_s"] - metrics["sampled"]["cost_s"]) / 1000)


def test_failed_early_exit_cannot_obtain_a_cheap_reward():
    report = SimpleNamespace(virtual_time_s=1, completion_certified_under_model=False,
                             error=None, exit_error=None)
    cost, success = penalized_cost(report, {"all_cleared": False})
    assert cost == 360001 and not success


@pytest.mark.parametrize("seed", [6000, 6095, 800000, 810000, 900000, 100000])
def test_training_rejects_nontraining_seeds(seed):
    assert not legal_seed(seed)
    with pytest.raises(ValueError, match="training ranges"):
        paired_episode((seed, {}, 16, 0, "v1", 1, time.time() + 60))


def test_actual_training_checkpoint_and_resume(tmp_path):
    directory = tmp_path / "trial"
    args = ["--output", str(directory), "--device", "cpu", "--hidden", "16",
            "--feature-version", "v1", "--workers", "0", "--pairs-per-update", "2",
            "--max-decisions", "1", "--updates", "1", "--max-wall-s", "60",
            "--deadline-utc", "2099-01-01T00:00:00+00:00"]
    assert main(args) == 0
    latest = torch.load(directory / "latest.pt", weights_only=False)
    initial = torch.load(directory / "random.pt", weights_only=False)
    assert latest["state"]["trainer"] == TRAINER
    assert latest["state"]["optimizer_steps"] == latest["state"]["update"] == 1
    assert any(not torch.equal(latest["model"][k], v) for k, v in initial["model"].items())
    with pytest.raises(SystemExit):
        main(args)
    args[args.index("--updates") + 1] = "2"
    assert main(args + ["--resume", str(directory / "latest.pt")]) == 0
    resumed = torch.load(directory / "latest.pt", weights_only=False)
    assert resumed["state"]["update"] == resumed["state"]["optimizer_steps"] == 2
    assert resumed["state"]["episodes"] == resumed["state"]["paired_baseline_episodes"] == 4
    assert resumed["state"]["next_seed"] == 100005
