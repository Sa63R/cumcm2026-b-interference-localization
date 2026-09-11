"""Training contracts: masking, exact total cost, PPO, and checkpoint recovery."""
import random

import pytest

torch = pytest.importorskip("torch")

from q4_rl.network import CandidateActorCritic, TorchPolicy, configure_cpu, pack_observations, load_policy
from q4_rl.train import (attach_returns, undiscounted_returns, imitation_update, ppo_update,
                         restore_checkpoint, save_checkpoint, training_case_spec, TrainingStop,
                         summarize_training_metrics)


@pytest.fixture(autouse=True)
def cpu():
    configure_cpu()


def observation(n=3):
    return {"global_features": [0.1 * i for i in range(10)],
            "candidate_features": [[(i + j) / 20 for j in range(16)] for i in range(n)]}


def test_masked_padding_does_not_change_policy_or_value():
    torch.manual_seed(1)
    model = CandidateActorCritic(hidden=16)
    single = model(*pack_observations([observation(2)]))
    batched = model(*pack_observations([observation(2), observation(7)]))
    assert torch.allclose(single[0][0], batched[0][0, :2], atol=1e-7)
    assert torch.allclose(single[1][0], batched[1][0], atol=1e-7)
    assert torch.isneginf(batched[0][0, 2:]).all()


def test_candidate_permutation_equivariance_and_invariant_critic():
    torch.manual_seed(2)
    model = CandidateActorCritic(hidden=16)
    row = observation(4)
    order = [2, 0, 3, 1]
    permuted = {**row, "candidate_features": [row["candidate_features"][i] for i in order]}
    logits, value = model(*pack_observations([row]))
    other_logits, other_value = model(*pack_observations([permuted]))
    assert torch.allclose(logits[0, order], other_logits[0], atol=1e-7)
    assert torch.allclose(value, other_value, atol=1e-7)


@pytest.mark.parametrize("costs", [[1., 2., 3.], [0., 600.], [600.]])
def test_undiscounted_cost_is_independent_of_number_of_decisions(costs):
    returns = undiscounted_returns(costs)
    assert returns[0] == pytest.approx(-sum(costs) / 1000)


def test_failure_penalty_preserves_actual_cost_and_full_fallback():
    records = [{"cost_s": 70.}, {"cost_s": 1930., "fallback_cost_s": 1800.}]
    result = attach_returns(records, actual_time_s=2000., success=False)
    assert records[0]["return"] == -360.
    assert records[-1]["terminal_penalty_s"] == 358000.
    assert sum(row["cost_s"] for row in records) == 2000.
    assert result["penalized_time_s"] == 360000.
    with pytest.raises(ValueError, match="complete episode"):
        attach_returns(records, actual_time_s=2001., success=True)


def test_invalid_or_empty_observation_rejected():
    with pytest.raises(ValueError):
        pack_observations([observation(0)])
    row = observation()
    row["global_features"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        pack_observations([row])


def make_records(model):
    policy = TorchPolicy(model, deterministic=False)
    for i in range(6):
        row = observation(2 + i % 3)
        policy(row["global_features"], row["candidate_features"])
    records = policy.records
    for i, row in enumerate(records):
        row["cost_s"] = 20. + i * 7.
    attach_returns(records, actual_time_s=sum(row["cost_s"] for row in records), success=True)
    return records


def test_real_ppo_update_changes_parameters_with_finite_loss():
    torch.manual_seed(3)
    model = CandidateActorCritic(hidden=16)
    records = make_records(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = ppo_update(model, optimizer, records, epochs=2, minibatch_size=3)
    assert result["updates"] == 4
    assert all(torch.isfinite(value).all() for value in model.parameters())
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_checkpoint_roundtrip_restores_optimizer_rng_and_cursor(tmp_path):
    torch.manual_seed(4)
    random.seed(4)
    model = CandidateActorCritic(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    records = make_records(model)
    ppo_update(model, optimizer, records, epochs=1, minibatch_size=3)
    path = tmp_path / "latest.pt"
    state = {"next_seed": 8000016, "wall_time_s": 12.5,
             "pending_batch": {"seeds": [8000014, 8000015], "action_seeds": [71, 72]}}
    save_checkpoint(path, model, optimizer, state, {"learning_rate": 0.001})
    expected_random, expected_torch = random.random(), torch.rand(4)
    restored, restored_optimizer, restored_state, _ = restore_checkpoint(path)
    assert random.random() == expected_random
    assert torch.equal(torch.rand(4), expected_torch)
    assert restored_state == state
    assert restored_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    assert load_policy(path)(**observation(2)) in (0, 1)


def test_resume_reproduces_next_gradient_update(tmp_path):
    torch.manual_seed(5)
    random.seed(5)
    model = CandidateActorCritic(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    records = make_records(model)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, model, optimizer, {"next_seed": 8000006}, {"learning_rate": 0.001})
    ppo_update(model, optimizer, records, epochs=2, minibatch_size=2)
    restored, restored_optimizer, _, _ = restore_checkpoint(path)
    ppo_update(restored, restored_optimizer, records, epochs=2, minibatch_size=2)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())


def test_training_partition_and_balanced_modes_are_fixed():
    assert training_case_spec(8000000) == {"family": "random", "source_mode": "mixed"}
    assert training_case_spec(8000008) == {"family": "random", "source_mode": "all_directional"}
    for seed in (7999999, 8100000, 8200000, 8300000):
        with pytest.raises(ValueError):
            training_case_spec(seed)


def test_administrative_stop_before_update_changes_no_parameters():
    model = CandidateActorCritic(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    records = make_records(model)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(TrainingStop):
        imitation_update(model, optimizer, records, stop_check=lambda: True)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_training_summary_keeps_failures_and_distinguishes_ratio_aggregation():
    runtime = {name: 0.1 for name in ("wall_time_s", "worker_cpu_s", "policy_wall_s", "policy_cpu_s",
                                     "posthoc_bound_wall_s", "posthoc_bound_cpu_s")}
    metrics = [dict(seed=8000000, actual_time_s=100., penalized_time_s=100.,
                    common_lower_bound_s=20., success=True, failed_clear_count=2, **runtime),
               dict(seed=8000001, actual_time_s=200., penalized_time_s=360000.,
                    common_lower_bound_s=80., success=False, failed_clear_count=3, **runtime)]
    before_rng = random.getstate()
    result = summarize_training_metrics(metrics)
    assert random.getstate() == before_rng
    assert result["full_clear_rate"] == 0.5
    assert result["mean_penalized_time_s"] == 180050.
    assert result["ratio_of_mean_penalized_time_to_mean_bound"] == 3601.
    assert result["mean_of_penalized_ratios"] == 2252.5
    assert result["all_clear_actual_time_over_mean_bound"] is None
    assert result["case_ratios"][1]["actual_time_over_lower_bound"] is None
    assert result["failed_clear_count"] == 5
    assert result["p95_actual_time_s"] == 195.
    assert result["sum_episode_worker_cpu_s"] == pytest.approx(0.2)
