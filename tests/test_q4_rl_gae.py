"""Actor-only GAE arithmetic, unchanged MC critic and episode transactions."""
from copy import deepcopy
import math

import pytest

torch = pytest.importorskip("torch")

from q4_rl.advantages import attach_actor_advantages, validate_gae_lambda
from q4_rl.train import attach_returns
from q4_rl.training_journal import read_batch
from tests.test_q4_rl_learner_threads import api, single_thread_runtime, assert_exact, checkpoint, args_for


def episode(success=True):
    rows = [dict(cost_s=1000., value=.4, terminal=False, log_prob=-.2, action_index=0),
            dict(cost_s=2000., value=.5, terminal=True, fallback_cost_s=1500., log_prob=-.8, action_index=1)]
    attach_returns(rows, actual_time_s=3000., success=success)
    return rows


@pytest.mark.parametrize("success,expected", [(True, [-2.15, -2.5]), (False, [-180.65, -359.5])])
def test_actual_fallback_failure_and_terminal_zero_arithmetic(success, expected):
    rows = episode(success)
    before = deepcopy(rows)
    attach_actor_advantages(rows, .5)
    assert [row["actor_advantage"] for row in rows] == pytest.approx(expected)
    for old, new in zip(before, rows):
        assert old == {key: value for key, value in new.items() if key != "actor_advantage"}
    # Fallback is already part of cost_s and was not charged a second time.
    assert sum(row["cost_s"] for row in rows) == 3000.


def test_lambda_zero_is_one_step_and_separate_episodes_never_bootstrap_each_other():
    first, second = episode(False), episode()
    attach_actor_advantages(first, 0.)
    assert [row["actor_advantage"] for row in first] == pytest.approx([-.9, -359.5])
    attach_actor_advantages(second, .97)
    independent = deepcopy(second)
    attach_actor_advantages(first, .97)
    assert second == independent
    with pytest.raises(ValueError, match="exactly one"):
        attach_actor_advantages(first+second, .97)
    second[-1]["terminal"] = False
    with pytest.raises(ValueError, match="terminal"):
        attach_actor_advantages(second, .97)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.01, 1.01, True, ".97"])
def test_invalid_lambda_is_rejected(value):
    with pytest.raises(ValueError, match="lambda"):
        validate_gae_lambda(value)


def test_lambda_one_leaves_rows_and_default_driver_exact(api, tmp_path, monkeypatch):
    train, fixture = api
    rows = episode(False)
    before = deepcopy(rows)
    attach_actor_advantages(rows, 1.)
    assert rows == before
    monkeypatch.setattr(train, "rollout", fixture.fake_rollout)
    old, explicit = tmp_path/"default", tmp_path/"explicit"
    train.main(fixture.arguments(old))
    train.main(fixture.arguments(explicit)+["--gae-lambda", "1"])
    a, b = checkpoint(old), checkpoint(explicit)
    for key in ("model", "optimizer", "rng", "config", "objective"):
        assert_exact(a[key], b[key])
    assert "gae_lambda" not in b["config"]
    assert b["objective"] == train.OBJECTIVE


def test_lambda_one_update_matches_exact_legacy_float32_subtraction(api):
    train, fixture = api
    model = fixture.network.make_model(hidden=16)
    records = fixture.records_for(model)
    initial = deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    legacy_advantages = (torch.tensor([row["return"] for row in records], dtype=torch.float32)
                         - torch.tensor([row["value"] for row in records], dtype=torch.float32)).tolist()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    attach_actor_advantages(records, 1.)
    train.ppo_update(model, optimizer, records, epochs=1, minibatch_size=2)
    expected_model, expected_optimizer, expected_rng = (
        deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), torch.get_rng_state().clone())
    model.load_state_dict(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    torch.set_rng_state(rng)
    train.ppo_update(model, optimizer, records, epochs=1, minibatch_size=2, actor_advantages=legacy_advantages)
    assert_exact(expected_model, model.state_dict())
    assert_exact(expected_optimizer, optimizer.state_dict())
    assert torch.equal(expected_rng, torch.get_rng_state())


def test_actor_update_keeps_mc_value_targets_and_complete_rollback(api, tmp_path, monkeypatch):
    train, fixture = api
    seen = []
    original_records = []
    def rollout(task):
        seen.append((task["seed"], task["action_seed"]))
        result = fixture.fake_rollout(task)
        original_records.append(deepcopy(result["records"]))
        return result
    monkeypatch.setattr(train, "rollout", rollout)
    real_update = train.ppo_update
    target_values = []
    real_loss = torch.nn.functional.smooth_l1_loss
    def value_loss(values, targets, *args, **kwargs):
        target_values.extend(targets.tolist())
        return real_loss(values, targets, *args, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "smooth_l1_loss", value_loss)
    def interrupted(model, optimizer, records, **kwargs):
        assert kwargs["actor_advantages"] == [row["actor_advantage"] for row in records]
        result = real_update(model, optimizer, records, **kwargs)
        assert result["updates"] > 0 and math.isfinite(result["value_loss"])
        expected = sorted(float(torch.tensor(row["return"])) for row in records)
        assert sorted(target_values) == expected
        raise train.TrainingStop("after complete actor GAE update")
    monkeypatch.setattr(train, "ppo_update", interrupted)
    output = tmp_path/"resume"
    args = args_for(fixture, output, 2, gae_lambda=.97)
    stopped = train.main(args)
    assert stopped["episodes"] == 0 and stopped["pending_batch"] is not None
    assert torch.get_num_threads() == 1
    initial = torch.load(output/"random.pt", weights_only=True)
    saved = checkpoint(output)
    for key in ("model", "optimizer"):
        assert_exact(initial[key], saved[key])
    assert saved["objective"] == {**train.OBJECTIVE, "lambda": .97,
        "actor_advantage": "gae", "critic_target": "undiscounted_mc"}
    raw = output/"batch-000000-attempt-000000.json.gz"
    original_bytes = raw.read_bytes()
    for original, logged in zip(original_records, read_batch(raw)):
        expected = deepcopy(original)
        attach_actor_advantages(expected, .97)
        assert logged["records"] == expected
    bad = deepcopy(saved)
    bad["objective"]["lambda"] = 1.
    torch.save(bad, tmp_path/"mismatch.pt")
    with pytest.raises(ValueError, match="objective"):
        train.restore_checkpoint(tmp_path/"mismatch.pt")
    with pytest.raises(SystemExit):
        train.main(args+["--gae-lambda", "1", "--resume", str(output/"latest.pt")])
    monkeypatch.setattr(train, "ppo_update", real_update)
    train.main(args+["--resume", str(output/"latest.pt")])
    n = len(seen)//2
    assert seen[:n] == seen[n:]
    assert raw.read_bytes() == original_bytes
    control = tmp_path/"control"
    train.main(args_for(fixture, control, 2, gae_lambda=.97))
    a, b = checkpoint(output), checkpoint(control)
    for key in ("model", "optimizer", "rng", "config", "objective"):
        assert_exact(a[key], b[key])


def test_bad_episode_is_preserved_before_advantage_validation_raises(api, tmp_path, monkeypatch):
    train, fixture = api
    def rollout(task):
        row = fixture.fake_rollout(task)
        row["records"][-1]["terminal"] = False
        return row
    monkeypatch.setattr(train, "rollout", rollout)
    with pytest.raises(ValueError, match="terminal"):
        train.main(args_for(fixture, tmp_path, gae_lambda=.97))
    assert len(read_batch(tmp_path/"batch-000000-attempt-000000.json.gz")) == 1
    assert checkpoint(tmp_path)["state"]["episodes"] == 0


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1"])
def test_cli_rejects_invalid_lambda_before_work(api, tmp_path, value):
    train, fixture = api
    with pytest.raises(SystemExit):
        train._arguments(fixture.arguments(tmp_path)+["--gae-lambda", value])
