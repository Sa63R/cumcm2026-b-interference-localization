"""Micro schema isolation, shared learning rules and full transaction recovery."""
import gzip
import json
from pathlib import Path
import random

import pytest

torch = pytest.importorskip("torch")

from q4_rl import network as macro_network
from q4_rl import train as macro_train
from q4_rl import micro_network as network
from q4_rl import micro_train as train


@pytest.fixture(autouse=True)
def cpu():
    network.configure_cpu()


def observation(n=3):
    return {"global_features": [.01*i for i in range(13)],
            "candidate_features": [[(i+j)/100 for j in range(50)] for i in range(n)]}


def records_for(model):
    policy = network.TorchPolicy(model, deterministic=False)
    for n in (2, 3, 4, 2):
        policy(**observation(n))
    for i, row in enumerate(policy.records):
        row.update(cost_s=10.+i*17., fallback_cost_s=50. if i == 3 else 0.,
                   terminal=i == 3, action_kind="measure")
    train.attach_returns(policy.records, actual_time_s=sum(r["cost_s"] for r in policy.records), success=True)
    return policy.records


def test_micro_masks_and_candidate_permutation_use_13_by_50_features():
    model = network.MicroCandidateActorCritic(hidden=16)
    row = observation(3)
    order = [2, 0, 1]
    permuted = {**row, "candidate_features": [row["candidate_features"][i] for i in order]}
    logits, value = model(*network.pack_observations([row]))
    batched_logits, batched_value = model(*network.pack_observations([permuted, observation(7)]))
    assert torch.allclose(logits[0, order], batched_logits[0, :3], atol=1e-7)
    assert torch.allclose(value[0], batched_value[0], atol=1e-7)
    assert torch.isneginf(batched_logits[0, 3:]).all()
    assert network.feature_schema()["version"] == "q4-micro-g1-v1"
    assert network.feature_schema()["candidate_features"][5] == "immediate_cost_upper"


def test_shared_learning_and_statistical_functions_are_not_forked():
    for name in ("ppo_update", "imitation_update", "attach_returns", "training_case_spec", "summarize_training_metrics"):
        assert getattr(train, name) is getattr(macro_train, name)
    with pytest.raises(ValueError, match="partition"):
        train.training_case_spec(8100000)
    rows = [{"cost_s": 50.}, {"cost_s": 1950., "fallback_cost_s": 1800.}]
    result = train.attach_returns(rows, actual_time_s=2000., success=False)
    assert rows[0]["return"] == -360.
    assert rows[-1]["terminal_penalty_s"] == 358000.
    assert result["penalized_time_s"] == 360000.
    assert sum(r["cost_s"] for r in rows) == 2000.


def test_micro_checkpoint_rejects_macro_both_directions_and_semantic_spoof(tmp_path):
    macro = macro_network.CandidateActorCritic(hidden=16)
    macro_path = tmp_path/"macro.pt"
    macro_train.save_checkpoint(macro_path, macro, torch.optim.Adam(macro.parameters()), {}, {"learning_rate": .001})
    for loader in (network.load_policy, train.restore_checkpoint):
        with pytest.raises(ValueError, match="micro checkpoint"):
            loader(macro_path)
    micro = network.MicroCandidateActorCritic(hidden=16)
    micro_path = tmp_path/"micro.pt"
    train.save_checkpoint(micro_path, micro, torch.optim.Adam(micro.parameters(), lr=.001), {}, {"learning_rate": .001})
    for loader in (macro_network.load_policy, macro_train.restore_checkpoint):
        with pytest.raises(ValueError):
            loader(micro_path)
    with pytest.raises(ValueError, match="macro"):
        train.save_checkpoint(tmp_path/"bad.pt", macro, torch.optim.Adam(macro.parameters()), {}, {"learning_rate": .001})
    saved = torch.load(micro_path, weights_only=True)
    saved["feature_schema"]["candidate_features"][5] = "minimum_immediate_cost"
    torch.save(saved, tmp_path/"spoof.pt")
    with pytest.raises(ValueError, match="semantics"):
        network.load_policy(tmp_path/"spoof.pt")
    with pytest.raises(ValueError, match="dimensions"):
        network.model_from_metadata(dict(architecture=network.ARCHITECTURE, global_dim=10, candidate_dim=16, hidden=16))


def test_save_does_not_advance_rng_and_resume_reproduces_next_update(tmp_path):
    torch.manual_seed(77)
    random.seed(77)
    model = network.MicroCandidateActorCritic(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    records = records_for(model)
    path = tmp_path/"micro.pt"
    before_rng, before_python = torch.get_rng_state().clone(), random.getstate()
    state = {"pending_batch": {"seeds": [8005001], "action_seeds": [41]}, "next_seed": 8005002}
    train.save_checkpoint(path, model, optimizer, state, {"learning_rate": .001})
    assert torch.equal(before_rng, torch.get_rng_state())
    assert before_python == random.getstate()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    update = train.ppo_update(model, optimizer, records, epochs=2, minibatch_size=2)
    assert update["updates"] == 4
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    restored, restored_optimizer, restored_state, _ = train.restore_checkpoint(path)
    train.ppo_update(restored, restored_optimizer, records, epochs=2, minibatch_size=2)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert restored_state == state
    assert network.load_policy(path)(**observation(2)) in (0, 1)
    with pytest.raises(ValueError, match="learning rate"):
        train.restore_checkpoint(path, learning_rate=.01)


def fake_rollout(task):
    """Exercise the actual actor/update/driver with no environment or hidden data."""
    model = network.model_from_metadata(task["network"])
    model.load_state_dict(task["model"])
    random.seed(task["action_seed"])
    torch.manual_seed(task["action_seed"])
    records = records_for(model)
    total = sum(r["cost_s"] for r in records)
    metrics = dict(seed=task["seed"], actual_time_s=total, penalized_time_s=total,
        common_lower_bound_s=20., success=True, failed_clear_count=0,
        **{k: .01 for k in ("wall_time_s", "worker_cpu_s", "policy_wall_s", "policy_cpu_s",
                            "posthoc_bound_wall_s", "posthoc_bound_cpu_s")})
    return dict(seed=task["seed"], action_seed=task["action_seed"], records=records, metrics=metrics,
                test_fixture=True)


def arguments(output, *, end=8005010):
    return ["--output", str(output), "--workers", "1", "--cpu-budget", "2", "--hidden", "16",
            "--warmstart-episodes", "0", "--epochs", "1", "--minibatch-size", "2",
            "--batch-episodes", "2", "--max-decisions", "8", "--scenario-start", "8005010",
            "--scenario-end", str(end), "--max-batches", "1", "--max-wall-seconds", "60",
            "--deadline", "2099-01-01T00:00:00+00:00"]


def test_main_update_interruption_rolls_back_and_replays_identically(tmp_path, monkeypatch):
    calls = []
    def tracked(task):
        calls.append((task["seed"], task["action_seed"]))
        return fake_rollout(task)
    monkeypatch.setattr(train, "rollout", tracked)
    real_update = train.ppo_update
    def interrupted(model, optimizer, records, **kwargs):
        real_update(model, optimizer, records, **kwargs)
        raise train.TrainingStop("scripted stop after parameters changed")
    monkeypatch.setattr(train, "ppo_update", interrupted)
    output = tmp_path/"resumed"
    stopped = train.main(arguments(output))
    assert stopped["episodes"] == stopped["batches"] == 0
    assert stopped["pending_batch"]["seeds"] == [8005010]
    interrupted_raw = output/"batch-000000-attempt-000000.json.gz"
    original_bytes = interrupted_raw.read_bytes()
    initial = torch.load(output/"random.pt", weights_only=True)
    reserved = torch.load(output/"latest.pt", weights_only=True)
    assert all(torch.equal(v, reserved["model"][k]) for k, v in initial["model"].items())
    monkeypatch.setattr(train, "ppo_update", real_update)
    resumed = train.main(arguments(output)+["--resume", str(output/"latest.pt")])
    assert resumed["episodes"] == 1 and resumed["pending_batch"] is None
    assert calls[0] == calls[1]
    assert interrupted_raw.read_bytes() == original_bytes
    assert (output/"batch-000000-attempt-000001.json.gz").exists()
    control = tmp_path/"control"
    train.main(arguments(control))
    a, b = (torch.load(p/"latest.pt", weights_only=True) for p in (output, control))
    assert all(torch.equal(v, b["model"][k]) for k, v in a["model"].items())
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])


def test_main_keeps_partial_raw_batch_and_reserves_all_seeds(tmp_path, monkeypatch):
    calls = []
    def sometimes_raises(task):
        calls.append(task["seed"])
        if len(calls) == 2:
            raise RuntimeError("scripted second worker failure")
        return fake_rollout(task)
    monkeypatch.setattr(train, "rollout", sometimes_raises)
    args = arguments(tmp_path, end=8005011)
    with pytest.raises(RuntimeError, match="second worker"):
        train.main(args)
    _, _, state, _ = train.restore_checkpoint(tmp_path/"latest.pt")
    assert state["episodes"] == 0
    assert state["pending_batch"]["seeds"] == [8005010, 8005011]
    raw = tmp_path/"batch-000000-attempt-000000.json.gz"
    old = raw.read_bytes()
    from q4_rl.training_journal import read_batch
    assert len(read_batch(raw)) == 1
    state = train.main(args+["--resume", str(tmp_path/"latest.pt")])
    assert state["episodes"] == 2
    assert calls == [8005010, 8005011, 8005010, 8005011]
    assert raw.read_bytes() == old
    with pytest.raises(SystemExit):
        train.main(args)  # Existing output is never overwritten as a new run.


def test_rollout_deadline_rejects_before_any_scene_creation(monkeypatch):
    from q4_rl import scenarios
    def forbidden(*args, **kwargs):
        raise AssertionError("scene constructed after expired deadline")
    monkeypatch.setattr(scenarios, "build_case", forbidden)
    result = train.rollout(dict(seed=8005000, action_seed=1, mode="ppo", deadline_epoch=0.))
    assert result["administrative_skip"] == "deadline_before_start"


@pytest.mark.parametrize("flag,value", [("--learning-rate", "nan"), ("--entropy-coefficient", "inf"),
                                       ("--scenario-start", "8100000"), ("--cpu-budget", "0")])
def test_invalid_resource_or_training_partition_is_rejected(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        train.main(arguments(tmp_path)+[flag, value])
