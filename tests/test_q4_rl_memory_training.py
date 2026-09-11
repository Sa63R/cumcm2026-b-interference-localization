"""G3 public wiring, schema isolation and transactional learning on tiny fixtures."""
import random

import pytest

torch = pytest.importorskip("torch")

from q4_rl import memory_controller as controller, memory_network as network, memory_train as train
from q4_rl import micro_controller as g1, micro_network, micro_train
from q4_rl import train as shared
from q4_rl.negative_memory import NegativeObservationMemory
from q4_rl.training_journal import read_batch
from simulator_client.state import Position
from strategies.search import _StopSearch
from tests.test_q4_rl_micro_controller import ScriptedClient


@pytest.fixture(autouse=True)
def cpu():
    network.configure_cpu()


@pytest.fixture
def compact_cover(monkeypatch):
    import planning.q4_directional_cover as cover
    monkeypatch.setattr(cover, "certified_cover_points", lambda profile:
        ((Position(100, 0), Position(200, 0)), {"passed": True, "test_only": True}))


def test_g1_actions_masks_certificates_and_first_50_features_remain_identical(compact_cover):
    for name in ("_candidates", "_can_measure", "_clear_certificate", "_terminal_gate",
                 "_execute_candidate", "_finish_with_baseline", "_resolve"):
        assert getattr(controller.Q4MemorySearch, name) is getattr(g1.Q4MicroSearch, name)
    observations = [[], []]
    def policy_for(index):
        def policy(g, candidates):
            observations[index].append((g, candidates))
            return 0
        return policy
    old = g1.Q4MicroSearch(ScriptedClient(), policy_for(0), max_decisions=64, max_expansions=0).run()
    new = controller.Q4MemorySearch(ScriptedClient(), policy_for(1), max_decisions=64, max_expansions=0).run()
    assert new.action_history == old.action_history and new.virtual_time_s == old.virtual_time_s
    assert new.completion_reason == old.completion_reason
    assert new.completion_certified_under_model == old.completion_certified_under_model
    assert len(observations[0]) == len(observations[1])
    for (old_g, old_c), (new_g, new_c) in zip(*observations):
        assert old_g == new_g and len(new_g) == 13
        assert old_c == [row[:50] for row in new_c]
        assert all(len(row) == 58 for row in new_c)
    memory = NegativeObservationMemory.replay([
        dict(action=row["action"], position=row["position"], channel=row["channel"],
             result=row["result"], accepted=True) for row in new.action_history])
    for field in ("compatible_counts", "unique_negative_counts", "presence_observed", "cleared"):
        assert new.learning["negative_memory"][field] == memory.state_summary()[field]


def test_rejected_request_does_not_update_memory_and_default_horizon_is_512(compact_cover):
    client = ScriptedClient()
    client.reject_measure = True
    search = controller.Q4MemorySearch(client, max_expansions=0)
    assert search.max_decisions == 512
    with pytest.raises(_StopSearch, match="request_rejected"):
        search._perform("measure", Position(0, 0), 1, "fixture")
    assert not search.negative_memory.history
    assert search.negative_memory.compatible.all()


def observation(n=3):
    return dict(global_features=[i/100. for i in range(13)],
        candidate_features=[[(i+j)/100. for j in range(58)] for i in range(n)])


def records_for(model):
    policy = network.TorchPolicy(model, deterministic=False)
    for n in (2, 3, 4, 2):
        policy(**observation(n))
    for i, record in enumerate(policy.records):
        record.update(cost_s=10.+i*17., fallback_cost_s=50. if i == 3 else 0.,
                      terminal=i == 3, action_kind="measure")
    train.attach_returns(policy.records, actual_time_s=sum(r["cost_s"] for r in policy.records), success=True)
    return policy.records


def test_g3_tensor_mask_permutation_and_shared_losses():
    model = network.make_model(hidden=16)
    row = observation(3)
    order = [2, 0, 1]
    other = {**row, "candidate_features": [row["candidate_features"][i] for i in order]}
    logits, value = model(*network.pack_observations([row]))
    other_logits, other_value = model(*network.pack_observations([other, observation(8)]))
    torch.testing.assert_close(logits[0, order], other_logits[0, :3])
    torch.testing.assert_close(value[0], other_value[0])
    assert torch.isneginf(other_logits[0, 3:]).all()
    for name in ("ppo_update", "imitation_update", "attach_returns", "training_case_spec", "summarize_training_metrics"):
        assert getattr(train, name) is getattr(shared, name)
    for update in (train.imitation_update, train.ppo_update):
        before = {key: value.clone() for key, value in model.state_dict().items()}
        result = update(model, torch.optim.Adam(model.parameters(), lr=.001), records_for(model),
                        epochs=1, minibatch_size=2)
        assert result["updates"] == 2
        assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_checkpoint_cross_schema_and_memory_semantics_are_strict(tmp_path):
    old = micro_network.make_model(hidden=16)
    old_path = tmp_path/"g1.pt"
    micro_train.save_checkpoint(old_path, old, torch.optim.Adam(old.parameters()), {}, {"learning_rate": .001})
    with pytest.raises(ValueError, match="G3 memory checkpoint"):
        network.load_policy(old_path)
    new = network.make_model(hidden=16)
    path = tmp_path/"g3.pt"
    train.save_checkpoint(path, new, torch.optim.Adam(new.parameters(), lr=.001), {}, {"learning_rate": .001})
    with pytest.raises(ValueError):
        micro_network.load_policy(path)
    with pytest.raises(ValueError, match="metadata"):
        train.save_checkpoint(tmp_path/"wrong.pt", old, torch.optim.Adam(old.parameters()), {}, {"learning_rate": .001})
    assert network.load_policy(path)(**observation()) in range(3)
    saved = torch.load(path, weights_only=True)
    saved["config"]["architecture"] = "induced"
    torch.save(saved, tmp_path/"mixed-config.pt")
    with pytest.raises(ValueError, match="architecture differ"):
        network.load_policy(tmp_path/"mixed-config.pt")
    saved["config"].pop("architecture")
    saved["feature_schema"]["negative_memory_version"] = "changed"
    torch.save(saved, tmp_path/"changed.pt")
    with pytest.raises(ValueError, match="schema"):
        network.load_policy(tmp_path/"changed.pt")
    with pytest.raises(SystemExit):
        train._arguments(["--output", str(tmp_path), "--initialize-micro-warmstart", str(old_path)])


def fake_rollout(task):
    model = network.model_from_metadata(task["network"])
    model.load_state_dict(task["model"])
    random.seed(task["action_seed"])
    torch.manual_seed(task["action_seed"])
    records = records_for(model)
    actual = sum(row["cost_s"] for row in records)
    metrics = dict(seed=task["seed"], actual_time_s=actual, penalized_time_s=actual,
        common_lower_bound_s=20., success=True, failed_clear_count=0,
        **{key: .01 for key in ("wall_time_s", "worker_cpu_s", "policy_wall_s", "policy_cpu_s",
                               "posthoc_bound_wall_s", "posthoc_bound_cpu_s")})
    return dict(seed=task["seed"], records=records, metrics=metrics, fixture_only=True)


def arguments(output):
    return ["--output", str(output), "--workers", "1", "--cpu-budget", "1", "--hidden", "16",
        "--epochs", "1", "--minibatch-size", "2", "--batch-episodes", "2", "--max-decisions", "8",
        "--scenario-start", "8006510", "--scenario-end", "8006511", "--max-batches", "1",
        "--max-wall-seconds", "60", "--deadline", "2099-01-01T00:00:00+00:00"]


def test_interrupted_update_restores_full_g3_transaction_and_exact_replay(tmp_path, monkeypatch):
    seen = []
    def worker(task):
        seen.append((task["seed"], task["action_seed"]))
        return fake_rollout(task)
    monkeypatch.setattr(train, "rollout", worker)
    real_update = train.ppo_update
    def interrupted(*args, **kwargs):
        real_update(*args, **kwargs)
        raise train.TrainingStop("fixture interrupt after parameters changed")
    monkeypatch.setattr(train, "ppo_update", interrupted)
    output = tmp_path/"resumed"
    state = train.main(arguments(output))
    assert state["episodes"] == 0 and state["pending_batch"]["seeds"] == [8006510, 8006511]
    old_raw = output/"batch-000000-attempt-000000.json.gz"
    old_bytes = old_raw.read_bytes()
    assert len(read_batch(old_raw)) == 2
    initial = torch.load(output/"random.pt", weights_only=True)
    reserved = torch.load(output/"latest.pt", weights_only=True)
    assert all(torch.equal(value, reserved["model"][key]) for key, value in initial["model"].items())
    monkeypatch.setattr(train, "ppo_update", real_update)
    state = train.main(arguments(output)+["--resume", str(output/"latest.pt")])
    assert state["episodes"] == 2 and state["pending_batch"] is None
    assert seen[:2] == seen[2:4] and old_raw.read_bytes() == old_bytes
    control = tmp_path/"control"
    train.main(arguments(control))
    a, b = (torch.load(folder/"latest.pt", weights_only=True) for folder in (output, control))
    assert all(torch.equal(value, b["model"][key]) for key, value in a["model"].items())
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])


def test_partial_return_preserves_raw_and_entire_reservation(tmp_path, monkeypatch):
    seen = []
    def worker(task):
        seen.append(task["seed"])
        if len(seen) == 2:
            raise RuntimeError("fixture worker failure")
        return fake_rollout(task)
    monkeypatch.setattr(train, "rollout", worker)
    with pytest.raises(RuntimeError, match="fixture worker"):
        train.main(arguments(tmp_path))
    old_raw = tmp_path/"batch-000000-attempt-000000.json.gz"
    before = old_raw.read_bytes()
    assert len(read_batch(old_raw)) == 1
    state = train.main(arguments(tmp_path)+["--resume", str(tmp_path/"latest.pt")])
    assert state["episodes"] == 2 and seen == [8006510, 8006511, 8006510, 8006511]
    assert old_raw.read_bytes() == before
