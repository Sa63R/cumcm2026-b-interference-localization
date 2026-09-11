"""CPU training contracts on numeric fixtures and actual scripted receipts.

These tests assess wiring, billing and recovery, never research performance.
No scenario rollout, validation database or official simulator is used.
"""
import copy
import math
import os
import random
import time

import pytest

torch = pytest.importorskip("torch")

from q4_rl import bundle_controller as controller, bundle_network as network, bundle_train as train
from q4_rl import train as shared
from q4_rl.training_journal import read_batch
from simulator_client.state import Position
from tests.test_q4_rl_micro_controller import ScriptedClient


@pytest.fixture(autouse=True)
def cpu():
    network.configure_cpu()


@pytest.fixture
def compact_cover(monkeypatch):
    import planning.q4_directional_cover as cover
    monkeypatch.setattr(cover, "certified_cover_points", lambda profile:
        ((Position(100, 0), Position(200, 0)), {"passed": True, "test_only": True}))


def observation(n=3):
    return dict(global_features=[i/100. for i in range(10)],
        candidate_features=[[(i+j)/100. for j in range(18)] for i in range(n)])


def records_for(model):
    policy = network.TorchPolicy(model, deterministic=False)
    for n in (2, 3, 4, 2):
        policy(**observation(n))
    for i, row in enumerate(policy.records):
        row.update(cost_s=10.+i*17., terminal=i == 3, action_kind="scan_bundle")
    train.attach_returns(policy.records,
        actual_time_s=sum(row["cost_s"] for row in policy.records), success=True)
    return policy.records


def test_dimensions_padding_permutation_and_cpu_contract():
    model = network.make_model(hidden=16)
    assert (model.global_dim, model.candidate_dim) == (10, 18)
    assert network.feature_schema()["version"] == "q4-scan-bundle-service-g1-v1"
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert all(p.device.type == "cpu" for p in model.parameters())
    row = observation(); order = [2, 0, 1]
    permuted = {**row, "candidate_features": [row["candidate_features"][i] for i in order]}
    logits, value = model(*network.pack_observations([row]))
    mixed_logits, mixed_value = model(*network.pack_observations([permuted, observation(9)]))
    torch.testing.assert_close(logits[0, order], mixed_logits[0, :3])
    torch.testing.assert_close(value[0], mixed_value[0])
    assert torch.isneginf(mixed_logits[0, 3:]).all()
    for invalid in ({**row, "global_features": [0.]*13},
                    {**row, "candidate_features": [[0.]*16]},
                    {**row, "candidate_features": [[math.inf]*18]}):
        with pytest.raises(ValueError):
            network.pack_observations([invalid])


@pytest.mark.parametrize("field,value", [
    ("version", "q4-ppo-cpu-v1"),
    ("controller_entrypoint", "q4_rl.memory_controller:run_q4_memory"),
    ("device", "cuda"),
    ("feature_schema.bundle_semantics_version", "different-order"),
    ("feature_schema.candidate_features", ["wrong-meaning"]*18),
    ("network.candidate_dim", 16),
    ("network.hidden", True),
    ("config.architecture", "induced"),
    ("config.initialization", {"from": "old-model"}),
    ("objective.gamma", .99),
])
def test_checkpoint_rejects_cross_schema_or_cost_semantics(tmp_path, field, value):
    model = network.make_model(hidden=16); path = tmp_path/"model.pt"
    train.save_checkpoint(path, model, torch.optim.Adam(model.parameters(), lr=.001),
                          {}, {"learning_rate": .001})
    saved = torch.load(path, map_location="cpu", weights_only=True)
    target = saved
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    torch.save(saved, path)
    for loader in (network.load_policy, train.restore_checkpoint):
        with pytest.raises(ValueError):
            loader(path)


@pytest.mark.parametrize("update_name", ["imitation_update", "ppo_update"])
def test_losses_update_actor_and_critic_and_checkpoint_roundtrip(tmp_path, update_name):
    assert getattr(train, update_name) is getattr(shared, update_name)
    torch.manual_seed(11)
    model = network.make_model(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = getattr(train, update_name)(model, optimizer, records_for(model), epochs=1, minibatch_size=2)
    assert result["updates"] == 2
    for prefix in ("actor.", "critic."):
        assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items()
                   if key.startswith(prefix))
    path = tmp_path/"model.pt"
    train.save_checkpoint(path, model, optimizer, {"batches": 1}, {"learning_rate": .001})
    loaded = network.load_policy(path)
    assert loaded(**observation()) in range(3)
    restored, restored_optimizer, state, config = train.restore_checkpoint(path)
    assert state == {"batches": 1} and config == {"learning_rate": .001}
    assert_nested_equal(model.state_dict(), restored.state_dict())
    assert_nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    with pytest.raises(ValueError, match="learning rate"):
        train.restore_checkpoint(path, learning_rate=.002)


@pytest.mark.parametrize("horizon,action_limit", [(1, 20000), (2, 20000), (128, 20000), (128, 5)])
def test_actual_group_boundaries_keep_full_mc_cost_including_fallback_and_exit(
        compact_cover, horizon, action_limit):
    client = ScriptedClient(); client.exit_cost = 7.25
    starts = []
    def choose(global_features, candidate_features):
        # Record actual pre-decision time only in this test observer.
        starts.append((len(search.report.action_history), client.state.virtual_time_s))
        return search._heuristic(search._candidates())
    search = controller.Q4BundleSearch(client, policy=choose, max_decisions=horizon,
                                       max_actions=action_limit, max_expansions=0)
    report = search.run(); records = copy.deepcopy(report.learning["transitions"])
    success = report.learning["training_success"]
    raw_times = [0.] + [row["virtual_time_s"] for row in report.action_history] + [report.virtual_time_s]
    primitive = [b-a for a,b in zip(raw_times, raw_times[1:])]
    metrics = train.attach_returns(records, actual_time_s=report.virtual_time_s, success=success)
    adjustment = metrics["failure_penalty_adjustment_s"]
    fine_returns = shared.undiscounted_returns(primitive, penalty_adjustment_s=adjustment)
    assert len(starts) == len(records)
    for i, (start_action, start_time) in enumerate(starts):
        end_time = starts[i+1][1] if i+1 < len(starts) else report.virtual_time_s
        assert records[i]["cost_s"] == pytest.approx(end_time-start_time)
        assert records[i]["return"] == pytest.approx(fine_returns[start_action])
        assert records[i]["return"] == pytest.approx(-(report.virtual_time_s-start_time+adjustment)/1000.)
    assert records[-1]["terminal"]
    assert sum(row["cost_s"] for row in records) == pytest.approx(report.virtual_time_s)
    assert report.learning["uncovered_cost_s"] == 7.25
    if horizon <= 2:
        assert report.learning["fallback_cost_s"] > 0
        assert records[-1]["fallback_cost_s"] == report.learning["fallback_cost_s"]
    if action_limit == 5:
        assert not success and records[0]["terminal_penalty_s"] > 0
        assert records[0]["return"] == -360.
    bad = copy.deepcopy(records); bad[-1]["cost_s"] -= 3.
    with pytest.raises(ValueError, match="complete episode"):
        train.attach_returns(bad, actual_time_s=report.virtual_time_s, success=success)


@pytest.mark.parametrize("mode", ["imitation", "ppo", "administrative"])
def test_real_rollout_wiring_uses_public_features_and_post_terminal_audit(compact_cover, monkeypatch, mode):
    import simulation
    import q4_rl.scenarios as scenarios
    import experiments.q4_comparison_bounds as bounds
    client = ScriptedClient(); client.exit_cost = 7.25
    truth_sentinel = {"fixture_only": True}
    chronology = []
    class SimulatorFixture:
        def __init__(self, case, **kwargs):
            assert case == "synthetic-constructor-fixture"
        def client(self):
            return client
        def evaluation(self):
            assert client.state.session == "exited"
            chronology.append("evaluation_after_exit")
            return dict(all_cleared=len([c for c in client.calls if c[0] == "clear"]) == 10,
                        failed_clear_count=0, ground_truth=truth_sentinel)
        def observation_history(self):
            return []
    def build_case(seed, **kwargs):
        assert seed == 8006620 and kwargs["split"] == "train"
        return "synthetic-constructor-fixture"
    original_returns = train.attach_returns
    def returns(*args, **kwargs):
        result = original_returns(*args, **kwargs)
        chronology.append("returns_fixed")
        return result
    def bound(truth):
        assert truth is truth_sentinel and chronology == ["evaluation_after_exit", "returns_fixed"]
        return {"common_lower_bound_s": 20.}
    monkeypatch.setattr(simulation, "LocalResearchSimulator", SimulatorFixture)
    monkeypatch.setattr(scenarios, "build_case", build_case)
    monkeypatch.setattr(train, "attach_returns", returns)
    monkeypatch.setattr(bounds, "common_bound", bound)
    if mode == "administrative":
        def interrupted(client, **kwargs):
            search = controller.Q4BundleSearch(client, max_expansions=0, **kwargs)
            def reply(point, channel):
                search.action_deadline_epoch = time.time()-1.
                return {"measure_result": "no_signal"}
            client.reply = reply
            return search.run()
        monkeypatch.setattr(controller, "run_q4_bundle", interrupted)
    model = network.make_model(hidden=16)
    result = train.rollout(dict(seed=8006620, action_seed=123, mode="imitation" if mode == "administrative" else mode,
        network=model.metadata(), model=model.state_dict(), max_decisions=1, deadline_epoch=time.time()+30.))
    rows = result["records"]
    assert len(rows) == 1 and len(rows[0]["global_features"]) == 10
    assert all(len(c) == 18 for c in rows[0]["candidate_features"])
    assert "seed" not in rows[0] and "ground_truth" not in rows[0]
    assert rows[0]["cost_s"] == result["metrics"]["actual_time_s"]
    assert ("log_prob" in rows[0]) == (mode == "ppo")
    assert result.get("administrative_skip") == ("training_deadline_during_episode" if mode == "administrative" else None)


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a,b):
            assert_nested_equal(left,right)
    else:
        assert a == b


def fake_rollout(task):
    model = network.model_from_metadata(task["network"]); model.load_state_dict(task["model"])
    random.seed(task["action_seed"]); torch.manual_seed(task["action_seed"])
    records = records_for(model); actual = sum(row["cost_s"] for row in records)
    metrics = dict(seed=task["seed"], actual_time_s=actual, penalized_time_s=actual,
        common_lower_bound_s=20., success=True, failed_clear_count=0,
        **{key: .01 for key in ("wall_time_s", "worker_cpu_s", "policy_wall_s", "policy_cpu_s",
                               "posthoc_bound_wall_s", "posthoc_bound_cpu_s")})
    return dict(seed=task["seed"], records=records, metrics=metrics, fixture_only=True)


def arguments(output, mode="ppo"):
    return ["--output", str(output), "--workers", "1", "--cpu-budget", "1", "--hidden", "16",
        "--epochs", "1", "--minibatch-size", "2", "--batch-episodes", "2", "--max-decisions", "8",
        "--warmstart-episodes", "2" if mode == "imitation" else "0",
        "--scenario-start", "8006620", "--scenario-end", "8006621", "--max-batches", "1",
        "--max-wall-seconds", "60", "--deadline", "2099-01-01T00:00:00+00:00"]


@pytest.mark.parametrize("mode", ["imitation", "ppo"])
def test_administrative_update_rollback_and_resume_exact_model_optimizer_rng(tmp_path, monkeypatch, mode):
    seen = []
    def worker(task):
        seen.append((task["seed"], task["action_seed"]))
        return fake_rollout(task)
    monkeypatch.setattr(train, "rollout", worker)
    name = "imitation_update" if mode == "imitation" else "ppo_update"
    real_update = getattr(train, name)
    def interrupted(*args, **kwargs):
        real_update(*args, **kwargs)
        raise train.TrainingStop("fixture stop after optimizer and RNG changed")
    monkeypatch.setattr(train, name, interrupted)
    output = tmp_path/"resumed"
    state = train.main(arguments(output, mode))
    assert state["episodes"] == 0 and state["pending_batch"]["seeds"] == [8006620,8006621]
    raw = output/"batch-000000-attempt-000000.json.gz"; original = raw.read_bytes()
    assert len(read_batch(raw)) == 2
    initial = torch.load(output/"random.pt", weights_only=True)
    reserved = torch.load(output/"latest.pt", weights_only=True)
    assert_nested_equal(initial["model"], reserved["model"])
    assert_nested_equal(initial["optimizer"], reserved["optimizer"])
    monkeypatch.setattr(train, name, real_update)
    state = train.main(arguments(output, mode)+["--resume", str(output/"latest.pt")])
    assert state["episodes"] == 2 and state["pending_batch"] is None
    assert seen[:2] == seen[2:4] and raw.read_bytes() == original
    control = tmp_path/"control"; train.main(arguments(control, mode))
    a,b = [torch.load(folder/"latest.pt", weights_only=True) for folder in (output,control)]
    for key in ("model", "optimizer", "rng"):
        assert_nested_equal(a[key], b[key])


def test_administrative_partial_group_is_journaled_but_never_learned(tmp_path, monkeypatch):
    def interrupted(task):
        return {**fake_rollout(task), "administrative_skip": "training_deadline_during_episode"}
    monkeypatch.setattr(train, "rollout", interrupted)
    def forbid_update(*args, **kwargs):
        pytest.fail("administratively truncated episode entered optimizer")
    monkeypatch.setattr(train, "ppo_update", forbid_update)
    state = train.main(arguments(tmp_path))
    assert state["episodes"] == state["batches"] == 0
    assert state["pending_batch"]["seeds"] == [8006620,8006621]
    raw = tmp_path/"batch-000000-attempt-000000.json.gz"
    assert all(row["administrative_skip"] for row in read_batch(raw))
    initial = torch.load(tmp_path/"random.pt", weights_only=True)
    reserved = torch.load(tmp_path/"latest.pt", weights_only=True)
    assert_nested_equal(initial["model"], reserved["model"])
    assert_nested_equal(initial["optimizer"], reserved["optimizer"])


def test_default_horizon_and_cpu_training_partition_contract(tmp_path):
    _, args = train._arguments(["--output", str(tmp_path), "--workers", "1", "--cpu-budget", "1"])
    assert args.max_decisions == 128 and train.configuration(args)["max_decisions"] == 128
    for extra in (["--scenario-start", "8100000"], ["--workers", "2", "--cpu-budget", "2"]):
        with pytest.raises(SystemExit):
            train._arguments(["--output", str(tmp_path), *extra])
