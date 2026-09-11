"""G3 completed BC initialization is schema-bound, fresh and resume-contained."""
from copy import deepcopy
import hashlib
import random

import pytest

torch = pytest.importorskip("torch")

from q4_rl import memory_train as train, memory_network as network
from q4_rl.memory_initialization import initialization_binding, load_memory_warmstart
from tests.test_q4_rl_memory_training import arguments, records_for, fake_rollout
from tests.test_q4_rl_micro_initialization import make_source as make_g1_source
from tests.test_q4_rl_learner_threads import assert_exact, checkpoint, single_thread_runtime


def make_source(path):
    torch.manual_seed(121)
    random.seed(122)
    model = network.make_model(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    train.imitation_update(model, optimizer, records_for(model), epochs=1, minibatch_size=2)
    state = dict(next_seed=8000002, episodes=2, attempted_episodes=2,
        warmstart_completed=2, ppo_batches=0, batches=1, wall_time_s=1.,
        pending_batch=None, next_attempt=1, stop_reason=None)
    config = dict(learning_rate=.002, warmstart_episodes=2, scenario_start=8000000,
                  scenario_end=8099999, hidden=16)
    train.save_checkpoint(path, model, optimizer, state, config)
    return torch.load(path, weights_only=True), hashlib.sha256(path.read_bytes()).hexdigest()


def initialized_args(output, source, digest):
    return arguments(output)+["--initialize-memory-warmstart", str(source),
        "--initialize-sha256", digest, "--warmstart-episodes", "0", "--gae-lambda", ".97"]


def test_sha_checked_before_decode_and_g1_checkpoint_is_forbidden(tmp_path, monkeypatch):
    g1 = tmp_path/"g1.pt"
    _, digest = make_g1_source(g1)
    with pytest.raises(ValueError, match="G3 memory checkpoint"):
        load_memory_warmstart(g1, digest)
    def forbidden(*args, **kwargs):
        raise AssertionError("decode ran before SHA validation")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_memory_warmstart(g1, "0"*64)


@pytest.mark.parametrize("mutation", ["incomplete", "zero", "ppo", "pending", "missing_pending",
                                      "episodes", "schema", "objective", "device", "binding"])
def test_incomplete_or_mismatched_source_is_rejected(tmp_path, mutation):
    path = tmp_path/"source.pt"
    saved, _ = make_source(path)
    if mutation == "incomplete": saved["state"]["warmstart_completed"] = 1
    elif mutation == "zero": saved["config"]["warmstart_episodes"] = 0
    elif mutation == "ppo": saved["state"]["ppo_batches"] = 1
    elif mutation == "pending": saved["state"]["pending_batch"] = {"seeds": [8000002]}
    elif mutation == "missing_pending": saved["state"].pop("pending_batch")
    elif mutation == "episodes": saved["state"]["episodes"] = 1
    elif mutation == "schema": saved["feature_schema"]["version"] = "other"
    elif mutation == "objective": saved["objective"]["gamma"] = .9
    elif mutation == "device": saved["device"] = "gpu"
    else: saved["config"]["initialization"] = dict(type="frozen-micro-bc-weights-v1", sha256="a"*64)
    torch.save(saved, path)
    with pytest.raises(ValueError):
        load_memory_warmstart(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_fresh_weights_adam_rng_and_self_contained_resume(tmp_path, monkeypatch):
    source = tmp_path/"source.pt"
    original, digest = make_source(source)
    original_bytes = source.read_bytes()
    observed = []
    def rollout(task):
        observed.append((task["seed"], task["action_seed"], task["mode"]))
        return fake_rollout(task)
    monkeypatch.setattr(train, "rollout", rollout)
    real_update = train.ppo_update
    def interrupted(*args, **kwargs):
        real_update(*args, **kwargs)
        raise train.TrainingStop("fixture stop after initialized GAE update")
    monkeypatch.setattr(train, "ppo_update", interrupted)
    output = tmp_path/"resume"
    state = train.main(initialized_args(output, source, digest))
    initial = torch.load(output/"initial.pt", weights_only=True)
    assert not (output/"random.pt").exists()
    assert_exact(original["model"], initial["model"])
    assert original["optimizer"]["state"] and initial["optimizer"]["state"] == {}
    assert initial["optimizer"]["param_groups"][0]["lr"] == .0003
    for field in ("episodes", "attempted_episodes", "warmstart_completed", "ppo_batches", "batches", "next_attempt"):
        assert initial["state"][field] == 0
    assert initial["state"]["next_seed"] == 8006510
    assert initial["rng"]["python"] == random.Random(424242).getstate()
    assert torch.equal(initial["rng"]["torch"], torch.Generator().manual_seed(424242).get_state())
    assert initial["config"]["initialization"] == initialization_binding(digest)
    assert str(tmp_path) not in repr(initial["config"])
    assert all(mode == "ppo" for _, _, mode in observed)
    assert state["episodes"] == 0 and state["pending_batch"] is not None
    assert source.read_bytes() == original_bytes
    monkeypatch.setattr(train, "ppo_update", real_update)
    # Resume never rereads a source checkpoint, even if it has moved or vanished.
    monkeypatch.setattr(train, "load_memory_warmstart", lambda *a, **k: pytest.fail("resume read source"))
    resume = arguments(output)+["--gae-lambda", ".97", "--resume", str(output/"latest.pt")]
    with pytest.raises(SystemExit):
        train.main(resume+["--gae-lambda", "1"])
    with pytest.raises(SystemExit):
        train.main(resume+["--initialize-memory-warmstart", str(source), "--initialize-sha256", digest])
    train.main(resume)
    assert observed[:2] == observed[2:]
    assert checkpoint(output)["config"]["initialization"] == initialization_binding(digest)
    assert network.load_policy(output/"latest.pt") is not None
    monkeypatch.setattr(train, "load_memory_warmstart", load_memory_warmstart)
    control = tmp_path/"control"
    train.main(initialized_args(control, source, digest))
    a, b = checkpoint(output), checkpoint(control)
    for key in ("model", "optimizer", "rng", "config", "objective"):
        assert_exact(a[key], b[key])
    with pytest.raises(SystemExit):
        train.main(initialized_args(control, source, digest))


def test_initialization_requires_explicit_new_interval_no_bc_and_exact_metadata(tmp_path):
    source = tmp_path/"source.pt"
    _, digest = make_source(source)
    args = initialized_args(tmp_path/"output", source, digest)
    for flag in ("--initialize-memory-warmstart", "--initialize-sha256", "--scenario-start", "--scenario-end"):
        modified = list(args)
        index = modified.index(flag)
        del modified[index:index+2]
        with pytest.raises(SystemExit):
            train._arguments(modified)
    with pytest.raises(SystemExit):
        train._arguments(args+["--warmstart-episodes", "1"])
    with pytest.raises(ValueError, match="metadata differs"):
        train.main(args+["--hidden", "32"])
    assert not (tmp_path/"output").exists()
