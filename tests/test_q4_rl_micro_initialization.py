"""Weight-only BC initialization and transactional PPO continuation; no scenes."""
from copy import deepcopy
import hashlib
import random

import pytest

torch = pytest.importorskip("torch")

from q4_rl import micro_network as network
from q4_rl import micro_train as train
from q4_rl.micro_initialization import initialization_binding, load_micro_warmstart
from tests.test_q4_rl_micro_training import arguments, records_for, fake_rollout


@pytest.fixture(autouse=True)
def cpu():
    network.configure_cpu()


def make_source(path, architecture="mlp"):
    torch.manual_seed(121)
    random.seed(122)
    model = network.make_model(architecture, hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    train.imitation_update(model, optimizer, records_for(model), epochs=1, minibatch_size=2)
    state = dict(next_seed=8000002, episodes=2, attempted_episodes=2,
        warmstart_completed=2, ppo_batches=0, batches=1, wall_time_s=1.,
        pending_batch=None, next_attempt=1, stop_reason=None)
    config = dict(learning_rate=.002, warmstart_episodes=2, scenario_start=8000000,
                  scenario_end=8099999, hidden=16)
    if architecture != "mlp":
        config["architecture"] = architecture
    train.save_checkpoint(path, model, optimizer, state, config)
    return torch.load(path, weights_only=True), hashlib.sha256(path.read_bytes()).hexdigest()


def initialized_args(output, source, digest, architecture="mlp"):
    return arguments(output)+["--initialize-micro-warmstart", str(source),
        "--initialize-sha256", digest, "--architecture", architecture]


def test_hash_is_checked_before_decoding_untrusted_or_mismatched_bytes(tmp_path, monkeypatch):
    source = tmp_path/"wrong.pt"
    source.write_bytes(b"not a checkpoint")
    def forbidden(*args, **kwargs):
        raise AssertionError("decoding preceded checksum validation")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_micro_warmstart(source, "0"*64)
    for value in (None, "", "0"*63, "g"*64):
        with pytest.raises(ValueError, match="64 hexadecimal"):
            initialization_binding(value)
    assert initialization_binding("A"*64)["sha256"] == "a"*64


@pytest.mark.parametrize("mutation", ["incomplete", "zero_target", "ppo", "pending",
                                       "missing_pending", "episodes", "schema", "device", "objective"])
def test_only_completed_micro_bc_without_ppo_is_accepted(tmp_path, mutation):
    source = tmp_path/"source.pt"
    saved, _ = make_source(source)
    if mutation == "incomplete": saved["state"]["warmstart_completed"] = 1
    elif mutation == "zero_target": saved["config"]["warmstart_episodes"] = 0
    elif mutation == "ppo": saved["state"]["ppo_batches"] = 1
    elif mutation == "pending": saved["state"]["pending_batch"] = {"seeds": [8000002]}
    elif mutation == "missing_pending": saved["state"].pop("pending_batch")
    elif mutation == "episodes": saved["state"]["episodes"] = 3
    elif mutation == "schema": saved["feature_schema"]["version"] = "q4-other"
    elif mutation == "device": saved["device"] = "gpu"
    else: saved["objective"]["gamma"] = .9
    torch.save(saved, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        load_micro_warmstart(source, digest)


@pytest.mark.parametrize("architecture", ["mlp", "induced"])
def test_new_run_copies_weights_but_resets_adam_counters_and_random_stream(tmp_path, monkeypatch, architecture):
    source = tmp_path/"source.pt"
    saved, digest = make_source(source, architecture)
    original_bytes = source.read_bytes()
    seen = []
    def worker(task):
        seen.append(deepcopy(task))
        return fake_rollout(task)
    monkeypatch.setattr(train, "rollout", worker)
    output = tmp_path/"new"
    state = train.main(initialized_args(output, source, digest, architecture))
    initial = torch.load(output/"initial.pt", weights_only=True)
    assert not (output/"random.pt").exists()
    assert all(torch.equal(value, initial["model"][key]) for key, value in saved["model"].items())
    assert saved["optimizer"]["state"] and initial["optimizer"]["state"] == {}
    assert initial["optimizer"]["param_groups"][0]["lr"] == .0003
    for field in ("episodes", "attempted_episodes", "warmstart_completed", "ppo_batches", "batches", "next_attempt"):
        assert initial["state"][field] == 0
    assert initial["state"]["next_seed"] == 8005010
    assert initial["state"]["pending_batch"] is None
    assert initial["config"]["initialization"] == initialization_binding(digest)
    assert str(tmp_path) not in repr(initial["config"])
    assert initial["rng"]["python"] == random.Random(424242).getstate()
    assert torch.equal(initial["rng"]["torch"], torch.Generator().manual_seed(424242).get_state())
    assert seen[0]["mode"] == "ppo"
    assert seen[0]["seed"] == 8005010
    assert seen[0]["action_seed"] == random.Random(424242).randrange(2**31)
    assert state["warmstart_completed"] == 0 and state["ppo_batches"] == 1
    assert source.read_bytes() == original_bytes
    with pytest.raises(SystemExit):
        train.main(initialized_args(output, source, digest, architecture))


def test_initializer_cli_requires_binding_no_repeat_bc_and_explicit_training_interval(tmp_path):
    source = tmp_path/"source.pt"
    _, digest = make_source(source)
    base = initialized_args(tmp_path/"new", source, digest)
    for flag in ("--initialize-micro-warmstart", "--initialize-sha256", "--scenario-start", "--scenario-end"):
        changed = list(base)
        index = changed.index(flag)
        del changed[index:index+2]
        with pytest.raises(SystemExit):
            train._arguments(changed)
    with pytest.raises(SystemExit):
        train._arguments(base+["--warmstart-episodes", "2"])
    with pytest.raises(ValueError, match="metadata differs"):
        train.main(base+["--hidden", "32"])
    with pytest.raises(ValueError, match="metadata differs"):
        train.main(base+["--architecture", "induced"])
    assert not (tmp_path/"new").exists()


def test_initialized_interruption_replays_and_resume_keeps_content_binding(tmp_path, monkeypatch):
    source = tmp_path/"source.pt"
    saved, digest = make_source(source)
    seen = []
    def worker(task):
        seen.append((task["seed"], task["action_seed"]))
        return fake_rollout(task)
    monkeypatch.setattr(train, "rollout", worker)
    update = train.ppo_update
    def interrupted(*args, **kwargs):
        update(*args, **kwargs)
        raise train.TrainingStop("scripted update interruption")
    monkeypatch.setattr(train, "ppo_update", interrupted)
    output = tmp_path/"resume"
    args = initialized_args(output, source, digest)
    stopped = train.main(args)
    assert stopped["episodes"] == 0 and stopped["pending_batch"] is not None
    monkeypatch.setattr(train, "ppo_update", update)
    with pytest.raises(SystemExit):
        train.main(arguments(output)+["--resume", str(output/"latest.pt")])
    changed = deepcopy(saved)
    next(iter(changed["model"].values())).add_(.1)
    alternative = tmp_path/"other.pt"
    torch.save(changed, alternative)
    alternative_digest = hashlib.sha256(alternative.read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        train.main(initialized_args(output, alternative, alternative_digest)+["--resume", str(output/"latest.pt")])
    # The source may move without changing the content binding.
    relocated = tmp_path/"same-bytes.pt"
    relocated.write_bytes(source.read_bytes())
    train.main(initialized_args(output, relocated, digest)+["--resume", str(output/"latest.pt")])
    assert seen[0] == seen[1]
    control = tmp_path/"control"
    train.main(initialized_args(control, source, digest))
    resumed, direct = (torch.load(folder/"latest.pt", weights_only=True) for folder in (output, control))
    assert all(torch.equal(value, direct["model"][key]) for key, value in resumed["model"].items())
    assert torch.equal(resumed["rng"]["torch"], direct["rng"]["torch"])
    assert resumed["config"] == direct["config"]
