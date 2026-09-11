"""Architecture-only micro set attention: no scenarios or simulator calls."""
from copy import deepcopy
import random

import pytest

torch = pytest.importorskip("torch")

from q4_rl import micro_network as network
from q4_rl import micro_train as train
from q4_rl.network import CandidateActorCritic
from tests.test_q4_rl_micro_training import observation, records_for, fake_rollout, arguments


@pytest.fixture(autouse=True)
def cpu():
    network.configure_cpu()


def test_default_mlp_weights_outputs_metadata_and_configuration_are_unchanged(tmp_path):
    torch.manual_seed(911)
    old = CandidateActorCritic(global_dim=13, candidate_dim=50, hidden=16)
    torch.manual_seed(911)
    default = network.make_model(hidden=16)
    assert type(default) is network.MicroCandidateActorCritic
    assert all(torch.equal(v, default.state_dict()[k]) for k, v in old.state_dict().items())
    tensors = network.pack_observations([observation(3)])
    assert all(torch.equal(a, b) for a, b in zip(old(*tensors), default(*tensors)))
    assert default.metadata() == dict(architecture=network.ARCHITECTURE, global_dim=13, candidate_dim=50, hidden=16)
    _, args = train._arguments(arguments(tmp_path))
    expected = {**train._configuration(args), "controller_entrypoint": network.CONTROLLER_ENTRYPOINT}
    assert train.configuration(args) == expected and "architecture" not in expected
    args.architecture = "induced"
    assert train.configuration(args) == {**expected, "architecture": "induced"}


@pytest.mark.parametrize("training", [False, True])
def test_attention_is_candidate_equivariant_and_padding_invariant(training):
    torch.manual_seed(912)
    model = network.make_model("induced", hidden=16).train(training)
    row = observation(5)
    order = [3, 0, 4, 1, 2]
    other = {**row, "candidate_features": [row["candidate_features"][i] for i in order]}
    logits, value = model(*network.pack_observations([row]))
    actual, other_value = model(*network.pack_observations([other, observation(11)]))
    torch.testing.assert_close(actual[0, :5], logits[0, order], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(other_value[:1], value, atol=1e-6, rtol=1e-5)
    assert torch.isneginf(actual[0, 5:]).all()


def test_masked_nan_padding_has_no_forward_effect_or_gradient():
    model = network.make_model("induced", hidden=16)
    g, x, mask = network.pack_observations([observation(2), observation(8)])
    expected = model(g, x, mask)
    x = x.masked_fill(~mask[..., None], float("nan")).requires_grad_()
    logits, value = model(g, x, mask)
    torch.testing.assert_close(logits[mask], expected[0][mask])
    torch.testing.assert_close(value, expected[1])
    (logits[mask].square().sum()+value.square().sum()).backward()
    assert torch.isfinite(x.grad).all()
    assert torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask]))


@pytest.mark.parametrize("non_cpu", ["global", "candidate", "mask", "model"])
def test_attention_rejects_non_cpu_without_initializing_gpu(non_cpu):
    model = network.make_model("induced", hidden=16)
    g, x, mask = network.pack_observations([observation()])
    if non_cpu == "global": g = g.to("meta")
    elif non_cpu == "candidate": x = x.to("meta")
    elif non_cpu == "mask": mask = mask.to("meta")
    else: model = model.to("meta")
    with pytest.raises(ValueError, match="CPU-only"):
        model(g, x, mask)


def test_exact_attention_metadata_and_config_are_required(tmp_path):
    model = network.make_model("induced", hidden=16)
    meta = model.metadata()
    assert meta == dict(architecture=network.INDUCED_ARCHITECTURE, global_dim=13,
                       candidate_dim=50, hidden=16, **network.INDUCED_SETTINGS)
    worker_model = network.model_from_metadata(meta)
    worker_model.load_state_dict(model.state_dict())
    assert worker_model.metadata() == meta
    for field, value in (("attention_heads", 4), ("attention_dim", 32), ("inducing_points", 8),
                         ("attention_blocks", 2), ("attention_refinement", "norm_ff")):
        bad = {**meta, field: value}
        with pytest.raises(ValueError, match="metadata"):
            network.model_from_metadata(bad)
    bad = deepcopy(meta)
    bad.pop("attention_heads")
    with pytest.raises(ValueError, match="metadata"):
        network.model_from_metadata(bad)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    with pytest.raises(ValueError):
        train.save_checkpoint(tmp_path/"bad.pt", model, optimizer, {}, {"learning_rate": .001})
    path = tmp_path/"good.pt"
    train.save_checkpoint(path, model, optimizer, {}, {"learning_rate": .001, "architecture": "induced"})
    saved = torch.load(path, weights_only=True)
    saved["config"].pop("architecture")
    torch.save(saved, tmp_path/"spoof.pt")
    with pytest.raises(ValueError, match="architecture differ"):
        network.load_policy(tmp_path/"spoof.pt")


@pytest.mark.parametrize("kind", ["ppo", "imitation"])
def test_shared_updates_change_attention_weights_and_remain_finite(kind):
    torch.manual_seed(913)
    model = network.make_model("induced", hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    records = records_for(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    update = train.ppo_update if kind == "ppo" else train.imitation_update
    result = update(model, optimizer, records, epochs=1, minibatch_size=2)
    assert result["updates"] == 2
    assert all(torch.isfinite(v).all() for v in model.parameters())
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items() if k.startswith("set_blocks."))


def test_attention_checkpoint_rng_and_next_gradient_update_are_exact(tmp_path):
    torch.manual_seed(914)
    random.seed(914)
    model = network.make_model("induced", hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    records = records_for(model)
    state = dict(next_seed=8005021, pending_batch=dict(seeds=[8005020], action_seeds=[1]))
    config = dict(learning_rate=.001, architecture="induced")
    rng = torch.get_rng_state().clone()
    train.save_checkpoint(tmp_path/"latest.pt", model, optimizer, state, config)
    assert torch.equal(rng, torch.get_rng_state())
    train.ppo_update(model, optimizer, records, epochs=2, minibatch_size=2)
    restored, opt, cursor, saved_config = train.restore_checkpoint(tmp_path/"latest.pt")
    assert cursor == state and saved_config == config
    assert type(restored) is network.MicroInducedActorCritic
    train.ppo_update(restored, opt, records, epochs=2, minibatch_size=2)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert network.load_policy(tmp_path/"latest.pt")(**observation(3)) in range(3)


def test_main_worker_and_interrupted_resume_keep_attention_architecture(tmp_path, monkeypatch):
    seen = []
    def worker(task):
        seen.append(task["network"])
        assert task["network"]["architecture"] == network.INDUCED_ARCHITECTURE
        return fake_rollout(task)
    monkeypatch.setattr(train, "rollout", worker)
    actual_update = train.ppo_update
    def interrupted(*args, **kwargs):
        actual_update(*args, **kwargs)
        raise train.TrainingStop("scripted update interruption")
    monkeypatch.setattr(train, "ppo_update", interrupted)
    args = arguments(tmp_path)+["--architecture", "induced"]
    state = train.main(args)
    assert state["episodes"] == 0 and state["pending_batch"] is not None
    monkeypatch.setattr(train, "ppo_update", actual_update)
    with pytest.raises(SystemExit):
        train.main(arguments(tmp_path)+["--resume", str(tmp_path/"latest.pt")])
    state = train.main(args+["--resume", str(tmp_path/"latest.pt")])
    assert state["episodes"] == 1 and state["pending_batch"] is None
    assert len(seen) == 2 and seen[0] == seen[1]


def benchmark(output):
    """Short paired CPU timings on fabricated features, not scenario training."""
    import hashlib
    import json
    from pathlib import Path
    import statistics
    import time
    path = Path(output)
    if path.exists():
        raise ValueError("benchmark output already exists")
    network.configure_cpu()
    batch_size, repetitions = 16, 12
    results = []
    for count in (32, 440, 636):
        generator = torch.Generator().manual_seed(941)
        g = torch.randn(batch_size, 13, generator=generator)
        x = torch.randn(batch_size, count, 50, generator=generator)
        mask = torch.ones(batch_size, count, dtype=torch.bool)
        models, optimizers, records, times = {}, {}, {}, {}
        for architecture in ("mlp", "induced"):
            torch.manual_seed(940)
            model = network.make_model(architecture, hidden=64)
            models[architecture] = model
            optimizers[architecture] = torch.optim.Adam(model.parameters(), lr=.0003)
            with torch.no_grad():
                logits, values = model(g, x, mask)
                old_log = torch.log_softmax(logits, -1)[:, 0]
            records[architecture] = [dict(global_features=ga, candidate_features=ca,
                action_index=0, log_prob=float(old_log[i]), value=float(values[i]),
                **{"return": -5.-i/10.}) for i, (ga, ca) in enumerate(zip(g.tolist(), x.tolist()))]
            times[architecture] = dict(inference_ms=[], shared_ppo_with_packing_ms=[])
        for repeat in range(repetitions+2):
            order = ("mlp", "induced") if repeat % 2 else ("induced", "mlp")
            for architecture in order:
                model = models[architecture]
                began = time.perf_counter()
                with torch.no_grad():
                    model(g[:1], x[:1], mask[:1])
                forward = (time.perf_counter()-began)*1000.
                began = time.perf_counter()
                train.ppo_update(model, optimizers[architecture], records[architecture],
                    epochs=1, minibatch_size=batch_size)
                update = (time.perf_counter()-began)*1000.
                if repeat >= 2:
                    times[architecture]["inference_ms"].append(forward)
                    times[architecture]["shared_ppo_with_packing_ms"].append(update)
        medians = {a: {k: statistics.median(v) for k, v in rows.items()} for a, rows in times.items()}
        results.append(dict(candidates=count, medians=medians,
            induced_over_mlp={k: medians["induced"][k]/medians["mlp"][k] for k in medians["mlp"]},
            parameters={a: sum(p.numel() for p in model.parameters()) for a, model in models.items()}, raw_ms=times))
    result = dict(kind="micro13x50-induced16-cpu-microbenchmark-v1", hidden=64,
        batch_size=batch_size, repetitions=repetitions, warmup=2, native_threads=1,
        attention=dict(network.INDUCED_SETTINGS), scenarios_used=0,
        notes="Single-thread CPU; alternating architecture order; actual shared PPO includes packing/backward/Adam. No scenario time or T/L exists.",
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path("src/q4_rl/micro_network.py"), Path("src/q4_rl/micro_train.py"), Path(__file__).relative_to(Path.cwd()))},
        results=results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps([{k: v for k, v in row.items() if k != "raw_ms"} for row in results]), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-output", required=True)
    benchmark(parser.parse_args().benchmark_output)
